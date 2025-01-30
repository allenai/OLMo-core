import warnings
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor

from ...distributed.utils import get_local_tensor
from ...exceptions import OLMoConfigurationError

__all__ = ["MoEMLP"]


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    @torch.amp.autocast_mode.custom_fwd(device_type="cuda")
    def forward(ctx: Any, x: torch.Tensor, scale: float):
        ctx.scale = scale
        return x

    @staticmethod
    @torch.amp.autocast_mode.custom_bwd(device_type="cuda")
    def backward(ctx: torch.Tensor, grad: torch.Tensor):
        return grad * ctx.scale, None  # type: ignore


_scale_gradient: Callable[[torch.Tensor, float], torch.Tensor] = _ScaleGradient.apply  # type: ignore


class MoEMLP(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        num_experts: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        self.gradient_scale: Optional[float] = None
        self.experts_per_rank = num_experts

        self.w1 = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                d_model,
                device=init_device,
                dtype=dtype,
            ),
        )
        self.w2 = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                d_model,
                device=init_device,
                dtype=dtype,
            ),
        )
        self.w3 = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                d_model,
                device=init_device,
                dtype=dtype,
            ),
        )

        self._gmm = None

        try:
            import grouped_gemm  # type: ignore

            self._gmm = grouped_gemm.ops.gmm
        except ImportError:
            warnings.warn(
                "Grouped GEMM not available, so the MoE will be substantially slower. "
                "Please install the 'grouped_gemm' package if possible.\n"
                "https://github.com/tgale96/grouped_gemm"
            )

    def scale_grad(self, w: torch.Tensor) -> torch.Tensor:
        if self.gradient_scale is None:
            return w
        return _scale_gradient(w, self.gradient_scale)

    def gmm(
        self, x: torch.Tensor, w: torch.Tensor, batch_sizes: torch.Tensor, trans_b: bool = False
    ) -> torch.Tensor:
        if self._gmm is not None:
            return self._gmm(x, w, batch_sizes, trans_b=trans_b)
        else:
            out = []
            start = 0
            for i, size in enumerate(batch_sizes.cpu().numpy()):
                rhs = w[i, :, :].t() if trans_b else w[i, :, :]
                out.append(x[start : start + size, :] @ rhs)
                start += size
            return torch.cat(out)

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        """
        Compute the expert outputs.

        :param x: The input of shape ``(total_tokens, d_model)``.
        :param tokens_per_expert: Specifies how many tokens go to each expert. Should be a
            1-D ``LongTensor``.
        """
        # Scale gradients and get local tensors (in case of expert parallelism).
        # shape (all): (experts_per_rank, hidden_size, d_model)
        w1, w2, w3 = (
            get_local_tensor(self.scale_grad(self.w1)),
            get_local_tensor(self.scale_grad(self.w2)),
            get_local_tensor(self.scale_grad(self.w3)),
        )

        # Compute the MLP.
        x1 = self.gmm(x, w1, tokens_per_expert, trans_b=True)
        x2 = self.gmm(x, w3, tokens_per_expert, trans_b=True)
        x1 = F.silu(x1) * x2
        return self.gmm(x1, w2, tokens_per_expert)

    def apply_ep(self, ep_mesh: DeviceMesh):
        """
        Apply expert parallelism.
        """
        if self.num_experts % ep_mesh.size() != 0:
            raise OLMoConfigurationError(
                f"'num_experts' ({self.num_experts}) must be divisible by the expert parallel degree ({ep_mesh.size()})."
            )

        self.experts_per_rank = self.num_experts // ep_mesh.size()
        self.gradient_scale = 1.0 / ep_mesh.size()

        self.register_parameter("w1", nn.Parameter(distribute_tensor(self.w1, ep_mesh, [Shard(0)])))
        self.register_parameter("w2", nn.Parameter(distribute_tensor(self.w2, ep_mesh, [Shard(0)])))
        self.register_parameter("w3", nn.Parameter(distribute_tensor(self.w3, ep_mesh, [Shard(0)])))
