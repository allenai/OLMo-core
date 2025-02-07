import warnings
from typing import Any, Callable, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard, distribute_tensor

from ...distributed.utils import get_local_tensor
from ...exceptions import OLMoConfigurationError

__all__ = ["MoEMLP", "DroplessMoEMLP"]


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


class MoEMLPBase(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        num_experts: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        self.gradient_scale: Optional[float] = None
        self.num_local_experts = num_experts
        self.hidden_sharding_degree = 1
        self.ep_pg: Optional[dist.ProcessGroup] = None

    def scale_grad(self, w: torch.Tensor) -> torch.Tensor:
        if self.gradient_scale is None:
            return w
        return _scale_gradient(w, self.gradient_scale)

    def apply_ep(
        self,
        ep_mesh: DeviceMesh,
        compile_enabled: bool = False,
        autograd_compile_enabled: bool = False,
    ):
        """
        Apply expert parallelism.
        """
        from torch.distributed._composable.replicate import replicate

        if ep_mesh.ndim > 2:
            raise RuntimeError("expert parallel mesh must be 1 or 2D")
        if not ep_mesh.mesh_dim_names:
            raise RuntimeError("expert parallel mesh must have named dimensions")

        shard_dim_name = ep_mesh.mesh_dim_names[-1]

        self.ep_pg = ep_mesh[shard_dim_name].get_group()
        num_shards = ep_mesh[shard_dim_name].size()

        if self.num_experts % num_shards != 0:
            raise OLMoConfigurationError(
                f"'num_experts' ({self.num_experts}) must be divisible by the expert parallel shard degree ({num_shards})."
            )

        self.num_local_experts = self.num_experts // num_shards
        self.gradient_scale = 1.0 / num_shards

        placements: List[Placement] = [Shard(0)]
        if ep_mesh.ndim > 1:
            placements.insert(0, Replicate())

        self.register_parameter("w1", nn.Parameter(distribute_tensor(self.w1, ep_mesh, placements)))  # type: ignore
        self.register_parameter("w2", nn.Parameter(distribute_tensor(self.w2, ep_mesh, placements)))  # type: ignore
        self.register_parameter("w3", nn.Parameter(distribute_tensor(self.w3, ep_mesh, placements)))  # type: ignore

        if ep_mesh.ndim > 1:
            if compile_enabled:
                if autograd_compile_enabled:
                    torch._dynamo.config.optimize_ddp = "python_reducer_without_compiled_forward"  # type: ignore
                else:
                    torch._dynamo.config.optimize_ddp = "ddp_optimizer"  # type: ignore

            replicate_dim_name = ep_mesh.mesh_dim_names[0]
            replicate(self, device_mesh=ep_mesh[replicate_dim_name])


class MoEMLP(MoEMLPBase):
    """
    A basic expert MLP module with SwiGLU activation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        num_experts: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__(d_model=d_model, hidden_size=hidden_size, num_experts=num_experts)
        self.w1 = nn.Parameter(
            torch.empty(
                num_experts,
                d_model,
                hidden_size,
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
                d_model,
                hidden_size,
                device=init_device,
                dtype=dtype,
            ),
        )

    def extra_repr(self):
        return f"d_model={self.d_model}, num_experts={self.num_experts}, hidden_size={self.hidden_size}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the expert outputs.

        :param x: The input of shape ``(num_local_experts, N, d_model)``.
        """
        # Scale gradients and get local tensors (in case of expert parallelism).
        # shape (all): (num_local_experts, hidden_size, d_model)
        w1, w2, w3 = (
            get_local_tensor(self.scale_grad(self.w1)),
            get_local_tensor(self.scale_grad(self.w2)),
            get_local_tensor(self.scale_grad(self.w3)),
        )

        # Compute the MLP.
        return torch.bmm(F.silu(torch.bmm(x, w1)) * torch.bmm(x, w3), w2)


class DroplessMoEMLP(MoEMLPBase):
    """
    A dropless expert MLP module with SwiGLU activation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        num_experts: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__(d_model=d_model, hidden_size=hidden_size, num_experts=num_experts)
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

    def forward(self, x: torch.Tensor, batch_size_per_expert: torch.Tensor) -> torch.Tensor:
        """
        Compute the expert outputs.

        :param x: The input of shape ``(*, d_model)``.
        :param batch_size_per_expert: Specifies how many items/tokens go to each expert. Should be a
            1-D ``LongTensor``.
        """
        # Scale gradients and get local tensors (in case of expert parallelism).
        # shape (all): (num_local_experts, hidden_size, d_model)
        w1, w2, w3 = (
            get_local_tensor(self.scale_grad(self.w1)),
            get_local_tensor(self.scale_grad(self.w2)),
            get_local_tensor(self.scale_grad(self.w3)),
        )

        # Compute the MLP.
        x1 = self.gmm(x, w1, batch_size_per_expert, trans_b=True)
        x2 = self.gmm(x, w3, batch_size_per_expert, trans_b=True)
        x1 = F.silu(x1) * x2
        return self.gmm(x1, w2, batch_size_per_expert)
