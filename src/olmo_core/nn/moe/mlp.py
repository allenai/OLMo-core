import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor

from ...config import Config, DType, StrEnum
from ...distributed.utils import get_local_tensor
from ...exceptions import OLMoConfigurationError

__all__ = ["MoEMLP", "DroplessMoEMLP", "MoEMLPConfig", "MoEMLPType"]


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


class MoEMLPType(StrEnum):
    """
    An enumeration of the different MoE expert MLP implementations.
    """

    default = "default"
    """
    ➡️ :class:`MoEMLP`
    """

    dropless = "dropless"
    """
    ➡️ :class:`DroplessMoEMLP`
    """


@dataclass
class MoEMLPConfig(Config):
    dtype: DType = DType.float32

    def num_params(self, d_model: int, num_experts: int, hidden_size: int) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        :param num_experts: Then number of experts.
        :param hidden_size: The hidden size of each expert.
        """
        return 3 * d_model * hidden_size * num_experts

    def num_active_params(self, d_model: int, top_k: int, hidden_size: int) -> int:
        return self.num_params(d_model, top_k, hidden_size)

    def build(
        self,
        *,
        name: MoEMLPType,
        d_model: int,
        num_experts: int,
        hidden_size: int,
        init_device: str = "cpu",
    ) -> "MoEMLPBase":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.update(
            dtype=kwargs.pop("dtype").as_pt(),
            d_model=d_model,
            num_experts=num_experts,
            hidden_size=hidden_size,
            init_device=init_device,
        )

        try:
            if name == MoEMLPType.default:
                return MoEMLP(**kwargs)
            elif name == MoEMLPType.dropless:
                return DroplessMoEMLP(**kwargs)
            else:
                raise NotImplementedError(name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{name}' {self.__class__.__name__}, {e}"
            ) from e


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
        self.experts_per_rank = num_experts
        self.hidden_sharding_degree = 1

    def scale_grad(self, w: torch.Tensor) -> torch.Tensor:
        if self.gradient_scale is None:
            return w
        return _scale_gradient(w, self.gradient_scale)

    def apply_ep(self, ep_mesh: DeviceMesh):
        """
        Apply expert parallelism.
        """
        if ep_mesh.ndim > 1:
            raise RuntimeError("local expert parallel sub-mesh must be 1-dimensional")
        num_shards = ep_mesh.size()
        if self.num_experts % num_shards != 0:
            raise OLMoConfigurationError(
                f"'num_experts' ({self.num_experts}) must be divisible by the expert parallel shard degree ({num_shards})."
            )

        self.experts_per_rank = self.num_experts // num_shards
        self.gradient_scale = 1.0 / num_shards

        self.register_parameter("w1", nn.Parameter(distribute_tensor(self.w1, ep_mesh, [Shard(0)])))  # type: ignore
        self.register_parameter("w2", nn.Parameter(distribute_tensor(self.w2, ep_mesh, [Shard(0)])))  # type: ignore
        self.register_parameter("w3", nn.Parameter(distribute_tensor(self.w3, ep_mesh, [Shard(0)])))  # type: ignore


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the expert outputs.

        :param x: The input of shape ``(num_local_experts, N, d_model)``.
        """
        # Scale gradients and get local tensors (in case of expert parallelism).
        # shape (all): (experts_per_rank, hidden_size, d_model)
        w1, w2, w3 = (
            get_local_tensor(self.scale_grad(self.w1)),
            get_local_tensor(self.scale_grad(self.w2)),
            get_local_tensor(self.scale_grad(self.w3)),
        )

        # Compute the MLP.
        x1 = torch.bmm(x, w1)
        x2 = torch.bmm(x, w3)
        x1 = F.silu(x1) * x2
        return torch.bmm(x1, w2)


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
        # shape (all): (experts_per_rank, hidden_size, d_model)
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
