from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh

from ...config import Config, StrEnum
from ...exceptions import OLMoConfigurationError
from .mlp import MoEMLP, MoEMLPConfig
from .parallel_mlp import ParallelDroplessMLP, ParallelMLP
from .router import MoERouterConfig
from .shared_mlp import SharedMLPConfig

__all__ = ["MoEBase", "DroplessMoE", "MoEConfig", "MoEType"]


class MoEType(StrEnum):
    """
    An enumeration of the different MoE implementations.
    """

    dropless = "dropless"
    """
    ➡️ :class:`DroplessMoE`
    """


@dataclass
class MoEConfig(Config):
    name: MoEType = MoEType.dropless
    """
    The name of the implementation.
    """
    num_experts: int = 1
    hidden_size: int = 256
    router: MoERouterConfig = field(default_factory=MoERouterConfig)
    mlp: MoEMLPConfig = field(default_factory=MoEMLPConfig)
    shared_mlp: Optional[SharedMLPConfig] = None
    lb_loss_weight: Optional[float] = 1.0
    z_loss_weight: Optional[float] = None

    def num_params(self, d_model: int) -> int:
        num_params = 0

        num_params += self.router.num_params(d_model, self.num_experts)
        num_params += self.mlp.num_params(d_model, self.num_experts, self.hidden_size)
        if self.shared_mlp is not None:
            num_params += self.shared_mlp.num_params(d_model, self.hidden_size)

        return num_params

    def build(self, d_model: int, *, num_layers: int, init_device: str = "cpu") -> "MoEBase":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            dtype=kwargs.pop("dtype").as_pt(),
            d_model=d_model,
            num_layers=num_layers,
            init_device=init_device,
        )

        try:
            if self.name == MoEType.dropless:
                return DroplessMoE(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class MoEBase(nn.Module):
    """
    Base class for MoE implementations.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_experts: int,
        hidden_size: int,
        router: MoERouterConfig,
        mlp: MoEMLPConfig,
        num_layers: int,
        shared_mlp: Optional[SharedMLPConfig] = None,
        init_device: str = "cpu",
        lb_loss_weight: Optional[float] = None,
        z_loss_weight: Optional[float] = None,
    ):
        super().__init__()
        self.router = router.build(d_model, num_experts, init_device=init_device)
        self.experts = self._init_parallel_mlp(
            mlp.build(d_model, num_experts, hidden_size, init_device=init_device)
        )
        self.shared_experts = (
            None
            if shared_mlp is None
            else shared_mlp.build(d_model, hidden_size, init_device=init_device)
        )
        self.num_layers = num_layers
        self.lb_loss_weight = lb_loss_weight
        self.z_loss_weight = z_loss_weight

    @abstractmethod
    @classmethod
    def _init_parallel_mlp(cls, mlp: MoEMLP) -> ParallelMLP:
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Run the MoE on the input ``x`` of shape ``(*, d_model)``.

        :param x: The input of shape ``(*, d_model)``.

        :returns: The output of the MoE layer, the optional load-balancing loss, and the optional
            router Z-loss.
        """
        expert_logits, expert_weights, exper_indices = self.router(x)
        out, batch_size_per_expert = self.experts(x, expert_weights, exper_indices)
        if self.shared_experts is not None:
            out = self.shared_experts(x, out, self.router.top_k)

        lb_loss: Optional[torch.Tensor] = None
        z_loss: Optional[torch.Tensor] = None
        if self.training and (self.lb_loss_weight is not None or self.z_loss_weight is not None):
            expert_logits = expert_logits.float()

            # Compute load-balancing loss.
            if self.lb_loss_weight is not None:
                expert_scores = expert_logits.softmax(dim=-1)
                total_bz = expert_scores.shape[0]
                scale = (self.router.num_experts * self.lb_loss_weight) / (
                    self.num_layers * total_bz * self.router.top_k
                )
                lb_loss = scale * torch.dot(batch_size_per_expert, expert_scores)

            # Compute router Z-loss.
            if self.z_loss_weight is not None:
                z_loss = torch.logsumexp(expert_logits, dim=-1).square().mean() * self.z_loss_weight

        return out, lb_loss, z_loss

    def apply_ep(self, ep_mesh: DeviceMesh):
        """
        Apply expert parallelism.
        """
        self.experts.apply_ep(ep_mesh)


class DroplessMoE(MoEBase):
    """
    A dropless MoE implementation.
    """

    @classmethod
    def _init_parallel_mlp(cls, mlp: MoEMLP) -> ParallelMLP:
        return ParallelDroplessMLP(mlp=mlp)
