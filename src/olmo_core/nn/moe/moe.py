from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard
from torch.distributed.tensor.parallel import PrepareModuleOutput, parallelize_module

from olmo_core.config import Config, StrEnum
from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.exceptions import OLMoConfigurationError

from .loss import MoELoadBalancingLoss, MoELoss, MoERouterZLoss
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

    def num_active_params(self, d_model: int) -> int:
        return (
            self.num_params(d_model)
            - self.mlp.num_params(d_model, self.num_experts, self.hidden_size)
            + self.mlp.num_active_params(d_model, self.router.top_k, self.hidden_size)
        )

    def build(self, d_model: int, *, num_layers: int, init_device: str = "cpu") -> "MoEBase":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
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
        self.losses: List[MoELoss] = []
        if lb_loss_weight is not None:
            self.losses.append(
                MoELoadBalancingLoss(
                    loss_weight=lb_loss_weight,
                    num_layers=num_layers,
                    num_experts=num_experts,
                    top_k=self.router.top_k,
                )
            )
        if z_loss_weight is not None:
            self.losses.append(
                MoERouterZLoss(
                    loss_weight=z_loss_weight, num_layers=num_layers, num_experts=num_experts
                )
            )

    def compute_losses(
        self, total_bz: Union[int, torch.Tensor], reset: bool = True
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for loss_fn in self.losses:
            out.update(loss_fn.compute(total_bz, reset=reset))
        return out

    def reset_losses(self):
        for loss_fn in self.losses:
            loss_fn.reset()

    @classmethod
    @abstractmethod
    def _init_parallel_mlp(cls, mlp: MoEMLP) -> ParallelMLP:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        if self.training and self.losses:
            expert_logits = expert_logits.float()
            for loss_fn in self.losses:
                loss_fn.update(expert_logits, batch_size_per_expert=batch_size_per_expert)

        return out

    def apply_ep(self, ep_mesh: Optional[DeviceMesh] = None):
        """
        Apply expert parallelism.
        """
        self.experts.apply_ep(ep_mesh)

    def apply_tp(
        self,
        tp_mesh: Optional[DeviceMesh] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        parallelize_module(
            self.router,
            device_mesh=tp_mesh,
            parallelize_plan=SequenceParallel(use_local_output=True),
        )
        self.experts.apply_ep(tp_mesh)
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleOutput(
                output_layouts=(Shard(1),),
                desired_output_layouts=(output_layouts or Replicate(),),
                use_local_output=use_local_output,
            ),
        )


class DroplessMoE(MoEBase):
    """
    A dropless MoE implementation.
    """

    @classmethod
    def _init_parallel_mlp(cls, mlp: MoEMLP) -> ParallelMLP:
        return ParallelDroplessMLP(mlp=mlp)
