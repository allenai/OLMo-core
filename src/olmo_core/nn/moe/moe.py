import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    PrepareModuleOutput,
    parallelize_module,
)

from olmo_core.config import DType, StrEnum
from olmo_core.distributed.parallel import (
    flatten_mesh,
    get_pp_stage_mesh,
    get_world_mesh,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.ops import attach_auxiliary_loss

from ..buffer_cache import BufferCache
from ..config import ModuleConfig
from ..feed_forward import FeedForwardConfig
from .loss import MoELoadBalancingLossGranularity
from .mlp import DroplessMoEMLP, MoEMLP
from .parallel_mlp import ParallelDroplessMLP, ParallelMLP, ParallelMLPBase
from .router import MoERouterConfig

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

__all__ = ["MoEBase", "MoE", "DroplessMoE", "MoEConfig", "MoEType"]


log = logging.getLogger(__name__)


class MoEType(StrEnum):
    """
    An enumeration of the different MoE implementations.
    """

    default = "default"
    """
    ➡️ :class:`MoE`
    """

    dropless = "dropless"
    """
    ➡️ :class:`DroplessMoE`
    """


@dataclass
class MoEConfig(ModuleConfig):
    name: MoEType = MoEType.default
    """
    The name of the implementation.
    """
    num_experts_list: List[int] = field(default_factory=lambda: [1])
    hidden_sizes_list: List[int] = field(default_factory=lambda: [1])
    capacity_factor: Optional[float] = None
    routers_list: List[MoERouterConfig] = field(default_factory=lambda: [MoERouterConfig])
    shared_mlp: Optional[FeedForwardConfig] = None
    lb_loss_weight: Optional[float] = 0.01
    lb_loss_granularity: MoELoadBalancingLossGranularity = (
        MoELoadBalancingLossGranularity.local_batch
    )
    z_loss_weight: Optional[float] = None
    scale_loss_by_num_layers: bool = True
    dtype: DType = DType.float32

    def num_params(self, d_model: int) -> int:
        num_params = 0
        for i in range(len(self.num_experts_list)):
            router = self.routers_list[i]
            num_experts = self.num_experts_list[i]
            hidden_size = self.hidden_sizes_list[i]
            num_params += router.num_params(d_model, num_experts)
            num_params += (
                3 * d_model * hidden_size * num_experts
            )
        if self.shared_mlp is not None:
            num_params += self.shared_mlp.num_params(d_model)
        return num_params

    def num_active_params(self, d_model: int) -> int:
        active_params = self.num_params(d_model)
        for i in range(len(self.num_experts_list)):
            router = self.routers_list[i]
            num_experts = self.num_experts_list[i]
            hidden_size = self.hidden_sizes_list[i]
            active_params -= (3 * d_model * hidden_size * num_experts)
            active_params += (3 * d_model * hidden_size * router.top_k)
        return active_params

    def build(
        self,
        d_model: int,
        *,
        n_layers: int = 1,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "MoEBase":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            d_model=d_model,
            n_layers=n_layers,
            init_device=init_device,
            dtype=kwargs.pop("dtype").as_pt(),
            cache=cache,
        )

        try:
            if self.name == MoEType.default:
                return MoE(**kwargs)
            elif self.name == MoEType.dropless:
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
        num_experts_list: List[int],
        hidden_sizes_list: List[int],
        routers_list: List[MoERouterConfig],
        shared_mlp: Optional[FeedForwardConfig] = None,
        init_device: str = "cpu",
        lb_loss_weight: Optional[float] = None,
        lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch,
        z_loss_weight: Optional[float] = None,
        n_layers: int = 1,
        scale_loss_by_num_layers: bool = True,
        dtype: torch.dtype = torch.float32,
        cache: Optional[BufferCache] = None,
        **kwargs,
    ):
        super().__init__()
        if scale_loss_by_num_layers:
            if lb_loss_weight is not None:
                lb_loss_weight = lb_loss_weight / n_layers
            if z_loss_weight is not None:
                z_loss_weight = z_loss_weight / n_layers

        self.routers_list = nn.ModuleList()
        self.experts_list = nn.ModuleList()
        for i in range(len(num_experts_list)):
            num_experts = num_experts_list[i]
            hidden_size = hidden_sizes_list[i]
            router = routers_list[i]
            self.routers_list.append(router.build(
                d_model,
                num_experts,
                lb_loss_weight=lb_loss_weight,
                lb_loss_granularity=lb_loss_granularity,
                z_loss_weight=z_loss_weight,
                dtype=dtype,
                init_device=init_device,
            ))
            self.experts_list.append(self._init_parallel_mlp(
                d_model=d_model,
                num_experts=num_experts,
                hidden_size=hidden_size,
                router=router,
                dtype=dtype,
                init_device=init_device,
                cache=cache,
                **kwargs,
            ))
        self.shared_mlp = (
            None
            if shared_mlp is None
            else shared_mlp.build(d_model, dtype=dtype, init_device=init_device)
        )
        self._ep_enabled = False

    @property
    def num_experts(self) -> int:
        # return self.router.num_experts
        return sum(router.num_experts for router in self.routers_list)

    @property
    def top_k_sum(self) -> int:
        # return self.router.top_k
        return sum(router.top_k for router in self.routers_list)

    @property
    def ep_enabled(self) -> bool:
        return self._ep_enabled

    def warmup_cache(self, max_local_microbatch_size: int):
        for experts in self.experts_list:
            experts.warmup_cache(max_local_microbatch_size)

    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        all_metrics: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}
        for i, router in enumerate(self.routers_list):
            all_metrics.update(router.compute_metrics(reset=reset, prefix=f"router {i}"))
        return all_metrics

    def reset_metrics(self):
        # self.router.reset_metrics()
        for router in self.routers_list:
            router.reset_metrics()

    def post_batch(self, dry_run: bool = False):
        """
        Should be called right after the final backward of a complete batch but before the optimizer step.
        """
        for router in self.routers_list:
            router.post_batch(dry_run=dry_run)
        # self.router.post_batch(dry_run=dry_run)

    @abstractmethod
    def _init_parallel_mlp(
        self,
        *,
        d_model: int,
        num_experts: int,
        hidden_size: int,
        router: MoERouterConfig,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        **kwargs,
    ) -> ParallelMLPBase:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    ) -> torch.Tensor:
        """
        Run the MoE on the input ``x`` of shape ``(*, d_model)``.

        :param x: The input of shape ``(*, d_model)``.

        :returns: The output of the MoE layer, the optional load-balancing loss, and the optional
            router Z-loss.
        """
        outs = []
        weights = []
        for i in range(len(self.routers_list)):
            router = self.routers_list[i]
            experts = self.experts_list[i]
            expert_weights, expert_indices, batch_size_per_expert, router_aux_loss = router(
                x, loss_div_factor=loss_div_factor
            )

            if router_aux_loss is not None:
                x = attach_auxiliary_loss(x, router_aux_loss)

            experts_out = experts(x, expert_weights, expert_indices, batch_size_per_expert)
            outs.append(experts_out * router.top_k)
            weights.append(router.top_k)

        if self.shared_mlp is not None:
            outs.append(self.shared_mlp(x))
            weights.append(1)

        out = sum(outs) / sum(weights)
        return out

    def apply_pp(self, pp_mesh: DeviceMesh):
        world_mesh = get_world_mesh()
        assert world_mesh is not None
        stage_mesh = get_pp_stage_mesh(world_mesh, pp_mesh)
        group = flatten_mesh(stage_mesh).get_group()
        # self.router.group = group
        for router in self.routers_list:
            router.group = group

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        """
        Apply expert parallelism.
        """
        # self.experts.apply_ep(ep_mesh, **kwargs)
        # self._ep_enabled = True
        for experts in self.experts_list:
            experts.apply_ep(ep_mesh, **kwargs)
        self._ep_enabled = True

    def prepare_experts_for_fsdp(self, **kwargs):
        """
        Should be called before wrapping this module with FSDP2.
        """
        # self.experts.prepare_experts_for_fsdp(**kwargs)
        for experts in self.experts_list:
            experts.prepare_experts_for_fsdp(**kwargs)

    def prepare_experts_for_ddp(self, **kwargs):
        """
        Should be called before wrapping this module with DDP2.
        """
        # self.experts.prepare_experts_for_ddp(**kwargs)
        for experts in self.experts_list:
            experts.prepare_experts_for_ddp(**kwargs)

    def apply_cp(self, cp_mesh: DeviceMesh):
        # self.router.apply_cp(cp_mesh)
        for router in self.routers_list:
            router.apply_cp(cp_mesh)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        # Sequence parallel for the most part.
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=None if input_layout is None else (input_layout,),
                desired_input_layouts=(Shard(1),),
                use_local_output=False,
            ),
        )

        # Sequence parallel.
        for router in self.routers_list:
            router.apply_tp(
                tp_mesh,
                input_layout=input_layout,
                output_layout=output_layout,
                use_local_output=use_local_output,
                float8_enabled=float8_enabled,
            )
        # self.router.apply_tp(tp_mesh, float8_enabled=float8_enabled)

        # Expert parallel.
        for experts in self.experts_list:
            experts.apply_tp(
                tp_mesh,
                input_layout=input_layout,
                output_layout=output_layout,
                use_local_output=use_local_output,
                float8_enabled=float8_enabled,
            )
        # self.experts.apply_tp(tp_mesh, float8_enabled=float8_enabled)

        # Model parallel.
        if self.shared_mlp is not None:
            self.shared_mlp.apply_tp(
                tp_mesh,
                input_layout=Shard(1),
                output_layout=Shard(1),
                use_local_output=True,
                float8_enabled=float8_enabled,
            )

        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleOutput(
                output_layouts=(Shard(1),),
                desired_output_layouts=(output_layout or Replicate(),),
                use_local_output=use_local_output,
            ),
        )

    def num_flops_per_token(self, seq_len: int) -> int:
        router_flops = 6 * sum(
            p.numel() for router in self.routers_list for p in router.parameters()
        )
        shared_mlp_flops = (
            self.shared_mlp.num_flops_per_token(seq_len) if self.shared_mlp is not None else 0
        )
        expert_flops = sum(
            experts.num_flops_per_token(seq_len) for experts in self.experts_list
        )
        return router_flops + shared_mlp_flops + expert_flops


class MoE(MoEBase):
    """
    A basic MoE implementation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_experts_list: List[int],
        hidden_sizes_list: List[int],
        routers_list: List[MoERouterConfig],
        shared_mlp: Optional[FeedForwardConfig] = None,
        capacity_factor: float = 1.2,
        init_device: str = "cpu",
        lb_loss_weight: Optional[float] = None,
        lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch,
        z_loss_weight: Optional[float] = None,
        scale_loss_by_num_layers: bool = True,
        n_layers: int = 1,
        dtype: torch.dtype = torch.float32,
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(
            d_model=d_model,
            num_experts_list=num_experts_list,
            hidden_sizes_list=hidden_sizes_list,
            routers_list=routers_list,
            shared_mlp=shared_mlp,
            init_device=init_device,
            lb_loss_weight=lb_loss_weight,
            lb_loss_granularity=lb_loss_granularity,
            z_loss_weight=z_loss_weight,
            scale_loss_by_num_layers=scale_loss_by_num_layers,
            n_layers=n_layers,
            dtype=dtype,
            capacity_factor=capacity_factor,
            cache=cache,
        )

    def _init_parallel_mlp(  # type: ignore[override]
        self,
        *,
        d_model: int,
        num_experts: int,
        hidden_size: int,
        capacity_factor: float,
        router: MoERouterConfig,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> ParallelMLP:
        return ParallelMLP(
            mlp=MoEMLP(
                d_model=d_model,
                hidden_size=hidden_size,
                num_experts=num_experts,
                dtype=dtype,
                init_device=init_device,
            ),
            top_k=router.top_k,
            capacity_factor=capacity_factor,
            cache=cache,
        )


class DroplessMoE(MoEBase):
    """
    A dropless MoE implementation.
    """

    def _init_parallel_mlp(  # type: ignore[override]
        self,
        *,
        d_model: int,
        num_experts: int,
        hidden_size: int,
        router: MoERouterConfig,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> ParallelDroplessMLP:
        return ParallelDroplessMLP(
            mlp=DroplessMoEMLP(
                d_model=d_model,
                num_experts=num_experts,
                hidden_size=hidden_size,
                dtype=dtype,
                init_device=init_device,
            ),
            top_k=router.top_k,
            cache=cache,
        )
