import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    PrepareModuleOutput,
    parallelize_module,
)

from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.parallel import (
    flatten_mesh,
    get_pp_stage_mesh,
    get_world_mesh,
)
from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.ops import attach_auxiliary_loss

from ..buffer_cache import BufferCache
from ..config import ModuleConfig
from ..feed_forward import FeedForwardConfig
from ..layer_norm import LayerNorm, LayerNormConfig, LayerNormType
from .loss import MoELoadBalancingLossGranularity
from .mlp import DroplessMoEMLP, MoEMLP
from .parallel_mlp import ParallelDroplessMLP, ParallelMLP, ParallelMLPBase
from .router import MoERouterConfig

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

__all__ = ["LatentMoEConfig", "MoEBase", "MoE", "DroplessMoE", "MoEConfig", "MoEType"]


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
class LatentMoEConfig(Config):
    """Configuration for running the routed MoE branch in a latent dimension."""

    latent_dim: int
    """Input and output dimension of the routed expert payloads and expert MLPs."""

    bias: bool = False
    """Whether to use bias in the model-to-latent and latent-to-model projections."""

    up_proj_input_norm: Optional[LayerNormConfig] = None
    """Normalization applied in latent space immediately before the up projection.

    When normalization is enabled, a bias-free RMSNorm is used if this is ``None``. Set this to
    another layer norm configuration to select a different implementation.
    """

    up_proj_input_norm_enabled: bool = False
    """Whether to apply ``up_proj_input_norm`` before the latent up projection."""

    def resolved_up_proj_input_norm(self) -> LayerNormConfig:
        return self.up_proj_input_norm or LayerNormConfig(name=LayerNormType.rms, bias=False)

    def num_params(self, d_model: int) -> int:
        params = 2 * d_model * self.latent_dim
        if self.bias:
            params += self.latent_dim + d_model
        if self.up_proj_input_norm_enabled:
            params += self.resolved_up_proj_input_norm().num_params(self.latent_dim)
        return params


@dataclass
class MoEConfig(ModuleConfig):
    name: MoEType = MoEType.default
    """
    The name of the implementation.
    """
    num_experts: int = 1
    hidden_size: int = 256
    capacity_factor: Optional[float] = None
    router: MoERouterConfig = field(default_factory=MoERouterConfig)
    shared_mlp: Optional[FeedForwardConfig] = None
    latent_moe: Optional[LatentMoEConfig] = None
    lb_loss_weight: Optional[float] = 0.01
    lb_loss_granularity: MoELoadBalancingLossGranularity = (
        MoELoadBalancingLossGranularity.local_batch
    )
    z_loss_weight: Optional[float] = None
    scale_loss_by_num_layers: bool = True
    dtype: DType = DType.float32

    def num_params(self, d_model: int) -> int:
        latent_dim = d_model if self.latent_moe is None else self.latent_moe.latent_dim
        num_params = 0
        num_params += self.router.num_params(d_model, self.num_experts)
        num_params += 3 * latent_dim * self.hidden_size * self.num_experts
        if self.shared_mlp is not None:
            num_params += self.shared_mlp.num_params(d_model)
        if self.latent_moe is not None:
            num_params += self.latent_moe.num_params(d_model)
        return num_params

    def num_active_params(self, d_model: int) -> int:
        latent_dim = d_model if self.latent_moe is None else self.latent_moe.latent_dim
        return (
            self.num_params(d_model)
            - (3 * latent_dim * self.hidden_size * self.num_experts)
            + (3 * latent_dim * self.hidden_size * self.router.top_k)
        )

    def build(
        self,
        d_model: int,
        *,
        n_layers: int = 1,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "MoEBase":
        if self.latent_moe is not None and not (0 < self.latent_moe.latent_dim < d_model):
            raise OLMoConfigurationError(
                "latent_moe.latent_dim must be greater than 0 and less than d_model "
                f"({d_model}), got {self.latent_moe.latent_dim}"
            )
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
        num_experts: int,
        hidden_size: int,
        router: MoERouterConfig,
        shared_mlp: Optional[FeedForwardConfig] = None,
        init_device: str = "cpu",
        lb_loss_weight: Optional[float] = None,
        lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch,
        z_loss_weight: Optional[float] = None,
        n_layers: int = 1,
        scale_loss_by_num_layers: bool = True,
        dtype: torch.dtype = torch.float32,
        cache: Optional[BufferCache] = None,
        latent_moe: Optional[LatentMoEConfig] = None,
        **kwargs,
    ):
        super().__init__()
        if scale_loss_by_num_layers:
            if lb_loss_weight is not None:
                lb_loss_weight = lb_loss_weight / n_layers
            if z_loss_weight is not None:
                z_loss_weight = z_loss_weight / n_layers

        latent_dim = d_model if latent_moe is None else latent_moe.latent_dim
        self.latent_down_proj: Optional[nn.Linear]
        self.latent_up_proj_input_norm: Optional[LayerNorm]
        self.latent_up_proj: Optional[nn.Linear]
        if latent_moe is None:
            self.latent_down_proj = None
            self.latent_up_proj_input_norm = None
            self.latent_up_proj = None
        else:
            self.latent_down_proj = nn.Linear(
                d_model,
                latent_dim,
                bias=latent_moe.bias,
                dtype=dtype,
                device=init_device,
            )
            self.latent_up_proj_input_norm = (
                latent_moe.resolved_up_proj_input_norm().build(latent_dim, init_device=init_device)
                if latent_moe.up_proj_input_norm_enabled
                else None
            )
            self.latent_up_proj = nn.Linear(
                latent_dim,
                d_model,
                bias=latent_moe.bias,
                dtype=dtype,
                device=init_device,
            )

        self.router = router.build(
            d_model,
            num_experts,
            lb_loss_weight=lb_loss_weight,
            lb_loss_granularity=lb_loss_granularity,
            z_loss_weight=z_loss_weight,
            dtype=dtype,
            init_device=init_device,
        )
        self.experts = self._init_parallel_mlp(
            d_model=latent_dim,
            num_experts=num_experts,
            hidden_size=hidden_size,
            dtype=dtype,
            init_device=init_device,
            cache=cache,
            **kwargs,
        )
        self.shared_mlp = (
            None
            if shared_mlp is None
            else shared_mlp.build(d_model, dtype=dtype, init_device=init_device)
        )
        self._ep_enabled = False

    @property
    def num_experts(self) -> int:
        return self.router.num_experts

    @property
    def top_k(self) -> int:
        return self.router.top_k

    @property
    def ep_enabled(self) -> bool:
        return self._ep_enabled

    def warmup_cache(self, max_local_microbatch_size: int):
        self.experts.warmup_cache(max_local_microbatch_size)

    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        return self.router.compute_metrics(reset=reset)

    def reset_metrics(self):
        self.router.reset_metrics()

    def post_batch(self, dry_run: bool = False):
        """
        Should be called right after the final backward of a complete batch but before the optimizer step.
        """
        self.router.post_batch(dry_run=dry_run)

    @abstractmethod
    def _init_parallel_mlp(
        self,
        *,
        d_model: int,
        num_experts: int,
        hidden_size: int,
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
        expert_weights, expert_indices, batch_size_per_expert, router_aux_loss = self.router(
            x, loss_div_factor=loss_div_factor
        )
        routed_x = self.prepare_routed_input(x)

        if router_aux_loss is not None:
            routed_x = attach_auxiliary_loss(routed_x, router_aux_loss)

        shared_out: Optional[torch.Tensor] = None
        if self.shared_mlp is not None:
            shared_out = self.shared_mlp(x)

        out = self.experts(routed_x, expert_weights, expert_indices, batch_size_per_expert)
        out = self.restore_routed_output(out)

        if shared_out is not None:
            shared_out = shared_out / (self.top_k + 1)
            out = shared_out.add(out, alpha=self.top_k / (self.top_k + 1))

        return out

    def prepare_routed_input(self, model_input: torch.Tensor) -> torch.Tensor:
        if self.latent_down_proj is None:
            return model_input
        return self.latent_down_proj(model_input)

    def restore_routed_output(self, routed_output: torch.Tensor) -> torch.Tensor:
        if self.latent_up_proj is None:
            return routed_output
        if self.latent_up_proj_input_norm is not None:
            routed_output = self.latent_up_proj_input_norm(routed_output)
        # grouped-gemm experts return BF16 regardless of their parameter/input dtype. Restore the
        # configured projection dtype before applying the dense up projection.
        routed_output = routed_output.to(dtype=self.latent_up_proj.weight.dtype)
        return self.latent_up_proj(routed_output)

    def apply_pp(self, pp_mesh: DeviceMesh):
        world_mesh = get_world_mesh()
        assert world_mesh is not None
        stage_mesh = get_pp_stage_mesh(world_mesh, pp_mesh)
        group = flatten_mesh(stage_mesh).get_group()
        self.router.group = group

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        """
        Apply expert parallelism.
        """
        self.experts.apply_ep(ep_mesh, **kwargs)
        self._ep_enabled = True

    def prepare_experts_for_fsdp(self, **kwargs):
        """
        Should be called before wrapping this module with FSDP2.
        """
        self.experts.prepare_experts_for_fsdp(**kwargs)

    def prepare_experts_for_ddp(self, **kwargs):
        """
        Should be called before wrapping this module with DDP2.
        """
        self.experts.prepare_experts_for_ddp(**kwargs)

    def apply_cp(self, cp_mesh: DeviceMesh):
        self.router.apply_cp(cp_mesh)

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
        self.router.apply_tp(tp_mesh, float8_enabled=float8_enabled)

        # The latent projection stack operates independently on sequence-sharded tokens. Replicate
        # its parameters and return local tensors, matching the router and expert input contract.
        for module in (
            self.latent_down_proj,
            self.latent_up_proj_input_norm,
            self.latent_up_proj,
        ):
            if module is not None:
                parallelize_module(
                    module,
                    device_mesh=tp_mesh,
                    parallelize_plan=SequenceParallel(use_local_output=True),
                )

        # Expert parallel.
        self.experts.apply_tp(tp_mesh, float8_enabled=float8_enabled)

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
        router_flops = 6 * sum(p.numel() for p in self.router.parameters())
        latent_projection_flops = 6 * sum(
            p.numel()
            for projection in (self.latent_down_proj, self.latent_up_proj)
            if projection is not None
            for p in projection.parameters()
        )
        shared_mlp_flops = (
            self.shared_mlp.num_flops_per_token(seq_len) if self.shared_mlp is not None else 0
        )
        expert_flops = self.experts.num_flops_per_token(seq_len)
        return router_flops + latent_projection_flops + shared_mlp_flops + expert_flops


class MoE(MoEBase):
    """
    A basic MoE implementation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_experts: int,
        hidden_size: int,
        router: MoERouterConfig,
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
        latent_moe: Optional[LatentMoEConfig] = None,
    ):
        super().__init__(
            d_model=d_model,
            num_experts=num_experts,
            hidden_size=hidden_size,
            router=router,
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
            latent_moe=latent_moe,
        )

    def _init_parallel_mlp(  # type: ignore[override]
        self,
        *,
        d_model: int,
        num_experts: int,
        hidden_size: int,
        capacity_factor: float,
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
            top_k=self.router.top_k,
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
            top_k=self.router.top_k,
            cache=cache,
        )
