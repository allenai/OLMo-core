import os
import threading
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Iterator, Optional, Tuple, Union, cast

import nvtx
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement, Shard
from torch.utils.checkpoint import checkpoint, CheckpointFunction

import olmo_core.nn.transformer.block
try:
    import torch.distributed._symmetric_memory as _symm_mem
except ImportError:
    _symm_mem = None  # type: ignore[assignment]

# Process-local cache for the EP->group "0" symmetric-memory alias.
# This makes repeated calls from multiple MoE blocks idempotent when they use
# the same EP group.
_EP_SYMM_GROUP0_ALIAS_LOCK = threading.Lock()
_EP_SYMM_GROUP0_ALIAS_RANKS: Optional[Tuple[int, ...]] = None

from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.utils import barrier, get_fs_local_rank, get_rank, get_world_size
from olmo_core.kernels import (
    ScaledGroupedMMPrequantizedRHS,
)
from olmo_core.kernels import olmo_symm_mem
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.fp8_weight import FP8WeightCacheSpec, FP8WeightStore
from olmo_core.ops import attach_auxiliary_loss
from olmo_core.utils import get_or_init_stream

from ..attention.base import SequenceMixerConfig
from ..buffer_cache import BufferCache
from ..functional import l2_normalize
from ..layer_norm import LayerNormConfig
from ..moe import MoERouterGatingFunction
from ..moe import MoERouterConfig as MoERouterConfigV1
from ..moe.loss import MoELoadBalancingLossGranularity
from ..moe.v2.routed_experts import (
    RoutedExperts,
    RoutedExpertsConfig,
    requires_host_side_split_sizes,
    use_torch_grouped_mm,
)
from ..moe.v2.router import MoERouterConfigV2, MoERouterV2
from ..moe.v2.shared_experts import SharedExperts
from ..moe.v2.shared_experts import SharedExpertsConfig
from ..moe.v2.fp8 import (
    MoERowwiseFP8Config,
    invalidate_rowwise_fp8_cache as _invalidate_rowwise_fp8_cache,
    normalize_rowwise_fp8_config,
    refresh_rowwise_fp8_cache as _refresh_rowwise_fp8_cache,
)
from ..moe.v2.ep_no_sync_buffers import (
    _NoSyncSymmSharedPool,
    resolve_ep_no_sync_rowwise_symm_options,
)
from ..moe.v2.ep_sync_1d import (
    combined_forward_ep_1d as _combined_forward_ep_1d,
)
from ..moe.v2.ep_sync_tbo import combined_forward_ep_tbo as _combined_forward_ep_tbo
from ..moe.v2.no_ep import combined_forward_no_ep as _combined_forward_no_ep
from ..moe.v2.ep_no_sync_1d import combined_forward_ep_no_sync_1d as _combined_forward_ep_no_sync_1d
from ..moe.v2.ep_no_sync_rowwise import (
    combined_forward_ep_no_sync_rowwise as _combined_forward_ep_no_sync_rowwise,
)
from ..moe.v2.ep_no_sync_rowwise_helpers import (
    add_ep_no_sync_rowwise_metrics,
    reset_ep_no_sync_rowwise_metrics,
)
from ..moe.v2.ep_no_sync_tbo_1d import (
    combined_forward_ep_no_sync_tbo as _combined_forward_ep_no_sync_tbo,
)
from ..moe.v2.ep_no_sync_tbo_rowwise import (
    combined_forward_ep_no_sync_tbo_rowwise as _combined_forward_ep_no_sync_tbo_rowwise,
)
from ..moe.v2.checkpointing import get_rowwise_checkpoint_state, is_checkpoint_recomputing
from ..moe.v2.activation_debug import maybe_dump_ep_no_sync_saved_activations
from olmo_core.nn.transformer.config import (
    TransformerBlockConfig,
    TransformerBlockType,
)


def _shared_up_gate_rhs(weight: torch.Tensor) -> torch.Tensor:
    return weight.unsqueeze(0)


def _shared_up_gate_rhs_for_dgrad(weight: torch.Tensor) -> torch.Tensor:
    return weight.transpose(0, 1).unsqueeze(0)


def _shared_down_rhs(weight: torch.Tensor) -> torch.Tensor:
    return weight


def _shared_down_rhs_for_dgrad(weight: torch.Tensor) -> torch.Tensor:
    return weight.transpose(1, 2)


_SHARED_UP_GATE_FP8_CACHE_SPECS = (
    FP8WeightCacheSpec("rhs", _shared_up_gate_rhs),
    FP8WeightCacheSpec("rhs_for_dgrad", _shared_up_gate_rhs_for_dgrad),
)
_SHARED_DOWN_FP8_CACHE_SPECS = (
    FP8WeightCacheSpec("rhs", _shared_down_rhs),
    FP8WeightCacheSpec("rhs_for_dgrad", _shared_down_rhs_for_dgrad),
)


@dataclass
class OLMoDDPTransformerBlockConfig(TransformerBlockConfig):

    shared_experts: Optional[SharedExpertsConfig] = None

    routed_experts: Optional[RoutedExpertsConfig] = None

    shared_experts_router: Optional[MoERouterConfigV2] = None

    routed_experts_router: Optional[MoERouterConfigV2] = None

    # TODO: These booleans overload the meaning of attention_norm and
    # feed_forward_norm. With use_pre_norm=True they are input/pre norms, while
    # the default/reordered path uses them as post-sublayer branch norms; peri
    # norm then adds separate *_input_norm modules for the pre norms. This makes
    # checkpoint conversion fragile across model families. Consider replacing
    # this with explicit placement fields, e.g.
    # attention_norm_placement/feed_forward_norm_placement in:
    #   {"none", "pre", "branch_post", "pre_and_branch_post"}
    # and stable module names like pre_attention_norm, post_attention_norm,
    # pre_feed_forward_norm, and post_feed_forward_norm, with compatibility
    # aliases for existing checkpoints.
    use_peri_norm: bool = False
    use_pre_norm: bool = False
    checkpoint_attn: bool = False
    checkpoint_permute_moe_unpermute: bool = False
    checkpoint_combined_ep_tbo: bool = False
    checkpoint_second_unpermute: bool = False
    ep_no_sync: bool = False
    ep_no_sync_use_2d_all_to_all: bool = False
    ep_no_sync_use_rowwise_all_to_all: bool = False
    ep_no_sync_rowwise_nblocks: int = 256
    ep_no_sync_share_dispatch_out: bool = False
    ep_no_sync_capacity_factor: float = 1.25
    ep_no_sync_shared_slots: int = 1
    ep_no_sync_share_combine_out: bool = False
    ep_no_sync_major_align: int = 1
    ep_no_sync_restore_unpermute_backend: str = "te_fused"
    ep_no_sync_rowwise_symm_dispatch_in: Optional[bool] = None
    ep_no_sync_rowwise_symm_combine_out: Optional[bool] = None
    ep_no_sync_rowwise_symm_combine_gather: Optional[bool] = None
    rowwise_fp8: Optional[MoERowwiseFP8Config] = None

    def build(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> olmo_core.nn.transformer.block.TransformerBlockBase:
        assert self.feed_forward is None and self.feed_forward_moe is None, (
            "OLMoDDPTransformerBlock does not support `feed_forward` or `feed_forward_moe`. "
            "Use `shared_experts` for dense/shared MLPs and `routed_experts` for routed MoE."
        )

        kwargs = self.as_dict(exclude_none=False, recurse=False)
        kwargs.pop("name")
        kwargs.pop("feed_forward") # from parent config
        kwargs.pop("feed_forward_moe") # from parent config
        kwargs.update(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            init_device=init_device,
            cache=cache,
        )


        if self.name == TransformerBlockType.moe_fused_v2:
            return OLMoDDPTransformerBlock(**kwargs)
        else:
            raise NotImplementedError(self.name)


    def num_params(self, d_model: int) -> int:
        block_params = 0

        block_params += self.sequence_mixer.num_params(d_model)
        if self.attention_norm is not None:
            block_params += self.attention_norm.num_params(d_model)
        if self.shared_experts is not None:
            block_params += self.shared_experts.num_params()
        if self.routed_experts is not None:
            block_params += self.routed_experts.num_params()
        if self.routed_experts_router is not None:
            block_params += self.routed_experts_router.num_params()
        if self.shared_experts_router is not None:
            block_params += self.shared_experts_router.num_params()
        if self.feed_forward_norm is not None:
            block_params += self.feed_forward_norm.num_params(d_model)
        if self.use_peri_norm:
            if self.attention_norm is not None:
                block_params += self.attention_norm.num_params(d_model)
            if self.feed_forward_norm is not None:
                block_params += self.feed_forward_norm.num_params(d_model)

        return block_params

    def num_active_params(self, d_model: int) -> int:
        block_params = 0

        block_params += self.sequence_mixer.num_params(d_model)
        if self.attention_norm is not None:
            block_params += self.attention_norm.num_params(d_model)
        if self.shared_experts is not None:
            block_params += self.shared_experts.num_params()
        if self.routed_experts is not None:
            assert self.routed_experts_router is not None, "routed_experts must have a router"
            block_params += self.routed_experts.num_active_params(self.routed_experts_router.top_k)
        if self.routed_experts_router is not None:
            block_params += self.routed_experts_router.num_params()
        if self.shared_experts_router is not None:
            block_params += self.shared_experts_router.num_params()
        if self.feed_forward_norm is not None:
            block_params += self.feed_forward_norm.num_params(d_model)
        if self.use_peri_norm:
            if self.attention_norm is not None:
                block_params += self.attention_norm.num_params(d_model)
            if self.feed_forward_norm is not None:
                block_params += self.feed_forward_norm.num_params(d_model)

        return block_params

    def flops_per_seq(self, d_model: int, seqlen: int) -> int:

        flops = 0

        # attention
        if hasattr(self.sequence_mixer, "flops_per_seq"):
            flops += self.sequence_mixer.flops_per_seq(d_model, seqlen)  # type: ignore[attr-defined]
        else:
            flops += self.sequence_mixer.build(
                d_model,
                layer_idx=0,
                n_layers=1,
                init_device="meta",
            ).num_flops_per_token(seqlen) * seqlen

        # router
        # (seq_len * d_model) * (d_model * num_total_experts)
        flops += 6 * seqlen * d_model * (
            (self.routed_experts_router.num_experts if self.routed_experts_router is not None else 0)
            + (self.shared_experts_router.num_experts if self.shared_experts_router is not None else 0)
        )

        # routed experts
        # (seq_len, d_model) * (d_model, expert_hidden_size) * top_k
        # x3 for swiglu (up, down, gate)
        # x3 for fwd+bwd
        # x2 for GEMM
        if self.routed_experts is not None:
            flops += (3 * 3 * 2) * seqlen * d_model * self.routed_experts.hidden_size *  self.routed_experts_router.top_k

        # shared experts
        # (seq_len, d_model) * (d_model, expert_hidden_size) * num_experts
        # x3 for swiglu (up, down, gate)
        # x3 for fwd+bwd
        # x2 for GEMM
        if self.shared_experts is not None:
            flops += (3 * 3 * 2) * seqlen * d_model * self.shared_experts.hidden_size * self.shared_experts.num_experts

        return flops


if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

class OLMoDDPTransformerBlock(olmo_core.nn.transformer.block.TransformerBlockBase):

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        attention_norm: LayerNormConfig,
        routed_experts_router: Optional[MoERouterConfigV2],
        shared_experts_router: Optional[MoERouterConfigV2],
        shared_experts: Optional[SharedExpertsConfig],
        routed_experts: Optional[RoutedExpertsConfig],
        feed_forward_norm: LayerNormConfig,
        use_peri_norm: bool = False,
        use_pre_norm: bool = False,
        dropout: float = 0.0,
        attention_residual_alpha: Optional[float] = None,
        feed_forward_residual_alpha: Optional[float] = None,
        checkpoint_attn = False,
        checkpoint_permute_moe_unpermute = False,
        checkpoint_combined_ep_tbo = False,
        checkpoint_second_unpermute=False,
        ep_no_sync: bool = False,
        ep_no_sync_use_2d_all_to_all: bool = False,
        ep_no_sync_use_rowwise_all_to_all: bool = False,
        ep_no_sync_rowwise_nblocks: int = 256,
        ep_no_sync_share_dispatch_out: bool = False,
        ep_no_sync_capacity_factor: float = 1.25,
        ep_no_sync_shared_slots: int = 1,
        ep_no_sync_share_combine_out: bool = False,
        ep_no_sync_major_align: int = 1,
        ep_no_sync_restore_unpermute_backend: str = "te_fused",
        ep_no_sync_rowwise_symm_dispatch_in: Optional[bool] = None,
        ep_no_sync_rowwise_symm_combine_out: Optional[bool] = None,
        ep_no_sync_rowwise_symm_combine_gather: Optional[bool] = None,
        rowwise_fp8: Optional[MoERowwiseFP8Config] = None,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)

        assert dropout == 0.0 or dropout is None, "OLMoDDPTransformerBlock does not support dropout"
        self.d_model = d_model
        self.block_idx = block_idx

        if attention_residual_alpha is not None:
            raise OLMoConfigurationError("OLMoDDPTransformerBlock does not support attention_residual_alpha")
        if feed_forward_residual_alpha is not None:
            raise OLMoConfigurationError("OLMoDDPTransformerBlock does not support feed_forward_residual_alpha")

        self.routed_experts: Optional[RoutedExperts]
        self.routed_experts_router: Optional[MoERouterV2]
        self.shared_experts: Optional[SharedExperts]
        self.shared_experts_router: Optional[MoERouterV2]
        self.rowwise_fp8 = normalize_rowwise_fp8_config(rowwise_fp8)
        self._rowwise_fp8_checked = False
        self._shared_rowwise_fp8_up_prequant: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._shared_rowwise_fp8_down_prequant: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._shared_rowwise_fp8_up_prequant_t: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._shared_rowwise_fp8_down_prequant_t: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._shared_rowwise_fp8_weight_versions: Optional[Tuple[int, int]] = None
        self._shared_rowwise_fp8_up_gate_weight: Optional[FP8WeightStore] = None
        self._shared_rowwise_fp8_down_weight: Optional[FP8WeightStore] = None
        if use_peri_norm and use_pre_norm:
            raise OLMoConfigurationError("use_peri_norm and use_pre_norm are mutually exclusive")
        self.use_peri_norm = use_peri_norm
        self.use_pre_norm = use_pre_norm

        ######## START: Attention ########
        self.attention = sequence_mixer.build(
            d_model, layer_idx=block_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        self.attention_norm = attention_norm.build(d_model, init_device=init_device)
        self.attention_input_norm = None
        if self.use_peri_norm:
            self.attention_input_norm = attention_norm.build(d_model, init_device=init_device)

        ######## END: Attention ########


        ######## START: MLP ########
        assert (routed_experts is not None) or (shared_experts is not None), "At least one of routed_experts or shared_experts must be provided"

        #### Optional: routed experts ####
        if routed_experts:
            # Routed Experts enabled
            assert routed_experts_router is not None, "Need routed_experts_router when using routed experts"
            routed_experts.rowwise_fp8 = normalize_rowwise_fp8_config(routed_experts.rowwise_fp8)
            if self.rowwise_fp8 is not None and routed_experts.rowwise_fp8 is None:
                routed_experts.rowwise_fp8 = self.rowwise_fp8
            self.routed_experts = routed_experts.build(init_device=init_device)
            owner_ref = weakref.ref(self)
            self.routed_experts.w_up_gate._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
            self.routed_experts.w_down._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
            self.routed_experts_router = routed_experts_router.build(init_device=init_device)
        else:
            # Routed Experts not enabled
            assert routed_experts_router is None, "Should not set routed_experts_router when not using routed experts"
            self.routed_experts = None
            self.routed_experts_router = None
        #### END: Optional: routed experts ####



        #### Optional: shared experts ####
        if shared_experts:
            # Shared Experts enabled
            self.shared_experts = shared_experts.build(init_device=init_device)
            owner_ref = weakref.ref(self)
            self.shared_experts.w_up_gate._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
            self.shared_experts.w_down._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
            self._shared_rowwise_fp8_up_gate_weight = FP8WeightStore(
                logical_name="shared_experts.w_up_gate",
                logical_shape=tuple(self.shared_experts.w_up_gate.shape),
                cache_specs=_SHARED_UP_GATE_FP8_CACHE_SPECS,
                anchor_param=self.shared_experts.w_up_gate,
                optimizer_enabled=(
                    self.rowwise_fp8 is not None
                    and self.rowwise_fp8.enabled
                    and self.rowwise_fp8.fp8_only_params
                ),
            )
            self._shared_rowwise_fp8_down_weight = FP8WeightStore(
                logical_name="shared_experts.w_down",
                logical_shape=tuple(self.shared_experts.w_down.shape),
                cache_specs=_SHARED_DOWN_FP8_CACHE_SPECS,
                anchor_param=self.shared_experts.w_down,
                optimizer_enabled=(
                    self.rowwise_fp8 is not None
                    and self.rowwise_fp8.enabled
                    and self.rowwise_fp8.fp8_only_params
                ),
            )
            # Shared Experts Router
            if shared_experts_router is not None:
                self.shared_experts_router = shared_experts_router.build(init_device=init_device)
            else:
                if shared_experts.num_experts > 1:
                    raise OLMoConfigurationError(
                        "Need shared_experts_router when using more than one shared expert"
                    )
                self.shared_experts_router = None
        else:
            # Shared Experts not enabled
            assert shared_experts_router is None, "Should not set shared_experts_router when not using shared experts"
            self.shared_experts = None
            self.shared_experts_router = None
        #### END: Optional: shared experts ####


        self.feed_forward_norm = feed_forward_norm.build(d_model, init_device=init_device)
        self.feed_forward_input_norm = None
        if self.use_peri_norm:
            self.feed_forward_input_norm = feed_forward_norm.build(d_model, init_device=init_device)
        ######## END: MLP ########

        self.ep_pg = None
        self._ep_enabled = False
        self.tp_pg = None
        self._tp_enabled = False


        # reuse the same event so that torch.compile can see the same object id and will not break the guard.
        self._dtoh_event: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event()) # cast to make pylance happy
        self._dtoh_event_send: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())
        self._dtoh_event_recv: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())
        self._before_rev_all2all_event: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())

        # same for tbo1
        self._dtoh_event1: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())
        self._dtoh_event_send1: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())
        self._dtoh_event_recv1: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())
        self._before_rev_all2all_event1: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())

        self.num_local_routed_experts: Optional[int] = self.routed_experts.num_experts if self.routed_experts else None


        self.checkpoint_attn = checkpoint_attn
        self.checkpoint_permute_moe_unpermute = checkpoint_permute_moe_unpermute
        self.checkpoint_combined_ep_tbo = checkpoint_combined_ep_tbo
        self.checkpoint_second_unpermute = checkpoint_second_unpermute
        self.ep_no_sync = ep_no_sync
        self.ep_no_sync_use_2d_all_to_all = ep_no_sync_use_2d_all_to_all
        self.ep_no_sync_use_rowwise_all_to_all = ep_no_sync_use_rowwise_all_to_all
        self.ep_no_sync_rowwise_nblocks = int(ep_no_sync_rowwise_nblocks)
        self.ep_no_sync_share_dispatch_out = ep_no_sync_share_dispatch_out
        if self.ep_no_sync_use_2d_all_to_all:
            raise OLMoConfigurationError(
                "ep_no_sync_use_2d_all_to_all=True is no longer supported: "
                "the 2D all_to_all path was removed due to correctness/performance issues."
            )
        self.ep_no_sync_capacity_factor = ep_no_sync_capacity_factor
        self.ep_no_sync_shared_slots = ep_no_sync_shared_slots
        self.ep_no_sync_share_combine_out = ep_no_sync_share_combine_out
        self.ep_no_sync_major_align = ep_no_sync_major_align
        self.ep_no_sync_restore_unpermute_backend = ep_no_sync_restore_unpermute_backend.lower()
        self.ep_no_sync_rowwise_symm_dispatch_in = ep_no_sync_rowwise_symm_dispatch_in
        self.ep_no_sync_rowwise_symm_combine_out = ep_no_sync_rowwise_symm_combine_out
        self.ep_no_sync_rowwise_symm_combine_gather = ep_no_sync_rowwise_symm_combine_gather
        self._ep_symm_group_name: Optional[str] = None
        self._ep_no_sync_symm_cache: Dict[str, torch.Tensor] = {}
        self._ep_no_sync_static_buffer_cache: Dict[Tuple[object, ...], object] = {}
        self._ep_no_sync_rowwise_fp8_static_buffer_cache: Dict[Tuple[object, ...], object] = {}
        self._ep_no_sync_rowwise_static_checkpoint_state: Optional[Tuple[bool, bool]] = None
        self._ep_no_sync_force_scratch_lifetime_buffers = False
        self._ep_no_sync_symm_lease_pools: Dict[str, object] = {}
        self._ep_no_sync_last_debug: Dict[str, torch.Tensor] = {}
        self._ep_no_sync_shared_pool: Optional[_NoSyncSymmSharedPool] = None
        self._ep_no_sync_shared_slot: int = 0
        self._ep_no_sync_te_backend_warned: bool = False
        # Row-wise EP no-sync metrics (populated only by combined_forward_ep_no_sync_rowwise()).
        self._ep_no_sync_rowwise_drop_tokens_sum: Optional[torch.Tensor] = None
        self._ep_no_sync_rowwise_total_tokens_sum: Optional[torch.Tensor] = None
        self._ep_no_sync_rowwise_symm_util_max: Optional[torch.Tensor] = None
        # self._ep_no_sync_forward_call_count: int = 0

        if self.ep_no_sync_capacity_factor <= 0:
            raise OLMoConfigurationError(
                f"ep_no_sync_capacity_factor must be > 0 (got {self.ep_no_sync_capacity_factor})"
            )
        if self.ep_no_sync_shared_slots < 1:
            raise OLMoConfigurationError(
                f"ep_no_sync_shared_slots must be >= 1 (got {self.ep_no_sync_shared_slots})"
            )
        if self.ep_no_sync_major_align < 1:
            raise OLMoConfigurationError(
                f"ep_no_sync_major_align must be >= 1 (got {self.ep_no_sync_major_align})"
            )
        if self.ep_no_sync_rowwise_nblocks < 0:
            raise OLMoConfigurationError(
                f"ep_no_sync_rowwise_nblocks must be >= 0 (got {self.ep_no_sync_rowwise_nblocks})"
            )
        if self.ep_no_sync_restore_unpermute_backend not in ("te_fused", "te_unfused", "cuda"):
            raise OLMoConfigurationError(
                "ep_no_sync_restore_unpermute_backend must be one of "
                "'te_fused'|'te_unfused'|'cuda' "
                f"(got {self.ep_no_sync_restore_unpermute_backend!r})"
            )

    def invalidate_rowwise_fp8_cache(self) -> None:
        _invalidate_rowwise_fp8_cache(self)

    def refresh_rowwise_fp8_cache(self) -> None:
        _refresh_rowwise_fp8_cache(self)

    def _shared_fp8_only_params_enabled(self) -> bool:
        cfg = self.rowwise_fp8
        return cfg is not None and cfg.enabled and cfg.fp8_only_params

    def _sync_shared_rowwise_fp8_weight_anchors(self) -> None:
        if (
            self.shared_experts is None
            or self._shared_rowwise_fp8_up_gate_weight is None
            or self._shared_rowwise_fp8_down_weight is None
        ):
            return
        fp8_only_params = self._shared_fp8_only_params_enabled()
        owner_ref = weakref.ref(self)
        self.shared_experts.w_up_gate._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        self.shared_experts.w_down._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        self._shared_rowwise_fp8_up_gate_weight.anchor_param = self.shared_experts.w_up_gate
        self._shared_rowwise_fp8_up_gate_weight.logical_shape = tuple(self.shared_experts.w_up_gate.shape)
        self._shared_rowwise_fp8_up_gate_weight.optimizer_enabled = fp8_only_params
        self._shared_rowwise_fp8_down_weight.anchor_param = self.shared_experts.w_down
        self._shared_rowwise_fp8_down_weight.logical_shape = tuple(self.shared_experts.w_down.shape)
        self._shared_rowwise_fp8_down_weight.optimizer_enabled = fp8_only_params
        if fp8_only_params:
            self.shared_experts.w_up_gate.requires_grad_(False)
            self.shared_experts.w_down.requires_grad_(False)

    def named_fp8_weight_stores(self) -> Iterator[tuple[str, FP8WeightStore]]:
        if self.routed_experts is not None:
            for name, weight in self.routed_experts.named_fp8_weight_stores():
                yield f"routed_experts.{name}", weight
        self._sync_shared_rowwise_fp8_weight_anchors()
        if self._shared_rowwise_fp8_up_gate_weight is not None:
            yield "shared_experts.w_up_gate", self._shared_rowwise_fp8_up_gate_weight
        if self._shared_rowwise_fp8_down_weight is not None:
            yield "shared_experts.w_down", self._shared_rowwise_fp8_down_weight

    def named_mxfp8_expert_weights(self) -> Iterator[tuple[str, object]]:
        yield from self.named_fp8_weight_stores()

    def zero_fp8_weight_store_grads(self, set_to_none: bool = True) -> None:
        for _, weight in self.named_fp8_weight_stores():
            weight.zero_grad(set_to_none=set_to_none)

    def zero_mxfp8_expert_weight_grads(self, set_to_none: bool = True) -> None:
        self.zero_fp8_weight_store_grads(set_to_none=set_to_none)

    def set_fp8_weight_store_main_grads_to_none(self) -> None:
        for _, weight in self.named_fp8_weight_stores():
            weight.set_main_grad_to_none()

    def set_mxfp8_expert_weight_main_grads_to_none(self) -> None:
        self.set_fp8_weight_store_main_grads_to_none()

    def disable_fp8_weight_anchor_grads(self) -> None:
        if self.routed_experts is not None:
            self.routed_experts.disable_mxfp8_expert_anchor_grads()
        if self._shared_fp8_only_params_enabled() and self.shared_experts is not None:
            self._sync_shared_rowwise_fp8_weight_anchors()
            self.shared_experts.w_up_gate.grad = None
            self.shared_experts.w_down.grad = None
            if hasattr(self.shared_experts.w_up_gate, "_main_grad_fp32"):
                self.shared_experts.w_up_gate._main_grad_fp32 = None  # type: ignore[attr-defined]
            if hasattr(self.shared_experts.w_down, "_main_grad_fp32"):
                self.shared_experts.w_down._main_grad_fp32 = None  # type: ignore[attr-defined]

    def disable_mxfp8_expert_anchor_grads(self) -> None:
        self.disable_fp8_weight_anchor_grads()

    def release_fp8_weight_anchor_storage(self) -> None:
        if self.routed_experts is not None:
            self.routed_experts.release_mxfp8_expert_anchor_storage()
        if not self._shared_fp8_only_params_enabled() or self.shared_experts is None:
            return
        self._sync_shared_rowwise_fp8_weight_anchors()
        if self._shared_rowwise_fp8_up_gate_weight is not None:
            self._shared_rowwise_fp8_up_gate_weight.release_anchor_storage()
        if self._shared_rowwise_fp8_down_weight is not None:
            self._shared_rowwise_fp8_down_weight.release_anchor_storage()
        self.shared_experts._fp8_anchor_storage_released = True

    def release_mxfp8_expert_anchor_storage(self) -> None:
        self.release_fp8_weight_anchor_storage()

    def _sync_fp8_weight_store_grad_from_anchor(
        self,
        weight: FP8WeightStore,
        anchor: torch.nn.Parameter,
    ) -> None:
        anchor_grad = getattr(anchor, "_main_grad_fp32", None)
        if anchor_grad is None:
            anchor_grad = anchor.grad
        if anchor_grad is None:
            return
        weight.replace_wgrad(anchor_grad)

    def sync_fp8_weight_store_grads_from_anchor(self) -> None:
        if self.routed_experts is not None:
            self.routed_experts.sync_mxfp8_expert_weight_grads_from_anchor()
        if (
            self._shared_fp8_only_params_enabled()
            and self.shared_experts is not None
            and self._shared_rowwise_fp8_up_gate_weight is not None
            and self._shared_rowwise_fp8_down_weight is not None
        ):
            self._sync_fp8_weight_store_grad_from_anchor(
                self._shared_rowwise_fp8_up_gate_weight,
                self.shared_experts.w_up_gate,
            )
            self._sync_fp8_weight_store_grad_from_anchor(
                self._shared_rowwise_fp8_down_weight,
                self.shared_experts.w_down,
            )

    def sync_mxfp8_expert_weight_grads_from_anchor(self) -> None:
        self.sync_fp8_weight_store_grads_from_anchor()

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        self.zero_mxfp8_expert_weight_grads(set_to_none=set_to_none)

    def purge_cuda_events(self):
        # set all events to None (so that the model can be deepcopied)
        self._dtoh_event = None # type: ignore[assignment]
        self._dtoh_event_send = None # type: ignore[assignment]
        self._dtoh_event_recv = None # type: ignore[assignment]
        self._before_rev_all2all_event = None # type: ignore[assignment]

        self._dtoh_event1 = None # type: ignore[assignment]
        self._dtoh_event_send1 = None # type: ignore[assignment]
        self._dtoh_event_recv1 = None # type: ignore[assignment]
        self._before_rev_all2all_event1 = None # type: ignore[assignment]

    def install_cuda_events(self):
        self._dtoh_event = cast(torch.cuda.Event, torch.cuda.Event())
        self._dtoh_event_send = cast(torch.cuda.Event, torch.cuda.Event())
        self._dtoh_event_recv = cast(torch.cuda.Event, torch.cuda.Event())
        self._before_rev_all2all_event = cast(torch.cuda.Event, torch.cuda.Event())

        self._dtoh_event1 = cast(torch.cuda.Event, torch.cuda.Event())
        self._dtoh_event_send1 = cast(torch.cuda.Event, torch.cuda.Event())
        self._dtoh_event_recv1 = cast(torch.cuda.Event, torch.cuda.Event())
        self._before_rev_all2all_event1 = cast(torch.cuda.Event, torch.cuda.Event())

    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        # compute shared and routed experts metrics
        # metrics_shared = self.shared_experts.compute_metrics(reset=reset)
        if self.routed_experts_router:
            metrics_routed = self.routed_experts_router.compute_metrics(reset=reset)
        else:
            metrics_routed = {}
        out = dict(metrics_routed)

        add_ep_no_sync_rowwise_metrics(self, out, ReduceType)

        if reset:
            reset_ep_no_sync_rowwise_metrics(self)

        # metrics = {
        #     "shared": metrics_shared,
        #     "routed": metrics_routed,
        # }
        return out

    def reset_metrics(self):
        # if self.shared_experts_router:
        #     self.shared_experts_router.reset_metrics()
        if self.routed_experts_router:
            self.routed_experts_router.reset_metrics()
        reset_ep_no_sync_rowwise_metrics(self)

    @property
    def has_routed_experts(self) -> bool:
        return self.routed_experts is not None

    @property
    def has_shared_experts(self) -> bool:
        return self.shared_experts is not None

    @property
    def is_shared_only(self) -> bool:
        return self.has_shared_experts and not self.has_routed_experts

    @property
    def is_moe(self) -> bool:
        return self.has_routed_experts

    @property
    def ep_enabled(self) -> bool:
        return self._ep_enabled

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    def get_dense_stream(self, for_x1=False) -> torch.cuda.Stream:
        if for_x1: # not used for now
            return get_or_init_stream(id='dense_x1', priority=20)
        else:
            return get_or_init_stream(id='dense', priority=20)

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.routed_experts:
            if self.ep_enabled:
                if self.ep_no_sync and self.training: # in eval mode, different ranks might get different input token counts, and no-sync can freeze
                    no_sync_forward = (
                        self.combined_forward_ep_no_sync_rowwise
                        if self.ep_no_sync_use_rowwise_all_to_all
                        else self.combined_forward_ep_no_sync_1d
                    )
                    debug_out = maybe_dump_ep_no_sync_saved_activations(
                        self,
                        x,
                        loss_div_factor=loss_div_factor,
                        forward_kwargs=kwargs,
                        no_sync_forward=no_sync_forward,
                    )
                    if debug_out is not None:
                        return debug_out
                    return no_sync_forward(
                        x, loss_div_factor=loss_div_factor, **kwargs
                    )
                return self.combined_forward_ep_1d(x, loss_div_factor=loss_div_factor, **kwargs)
            else:
                return self.combined_forward_no_ep(x, loss_div_factor=loss_div_factor, **kwargs)
        else:
            # only shared_experts
            return self.combined_forward_shared_only(x, loss_div_factor=loss_div_factor, **kwargs)

    def apply_pp(self, pp_mesh: DeviceMesh):
        pass # nothing to do

    def _ensure_ep_no_sync_symm_backend(self):
        if olmo_symm_mem.is_enabled():
            if not torch.cuda.is_available():
                raise RuntimeError("EP no-sync requires CUDA")
            return

        if _symm_mem is None:
            raise RuntimeError(
                "EP no-sync requires torch.distributed._symmetric_memory, but it is unavailable"
            )

        if not torch.cuda.is_available():
            raise RuntimeError("EP no-sync requires CUDA")

        device = torch.device("cuda", torch.cuda.current_device())
        current_backend = _symm_mem.get_backend(device)
        if current_backend is not None and current_backend.upper() == "NVSHMEM":
            return

        if not _symm_mem.is_nvshmem_available():
            raise RuntimeError(
                "EP no-sync requires NVSHMEM-backed symmetric memory for "
                "all_to_all_vdev, but NVSHMEM is not available in this "
                "PyTorch build/environment."
            )

        try:
            _symm_mem.set_backend("NVSHMEM")
        except Exception as e:
            try:
                backend_after = _symm_mem.get_backend(device)
            except Exception:
                backend_after = None
            raise RuntimeError(
                "EP no-sync requires NVSHMEM-backed symmetric memory for "
                "all_to_all_vdev. Failed to switch backend to NVSHMEM "
                f"(current={current_backend}, after_error={backend_after}): {e}. "
                "Call torch.distributed._symmetric_memory.set_backend('NVSHMEM') "
                "before any symmetric-memory allocations."
            ) from e

    def _try_alias_ep_group_as_world_for_symm_mem(self) -> bool:
        """
        Try to alias EP group metadata as symmetric-memory group "0".

        NVSHMEM-backed ``symm_mem.empty()`` bootstraps through group "0" before
        the later tensor rendezvous sees the EP process group. Each process can
        safely alias its local EP island as group "0" before the first symmetric
        allocation.
        """
        global _EP_SYMM_GROUP0_ALIAS_RANKS
        if _symm_mem is None or self.ep_pg is None:
            return False
        if dist.group.WORLD is None:
            return False
        if self.ep_pg.group_name == dist.group.WORLD.group_name:
            return True

        try:
            import torch.distributed.distributed_c10d as c10d
        except Exception:
            return False

        try:
            alias_ranks = tuple(sorted(c10d._world.pg_group_ranks[self.ep_pg].keys()))
            with _EP_SYMM_GROUP0_ALIAS_LOCK:
                # Idempotent success: alias already installed for the same EP group.
                if _EP_SYMM_GROUP0_ALIAS_RANKS == alias_ranks:
                    return True

                # If group "0" is already registered by a different context,
                # do not overwrite global process state.
                if _symm_mem.is_symm_mem_enabled_for_group("0"):
                    return False

                if not self._register_process_group_for_symm_mem(self.ep_pg, "0"):
                    return False
                _EP_SYMM_GROUP0_ALIAS_RANKS = alias_ranks
                return True
        except Exception:
            return False

    @staticmethod
    def _register_process_group_for_symm_mem(
        group: dist.ProcessGroup,
        group_name: str,
    ) -> bool:
        if _symm_mem is None:
            return False
        if _symm_mem.is_symm_mem_enabled_for_group(group_name):
            return True
        try:
            import torch.distributed.distributed_c10d as c10d
            from torch._C._distributed_c10d import _SymmetricMemory
        except Exception:
            return False

        try:
            global_ranks = sorted(c10d._world.pg_group_ranks[group].keys())
            global_ranks_str = "_".join(map(str, global_ranks))
            store = c10d.PrefixStore(
                f"symmetric_memory-{global_ranks_str}",
                c10d._get_process_group_store(group),
            )
            _SymmetricMemory.set_group_info(
                group_name,
                dist.get_rank(group),
                dist.get_world_size(group),
                store,
            )
            group_to_store = getattr(_symm_mem, "_group_name_to_store", None)
            if isinstance(group_to_store, dict):
                group_to_store[group_name] = store
            return True
        except Exception:
            return False

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        assert self.routed_experts is not None, "ep can only be applied when routed_experts is enabled"
        ep_dp_mesh = ep_mesh['ep_dp']
        ep_mp_mesh = ep_mesh['ep_mp']
        ep_pg = kwargs.get("ep_pg")
        self.ep_mesh = ep_mesh
        self.routed_experts.apply_ep(
            ep_mesh
        )
        owner_ref = weakref.ref(self)
        self.routed_experts.w_up_gate._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        self.routed_experts.w_down._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        _invalidate_rowwise_fp8_cache(self)
        self.num_local_routed_experts = self.routed_experts.num_local_experts
        self._ep_enabled = True
        self.ep_pg = ep_pg if ep_pg is not None else ep_mp_mesh.get_group()

        if self.ep_no_sync:
            if olmo_symm_mem.is_enabled() and not self.ep_no_sync_use_rowwise_all_to_all:
                raise RuntimeError(
                    "OLMo-owned symmetric memory currently supports only the rowwise "
                    "EP no-sync path. Set OLMO_USE_OWN_SYMM_MEM=0 to use the legacy "
                    "torch.ops.symm_mem.all_to_all_vdev path."
                )
            if _symm_mem is None and not olmo_symm_mem.is_enabled():
                raise RuntimeError(
                    "EP no-sync requires torch.distributed._symmetric_memory, but it is unavailable"
                )
            self._ensure_ep_no_sync_symm_backend()
            assert self.ep_pg is not None
            group_name = self.ep_pg.group_name
            assert dist.group.WORLD is not None, "torch.distributed.group.WORLD must be initialized for EP no-sync to work"
            world_group_name = dist.group.WORLD.group_name
            alias_group0_active = False
            try:
                if olmo_symm_mem.is_enabled():
                    olmo_symm_mem.register_group(self.ep_pg, device=torch.device("cuda", torch.cuda.current_device()))
                else:
                    if not self._register_process_group_for_symm_mem(self.ep_pg, group_name):
                        raise RuntimeError(
                            f"Failed to register EP group '{group_name}' for symmetric memory support"
                        )
                    if os.getenv("OLMO_EP_NO_SYNC_ALIAS_GROUP0", "1").lower() not in ("", "0", "false", "no"):
                        alias_group0_active = self._try_alias_ep_group_as_world_for_symm_mem()
                        if not alias_group0_active:
                            raise RuntimeError(
                                f"Failed to alias EP group '{group_name}' as group '0' for symmetric memory support"
                            )

            except Exception as e:
                raise RuntimeError(
                    f"Failed to enable symmetric memory for EP group '{group_name}' "
                    f"(world='{world_group_name}', alias_group0_active={alias_group0_active}) "
                    f"(block={self.block_idx}, rank={get_rank(self.ep_pg)}): {e}"
                ) from e
            self._ep_symm_group_name = group_name
            if self.ep_no_sync_use_rowwise_all_to_all:
                resolve_ep_no_sync_rowwise_symm_options(self)
            self._ep_no_sync_symm_cache.clear()
            self._ep_no_sync_static_buffer_cache.clear()
            self._ep_no_sync_rowwise_fp8_static_buffer_cache.clear()
            self._ep_no_sync_symm_lease_pools.clear()
            self._ep_no_sync_shared_pool = None
            self._ep_no_sync_shared_slot = 0
            self._ep_no_sync_te_backend_warned = False

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        raise NotImplementedError("TP is not supported in MoEFusedV1TransformerBlock")

    def apply_cp(self, cp_mesh: DeviceMesh, ring=None, uly=None):
        self.attention.apply_cp(cp_mesh, ring=ring, uly=uly)
        if self.routed_experts_router is not None:
            self.routed_experts_router.apply_cp(cp_mesh)
        if self.shared_experts_router is not None:
            self.shared_experts_router.apply_cp(cp_mesh)

    def apply_fsdp(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError("FSDP is not supported in OLMoDDPTransformerBlock")

    def apply_compile(self):
        self.compile(
            fullgraph=False,
            # dynamic=False
        )

        # NOTE: the tbo might be called by the outer model directly (by block.combined_forward_ep_tbo(x, ...) instead of block(x, ...)), so need to compile it here as well
        self.combined_forward_ep_tbo = torch.compile(self.combined_forward_ep_tbo)
        self._res_norm_attn = torch.compile(self._res_norm_attn)


    @property
    def ep_world_size(self) -> int:
        if self.ep_pg is not None:
            return get_world_size(self.ep_pg)
        else:
            return 1

    def _attach_routed_aux_loss(
        self,
        x: torch.Tensor,
        routed_expert_router_aux_loss_info: Optional[Tuple[object, ...]],
        *,
        accumulate_metrics: Optional[bool] = None,
    ) -> torch.Tensor:
        if routed_expert_router_aux_loss_info is None:
            return x
        assert self.routed_experts_router is not None
        if accumulate_metrics is None:
            accumulate_metrics = not is_checkpoint_recomputing()
        routed_expert_router_aux_loss = self.routed_experts_router.compute_aux_loss(
            *routed_expert_router_aux_loss_info,
            accumulate_metrics=accumulate_metrics,
        )
        if routed_expert_router_aux_loss is None:
            return x
        return attach_auxiliary_loss(x, routed_expert_router_aux_loss)

    @torch.compiler.disable
    def sync_dtoh_event(self):
        assert self._dtoh_event is not None
        dtoh_event = cast(torch.cuda.Event, self._dtoh_event)
        dtoh_event.synchronize()

    # -------------------------------------------------------------------------
    # Forward path entry points
    # -------------------------------------------------------------------------
    def combined_forward_shared_only(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward for shared-only blocks, equivalent to a dense DDP-stack MLP."""
        assert self.routed_experts is None
        assert self.routed_experts_router is None
        assert self.shared_experts is not None

        block_inp = x
        del x

        attn_res_out = self._res_norm_attn(block_inp, **kwargs)
        kwargs.pop("max_doc_len", None)
        kwargs.pop("cu_doc_lens", None)
        mlp_inp = self._prepare_moe_input(attn_res_out)

        if self.shared_experts_router:
            shared_expert_weights, _, _, _ = self.shared_experts_router(
                mlp_inp,
                True,
                loss_div_factor=loss_div_factor,
            )
        else:
            shared_expert_weights = None

        if mlp_inp.is_cuda:
            wait_stream_no_compile(
                this_stream=self.get_dense_stream(),
                other_stream=torch.cuda.current_stream(),
            )
            with torch.cuda.stream(self.get_dense_stream()):
                shared_out = self.shared_experts(mlp_inp)
                mlp_out = self._mix_shared_out(
                    shared_out,
                    shared_expert_weights,
                    attn_res_out.shape,
                )
            wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())
        else:
            shared_out = self.shared_experts(mlp_inp)
            mlp_out = self._mix_shared_out(
                shared_out,
                shared_expert_weights,
                attn_res_out.shape,
            )

        return self._res_norm_mlp(attn_res_out, mlp_out)

    def combined_forward_no_ep(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return _combined_forward_no_ep(
            self,
            x,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    def combined_forward_ep_no_sync_1d(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return _combined_forward_ep_no_sync_1d(
            self,
            x,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    def combined_forward_ep_no_sync_rowwise(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        checkpoint_state = self._ep_no_sync_rowwise_static_checkpoint_state
        if checkpoint_state is None:
            checkpoint_state = get_rowwise_checkpoint_state()
        activation_checkpointing, accumulate_routed_aux_loss_metrics = checkpoint_state
        return _combined_forward_ep_no_sync_rowwise(
            self,
            x,
            activation_checkpointing=activation_checkpointing,
            accumulate_routed_aux_loss_metrics=accumulate_routed_aux_loss_metrics,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    def combined_forward_ep_no_sync_tbo(
        self,
        x0: torch.Tensor,
        x1_ctx: object,
        x1_is_fresh: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, object]:
        if self.ep_no_sync_use_rowwise_all_to_all:
            return _combined_forward_ep_no_sync_tbo_rowwise(
                self,
                x0,
                x1_ctx,
                x1_is_fresh,
                loss_div_factor=loss_div_factor,
                **kwargs,
            )
        return _combined_forward_ep_no_sync_tbo(
            self,
            x0,
            x1_ctx,
            x1_is_fresh,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    def combined_forward_ep_1d(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return _combined_forward_ep_1d(
            self,
            x,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    def combined_forward_ep_tbo(
        self,
        x0: torch.Tensor,
        x1_ctx: object,
        x1_is_fresh: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, object]:
        return _combined_forward_ep_tbo(
            self,
            x0,
            x1_ctx,
            x1_is_fresh,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    def checkpointed_combined_forward_ep_tbo(
        self,
        x0: torch.Tensor,
        x1_ctx: object,
        x1_is_fresh: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, object]:
        if self.checkpoint_combined_ep_tbo:
            out = checkpoint(
                self.combined_forward_ep_tbo,
                x0,
                x1_ctx,
                x1_is_fresh,
                loss_div_factor=loss_div_factor,
                use_reentrant=False,
                **kwargs,
            )
            return cast(Tuple[torch.Tensor, object], out)
        else:
            return self.combined_forward_ep_tbo(
                x0,
                x1_ctx,
                x1_is_fresh,
                loss_div_factor=loss_div_factor,
                **kwargs,
            )

    # -------------------------------------------------------------------------
    # Shared forward helpers
    # -------------------------------------------------------------------------
    def _prepare_moe_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_peri_norm:
            assert self.feed_forward_input_norm is not None
            return self.feed_forward_input_norm(x)
        if self.use_pre_norm:
            return self.feed_forward_norm(x)
        return x

    def _merge_routed_and_shared(
        self,
        routed_out: torch.Tensor,
        mixed_shared_out: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.shared_experts is None:
            return routed_out
        if mixed_shared_out is None:
            raise RuntimeError("shared_experts is enabled but mixed_shared_out is missing")
        return routed_out + mixed_shared_out

    def _mix_shared_out(
        self,
        shared_out: torch.Tensor,
        shared_expert_weights: Optional[torch.Tensor],
        output_shape: torch.Size,
    ) -> torch.Tensor:
        assert self.shared_experts is not None
        B, S, D = output_shape
        if self.shared_experts_router is None:
            if self.shared_experts.num_experts != 1:
                raise RuntimeError("shared_experts_router is required for multiple shared experts")
            return shared_out.squeeze(0)

        if shared_expert_weights is None:
            raise RuntimeError("shared_experts_router is enabled but shared expert weights are missing")
        _, _, E_s = shared_expert_weights.shape
        return torch.bmm(
            shared_expert_weights.to(shared_out.dtype).reshape(B * S, 1, E_s),
            shared_out.permute(1, 2, 0, 3).contiguous().view(B * S, E_s, D),
        ).squeeze(1).view(B, S, D)

    def _res_norm_mlp(self, residual: torch.Tensor, mlp_out: torch.Tensor) -> torch.Tensor:
        if self.use_pre_norm:
            return residual + mlp_out
        return residual + self.feed_forward_norm(mlp_out)

    def _res_norm_attn(
        self,
        block_inp,
        **kwargs
    ) -> torch.Tensor:
        attn_in = block_inp
        if self.use_peri_norm:
            assert self.attention_input_norm is not None
            attn_in = self.attention_input_norm(block_inp)
        elif self.use_pre_norm:
            attn_in = self.attention_norm(block_inp)
            return block_inp + self.attention(attn_in, **kwargs)
        attn_res_out = block_inp + self.attention_norm(self.attention(attn_in, **kwargs))
        return attn_res_out

    def _checkpointed_res_norm_attn(
        self,
        block_inp,
        **kwargs
    ) -> torch.Tensor:
        if self.checkpoint_attn:
            out = checkpoint(
                self._res_norm_attn,
                block_inp,
                use_reentrant=False,
                **kwargs,
            )
            return cast(torch.Tensor, out)
        else:
            return self._res_norm_attn(block_inp, **kwargs)


    def post_batch(self, dry_run: bool = False):
        """
        Should be called right after the final backward of a complete batch but before the optimizer step.
        """
        if self.shared_experts_router:
            self.shared_experts_router.post_batch(dry_run=dry_run)
        if self.routed_experts_router:
            self.routed_experts_router.post_batch(dry_run=dry_run)


MoEFusedV2TransformerBlockConfig = OLMoDDPTransformerBlockConfig
MoEFusedV2TransformerBlock = OLMoDDPTransformerBlock

__all__ = [
    "OLMoDDPTransformerBlock",
    "OLMoDDPTransformerBlockConfig",
    "MoEFusedV2TransformerBlock",
    "MoEFusedV2TransformerBlockConfig",
]
