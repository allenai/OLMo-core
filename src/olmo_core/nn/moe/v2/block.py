import threading
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union, cast

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
from olmo_core.ops import attach_auxiliary_loss
from olmo_core.kernels import (
    ScaledGroupedMMPrequantizedRHS,
)
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import get_or_init_stream

from ...attention import AttentionConfig, RingAttentionLoadBalancerType
from ...buffer_cache import BufferCache
from ...functional import l2_normalize
from ...layer_norm import LayerNormConfig
from ...moe import MoERouterGatingFunction
from ...moe import MoERouterConfig as MoERouterConfigV1
from ...moe.loss import MoELoadBalancingLossGranularity
from ...moe.utils import (
    wait_stream_no_compile,
)
from .routed_experts import RoutedExperts, RoutedExpertsConfig, requires_host_side_split_sizes, use_torch_grouped_mm
from .router import MoERouterConfigV2, MoERouterV2
from .shared_experts import SharedExperts
from .shared_experts import SharedExpertsConfig
from .fp8 import (
    MoERowwiseFP8Config,
    invalidate_rowwise_fp8_cache as _invalidate_rowwise_fp8_cache,
    maybe_refresh_shared_rowwise_fp8_cache as _maybe_refresh_shared_rowwise_fp8_cache,
    normalize_rowwise_fp8_config,
    refresh_rowwise_fp8_cache as _refresh_rowwise_fp8_cache,
    shared_experts_forward1_rowwise_fp8 as _shared_experts_forward1_rowwise_fp8,
    shared_experts_forward2_rowwise_fp8 as _shared_experts_forward2_rowwise_fp8,
)
from .ep_no_sync_state import (
    _NoSyncStageAState,
    _NoSyncStageDState,
    _NoSyncSymmBuffers,
    _NoSyncSymmSharedPool,
    _NoSyncTboPendingContext,
)
from .ep_no_sync_buffers import (
    compute_ep_no_sync_rank_capacity as _compute_ep_no_sync_rank_capacity,
    ep_no_sync_slot_for_lane as _ep_no_sync_slot_for_lane,
    get_ep_no_sync_buffers as _get_ep_no_sync_buffers,
    get_ep_no_sync_group_name as _get_ep_no_sync_group_name,
    get_or_init_ep_no_sync_symm_tensor as _get_or_init_ep_no_sync_symm_tensor,
    iter_ep_no_sync_symm_tensors as _iter_ep_no_sync_symm_tensors,
    resolve_ep_no_sync_chunk_reorder_backend as _resolve_ep_no_sync_chunk_reorder_backend,
)
from .ep_sync import (
    checkpointed_permute_routed_experts_unpermute as _checkpointed_permute_routed_experts_unpermute,
    combined_forward_ep as _combined_forward_ep,
    routed_experts_unpermute as _routed_experts_unpermute,
)
from .ep_sync_tbo import combined_forward_ep_tbo as _combined_forward_ep_tbo
from .no_ep import combined_forward_no_ep as _combined_forward_no_ep
from .ep_no_sync_legacy import combined_forward_ep_no_sync_legacy as _combined_forward_ep_no_sync_legacy
from .ep_no_sync_rowwise import (
    build_rowwise_combine_2d_route_to_packed as _build_rowwise_combine_2d_route_to_packed,
    build_rowwise_route_maps as _build_rowwise_route_maps,
    combined_forward_ep_no_sync_rowwise as _combined_forward_ep_no_sync_rowwise,
)
from .ep_no_sync_routing import (
    build_keep_reorder as _build_keep_reorder,
    build_padded_local_expert_batch_sizes_from_layout as _build_padded_local_expert_batch_sizes_from_layout,
    build_tail_keep_quota as _build_tail_keep_quota,
    restore_drop_unpermute_1d as _restore_drop_unpermute_1d,
    sync_tail_drop_allowed_splits_single_a2a as _sync_tail_drop_allowed_splits_single_a2a,
)
from .ep_no_sync_tbo_legacy import (
    combined_forward_ep_no_sync_tbo as _combined_forward_ep_no_sync_tbo,
    ep_no_sync_stage_a as _ep_no_sync_stage_a,
    ep_no_sync_stage_c_launch as _ep_no_sync_stage_c_launch,
    ep_no_sync_stage_d_launch as _ep_no_sync_stage_d_launch,
    ep_no_sync_stage_e as _ep_no_sync_stage_e,
    ep_no_sync_stage_tail as _ep_no_sync_stage_tail,
)
from .checkpointing import is_checkpoint_recomputing
from .activation_debug import maybe_dump_ep_no_sync_saved_activations
from .metrics import (
    accumulate_ep_no_sync_rowwise_metrics as _accumulate_ep_no_sync_rowwise_metrics,
    add_ep_no_sync_rowwise_metrics as _add_ep_no_sync_rowwise_metrics,
    reset_ep_no_sync_rowwise_metrics as _reset_ep_no_sync_rowwise_metrics,
)
from olmo_core.nn.transformer.config import (
    TransformerBlockConfig,
    TransformerBlockType,
)

@dataclass
class MoEFusedV2TransformerBlockConfig(TransformerBlockConfig):

    shared_experts: Optional[SharedExpertsConfig] = None

    routed_experts: Optional[RoutedExpertsConfig] = None

    shared_experts_router: Optional[MoERouterConfigV2] = None

    routed_experts_router: Optional[MoERouterConfigV2] = None

    use_peri_norm: bool = False
    checkpoint_attn: bool = False
    checkpoint_permute_moe_unpermute: bool = False
    checkpoint_combined_ep_tbo: bool = False
    checkpoint_second_unpermute: bool = False
    ep_no_sync: bool = False
    ep_no_sync_use_2d_all_to_all: bool = False
    ep_no_sync_use_rowwise_all_to_all: bool = False
    ep_no_sync_rowwise_nblocks: int = 256
    ep_no_sync_share_dispatch_out: bool = False
    ep_no_sync_capacity_factor: float = 1.125
    ep_no_sync_shared_slots: int = 1
    ep_no_sync_share_combine_out: bool = False
    ep_no_sync_major_align: int = 1
    ep_no_sync_restore_unpermute_backend: str = "te_fused"
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
        assert self.feed_forward is None and self.feed_forward_moe is None, "MoEFusedV2TransformerBlock does not support `feed_forward` or `feed_forward_moe` (use TransformerBlockConfig instead). Set `shared_experts` and `routed_experts` instead."

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
            return MoEFusedV2TransformerBlock(**kwargs)
        else:
            raise NotImplementedError(self.name)


    def num_params(self, d_model: int) -> int:
        block_params = 0

        block_params += self.attention.num_params(d_model)
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

        block_params += self.attention.num_params(d_model)
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
        flops += self.attention.flops_per_seq(d_model, seqlen)

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

class MoEFusedV2TransformerBlock(olmo_core.nn.transformer.block.TransformerBlockBase):

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        attention: AttentionConfig,
        attention_norm: LayerNormConfig,
        routed_experts_router: Optional[MoERouterConfigV2],
        shared_experts_router: Optional[MoERouterConfigV2],
        shared_experts: Optional[SharedExpertsConfig],
        routed_experts: Optional[RoutedExpertsConfig],
        feed_forward_norm: LayerNormConfig,
        use_peri_norm: bool = False,
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
        ep_no_sync_rowwise_nblocks: int = 0,
        ep_no_sync_share_dispatch_out: bool = True,
        ep_no_sync_capacity_factor: float = 2.0,
        ep_no_sync_shared_slots: int = 2,
        ep_no_sync_share_combine_out: bool = False,
        ep_no_sync_major_align: int = 1,
        ep_no_sync_restore_unpermute_backend: str = "te_fused",
        rowwise_fp8: Optional[MoERowwiseFP8Config] = None,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)

        assert dropout == 0.0 or dropout is None, "MoEFusedV2TransformerBlock does not support dropout"
        self.d_model = d_model
        self.block_idx = block_idx

        if attention_residual_alpha is not None:
            raise OLMoConfigurationError("MoEFusedV2TransformerBlock does not support attention_residual_alpha")
        if feed_forward_residual_alpha is not None:
            raise OLMoConfigurationError("MoEFusedV2TransformerBlock does not support feed_forward_residual_alpha")

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
        self.use_peri_norm = use_peri_norm

        ######## START: Attention ########
        self.attention = attention.build(
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
            # Shared Experts Router
            if shared_experts.num_experts > 1:
                # Need router if more than one experts
                assert shared_experts_router is not None, "Need shared_experts_router when using shared experts with more than one expert"
                self.shared_experts_router = shared_experts_router.build(init_device=init_device)
            else:
                assert shared_experts_router is None, "Should not set shared_experts_router when using only one shared expert"
                # No router if just one
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
        self._ep_symm_group_name: Optional[str] = None
        self._ep_no_sync_symm_cache: Dict[str, torch.Tensor] = {}
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
        if self.ep_no_sync_restore_unpermute_backend not in ("te_fused", "te_legacy", "cuda"):
            raise OLMoConfigurationError(
                "ep_no_sync_restore_unpermute_backend must be one of "
                "'te_fused'|'te_legacy'|'cuda' "
                f"(got {self.ep_no_sync_restore_unpermute_backend!r})"
            )



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

        _add_ep_no_sync_rowwise_metrics(self, out, ReduceType)

        if reset:
            self._reset_ep_no_sync_rowwise_metrics()

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
        self._reset_ep_no_sync_rowwise_metrics()

    def _reset_ep_no_sync_rowwise_metrics(self):
        _reset_ep_no_sync_rowwise_metrics(self)

    def _accumulate_ep_no_sync_rowwise_metrics(
        self,
        *,
        drop_token_cnt: torch.Tensor,
        num_out_tokens: int,
        recv_splits_by_src_local: torch.Tensor,
        rank_capacity: int,
    ) -> None:
        _accumulate_ep_no_sync_rowwise_metrics(
            self,
            drop_token_cnt=drop_token_cnt,
            num_out_tokens=num_out_tokens,
            recv_splits_by_src_local=recv_splits_by_src_local,
            rank_capacity=rank_capacity,
        )


    @property
    def is_moe(self) -> bool:
        return True

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

    def get_ep_no_sync_comm_stream(self) -> torch.cuda.Stream:
        return get_or_init_stream(id=f"ep_no_sync_comm_block_{self.block_idx}", priority=0)

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
                    debug_out = maybe_dump_ep_no_sync_saved_activations(
                        self,
                        x,
                        loss_div_factor=loss_div_factor,
                        forward_kwargs=kwargs,
                    )
                    if debug_out is not None:
                        return debug_out
                    return self.combined_forward_ep_no_sync(
                        x, loss_div_factor=loss_div_factor, **kwargs
                    )
                return self.combined_forward_ep(x, loss_div_factor=loss_div_factor, **kwargs)
            else:
                return self.combined_forward_no_ep(x, loss_div_factor=loss_div_factor, **kwargs)
        else:
            # only shared_experts
            return self.combined_forward_shared_only(x, loss_div_factor=loss_div_factor, **kwargs)

    def apply_pp(self, pp_mesh: DeviceMesh):
        pass # nothing to do

    def _ensure_ep_no_sync_symm_backend(self):
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
        Try to alias EP group metadata as symmetric-memory group "0" so NVSHMEM
        allocator bootstrap follows EP group topology instead of WORLD.

        Returns True when aliasing is active (or unnecessary because EP==WORLD),
        otherwise False so caller can fall back to WORLD bootstrap.
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
            from torch._C._distributed_c10d import _SymmetricMemory
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

                global_ranks_str = "_".join(map(str, alias_ranks))
                store = c10d.PrefixStore(
                    f"symmetric_memory-{global_ranks_str}",
                    c10d._get_process_group_store(self.ep_pg),
                )
                _SymmetricMemory.set_group_info(
                    "0",
                    dist.get_rank(self.ep_pg),
                    dist.get_world_size(self.ep_pg),
                    store,
                )
                # Keep Python bookkeeping in sync to avoid duplicate registration.
                group_to_store = getattr(_symm_mem, "_group_name_to_store", None)
                if isinstance(group_to_store, dict):
                    group_to_store["0"] = store
                _EP_SYMM_GROUP0_ALIAS_RANKS = alias_ranks
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
        self.invalidate_rowwise_fp8_cache()
        self.num_local_routed_experts = self.routed_experts.num_local_experts
        self._ep_enabled = True
        self.ep_pg = ep_pg if ep_pg is not None else ep_mp_mesh.get_group()

        if self.ep_no_sync:
            if _symm_mem is None:
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
                _symm_mem.enable_symm_mem_for_group(group_name)
                # Default path: alias EP group as group "0" so NVSHMEM allocator
                # bootstrap tracks EP topology. This keeps 2x8 intra-node teams
                # from inheriting WORLD inter-node behavior.
                alias_group0_active = self._try_alias_ep_group_as_world_for_symm_mem()
                if not alias_group0_active:
                    # Fallback path for environments where aliasing private APIs
                    # are unavailable or group "0" is already occupied.
                    # _symm_mem.enable_symm_mem_for_group(world_group_name)
                    # Option: hard fail
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
            self._ep_no_sync_symm_cache.clear()
            self._ep_no_sync_shared_pool = None
            self._ep_no_sync_shared_slot = 0
            self._ep_no_sync_te_backend_warned = False

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        raise NotImplementedError("TP is not supported in MoEFusedV1TransformerBlock")

    def apply_cp(self, cp_mesh: DeviceMesh, load_balancer: RingAttentionLoadBalancerType):
        raise NotImplementedError("CP is not supported in MoEFusedV1TransformerBlock")
        self.attention.apply_cp(cp_mesh, load_balancer)
        self.shared_experts.apply_cp(cp_mesh)
        self.routed_experts.apply_cp(cp_mesh)

    def apply_fsdp(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError("FSDP is not supported in MoEFusedV2TransformerBlock")

    def apply_compile(self):
        self.compile(
            fullgraph=False,
            # dynamic=False
        )

        # NOTE: the tbo might be called by the outer model directly (by block.combined_forward_ep_tbo(x, ...) instead of block(x, ...)), so need to compile it here as well
        self.combined_forward_ep_tbo = torch.compile(self.combined_forward_ep_tbo)
        self._res_norm_attn = torch.compile(self._res_norm_attn)
        self._routed_experts_unpermute = torch.compile(self._routed_experts_unpermute)


    @property
    def ep_world_size(self) -> int:
        if self.ep_pg is not None:
            return get_world_size(self.ep_pg)
        else:
            return 1

    def router_forward(
        self,
        router: MoERouterV2,
        local_x: torch.Tensor,
        scores_only: bool,
        loss_div_factor: Optional[Union[torch.Tensor, float]],
    ):
        return router(
            local_x,
            scores_only,
            loss_div_factor=loss_div_factor # scalar
        )

    def _attach_routed_aux_loss(
        self,
        x: torch.Tensor,
        routed_expert_router_aux_loss_info: Optional[Tuple[object, ...]],
    ) -> torch.Tensor:
        if routed_expert_router_aux_loss_info is None:
            return x
        assert self.routed_experts_router is not None
        routed_expert_router_aux_loss = self.routed_experts_router.compute_aux_loss(
            *routed_expert_router_aux_loss_info,
            accumulate_metrics=not is_checkpoint_recomputing(),
        )
        if routed_expert_router_aux_loss is None:
            return x
        return attach_auxiliary_loss(x, routed_expert_router_aux_loss)

    def invalidate_rowwise_fp8_cache(self) -> None:
        _invalidate_rowwise_fp8_cache(self)

    @torch.no_grad()
    def refresh_rowwise_fp8_cache(self) -> None:
        _refresh_rowwise_fp8_cache(self)

    def _maybe_refresh_shared_rowwise_fp8_cache(self) -> None:
        _maybe_refresh_shared_rowwise_fp8_cache(self)

    def _shared_experts_forward1_rowwise_fp8(
        self,
        x: torch.Tensor,
        *,
        use_fast_accum: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _shared_experts_forward1_rowwise_fp8(
            self,
            x,
            use_fast_accum=use_fast_accum,
        )

    def _shared_experts_forward2_rowwise_fp8(
        self,
        up: torch.Tensor,
        gate: torch.Tensor,
        xshape: torch.Size,
        *,
        use_fast_accum: bool,
    ) -> torch.Tensor:
        return _shared_experts_forward2_rowwise_fp8(
            self,
            up,
            gate,
            xshape,
            use_fast_accum=use_fast_accum,
        )

    def _get_ep_no_sync_group_name(self) -> str:
        return _get_ep_no_sync_group_name(self)

    def _ep_no_sync_slot_for_lane(self, lane_id: int) -> int:
        return _ep_no_sync_slot_for_lane(self, lane_id)

    def _resolve_ep_no_sync_chunk_reorder_backend(self) -> str:
        return _resolve_ep_no_sync_chunk_reorder_backend()

    def _get_or_init_ep_no_sync_symm_tensor(
        self,
        *,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return _get_or_init_ep_no_sync_symm_tensor(
            self,
            name=name,
            shape=shape,
            dtype=dtype,
            device=device,
        )

    @torch.compiler.disable # to reduce Dynamo/AOT overhead
    def _get_ep_no_sync_buffers(
        self,
        *,
        dispatch_in_cap: int,
        dispatch_out_cap: int,
        combine_in_cap: int,
        combine_out_cap: int,
        d_model: int,
        dtype: torch.dtype,
        device: torch.device,
        slot_idx: Optional[int] = None,
        need_dispatch_in: bool = True,
        need_dispatch_meta: bool = True,
        need_dispatch_out: bool = True,
        need_combine_in: bool = True,
        need_combine_meta: bool = True,
        need_combine_out: bool = True,
    ) -> _NoSyncSymmBuffers:
        return _get_ep_no_sync_buffers(
            self,
            dispatch_in_cap=dispatch_in_cap,
            dispatch_out_cap=dispatch_out_cap,
            combine_in_cap=combine_in_cap,
            combine_out_cap=combine_out_cap,
            d_model=d_model,
            dtype=dtype,
            device=device,
            slot_idx=slot_idx,
            need_dispatch_in=need_dispatch_in,
            need_dispatch_meta=need_dispatch_meta,
            need_dispatch_out=need_dispatch_out,
            need_combine_in=need_combine_in,
            need_combine_meta=need_combine_meta,
            need_combine_out=need_combine_out,
        )

    def iter_ep_no_sync_symm_tensors(self):
        yield from _iter_ep_no_sync_symm_tensors(self)

    def _compute_ep_no_sync_rank_capacity(self, num_out_tokens: int) -> int:
        return _compute_ep_no_sync_rank_capacity(self, num_out_tokens)

    def _build_padded_local_expert_batch_sizes_from_layout(
        self,
        *,
        splits: torch.Tensor,
        offsets: torch.Tensor,
        total_rows: int,
    ) -> torch.Tensor:
        return _build_padded_local_expert_batch_sizes_from_layout(
            self,
            splits=splits,
            offsets=offsets,
            total_rows=total_rows,
        )

    def _build_tail_keep_quota(
        self,
        recv_counts_per_src_local_expert: torch.Tensor,
        rank_capacity: int,
    ) -> torch.Tensor:
        return _build_tail_keep_quota(
            self,
            recv_counts_per_src_local_expert,
            rank_capacity,
        )

    @nvtx.annotate("SyncTokenCount", color="green")
    def _sync_tail_drop_allowed_splits_single_a2a(
        self,
        requested_splits: torch.Tensor,
        *,
        rank_capacity: int,
        return_keep_matrix: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        return _sync_tail_drop_allowed_splits_single_a2a(
            self,
            requested_splits,
            rank_capacity=rank_capacity,
            return_keep_matrix=return_keep_matrix,
        )


    @nvtx.annotate('_build_keep_reorder')
    def _build_keep_reorder(
        self,
        requested_splits: torch.Tensor,
        keep_splits: torch.Tensor,
        num_out_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del self
        return _build_keep_reorder(
            requested_splits,
            keep_splits,
            num_out_tokens,
        )

    @nvtx.annotate("_build_rowwise_route_maps")
    def _build_rowwise_route_maps(
        self,
        *,
        routing_map: torch.Tensor,
        allowed_splits: torch.Tensor,
        keep_from_src_dest_local: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _build_rowwise_route_maps(
            self,
            routing_map=routing_map,
            allowed_splits=allowed_splits,
            keep_from_src_dest_local=keep_from_src_dest_local,
        )

    @nvtx.annotate("_build_rowwise_combine_2d_route_to_packed")
    def _build_rowwise_combine_2d_route_to_packed(
        self,
        *,
        route_to_packed: torch.Tensor,
        dst_ranks: torch.Tensor,
        dst_rows: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _build_rowwise_combine_2d_route_to_packed(
            self,
            route_to_packed=route_to_packed,
            dst_ranks=dst_ranks,
            dst_rows=dst_rows,
        )

    @nvtx.annotate("_restore_drop_unpermute_1d")
    def _restore_drop_unpermute_1d(
        self,
        *,
        combine_out: torch.Tensor,
        local_inverse_reorder_indices: torch.Tensor,
        packed_keep_mask: torch.Tensor,
        num_kept: torch.Tensor,
        reversed_local_x_permutation_mapping: torch.Tensor,
        local_x_global_routed_expert_weights: torch.Tensor,
        hidden_shape_before_permute: torch.Size,
        row_id_map_is_packed: bool = False,
        backward_grad_input_buffer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _restore_drop_unpermute_1d(
            self,
            combine_out=combine_out,
            local_inverse_reorder_indices=local_inverse_reorder_indices,
            packed_keep_mask=packed_keep_mask,
            num_kept=num_kept,
            reversed_local_x_permutation_mapping=reversed_local_x_permutation_mapping,
            local_x_global_routed_expert_weights=local_x_global_routed_expert_weights,
            hidden_shape_before_permute=hidden_shape_before_permute,
            row_id_map_is_packed=row_id_map_is_packed,
            backward_grad_input_buffer=backward_grad_input_buffer,
        )



    def combined_forward_shared_only(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Reserved for no-routed-experts case (only shared experts), equivalent to a dense model"""
        assert self.routed_experts is None
        assert self.routed_experts_router is None
        assert self.shared_experts is not None
        raise NotImplementedError("combined_forward_shared_only is not implemented")


    @torch.compiler.disable
    def sync_dtoh_event(self):
        assert self._dtoh_event is not None
        dtoh_event = cast(torch.cuda.Event, self._dtoh_event)
        dtoh_event.synchronize()

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

    def merge_shared_and_routed_out(
        self,
        shared_out: torch.Tensor,
        shared_factor: float,
        routed_out: torch.Tensor,
        routed_factor: float,
    ) -> torch.Tensor:
        # Combine shared and routed outputs
        return shared_out * shared_factor + routed_out * routed_factor

    def combined_forward_ep_no_sync(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return _combined_forward_ep_no_sync_legacy(
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
        return _combined_forward_ep_no_sync_rowwise(
            self,
            x,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    def _ep_no_sync_stage_a(
        self,
        x: torch.Tensor,
        *,
        lane_id: int,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> _NoSyncStageAState:
        return _ep_no_sync_stage_a(
            self,
            x,
            lane_id=lane_id,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    def _ep_no_sync_stage_d_launch(self, a_state: _NoSyncStageAState) -> _NoSyncStageDState:
        return _ep_no_sync_stage_d_launch(self, a_state)

    def _ep_no_sync_stage_e(self, d_state: _NoSyncStageDState) -> _NoSyncTboPendingContext:
        return _ep_no_sync_stage_e(self, d_state)

    def _ep_no_sync_stage_c_launch(
        self, pending_ctx: _NoSyncTboPendingContext
    ) -> _NoSyncTboPendingContext:
        return _ep_no_sync_stage_c_launch(self, pending_ctx)

    def _ep_no_sync_stage_tail(self, pending_ctx: _NoSyncTboPendingContext) -> torch.Tensor:
        return _ep_no_sync_stage_tail(self, pending_ctx)

    def combined_forward_ep_no_sync_tbo(
        self,
        x0: torch.Tensor,
        x1_ctx: object,
        x1_is_fresh: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, _NoSyncTboPendingContext]:
        return _combined_forward_ep_no_sync_tbo(
            self,
            x0,
            x1_ctx,
            x1_is_fresh,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    def combined_forward_ep(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return _combined_forward_ep(
            self,
            x,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )



    def _routed_experts_unpermute(
        self,
        global_x,
        global_x_local_expert_indices,
        parallel_batch_size_per_local_expert_cpu,
        hidden_shape_before_permute2,
        reversed_global_x_permutation_mapping,
    ):
        return _routed_experts_unpermute(
            self,
            global_x,
            global_x_local_expert_indices,
            parallel_batch_size_per_local_expert_cpu,
            hidden_shape_before_permute2,
            reversed_global_x_permutation_mapping,
        )

    def _attention_norm_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention_norm(x)

    def _feed_forward_norm_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward_norm(x)

    def _prepare_moe_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_peri_norm:
            assert self.feed_forward_input_norm is not None
            return self.feed_forward_input_norm(x)
        return x

    def _res_norm_mlp(self, residual: torch.Tensor, mlp_out: torch.Tensor) -> torch.Tensor:
        return residual + self.feed_forward_norm(mlp_out)


    def _checkpointed_permute_routed_experts_unpermute(
        self,
        global_x,
        global_x_local_expert_indices,
        parallel_batch_size_per_local_expert_cpu
    ) -> torch.Tensor:
        return _checkpointed_permute_routed_experts_unpermute(
            self,
            global_x,
            global_x_local_expert_indices,
            parallel_batch_size_per_local_expert_cpu,
        )


    def _res_norm_attn(
        self,
        block_inp,
        **kwargs
    ) -> torch.Tensor:
        attn_in = block_inp
        if self.use_peri_norm:
            assert self.attention_input_norm is not None
            attn_in = self.attention_input_norm(block_inp)
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
