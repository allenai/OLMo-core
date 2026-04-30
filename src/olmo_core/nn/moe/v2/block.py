import threading
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement
from torch.utils.checkpoint import checkpoint

import olmo_core.nn.transformer.block

try:
    import torch.distributed._symmetric_memory as _symm_mem
except ImportError:
    _symm_mem = None  # type: ignore[assignment]

from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.kernels import ScaledGroupedMMPrequantizedRHS
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerBlockType
from olmo_core.ops import attach_auxiliary_loss
from olmo_core.utils import get_or_init_stream

from ...attention import AttentionConfig, RingAttentionLoadBalancerType
from ...buffer_cache import BufferCache
from ...layer_norm import LayerNormConfig
from .activation_debug import maybe_dump_ep_no_sync_saved_activations
from .checkpointing import is_checkpoint_recomputing
from .ep_no_sync_1d import (
    combined_forward_ep_no_sync_1d as _combined_forward_ep_no_sync_1d,
)
from .ep_no_sync_buffers import _NoSyncSymmSharedPool, _NoSyncTboPendingContext
from .ep_no_sync_rowwise import (
    combined_forward_ep_no_sync_rowwise as _combined_forward_ep_no_sync_rowwise,
)
from .ep_no_sync_rowwise_helpers import (
    add_ep_no_sync_rowwise_metrics,
    reset_ep_no_sync_rowwise_metrics,
)
from .ep_no_sync_tbo_1d import (
    combined_forward_ep_no_sync_tbo as _combined_forward_ep_no_sync_tbo,
)
from .ep_sync_1d import combined_forward_ep_1d as _combined_forward_ep_1d
from .ep_sync_tbo import combined_forward_ep_tbo as _combined_forward_ep_tbo
from .fp8 import MoERowwiseFP8Config
from .fp8 import invalidate_rowwise_fp8_cache as _invalidate_rowwise_fp8_cache
from .fp8 import normalize_rowwise_fp8_config
from .no_ep import combined_forward_no_ep as _combined_forward_no_ep
from .routed_experts import RoutedExperts, RoutedExpertsConfig
from .router import MoERouterConfigV2, MoERouterV2
from .shared_experts import SharedExperts, SharedExpertsConfig

# Process-local cache for the EP->group "0" symmetric-memory alias.
# This makes repeated calls from multiple MoE blocks idempotent when they use
# the same EP group.
_EP_SYMM_GROUP0_ALIAS_LOCK = threading.Lock()
_EP_SYMM_GROUP0_ALIAS_RANKS: Optional[Tuple[int, ...]] = None


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
    ep_no_sync_capacity_factor: float = 1.25
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
        assert (
            self.feed_forward is None and self.feed_forward_moe is None
        ), "MoEFusedV2TransformerBlock does not support `feed_forward` or `feed_forward_moe` (use TransformerBlockConfig instead). Set `shared_experts` and `routed_experts` instead."

        kwargs = self.as_dict(exclude_none=False, recurse=False)
        kwargs.pop("name")
        kwargs.pop("feed_forward")  # from parent config
        kwargs.pop("feed_forward_moe")  # from parent config
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
        flops += (
            6
            * seqlen
            * d_model
            * (
                (
                    self.routed_experts_router.num_experts
                    if self.routed_experts_router is not None
                    else 0
                )
                + (
                    self.shared_experts_router.num_experts
                    if self.shared_experts_router is not None
                    else 0
                )
            )
        )

        # routed experts
        # (seq_len, d_model) * (d_model, expert_hidden_size) * top_k
        # x3 for swiglu (up, down, gate)
        # x3 for fwd+bwd
        # x2 for GEMM
        if self.routed_experts is not None:
            assert self.routed_experts_router is not None
            flops += (
                (3 * 3 * 2)
                * seqlen
                * d_model
                * self.routed_experts.hidden_size
                * self.routed_experts_router.top_k
            )

        # shared experts
        # (seq_len, d_model) * (d_model, expert_hidden_size) * num_experts
        # x3 for swiglu (up, down, gate)
        # x3 for fwd+bwd
        # x2 for GEMM
        if self.shared_experts is not None:
            flops += (
                (3 * 3 * 2)
                * seqlen
                * d_model
                * self.shared_experts.hidden_size
                * self.shared_experts.num_experts
            )

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
        checkpoint_attn=False,
        checkpoint_permute_moe_unpermute=False,
        checkpoint_combined_ep_tbo=False,
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
        rowwise_fp8: Optional[MoERowwiseFP8Config] = None,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)

        assert (
            dropout == 0.0 or dropout is None
        ), "MoEFusedV2TransformerBlock does not support dropout"
        self.d_model = d_model
        self.block_idx = block_idx

        if attention_residual_alpha is not None:
            raise OLMoConfigurationError(
                "MoEFusedV2TransformerBlock does not support attention_residual_alpha"
            )
        if feed_forward_residual_alpha is not None:
            raise OLMoConfigurationError(
                "MoEFusedV2TransformerBlock does not support feed_forward_residual_alpha"
            )

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
        assert (routed_experts is not None) or (
            shared_experts is not None
        ), "At least one of routed_experts or shared_experts must be provided"

        #### Optional: routed experts ####
        if routed_experts:
            # Routed Experts enabled
            assert (
                routed_experts_router is not None
            ), "Need routed_experts_router when using routed experts"
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
            assert (
                routed_experts_router is None
            ), "Should not set routed_experts_router when not using routed experts"
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
                assert (
                    shared_experts_router is not None
                ), "Need shared_experts_router when using shared experts with more than one expert"
                self.shared_experts_router = shared_experts_router.build(init_device=init_device)
            else:
                assert (
                    shared_experts_router is None
                ), "Should not set shared_experts_router when using only one shared expert"
                # No router if just one
                self.shared_experts_router = None
        else:
            # Shared Experts not enabled
            assert (
                shared_experts_router is None
            ), "Should not set shared_experts_router when not using shared experts"
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
        self._dtoh_event: torch.cuda.Event = cast(
            torch.cuda.Event, torch.cuda.Event()
        )  # cast to make pylance happy
        self._dtoh_event_send: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())
        self._dtoh_event_recv: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())
        self._before_rev_all2all_event: torch.cuda.Event = cast(
            torch.cuda.Event, torch.cuda.Event()
        )

        # same for tbo1
        self._dtoh_event1: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())
        self._dtoh_event_send1: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())
        self._dtoh_event_recv1: torch.cuda.Event = cast(torch.cuda.Event, torch.cuda.Event())
        self._before_rev_all2all_event1: torch.cuda.Event = cast(
            torch.cuda.Event, torch.cuda.Event()
        )

        self.num_local_routed_experts: Optional[int] = (
            self.routed_experts.num_experts if self.routed_experts else None
        )

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
        if self.ep_no_sync_restore_unpermute_backend not in ("te_fused", "te_unfused", "cuda"):
            raise OLMoConfigurationError(
                "ep_no_sync_restore_unpermute_backend must be one of "
                "'te_fused'|'te_unfused'|'cuda' "
                f"(got {self.ep_no_sync_restore_unpermute_backend!r})"
            )

    def purge_cuda_events(self):
        # set all events to None (so that the model can be deepcopied)
        self._dtoh_event = None  # type: ignore[assignment]
        self._dtoh_event_send = None  # type: ignore[assignment]
        self._dtoh_event_recv = None  # type: ignore[assignment]
        self._before_rev_all2all_event = None  # type: ignore[assignment]

        self._dtoh_event1 = None  # type: ignore[assignment]
        self._dtoh_event_send1 = None  # type: ignore[assignment]
        self._dtoh_event_recv1 = None  # type: ignore[assignment]
        self._before_rev_all2all_event1 = None  # type: ignore[assignment]

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
    def is_moe(self) -> bool:
        return True

    @property
    def ep_enabled(self) -> bool:
        return self._ep_enabled

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    def get_dense_stream(self, for_x1=False) -> torch.cuda.Stream:
        if for_x1:  # not used for now
            return get_or_init_stream(id="dense_x1", priority=20)
        else:
            return get_or_init_stream(id="dense", priority=20)

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.routed_experts:
            if self.ep_enabled:
                if (
                    self.ep_no_sync and self.training
                ):  # in eval mode, different ranks might get different input token counts, and no-sync can freeze
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
                    return no_sync_forward(x, loss_div_factor=loss_div_factor, **kwargs)
                return self.combined_forward_ep_1d(x, loss_div_factor=loss_div_factor, **kwargs)
            else:
                return self.combined_forward_no_ep(x, loss_div_factor=loss_div_factor, **kwargs)
        else:
            # only shared_experts
            return self.combined_forward_shared_only(x, loss_div_factor=loss_div_factor, **kwargs)

    def apply_pp(self, pp_mesh: DeviceMesh):
        pass  # nothing to do

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
        assert (
            self.routed_experts is not None
        ), "ep can only be applied when routed_experts is enabled"
        ep_dp_mesh = ep_mesh["ep_dp"]
        ep_mp_mesh = ep_mesh["ep_mp"]
        ep_pg = kwargs.get("ep_pg")
        self.ep_mesh = ep_mesh
        self.routed_experts.apply_ep(ep_mesh)
        owner_ref = weakref.ref(self)
        self.routed_experts.w_up_gate._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        self.routed_experts.w_down._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        _invalidate_rowwise_fp8_cache(self)
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
            assert (
                dist.group.WORLD is not None
            ), "torch.distributed.group.WORLD must be initialized for EP no-sync to work"
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

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        load_balancer: RingAttentionLoadBalancerType,
        head_stride: int = 1,
    ):
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
        self.combined_forward_ep_tbo = torch.compile(self.combined_forward_ep_tbo)  # type: ignore[method-assign]
        self._res_norm_attn = torch.compile(self._res_norm_attn)  # type: ignore[method-assign]

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
        """Reserved for no-routed-experts case (only shared experts), equivalent to a dense model"""
        assert self.routed_experts is None
        assert self.routed_experts_router is None
        assert self.shared_experts is not None
        raise NotImplementedError("combined_forward_shared_only is not implemented")

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
        return _combined_forward_ep_no_sync_rowwise(
            self,
            x,
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
    ) -> Tuple[torch.Tensor, _NoSyncTboPendingContext]:
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

    def _res_norm_mlp(self, residual: torch.Tensor, mlp_out: torch.Tensor) -> torch.Tensor:
        return residual + self.feed_forward_norm(mlp_out)

    def _res_norm_attn(self, block_inp, **kwargs) -> torch.Tensor:
        attn_in = block_inp
        if self.use_peri_norm:
            assert self.attention_input_norm is not None
            attn_in = self.attention_input_norm(block_inp)
        attn_res_out = block_inp + self.attention_norm(self.attention(attn_in, **kwargs))
        return attn_res_out

    def _checkpointed_res_norm_attn(self, block_inp, **kwargs) -> torch.Tensor:
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
