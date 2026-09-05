import logging
from collections import defaultdict
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from olmo_core.data.utils import get_cumulative_document_lengths
from olmo_core.distributed.parallel import get_pp_mesh
from olmo_core.distributed.utils import hide_from_torch, unhide_from_torch
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.utils import get_default_device, mark_dynamic, move_to_device

from ..attention import (
    Attention,
    FusedAttention,
    RingAttentionLoadBalancer,
    SequenceMixer,
)
from ..attention.chunked_mask import (
    build_chunk_ids_from_tokens,
    collapse_roles_to_causal,
    mask_mix_standard_prob,
)
from ..buffer_cache import BufferCache
from ..functional import l2_normalize
from ..layer_norm import LayerNormConfig
from ..lm_head import LMHeadConfig, LMOutputWithLoss
from ..moe import MoEBase
from ..rope import RoPEBuffers, RotaryEmbeddingBase
from ..utils import selective_checkpointing_context_fn
from .block import (
    MoETransformerBlock,
    NormalizedTransformerBlock,
    TransformerBlock,
    TransformerBlockBase,
)
from .config import (
    TransformerActivationCheckpointingMode,
    TransformerBlockConfig,
    TransformerDataParallelWrappingStrategy,
    resolve_block_configs,
)
from .init import InitMethod

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

__all__ = [
    "Transformer",
    "NormalizedTransformer",
    "MoETransformer",
    "TransformerDataParallelWrappingStrategy",
    "TransformerActivationCheckpointingMode",
]


log = logging.getLogger(__name__)


class Transformer(nn.Module):
    """
    A typical "Llama-style" transformer implementation.

    :param d_model: The model dimensionality.
    :param vocab_size: The vocab size.
    :param n_layers: The number of transformer layers/blocks.
    :param block: The block configuration. Can be a single block config or a dict of named blocks.
    :param layer_norm: The layer norm config for the final layer norm.
    :param bias: Whether to use a bias in the final linear layer.
    :param dtype: The datatype to use for the linear output layer.
    :param init_device: The device used when initializing parameters.
    :param init_seed: The seed used when initializing parameters.
    :param init_std: The standard deviation used when initializing parameters.
    :param embedding_init_std: The standard deviation used when initializing the embeddings.
    :param block_overrides: Overrides for specific blocks. Not supported if `block` is a dict of named blocks.
    :param block_pattern: The pattern of blocks to use. Required if `block` is a dict of named blocks.
    :param embed_scale: The scale factor for the embeddings.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        block: TransformerBlockConfig | dict[str, TransformerBlockConfig],
        lm_head: LMHeadConfig,
        embedding_norm: Optional[LayerNormConfig] = None,
        dtype: torch.dtype = torch.float32,
        init_method: InitMethod = InitMethod.normal,
        init_device: str = "cpu",
        init_seed: int = 0,
        init_std: float = 0.02,
        embedding_init_std: Optional[float] = None,
        block_overrides: Optional[Dict[int, TransformerBlockConfig]] = None,
        block_pattern: Optional[List[str]] = None,
        embed_scale: Optional[float] = None,
    ):
        super().__init__()

        cache = BufferCache()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dtype = dtype
        self.embed_scale = embed_scale
        # Optional document-chunked-attention config (see ``enable_document_chunk_attention``). When
        # set, ``_prepare_inputs`` reconstructs per-token ``chunk_ids`` from the boundary tokens and
        # forwards them to every block (for :class:`DocumentLandmarkAttention`).
        self._document_chunk_attention: Optional[Dict[str, Any]] = None
        # Soft-token document pooling ("B1"): set by ``enable_pooled_soft_tokens``.
        self._pooled_soft_tokens: Optional[Dict[str, Any]] = None
        self.pooled_projector: Optional[nn.Module] = None
        self._pooled_keep_holder: Optional[Any] = None
        # Role-gated FFN ("flexible-compute FFN"): set by ``enable_role_gated_ffn``.
        self._role_gated_ffn: Optional[Dict[str, Any]] = None
        # Nested-width FFN mixture (learned router): set by ``enable_nested_ffn_moe``.
        self._nested_ffn_moe: Optional[Dict[str, Any]] = None
        # Per-layer KV-cache allocation router: set by ``enable_kv_route``.
        self._kv_route: Optional[Dict[str, Any]] = None
        # Joint FFN+attention budget over both routers: set by ``joint_budget.install_joint_budget``.
        self._joint_budget: Optional[Dict[str, Any]] = None
        # Per-token block skipping (mixture-of-depths router): set by ``enable_block_skip``.
        self._block_skip: Optional[Dict[str, Any]] = None

        self.embeddings = nn.Embedding(vocab_size, d_model, dtype=dtype, device=init_device)
        self.embedding_norm = (
            None
            if embedding_norm is None
            else embedding_norm.build(
                d_model,
                init_device=init_device,
            )
        )

        block_configs: List[TransformerBlockConfig] = resolve_block_configs(
            n_layers=n_layers,
            block=block,
            block_pattern=block_pattern,
            block_overrides=block_overrides,
        )

        self.blocks = nn.ModuleDict()
        for block_idx in range(n_layers):
            self.blocks[str(block_idx)] = self._validate_block(
                block_configs[block_idx].build(
                    d_model=d_model,
                    block_idx=block_idx,
                    n_layers=n_layers,
                    init_device=init_device,
                    cache=cache,
                )
            )
        self.lm_head = lm_head.build(
            d_model=d_model, vocab_size=vocab_size, init_device=init_device
        )

        self.init_device = init_device
        self.init_method = InitMethod(init_method)
        self.init_seed = init_seed
        self.init_std = init_std
        self.embedding_init_std = embedding_init_std

        self._cache = cache
        self._pp_enabled = False
        self._pp_group_size = 1
        self._fp8_enabled = False
        self._precompute_float8_dynamic_scale_for_fsdp = False
        self._compile_enabled = False
        self._device: Optional[torch.device] = None
        self._cp_load_balancer: Optional[RingAttentionLoadBalancer] = None
        self._tp_enabled = False
        self._tp_mesh: Optional[DeviceMesh] = None
        self._fsdp_enabled = False

        # Cache the value of these properties up-front in case the parameters are removed
        # later, like for pipeline parallelism.
        self.num_params
        self.num_non_embedding_params

    def _validate_block(self, block: TransformerBlockBase) -> TransformerBlockBase:
        return block

    def enable_document_chunk_attention(
        self,
        doc_start_id: int,
        doc_end_id: int,
        eos_id: int,
        mode: str = "chunked",
        pad_id: Optional[int] = None,
        standard_mix_prob: float = 0.0,
        mix_start_p: float = 0.0,
        mix_end_p: float = 0.0,
        mix_total_forwards: int = 0,
        mix_seed: int = 42,
        mix_log_interval: int = 500,
    ) -> None:
        """
        Enable runtime ``chunk_ids`` reconstruction for document-chunked landmark attention.

        When enabled, :meth:`forward` reconstructs a per-token ``chunk_id`` role tensor from the
        ``<|doc_start|>`` / ``<|doc_end|>`` boundary tokens (and the EOS pad terminator) on every
        step and forwards it to each block, so :class:`~olmo_core.nn.attention.DocumentLandmarkAttention`
        can build its chunked mask. All attention layers must accept a ``chunk_ids`` keyword (i.e. the
        model is uniformly ``document_landmark``); mixing with other attention types via this path is
        not supported. Context parallelism with chunked attention is not yet supported.

        **Mask mixing** (optional): during *training*, each forward independently collapses a random
        subset of examples from the chunked mask to plain causal (full) attention, by setting their
        roles to all-FREE (see :func:`~olmo_core.nn.attention.chunked_mask.collapse_roles_to_causal`).
        The per-example probability ``p`` is either static (``standard_mix_prob``) or a linear
        curriculum (``mix_start_p -> mix_end_p`` over ``mix_total_forwards`` forwards). ``p == 0`` (all
        defaults) is a no-op -- pure-chunked training stays bit-identical. Static and curriculum mixing
        are mutually exclusive.

        :param doc_start_id: The ``<|doc_start|>`` token id.
        :param doc_end_id: The ``<|doc_end|>`` token id.
        :param eos_id: The EOS / document terminator (everything after the first one is padding).
        :param mode: ``"chunked"`` (no SINK) or ``"modified_swa"`` (mark the instruction prefix SINK).
        :param pad_id: Optional dedicated interior-padding id (window fill for the landmark variant);
            when set, those positions reconstruct to ``PAD`` (non-attendable) rather than ``FREE``.
        :param standard_mix_prob: Static per-example mask-mix probability (constant every forward).
        :param mix_start_p: Curriculum start probability (at forward 0).
        :param mix_end_p: Curriculum end probability (at ``mix_total_forwards``).
        :param mix_total_forwards: Number of forwards over which the curriculum anneals linearly.
        :param mix_seed: Base seed for the deterministic per-(forward, example) mix coin.
        :param mix_log_interval: Log a ``[curriculum]`` line (current ``p`` + cumulative collapse
            count) every this many training forwards.

        :raises OLMoConfigurationError: If static and curriculum mixing are both requested, or a
            curriculum is requested without ``mix_total_forwards > 0``.
        """
        curriculum = mix_start_p > 0.0 or mix_end_p > 0.0
        if standard_mix_prob > 0.0 and curriculum:
            raise OLMoConfigurationError(
                "Mask mixing: 'standard_mix_prob' (static) is mutually exclusive with the curriculum "
                "('mix_start_p' / 'mix_end_p'); set only one."
            )
        if curriculum and mix_total_forwards <= 0:
            raise OLMoConfigurationError(
                "Mask-mix curriculum ('mix_start_p' / 'mix_end_p') requires 'mix_total_forwards' > 0 "
                "(the number of forwards over which p anneals)."
            )
        mix: Optional[Dict[str, Any]] = None
        if standard_mix_prob > 0.0 or curriculum:
            mix = {
                "standard_mix_prob": float(standard_mix_prob),
                "mix_start_p": float(mix_start_p),
                "mix_end_p": float(mix_end_p),
                "mix_total_forwards": int(mix_total_forwards),
                "mix_seed": int(mix_seed),
                "log_interval": max(1, int(mix_log_interval)),
                "forward_idx": 0,
                "n_collapsed": 0,
            }
        self._document_chunk_attention = {
            "doc_start_id": int(doc_start_id),
            "doc_end_id": int(doc_end_id),
            "eos_id": int(eos_id),
            "mode": mode,
            "pad_id": None if pad_id is None else int(pad_id),
            "mix": mix,
        }

    def enable_pooled_soft_tokens(
        self,
        doc_start_id: int,
        doc_end_id: int,
        eos_id: int,
        *,
        placeholder_id: int,
        keep_prob: float = 0.1,
        keep_seed: int = 42,
        projector_hidden: Optional[int] = None,
        aux_match_weight: float = 0.0,
        aux_queries: int = 16,
        aux_max_shadows: int = 8,
        detach_soft_kv: bool = False,
        distill_prob: float = 0.0,
        distill_weight: float = 1.0,
        distill_layer_stride: int = 4,
        oracle_cache: Optional[Any] = None,
    ) -> None:
        """
        Enable train-time soft-token document pooling ("B1"; see :mod:`olmo_core.nn.pooled_soft_token`).

        During **training** forwards, each context document (identified from the
        ``<|doc_start|>``/``<|doc_end|>`` markers) is either kept (real tokens) or replaced by a
        single soft token -- the :class:`~olmo_core.nn.pooled_soft_token.PooledDocProjector` output
        on the document's mean input embedding -- at the document's center position, with original
        ``position_ids`` threaded to RoPE. The keep set is gold + random negatives via
        :func:`~olmo_core.nn.attention.pooled_doc_kv.install_pooled_doc_keep`, or a seeded random
        ``keep_prob`` fraction as fallback. Eval/generation forwards are untouched (full attention
        over real tokens -- the transfer condition), as are rows without markers.

        Creates the ``pooled_projector`` submodule -- call this BEFORE building the optimizer, and
        call ``pooled_projector.reset_parameters()`` after loading a base checkpoint without
        projector keys. Not supported together with context/pipeline parallelism or
        ``enable_document_chunk_attention``.

        :param placeholder_id: Token id emitted at soft slots (embedding is overwritten; use a
            repaired reserved id, e.g. the landmark id).
        """
        if self._document_chunk_attention is not None:
            raise OLMoConfigurationError(
                "enable_pooled_soft_tokens is mutually exclusive with "
                "enable_document_chunk_attention (the compacted sequence is plain causal)."
            )
        from ..pooled_soft_token import PooledDocProjector

        emb_weight = self.embeddings.weight  # type: ignore[union-attr]
        self.pooled_projector = PooledDocProjector(
            self.d_model,
            hidden=projector_hidden,
            dtype=emb_weight.dtype,
            init_device="meta" if emb_weight.device.type == "meta" else str(emb_weight.device),
        )
        self._pooled_soft_tokens = {
            "doc_start_id": int(doc_start_id),
            "doc_end_id": int(doc_end_id),
            "eos_id": int(eos_id),
            "placeholder_id": int(placeholder_id),
            "keep_prob": float(keep_prob),
            "keep_seed": int(keep_seed),
            # Aux attention-contribution matching (see pooled_soft_token.aux_matching_loss): train
            # the projector ONLINE so a soft token's per-layer KV reproduces its doc's real
            # attention behavior, using the keep set as the supervision source. > 0 enables
            # shadows + the position-causal masked attention path.
            "aux_match_weight": float(aux_match_weight),
            "aux_queries": int(aux_queries),
            "aux_max_shadows": int(aux_max_shadows),
            # Treat pooled slots as STATIC KV: sever the LM loss's backward through the slot
            # columns' K/V at EVERY layer (and their input injection), so the main model receives
            # task gradient only through real tokens and cannot co-invent a private "summary
            # language" (the co-drift channel). The projector then trains ONLY via the aux
            # shadow objective (frozen if aux_match_weight == 0).
            "detach_soft_kv": bool(detach_soft_kv),
            # Paired consistency distillation: with probability distill_prob a training forward
            # runs BOTH the full pass (LM gradient -> protects the full-attention pathway from
            # co-drift) and the compressed pass, matching the student's hidden states at the
            # divergence layers (probe: L16+) to the DETACHED teacher at answer positions.
            # The coin is seeded on a shared forward counter so every FSDP rank takes the same
            # branch (asymmetric double-forwards would desynchronize the collectives).
            "distill_prob": float(distill_prob),
            "distill_weight": float(distill_weight),
            "distill_layers": sorted(
                set(range(int(self.n_layers * 0.4), self.n_layers, max(1, distill_layer_stride)))
                | {self.n_layers - 1}
            ),
            "_distill_counter": 0,
            # Oracle slot cache (olmo_core.nn.oracle_slot.OracleSlotCache): when set, pooled
            # slots' per-layer K/V are OVERRIDDEN with precomputed oracle log-mass slots instead
            # of whatever the network computes from the projected soft token -- the
            # maximal-fidelity static slot. Docs missing from the cache fall back to the
            # projector path.
            "oracle_cache": oracle_cache,
            "_oracle_stats": {"hits": 0, "misses": 0, "calls": 0},
        }

    def enable_role_gated_ffn(
        self,
        doc_start_id: int,
        doc_end_id: int,
        eos_id: int,
        *,
        start_layer: int = 4,
        pad_id: Optional[int] = None,
    ) -> None:
        """
        Enable role-gated FFN compute (see :mod:`olmo_core.nn.role_gated_ffn`): context-document
        tokens (between ``<|doc_start|>``/``<|doc_end|>`` markers) skip the full FFN from
        ``start_layer`` on -- identity residual -- while free/query/answer tokens (and generated
        tokens at decode time) keep it. The gate applies to EVERY forward (train, eval, prefill),
        so training and inference see identical routing; no new parameters, so base checkpoints
        load untouched.

        Call BEFORE building the optimizer / applying data parallelism.

        :param start_layer: First gated layer (earlier layers keep the full FFN everywhere).
        """
        from ..role_gated_ffn import RoleGateHolder, install_role_gated_ffn

        holder = RoleGateHolder()
        gated = install_role_gated_ffn(self.blocks, holder, start_layer=start_layer)
        if not gated:
            raise OLMoConfigurationError("enable_role_gated_ffn gated no blocks")
        self._role_gated_ffn = {
            "doc_start_id": int(doc_start_id),
            "doc_end_id": int(doc_end_id),
            "eos_id": int(eos_id),
            "pad_id": pad_id,
            "start_layer": int(start_layer),
            "holder": holder,
        }
        log.info("Role-gated FFN enabled on %d blocks (start_layer=%d)", len(gated), start_layer)

    def enable_nested_ffn_moe(
        self,
        *,
        start_layer: int = 4,
        divisors: Sequence[float] = (1, 4, 16, 64),
        include_null: bool = True,
        target_cost: float = 0.05,
        budget_weight: float = 1.0,
        hinge_power: int = 1,
        two_sided: bool = False,
        target_anneal_calls: int = 0,
        explore_prob: float = 0.0,
        explore_anneal_calls: int = 0,
        recon_frac: float = 0.0,
        recon_weight: float = 0.0,
        entropy_weight: float = 0.0,
        seed: int = 0,
        layer_curriculum_calls: int = 0,
        width_multiple: int = 8,
        trainable_width: int = 0,
    ) -> None:
        """
        Enable the nested-width FFN mixture (see :mod:`olmo_core.nn.nested_ffn_moe`): from
        ``start_layer`` on, a learned per-token router picks one of several nested FFN widths
        (down to a zero-cost null rung), and a budget hinge loss pushes the mean per-token FFN
        cost under ``target_cost``.

        Adds a small router and per-rung gains per gated block -- NEW state-dict keys, so the base
        checkpoint must be re-saved with them (``bake_ffn_moe_into_base.py``). The router is
        initialized to select the full rung with probability ~1, so an enabled but untrained model
        reproduces its base exactly.

        Call BEFORE building the optimizer / applying data parallelism.

        :param start_layer: First routed layer.
        :param divisors: Cost divisors for the rungs, e.g. ``(1, 4, 16, 64)``.
        :param include_null: Append a zero-compute null rung.
        :param target_cost: Mean per-token FFN cost the budget hinge allows for free.
        :param budget_weight: Weight of the budget hinge.
        :param recon_frac: Fraction of tokens carrying a local full-FFN reconstruction target.
        :param seed: Base seed for the exploration draws (kept deterministic per forward so
            activation-checkpoint recompute reproduces the routing).
        :param layer_curriculum_calls: If > 0, routing opens from the last layer downward to
            ``start_layer`` linearly over this many forwards (see
            :meth:`NestedFFNHolder.current_min_layer`).
        :param width_multiple: Rung widths are floored to a multiple of this (minimum = this).
            ``1`` allows a single-hidden-unit rung, e.g. divisor 9728 on Qwen3-4B.

        :raises OLMoConfigurationError: If no blocks were routed.
        """
        from ..nested_ffn_moe import (
            NestedFFNHolder,
            install_nested_ffn_moe,
            resolve_rung_widths,
        )

        first_block = next(iter(self.blocks.values()))
        hidden_size = first_block.feed_forward.w1.out_features  # type: ignore[union-attr]
        widths, costs = resolve_rung_widths(
            hidden_size, divisors, include_null=include_null, multiple_of=width_multiple
        )
        holder = NestedFFNHolder(
            costs,
            target_cost=target_cost,
            budget_weight=budget_weight,
            hinge_power=hinge_power,
            two_sided=two_sided,
            target_anneal_calls=target_anneal_calls,
            explore_prob=explore_prob,
            explore_anneal_calls=explore_anneal_calls,
            recon_frac=recon_frac,
            recon_weight=recon_weight,
            entropy_weight=entropy_weight,
            seed=seed,
            start_layer=start_layer,
            n_layers=len(self.blocks),
            layer_curriculum_calls=layer_curriculum_calls,
        )
        routed = install_nested_ffn_moe(
            self.blocks, holder, start_layer=start_layer, widths=widths, costs=costs,
            trainable_width=trainable_width,
        )
        if not routed:
            raise OLMoConfigurationError("enable_nested_ffn_moe routed no blocks")
        self._nested_ffn_moe = {
            "start_layer": int(start_layer),
            "widths": widths,
            "costs": costs,
            "holder": holder,
        }
        log.info(
            "Nested-FFN MoE enabled on %d blocks (start_layer=%d) rungs=%s costs=%s target=%.4f",
            len(routed),
            start_layer,
            widths,
            [round(c, 5) for c in costs],
            target_cost,
        )

    def enable_kv_route(
        self,
        *,
        start_layer: int = 0,
        target: float = 0.5,
        budget_weight: float = 1.0,
        two_sided: bool = True,
        target_anneal_calls: int = 0,
        explore_prob: float = 0.0,
        explore_anneal_calls: int = 0,
        seed: int = 0,
    ) -> None:
        """
        Enable learned per-layer KV-cache allocation (see :mod:`olmo_core.nn.attention.kv_route`):
        every plain full-attention layer at or after ``start_layer`` gets a per-token keep/drop
        router; dropped keys leave that layer's cache, and a budget term pulls the mean keep
        fraction (over tokens and routed layers) to ``target``.

        Adds NEW state-dict keys (``blocks.<i>.attention._kvr_router.w.*``), initialised to keep
        everything so an enabled-but-untrained model reproduces its base exactly. Call BEFORE
        building the optimizer / applying data parallelism.

        :raises OLMoConfigurationError: If no attention layer was routed.
        """
        from ..attention.kv_route import KVRouteHolder, install_kv_route

        holder = KVRouteHolder(
            target=target,
            budget_weight=budget_weight,
            two_sided=two_sided,
            target_anneal_calls=target_anneal_calls,
            explore_prob=explore_prob,
            explore_anneal_calls=explore_anneal_calls,
            seed=seed,
            start_layer=start_layer,
            n_layers=len(self.blocks),
        )
        routed = install_kv_route(self.blocks, holder, start_layer=start_layer)
        if not routed:
            raise OLMoConfigurationError("enable_kv_route routed no attention layers")
        self._kv_route = {"start_layer": int(start_layer), "routed": routed, "holder": holder}
        log.info(
            "KV routing enabled on %d attention layers %s (start_layer=%d) target=%.3f",
            len(routed), routed, start_layer, target,
        )

    def enable_block_skip(
        self,
        *,
        start_layer: int = 0,
        target: float = 0.5,
        budget_weight: float = 1.0,
        two_sided: bool = True,
        target_anneal_calls: int = 0,
        seed: int = 0,
    ) -> None:
        """
        Enable learned per-token block skipping (see :mod:`olmo_core.nn.block_skip`): every block at
        or after ``start_layer`` gets a router deciding per token whether the block runs; skipped
        tokens pass the residual stream unchanged and are not keys in that block's attention. Adds
        NEW state-dict keys ``blocks.<i>._bskip_router.w.*`` initialised to run everything.

        :raises OLMoConfigurationError: If no block was routed.
        """
        from ..block_skip import BlockSkipHolder, install_block_skip

        holder = BlockSkipHolder(
            target=target,
            budget_weight=budget_weight,
            two_sided=two_sided,
            target_anneal_calls=target_anneal_calls,
            seed=seed,
            start_layer=start_layer,
            n_layers=len(self.blocks),
        )
        routed = install_block_skip(self.blocks, holder, start_layer=start_layer)
        if not routed:
            raise OLMoConfigurationError("enable_block_skip routed no blocks")
        self._block_skip = {"start_layer": int(start_layer), "routed": routed, "holder": holder}
        log.info(
            "Block skipping enabled on %d blocks (start_layer=%d) target=%.3f",
            len(routed), start_layer, target,
        )

    def _set_role_gate_mask(self, input_ids: torch.Tensor) -> None:
        """Recompute the FFN gate mask from the (possibly compacted) token stream."""
        cfg = self._role_gated_ffn
        assert cfg is not None
        holder = cfg["holder"]
        if input_ids.is_floating_point():
            holder.clear()  # PP hidden states: cannot derive roles here
            return
        chunk_ids = build_chunk_ids_from_tokens(
            input_ids,
            doc_start_id=cfg["doc_start_id"],
            doc_end_id=cfg["doc_end_id"],
            eos_id=cfg["eos_id"],
            mode="chunked",
            pad_id=cfg["pad_id"],
        )
        holder.set_from_chunk_ids(move_to_device(chunk_ids, self.device))

    def _compact_pooled_soft_tokens(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        ignore_index: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple]]:
        """
        Apply soft-token compaction for this training forward. Returns
        ``(input_ids, labels, position_ids, soft_inject)`` -- unchanged inputs and ``None``s when
        the batch has no context documents. ``soft_inject = (rows, cols, mean_embeds)`` is consumed
        after the embedding lookup.
        """
        from ..attention.pooled_doc_kv import resolve_keep_docs
        from ..pooled_soft_token import compact_pooled_rows

        cfg = self._pooled_soft_tokens
        assert cfg is not None
        if self._cp_load_balancer is not None or self._pp_enabled:
            raise OLMoConfigurationError(
                "pooled soft tokens are not supported with context/pipeline parallelism"
            )
        chunk_ids = build_chunk_ids_from_tokens(
            input_ids,
            doc_start_id=cfg["doc_start_id"],
            doc_end_id=cfg["doc_end_id"],
            eos_id=cfg["eos_id"],
            mode="chunked",
        )
        n_docs = int(chunk_ids.max().item()) + 1
        if n_docs <= 0:
            return None
        keep = resolve_keep_docs(
            chunk_ids,
            n_docs,
            holder=self._pooled_keep_holder,
            keep_prob=cfg["keep_prob"],
            keep_seed=cfg["keep_seed"],
        )
        # Mean input embedding per pooled doc (from the ORIGINAL row), the projector's feature.
        emb = self.embeddings(input_ids)  # type: ignore[misc]
        B, T, D = emb.shape
        cid = chunk_ids.to(torch.long)
        is_ctx = cid >= 0
        flat = (torch.arange(B, device=emb.device)[:, None] * n_docs + cid.clamp(min=0)).reshape(
            -1
        )[is_ctx.reshape(-1)]
        sums = torch.zeros(B * n_docs, D, dtype=emb.dtype, device=emb.device).index_add(
            0, flat, emb.reshape(B * T, D)[is_ctx.reshape(-1)]
        )
        counts = (
            torch.zeros(B * n_docs, dtype=torch.float32, device=emb.device)
            .index_add(0, flat, torch.ones_like(flat, dtype=torch.float32))
            .clamp(min=1.0)
        )
        doc_means = (sums / counts.unsqueeze(-1).to(emb.dtype)).reshape(B, n_docs, D)

        cb = compact_pooled_rows(
            input_ids,
            labels,
            chunk_ids,
            keep,
            placeholder_id=cfg["placeholder_id"],
            pad_token_id=cfg["eos_id"],
            ignore_index=ignore_index,
            add_shadows=cfg["aux_match_weight"] > 0.0,
            max_shadows_per_row=cfg["aux_max_shadows"],
        )
        # Projector features for pooled slots AND (when aux is on) the shadow candidates -- both
        # are P(mean input embeds of the doc), same distribution, same module.
        inj_rows = torch.cat([cb.soft_rows, cb.shadow_rows])
        inj_cols = torch.cat([cb.soft_cols, cb.shadow_cols])
        inj_docs = torch.cat([cb.soft_docs, cb.shadow_docs])
        mean_feats = doc_means[inj_rows, inj_docs]

        # Oracle slot lookup: hash each pooled doc's token span, gather its cached per-layer
        # slots. Docs missing from the cache stay on the projector path.
        oracle_ovr = None
        cache = cfg.get("oracle_cache")
        if cache is not None and cb.soft_rows.numel() > 0:
            import numpy as np

            from ..oracle_slot import doc_hash64

            ids_np = input_ids.cpu().numpy()
            cid_np = cid.cpu().numpy()
            s_rows = cb.soft_rows.cpu().numpy()
            s_docs = cb.soft_docs.cpu().numpy()
            row_order: Dict[int, Tuple[Any, Any]] = {}
            hashes = []
            for b, d in zip(s_rows.tolist(), s_docs.tolist()):
                if b not in row_order:
                    order = np.argsort(cid_np[b], kind="stable")
                    row_order[b] = (order, cid_np[b][order])
                order, srt = row_order[b]
                span_lo = int(np.searchsorted(srt, d))
                span_hi = int(np.searchsorted(srt, d, side="right"))
                hashes.append(doc_hash64(ids_np[b][order[span_lo:span_hi]]))
            idx = cache.lookup(hashes)
            found = idx >= 0
            st = cfg["_oracle_stats"]
            st["hits"] += int(found.sum())
            st["misses"] += int((~found).sum())
            st["calls"] += 1
            if st["misses"] > 0 and st["calls"] % 100 == 1:
                log.warning(
                    "[oracle-slot] cache misses so far: %d / %d slots (missing docs fall back "
                    "to the projector)",
                    st["misses"],
                    st["hits"] + st["misses"],
                )
            if st["calls"] >= 10 and st["hits"] == 0:
                raise RuntimeError(
                    "[oracle-slot] 0 cache hits after 10 forwards -- the doc-hash scheme "
                    "almost certainly mismatches the cache builder; refusing to silently "
                    "train plain B1"
                )
            if found.any():
                sel = torch.from_numpy(np.flatnonzero(found)).to(cb.soft_rows.device)
                slots, biases = cache.gather(idx[found])
                rows = cb.soft_rows[sel]
                cols = cb.soft_cols[sel]
                oracle_ovr = {
                    "rows": rows,
                    "cols": cols,
                    "pos": cb.position_ids[rows, cols],
                    "slots": slots.to(input_ids.device, non_blocking=True),
                    "biases": biases.to(input_ids.device, non_blocking=True),
                }
        return cb, (inj_rows, inj_cols, mean_feats), oracle_ovr

    def _run_blocks(
        self,
        h: torch.Tensor,
        all_block_kwargs: Dict[str, Any],
        per_block_kwargs: Dict[int, Dict[str, Any]],
        capture_layers: Optional[Set[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Run the block stack; optionally capture (attached) hidden states at given layer indices."""
        captured: Dict[int, torch.Tensor] = {}
        for block_key, block in self.blocks.items():
            block_idx = int(block_key)
            block_kwargs = per_block_kwargs.get(block_idx, {})
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
            if getattr(block, "_bskip", None) is not None:
                from ..block_skip import block_skip_forward

                h = block_skip_forward(block, h, {**all_block_kwargs, **block_kwargs})
            else:
                h = block(h, **all_block_kwargs, **block_kwargs)
            if capture_layers is not None and block_idx in capture_layers:
                captured[block_idx] = h
        return h, captured

    def set_landmark_eval_top_k(self, top_k: Optional[int]) -> int:
        """
        Enable inference-only hard top-k landmark retrieval on every eager landmark attention layer
        (e.g. :class:`~olmo_core.nn.attention.DocumentLandmarkAttention`): each query attends only the
        ``top_k`` highest-scoring landmark blocks. ``None`` restores exact (all-block) attention.
        Training is unaffected. Returns the number of layers updated.

        :param top_k: Number of landmark blocks to keep per query, or ``None`` for exact.
        """
        n = 0
        for module in self.modules():
            if hasattr(module, "set_eval_top_k") and callable(module.set_eval_top_k):
                module.set_eval_top_k(top_k)
                n += 1
        return n

    def compute_auxiliary_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        del reset
        return {}

    def reset_auxiliary_metrics(self):
        pass

    @property
    def pp_enabled(self) -> bool:
        return self._pp_enabled

    @property
    def fp8_enabled(self) -> bool:
        return self._fp8_enabled

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    @property
    def fsdp_enabled(self) -> bool:
        return self._fsdp_enabled

    @property
    def is_moe(self) -> bool:
        return False

    @property
    def device(self) -> torch.device:
        if self._device is None:
            for p in self.parameters():
                if p.numel() > 0:
                    self._device = p.device
                    break
            else:
                self._device = get_default_device()
        return self._device

    @property
    def compile_enabled(self) -> bool:
        return self._compile_enabled

    def get_rope_buffers(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> Dict[int, Optional[RoPEBuffers]]:
        """
        Get the RoPE buffers to pass to each layer.
        """
        if device is None:
            device = self.device
        rope_buffers = {}
        for key, block in self.blocks.items():
            if isinstance(block.attention, (Attention, FusedAttention)):
                rope = cast(Optional[RotaryEmbeddingBase], block.attention.rope)
                rope_buffers[int(key)] = None if rope is None else rope.get_buffers(seq_len, device)
            else:
                rope_buffers[int(key)] = None
        return rope_buffers

    @torch.no_grad()
    def init_weights(
        self,
        *,
        max_seq_len: Optional[int] = None,
        max_local_microbatch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        world_mesh: Optional[DeviceMesh] = None,
        model_part_idx: int = 0,
    ) -> torch.Generator:
        """
        Initialize the model weights.

        :param max_seq_len: The maximum sequence length expected. This is used
            to warm up the RoPE cache.
        :param max_local_microbatch_size: The maximum local (rank) micro-batch size (in tokens)
            expected. This is used to warm-up some MoE cache.
        :param device: The device the local copy of the model will be trained on.
        :param model_part_idx: The local index of this model part on the current rank.
            With interleaved pipeline schedules a single rank can own multiple model
            chunks, and each must receive a distinct seed; otherwise their parameters
            would be identical.
        """
        device = device or self.device
        self.to_empty(device=device)

        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()  # type: ignore

        seed = self.init_seed
        if world_mesh is not None and self.pp_enabled:
            pp_mesh = get_pp_mesh(world_mesh)
            seed += pp_mesh.get_local_rank() + model_part_idx * pp_mesh.size()

        generator = torch.Generator(device).manual_seed(seed)

        if self.embeddings is not None:
            self.init_method.init_embeddings(
                self.embeddings,
                d_model=self.d_model,
                embed_scale=self.embed_scale,
                std=(
                    self.embedding_init_std
                    if self.embedding_init_std is not None
                    else self.init_std
                ),
                generator=generator,
            )

        for block in self.blocks.values():
            # This might fail if it's wrapped.
            #  assert isinstance(block, TransformerBlock)
            block = cast(TransformerBlock, block)
            att = cast(SequenceMixer, block.attention)

            # Attention weights.
            self.init_method.init_attention(
                att,
                d_model=self.d_model,
                block_idx=block.block_idx,
                num_blocks=self.n_layers,
                std=self.init_std,
                generator=generator,
            )

            # Feed-forward weights.
            if hasattr(block, "feed_forward"):
                self.init_method.init_feed_forward(
                    block.feed_forward,
                    d_model=self.d_model,
                    block_idx=block.block_idx,
                    num_blocks=self.n_layers,
                    std=self.init_std,
                    generator=generator,
                )

            # MoE weights.
            if hasattr(block, "feed_forward_moe"):
                block = cast(MoETransformerBlock, block)
                if max_local_microbatch_size is not None:
                    block.feed_forward_moe.warmup_cache(max_local_microbatch_size)
                self.init_method.init_feed_forward_moe(
                    block.feed_forward_moe,
                    d_model=self.d_model,
                    block_idx=block.block_idx,
                    num_blocks=self.n_layers,
                    std=self.init_std,
                    generator=generator,
                )

            if isinstance(att, (Attention, FusedAttention)):
                # Warm up attention backend cache.
                if max_seq_len is not None and att.backend is not None:
                    att.backend.warmup_cache(max_seq_len, device)

                # Warm up RoPE cache.
                if max_seq_len is not None and att.rope is not None:
                    att.rope.warmup_cache(max_seq_len, device)

        if self.lm_head is not None:
            self.init_method.init_final_w_out(
                self.lm_head.w_out,
                d_model=self.d_model,
                std=self.init_std,
                generator=generator,
            )

        return generator

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Dict[str, Any],
        Dict[int, Dict[str, Any]],
        Dict[str, Any],
    ]:
        # NOTE: with pipeline parallelism input_ids might actually be an intermediate output,
        # so we have to be careful here.
        B, S = input_ids.shape[:2]

        all_block_kwargs: Dict[str, Any] = {}
        per_block_kwargs: Dict[int, Dict[str, Any]] = defaultdict(dict)
        lm_head_kwargs: Dict[str, Any] = dict(
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            return_logits=return_logits,
            logits_to_keep=logits_to_keep,
        )

        if loss_div_factor is not None:
            loss_div_factor = move_to_device(loss_div_factor, self.device)
            lm_head_kwargs["loss_div_factor"] = loss_div_factor
            all_block_kwargs["loss_div_factor"] = loss_div_factor

        # Prepare document length inputs.
        max_doc_len: Optional[int] = None
        cu_doc_lens: Optional[torch.Tensor] = None
        doc_lens: Optional[torch.Tensor] = None
        cache_leftpad: Optional[torch.Tensor] = kwargs.pop("cache_leftpad", None)

        # Explicit per-token positions (soft-token pooling: kept tokens keep their ORIGINAL
        # positions so RoPE geometry matches full-attention inference). Threaded to every block.
        if (position_ids := kwargs.pop("position_ids", None)) is not None:
            if self._cp_load_balancer is not None:
                raise NotImplementedError("position_ids is not supported with context parallelism")
            all_block_kwargs["position_ids"] = move_to_device(position_ids, self.device)
        # Soft-token aux matching: the position-causal masked-attention bias + the shared per-layer
        # capture dict (each Attention appends its q/k/v slices for the matching loss).
        if (attn_bias := kwargs.pop("attn_bias", None)) is not None:
            all_block_kwargs["attn_bias"] = move_to_device(attn_bias, self.device)
        if (aux_capture := kwargs.pop("aux_capture", None)) is not None:
            all_block_kwargs["aux_capture"] = aux_capture
        if (kv_grad_mask := kwargs.pop("kv_grad_mask", None)) is not None:
            all_block_kwargs["kv_grad_mask"] = move_to_device(kv_grad_mask, self.device)
        # Oracle slot K/V overrides (soft-token pooling): one gathered slot stack for the batch,
        # sliced per layer into per-block kwargs (each Attention rotates + injects its own slice).
        if (oracle_ovr := kwargs.pop("soft_kv_override_layers", None)) is not None:
            slots = move_to_device(oracle_ovr["slots"], self.device)
            biases = move_to_device(oracle_ovr["biases"], self.device)
            rows = move_to_device(oracle_ovr["rows"], self.device)
            cols = move_to_device(oracle_ovr["cols"], self.device)
            slot_pos = move_to_device(oracle_ovr["pos"], self.device)
            for li in range(slots.shape[1]):
                per_block_kwargs[li]["soft_kv_override"] = {
                    "rows": rows,
                    "cols": cols,
                    "pos": slot_pos,
                    "k": slots[:, li, 0],
                    "v": slots[:, li, 1],
                    "bias": biases[:, li],
                }

        if (doc_lens := kwargs.pop("doc_lens", None)) is not None and (
            max_doc_lens := kwargs.pop("max_doc_lens", None)
        ) is not None:
            max_doc_len = max(max_doc_lens)
            cu_doc_lens = get_cumulative_document_lengths(doc_lens)

        if self._document_chunk_attention is not None and self._cp_load_balancer is not None:
            raise NotImplementedError(
                "Document-chunked landmark attention (enable_document_chunk_attention) is not yet "
                "supported together with context parallelism (chunk_ids are not sequence-sharded)."
            )

        # Shard inputs and RoPE buffers on sequence dimension if using context parallelism.
        if (cp_load_balancer := self._cp_load_balancer) is not None:
            inputs = [input_ids]
            seq_dims = [1]
            pad_values: List[Union[int, float]] = [0]
            keys = ["input_ids"]

            # NOTE: initialize buffer(s) on CPU to avoid possible host-device sync when sharding.
            for block_idx, rope_buffers in self.get_rope_buffers(S, torch.device("cpu")).items():
                if rope_buffers is not None:
                    # Also shard RoPE buffers based on the context parallelism load balancer.
                    if rope_buffers.pos_sin is not None:
                        inputs.append(rope_buffers.pos_sin)
                        seq_dims.append(0)
                        pad_values.append(0.0)
                        keys.append(f"block_{block_idx}.pos_sin")
                    if rope_buffers.pos_cos is not None:
                        inputs.append(rope_buffers.pos_cos)
                        seq_dims.append(0)
                        pad_values.append(0.0)
                        keys.append(f"block_{block_idx}.pos_cos")
                    if rope_buffers.freqs_cis is not None:
                        inputs.append(rope_buffers.freqs_cis)
                        seq_dims.append(0)
                        pad_values.append(0.0)
                        keys.append(f"block_{block_idx}.freqs_cis")

            if labels is not None:
                inputs.append(labels)
                seq_dims.append(1)
                pad_values.append(ignore_index)
                keys.append("labels")

            if cache_leftpad is not None:
                raise NotImplementedError("cache_leftpad is not supported with context parallelism")

            if cu_doc_lens is not None:
                # NOTE: Can only shard properly here if 'input_ids' is flat, i.e. a single instance.
                # TODO: (epwalsh) We could just flatten all of the inputs here, but then we risk going
                # beyond the model's maximum sequence length, which might be okay at least
                # with relative positional encodings, but then again if you're resorting to context
                # parallelism you can probably only fit a single instance at a time anyway.
                if B != 1:
                    raise RuntimeError(
                        f"Rank micro-batches must consist of a single instance when using "
                        f"context parallelism with intra-document masking (got {B} instances)"
                    )
                inputs, additional_inputs = cp_load_balancer.batch_shard_by_document(
                    inputs=inputs,
                    seq_dims=seq_dims,
                    cu_doc_lens=cu_doc_lens,
                    pad_values=pad_values,
                    length_multiple=16,
                )
                for key, value in additional_inputs.items():
                    all_block_kwargs[key] = move_to_device(value, self.device)

            else:
                inputs = cp_load_balancer.batch_shard(
                    inputs=inputs,
                    seq_dims=seq_dims,
                    pad_values=pad_values,
                )

            for key, value in zip(keys, inputs):
                if key.startswith("block_"):
                    block_key, key = key.split(".", 1)
                    block_idx = int(block_key.replace("block_", ""))
                    per_block_kwargs[block_idx][key] = move_to_device(value, self.device)
                else:
                    all_block_kwargs[key] = move_to_device(value, self.device)

            input_ids = all_block_kwargs.pop("input_ids")
            labels = all_block_kwargs.pop("labels", None)
        else:
            input_ids = move_to_device(input_ids, self.device)
            labels = move_to_device(labels, self.device)

            if (max_doc_len is not None or cu_doc_lens is not None) and cache_leftpad is not None:
                raise ValueError("max_doc_len/cu_doc_lens and cache_leftpad are mutually exclusive")
            if max_doc_len is not None or cu_doc_lens is not None:
                all_block_kwargs["max_doc_len"] = max_doc_len
                all_block_kwargs["cu_doc_lens"] = move_to_device(cu_doc_lens, self.device)
            if cache_leftpad is not None:
                all_block_kwargs["cache_leftpad"] = move_to_device(cache_leftpad, self.device)

            # Document-chunked landmark attention: reconstruct per-token chunk_ids from the boundary
            # tokens and forward them to every block (see ``enable_document_chunk_attention``).
            if self._document_chunk_attention is not None:
                cfg = self._document_chunk_attention
                chunk_ids = build_chunk_ids_from_tokens(
                    input_ids,
                    doc_start_id=cfg["doc_start_id"],
                    doc_end_id=cfg["doc_end_id"],
                    eos_id=cfg["eos_id"],
                    mode=cfg["mode"],
                    pad_id=cfg.get("pad_id"),
                )
                # Mask mixing (training only): with a scheduled, seeded probability, collapse a random
                # subset of examples to plain causal by neutralizing their roles to all-FREE. Runs once
                # per forward on the shared chunk_ids (before they are threaded to every block), so all
                # layers and any AC recompute see the same mixed roles. p == 0 (no mix configured, or a
                # forward where no example is drawn) leaves chunk_ids untouched -> bit-identical to
                # pure chunked. Kept eager (python-seeded coin + counter) -- exclude from torch.compile.
                mix = cfg.get("mix")
                if mix is not None and self.training:
                    idx = mix["forward_idx"]
                    p = mask_mix_standard_prob(
                        idx,
                        standard_mix_prob=mix["standard_mix_prob"],
                        mix_start_p=mix["mix_start_p"],
                        mix_end_p=mix["mix_end_p"],
                        mix_total_forwards=mix["mix_total_forwards"],
                    )
                    mix["forward_idx"] = idx + 1
                    new_chunk_ids = collapse_roles_to_causal(
                        chunk_ids, p, forward_idx=mix["forward_idx"], mix_seed=mix["mix_seed"]
                    )
                    if (
                        new_chunk_ids is not chunk_ids
                    ):  # at least one example collapsed this forward
                        mix["n_collapsed"] += 1
                    chunk_ids = new_chunk_ids
                    if p > 0.0 and mix["forward_idx"] % mix["log_interval"] == 0:
                        log.info(
                            f"[curriculum] forward={mix['forward_idx']} p_standard={p:.3f} "
                            f"collapsed_forwards={mix['n_collapsed']}"
                        )
                all_block_kwargs["chunk_ids"] = move_to_device(chunk_ids, self.device)

        if "cu_doc_lens" in all_block_kwargs:
            mark_dynamic(all_block_kwargs["cu_doc_lens"], 0, strict=False)  # type: ignore[arg-type]

        return (
            input_ids,
            labels,
            all_block_kwargs,
            per_block_kwargs,
            lm_head_kwargs,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run the transformer on the token input IDs.

        :param input_ids: The token input IDs, shape ``(batch_size, seq_len)``.
        :param labels: The token labels, shape ``(batch_size, seq_len)``.
        :param ignore_index: The index to ignore in the loss computation. Default is -100.
        :param loss_reduction: The reduction method for the loss. Can be "mean", "sum", or "none".
        :param z_loss_multiplier: Optional multiplier for the z-loss regularization term.
        :param loss_div_factor: Optional divisor for the loss, can be a scalar or tensor.
        :param return_logits: Whether to return logits along with the loss when labels are provided.
        :param logits_to_keep: Number of positions to keep from the end of the sequence (if int),
            or tensor specifying which positions to keep. Default is 0 (keep all).

        :returns: The logits if ``labels`` is ``None`` or the losses if ``labels`` is not ``None``.
        """
        # Soft-token document pooling ("B1"): during training, compact the sequence (pooled docs ->
        # one placeholder each) BEFORE anything else sees it. Eval/generation forwards untouched.
        soft_inject: Optional[Tuple] = None
        aux_ctx: Optional[Dict[str, Any]] = None
        distill_teacher: Optional[Dict[str, Any]] = None
        if (
            self._pooled_soft_tokens is not None
            and self.training
            and not input_ids.is_floating_point()  # PP passes hidden states; only act on token ids
        ):
            # Paired-distillation coin: synchronized across ranks via the shared forward counter.
            pst0 = self._pooled_soft_tokens
            paired = False
            if pst0["distill_prob"] > 0.0 and labels is not None:
                import random as _random

                cnt = pst0["_distill_counter"]
                pst0["_distill_counter"] = cnt + 1
                paired = (
                    _random.Random(f"distill:{pst0['keep_seed']}:{cnt}").random()
                    < pst0["distill_prob"]
                )
            if paired:
                # TEACHER pass: the ORIGINAL row, plain full attention, WITH LM gradient.
                t_ids, t_labels, t_abk, t_pbk, t_lmk = self._prepare_inputs(
                    input_ids,
                    labels,
                    ignore_index=ignore_index,
                    loss_reduction=loss_reduction,
                    z_loss_multiplier=z_loss_multiplier,
                    loss_div_factor=loss_div_factor,
                    return_logits=False,
                )
                h_t = self.embeddings(t_ids)  # type: ignore[misc]
                if self.embed_scale is not None:
                    h_t = h_t * self.embed_scale
                if self.embedding_norm is not None:
                    h_t = self.embedding_norm(h_t)
                h_t, t_caps = self._run_blocks(
                    h_t, t_abk, t_pbk, capture_layers=set(pst0["distill_layers"])
                )
                t_lmk["labels"] = t_labels
                t_out = self.lm_head(h_t, **t_lmk)  # type: ignore[misc]
                # Distill positions: every labeled (answer) position PLUS sampled FREE-region
                # positions -- free tokens survive compaction, so both passes have them, and
                # matching across the whole row constrains the pathway far more per teacher
                # than answer positions alone (the 32k signal-density fix).
                ans_r, ans_c = (t_labels != ignore_index).nonzero(as_tuple=True)
                roles_t = build_chunk_ids_from_tokens(
                    t_ids,
                    doc_start_id=pst0["doc_start_id"],
                    doc_end_id=pst0["doc_end_id"],
                    eos_id=pst0["eos_id"],
                    mode="chunked",
                )
                free_r, free_c = (roles_t == -1).nonzero(as_tuple=True)
                n_extra = min(96 * t_ids.shape[0], int(free_r.numel()))
                if n_extra > 0:
                    sel_f = torch.randperm(free_r.numel(), device=free_r.device)[:n_extra]
                    d_rows = torch.cat([ans_r, free_r[sel_f]])
                    d_pos = torch.cat([ans_c, free_c[sel_f]])
                else:
                    d_rows, d_pos = ans_r, ans_c
                distill_teacher = {
                    "loss_full": t_out.loss,
                    "rows": d_rows,
                    "pos": d_pos,
                    "caps": {li: t_caps[li][d_rows, d_pos].detach() for li in t_caps},
                }
            n_tokens_in = int(input_ids.numel())
            compacted = self._compact_pooled_soft_tokens(input_ids, labels, ignore_index)
            if compacted is not None:
                cb, soft_inject, oracle_ovr = compacted
                input_ids, labels = cb.input_ids, cb.labels
                # Compaction accounting for the FLOP meter (accumulated across microbatches; the
                # FlopMeterCallback reads and resets it every step).
                stats = getattr(self, "_soft_token_compaction", None)
                if stats is None:
                    stats = self._soft_token_compaction = {"tokens_in": 0, "tokens_out": 0, "rows": 0}
                stats["tokens_in"] += n_tokens_in
                stats["tokens_out"] += int(input_ids.numel())
                stats["rows"] += int(input_ids.shape[0])
                kwargs["position_ids"] = cb.position_ids
                if oracle_ovr is not None:
                    kwargs["soft_kv_override_layers"] = oracle_ovr
                    # Slot biases need the additive-bias SDPA path: position-causality is
                    # equivalent to sequence-causality on the compacted row (content is sorted
                    # by original position), so this only changes the kernel, not the math.
                    from ..pooled_soft_token import build_position_causal_bias

                    kwargs.setdefault(
                        "attn_bias",
                        build_position_causal_bias(
                            cb,
                            dtype=self.embeddings.weight.dtype,  # type: ignore[union-attr]
                            device=input_ids.device,
                        ),
                    )
                pst = self._pooled_soft_tokens
                if pst["detach_soft_kv"] and cb.soft_rows.numel() > 0:
                    kv_grad_mask = torch.ones_like(cb.input_ids, dtype=torch.bool)
                    kv_grad_mask[cb.soft_rows, cb.soft_cols] = False
                    kwargs["kv_grad_mask"] = kv_grad_mask
                if (
                    pst["aux_match_weight"] > 0.0
                    and cb.shadow_rows.numel() > 0
                    and labels is not None
                ):
                    from ..pooled_soft_token import build_position_causal_bias

                    kwargs["attn_bias"] = build_position_causal_bias(
                        cb, dtype=self.embeddings.weight.dtype, device=input_ids.device  # type: ignore[union-attr]
                    )
                    lab_rows, lab_cols = (labels != ignore_index).nonzero(as_tuple=True)
                    if lab_rows.numel() > 0:
                        nq = min(pst["aux_queries"] * cb.input_ids.shape[0], int(lab_rows.numel()))
                        sel = torch.randperm(lab_rows.numel(), device=lab_rows.device)[:nq]
                        valid = cb.shadow_doc_cols >= 0
                        n_sh = cb.shadow_rows.numel()
                        aux_ctx = {
                            "rows_q": lab_rows[sel],
                            "cols_q": lab_cols[sel],
                            "rows_kv": cb.shadow_rows[:, None].expand_as(cb.shadow_doc_cols)[valid],
                            "cols_kv": cb.shadow_doc_cols[valid],
                            "doc_of_kv": torch.arange(n_sh, device=lab_rows.device)[
                                :, None
                            ].expand_as(cb.shadow_doc_cols)[valid],
                            "rows_sh": cb.shadow_rows,
                            "cols_sh": cb.shadow_cols,
                            "log_len": cb.shadow_log_len,
                            "layers": [],
                        }
                        kwargs["aux_capture"] = aux_ctx
        # Role-gated FFN: gate mask from the FINAL token stream (post-compaction when the
        # soft-token path rewrote input_ids), so kept-doc tokens are gated in compacted rows too.
        if self._role_gated_ffn is not None:
            self._set_role_gate_mask(input_ids)
        # Nested-FFN router: reset per-forward accumulators and advance the budget/exploration
        # schedules. Loss terms are only collected when we are actually computing a loss.
        if self._nested_ffn_moe is not None:
            self._nested_ffn_moe["holder"].begin_forward(collect_loss=labels is not None)
        if self._kv_route is not None:
            self._kv_route["holder"].begin_forward(collect_loss=labels is not None)
        if self._block_skip is not None:
            self._block_skip["holder"].begin_forward(collect_loss=labels is not None)

        (
            input_ids,
            labels,
            all_block_kwargs,
            per_block_kwargs,
            lm_head_kwargs,
        ) = self._prepare_inputs(
            input_ids,
            labels,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            loss_div_factor=loss_div_factor,
            return_logits=return_logits,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # Get embeddings but pass-through for non-existent layers to allow easy
        # pipeline parallel configuration.
        h = self.embeddings(input_ids) if self.embeddings is not None else input_ids
        if soft_inject is not None:
            # Overwrite the placeholder embeddings with the projector's soft doc tokens. Under
            # detach_soft_kv the POOLED slots are injected detached (static-KV semantics; the
            # per-layer K/V cut happens in Attention via kv_grad_mask) while SHADOW candidates
            # stay attached so the aux objective can train the projector.
            s_rows, s_cols, mean_feats = soft_inject
            assert self.pooled_projector is not None
            h = h.clone()
            soft_vecs = self.pooled_projector(mean_feats).to(h.dtype)
            if self._pooled_soft_tokens["detach_soft_kv"]:  # type: ignore[index]
                n_slots = s_rows.shape[0] - (
                    aux_ctx["rows_sh"].shape[0] if aux_ctx is not None else 0
                )
                soft_vecs = torch.cat([soft_vecs[:n_slots].detach(), soft_vecs[n_slots:]], dim=0)
            h[s_rows, s_cols] = soft_vecs
        if self.embeddings is not None and self.embed_scale is not None:
            h = h * self.embed_scale
        if self.embedding_norm is not None:
            h = self.embedding_norm(h)

        # Run each block (capturing student hiddens at the distill layers on paired forwards).
        cap_set = (
            set(self._pooled_soft_tokens["distill_layers"])  # type: ignore[index]
            if distill_teacher is not None
            else None
        )
        h, s_caps = self._run_blocks(h, all_block_kwargs, per_block_kwargs, capture_layers=cap_set)

        # Aux attention-contribution matching loss (soft-token pooling; see aux_matching_loss).
        aux_loss: Optional[torch.Tensor] = None
        if aux_ctx is not None and aux_ctx["layers"]:
            from ..pooled_soft_token import aux_matching_loss

            first_block = next(iter(self.blocks.values()))
            scale = first_block.attention.head_dim**-0.5  # type: ignore[union-attr]
            aux_loss = aux_matching_loss(
                [
                    (qs, kd, vd, aux_ctx["doc_of_kv"], ksh, vsh)
                    for (qs, kd, vd, ksh, vsh) in aux_ctx["layers"]
                ],
                q_rows=aux_ctx["rows_q"],
                shadow_rows=aux_ctx["rows_sh"],
                shadow_log_len=aux_ctx["log_len"],
                scale=scale,
            )

        # Get final logits but again pass-through in case of pipeline parallelism.
        if self.lm_head is not None:
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
                if labels is not None:
                    mark_dynamic(labels, (0, 1), strict=False)
            # NOTE: When TP is active we can't pass 'labels=None' or the hook from 'PrepareModuleInput'
            # will throw an exception.
            if labels is not None:
                lm_head_kwargs["labels"] = labels
            out = self.lm_head(h, **lm_head_kwargs)
            if isinstance(out, LMOutputWithLoss) and out.loss is not None:
                loss = out.loss
                if aux_loss is not None:
                    w = self._pooled_soft_tokens["aux_match_weight"]  # type: ignore[index]
                    loss = loss + w * aux_loss.to(loss.dtype)
                if self._nested_ffn_moe is not None:
                    nffn_loss = self._nested_ffn_moe["holder"].regularization_loss()
                    if nffn_loss is not None:
                        loss = loss + nffn_loss.to(loss.dtype)
                if self._kv_route is not None:
                    kvr_loss = self._kv_route["holder"].regularization_loss()
                    if kvr_loss is not None:
                        loss = loss + kvr_loss.to(loss.dtype)
                if self._block_skip is not None:
                    bs_loss = self._block_skip["holder"].regularization_loss()
                    if bs_loss is not None:
                        loss = loss + bs_loss.to(loss.dtype)
                if self._joint_budget is not None:
                    from ..joint_budget import joint_budget_loss

                    jb_loss = joint_budget_loss(self)
                    if jb_loss is not None:
                        loss = loss + jb_loss.to(loss.dtype)
                if distill_teacher is not None and labels is not None:
                    # Map teacher (row, ORIGINAL position) -> student compacted column via the
                    # per-row ascending position_ids (free/answer tokens survive compaction).
                    pos_ids_s = all_block_kwargs.get("position_ids")
                    s_r = distill_teacher["rows"]
                    if pos_ids_s is not None:
                        s_c = (
                            torch.searchsorted(
                                pos_ids_s[s_r].contiguous(),
                                distill_teacher["pos"][:, None].contiguous(),
                            )
                            .squeeze(-1)
                            .clamp(max=pos_ids_s.shape[1] - 1)
                        )
                    else:
                        s_c = distill_teacher["pos"]
                    terms = []
                    for li, t_h in distill_teacher["caps"].items():
                        s_h = s_caps[li][s_r, s_c].float()
                        t_f = t_h.float()
                        terms.append(
                            ((s_h - t_f) ** 2).sum(-1).mean()
                            / t_f.pow(2).sum(-1).mean().clamp(min=1e-6)
                        )
                    d_loss = torch.stack(terms).mean()
                    w_d = self._pooled_soft_tokens["distill_weight"]  # type: ignore[index]
                    loss = loss + distill_teacher["loss_full"] + w_d * d_loss.to(loss.dtype)
                if loss is not out.loss:
                    out = LMOutputWithLoss(out.logits, loss, out.ce_loss, out.z_loss)
            return out
        else:
            return h

    def apply_fp8(self, float8_config: Float8Config):
        """
        Use an FP8 recipe on most linear layers.
        """
        if not float8_config.enabled:
            return

        modules_to_ignore = set()
        if self.lm_head is not None:
            modules_to_ignore.add("lm_head.w_out")
        if float8_config.modules_to_ignore is not None:
            modules_to_ignore.update(float8_config.modules_to_ignore)

        float8_config.apply_float8_linear(self, modules_to_ignore=modules_to_ignore)

        self._fp8_enabled = True
        self._precompute_float8_dynamic_scale_for_fsdp = (
            float8_config.should_precompute_float8_dynamic_scale_for_fsdp
        )

    def apply_pp(self, pp_mesh: DeviceMesh):
        """
        Prepare the model for pipeline parallelism after it's been split into stages.
        """
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_pp(pp_mesh)
        self._pp_enabled = True
        self._pp_group_size = pp_mesh.size()

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: Optional[bool] = None):
        """
        Apply tensor parallelism to the model.

        :param loss_parallel: Set to ``True`` if parallelizing the loss function as well.
        :param float8_enabled: Set this to ``True`` if training with float8 linear layers.
        """
        if float8_enabled is None:
            float8_enabled = self.fp8_enabled
        elif not float8_enabled and self.fp8_enabled:
            raise OLMoConfigurationError(
                "Got 'float8_enabled=False', but FP8 has already been enabled"
            )

        if self.embeddings is not None:
            parallelize_module(
                self.embeddings,
                device_mesh=tp_mesh,
                parallelize_plan=RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                    use_local_output=False,
                ),
            )
        if self.embedding_norm is not None:
            parallelize_module(
                self.embedding_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
            )

        # Apply tensor/sequence parallelism to every transformer block.
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_tp(tp_mesh, input_layout=Shard(1), float8_enabled=float8_enabled)

        if self.lm_head is not None:
            self.lm_head.apply_tp(tp_mesh, input_layouts=(Shard(1), Replicate()))

        self._tp_enabled = True
        self._tp_mesh = tp_mesh

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: RingContextParallelStyle | None = None,
        uly: UlyssesContextParallelStyle | None = None,
    ):
        """
        Prepare the model for context-parallelism (CP).

        :param cp_mesh: The CP device mesh.
        :param ring: The ring context parallel style.
        :param uly: The ulysses context parallel style.
        """
        if ring is not None:
            self._cp_load_balancer = ring.load_balancer.build(cp_mesh)
        elif uly is not None:
            self._cp_load_balancer = uly.load_balancer.build(cp_mesh)

        for block in self.blocks.values():
            cast(TransformerBlockBase, block).apply_cp(cp_mesh, ring=ring, uly=uly)
        if self.lm_head is not None:
            self.lm_head.apply_cp(cp_mesh)

    def apply_activation_checkpointing(
        self,
        mode: TransformerActivationCheckpointingMode,
        block_interval: Optional[int] = None,
        modules: Optional[List[str]] = None,
        activation_memory_budget: Optional[float] = None,
    ):
        """
        Apply activation checkpointing to the model.

        :param mode: Determines how to apply activation checkpointing.
        :param block_interval: Required when :data:`mode` is "selected_blocks". Determines
            which blocks are wrapped.
        :param modules: Required when :data:`mode` is "selected_modules". A list of modules names
            to wrap for activation checkpointing. Globs are supported.
        :param activation_memory_budget: The memory budget for activation checkpointing in the range
            [0, 1]. 0 corresponds to the memory usage when recomputing all activations, and 1
            corresponds to the memory usage when recomputing no activations (which is the default).
            Requires compilation to be enabled.
        """

        if mode == TransformerActivationCheckpointingMode.budget:
            if activation_memory_budget is None:
                raise ValueError("'activation_memory_budget' is required for 'budget' mode")
            if activation_memory_budget < 0 or activation_memory_budget > 1:
                raise ValueError("'activation_memory_budget' must be in the range [0, 1]")
            torch._functorch.config.activation_memory_budget = activation_memory_budget
            return

        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper as ptd_checkpoint_wrapper,
        )

        if (
            mode == TransformerActivationCheckpointingMode.selected_blocks
            and block_interval is None
        ):
            raise ValueError("'block_interval' is required for 'selected_blocks' mode")

        if mode == TransformerActivationCheckpointingMode.selected_modules and modules is None:
            raise ValueError("'modules' is required for 'selected_modules' mode")

        # TODO: only preserve RNG state if dropout is active
        preserve_rng_state = False

        if mode == TransformerActivationCheckpointingMode.selected_modules:
            from fnmatch import fnmatch

            assert modules is not None
            wrapped_modules: Set[str] = set()
            for name, module in self.named_modules():
                for pattern in modules:
                    if fnmatch(name, pattern):
                        break
                else:
                    continue

                if isinstance(module, MoEBase):
                    raise OLMoConfigurationError(
                        "Wrapping an entire MoE module for activation checkpointing is not supported. "
                        "Please try a finer-grained wrapping strategy."
                    )

                # NOTE: have to be careful not to try to wrap submodules of modules that have been wrapped.
                parent_name = ".".join(name.split(".")[:-1])
                if parent_name in wrapped_modules:
                    continue

                parent = self if not parent_name else self.get_submodule(parent_name)
                module = ptd_checkpoint_wrapper(module, preserve_rng_state=preserve_rng_state)
                parent.register_module(name.split(".")[-1], module)
                log.info(f"Wrapped '{name}' for activation checkpointing")
                wrapped_modules.add(name)
        else:
            for block_idx, block in enumerate(self.blocks.values()):
                if mode == TransformerActivationCheckpointingMode.selected_blocks:
                    assert block_interval is not None
                    if block_idx % block_interval == 0:
                        if isinstance(block, MoETransformerBlock):
                            raise OLMoConfigurationError(
                                "Wrapping MoE blocks for activation checkpointing is not supported."
                            )
                        block = ptd_checkpoint_wrapper(block, preserve_rng_state=preserve_rng_state)
                elif mode == TransformerActivationCheckpointingMode.full:
                    if isinstance(block, MoETransformerBlock):
                        raise OLMoConfigurationError(
                            "Wrapping MoE blocks for activation checkpointing is not supported."
                        )
                    block = ptd_checkpoint_wrapper(block, preserve_rng_state=preserve_rng_state)
                elif mode == TransformerActivationCheckpointingMode.selected_ops:
                    block = ptd_checkpoint_wrapper(
                        block,
                        context_fn=selective_checkpointing_context_fn,
                        preserve_rng_state=preserve_rng_state,
                    )

                self.blocks.register_module(str(block_idx), block)

    def apply_compile(self):
        """
        Apply ``torch.compile()`` to each transformer block, which makes compilation efficient
        due to repeated structure.

        .. warning::
            This must be called after :meth:`apply_activation_checkpointing()` but
            before :meth:`apply_fsdp()` or :meth:`apply_ddp()`.
        """
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_compile()

        if self.lm_head is not None:
            self.lm_head.compile(fullgraph=False)

        torch.compiler.config.dynamic_sources += "L['kwargs']['max_doc_len'],"
        self._compile_enabled = True

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
    ):
        """
        Apply FSDP(2) to the model.

        .. warning::
            This should generally be called last if using any other parallelism strategies or optimizations
            like :meth:`apply_compile()`.

        :param dp_mesh: The model data parallel device mesh.
        :param param_dtype: The data type to materialize params in. Defaults to the current param dtype.
        :param reduce_dtype: The data type for gradient reduction.
        :pp_enabled: If pipeline parallelism is also enabled.
        :prefetch_factor: For tuning the prefetch settings. 0 is the default, and higher values result
            in more aggressive prefetching.
        :wrapping_strategy: The wrapping strategy.
        """
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype or self.dtype, reduce_dtype=reduce_dtype
        )
        # NOTE: blocks get a policy with 'cast_forward_inputs' DISABLED, which is required for
        # correctness whenever activation checkpointing wraps a block (the AC wrapper sits
        # *outside* the block, so FSDP's pre-forward hook runs *inside* the checkpointed region).
        # FSDP2 only casts forward inputs in 'FSDPState._pre_forward', which early-returns when
        # the state is 'PRE_BACKWARD' -- i.e. exactly during the AC recompute. So any float32
        # tensor passed to a block as an argument reaches the block as 'param_dtype' in the
        # forward but as float32 in the recompute. Under context parallelism that's precisely the
        # CP-sharded RoPE buffers ('pos_sin'/'pos_cos', built in fp32 by '_prepare_inputs'), and
        # since torch.compile guards on input dtypes the recompute recompiles the block into a
        # differently-partitioned graph that saves a different sequence of tensors. The result is
        #   CheckpointError: torch.utils.checkpoint: Recomputed values for the following tensors
        #   have different metadata than during the forward pass.
        # on the very first backward (seen on 128k-context CP=2 runs; see
        # 'debug/cp_ac_rope_dtype/').
        # Disabling the cast is safe: the hidden state entering a block is already in
        # 'param_dtype' (the embedding layer is sharded with the same policy and every layer norm
        # preserves its input dtype), and the RoPE buffers are meant to stay in fp32 anyway --
        # 'RotaryEmbedding.forward' casts them itself with 'type_as'.
        block_mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype or self.dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=False,
        )
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)
        # For PP, do not reshard after forward to avoid per-microbatch all-gathers,
        # which can be expensive and non-overlapped
        reshard_after_forward = False if pp_enabled else True

        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_fsdp(
                dp_mesh=dp_mesh,
                prefetch_factor=prefetch_factor,
                wrapping_strategy=wrapping_strategy,
                reshard_after_forward=reshard_after_forward,
                mp_policy=block_mp_policy,
            )

        if self.embeddings is not None:
            fully_shard(
                self.embeddings,
                reshard_after_forward=reshard_after_forward,
                **fsdp_config,
            )
            # Embedding params are not needed for backwards computation.
            cast(FSDPModule, self.embeddings).set_unshard_in_backward(False)

        if wrapping_strategy != TransformerDataParallelWrappingStrategy.blocks:
            if self.embedding_norm is not None:
                fully_shard(self.embedding_norm, **fsdp_config)
            if self.lm_head is not None:
                fully_shard(self.lm_head, reshard_after_forward=False, **fsdp_config)

        fully_shard(self, reshard_after_forward=reshard_after_forward, **fsdp_config)
        # Some inputs need to be on CPU initially, but FSDP will move everything to model's
        # device if we don't hide it.
        self.register_forward_pre_hook(_hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True)
        self.register_forward_pre_hook(
            _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        )

        if prefetch_factor > 0:
            blocks = cast(List[FSDPModule], list(self.blocks.values()))
            for i in range(len(blocks)):
                block = blocks[i]
                if i + 1 < len(blocks):
                    block.set_modules_to_forward_prefetch(blocks[i + 1 : i + 1 + prefetch_factor])
                elif isinstance(self.lm_head, FSDPModule):
                    block.set_modules_to_forward_prefetch([self.lm_head])

        self._fsdp_enabled = True

    def apply_ddp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        compile_enabled: bool = False,
        autograd_compile_enabled: bool = False,
    ):
        """
        Apply DDP to the model.
        """
        from torch.distributed._composable.replicate import replicate

        # Cast model explicitly to the specified dtype before applying DDP
        target_dtype = param_dtype or self.dtype
        if target_dtype != self.dtype:
            self.to(dtype=target_dtype)

        # Adapted from
        # https://github.com/pytorch/torchtitan/blob/90c889e972b56b9faadebbb78fc985dedc537ed9/torchtitan/parallelisms/parallelize_llama.py#L328
        if compile_enabled:
            if autograd_compile_enabled:
                torch._dynamo.config.optimize_ddp = "python_reducer_without_compiled_forward"  # type: ignore
            else:
                torch._dynamo.config.optimize_ddp = "ddp_optimizer"  # type: ignore

        replicate(self, device_mesh=dp_mesh, bucket_cap_mb=100)
        # Some inputs need to be on CPU initially, but DDP will move everything to model's
        # device if we don't hide it.
        self.register_forward_pre_hook(_hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True)
        self.register_forward_pre_hook(
            _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        )

    @cached_property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @cached_property
    def num_non_embedding_params(self) -> int:
        return self.num_params - self.embeddings.weight.numel()

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Returns the idealized number of flops per token for the given sequence length. Purposefully
        does not account for wasted flops due to padding, recomputation, etc.
        """
        flops_per_token = 0
        blocks = cast(List[TransformerBlockBase], list(self.blocks.values()))
        for block in blocks:
            flops_per_token += block.num_flops_per_token(seq_len)
        if self.lm_head is not None:
            flops_per_token += self.lm_head.num_flops_per_token(seq_len)
        return flops_per_token

    def post_batch(self, dry_run: bool = False):
        """
        Should be called right after the final backward of a complete batch but before the optimizer step.
        """
        del dry_run

    def post_optim_step(self):
        """
        Should be called right after an optimizer step.
        """
        if self.fp8_enabled and self._precompute_float8_dynamic_scale_for_fsdp:
            from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

            precompute_float8_dynamic_scale_for_fsdp(self)


@beta_feature
class NormalizedTransformer(Transformer):
    """
    A nGPT transformer implementation, to be used with the :class:`NormalizedTransformerBlock` block
    type.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        block: TransformerBlockConfig | dict[str, TransformerBlockConfig],
        lm_head: LMHeadConfig,
        dtype: torch.dtype = torch.float32,
        init_method: InitMethod = InitMethod.normalized,
        init_device: str = "cpu",
        init_seed: int = 0,
        init_std: float = 0.02,
        embedding_init_std: Optional[float] = None,
        block_overrides: Optional[Dict[int, TransformerBlockConfig]] = None,
        block_pattern: Optional[List[str]] = None,
    ):
        super().__init__(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=lm_head,
            dtype=dtype,
            init_method=init_method,
            init_device=init_device,
            init_seed=init_seed,
            init_std=init_std,
            embedding_init_std=embedding_init_std,
            block_overrides=block_overrides,
            block_pattern=block_pattern,
        )

    def _validate_block(self, block: TransformerBlockBase) -> TransformerBlockBase:
        if not isinstance(block, NormalizedTransformerBlock):
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}' requires a '{NormalizedTransformerBlock.__name__}' block"
            )
        return block

    @torch.no_grad()
    def init_weights(self, *args, **kwargs) -> torch.Generator:
        generator = super().init_weights(*args, **kwargs)
        self.normalize_matrices()
        return generator

    @torch.no_grad()
    def normalize_matrices(self):
        """
        Normalize the weights in all matrices. This should be called after each optimizer step, which
        the :class:`~olmo_core.train.train_module.TransformerTrainModule` will handle for you.
        """
        if self.embeddings is not None:
            self._normalize_matrix(self.embeddings.weight)

        for block in self.blocks.values():
            if hasattr(block, "normalize_matrices"):
                block.normalize_matrices()  # type: ignore

        if self.lm_head is not None:
            self.lm_head.normalize_matrices()  # type: ignore

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        float8_enabled: Optional[bool] = None,
    ):
        del tp_mesh, float8_enabled

        raise NotImplementedError(
            "TP is not implemented yet for the normalized transformer variant"
        )

    def apply_compile(self):
        super().apply_compile()
        self.normalize_matrices = torch.compile(self.normalize_matrices)

    def post_optim_step(self):
        super().post_optim_step()
        self.normalize_matrices()


@beta_feature
class MoETransformer(Transformer):
    """
    An MoE transformer implementation, to be used with one of the
    :class:`MoETransformerBlock` block types.
    """

    @property
    def is_moe(self) -> bool:
        return True

    def compute_auxiliary_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        mean_offset = 1.0
        if self.pp_enabled:
            # Change the divisor to 'world_size // pp_group_size'
            mean_offset = self._pp_group_size

        out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}
        for block_idx, block in self.blocks.items():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            block_metrics = block.compute_metrics(reset=reset)
            for metric_name, (metric_val, reduce_type) in block_metrics.items():
                out[f"block {int(block_idx):02d}/{metric_name}"] = (
                    metric_val,
                    reduce_type,
                )

                if self.pp_enabled and reduce_type == ReduceType.mean:
                    metric_val = metric_val.float() * mean_offset

                if metric_name not in out:
                    out[metric_name] = (metric_val, reduce_type)
                elif reduce_type in (ReduceType.mean, ReduceType.sum):
                    out[metric_name] = (
                        out[metric_name][0] + metric_val,
                        reduce_type,
                    )
                elif reduce_type == ReduceType.max:
                    out[metric_name] = (
                        torch.max(out[metric_name][0], metric_val),
                        reduce_type,
                    )
                else:
                    raise NotImplementedError(reduce_type)
        return out

    def reset_auxiliary_metrics(self):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            cast(MoETransformerBlock, block).reset_metrics()

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            block.apply_ep(ep_mesh, **kwargs)

    def prepare_experts_for_fsdp(
        self,
        world_mesh: DeviceMesh,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
    ):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            reshard_after_forward = True
            if pp_enabled or block.ep_enabled or block.tp_enabled:
                reshard_after_forward = False
            block.feed_forward_moe.prepare_experts_for_fsdp(
                world_mesh=world_mesh,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=param_dtype or self.dtype, reduce_dtype=reduce_dtype
                ),
                reshard_after_forward=reshard_after_forward,
            )

    def prepare_experts_for_ddp(self, world_mesh: DeviceMesh):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            cast(MoETransformerBlock, block).feed_forward_moe.prepare_experts_for_ddp(
                world_mesh=world_mesh,
            )

    def post_batch(self, dry_run: bool = False):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            block.feed_forward_moe.post_batch(dry_run=dry_run)


def _hide_cpu_inputs_from_torch(m, args, kwargs) -> Optional[Tuple[Any, Dict[str, Any]]]:
    del m
    if (doc_lens := kwargs.get("doc_lens")) is not None:
        kwargs["doc_lens"] = hide_from_torch(doc_lens)
    return (args, kwargs)


def _unhide_cpu_inputs_from_torch(m, args, kwargs) -> Optional[Tuple[Any, Dict[str, Any]]]:
    del m
    if (doc_lens := kwargs.get("doc_lens")) is not None:
        kwargs["doc_lens"] = unhide_from_torch(doc_lens)
    return (args, kwargs)
