#!/usr/bin/env python3
"""Build the active-matched geometry/GDN-expand-v=2 family at every size.

The 480M, 810M, and 1.2B models adopt the corresponding dense ladder's
450M, 810M, and 1.4B width/depth/head geometry and every-fifth-layer global
attention pattern. As in the first 275M ``geometry_only`` experiment, they
retain the MoE recipe, dense-first FFN, initialization, and the size's existing
GQA ratio. The optional NoPE profile changes only the full-attention layers'
``rope`` field. Attention gating adds only the dense ladder's elementwise,
full-precision gate and is supported with either RoPE or NoPE.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

from olmo_core.nn.attention import (
    AttentionConfig,
    GateConfig,
    GatedDeltaNet2Config,
    GateGranularity,
    KimiDeltaAttentionConfig,
)
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.ddp import OLMoDDPTransformerBlockConfig
from olmo_core.nn.transformer import OLMoDDPModelConfig
from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_275m import (
    build_geometry_matched_model_config as build_geometry_matched_275m_model_config,
)
from scripts.train.jacobm_olmoe_ladder.v2.models.hybrid_wide import (
    build_hybrid_model_config,
)


MODEL_SIZES = ("275m", "480m", "810m", "1p2b")
TOP_K = 8
HEAD_DIM = 128
GDN_EXPAND_V = 2.0


@dataclass(frozen=True)
class Geometry:
    dense_rung: str
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    expert_hidden_size: int
    expected_active_params: int
    expected_active_non_embedding_params: int
    expected_total_params: int
    dense_reference_active_params: int
    dense_reference_active_non_embedding_params: int

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(range(4, self.n_layers, 5))

    @property
    def gdn_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.n_layers) if i not in self.full_attention_layers)


GEOMETRIES = {
    # The established 275M configuration remains sourced from its original
    # audited builder. Its expert width was intentionally held at 664.
    "275m": Geometry(
        dense_rung="275m",
        d_model=640,
        n_layers=10,
        n_heads=8,
        n_kv_heads=4,
        expert_hidden_size=664,
        expected_active_params=290_782_080,
        expected_active_non_embedding_params=226_556_800,
        expected_total_params=3_136_314_240,
        dense_reference_active_params=275_493_760,
        dense_reference_active_non_embedding_params=211_268_480,
    ),
    "480m": Geometry(
        dense_rung="450m",
        d_model=768,
        n_layers=15,
        n_heads=8,
        n_kv_heads=4,
        expert_hidden_size=840,
        expected_active_params=501_137_856,
        expected_active_non_embedding_params=424_067_520,
        expected_total_params=7_220_707_776,
        dense_reference_active_params=454_118_400,
        dense_reference_active_non_embedding_params=377_047_040,
    ),
    "810m": Geometry(
        dense_rung="810m",
        d_model=1024,
        n_layers=15,
        n_heads=16,
        n_kv_heads=8,
        expert_hidden_size=1032,
        expected_active_params=858_237_056,
        expected_active_non_embedding_params=755_476_608,
        expected_total_params=11_865_532_544,
        dense_reference_active_params=810_354_816,
        dense_reference_active_non_embedding_params=707_594_368,
    ),
    "1p2b": Geometry(
        dense_rung="1.4b",
        d_model=1280,
        n_layers=20,
        n_heads=16,
        n_kv_heads=8,
        expert_hidden_size=952,
        expected_active_params=1_289_441_280,
        expected_active_non_embedding_params=1_160_990_720,
        expected_total_params=18_515_005_440,
        dense_reference_active_params=1_422_110_720,
        dense_reference_active_non_embedding_params=1_293_660_160,
    ),
}

# GDN2 changes only the recurrent mixer generation. These exact counts are the
# gated-NoPE GDN1 counts plus the per-mixer GDN2 parameter delta at each size.
EXPECTED_GDN2_GATED_NOPE_COUNTS = {
    "275m": (306_191_168, 241_965_888, 3_151_723_328),
    "480m": (526_979_424, 449_909_088, 7_246_549_344),
    "810m": (914_540_480, 811_780_032, 11_921_835_968),
    "1p2b": (1_376_964_352, 1_248_513_792, 18_602_528_512),
}
EXPECTED_CANONICAL_GDN2_GATED_NOPE_COUNTS = {
    "275m": (284_915_520, 220_690_240, 3_130_447_680),
    "480m": (489_954_144, 412_883_808, 7_209_524_064),
    "810m": (823_189_952, 720_429_504, 11_830_485_440),
    "1p2b": (1_228_949_248, 1_100_498_688, 18_454_513_408),
}
EXPECTED_GDN2_GATED_NOPE_COUNTS_BY_EXPAND_V = {
    1.0: EXPECTED_CANONICAL_GDN2_GATED_NOPE_COUNTS,
    2.0: EXPECTED_GDN2_GATED_NOPE_COUNTS,
}


def _geometry(model_size: str) -> Geometry:
    try:
        return GEOMETRIES[model_size]
    except KeyError as exc:
        raise ValueError(f"unknown model size {model_size!r}; choose one of {MODEL_SIZES}") from exc


def _resize_moe_block(
    block: OLMoDDPTransformerBlockConfig,
    *,
    d_model: int,
    expert_hidden_size: int,
) -> None:
    if block.shared_experts is None:
        raise ValueError("expected a shared expert config")
    if block.routed_experts is None or block.routed_experts_router is None:
        raise ValueError("expected routed expert and router configs")
    block.shared_experts.d_model = d_model
    block.shared_experts.hidden_size = expert_hidden_size
    block.routed_experts.d_model = d_model
    block.routed_experts.hidden_size = expert_hidden_size
    block.routed_experts_router.d_model = d_model
    if block.routed_experts_router.top_k != TOP_K:
        raise ValueError(
            f"expected routed top_k={TOP_K}, found {block.routed_experts_router.top_k}"
        )


def _dense_first_block(
    block: OLMoDDPTransformerBlockConfig,
    expert_hidden_size: int,
) -> OLMoDDPTransformerBlockConfig:
    dense = deepcopy(block)
    assert dense.shared_experts is not None
    dense.shared_experts.hidden_size = (TOP_K + 1) * expert_hidden_size
    dense.routed_experts = None
    dense.routed_experts_router = None
    dense.ep = None
    dense.rowwise_fp8 = None
    return dense


def _full_attention_block(
    block: OLMoDDPTransformerBlockConfig,
    attention_template: AttentionConfig,
    geometry: Geometry,
    *,
    rope: bool,
    attention_gate: bool,
) -> OLMoDDPTransformerBlockConfig:
    full = deepcopy(block)
    attention = deepcopy(attention_template)
    attention.n_heads = geometry.n_heads
    attention.n_kv_heads = geometry.n_kv_heads
    attention.head_dim = HEAD_DIM
    attention.sliding_window = None
    attention.gate = (
        GateConfig(
            granularity=GateGranularity.elementwise,
            full_precision=True,
        )
        if attention_gate
        else None
    )
    if not rope:
        attention.rope = None
    full.sequence_mixer = attention
    return full


def _build_geometry_matched_scale_model_config(
    model_size: str,
    *,
    rope: bool,
    attention_gate: bool,
) -> OLMoDDPModelConfig:
    """Build one geometry-matched model before cross-profile parity checks."""

    geometry = _geometry(model_size)
    if model_size == "275m":
        return build_geometry_matched_275m_model_config(
            (
                "geometry_rope_gated"
                if rope and attention_gate
                else "geometry_only"
                if rope
                else ("geometry_nope_gated" if attention_gate else "geometry_nope")
            )
        )

    parent = build_hybrid_model_config(model_size)
    candidate = deepcopy(parent)
    candidate.d_model = geometry.d_model
    candidate.n_layers = geometry.n_layers
    candidate.embed_scale = math.sqrt(geometry.d_model)

    moe_block = deepcopy(
        next(
            block
            for block in parent.resolved_block_configs
            if isinstance(block.sequence_mixer, GatedDeltaNetConfig)
        )
    )
    _resize_moe_block(
        moe_block,
        d_model=geometry.d_model,
        expert_hidden_size=geometry.expert_hidden_size,
    )
    gdn = cast(GatedDeltaNetConfig, moe_block.sequence_mixer)
    gdn.n_heads = geometry.n_heads
    gdn.n_v_heads = geometry.n_heads
    gdn.head_dim = HEAD_DIM
    gdn.expand_v = GDN_EXPAND_V
    candidate.block = moe_block

    attention_template = cast(
        AttentionConfig,
        next(
            block.sequence_mixer
            for block in parent.resolved_block_configs
            if isinstance(block.sequence_mixer, AttentionConfig)
            and block.sequence_mixer.sliding_window is None
        ),
    )
    overrides: dict[int, OLMoDDPTransformerBlockConfig] = {
        0: _dense_first_block(moe_block, geometry.expert_hidden_size),
    }
    for layer_idx in geometry.full_attention_layers:
        overrides[layer_idx] = _full_attention_block(
            moe_block,
            attention_template,
            geometry,
            rope=rope,
            attention_gate=attention_gate,
        )
    candidate.block_overrides = overrides
    candidate.validate()

    resolved = candidate.resolved_block_configs
    actual_gdn_layers = tuple(
        i
        for i, block in enumerate(resolved)
        if isinstance(block.sequence_mixer, GatedDeltaNetConfig)
    )
    actual_full_attention_layers = tuple(
        i for i, block in enumerate(resolved) if isinstance(block.sequence_mixer, AttentionConfig)
    )
    if actual_gdn_layers != geometry.gdn_layers:
        raise ValueError(
            f"unexpected {model_size} GDN layers: expected {geometry.gdn_layers}, "
            f"found {actual_gdn_layers}"
        )
    if actual_full_attention_layers != geometry.full_attention_layers:
        raise ValueError(
            f"unexpected {model_size} full-attention layers: expected "
            f"{geometry.full_attention_layers}, found {actual_full_attention_layers}"
        )
    if resolved[0].routed_experts is not None:
        raise ValueError("layer 0 must retain the dense-first FFN design")
    if candidate.init_std != 0.01:
        raise ValueError("geometry family must retain init_std=0.01")
    for layer_idx in geometry.gdn_layers:
        layer_gdn = cast(GatedDeltaNetConfig, resolved[layer_idx].sequence_mixer)
        if (
            layer_gdn.n_heads != geometry.n_heads
            or layer_gdn.n_v_heads != geometry.n_heads
            or layer_gdn.head_dim != HEAD_DIM
            or layer_gdn.expand_v != GDN_EXPAND_V
        ):
            raise ValueError(f"unexpected GDN shape at {model_size} layer {layer_idx}")
    for layer_idx in geometry.full_attention_layers:
        attention = cast(AttentionConfig, resolved[layer_idx].sequence_mixer)
        if (
            attention.n_heads != geometry.n_heads
            or attention.n_kv_heads != geometry.n_kv_heads
            or attention.head_dim != HEAD_DIM
            or (attention.rope is None) != (not rope)
        ):
            raise ValueError(f"unexpected full-attention shape at {model_size} layer {layer_idx}")
        if attention_gate:
            if (
                attention.gate is None
                or attention.gate.granularity != GateGranularity.elementwise
                or not attention.gate.full_precision
            ):
                raise ValueError(
                    f"unexpected attention gate at {model_size} layer {layer_idx}"
                )
        elif attention.gate is not None:
            raise ValueError(f"{model_size} layer {layer_idx} must remain ungated")

    gate_params = (
        geometry.d_model
        * geometry.n_heads
        * HEAD_DIM
        * len(geometry.full_attention_layers)
        if attention_gate
        else 0
    )
    expected_counts = (
        geometry.expected_active_params + gate_params,
        geometry.expected_active_non_embedding_params + gate_params,
        geometry.expected_total_params + gate_params,
    )
    actual_counts = (
        candidate.num_active_params,
        candidate.num_active_non_embedding_params,
        candidate.num_params,
    )
    if actual_counts != expected_counts:
        raise ValueError(
            f"unexpected {model_size} parameter counts: expected {expected_counts}, "
            f"found {actual_counts}"
        )
    return candidate


def _assert_nope_only_changes_rope(
    model_size: str,
    rope_model: OLMoDDPModelConfig,
    nope_model: OLMoDDPModelConfig,
) -> None:
    rope_counts = (
        rope_model.num_active_params,
        rope_model.num_active_non_embedding_params,
        rope_model.num_params,
    )
    nope_counts = (
        nope_model.num_active_params,
        nope_model.num_active_non_embedding_params,
        nope_model.num_params,
    )
    if nope_counts != rope_counts:
        raise ValueError(
            f"{model_size} NoPE changed parameter counts: RoPE={rope_counts}, NoPE={nope_counts}"
        )

    normalized_nope = deepcopy(nope_model)
    assert normalized_nope.block_overrides is not None
    assert rope_model.block_overrides is not None
    for layer_idx in _geometry(model_size).full_attention_layers:
        nope_attention = cast(
            AttentionConfig,
            normalized_nope.block_overrides[layer_idx].sequence_mixer,
        )
        rope_attention = cast(
            AttentionConfig,
            rope_model.block_overrides[layer_idx].sequence_mixer,
        )
        if nope_attention.rope is not None or rope_attention.rope is None:
            raise ValueError(f"{model_size} layer {layer_idx} did not form a RoPE/NoPE pair")
        nope_attention.rope = deepcopy(rope_attention.rope)

    if normalized_nope.as_dict() != rope_model.as_dict():
        raise ValueError(f"{model_size} NoPE changed fields other than full-attention RoPE")


def build_geometry_matched_scale_model_config(
    model_size: str,
    *,
    rope: bool = True,
    attention_gate: bool = False,
) -> OLMoDDPModelConfig:
    """Build a strictly audited geometry-matched model profile."""

    candidate = _build_geometry_matched_scale_model_config(
        model_size,
        rope=rope,
        attention_gate=attention_gate,
    )
    if not rope and not attention_gate:
        rope_model = _build_geometry_matched_scale_model_config(
            model_size,
            rope=True,
            attention_gate=False,
        )
        _assert_nope_only_changes_rope(model_size, rope_model, candidate)
    elif attention_gate:
        ungated = _build_geometry_matched_scale_model_config(
            model_size,
            rope=rope,
            attention_gate=False,
        )
        normalized = deepcopy(candidate)
        assert normalized.block_overrides is not None
        for layer_idx in _geometry(model_size).full_attention_layers:
            attention = cast(
                AttentionConfig,
                normalized.block_overrides[layer_idx].sequence_mixer,
            )
            if (
                attention.gate is None
                or attention.gate.granularity != GateGranularity.elementwise
                or not attention.gate.full_precision
            ):
                raise ValueError(
                    f"{model_size} layer {layer_idx} does not have the expected gate"
                )
            attention.gate = None
        if normalized.as_dict() != ungated.as_dict():
            raise ValueError(
                f"{model_size} gated profile changed fields other than attention.gate"
            )
        if rope:
            gated_nope = _build_geometry_matched_scale_model_config(
                model_size,
                rope=False,
                attention_gate=True,
            )
            _assert_nope_only_changes_rope(model_size, candidate, gated_nope)
    return candidate


def build_geometry_matched_scale_gdn2_model_config(
    model_size: str,
    *,
    rope: bool = False,
    attention_gate: bool = True,
    expand_v: float = 2.0,
    allow_neg_eigval: bool = True,
    disable_recompute: bool = False,
) -> OLMoDDPModelConfig:
    """Replace only GDN1 with GDN2 in a geometry-matched scale model."""

    if expand_v not in EXPECTED_GDN2_GATED_NOPE_COUNTS_BY_EXPAND_V:
        raise ValueError(
            f"unsupported audited GDN2 expand_v={expand_v}; expected one of "
            f"{tuple(EXPECTED_GDN2_GATED_NOPE_COUNTS_BY_EXPAND_V)}"
        )

    parent = build_geometry_matched_scale_model_config(
        model_size,
        rope=rope,
        attention_gate=attention_gate,
    )
    geometry = _geometry(model_size)
    resolved = parent.resolved_block_configs
    old_gdn = cast(GatedDeltaNetConfig, resolved[geometry.gdn_layers[0]].sequence_mixer)
    gdn2 = GatedDeltaNet2Config(
        n_heads=old_gdn.n_heads,
        n_v_heads=old_gdn.n_v_heads,
        head_dim=old_gdn.head_dim,
        expand_v=expand_v,
        allow_neg_eigval=allow_neg_eigval,
        conv_size=old_gdn.conv_size,
        conv_bias=old_gdn.conv_bias,
        disable_recompute=disable_recompute,
        norm_eps=old_gdn.norm_eps,
        dtype=old_gdn.dtype,
    )

    candidate = deepcopy(parent)
    default_block = deepcopy(resolved[next(i for i in geometry.gdn_layers if i != 0)])
    default_block.sequence_mixer = deepcopy(gdn2)
    dense_first = deepcopy(resolved[0])
    dense_first.sequence_mixer = deepcopy(gdn2)
    candidate.block = default_block
    candidate.block_overrides = {
        0: dense_first,
        **{
            layer_idx: deepcopy(resolved[layer_idx])
            for layer_idx in geometry.full_attention_layers
        },
    }
    candidate.validate()

    candidate_resolved = candidate.resolved_block_configs
    actual_gdn2_layers = tuple(
        layer_idx
        for layer_idx, block in enumerate(candidate_resolved)
        if isinstance(block.sequence_mixer, GatedDeltaNet2Config)
    )
    actual_full_attention_layers = tuple(
        layer_idx
        for layer_idx, block in enumerate(candidate_resolved)
        if isinstance(block.sequence_mixer, AttentionConfig)
    )
    if actual_gdn2_layers != geometry.gdn_layers:
        raise ValueError(
            f"unexpected {model_size} GDN2 layers: expected {geometry.gdn_layers}, "
            f"found {actual_gdn2_layers}"
        )
    if actual_full_attention_layers != geometry.full_attention_layers:
        raise ValueError(
            f"unexpected {model_size} full-attention layers: expected "
            f"{geometry.full_attention_layers}, found {actual_full_attention_layers}"
        )
    if candidate_resolved[0].routed_experts is not None:
        raise ValueError("GDN2 scale model must retain the dense-first layer-0 FFN")
    for layer_idx in geometry.gdn_layers:
        layer_gdn2 = cast(GatedDeltaNet2Config, candidate_resolved[layer_idx].sequence_mixer)
        if (
            layer_gdn2.expand_v != expand_v
            or layer_gdn2.allow_neg_eigval != allow_neg_eigval
        ):
            raise ValueError(
                f"unexpected GDN2 stability settings at {model_size} layer {layer_idx}"
            )

    # Normalize GDN2 back to GDN1 and demand exact config equality. This proves
    # that width/depth, MoE, attention, positional encoding, norms, and init did
    # not change as a side effect of the mixer swap.
    normalized = deepcopy(candidate)
    normalized_default = deepcopy(default_block)
    normalized_default.sequence_mixer = deepcopy(old_gdn)
    normalized_dense = deepcopy(dense_first)
    normalized_dense.sequence_mixer = deepcopy(old_gdn)
    normalized.block = normalized_default
    normalized.block_overrides = {
        0: normalized_dense,
        **{
            layer_idx: deepcopy(resolved[layer_idx])
            for layer_idx in geometry.full_attention_layers
        },
    }
    if normalized.as_dict() != parent.as_dict():
        raise ValueError(f"{model_size} GDN2 conversion changed fields beyond the mixer")

    actual_counts = (
        candidate.num_active_params,
        candidate.num_active_non_embedding_params,
        candidate.num_params,
    )
    if not rope and attention_gate:
        expected_counts = EXPECTED_GDN2_GATED_NOPE_COUNTS_BY_EXPAND_V[expand_v][model_size]
        if actual_counts != expected_counts:
            raise ValueError(
                f"unexpected {model_size} gated-NoPE GDN2 expand_v={expand_v:g} "
                f"counts: expected "
                f"{expected_counts}, found {actual_counts}"
            )
    else:
        parent_counts = (
            parent.num_active_params,
            parent.num_active_non_embedding_params,
            parent.num_params,
        )
        expected_delta = (
            gdn2.num_params(geometry.d_model) - old_gdn.num_params(geometry.d_model)
        ) * len(geometry.gdn_layers)
        if any(
            actual != base + expected_delta
            for actual, base in zip(actual_counts, parent_counts, strict=True)
        ):
            raise ValueError(
                f"unexpected {model_size} GDN2 parameter delta: expected "
                f"{expected_delta:,}, parent={parent_counts}, found={actual_counts}"
            )
    return candidate


def build_geometry_matched_scale_kda_model_config(
    model_size: str,
) -> OLMoDDPModelConfig:
    """Replace only GDN1 with canonical KDA in the gated-NoPE scale geometry."""

    if model_size == "275m":
        from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_275m import (
            build_geometry_matched_kda_model_config,
        )

        return build_geometry_matched_kda_model_config()

    parent = build_geometry_matched_scale_model_config(
        model_size,
        rope=False,
        attention_gate=True,
    )
    geometry = _geometry(model_size)
    resolved = parent.resolved_block_configs
    old_gdn = cast(GatedDeltaNetConfig, resolved[geometry.gdn_layers[0]].sequence_mixer)
    kda = KimiDeltaAttentionConfig(
        n_heads=old_gdn.n_heads,
        n_v_heads=old_gdn.n_heads,
        head_dim=old_gdn.head_dim,
        expand_v=1.0,
        allow_neg_eigval=False,
        conv_size=old_gdn.conv_size,
        conv_bias=old_gdn.conv_bias,
        norm_eps=old_gdn.norm_eps,
        dtype=old_gdn.dtype,
    )

    candidate = deepcopy(parent)
    default_idx = next(i for i in geometry.gdn_layers if i != 0)
    default_block = deepcopy(resolved[default_idx])
    default_block.sequence_mixer = deepcopy(kda)
    dense_first = deepcopy(resolved[0])
    dense_first.sequence_mixer = deepcopy(kda)
    candidate.block = default_block
    candidate.block_overrides = {
        0: dense_first,
        **{
            layer_idx: deepcopy(resolved[layer_idx])
            for layer_idx in geometry.full_attention_layers
        },
    }
    candidate.validate()

    candidate_resolved = candidate.resolved_block_configs
    actual_kda_layers = tuple(
        layer_idx
        for layer_idx, block in enumerate(candidate_resolved)
        if isinstance(block.sequence_mixer, KimiDeltaAttentionConfig)
    )
    actual_full_attention_layers = tuple(
        layer_idx
        for layer_idx, block in enumerate(candidate_resolved)
        if isinstance(block.sequence_mixer, AttentionConfig)
    )
    if actual_kda_layers != geometry.gdn_layers:
        raise ValueError(
            f"unexpected {model_size} KDA layers: expected {geometry.gdn_layers}, "
            f"found {actual_kda_layers}"
        )
    if actual_full_attention_layers != geometry.full_attention_layers:
        raise ValueError(
            f"unexpected {model_size} full-attention layers: expected "
            f"{geometry.full_attention_layers}, found {actual_full_attention_layers}"
        )
    if candidate_resolved[0].routed_experts is not None:
        raise ValueError("KDA scale model must retain the dense-first layer-0 FFN")
    for layer_idx in geometry.gdn_layers:
        layer_kda = cast(
            KimiDeltaAttentionConfig,
            candidate_resolved[layer_idx].sequence_mixer,
        )
        if (
            layer_kda.n_heads != geometry.n_heads
            or layer_kda.n_v_heads != geometry.n_heads
            or layer_kda.head_dim != HEAD_DIM
            or layer_kda.expand_v != 1.0
            or layer_kda.allow_neg_eigval
        ):
            raise ValueError(f"unexpected canonical KDA settings at {model_size} layer {layer_idx}")

    # Normalize KDA back to GDN1 and require byte-for-byte config equality.
    # This catches any accidental change to geometry, MoE, full attention,
    # positional encoding, gating, norms, or initialization.
    normalized = deepcopy(candidate)
    normalized_default = deepcopy(default_block)
    normalized_default.sequence_mixer = deepcopy(old_gdn)
    normalized_dense = deepcopy(dense_first)
    normalized_dense.sequence_mixer = deepcopy(old_gdn)
    normalized.block = normalized_default
    normalized.block_overrides = {
        0: normalized_dense,
        **{
            layer_idx: deepcopy(resolved[layer_idx])
            for layer_idx in geometry.full_attention_layers
        },
    }
    if normalized.as_dict() != parent.as_dict():
        raise ValueError(f"{model_size} KDA conversion changed fields beyond the mixer")

    parent_counts = (
        parent.num_active_params,
        parent.num_active_non_embedding_params,
        parent.num_params,
    )
    actual_counts = (
        candidate.num_active_params,
        candidate.num_active_non_embedding_params,
        candidate.num_params,
    )
    expected_delta = (
        kda.num_params(geometry.d_model) - old_gdn.num_params(geometry.d_model)
    ) * len(geometry.gdn_layers)
    if any(
        actual != base + expected_delta
        for actual, base in zip(actual_counts, parent_counts, strict=True)
    ):
        raise ValueError(
            f"unexpected {model_size} KDA parameter delta: expected {expected_delta:,}, "
            f"parent={parent_counts}, found={actual_counts}"
        )
    return candidate


def parameter_summary(
    model_size: str,
    *,
    rope: bool = True,
    attention_gate: bool = False,
) -> dict[str, Any]:
    geometry = _geometry(model_size)
    parent = build_hybrid_model_config(model_size)
    candidate = build_geometry_matched_scale_model_config(
        model_size,
        rope=rope,
        attention_gate=attention_gate,
    )
    return {
        "model_size": model_size,
        "dense_geometry_rung": geometry.dense_rung,
        "d_model": geometry.d_model,
        "n_layers": geometry.n_layers,
        "gdn_layers": list(geometry.gdn_layers),
        "full_attention_layers": list(geometry.full_attention_layers),
        "dense_ffn_layers": [0],
        "n_heads": geometry.n_heads,
        "n_kv_heads": geometry.n_kv_heads,
        "head_dim": HEAD_DIM,
        "expert_hidden_size": geometry.expert_hidden_size,
        "dense_first_hidden_size": (TOP_K + 1) * geometry.expert_hidden_size,
        "num_experts": 256,
        "top_k": TOP_K,
        "shared_experts": 1,
        "gdn_expand_v": GDN_EXPAND_V,
        "rope": rope,
        "attention_gate": attention_gate,
        "init_std": candidate.init_std,
        "active_params": candidate.num_active_params,
        "active_non_embedding_params": candidate.num_active_non_embedding_params,
        "total_params": candidate.num_params,
        "delta_vs_current_hybrid_active": candidate.num_active_params - parent.num_active_params,
        "delta_fraction_vs_current_hybrid_active": (
            candidate.num_active_params / parent.num_active_params - 1
        ),
        "delta_vs_current_hybrid_active_non_embedding": (
            candidate.num_active_non_embedding_params - parent.num_active_non_embedding_params
        ),
        "delta_fraction_vs_current_hybrid_active_non_embedding": (
            candidate.num_active_non_embedding_params / parent.num_active_non_embedding_params - 1
        ),
        "dense_reference_active_params": geometry.dense_reference_active_params,
        "dense_reference_active_non_embedding_params": (
            geometry.dense_reference_active_non_embedding_params
        ),
    }


def gdn2_parameter_summary(model_size: str) -> dict[str, Any]:
    """Return the audited gated-NoPE GDN2 scale configuration."""

    geometry = _geometry(model_size)
    parent = build_geometry_matched_scale_model_config(
        model_size,
        rope=False,
        attention_gate=True,
    )
    candidate = build_geometry_matched_scale_gdn2_model_config(model_size)
    return {
        "model_size": model_size,
        "dense_geometry_rung": geometry.dense_rung,
        "d_model": geometry.d_model,
        "n_layers": geometry.n_layers,
        "gdn2_layers": list(geometry.gdn_layers),
        "full_attention_layers": list(geometry.full_attention_layers),
        "dense_ffn_layers": [0],
        "n_heads": geometry.n_heads,
        "n_kv_heads": geometry.n_kv_heads,
        "head_dim": HEAD_DIM,
        "expert_hidden_size": geometry.expert_hidden_size,
        "dense_first_hidden_size": (TOP_K + 1) * geometry.expert_hidden_size,
        "num_experts": 256,
        "top_k": TOP_K,
        "shared_experts": 1,
        "gdn_expand_v": GDN_EXPAND_V,
        "rope": False,
        "attention_gate": True,
        "attention_gate_granularity": "elementwise",
        "attention_gate_full_precision": True,
        "gdn2_disable_recompute": False,
        "init_std": candidate.init_std,
        "active_params": candidate.num_active_params,
        "active_non_embedding_params": candidate.num_active_non_embedding_params,
        "total_params": candidate.num_params,
        "delta_vs_gdn1_active": candidate.num_active_params - parent.num_active_params,
        "delta_vs_gdn1_active_non_embedding": (
            candidate.num_active_non_embedding_params
            - parent.num_active_non_embedding_params
        ),
        "delta_vs_gdn1_total": candidate.num_params - parent.num_params,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=MODEL_SIZES, action="append")
    parser.add_argument(
        "--nope",
        action="store_true",
        help="Build the strictly matched NoPE profile instead of the RoPE profile",
    )
    parser.add_argument(
        "--attention-gate",
        action="store_true",
        help="Add the isolated elementwise attention gate to the selected RoPE/NoPE profile",
    )
    parser.add_argument(
        "--gdn2",
        action="store_true",
        help="Audit the gated-NoPE GDN2 swap at the selected sizes",
    )
    args = parser.parse_args()
    model_sizes = args.model_size or list(MODEL_SIZES)
    print(
        json.dumps(
            [
                (
                    gdn2_parameter_summary(size)
                    if args.gdn2
                    else parameter_summary(
                        size,
                        rope=not args.nope,
                        attention_gate=args.attention_gate,
                    )
                )
                for size in model_sizes
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
