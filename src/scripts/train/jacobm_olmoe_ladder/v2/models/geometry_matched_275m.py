#!/usr/bin/env python3
"""Design candidate 275M MoE hybrids on the dense ladder's model geometry.

The primary ``geometry_only`` profile isolates width, depth, mixer placement,
head geometry, and GDN value expansion from the already-tested hybrid. It
deliberately retains our current GQA ratio and ungated RoPE full-attention
blocks. ``geometry_nope`` changes only those two full-attention blocks from
RoPE to NoPE. ``geometry_nope_gated`` adds the dense ladder's elementwise
attention gate while retaining our 8-Q/4-KV attention shape, so gating can be
tested without confounding it with the KV-head ratio. ``geometry_rope_gated``
is the corresponding interaction control: it restores RoPE while changing
nothing else about the gated NoPE profile. The ``dense_attention`` profile
additionally matches the dense 275M rung's KV-head count and elementwise
attention gate, while deliberately retaining RoPE and the current initialization
for separate interventions.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, cast

from olmo_core.nn.attention import (
    AttentionConfig,
    GateConfig,
    GatedDeltaNet2Config,
    GateGranularity,
    KimiDeltaAttentionConfig,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.ddp import OLMoDDPTransformerBlockConfig
from olmo_core.nn.moe import LatentMoEConfig
from olmo_core.nn.transformer import OLMoDDPModelConfig
from scripts.train.jacobm_olmoe_ladder.v2.models.hybrid_wide import (
    build_hybrid_model_config,
)

D_MODEL = 640
N_LAYERS = 10
N_HEADS = 8
HEAD_DIM = 128
FULL_ATTENTION_LAYERS = (4, 9)
GDN_LAYERS = tuple(i for i in range(N_LAYERS) if i not in FULL_ATTENTION_LAYERS)
NUM_EXPERTS = 256
TOP_K = 8
SWA_WINDOW_SIZE = 2_048

# Checked-in dense-ladder 275M reference values.
DENSE_REFERENCE_ACTIVE_PARAMS = 275_493_760
DENSE_REFERENCE_ACTIVE_NON_EMBEDDING_PARAMS = 211_268_480


@dataclass(frozen=True)
class Profile:
    expert_hidden_size: int
    n_kv_heads: int
    attention_gate: bool
    gdn_expand_v: float
    rope: bool


PROFILES = {
    # Keep the already-audited geometry candidate unchanged except for adopting
    # the dense hybrid's expand_v=2. This lands within 1% of the current 288M
    # hybrid's active-parameter count without retuning the MoE widths.
    "geometry_only": Profile(
        expert_hidden_size=664,
        n_kv_heads=4,
        attention_gate=False,
        gdn_expand_v=2.0,
        rope=True,
    ),
    # Change only positional encoding in the two full-attention layers.
    "geometry_nope": Profile(
        expert_hidden_size=664,
        n_kv_heads=4,
        attention_gate=False,
        gdn_expand_v=2.0,
        rope=False,
    ),
    # Add only the dense ladder's elementwise, full-precision sigmoid gate to
    # the NoPE profile. Keep the existing MoE widths and 8-Q/4-KV GQA shape so
    # this is a clean attention-gating intervention.
    "geometry_nope_gated": Profile(
        expert_hidden_size=664,
        n_kv_heads=4,
        attention_gate=True,
        gdn_expand_v=2.0,
        rope=False,
    ),
    # Restore RoPE on the gated profile without changing attention head
    # geometry, MoE widths, initialization, or any optimization setting.
    "geometry_rope_gated": Profile(
        expert_hidden_size=664,
        n_kv_heads=4,
        attention_gate=True,
        gdn_expand_v=2.0,
        rope=True,
    ),
    # Also match the dense 275M full-attention shape (8 Q / 8 KV) and its
    # elementwise gate. 648 gives a near-exact active match with 8-wide tensor
    # alignment; odd hidden widths get closer numerically but are poor
    # production shapes.
    "dense_attention": Profile(
        expert_hidden_size=648,
        n_kv_heads=8,
        attention_gate=True,
        gdn_expand_v=2.0,
        rope=True,
    ),
}

EXPECTED_PROFILE_COUNTS = {
    "geometry_only": (290_782_080, 226_556_800, 3_136_314_240),
    "geometry_nope": (290_782_080, 226_556_800, 3_136_314_240),
    "geometry_nope_gated": (292_092_800, 227_867_520, 3_137_624_960),
    "geometry_rope_gated": (292_092_800, 227_867_520, 3_137_624_960),
    "dense_attention": (290_638_720, 226_413_440, 3_067_603_840),
}
EXPECTED_SWA_COUNTS = (265_665_280, 201_440_000, 3_111_197_440)
EXPECTED_GDN2_COUNTS_BY_EXPAND_V = {
    1.0: (284_915_520, 220_690_240, 3_130_447_680),
    2.0: (306_191_168, 241_965_888, 3_151_723_328),
}
EXPECTED_KDA_COUNTS_BY_SETTINGS = {
    (1.0, False, 664): (274_470_720, 210_245_440, 3_120_002_880),
    (2.0, True, 664): (290_503_488, 226_278_208, 3_136_035_648),
    # First production-shaped MXFP8 candidate. Changing only the expert width
    # from 664 to 672 also changes the dense-first FFN from 5,976 to 6,048.
    (2.0, True, 672): (291_885_888, 227_660_608, 3_171_701_568),
}

# Experimental six-layer mixer motif:
#
#   KDA -> SWA -> KDA -> SWA -> KDA -> full attention
#
# Repeating it twice gives an exact 1:1 split between KDA and ordinary
# attention while keeping a 2:1 split between local and global ordinary
# attention. Twelve layers is the smallest depth near the current ten-layer
# KDA parent that can contain whole motifs. Narrowing only the expert hidden
# dimension to 552 keeps the active size within 0.14% of that parent.
KDA_MIXED6_N_LAYERS = 12
KDA_MIXED6_EXPERT_HIDDEN_SIZE = 552
KDA_MIXED6_KDA_LAYERS = (0, 2, 4, 6, 8, 10)
KDA_MIXED6_SWA_LAYERS = (1, 3, 7, 9)
KDA_MIXED6_FULL_ATTENTION_LAYERS = (5, 11)
EXPECTED_KDA_MIXED6_COUNTS = (290_904_368, 226_679_088, 3_182_147_888)
EXPECTED_KDA_MIXED6_LATENT2X_COUNTS = (297_212_208, 232_986_928, 3_188_455_728)

# LatentMoE qualification starts with the two compression ratios discussed for
# the 275M rung: a conservative 2x projection and the paper's default 4x
# projection. Expert count, top-k, hidden width, and the surrounding KDA hybrid
# are deliberately held fixed so this is an isolated latent-width experiment.
LATENT_MOE_ROUTED_DIMS_BY_COMPRESSION = {
    2: 320,
    4: 160,
}
EXPECTED_LATENT_MOE_COUNTS = {
    (2, False, None): (248_294_208, 184_068_928, 1_671_060_288),
    (4, False, None): (223_503_168, 159_277_888, 934_886_208),
    (2, True, None): (295_664_448, 231_439_168, 3_141_196_608),
    (4, True, None): (296_770_368, 232_545_088, 3_142_302_528),
    # Keep paper-matched top-32 routing but stay below the grouped-MM
    # implementation's strict group_count < 1024 limit on EP1.
    (4, True, 1000): (296_632_128, 232_406_848, 3_073_320_768),
}

# A strict 1:1 hybrid alternates a recurrent/local mixer at even layers with
# gated global attention at odd layers. Layer count is the only sizing knob:
# all widths, heads, MoE dimensions, initialization, and mixer settings remain
# identical to the existing gated-RoPE systems comparison. Ten layers is the
# closest even-depth match for GDN1/GDN2; twelve layers is the closest for SWA.
ONE_TO_ONE_N_LAYERS = {
    "gdn1": 10,
    "gdn2": 10,
    "swa": 12,
}
EXPECTED_ONE_TO_ONE_COUNTS = {
    "gdn1": (284_148_560, 219_923_280, 3_129_680_720),
    "gdn2": (292_960_040, 228_734_760, 3_138_492_200),
    "swa": (295_500_032, 231_274_752, 3_773_372_672),
    "swa_10l": (267_631_360, 203_406_080, 3_113_163_520),
}


def _resize_moe_block(block: OLMoDDPTransformerBlockConfig, expert_hidden_size: int) -> None:
    if block.shared_experts is None:
        raise ValueError("expected a shared expert config")
    if block.routed_experts is None or block.routed_experts_router is None:
        raise ValueError("expected routed expert and router configs")
    block.shared_experts.d_model = D_MODEL
    block.shared_experts.hidden_size = expert_hidden_size
    block.routed_experts.d_model = D_MODEL
    block.routed_experts.hidden_size = expert_hidden_size
    block.routed_experts_router.d_model = D_MODEL
    if block.routed_experts_router.top_k != TOP_K:
        raise ValueError(f"expected top_k={TOP_K}, found {block.routed_experts_router.top_k}")


def _dense_first_block(
    block: OLMoDDPTransformerBlockConfig,
    expert_hidden_size: int,
) -> OLMoDDPTransformerBlockConfig:
    dense = deepcopy(block)
    assert dense.shared_experts is not None
    # Match the active FFN width of top-k routed experts plus one shared expert.
    dense.shared_experts.hidden_size = (TOP_K + 1) * expert_hidden_size
    dense.routed_experts = None
    dense.routed_experts_router = None
    dense.ep = None
    dense.rowwise_fp8 = None
    return dense


def _full_attention_block(
    block: OLMoDDPTransformerBlockConfig,
    attention_template: AttentionConfig,
    profile: Profile,
) -> OLMoDDPTransformerBlockConfig:
    full = deepcopy(block)
    attention = deepcopy(attention_template)
    attention.n_heads = N_HEADS
    attention.n_kv_heads = profile.n_kv_heads
    attention.head_dim = HEAD_DIM
    attention.sliding_window = None
    attention.gate = (
        GateConfig(
            granularity=GateGranularity.elementwise,
            full_precision=True,
        )
        if profile.attention_gate
        else None
    )
    if not profile.rope:
        attention.rope = None
    full.sequence_mixer = attention
    return full


def build_geometry_matched_model_config(
    profile_name: str = "geometry_only",
) -> OLMoDDPModelConfig:
    """Build one geometry-matched candidate while retaining RoPE and initialization."""

    try:
        profile = PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown profile {profile_name!r}; choose one of {tuple(PROFILES)}"
        ) from exc

    parent = build_hybrid_model_config("275m")
    candidate = deepcopy(parent)
    candidate.d_model = D_MODEL
    candidate.n_layers = N_LAYERS
    candidate.embed_scale = math.sqrt(D_MODEL)

    # Use the tested GDN block as the base for all layers, then adopt the
    # profile's explicitly audited value expansion.
    moe_block = deepcopy(parent.resolved_block_configs[2])
    _resize_moe_block(moe_block, profile.expert_hidden_size)
    gdn = cast(GatedDeltaNetConfig, moe_block.sequence_mixer)
    gdn.n_heads = N_HEADS
    gdn.n_v_heads = N_HEADS
    gdn.head_dim = HEAD_DIM
    gdn.expand_v = profile.gdn_expand_v
    candidate.block = moe_block

    attention_template = cast(AttentionConfig, parent.resolved_block_configs[0].sequence_mixer)
    overrides: dict[int, OLMoDDPTransformerBlockConfig] = {
        0: _dense_first_block(moe_block, profile.expert_hidden_size),
    }
    for layer_idx in FULL_ATTENTION_LAYERS:
        overrides[layer_idx] = _full_attention_block(moe_block, attention_template, profile)
    candidate.block_overrides = overrides
    candidate.validate()

    resolved = candidate.resolved_block_configs
    actual_gdn = tuple(
        i
        for i, block in enumerate(resolved)
        if isinstance(block.sequence_mixer, GatedDeltaNetConfig)
    )
    actual_full = tuple(
        i for i, block in enumerate(resolved) if isinstance(block.sequence_mixer, AttentionConfig)
    )
    if actual_gdn != GDN_LAYERS or actual_full != FULL_ATTENTION_LAYERS:
        raise ValueError(f"unexpected mixer pattern: GDN={actual_gdn}, full={actual_full}")
    if resolved[0].routed_experts is not None:
        raise ValueError("layer 0 must retain the dense-first FFN design")
    if candidate.init_std != 0.01:
        raise ValueError("geometry candidate must retain init_std=0.01")
    if any(
        cast(GatedDeltaNetConfig, resolved[i].sequence_mixer).expand_v != profile.gdn_expand_v
        for i in GDN_LAYERS
    ):
        raise ValueError(f"geometry candidate must use expand_v={profile.gdn_expand_v:g}")
    if profile.rope:
        if any(cast(AttentionConfig, resolved[i].sequence_mixer).rope is None for i in actual_full):
            raise ValueError(f"{profile_name} must retain RoPE")
    elif any(
        cast(AttentionConfig, resolved[i].sequence_mixer).rope is not None for i in actual_full
    ):
        raise ValueError(f"{profile_name} must use NoPE")
    for layer_idx in actual_full:
        attention = cast(AttentionConfig, resolved[layer_idx].sequence_mixer)
        if attention.n_kv_heads != profile.n_kv_heads:
            raise ValueError(
                f"{profile_name} layer {layer_idx} must use {profile.n_kv_heads} KV heads"
            )
        if profile.attention_gate:
            if (
                attention.gate is None
                or attention.gate.granularity != GateGranularity.elementwise
                or not attention.gate.full_precision
            ):
                raise ValueError(
                    f"{profile_name} layer {layer_idx} must use the dense ladder's "
                    "full-precision elementwise attention gate"
                )
        elif attention.gate is not None:
            raise ValueError(f"{profile_name} layer {layer_idx} must remain ungated")
    actual_counts = (
        candidate.num_active_params,
        candidate.num_active_non_embedding_params,
        candidate.num_params,
    )
    if actual_counts != EXPECTED_PROFILE_COUNTS[profile_name]:
        raise ValueError(
            f"unexpected {profile_name} parameter counts: expected "
            f"{EXPECTED_PROFILE_COUNTS[profile_name]}, found {actual_counts}"
        )

    return candidate


def build_geometry_matched_gdn2_model_config(
    *,
    rope: bool = True,
    expand_v: float = 2.0,
    allow_neg_eigval: bool = True,
    disable_recompute: bool = False,
) -> OLMoDDPModelConfig:
    """Swap GDN1 for GDN2 in a 275M gated geometry candidate.

    All geometry, MoE, full-attention, initialization, and optimization-facing
    settings remain unchanged. ``rope=False`` selects the corresponding gated
    NoPE parent. The defaults retain GDN1's ``expand_v=2`` and
    ``allow_neg_eigval=True`` settings; explicit overrides support isolated
    stability ablations without changing any surrounding model geometry.
    """

    try:
        expected_counts = EXPECTED_GDN2_COUNTS_BY_EXPAND_V[expand_v]
    except KeyError as exc:
        raise ValueError(
            f"unsupported audited GDN2 expand_v={expand_v}; expected one of "
            f"{tuple(EXPECTED_GDN2_COUNTS_BY_EXPAND_V)}"
        ) from exc

    profile_name = "geometry_rope_gated" if rope else "geometry_nope_gated"
    candidate = build_geometry_matched_model_config(profile_name)
    resolved = candidate.resolved_block_configs
    old_gdn = cast(GatedDeltaNetConfig, resolved[1].sequence_mixer)
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

    default_block = deepcopy(resolved[1])
    default_block.sequence_mixer = deepcopy(gdn2)
    dense_first = deepcopy(resolved[0])
    dense_first.sequence_mixer = deepcopy(gdn2)
    candidate.block = default_block
    candidate.block_overrides = {
        0: dense_first,
        **{layer_idx: deepcopy(resolved[layer_idx]) for layer_idx in FULL_ATTENTION_LAYERS},
    }
    candidate.validate()

    actual_gdn2 = tuple(
        layer_idx
        for layer_idx, block in enumerate(candidate.resolved_block_configs)
        if isinstance(block.sequence_mixer, GatedDeltaNet2Config)
    )
    actual_full = tuple(
        layer_idx
        for layer_idx, block in enumerate(candidate.resolved_block_configs)
        if isinstance(block.sequence_mixer, AttentionConfig)
    )
    if actual_gdn2 != GDN_LAYERS or actual_full != FULL_ATTENTION_LAYERS:
        raise ValueError(f"unexpected mixer pattern: GDN2={actual_gdn2}, full={actual_full}")
    if candidate.resolved_block_configs[0].routed_experts is not None:
        raise ValueError("GDN2 candidate must retain the dense-first layer-0 FFN")
    if any(
        (
            cast(AttentionConfig, candidate.resolved_block_configs[layer_idx].sequence_mixer).rope
            is None
        )
        != (not rope)
        for layer_idx in FULL_ATTENTION_LAYERS
    ):
        raise ValueError(f"GDN2 {profile_name} positional encoding does not match rope={rope}")
    actual_counts = (
        candidate.num_active_params,
        candidate.num_active_non_embedding_params,
        candidate.num_params,
    )
    if actual_counts != expected_counts:
        raise ValueError(
            f"unexpected GDN2 parameter counts: expected {expected_counts}, found {actual_counts}"
        )
    return candidate


def build_geometry_matched_kda_model_config(
    *,
    expand_v: float = 1.0,
    allow_neg_eigval: bool = False,
    expert_hidden_size: int = 664,
) -> OLMoDDPModelConfig:
    """Replace GDN1 with KDA in the gated-NoPE 275M geometry.

    The defaults follow the released Kimi-Linear configuration. The audited
    ``expand_v=2`` / negative-eigenvalue profile matches the recurrent settings
    of the original geometry-matched GDN1 family. Full attention, MoE geometry,
    dense-first placement, and initialization are inherited unchanged.
    """

    settings = (float(expand_v), bool(allow_neg_eigval), int(expert_hidden_size))
    try:
        expected_counts = EXPECTED_KDA_COUNTS_BY_SETTINGS[settings]
    except KeyError as exc:
        raise ValueError(f"unsupported audited KDA settings: {settings}") from exc

    candidate = build_geometry_matched_model_config("geometry_nope_gated")
    resolved = candidate.resolved_block_configs
    old_gdn = cast(GatedDeltaNetConfig, resolved[1].sequence_mixer)
    kda = KimiDeltaAttentionConfig(
        n_heads=old_gdn.n_heads,
        n_v_heads=old_gdn.n_heads,
        head_dim=old_gdn.head_dim,
        expand_v=expand_v,
        allow_neg_eigval=allow_neg_eigval,
        conv_size=old_gdn.conv_size,
        conv_bias=old_gdn.conv_bias,
        norm_eps=old_gdn.norm_eps,
        dtype=old_gdn.dtype,
    )

    default_block = deepcopy(resolved[1])
    default_block.sequence_mixer = deepcopy(kda)
    _resize_moe_block(default_block, expert_hidden_size)
    dense_first = deepcopy(resolved[0])
    dense_first.sequence_mixer = deepcopy(kda)
    assert dense_first.shared_experts is not None
    dense_first.shared_experts.hidden_size = (TOP_K + 1) * expert_hidden_size
    full_attention_blocks = {
        layer_idx: deepcopy(resolved[layer_idx]) for layer_idx in FULL_ATTENTION_LAYERS
    }
    for block in full_attention_blocks.values():
        _resize_moe_block(block, expert_hidden_size)
    candidate.block = default_block
    candidate.block_overrides = {
        0: dense_first,
        **full_attention_blocks,
    }
    candidate.validate()

    actual_kda = tuple(
        layer_idx
        for layer_idx, block in enumerate(candidate.resolved_block_configs)
        if isinstance(block.sequence_mixer, KimiDeltaAttentionConfig)
    )
    actual_full = tuple(
        layer_idx
        for layer_idx, block in enumerate(candidate.resolved_block_configs)
        if isinstance(block.sequence_mixer, AttentionConfig)
    )
    if actual_kda != GDN_LAYERS or actual_full != FULL_ATTENTION_LAYERS:
        raise ValueError(f"unexpected mixer pattern: KDA={actual_kda}, full={actual_full}")
    if candidate.resolved_block_configs[0].routed_experts is not None:
        raise ValueError("KDA candidate must retain the dense-first layer-0 FFN")
    if (
        candidate.resolved_block_configs[0].shared_experts.hidden_size
        != (TOP_K + 1) * expert_hidden_size
    ):
        raise ValueError("KDA candidate has an unexpected dense-first FFN width")
    for layer_idx in range(1, candidate.n_layers):
        block = candidate.resolved_block_configs[layer_idx]
        if block.shared_experts is None or block.routed_experts is None:
            raise ValueError(f"KDA candidate layer {layer_idx} must retain MoE experts")
        if (
            block.shared_experts.hidden_size != expert_hidden_size
            or block.routed_experts.hidden_size != expert_hidden_size
        ):
            raise ValueError(f"KDA candidate layer {layer_idx} has an unexpected expert width")
    for layer_idx in FULL_ATTENTION_LAYERS:
        attention = cast(
            AttentionConfig, candidate.resolved_block_configs[layer_idx].sequence_mixer
        )
        if attention.rope is not None:
            raise ValueError(f"KDA candidate full-attention layer {layer_idx} must use NoPE")
        if (
            attention.gate is None
            or attention.gate.granularity != GateGranularity.elementwise
            or not attention.gate.full_precision
        ):
            raise ValueError(
                f"KDA candidate full-attention layer {layer_idx} must retain elementwise gating"
            )
    if any(
        cast(KimiDeltaAttentionConfig, candidate.resolved_block_configs[i].sequence_mixer).expand_v
        != expand_v
        or cast(
            KimiDeltaAttentionConfig, candidate.resolved_block_configs[i].sequence_mixer
        ).allow_neg_eigval
        != allow_neg_eigval
        for i in GDN_LAYERS
    ):
        raise ValueError(
            f"KDA candidate must use expand_v={expand_v:g} and allow_neg_eigval={allow_neg_eigval}"
        )
    actual_counts = (
        candidate.num_active_params,
        candidate.num_active_non_embedding_params,
        candidate.num_params,
    )
    if actual_counts != expected_counts:
        raise ValueError(
            f"unexpected KDA parameter counts: expected {expected_counts}, found {actual_counts}"
        )
    return candidate


def build_geometry_matched_kda_mixed6_model_config() -> OLMoDDPModelConfig:
    """Build the active-matched 3-KDA/2-SWA/1-full-attention motif.

    The current promoted KDA settings are retained: ``expand_v=2``, negative
    eigenvalues, NoPE, elementwise full-precision attention gates, 8-Q/4-KV
    GQA, a 2,048-token SWA window, and the dense-first FFN. The exact six-layer
    motif is repeated twice. Only depth and expert hidden size change to
    preserve the motif and recover the parent's active parameter count.
    """

    source = build_geometry_matched_kda_model_config(
        expand_v=2.0,
        allow_neg_eigval=True,
        expert_hidden_size=664,
    )
    source_resolved = source.resolved_block_configs

    candidate = deepcopy(source)
    candidate.n_layers = KDA_MIXED6_N_LAYERS

    kda_block = deepcopy(source_resolved[1])
    _resize_moe_block(kda_block, KDA_MIXED6_EXPERT_HIDDEN_SIZE)

    dense_first = deepcopy(source_resolved[0])
    assert dense_first.shared_experts is not None
    dense_first.shared_experts.hidden_size = (TOP_K + 1) * KDA_MIXED6_EXPERT_HIDDEN_SIZE

    full_attention = deepcopy(source_resolved[FULL_ATTENTION_LAYERS[0]])
    _resize_moe_block(full_attention, KDA_MIXED6_EXPERT_HIDDEN_SIZE)

    sliding_attention = deepcopy(full_attention)
    sliding_mixer = cast(AttentionConfig, sliding_attention.sequence_mixer)
    sliding_mixer.sliding_window = SlidingWindowAttentionConfig(
        pattern=[SWA_WINDOW_SIZE],
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=False,
    )

    candidate.block = kda_block
    candidate.block_overrides = {
        0: dense_first,
        **{layer_idx: deepcopy(sliding_attention) for layer_idx in KDA_MIXED6_SWA_LAYERS},
        **{layer_idx: deepcopy(full_attention) for layer_idx in KDA_MIXED6_FULL_ATTENTION_LAYERS},
    }
    candidate.validate()

    resolved = candidate.resolved_block_configs
    actual_kda: list[int] = []
    actual_swa: list[int] = []
    actual_full: list[int] = []
    for layer_idx, block in enumerate(resolved):
        mixer = block.sequence_mixer
        if isinstance(mixer, KimiDeltaAttentionConfig):
            actual_kda.append(layer_idx)
        elif isinstance(mixer, AttentionConfig):
            if mixer.sliding_window is not None and mixer.sliding_window.should_use_swa(
                layer_idx, candidate.n_layers
            ):
                actual_swa.append(layer_idx)
            else:
                actual_full.append(layer_idx)
        else:
            raise TypeError(
                f"unexpected mixed6 sequence mixer at layer {layer_idx}: {type(mixer).__name__}"
            )

    actual_pattern = (tuple(actual_kda), tuple(actual_swa), tuple(actual_full))
    expected_pattern = (
        KDA_MIXED6_KDA_LAYERS,
        KDA_MIXED6_SWA_LAYERS,
        KDA_MIXED6_FULL_ATTENTION_LAYERS,
    )
    if actual_pattern != expected_pattern:
        raise ValueError(
            f"unexpected mixed6 pattern: expected {expected_pattern}, found {actual_pattern}"
        )
    if resolved[0].routed_experts is not None:
        raise ValueError("mixed6 model must retain the dense-first layer-0 FFN")
    if candidate.init_std != source.init_std:
        raise ValueError("mixed6 model changed initialization")

    for layer_idx in KDA_MIXED6_KDA_LAYERS:
        kda = cast(KimiDeltaAttentionConfig, resolved[layer_idx].sequence_mixer)
        if kda.expand_v != 2.0 or not kda.allow_neg_eigval:
            raise ValueError(
                f"mixed6 KDA layer {layer_idx} must use expand_v=2 and negative eigenvalues"
            )
    for layer_idx in (*KDA_MIXED6_SWA_LAYERS, *KDA_MIXED6_FULL_ATTENTION_LAYERS):
        attention = cast(AttentionConfig, resolved[layer_idx].sequence_mixer)
        if attention.rope is not None:
            raise ValueError(f"mixed6 attention layer {layer_idx} must use NoPE")
        if (
            attention.gate is None
            or attention.gate.granularity != GateGranularity.elementwise
            or not attention.gate.full_precision
        ):
            raise ValueError(f"mixed6 attention layer {layer_idx} must retain elementwise gating")
    for layer_idx in KDA_MIXED6_SWA_LAYERS:
        attention = cast(AttentionConfig, resolved[layer_idx].sequence_mixer)
        assert attention.sliding_window is not None
        if (
            attention.sliding_window.get_window_size(layer_idx, candidate.n_layers)
            != SWA_WINDOW_SIZE
        ):
            raise ValueError(
                f"mixed6 SWA layer {layer_idx} must use a {SWA_WINDOW_SIZE}-token window"
            )
    for layer_idx in KDA_MIXED6_FULL_ATTENTION_LAYERS:
        attention = cast(AttentionConfig, resolved[layer_idx].sequence_mixer)
        if attention.sliding_window is not None:
            raise ValueError(f"mixed6 full-attention layer {layer_idx} cannot use SWA")

    for layer_idx in range(1, candidate.n_layers):
        block = resolved[layer_idx]
        if block.shared_experts is None or block.routed_experts is None:
            raise ValueError(f"mixed6 layer {layer_idx} must retain MoE experts")
        if (
            block.shared_experts.hidden_size != KDA_MIXED6_EXPERT_HIDDEN_SIZE
            or block.routed_experts.hidden_size != KDA_MIXED6_EXPERT_HIDDEN_SIZE
        ):
            raise ValueError(f"mixed6 layer {layer_idx} has an unexpected expert width")

    actual_counts = (
        candidate.num_active_params,
        candidate.num_active_non_embedding_params,
        candidate.num_params,
    )
    if actual_counts != EXPECTED_KDA_MIXED6_COUNTS:
        raise ValueError(
            f"unexpected mixed6 parameter counts: expected "
            f"{EXPECTED_KDA_MIXED6_COUNTS}, found {actual_counts}"
        )

    return candidate


def build_geometry_matched_kda_mixed6_latent2x_model_config() -> OLMoDDPModelConfig:
    """Add paper-matched L=2 LatentMoE to the mixed-attention candidate.

    This retains the 12-layer 3-KDA/2-SWA/1-full-attention motif and its
    552-wide experts. Only the routed expert path is projected from 640 to
    320, while routed expert count and top-k are doubled to 512 and 16.
    Routing continues to see the full 640-wide token representation.
    """

    candidate = build_geometry_matched_kda_mixed6_model_config()
    resolved = candidate.resolved_block_configs

    def with_latent_moe(block: OLMoDDPTransformerBlockConfig) -> OLMoDDPTransformerBlockConfig:
        latent = deepcopy(block)
        if latent.routed_experts is None or latent.routed_experts_router is None:
            raise ValueError("mixed6 LatentMoE requires a routed expert and router")
        latent.latent_moe = LatentMoEConfig(
            latent_dim=LATENT_MOE_ROUTED_DIMS_BY_COMPRESSION[2],
            up_proj_input_norm_enabled=False,
        )
        latent.routed_experts.d_model = LATENT_MOE_ROUTED_DIMS_BY_COMPRESSION[2]
        latent.routed_experts_router.d_model = D_MODEL
        latent.routed_experts.num_experts *= 2
        latent.routed_experts_router.num_experts *= 2
        latent.routed_experts_router.top_k *= 2
        return latent

    candidate.block = with_latent_moe(resolved[2])
    candidate.block_overrides = {
        0: deepcopy(resolved[0]),
        **{
            layer_idx: with_latent_moe(resolved[layer_idx])
            for layer_idx in range(1, candidate.n_layers)
        },
    }
    candidate.validate()

    for layer_idx, block in enumerate(candidate.resolved_block_configs):
        if layer_idx == 0:
            if block.latent_moe is not None or block.routed_experts is not None:
                raise ValueError("mixed6 LatentMoE must retain the dense-first layer 0")
            continue
        if block.latent_moe is None or block.latent_moe.latent_dim != 320:
            raise ValueError(f"mixed6 LatentMoE layer {layer_idx} has the wrong latent width")
        if block.routed_experts is None or block.routed_experts_router is None:
            raise ValueError(f"mixed6 LatentMoE layer {layer_idx} lost its routed branch")
        if (
            block.routed_experts.d_model != 320
            or block.routed_experts.hidden_size != KDA_MIXED6_EXPERT_HIDDEN_SIZE
            or block.routed_experts.num_experts != 512
            or block.routed_experts_router.d_model != D_MODEL
            or block.routed_experts_router.num_experts != 512
            or block.routed_experts_router.top_k != 16
        ):
            raise ValueError(f"mixed6 LatentMoE layer {layer_idx} has inconsistent expert settings")
        if block.shared_experts is None or block.shared_experts.d_model != D_MODEL:
            raise ValueError(f"mixed6 LatentMoE layer {layer_idx} changed its shared expert")

    actual_counts = (
        candidate.num_active_params,
        candidate.num_active_non_embedding_params,
        candidate.num_params,
    )
    if actual_counts != EXPECTED_KDA_MIXED6_LATENT2X_COUNTS:
        raise ValueError(
            "unexpected mixed6 LatentMoE counts: expected "
            f"{EXPECTED_KDA_MIXED6_LATENT2X_COUNTS}, found {actual_counts}"
        )

    return candidate


def build_geometry_matched_kda_latent_moe_model_config(
    *,
    compression: int,
    up_proj_input_norm_enabled: bool = False,
    scale_experts_with_compression: bool = False,
    num_experts_override: int | None = None,
) -> OLMoDDPModelConfig:
    """Add LatentMoE to the promoted 275M KDA recipe.

    Only the routed expert branch is projected. Routing decisions are made from
    the full-width token representation, while the routed experts operate at
    ``d_model / compression``. The shared expert, attention/recurrent mixers,
    residual stream, and expert hidden width stay unchanged. When
    ``scale_experts_with_compression`` is enabled, both total experts and top-k
    are multiplied by the compression ratio, matching the paper's
    parameter/compute-matched recipe. The optional pre-up-projection RMSNorm is
    exposed explicitly so a later ablation cannot silently change the architecture.
    ``num_experts_override`` supports the explicitly audited 1,000-expert L=4
    EP1 approximation while leaving top-k at the paper-matched value of 32.
    """

    try:
        latent_dim = LATENT_MOE_ROUTED_DIMS_BY_COMPRESSION[compression]
    except KeyError as exc:
        raise ValueError(
            f"unsupported LatentMoE compression={compression}; expected one of "
            f"{tuple(LATENT_MOE_ROUTED_DIMS_BY_COMPRESSION)}"
        ) from exc
    if num_experts_override is not None:
        if not scale_experts_with_compression:
            raise ValueError("num_experts_override requires paper-matched expert scaling")
        if num_experts_override <= TOP_K * compression:
            raise ValueError("num_experts_override must exceed the routed top-k")

    candidate = build_geometry_matched_kda_model_config(
        expand_v=2.0,
        allow_neg_eigval=True,
        expert_hidden_size=664,
    )
    resolved = candidate.resolved_block_configs

    def with_latent_moe(block: OLMoDDPTransformerBlockConfig) -> OLMoDDPTransformerBlockConfig:
        latent = deepcopy(block)
        if latent.routed_experts is None or latent.routed_experts_router is None:
            raise ValueError("LatentMoE requires a routed expert and router")
        latent.latent_moe = LatentMoEConfig(
            latent_dim=latent_dim,
            up_proj_input_norm_enabled=up_proj_input_norm_enabled,
        )
        latent.routed_experts.d_model = latent_dim
        latent.routed_experts_router.d_model = D_MODEL
        if scale_experts_with_compression:
            latent.routed_experts.num_experts *= compression
            latent.routed_experts_router.num_experts *= compression
            latent.routed_experts_router.top_k *= compression
        if num_experts_override is not None:
            latent.routed_experts.num_experts = num_experts_override
            latent.routed_experts_router.num_experts = num_experts_override
        return latent

    candidate.block = with_latent_moe(resolved[1])
    candidate.block_overrides = {
        0: deepcopy(resolved[0]),
        **{layer_idx: with_latent_moe(resolved[layer_idx]) for layer_idx in FULL_ATTENTION_LAYERS},
    }
    candidate.validate()

    for layer_idx, block in enumerate(candidate.resolved_block_configs):
        if layer_idx == 0:
            if block.latent_moe is not None or block.routed_experts is not None:
                raise ValueError("LatentMoE candidate must retain the dense-first layer 0")
            continue
        if block.latent_moe is None:
            raise ValueError(f"LatentMoE candidate layer {layer_idx} is missing latent_moe")
        if block.latent_moe.latent_dim != latent_dim:
            raise ValueError(f"LatentMoE candidate layer {layer_idx} has the wrong latent width")
        if block.latent_moe.up_proj_input_norm_enabled != up_proj_input_norm_enabled:
            raise ValueError(f"LatentMoE candidate layer {layer_idx} has the wrong norm setting")
        if block.routed_experts is None or block.routed_experts_router is None:
            raise ValueError(f"LatentMoE candidate layer {layer_idx} lost its routed branch")
        if (
            block.routed_experts.d_model != latent_dim
            or block.routed_experts_router.d_model != D_MODEL
        ):
            raise ValueError(f"LatentMoE candidate layer {layer_idx} has inconsistent dimensions")
        if block.shared_experts is None or block.shared_experts.d_model != D_MODEL:
            raise ValueError(f"LatentMoE candidate layer {layer_idx} changed its shared expert")
        expected_num_experts = num_experts_override or (
            NUM_EXPERTS * compression if scale_experts_with_compression else NUM_EXPERTS
        )
        expected_top_k = TOP_K * compression if scale_experts_with_compression else TOP_K
        if (
            block.routed_experts.hidden_size != 664
            or block.routed_experts.num_experts != expected_num_experts
            or block.routed_experts_router.num_experts != expected_num_experts
            or block.routed_experts_router.top_k != expected_top_k
        ):
            raise ValueError(
                f"LatentMoE candidate layer {layer_idx} has the wrong expert width/count/top-k"
            )

    actual_counts = (
        candidate.num_active_params,
        candidate.num_active_non_embedding_params,
        candidate.num_params,
    )
    expected_counts = EXPECTED_LATENT_MOE_COUNTS[
        (compression, scale_experts_with_compression, num_experts_override)
    ]
    if not up_proj_input_norm_enabled and actual_counts != expected_counts:
        raise ValueError(
            f"unexpected LatentMoE {compression}x parameter counts: "
            f"expected {expected_counts}, found {actual_counts}"
        )

    return candidate


def build_geometry_matched_swa_model_config() -> OLMoDDPModelConfig:
    """Replace only the geometry RoPE-gated profile's GDN mixers with SWA."""

    candidate = build_geometry_matched_model_config("geometry_rope_gated")
    resolved = candidate.resolved_block_configs

    # The same pattern is attached to every restored attention mixer. It gives
    # layers 0--3 and 5--8 a 2,048-token window while layers 4 and 9 remain the
    # separately configured global-attention overrides.
    sliding_window = SlidingWindowAttentionConfig(
        pattern=[SWA_WINDOW_SIZE, SWA_WINDOW_SIZE, SWA_WINDOW_SIZE, SWA_WINDOW_SIZE, -1],
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
    )
    attention_template = deepcopy(
        cast(AttentionConfig, resolved[FULL_ATTENTION_LAYERS[0]].sequence_mixer)
    )
    attention_template.gate = None
    attention_template.sliding_window = sliding_window

    default_block = deepcopy(resolved[1])
    default_block.sequence_mixer = deepcopy(attention_template)
    dense_first = deepcopy(resolved[0])
    dense_first.sequence_mixer = deepcopy(attention_template)
    candidate.block = default_block
    candidate.block_overrides = {
        0: dense_first,
        **{layer_idx: deepcopy(resolved[layer_idx]) for layer_idx in FULL_ATTENTION_LAYERS},
    }
    candidate.validate()

    actual_swa: list[int] = []
    actual_full: list[int] = []
    for layer_idx, block in enumerate(candidate.resolved_block_configs):
        attention = cast(AttentionConfig, block.sequence_mixer)
        if attention.sliding_window is not None and attention.sliding_window.should_use_swa(
            layer_idx, candidate.n_layers
        ):
            actual_swa.append(layer_idx)
        else:
            actual_full.append(layer_idx)
    if tuple(actual_swa) != GDN_LAYERS or tuple(actual_full) != FULL_ATTENTION_LAYERS:
        raise ValueError(
            f"unexpected SWA mixer pattern: SWA={tuple(actual_swa)}, full={tuple(actual_full)}"
        )
    if candidate.resolved_block_configs[0].routed_experts is not None:
        raise ValueError("SWA control must retain the dense-first layer-0 FFN")
    for layer_idx in actual_swa:
        attention = cast(
            AttentionConfig, candidate.resolved_block_configs[layer_idx].sequence_mixer
        )
        if attention.gate is not None:
            raise ValueError(f"restored SWA layer {layer_idx} must remain ungated")
        if attention.rope is None:
            raise ValueError(f"restored SWA layer {layer_idx} must retain RoPE")
        assert attention.sliding_window is not None
        if (
            attention.sliding_window.get_window_size(layer_idx, candidate.n_layers)
            != SWA_WINDOW_SIZE
        ):
            raise ValueError(f"restored SWA layer {layer_idx} must use a 2,048-token window")
    for layer_idx in actual_full:
        attention = cast(
            AttentionConfig, candidate.resolved_block_configs[layer_idx].sequence_mixer
        )
        if (
            attention.gate is None
            or attention.gate.granularity != GateGranularity.elementwise
            or not attention.gate.full_precision
        ):
            raise ValueError(f"global-attention layer {layer_idx} must retain its elementwise gate")

    actual_counts = (
        candidate.num_active_params,
        candidate.num_active_non_embedding_params,
        candidate.num_params,
    )
    if actual_counts != EXPECTED_SWA_COUNTS:
        raise ValueError(
            f"unexpected SWA-control parameter counts: expected {EXPECTED_SWA_COUNTS}, "
            f"found {actual_counts}"
        )

    return candidate


def build_geometry_matched_one_to_one_model_config(
    mixer: Literal["gdn1", "gdn2", "swa"],
    *,
    gdn2_disable_recompute: bool = False,
    swa_n_layers: Literal[10, 12] | None = None,
) -> OLMoDDPModelConfig:
    """Build the gated-RoPE geometry with a strict 1:1 mixer/attention ratio.

    The model alternates a recurrent or local-attention layer at even indices
    with gated global RoPE attention at odd indices. Only ``n_layers`` and the
    resulting layer placement differ from the existing throughput controls.
    """

    if mixer not in ONE_TO_ONE_N_LAYERS:
        raise ValueError(f"unknown 1:1 mixer {mixer!r}")

    source = build_geometry_matched_model_config("geometry_rope_gated")
    source_resolved = source.resolved_block_configs
    if swa_n_layers is not None and mixer != "swa":
        raise ValueError("swa_n_layers is only valid for the SWA 1:1 control")
    n_layers = swa_n_layers if swa_n_layers is not None else ONE_TO_ONE_N_LAYERS[mixer]
    full_attention_layers = tuple(range(1, n_layers, 2))
    hybrid_layers = tuple(range(0, n_layers, 2))
    if len(full_attention_layers) != len(hybrid_layers):
        raise ValueError(f"1:1 layout requires an even layer count, got {n_layers}")

    candidate = deepcopy(source)
    candidate.n_layers = n_layers
    default_block = deepcopy(source_resolved[1])
    dense_first = deepcopy(source_resolved[0])
    full_attention = deepcopy(source_resolved[FULL_ATTENTION_LAYERS[0]])

    if mixer == "gdn2":
        old_gdn = cast(GatedDeltaNetConfig, default_block.sequence_mixer)
        recurrent = GatedDeltaNet2Config(
            n_heads=old_gdn.n_heads,
            n_v_heads=old_gdn.n_v_heads,
            head_dim=old_gdn.head_dim,
            expand_v=old_gdn.expand_v,
            allow_neg_eigval=old_gdn.allow_neg_eigval,
            conv_size=old_gdn.conv_size,
            conv_bias=old_gdn.conv_bias,
            disable_recompute=gdn2_disable_recompute,
            norm_eps=old_gdn.norm_eps,
            dtype=old_gdn.dtype,
        )
        default_block.sequence_mixer = deepcopy(recurrent)
        dense_first.sequence_mixer = deepcopy(recurrent)
    elif mixer == "swa":
        sliding_window = SlidingWindowAttentionConfig(
            pattern=[SWA_WINDOW_SIZE, -1],
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=True,
        )
        local_attention = deepcopy(cast(AttentionConfig, full_attention.sequence_mixer))
        local_attention.gate = None
        local_attention.sliding_window = sliding_window
        default_block.sequence_mixer = deepcopy(local_attention)
        dense_first.sequence_mixer = deepcopy(local_attention)

    candidate.block = default_block
    candidate.block_overrides = {
        0: dense_first,
        **{layer_idx: deepcopy(full_attention) for layer_idx in full_attention_layers},
    }
    candidate.validate()

    resolved = candidate.resolved_block_configs
    actual_full: list[int] = []
    actual_hybrid: list[int] = []
    for layer_idx, block in enumerate(resolved):
        sequence_mixer = block.sequence_mixer
        if isinstance(sequence_mixer, AttentionConfig):
            if (
                sequence_mixer.sliding_window is not None
                and sequence_mixer.sliding_window.should_use_swa(layer_idx, n_layers)
            ):
                actual_hybrid.append(layer_idx)
            else:
                actual_full.append(layer_idx)
        elif isinstance(sequence_mixer, (GatedDeltaNetConfig, GatedDeltaNet2Config)):
            actual_hybrid.append(layer_idx)
        else:
            raise TypeError(
                f"unexpected {mixer} sequence mixer at layer {layer_idx}: "
                f"{type(sequence_mixer).__name__}"
            )

    if tuple(actual_hybrid) != hybrid_layers or tuple(actual_full) != full_attention_layers:
        raise ValueError(
            f"unexpected {mixer} 1:1 pattern: hybrid={tuple(actual_hybrid)}, "
            f"full={tuple(actual_full)}"
        )
    if resolved[0].routed_experts is not None:
        raise ValueError(f"{mixer} 1:1 model must retain the dense-first layer-0 FFN")
    if candidate.init_std != source.init_std:
        raise ValueError(f"{mixer} 1:1 model changed initialization")

    for layer_idx in full_attention_layers:
        attention = cast(AttentionConfig, resolved[layer_idx].sequence_mixer)
        if attention.sliding_window is not None or attention.rope is None:
            raise ValueError(f"{mixer} layer {layer_idx} must use global RoPE attention")
        if (
            attention.gate is None
            or attention.gate.granularity != GateGranularity.elementwise
            or not attention.gate.full_precision
        ):
            raise ValueError(
                f"{mixer} layer {layer_idx} must retain full-precision elementwise gating"
            )

    if mixer == "gdn1" and any(
        not isinstance(resolved[layer_idx].sequence_mixer, GatedDeltaNetConfig)
        for layer_idx in hybrid_layers
    ):
        raise ValueError("GDN1 1:1 model contains a non-GDN1 hybrid layer")
    if mixer == "gdn2" and any(
        not isinstance(resolved[layer_idx].sequence_mixer, GatedDeltaNet2Config)
        for layer_idx in hybrid_layers
    ):
        raise ValueError("GDN2 1:1 model contains a non-GDN2 hybrid layer")
    if mixer == "swa":
        for layer_idx in hybrid_layers:
            attention = cast(AttentionConfig, resolved[layer_idx].sequence_mixer)
            if attention.gate is not None or attention.rope is None:
                raise ValueError(f"SWA layer {layer_idx} must retain ungated local RoPE attention")
            assert attention.sliding_window is not None
            if attention.sliding_window.get_window_size(layer_idx, n_layers) != SWA_WINDOW_SIZE:
                raise ValueError(f"SWA layer {layer_idx} must use a {SWA_WINDOW_SIZE}-token window")

    actual_counts = (
        candidate.num_active_params,
        candidate.num_active_non_embedding_params,
        candidate.num_params,
    )
    expected_counts_key = "swa_10l" if mixer == "swa" and n_layers == 10 else mixer
    if actual_counts != EXPECTED_ONE_TO_ONE_COUNTS[expected_counts_key]:
        raise ValueError(
            f"unexpected {mixer} 1:1 parameter counts: expected "
            f"{EXPECTED_ONE_TO_ONE_COUNTS[expected_counts_key]}, found {actual_counts}"
        )

    return candidate


def parameter_summary(profile_name: str) -> dict[str, Any]:
    parent = build_hybrid_model_config("275m")
    candidate = build_geometry_matched_model_config(profile_name)
    profile = PROFILES[profile_name]
    return {
        "profile": profile_name,
        "d_model": candidate.d_model,
        "n_layers": candidate.n_layers,
        "gdn_layers": list(GDN_LAYERS),
        "full_attention_layers": list(FULL_ATTENTION_LAYERS),
        "dense_ffn_layers": [0],
        "n_heads": N_HEADS,
        "n_kv_heads": profile.n_kv_heads,
        "head_dim": HEAD_DIM,
        "attention_gate": profile.attention_gate,
        "expert_hidden_size": profile.expert_hidden_size,
        "dense_first_hidden_size": (TOP_K + 1) * profile.expert_hidden_size,
        "num_experts": 256,
        "top_k": TOP_K,
        "shared_experts": 1,
        "gdn_expand_v": profile.gdn_expand_v,
        "rope": profile.rope,
        "init_std": candidate.init_std,
        "active_params": candidate.num_active_params,
        "active_non_embedding_params": candidate.num_active_non_embedding_params,
        "total_params": candidate.num_params,
        "delta_vs_current_hybrid_active": (candidate.num_active_params - parent.num_active_params),
        "delta_fraction_vs_current_hybrid_active": (
            candidate.num_active_params / parent.num_active_params - 1
        ),
        "delta_vs_current_hybrid_active_non_embedding": (
            candidate.num_active_non_embedding_params - parent.num_active_non_embedding_params
        ),
        "delta_fraction_vs_current_hybrid_active_non_embedding": (
            candidate.num_active_non_embedding_params / parent.num_active_non_embedding_params - 1
        ),
        "delta_vs_dense_reference_active": candidate.num_active_params
        - DENSE_REFERENCE_ACTIVE_PARAMS,
        "delta_fraction_vs_dense_reference_active": candidate.num_active_params
        / DENSE_REFERENCE_ACTIVE_PARAMS
        - 1,
        "delta_vs_dense_reference_active_non_embedding": (
            candidate.num_active_non_embedding_params - DENSE_REFERENCE_ACTIVE_NON_EMBEDDING_PARAMS
        ),
        "delta_fraction_vs_dense_reference_active_non_embedding": (
            candidate.num_active_non_embedding_params / DENSE_REFERENCE_ACTIVE_NON_EMBEDDING_PARAMS
            - 1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), action="append")
    args = parser.parse_args()
    profiles = args.profile or list(PROFILES)
    print(json.dumps([parameter_summary(profile) for profile in profiles], indent=2))


if __name__ == "__main__":
    main()
