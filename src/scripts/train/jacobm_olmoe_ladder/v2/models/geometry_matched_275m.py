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
from typing import Any, cast

from olmo_core.nn.attention import (
    AttentionConfig,
    GateConfig,
    GateGranularity,
    GatedDeltaNet2Config,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.ddp import OLMoDDPTransformerBlockConfig
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
EXPECTED_GDN2_COUNTS = (306_191_168, 241_965_888, 3_151_723_328)


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
        if any(
            cast(AttentionConfig, resolved[i].sequence_mixer).rope is None for i in actual_full
        ):
            raise ValueError(f"{profile_name} must retain RoPE")
    elif any(cast(AttentionConfig, resolved[i].sequence_mixer).rope is not None for i in actual_full):
        raise ValueError(f"{profile_name} must use NoPE")
    for layer_idx in actual_full:
        attention = cast(AttentionConfig, resolved[layer_idx].sequence_mixer)
        if attention.n_kv_heads != profile.n_kv_heads:
            raise ValueError(
                f"{profile_name} layer {layer_idx} must use "
                f"{profile.n_kv_heads} KV heads"
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
    *, rope: bool = True, disable_recompute: bool = False
) -> OLMoDDPModelConfig:
    """Swap GDN1 for GDN2 in a 275M gated geometry candidate.

    All geometry, MoE, full-attention, initialization, and optimization-facing
    settings remain unchanged. ``rope=False`` selects the corresponding gated
    NoPE parent. ``allow_neg_eigval`` is retained from GDN1 so the mixer
    generation is the only other architectural intervention.
    """

    profile_name = "geometry_rope_gated" if rope else "geometry_nope_gated"
    candidate = build_geometry_matched_model_config(profile_name)
    resolved = candidate.resolved_block_configs
    old_gdn = cast(GatedDeltaNetConfig, resolved[1].sequence_mixer)
    gdn2 = GatedDeltaNet2Config(
        n_heads=old_gdn.n_heads,
        n_v_heads=old_gdn.n_v_heads,
        head_dim=old_gdn.head_dim,
        expand_v=old_gdn.expand_v,
        allow_neg_eigval=old_gdn.allow_neg_eigval,
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
        (cast(AttentionConfig, candidate.resolved_block_configs[layer_idx].sequence_mixer).rope
         is None)
        != (not rope)
        for layer_idx in FULL_ATTENTION_LAYERS
    ):
        raise ValueError(f"GDN2 {profile_name} positional encoding does not match rope={rope}")
    actual_counts = (
        candidate.num_active_params,
        candidate.num_active_non_embedding_params,
        candidate.num_params,
    )
    if actual_counts != EXPECTED_GDN2_COUNTS:
        raise ValueError(
            f"unexpected GDN2 parameter counts: expected {EXPECTED_GDN2_COUNTS}, "
            f"found {actual_counts}"
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
        if attention.sliding_window.get_window_size(layer_idx, candidate.n_layers) != SWA_WINDOW_SIZE:
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
