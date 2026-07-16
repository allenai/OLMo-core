#!/usr/bin/env python3
"""Design candidate 275M MoE hybrids on the dense ladder's model geometry.

The primary ``geometry_only`` profile isolates width, depth, mixer placement,
and head geometry from the already-tested hybrid. It deliberately retains our
current GQA ratio and ungated RoPE full-attention blocks. The
``dense_attention`` profile additionally matches the dense 275M rung's KV-head
count and elementwise attention gate, while still holding NoPE, ``expand_v=2``,
and initialization for their own later interventions.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

from olmo_core.nn.attention import AttentionConfig, GateConfig, GateGranularity
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

# Checked-in dense-ladder 275M reference values.
DENSE_REFERENCE_ACTIVE_PARAMS = 275_493_760
DENSE_REFERENCE_ACTIVE_NON_EMBEDDING_PARAMS = 211_268_480


@dataclass(frozen=True)
class Profile:
    expert_hidden_size: int
    n_kv_heads: int
    attention_gate: bool


PROFILES = {
    # Closest practical active match while changing only the already-planned
    # geometry bundle. 664 is BF16-tile aligned and keeps the dense first FFN
    # equal to top-k routed width plus shared-expert width.
    "geometry_only": Profile(
        expert_hidden_size=664,
        n_kv_heads=4,
        attention_gate=False,
    ),
    # Also match the dense 275M full-attention shape (8 Q / 8 KV) and its
    # elementwise gate. 648 gives a near-exact active match with 8-wide tensor
    # alignment; odd hidden widths get closer numerically but are poor
    # production shapes.
    "dense_attention": Profile(
        expert_hidden_size=648,
        n_kv_heads=8,
        attention_gate=True,
    ),
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
    attention.d_attn = None
    attention.sliding_window = None
    attention.gate = (
        GateConfig(
            granularity=GateGranularity.elementwise,
            full_precision=True,
        )
        if profile.attention_gate
        else None
    )
    full.sequence_mixer = attention
    return full


def build_geometry_matched_model_config(
    profile_name: str = "geometry_only",
) -> OLMoDDPModelConfig:
    """Build one geometry-matched candidate without changing NoPE/init/expand-v."""

    try:
        profile = PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(f"unknown profile {profile_name!r}; choose one of {tuple(PROFILES)}") from exc

    parent = build_hybrid_model_config("275m")
    candidate = deepcopy(parent)
    candidate.d_model = D_MODEL
    candidate.n_layers = N_LAYERS
    candidate.embed_scale = math.sqrt(D_MODEL)

    # Use a tested expand_v=1 GDN block as the base for all layers.
    moe_block = deepcopy(parent.resolved_block_configs[2])
    _resize_moe_block(moe_block, profile.expert_hidden_size)
    gdn = cast(GatedDeltaNetConfig, moe_block.sequence_mixer)
    gdn.n_heads = N_HEADS
    gdn.n_v_heads = N_HEADS
    gdn.head_dim = HEAD_DIM
    gdn.expand_v = 1.0
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
        i
        for i, block in enumerate(resolved)
        if isinstance(block.sequence_mixer, AttentionConfig)
    )
    if actual_gdn != GDN_LAYERS or actual_full != FULL_ATTENTION_LAYERS:
        raise ValueError(f"unexpected mixer pattern: GDN={actual_gdn}, full={actual_full}")
    if resolved[0].routed_experts is not None:
        raise ValueError("layer 0 must retain the dense-first FFN design")
    if candidate.init_std != 0.01:
        raise ValueError("geometry candidate must retain init_std=0.01")
    if any(
        cast(GatedDeltaNetConfig, resolved[i].sequence_mixer).expand_v != 1.0
        for i in GDN_LAYERS
    ):
        raise ValueError("geometry candidate must retain expand_v=1")
    if any(cast(AttentionConfig, resolved[i].sequence_mixer).rope is None for i in actual_full):
        raise ValueError("geometry candidate must retain RoPE")

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
        "gdn_expand_v": 1.0,
        "rope": True,
        "init_std": candidate.init_std,
        "active_params": candidate.num_active_params,
        "active_non_embedding_params": candidate.num_active_non_embedding_params,
        "total_params": candidate.num_params,
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
            candidate.num_active_non_embedding_params
            - DENSE_REFERENCE_ACTIVE_NON_EMBEDDING_PARAMS
        ),
        "delta_fraction_vs_dense_reference_active_non_embedding": (
            candidate.num_active_non_embedding_params
            / DENSE_REFERENCE_ACTIVE_NON_EMBEDDING_PARAMS
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
