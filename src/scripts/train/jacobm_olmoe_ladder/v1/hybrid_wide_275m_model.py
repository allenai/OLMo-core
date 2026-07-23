#!/usr/bin/env python3
"""Parameter-audited 275M hybrid model derived from integration-wide.

This isolates the hybrid mixer intervention: every layer that used sliding-window
attention (SWA) becomes Gated DeltaNet (GDN), while all full-attention layers,
the dense prefix, depth, width, MoE geometry, norms, RoPE, and initialization stay
unchanged.

The dense mainline ladder uses ``expand_v=2``. At the integration-wide width that
would raise active parameters by about 7.1%. This isolated control uses
``expand_v=1``, matching the existing MoE GDN implementation while keeping the
active-parameter increase below 3%. The builder accepts another expansion for an
explicit follow-up.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.transformer import OLMoDDPModelConfig
from scripts.train.jacobm_olmoe_ladder.v2.moe_v2_core_adapter import (
    build_model_config_from_recorded,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
BASE_EXPERIMENT_CONFIG = (
    REPO_ROOT / "JACOBM_DDP_CONFIGS/pretraining/integration_wide/275m/cx1/config.json"
)

HYBRID_GDN_EXPAND_V = 1.0
DENSE_LADDER_GDN_EXPAND_V = 2.0
MAX_ACTIVE_PARAMETER_DELTA_FRACTION = 0.03


def load_wide_model_config(
    experiment_config: Path = BASE_EXPERIMENT_CONFIG,
) -> OLMoDDPModelConfig:
    return build_model_config_from_recorded(experiment_config)


def sliding_window_layers(config: OLMoDDPModelConfig) -> tuple[int, ...]:
    layers: list[int] = []
    for layer_idx, block in enumerate(config.resolved_block_configs):
        mixer = block.sequence_mixer
        if not isinstance(mixer, AttentionConfig) or mixer.sliding_window is None:
            continue
        if mixer.sliding_window.should_use_swa(layer_idx, config.n_layers):
            layers.append(layer_idx)
    return tuple(layers)


def build_hybrid_model_config(
    *,
    gdn_expand_v: float = HYBRID_GDN_EXPAND_V,
    base_experiment_config: Path = BASE_EXPERIMENT_CONFIG,
) -> OLMoDDPModelConfig:
    config = load_wide_model_config(base_experiment_config)
    swa_layers = sliding_window_layers(config)
    if swa_layers != (2, 4, 6, 8, 10):
        raise ValueError(f"Unexpected integration-wide SWA layers: {swa_layers}")

    overrides = dict(config.block_overrides or {})
    for layer_idx in swa_layers:
        block = deepcopy(config.resolved_block_configs[layer_idx])
        block.sequence_mixer = GatedDeltaNetConfig(
            n_heads=8,
            n_v_heads=8,
            head_dim=128,
            expand_v=gdn_expand_v,
            allow_neg_eigval=True,
            conv_size=4,
            conv_bias=False,
            norm_eps=1e-5,
            dtype=DType.float32,
        )
        overrides[layer_idx] = block

    config.block_overrides = overrides
    config.validate()
    return config


def parameter_summary() -> dict[str, Any]:
    base = load_wide_model_config()
    hybrid = build_hybrid_model_config()
    dense_default = build_hybrid_model_config(gdn_expand_v=DENSE_LADDER_GDN_EXPAND_V)
    active_delta = hybrid.num_active_params - base.num_active_params
    active_delta_fraction = active_delta / base.num_active_params
    if abs(active_delta_fraction) > MAX_ACTIVE_PARAMETER_DELTA_FRACTION:
        raise ValueError(
            f"Hybrid differs by {active_delta_fraction:.4%}, "
            f"above {MAX_ACTIVE_PARAMETER_DELTA_FRACTION:.4%}"
        )
    return {
        "source": "integration_wide/275m",
        "unchanged": [
            "d_model",
            "d_attn",
            "n_layers",
            "full_attention_layers",
            "dense_prefix",
            "moe_geometry",
            "norms",
            "rope",
            "initialization",
        ],
        "source_swa_layers": list(sliding_window_layers(base)),
        "hybrid_gdn_layers": list(sliding_window_layers(base)),
        "full_attention_layers": [0, 1, 3, 5, 7, 9, 11],
        "gdn": {
            "n_heads": 8,
            "n_v_heads": 8,
            "head_dim": 128,
            "expand_v": HYBRID_GDN_EXPAND_V,
            "allow_neg_eigval": True,
            "conv_size": 4,
        },
        "base": {
            "active_params": base.num_active_params,
            "active_non_embedding_params": base.num_active_non_embedding_params,
            "total_params": base.num_params,
        },
        "hybrid_expand_v_1": {
            "active_params": hybrid.num_active_params,
            "active_non_embedding_params": hybrid.num_active_non_embedding_params,
            "total_params": hybrid.num_params,
            "active_delta": active_delta,
            "active_delta_fraction": active_delta_fraction,
        },
        "dense_expand_v_2_reference": {
            "active_params": dense_default.num_active_params,
            "active_non_embedding_params": dense_default.num_active_non_embedding_params,
            "total_params": dense_default.num_params,
            "active_delta": dense_default.num_active_params - base.num_active_params,
            "active_delta_fraction": (
                dense_default.num_active_params - base.num_active_params
            )
            / base.num_active_params,
        },
    }


if __name__ == "__main__":
    print(json.dumps(parameter_summary(), indent=2, sort_keys=True))
