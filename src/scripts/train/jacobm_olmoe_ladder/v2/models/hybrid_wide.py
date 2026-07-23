#!/usr/bin/env python3
"""Audited integration-wide to GDN-hybrid model-config transformation.

The transformation replaces only sliding-window attention mixers with
GatedDeltaNet. It preserves width, depth, full-attention layers, dense/MoE
placement and geometry, norms, RoPE, initialization, and every trainer-facing
model field. ``expand_v=1`` is the active-near-matched control tested at 275M.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.transformer import OLMoDDPModelConfig
from scripts.train.jacobm_olmoe_ladder.v2.moe_v2_core_adapter import (
    build_model_config_from_recorded,
)


REPO_ROOT = Path(__file__).resolve().parents[6]
MODEL_SIZES = ("275m", "480m", "810m", "1p2b")
SOURCE_CONFIGS = {
    size: REPO_ROOT
    / "JACOBM_DDP_CONFIGS"
    / "pretraining"
    / "integration_wide"
    / size
    / "cx1"
    / "config.json"
    for size in MODEL_SIZES
}
EXPECTED_SWA_LAYERS = {
    "275m": (2, 4, 6, 8, 10),
    "480m": (2, 4, 6, 8, 10, 12, 14),
    "810m": (2, 4, 6, 8, 10, 12, 14, 16, 18),
    "1p2b": (2, 4, 6, 8, 10, 12, 14, 16, 18, 20),
}

HYBRID_GDN_EXPAND_V = 1.0
GDN_HEAD_DIM = 128
MAX_ACTIVE_PARAMETER_DELTA_FRACTION = 0.06


def _validate_model_size(model_size: str) -> None:
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Unknown model size {model_size!r}; choose one of {MODEL_SIZES}")


def load_wide_model_config(
    model_size: str,
    *,
    experiment_config: Path | None = None,
) -> OLMoDDPModelConfig:
    """Load one converted integration-wide model config.

    The model payloads are identical across Cx1/Cx2/Cx4/Cx8 for every size;
    Cx1 is the canonical source so the transformation has one input per size.
    """

    _validate_model_size(model_size)
    path = experiment_config or SOURCE_CONFIGS[model_size]
    return build_model_config_from_recorded(path)


def sliding_window_layers(config: OLMoDDPModelConfig) -> tuple[int, ...]:
    layers: list[int] = []
    for layer_idx, block in enumerate(config.resolved_block_configs):
        mixer = block.sequence_mixer
        if not isinstance(mixer, AttentionConfig) or mixer.sliding_window is None:
            continue
        if mixer.sliding_window.should_use_swa(layer_idx, config.n_layers):
            layers.append(layer_idx)
    return tuple(layers)


def full_attention_layers(config: OLMoDDPModelConfig) -> tuple[int, ...]:
    return tuple(
        layer_idx
        for layer_idx, block in enumerate(config.resolved_block_configs)
        if isinstance(block.sequence_mixer, AttentionConfig)
        and layer_idx not in sliding_window_layers(config)
    )


def build_hybrid_model_config(
    model_size: str,
    *,
    gdn_expand_v: float = HYBRID_GDN_EXPAND_V,
    base_experiment_config: Path | None = None,
) -> OLMoDDPModelConfig:
    """Build the active-near-matched hybrid config for ``model_size``."""

    base = load_wide_model_config(model_size, experiment_config=base_experiment_config)
    swa_layers = sliding_window_layers(base)
    expected_swa_layers = EXPECTED_SWA_LAYERS[model_size]
    if swa_layers != expected_swa_layers:
        raise ValueError(
            f"Unexpected {model_size} integration-wide SWA layers: "
            f"expected {expected_swa_layers}, found {swa_layers}"
        )

    hybrid = deepcopy(base)
    overrides = dict(hybrid.block_overrides or {})
    for layer_idx in swa_layers:
        block = deepcopy(base.resolved_block_configs[layer_idx])
        attention = cast(AttentionConfig, block.sequence_mixer)
        head_dim = attention.head_dim or base.d_model // attention.n_heads
        if head_dim != GDN_HEAD_DIM:
            raise ValueError(
                f"Unexpected {model_size} layer {layer_idx} head dim: "
                f"expected {GDN_HEAD_DIM}, found {head_dim}"
            )
        block.sequence_mixer = GatedDeltaNetConfig(
            n_heads=attention.n_heads,
            n_v_heads=attention.n_heads,
            head_dim=head_dim,
            expand_v=gdn_expand_v,
            allow_neg_eigval=True,
            conv_size=4,
            conv_bias=False,
            norm_eps=1e-5,
            dtype=DType.float32,
        )
        overrides[layer_idx] = block

    hybrid.block_overrides = overrides
    hybrid.validate()

    resolved = hybrid.resolved_block_configs
    actual_gdn_layers = tuple(
        layer_idx
        for layer_idx, block in enumerate(resolved)
        if isinstance(block.sequence_mixer, GatedDeltaNetConfig)
    )
    if actual_gdn_layers != expected_swa_layers:
        raise ValueError(
            f"Unexpected {model_size} GDN layers after conversion: {actual_gdn_layers}"
        )

    for layer_idx in range(base.n_layers):
        base_block = base.resolved_block_configs[layer_idx]
        hybrid_block = resolved[layer_idx]
        base_block_payload = base_block.as_dict()
        hybrid_block_payload = hybrid_block.as_dict()
        base_block_payload.pop("sequence_mixer")
        hybrid_block_payload.pop("sequence_mixer")
        if base_block_payload != hybrid_block_payload:
            raise ValueError(f"Non-mixer block fields changed at {model_size} layer {layer_idx}")

    delta_fraction = (hybrid.num_active_params - base.num_active_params) / base.num_active_params
    if abs(delta_fraction) > MAX_ACTIVE_PARAMETER_DELTA_FRACTION:
        raise ValueError(
            f"{model_size} active-parameter delta {delta_fraction:.3%} exceeds "
            f"the {MAX_ACTIVE_PARAMETER_DELTA_FRACTION:.1%} near-match limit"
        )
    return hybrid


def parameter_summary(model_size: str) -> dict[str, Any]:
    base = load_wide_model_config(model_size)
    hybrid = build_hybrid_model_config(model_size)
    active_delta = hybrid.num_active_params - base.num_active_params
    non_embedding_delta = (
        hybrid.num_active_non_embedding_params - base.num_active_non_embedding_params
    )
    total_delta = hybrid.num_params - base.num_params
    return {
        "model_size": model_size,
        "source": str(SOURCE_CONFIGS[model_size].relative_to(REPO_ROOT)),
        "gdn_layers": list(EXPECTED_SWA_LAYERS[model_size]),
        "full_attention_layers": list(full_attention_layers(base)),
        "gdn": {
            "n_heads": cast(
                GatedDeltaNetConfig,
                hybrid.resolved_block_configs[EXPECTED_SWA_LAYERS[model_size][0]].sequence_mixer,
            ).n_heads,
            "n_v_heads": cast(
                GatedDeltaNetConfig,
                hybrid.resolved_block_configs[EXPECTED_SWA_LAYERS[model_size][0]].sequence_mixer,
            ).n_v_heads,
            "head_dim": GDN_HEAD_DIM,
            "expand_v": HYBRID_GDN_EXPAND_V,
        },
        "base": {
            "active_params": base.num_active_params,
            "active_non_embedding_params": base.num_active_non_embedding_params,
            "total_params": base.num_params,
        },
        "hybrid": {
            "active_params": hybrid.num_active_params,
            "active_non_embedding_params": hybrid.num_active_non_embedding_params,
            "total_params": hybrid.num_params,
            "active_delta": active_delta,
            "active_delta_fraction": active_delta / base.num_active_params,
            "active_non_embedding_delta": non_embedding_delta,
            "active_non_embedding_delta_fraction": (
                non_embedding_delta / base.num_active_non_embedding_params
            ),
            "total_delta": total_delta,
            "total_delta_fraction": total_delta / base.num_params,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=MODEL_SIZES, action="append")
    args = parser.parse_args()
    model_sizes = args.model_size or list(MODEL_SIZES)
    print(json.dumps([parameter_summary(size) for size in model_sizes], indent=2))


if __name__ == "__main__":
    main()
