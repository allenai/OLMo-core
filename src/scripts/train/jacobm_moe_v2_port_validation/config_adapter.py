"""Narrow config adapter for validating the olmo-ddp -> moe-v2-core port.

The upstreamable branch intentionally keeps one block-level ``layer_norm``
config and uses attention ``head_dim`` instead of olmo-ddp's ``d_attn``.  This
module translates only those two representation changes.  Everything else is
left untouched so the checkpoint schema and deterministic forward pass can act
as the compatibility oracle.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.transformer import OLMoDDPModelConfig


def load_recorded_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def adapt_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Adapt a recorded olmo-ddp model payload without changing its semantics."""

    model = copy.deepcopy(payload)
    blocks = [model["block"], *(model.get("block_overrides") or {}).values()]
    for block in blocks:
        attention_norm = block.pop("attention_norm")
        feed_forward_norm = block.pop("feed_forward_norm")
        if attention_norm != feed_forward_norm:
            raise ValueError(
                "The port's single layer_norm field cannot represent unequal "
                "attention/feed-forward norm configs"
            )
        block["layer_norm"] = attention_norm

        mixer = block["sequence_mixer"]
        if "d_attn" in mixer:
            d_attn = int(mixer.pop("d_attn"))
            n_heads = int(mixer["n_heads"])
            if d_attn % n_heads:
                raise ValueError(f"d_attn={d_attn} is not divisible by n_heads={n_heads}")
            head_dim = d_attn // n_heads
            recorded_head_dim = mixer.get("head_dim")
            if recorded_head_dim is not None and int(recorded_head_dim) != head_dim:
                raise ValueError(
                    f"Conflicting attention head dimensions: {recorded_head_dim} vs {head_dim}"
                )
            mixer["head_dim"] = head_dim
    return model


def build_model_config(config_path: Path) -> OLMoDDPModelConfig:
    recorded = load_recorded_config(config_path)
    model = OLMoDDPModelConfig.from_dict(adapt_model_payload(recorded["model"]))
    model.validate()

    # The source family always uses explicit 128-wide attention heads.  This
    # catches an accidental d_attn -> head_dim mapping that would still build.
    for layer_idx, block in enumerate(model.resolved_block_configs):
        mixer = block.sequence_mixer
        if isinstance(mixer, AttentionConfig) and mixer.head_dim != 128:
            raise ValueError(
                f"Layer {layer_idx} has unexpected attention head_dim={mixer.head_dim}"
            )
    return model
