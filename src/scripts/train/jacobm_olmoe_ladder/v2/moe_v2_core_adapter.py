"""Compatibility helpers for Jacob's ``olmo-ddp`` experiment artifacts.

The upstreamable ``moe-v2-core`` branch deliberately represents several model
settings differently from the branch that produced our recorded configs:

* one block-level ``layer_norm`` replaces the two identical norm configs; and
* attention records ``head_dim`` instead of the aggregate ``d_attn`` width;
* expert-parallel controls live in ``ExpertParallelConfig``; and
* a legacy dense first block is represented as a shared-only DDP block.

These helpers adapt only those representation changes (plus the equivalent
``truncate=true`` YaRN representation). Checkpoint tensor names and values are
not rewritten. The optimizer helper similarly removes Muon-only controls only
after proving they were disabled for every parameter group.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.transformer import OLMoDDPModelConfig


EP_CONFIG_CLASS = "olmo_core.nn.moe.v2.ep_config.ExpertParallelConfig"
MODEL_CONFIG_CLASS = "olmo_core.nn.transformer.config.OLMoDDPModelConfig"
BLOCK_CONFIG_CLASS = "olmo_core.nn.ddp.block.OLMoDDPTransformerBlockConfig"
SHARED_EXPERTS_CONFIG_CLASS = "olmo_core.nn.moe.v2.shared_experts.SharedExpertsConfig"


def _convert_legacy_dense_block(
    block: Mapping[str, Any], *, d_model: int
) -> dict[str, Any]:
    """Represent a legacy dense FFN as an equivalent shared-only DDP block."""

    feed_forward = block.get("feed_forward")
    if feed_forward is None:
        raise ValueError("Expected a legacy dense block with feed_forward")
    if block.get("name") != "peri_norm":
        raise ValueError(
            f"Only legacy peri_norm dense blocks are supported, got {block.get('name')!r}"
        )
    if feed_forward.get("bias", True):
        raise ValueError("Dense checkpoint conversion requires bias=false")
    if feed_forward.get("activation", "silu") != "silu":
        raise ValueError("Dense checkpoint conversion requires SwiGLU/SILU")
    if block.get("dropout", 0.0) not in (None, 0.0):
        raise ValueError("Dense checkpoint conversion requires dropout=0")
    for alpha_name in ("attention_residual_alpha", "feed_forward_residual_alpha"):
        if block.get(alpha_name, 1.0) not in (None, 1.0):
            raise ValueError(f"Dense checkpoint conversion requires {alpha_name}=1")
    for norm_name in ("attention_norm", "feed_forward_norm"):
        if block[norm_name].get("bias", False):
            raise ValueError("Dense checkpoint conversion does not support norm bias tensors")

    dtype = feed_forward.get("dtype")
    if dtype is None:
        dtype = block["sequence_mixer"].get("dtype", "float32")

    return {
        "sequence_mixer": copy.deepcopy(block["sequence_mixer"]),
        "attention_norm": copy.deepcopy(block["attention_norm"]),
        "feed_forward_norm": copy.deepcopy(block["feed_forward_norm"]),
        "name": "moe_fused_v2",
        "shared_experts": {
            "d_model": d_model,
            "hidden_size": int(feed_forward["hidden_size"]),
            "num_experts": 1,
            "bias": False,
            "dtype": dtype,
            "activation": "swiglu",
            "_CLASS_": SHARED_EXPERTS_CONFIG_CLASS,
        },
        "use_peri_norm": True,
        "checkpoint_attn": False,
        "checkpoint_permute_moe_unpermute": False,
        "checkpoint_second_unpermute": False,
        "_CLASS_": BLOCK_CONFIG_CLASS,
    }


def _adapt_legacy_ep_config(block: dict[str, Any], *, tbo: bool) -> None:
    """Translate the former flat EP controls into ``ExpertParallelConfig``."""

    legacy_ep_keys = {key for key in block if key.startswith("ep_no_sync")}
    if "checkpoint_combined_ep_tbo" in block:
        legacy_ep_keys.add("checkpoint_combined_ep_tbo")

    if "ep" in block:
        if legacy_ep_keys:
            raise ValueError(
                "Block mixes canonical 'ep' with legacy controls: "
                + ", ".join(sorted(legacy_ep_keys))
            )
        ep = block["ep"]
        if ep is not None:
            ep["_CLASS_"] = EP_CONFIG_CLASS
        return

    if not legacy_ep_keys and block.get("routed_experts") is None:
        return

    supported = {
        "ep_no_sync",
        "ep_no_sync_use_2d_all_to_all",
        "ep_no_sync_use_rowwise_all_to_all",
        "ep_no_sync_rowwise_nblocks",
        "ep_no_sync_share_dispatch_out",
        "ep_no_sync_capacity_factor",
        "ep_no_sync_shared_slots",
        "ep_no_sync_share_combine_out",
        "ep_no_sync_major_align",
        "ep_no_sync_restore_unpermute_backend",
        "checkpoint_combined_ep_tbo",
    }
    unsupported = legacy_ep_keys - supported
    if unsupported:
        raise ValueError("Unsupported legacy EP controls: " + ", ".join(sorted(unsupported)))
    if block.get("ep_no_sync_use_2d_all_to_all", False):
        raise ValueError("Legacy 2D all-to-all has no moe-v2-core equivalent")

    no_sync = bool(block.get("ep_no_sync", False))
    rowwise = bool(block.get("ep_no_sync_use_rowwise_all_to_all", False))
    if rowwise and not no_sync:
        raise ValueError("Legacy rowwise all-to-all requires ep_no_sync=true")
    path = "rowwise_nvshmem" if rowwise else ("no_sync_1d" if no_sync else "sync_1d")
    if tbo and path != "rowwise_nvshmem":
        raise ValueError("Two-batch overlap requires rowwise no-sync EP")

    block["ep"] = {
        "path": path,
        "schedule": "tbo" if tbo else "normal",
        "capacity_factor": float(block.get("ep_no_sync_capacity_factor", 1.25)),
        "shared_slots": int(block.get("ep_no_sync_shared_slots", 1)),
        "major_align": int(block.get("ep_no_sync_major_align", 1)),
        "rowwise_nblocks": int(block.get("ep_no_sync_rowwise_nblocks", 32)),
        "share_dispatch_out": bool(block.get("ep_no_sync_share_dispatch_out", False)),
        "share_combine_out": bool(block.get("ep_no_sync_share_combine_out", False)),
        "restore_unpermute_backend": block.get(
            "ep_no_sync_restore_unpermute_backend", "te_fused"
        ),
        "checkpoint_tbo": bool(block.get("checkpoint_combined_ep_tbo", False)),
        "_CLASS_": EP_CONFIG_CLASS,
    }
    for key in supported:
        block.pop(key, None)


def load_recorded_config(path: Path) -> dict[str, Any]:
    """Load a recorded experiment or checkpoint config JSON object."""

    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def adapt_model_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Translate an ``olmo-ddp`` model payload without changing semantics."""

    model = copy.deepcopy(dict(payload))
    d_model = int(model["d_model"])
    if model["block"].get("feed_forward") is not None:
        model["block"] = _convert_legacy_dense_block(model["block"], d_model=d_model)
    else:
        model["block"]["name"] = "moe_fused_v2"
        model["block"]["_CLASS_"] = BLOCK_CONFIG_CLASS

    overrides = model.get("block_overrides") or {}
    for layer_idx, block in tuple(overrides.items()):
        if block.get("feed_forward") is not None:
            overrides[layer_idx] = _convert_legacy_dense_block(block, d_model=d_model)
        else:
            block["name"] = "moe_fused_v2"
            block["_CLASS_"] = BLOCK_CONFIG_CLASS
    model["block_overrides"] = overrides or None
    model["_CLASS_"] = MODEL_CONFIG_CLASS

    blocks = [model["block"], *(model.get("block_overrides") or {}).values()]
    for block in blocks:
        _adapt_legacy_ep_config(block, tbo=bool(model.get("two_batch_overlap", False)))

        attention_norm = block.pop("attention_norm", None)
        feed_forward_norm = block.pop("feed_forward_norm", None)
        recorded_layer_norm = block.get("layer_norm")

        if attention_norm is not None or feed_forward_norm is not None:
            if attention_norm != feed_forward_norm:
                raise ValueError(
                    "moe-v2-core's single layer_norm cannot represent unequal "
                    "attention/feed-forward norm configs"
                )
            if recorded_layer_norm is not None and recorded_layer_norm != attention_norm:
                raise ValueError("Recorded layer_norm conflicts with the split norm configs")
            block["layer_norm"] = attention_norm
        elif recorded_layer_norm is None:
            raise ValueError("Block has neither layer_norm nor split norm configs")

        mixer = block["sequence_mixer"]
        mixer.pop("type", None)
        if mixer.get("_CLASS_", "").endswith("AttentionConfig"):
            # The source branch recorded this as a concrete false default; the
            # upstream API made it optional, where None also disables it.
            mixer.setdefault("mxfp8_save_qkv_for_backward", False)
        d_attn = mixer.pop("d_attn", None)
        if d_attn is not None:
            d_attn = int(d_attn)
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

        scaling = (mixer.get("rope") or {}).get("scaling")
        if scaling is not None and "truncate" in scaling:
            # This upstream revision always floors/ceils the YaRN correction
            # bounds, which is exactly the later branch's truncate=true mode.
            if not scaling.pop("truncate"):
                raise ValueError("moe-v2-core cannot represent YaRN truncate=false")

    return model


def adapt_train_module_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove inert Muon controls after proving this recipe selected AdamW."""

    train_module = copy.deepcopy(dict(payload))
    optim = train_module["optim"]
    for override in optim.get("group_overrides") or []:
        use_muon = override.get("opts", {}).pop("use_muon", False)
        if use_muon:
            raise ValueError("Cannot migrate a parameter group with use_muon=true")

    for field in (
        "muon_momentum",
        "muon_nesterov",
        "muon_ns_coefficients",
        "muon_eps",
        "muon_ns_steps",
        "muon_adjust_lr_fn",
    ):
        optim.pop(field, None)
    return train_module


def build_model_config_from_payload(payload: Mapping[str, Any]) -> OLMoDDPModelConfig:
    """Build and audit an upstream model config from a recorded model payload."""

    model = OLMoDDPModelConfig.from_dict(adapt_model_payload(payload))
    model.validate()
    for layer_idx, block in enumerate(model.resolved_block_configs):
        mixer = block.sequence_mixer
        if isinstance(mixer, AttentionConfig) and mixer.head_dim != 128:
            raise ValueError(f"Layer {layer_idx} has unexpected attention head_dim={mixer.head_dim}")
    return model


def build_model_config_from_recorded(path: Path) -> OLMoDDPModelConfig:
    """Load an experiment/checkpoint config and build its adapted model config."""

    recorded = load_recorded_config(path)
    return build_model_config_from_payload(recorded["model"])
