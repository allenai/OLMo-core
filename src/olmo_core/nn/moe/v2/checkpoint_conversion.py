"""Checkpoint conversion helpers for legacy OLMoE v2 models.

The legacy OLMoE v2 stack represented dense layers with ``FeedForward`` while
the OLMoDDP stack represents them as shared-only MoE blocks.  The rest of the
model parameter layout is intentionally kept unchanged by this converter.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

import torch

from olmo_core.nn.transformer.config import OLMoDDPModelConfig


MODEL_CONFIG_CLASS = "olmo_core.nn.transformer.config.OLMoDDPModelConfig"
BLOCK_CONFIG_CLASS = "olmo_core.nn.ddp.block.OLMoDDPTransformerBlockConfig"
EP_CONFIG_CLASS = "olmo_core.nn.moe.v2.ep_config.ExpertParallelConfig"
SHARED_EXPERTS_CONFIG_CLASS = "olmo_core.nn.moe.v2.shared_experts.SharedExpertsConfig"
TRAIN_MODULE_CONFIG_CLASS = (
    "olmo_core.train.train_module.transformer.config.OLMoDDPTrainModuleConfig"
)
OPTIM_CONFIG_CLASS = "olmo_core.optim.moe_optimizer.OLMoDDPOptimizerConfig"


@dataclass(frozen=True)
class DenseLayerSpec:
    layer_idx: int
    d_model: int
    hidden_size: int


def _block_for_layer(model_config: Mapping[str, Any], layer_idx: int) -> Mapping[str, Any]:
    overrides = model_config.get("block_overrides") or {}
    return overrides.get(str(layer_idx), overrides.get(layer_idx, model_config["block"]))


def get_legacy_dense_layer_specs(config: Mapping[str, Any]) -> list[DenseLayerSpec]:
    """Find dense ``FeedForward`` layers in a recorded legacy experiment config."""

    model_config = config["model"]
    d_model = int(model_config["d_model"])
    specs: list[DenseLayerSpec] = []
    for layer_idx in range(int(model_config["n_layers"])):
        block_config = _block_for_layer(model_config, layer_idx)
        feed_forward = block_config.get("feed_forward")
        if feed_forward is not None:
            specs.append(
                DenseLayerSpec(
                    layer_idx=layer_idx,
                    d_model=d_model,
                    hidden_size=int(feed_forward["hidden_size"]),
                )
            )
    return specs


def _convert_legacy_ep_config(block_config: Dict[str, Any], *, tbo: bool) -> None:
    if "ep" in block_config:
        legacy_ep_keys = {key for key in block_config if key.startswith("ep_no_sync")}
        if "checkpoint_combined_ep_tbo" in block_config:
            legacy_ep_keys.add("checkpoint_combined_ep_tbo")
        if legacy_ep_keys:
            raise ValueError(
                "Block config mixes new 'ep' with legacy options: "
                + ", ".join(sorted(legacy_ep_keys))
            )
        ep = block_config["ep"]
        if ep is not None:
            ep["_CLASS_"] = EP_CONFIG_CLASS
        return

    legacy_ep_keys = {key for key in block_config if key.startswith("ep_no_sync")}
    if not legacy_ep_keys and block_config.get("routed_experts") is None:
        return

    supported_keys = {
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
    }
    unsupported_keys = legacy_ep_keys - supported_keys
    if unsupported_keys:
        raise ValueError(
            "Unsupported legacy expert-parallel options: " + ", ".join(sorted(unsupported_keys))
        )

    if block_config.get("ep_no_sync_use_2d_all_to_all", False):
        raise ValueError("Legacy 2D all-to-all EP does not have an OLMoDDP conversion")

    no_sync = bool(block_config.get("ep_no_sync", False))
    rowwise = bool(block_config.get("ep_no_sync_use_rowwise_all_to_all", False))
    if rowwise and not no_sync:
        raise ValueError("Rowwise all-to-all was set while legacy ep_no_sync was disabled")

    if rowwise:
        path = "rowwise_nvshmem"
    elif no_sync:
        path = "no_sync_1d"
    else:
        path = "sync_1d"

    ep: Dict[str, Any] = {
        "path": path,
        "schedule": "tbo" if tbo else "normal",
        "capacity_factor": float(block_config.get("ep_no_sync_capacity_factor", 1.25)),
        "shared_slots": int(block_config.get("ep_no_sync_shared_slots", 1)),
        "major_align": int(block_config.get("ep_no_sync_major_align", 1)),
        "rowwise_nblocks": int(block_config.get("ep_no_sync_rowwise_nblocks", 32)),
        "share_dispatch_out": bool(block_config.get("ep_no_sync_share_dispatch_out", False)),
        "share_combine_out": bool(block_config.get("ep_no_sync_share_combine_out", False)),
        "restore_unpermute_backend": block_config.get(
            "ep_no_sync_restore_unpermute_backend", "te_fused"
        ),
        "checkpoint_tbo": bool(block_config.get("checkpoint_combined_ep_tbo", False)),
        "_CLASS_": EP_CONFIG_CLASS,
    }
    if tbo and path != "rowwise_nvshmem":
        raise ValueError("Two-batch overlap requires legacy rowwise no-sync EP")
    block_config["ep"] = ep

    for key in supported_keys:
        block_config.pop(key, None)
    block_config.pop("checkpoint_combined_ep_tbo", None)


def _convert_dense_block(block_config: Mapping[str, Any], *, d_model: int) -> Dict[str, Any]:
    feed_forward = block_config.get("feed_forward")
    if feed_forward is None:
        raise ValueError("Expected a legacy dense block with feed_forward")
    if block_config.get("name") != "peri_norm":
        raise ValueError(
            f"Only legacy peri_norm dense blocks are supported, got {block_config.get('name')!r}"
        )
    if feed_forward.get("bias", True):
        raise ValueError("Dense checkpoint conversion currently requires bias=False")
    if feed_forward.get("activation", "silu") != "silu":
        raise ValueError("Dense checkpoint conversion currently requires SwiGLU/SILU")
    if block_config.get("dropout", 0.0) not in (None, 0.0):
        raise ValueError("Dense checkpoint conversion currently requires dropout=0")
    for alpha_name in ("attention_residual_alpha", "feed_forward_residual_alpha"):
        if block_config.get(alpha_name, 1.0) not in (None, 1.0):
            raise ValueError(f"Dense checkpoint conversion currently requires {alpha_name}=1")
    for norm_name in ("attention_norm", "feed_forward_norm"):
        if block_config[norm_name].get("bias", False):
            raise ValueError("Dense checkpoint conversion does not support norm bias tensors")

    dtype = feed_forward.get("dtype")
    if dtype is None:
        dtype = block_config["sequence_mixer"].get("dtype", "float32")

    converted: Dict[str, Any] = {
        "sequence_mixer": copy.deepcopy(block_config["sequence_mixer"]),
        "attention_norm": copy.deepcopy(block_config["attention_norm"]),
        "feed_forward_norm": copy.deepcopy(block_config["feed_forward_norm"]),
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
    return converted


def _convert_moe_block(block_config: Mapping[str, Any], *, tbo: bool) -> Dict[str, Any]:
    converted = copy.deepcopy(dict(block_config))
    converted["name"] = "moe_fused_v2"
    converted["_CLASS_"] = BLOCK_CONFIG_CLASS
    _convert_legacy_ep_config(converted, tbo=tbo)
    return converted


def convert_legacy_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a target config that uses canonical OLMoDDP config classes."""

    converted = copy.deepcopy(dict(config))
    model_config = converted["model"]
    d_model = int(model_config["d_model"])
    tbo = bool(model_config.get("two_batch_overlap", False))

    block = model_config["block"]
    if block.get("feed_forward") is not None:
        model_config["block"] = _convert_dense_block(block, d_model=d_model)
    else:
        model_config["block"] = _convert_moe_block(block, tbo=tbo)

    overrides = model_config.get("block_overrides") or {}
    converted_overrides: Dict[str, Any] = {}
    for layer_idx, override in overrides.items():
        if override.get("feed_forward") is not None:
            converted_override = _convert_dense_block(override, d_model=d_model)
        else:
            converted_override = _convert_moe_block(override, tbo=tbo)
        converted_overrides[str(layer_idx)] = converted_override
    model_config["block_overrides"] = converted_overrides or None
    model_config["_CLASS_"] = MODEL_CONFIG_CLASS

    train_module = converted.get("train_module")
    if isinstance(train_module, dict):
        train_module["_CLASS_"] = TRAIN_MODULE_CONFIG_CLASS
        optim = train_module.get("optim")
        if isinstance(optim, dict):
            optim["_CLASS_"] = OPTIM_CONFIG_CLASS

    return converted


def _take_tensor(
    state_dict: Dict[str, torch.Tensor], key: str, expected_numel: int
) -> torch.Tensor:
    try:
        tensor = state_dict.pop(key)
    except KeyError as exc:
        raise KeyError(f"Missing required legacy tensor: {key}") from exc
    if tensor.numel() != expected_numel:
        raise ValueError(
            f"Unexpected size for {key}: expected {expected_numel:,} elements, "
            f"got {tensor.numel():,}"
        )
    return tensor


def convert_legacy_model_state(
    state_dict: Mapping[str, torch.Tensor], dense_layers: Iterable[DenseLayerSpec]
) -> Dict[str, torch.Tensor]:
    """Convert legacy optimizer-main tensors into the OLMoDDP parameter layout."""

    output = dict(state_dict)
    for spec in dense_layers:
        prefix = f"module.blocks.{spec.layer_idx}"
        d_model, hidden_size = spec.d_model, spec.hidden_size

        w1 = _take_tensor(
            output,
            f"{prefix}.feed_forward.w1.weight.main",
            hidden_size * d_model,
        ).view(hidden_size, d_model)
        w2 = _take_tensor(
            output,
            f"{prefix}.feed_forward.w2.weight.main",
            d_model * hidden_size,
        ).view(d_model, hidden_size)
        w3 = _take_tensor(
            output,
            f"{prefix}.feed_forward.w3.weight.main",
            hidden_size * d_model,
        ).view(hidden_size, d_model)

        # Legacy FeedForward computes silu(x @ w1.T) * (x @ w3.T).
        # SharedExperts stores a column-packed [up | gate] matrix.
        output[f"{prefix}.shared_experts.w_up_gate.main"] = (
            torch.cat((w3.T, w1.T), dim=1).contiguous().view(-1)
        )
        output[f"{prefix}.shared_experts.w_down.main"] = w2.T.contiguous().unsqueeze(0).view(-1)

        norm_mapping = {
            "attention_norm": "attention_input_norm",
            "post_attention_norm": "attention_norm",
            "feed_forward_norm": "feed_forward_input_norm",
            "post_feed_forward_norm": "feed_forward_norm",
        }
        norm_tensors: Dict[str, torch.Tensor] = {}
        for source_name, target_name in norm_mapping.items():
            norm_tensors[target_name] = _take_tensor(
                output,
                f"{prefix}.{source_name}.weight.main",
                d_model,
            )
        for target_name, tensor in norm_tensors.items():
            output[f"{prefix}.{target_name}.weight.main"] = tensor

    return output


def expected_olmo_ddp_main_tensors(
    converted_config: Mapping[str, Any],
) -> Dict[str, tuple[torch.Size, torch.dtype]]:
    """Build the target model on ``meta`` and return its optimizer-main schema."""

    model_config = OLMoDDPModelConfig.from_dict(dict(converted_config["model"]))
    model_config.validate()
    model = model_config.build(init_device="meta")
    return {
        # OLMoDDPOptimizer always stores flattened master parameters in fp32,
        # even when the model parameters themselves are bf16.
        f"module.{name}.main": (parameter.shape, torch.float32)
        for name, parameter in model.named_parameters()
    }


def validate_main_tensor_schema(
    state_dict: Mapping[str, torch.Tensor],
    expected: Mapping[str, tuple[torch.Size, torch.dtype]],
) -> None:
    actual_keys = set(state_dict)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "Converted model tensor keys do not match OLMoDDP model schema. "
            f"Missing={missing[:20]}, unexpected={unexpected[:20]}"
        )

    errors: list[str] = []
    for key, tensor in state_dict.items():
        expected_shape, expected_dtype = expected[key]
        expected_numel = expected_shape.numel()
        if tensor.numel() != expected_numel:
            errors.append(
                f"{key}: expected {tuple(expected_shape)} ({expected_numel} elements), "
                f"got {tuple(tensor.shape)} ({tensor.numel()} elements)"
            )
        if tensor.dtype != expected_dtype:
            errors.append(f"{key}: expected dtype {expected_dtype}, got {tensor.dtype}")
    if errors:
        raise ValueError("Converted tensor schema errors:\n" + "\n".join(errors[:20]))
