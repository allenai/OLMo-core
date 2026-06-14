"""
Convert Qwen3 and Qwen3.5/3.6 MoE Hugging Face checkpoints to OLMo-core.

For Qwen3.5/3.6 this intentionally converts only ``model.language_model`` and
``lm_head`` weights. Qwen's vision tower and MTP module are skipped because the
OLMo pretraining path here is text-only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import save_model_and_optim_state
from olmo_core.io import copy_file, join_path
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.moe.v2.qwen import build_qwen3_moe_config_from_hf_config
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def _is_local_dir(path_or_repo: str) -> bool:
    return Path(path_or_repo).is_dir()


def _load_hf_config_dict(model_name_or_path: str, revision: str) -> dict[str, Any]:
    cfg = AutoConfig.from_pretrained(model_name_or_path, revision=revision, trust_remote_code=False)
    return cfg.to_dict()


def _tokenizer_config_from_qwen_hf(hf_model: str, hf_config: Mapping[str, Any]) -> TokenizerConfig:
    text_config = hf_config.get("text_config", hf_config)
    eos_token_id = text_config.get("eos_token_id")
    if eos_token_id is None:
        raise ValueError("Qwen text config is missing eos_token_id")
    pad_token_id = text_config.get("pad_token_id")
    if pad_token_id is None:
        pad_token_id = eos_token_id
    return TokenizerConfig(
        vocab_size=int(text_config["vocab_size"]),
        eos_token_id=int(eos_token_id),
        pad_token_id=int(pad_token_id),
        bos_token_id=text_config.get("bos_token_id"),
        identifier=hf_model,
    )


def _snapshot_or_path(
    model_name_or_path: str,
    revision: str,
    cache_dir: str | None,
    *,
    required_shards: set[str] | None = None,
) -> Path:
    if _is_local_dir(model_name_or_path):
        return Path(model_name_or_path)
    weight_patterns = sorted(required_shards) if required_shards is not None else ["*.safetensors"]
    return Path(
        snapshot_download(
            model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                *weight_patterns,
                "tokenizer*",
                "*.json",
                "*.tiktoken",
                "*.model",
                "vocab*",
                "merges.txt",
            ],
        )
    )


def _index_path(model_name_or_path: str, revision: str, cache_dir: str | None) -> Path:
    if _is_local_dir(model_name_or_path):
        return Path(model_name_or_path) / "model.safetensors.index.json"
    return Path(
        hf_hub_download(
            model_name_or_path,
            "model.safetensors.index.json",
            revision=revision,
            cache_dir=cache_dir,
        )
    )


class SafeTensorKeyLoader:
    def __init__(self, checkpoint_dir: Path, weight_map: Mapping[str, str]):
        self.checkpoint_dir = checkpoint_dir
        self.weight_map = weight_map
        self._current_file: str | None = None
        self._handle: Any | None = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.__exit__(None, None, None)
            self._handle = None
            self._current_file = None

    def get(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        if shard != self._current_file:
            self.close()
            self._handle = safe_open(self.checkpoint_dir / shard, framework="pt", device="cpu")
            self._handle.__enter__()
            self._current_file = shard
        assert self._handle is not None
        return self._handle.get_tensor(key)


def _copy_param(target: torch.Tensor, value: torch.Tensor, key: str) -> None:
    if target.shape != value.shape:
        raise ValueError(f"{key}: target shape {tuple(target.shape)} != source shape {tuple(value.shape)}")
    target.copy_(value.to(device=target.device, dtype=target.dtype))


def _copy_flat_param(target: torch.Tensor, value: torch.Tensor, key: str) -> None:
    if target.numel() != value.numel():
        raise ValueError(f"{key}: target numel {target.numel()} != source numel {value.numel()}")
    target.copy_(value.reshape(target.shape).to(device=target.device, dtype=target.dtype))


def _is_qwen35_text_config(text_config: Mapping[str, Any]) -> bool:
    return "linear_num_key_heads" in text_config


def _num_experts(text_config: Mapping[str, Any]) -> int:
    for key in ("num_experts", "num_local_experts", "num_routed_experts", "n_routed_experts"):
        if key in text_config:
            return int(text_config[key])
    raise KeyError("Qwen MoE config is missing num_experts/num_local_experts")


def _make_config_overrides_for_layer_limit(
    text_config: Mapping[str, Any],
    max_layers: int | None,
) -> dict[str, Any]:
    if max_layers is None:
        return {}
    total_layers = int(text_config["num_hidden_layers"])
    if max_layers < 1 or max_layers > total_layers:
        raise ValueError(f"--max-layers must be in [1, {total_layers}], got {max_layers}")
    overrides: dict[str, Any] = {"n_layers": max_layers}
    if "layer_types" in text_config:
        overrides["layer_types"] = tuple(text_config["layer_types"][:max_layers])
    else:
        overrides["layer_types"] = ("full_attention",) * max_layers
    return overrides


def _assign_qwen35_text_weights(
    model_state: dict[str, torch.Tensor],
    loader: SafeTensorKeyLoader,
    text_config: Mapping[str, Any],
    *,
    n_layers: int,
    dry_run: bool = False,
) -> tuple[set[str], set[str]]:
    required_hf_keys: set[str] = set()
    assigned_olmo_keys: set[str] = set()

    def require(hf_key: str) -> torch.Tensor | None:
        required_hf_keys.add(hf_key)
        if dry_run:
            return None
        return loader.get(hf_key)

    def assign(olmo_key: str, value: torch.Tensor | None) -> None:
        assigned_olmo_keys.add(olmo_key)
        if dry_run:
            return
        assert value is not None
        _copy_param(model_state[olmo_key], value, olmo_key)

    def assign_flat(olmo_key: str, value: torch.Tensor | None) -> None:
        assigned_olmo_keys.add(olmo_key)
        if dry_run:
            return
        assert value is not None
        _copy_flat_param(model_state[olmo_key], value, olmo_key)

    assign("embeddings.weight", require("model.language_model.embed_tokens.weight"))
    assign("lm_head.norm.weight", require("model.language_model.norm.weight"))
    assign("lm_head.w_out.weight", require("lm_head.weight"))

    layer_types = list(text_config["layer_types"])
    key_dim = int(text_config["linear_num_key_heads"]) * int(text_config["linear_key_head_dim"])
    value_dim = int(text_config["linear_num_value_heads"]) * int(text_config["linear_value_head_dim"])
    hidden_size = int(text_config["hidden_size"])
    moe_hidden = int(text_config["moe_intermediate_size"])
    shared_hidden = int(text_config["shared_expert_intermediate_size"])

    for layer_idx in range(n_layers):
        prefix_hf = f"model.language_model.layers.{layer_idx}"
        prefix_olmo = f"blocks.{layer_idx}"

        assign(f"{prefix_olmo}.attention_norm.weight", require(f"{prefix_hf}.input_layernorm.weight"))
        assign(
            f"{prefix_olmo}.feed_forward_norm.weight",
            require(f"{prefix_hf}.post_attention_layernorm.weight"),
        )

        if layer_types[layer_idx] == "linear_attention":
            qkv = require(f"{prefix_hf}.linear_attn.in_proj_qkv.weight")
            conv = require(f"{prefix_hf}.linear_attn.conv1d.weight")
            if not dry_run:
                assert qkv is not None and conv is not None
                q, k, v = qkv.split([key_dim, key_dim, value_dim], dim=0)
                q_conv, k_conv, v_conv = conv.split([key_dim, key_dim, value_dim], dim=0)
                assign(f"{prefix_olmo}.attention.w_q.weight", q)
                assign(f"{prefix_olmo}.attention.w_k.weight", k)
                assign(f"{prefix_olmo}.attention.w_v.weight", v)
                assign(f"{prefix_olmo}.attention.q_conv1d.weight", q_conv)
                assign(f"{prefix_olmo}.attention.k_conv1d.weight", k_conv)
                assign(f"{prefix_olmo}.attention.v_conv1d.weight", v_conv)
            else:
                for suffix in (
                    "w_q.weight",
                    "w_k.weight",
                    "w_v.weight",
                    "q_conv1d.weight",
                    "k_conv1d.weight",
                    "v_conv1d.weight",
                ):
                    assigned_olmo_keys.add(f"{prefix_olmo}.attention.{suffix}")

            assign(f"{prefix_olmo}.attention.w_a.weight", require(f"{prefix_hf}.linear_attn.in_proj_a.weight"))
            assign(f"{prefix_olmo}.attention.w_b.weight", require(f"{prefix_hf}.linear_attn.in_proj_b.weight"))
            assign(f"{prefix_olmo}.attention.w_g.weight", require(f"{prefix_hf}.linear_attn.in_proj_z.weight"))
            assign(f"{prefix_olmo}.attention.w_out.weight", require(f"{prefix_hf}.linear_attn.out_proj.weight"))
            assign(f"{prefix_olmo}.attention.o_norm.weight", require(f"{prefix_hf}.linear_attn.norm.weight"))
            assign(f"{prefix_olmo}.attention.A_log", require(f"{prefix_hf}.linear_attn.A_log"))
            assign(f"{prefix_olmo}.attention.dt_bias", require(f"{prefix_hf}.linear_attn.dt_bias"))
        elif layer_types[layer_idx] == "full_attention":
            q_proj = require(f"{prefix_hf}.self_attn.q_proj.weight")
            if not dry_run:
                assert q_proj is not None
                head_dim = int(text_config["head_dim"])
                num_heads = int(text_config["num_attention_heads"])
                q_proj_by_head = q_proj.view(num_heads, 2 * head_dim, hidden_size)
                q = q_proj_by_head[:, :head_dim, :].reshape(num_heads * head_dim, hidden_size)
                gate = q_proj_by_head[:, head_dim:, :].reshape(num_heads * head_dim, hidden_size)
                assign(f"{prefix_olmo}.attention.w_q.weight", q)
                assign(f"{prefix_olmo}.attention.w_g.weight", gate)
            else:
                assigned_olmo_keys.add(f"{prefix_olmo}.attention.w_q.weight")
                assigned_olmo_keys.add(f"{prefix_olmo}.attention.w_g.weight")
            assign(f"{prefix_olmo}.attention.w_k.weight", require(f"{prefix_hf}.self_attn.k_proj.weight"))
            assign(f"{prefix_olmo}.attention.w_v.weight", require(f"{prefix_hf}.self_attn.v_proj.weight"))
            assign(f"{prefix_olmo}.attention.w_out.weight", require(f"{prefix_hf}.self_attn.o_proj.weight"))
            assign(f"{prefix_olmo}.attention.q_norm.weight", require(f"{prefix_hf}.self_attn.q_norm.weight"))
            assign(f"{prefix_olmo}.attention.k_norm.weight", require(f"{prefix_hf}.self_attn.k_norm.weight"))
        else:
            raise ValueError(f"Unsupported layer type at {layer_idx}: {layer_types[layer_idx]!r}")

        assign_flat(
            f"{prefix_olmo}.routed_experts_router.weight",
            require(f"{prefix_hf}.mlp.gate.weight"),
        )
        gate_up = require(f"{prefix_hf}.mlp.experts.gate_up_proj")
        if not dry_run:
            assert gate_up is not None
            gate, up = gate_up.split(moe_hidden, dim=1)
            assign(f"{prefix_olmo}.routed_experts.w_up_gate", torch.cat((up, gate), dim=1))
        else:
            assigned_olmo_keys.add(f"{prefix_olmo}.routed_experts.w_up_gate")

        down = require(f"{prefix_hf}.mlp.experts.down_proj")
        if not dry_run:
            assert down is not None
            assign(f"{prefix_olmo}.routed_experts.w_down", down.transpose(1, 2).contiguous())
        else:
            assigned_olmo_keys.add(f"{prefix_olmo}.routed_experts.w_down")

        shared_up = require(f"{prefix_hf}.mlp.shared_expert.up_proj.weight")
        shared_gate = require(f"{prefix_hf}.mlp.shared_expert.gate_proj.weight")
        if not dry_run:
            assert shared_up is not None and shared_gate is not None
            shared_up_gate = torch.cat((shared_up.t(), shared_gate.t()), dim=1).contiguous()
            if shared_up_gate.shape != (hidden_size, 2 * shared_hidden):
                raise ValueError(
                    f"{prefix_hf}.mlp.shared_expert up/gate produced {shared_up_gate.shape}"
                )
            assign(f"{prefix_olmo}.shared_experts.w_up_gate", shared_up_gate)
        else:
            assigned_olmo_keys.add(f"{prefix_olmo}.shared_experts.w_up_gate")

        shared_down = require(f"{prefix_hf}.mlp.shared_expert.down_proj.weight")
        if not dry_run:
            assert shared_down is not None
            assign(
                f"{prefix_olmo}.shared_experts.w_down",
                shared_down.t().contiguous().unsqueeze(0),
            )
        else:
            assigned_olmo_keys.add(f"{prefix_olmo}.shared_experts.w_down")

        assign_flat(
            f"{prefix_olmo}.shared_experts_router.weight",
            require(f"{prefix_hf}.mlp.shared_expert_gate.weight"),
        )

    return required_hf_keys, assigned_olmo_keys


def _assign_qwen3_moe_weights(
    model_state: dict[str, torch.Tensor],
    loader: SafeTensorKeyLoader,
    text_config: Mapping[str, Any],
    *,
    n_layers: int,
    dry_run: bool = False,
) -> tuple[set[str], set[str]]:
    required_hf_keys: set[str] = set()
    assigned_olmo_keys: set[str] = set()

    def require(hf_key: str) -> torch.Tensor | None:
        required_hf_keys.add(hf_key)
        if dry_run:
            return None
        return loader.get(hf_key)

    def assign(olmo_key: str, value: torch.Tensor | None) -> None:
        assigned_olmo_keys.add(olmo_key)
        if dry_run:
            return
        assert value is not None
        _copy_param(model_state[olmo_key], value, olmo_key)

    def assign_flat(olmo_key: str, value: torch.Tensor | None) -> None:
        assigned_olmo_keys.add(olmo_key)
        if dry_run:
            return
        assert value is not None
        _copy_flat_param(model_state[olmo_key], value, olmo_key)

    def assign_slice(olmo_key: str, value: torch.Tensor | None, target: torch.Tensor) -> None:
        assigned_olmo_keys.add(olmo_key)
        if dry_run:
            return
        assert value is not None
        _copy_param(target, value, olmo_key)

    assign("embeddings.weight", require("model.embed_tokens.weight"))
    assign("lm_head.norm.weight", require("model.norm.weight"))
    assign("lm_head.w_out.weight", require("lm_head.weight"))

    num_experts = _num_experts(text_config)
    moe_hidden = int(text_config["moe_intermediate_size"])

    for layer_idx in range(n_layers):
        prefix_hf = f"model.layers.{layer_idx}"
        prefix_olmo = f"blocks.{layer_idx}"

        assign(f"{prefix_olmo}.attention_norm.weight", require(f"{prefix_hf}.input_layernorm.weight"))
        assign(
            f"{prefix_olmo}.feed_forward_norm.weight",
            require(f"{prefix_hf}.post_attention_layernorm.weight"),
        )
        assign(f"{prefix_olmo}.attention.w_q.weight", require(f"{prefix_hf}.self_attn.q_proj.weight"))
        assign(f"{prefix_olmo}.attention.w_k.weight", require(f"{prefix_hf}.self_attn.k_proj.weight"))
        assign(f"{prefix_olmo}.attention.w_v.weight", require(f"{prefix_hf}.self_attn.v_proj.weight"))
        assign(f"{prefix_olmo}.attention.w_out.weight", require(f"{prefix_hf}.self_attn.o_proj.weight"))
        assign(f"{prefix_olmo}.attention.q_norm.weight", require(f"{prefix_hf}.self_attn.q_norm.weight"))
        assign(f"{prefix_olmo}.attention.k_norm.weight", require(f"{prefix_hf}.self_attn.k_norm.weight"))
        assign_flat(
            f"{prefix_olmo}.routed_experts_router.weight",
            require(f"{prefix_hf}.mlp.gate.weight"),
        )

        up_gate_key = f"{prefix_olmo}.routed_experts.w_up_gate"
        down_key = f"{prefix_olmo}.routed_experts.w_down"
        up_gate_target = model_state[up_gate_key]
        down_target = model_state[down_key]
        for expert_idx in range(num_experts):
            expert_prefix = f"{prefix_hf}.mlp.experts.{expert_idx}"
            up = require(f"{expert_prefix}.up_proj.weight")
            gate = require(f"{expert_prefix}.gate_proj.weight")
            down = require(f"{expert_prefix}.down_proj.weight")
            if dry_run:
                continue
            assert up is not None and gate is not None and down is not None
            assign_slice(
                up_gate_key,
                up,
                up_gate_target[expert_idx, :moe_hidden, :],
            )
            assign_slice(
                up_gate_key,
                gate,
                up_gate_target[expert_idx, moe_hidden:, :],
            )
            assign_slice(
                down_key,
                down.t().contiguous(),
                down_target[expert_idx],
            )
        if dry_run:
            assigned_olmo_keys.add(up_gate_key)
            assigned_olmo_keys.add(down_key)

    return required_hf_keys, assigned_olmo_keys


def _assign_qwen_weights(
    model_state: dict[str, torch.Tensor],
    loader: SafeTensorKeyLoader,
    text_config: Mapping[str, Any],
    *,
    n_layers: int,
    dry_run: bool = False,
) -> tuple[set[str], set[str]]:
    if _is_qwen35_text_config(text_config):
        return _assign_qwen35_text_weights(
            model_state,
            loader,
            text_config,
            n_layers=n_layers,
            dry_run=dry_run,
        )
    return _assign_qwen3_moe_weights(
        model_state,
        loader,
        text_config,
        n_layers=n_layers,
        dry_run=dry_run,
    )


def convert_qwen3_moe_hf_to_olmo(
    *,
    hf_model: str,
    output_path: str | Path,
    revision: str = "main",
    cache_dir: str | None = "/workspace/checkpoint",
    dtype: DType = DType.bfloat16,
    device: torch.device = torch.device("cpu"),
    attention_backend: AttentionBackendName = AttentionBackendName.flash_4,
    max_layers: int | None = None,
    dry_run: bool = False,
    save_overwrite: bool = False,
) -> None:
    hf_config = _load_hf_config_dict(hf_model, revision)
    text_config = hf_config.get("text_config", hf_config)
    layer_overrides = _make_config_overrides_for_layer_limit(text_config, max_layers)
    model_config = build_qwen3_moe_config_from_hf_config(
        hf_config,
        dtype=dtype,
        attention_backend=attention_backend,
        compile_friendly_recompute=True,
        **layer_overrides,
    )

    index_file = _index_path(hf_model, revision, cache_dir)
    index = json.loads(index_file.read_text())
    weight_map = index["weight_map"]

    log.info("Building OLMo model skeleton")
    model = model_config.build(init_device="meta")
    if not dry_run:
        model.to_empty(device=device)
    model_state = {k: v for k, v in model.state_dict().items()}

    missing_olmo = set(model_state)
    loader = SafeTensorKeyLoader(index_file.parent, weight_map)
    try:
        required_hf, assigned_olmo = _assign_qwen_weights(
            model_state,
            loader,
            text_config,
            n_layers=model_config.n_layers,
            dry_run=True,
        )
    finally:
        loader.close()

    missing_hf = sorted(required_hf - set(weight_map))
    missing_olmo -= assigned_olmo
    if missing_hf:
        raise RuntimeError(f"Missing required HF weights: {missing_hf[:20]}")
    if missing_olmo:
        raise RuntimeError(f"Unassigned OLMo weights: {sorted(missing_olmo)[:20]}")

    output_path = Path(output_path)
    if dry_run:
        log.info("Dry run successful: %d HF keys cover %d OLMo tensors", len(required_hf), len(assigned_olmo))
        return

    required_shards = {weight_map[key] for key in required_hf}
    checkpoint_dir = _snapshot_or_path(
        hf_model,
        revision,
        cache_dir,
        required_shards=required_shards,
    )
    loader = SafeTensorKeyLoader(checkpoint_dir, weight_map)
    try:
        required_hf_actual, assigned_olmo_actual = _assign_qwen_weights(
            model_state,
            loader,
            text_config,
            n_layers=model_config.n_layers,
            dry_run=False,
        )
    finally:
        loader.close()
    if required_hf_actual != required_hf or assigned_olmo_actual != assigned_olmo:
        raise RuntimeError("Qwen conversion dry-run and materialized assignment coverage diverged")

    model_and_optim_dir = join_path(output_path, "model_and_optim")
    log.info("Saving converted OLMo-core checkpoint to %s", model_and_optim_dir)
    save_model_and_optim_state(model_and_optim_dir, model, save_overwrite=save_overwrite)

    tokenizer_config = _tokenizer_config_from_qwen_hf(hf_model, hf_config)
    converted = (
        "model.language_model text tower + lm_head"
        if _is_qwen35_text_config(text_config)
        else "model text tower + lm_head"
    )
    skipped = ["model.visual", "mtp"] if _is_qwen35_text_config(text_config) else []
    experiment_config_dict = {
        "model": model_config.as_config_dict(),
        "dataset": {
            "tokenizer": tokenizer_config.as_config_dict(),
        },
        "conversion": {
            "source": hf_model,
            "revision": revision,
            "converted": converted,
            "max_layers": max_layers,
            "skipped": skipped,
        },
    }
    config_path = join_path(output_path, "config.json")
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        json.dump(experiment_config_dict, temp_file, indent=2)
        temp_file.flush()
        if hasattr(os, "fdatasync"):
            os.fdatasync(temp_file)
        temp_name = temp_file.name
    copy_file(temp_name, config_path, save_overwrite=True)
    Path(temp_name).unlink(missing_ok=True)
    log.info("Wrote partial experiment config to %s", config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--cache-dir", default="/workspace/checkpoint")
    parser.add_argument("--dtype", choices=[d.value for d in DType], default=DType.bfloat16.value)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--attention-backend",
        choices=[b.value for b in AttentionBackendName],
        default=AttentionBackendName.flash_4.value if torch.cuda.is_available() else AttentionBackendName.torch.value,
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Convert only the first N transformer layers, useful for OOM-safe smoke tests.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-overwrite", action="store_true")
    args = parser.parse_args()

    prepare_cli_environment()
    convert_qwen3_moe_hf_to_olmo(
        hf_model=args.hf_model,
        output_path=args.output_path,
        revision=args.revision,
        cache_dir=args.cache_dir,
        dtype=DType(args.dtype),
        device=torch.device(args.device),
        attention_backend=AttentionBackendName(args.attention_backend),
        max_layers=args.max_layers,
        dry_run=args.dry_run,
        save_overwrite=args.save_overwrite,
    )


if __name__ == "__main__":
    main()
