"""
Convert NVIDIA Nemotron 3 Nano MoE Hugging Face checkpoints to OLMo-core.

This targets the text-only Nemotron-H backbone and language-model head.
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
from olmo_core.nn.moe.v2.nemotron import (
    NEMOTRON3_NANO_MODEL_ID,
    build_nemotron3_nano_config_from_hf_config,
)
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def _is_local_dir(path_or_repo: str) -> bool:
    return Path(path_or_repo).is_dir()


def _load_hf_config_dict(model_name_or_path: str, revision: str) -> dict[str, Any]:
    cfg = AutoConfig.from_pretrained(model_name_or_path, revision=revision, trust_remote_code=False)
    return cfg.to_dict()


def _tokenizer_config_from_hf(hf_model: str, hf_config: Mapping[str, Any]) -> TokenizerConfig:
    return TokenizerConfig(
        vocab_size=int(hf_config["vocab_size"]),
        eos_token_id=int(hf_config["eos_token_id"]),
        pad_token_id=int(hf_config["pad_token_id"]),
        bos_token_id=hf_config.get("bos_token_id"),
        identifier=hf_model,
    )


def _snapshot_or_path(model_name_or_path: str, revision: str, cache_dir: str | None) -> Path:
    if _is_local_dir(model_name_or_path):
        return Path(model_name_or_path)
    return Path(
        snapshot_download(
            model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "*.safetensors",
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


def _assign_nemotron_weights(
    model_state: dict[str, torch.Tensor],
    loader: SafeTensorKeyLoader,
    hf_config: Mapping[str, Any],
    *,
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

    assign("embeddings.weight", require("backbone.embeddings.weight"))
    assign("lm_head.norm.weight", require("backbone.norm_f.weight"))
    assign("lm_head.w_out.weight", require("lm_head.weight"))

    layer_types = list(hf_config["layers_block_type"])
    num_experts = int(hf_config["n_routed_experts"])

    for layer_idx, layer_type in enumerate(layer_types):
        prefix_hf = f"backbone.layers.{layer_idx}"
        prefix_olmo = f"blocks.{layer_idx}"
        assign(f"{prefix_olmo}.norm.weight", require(f"{prefix_hf}.norm.weight"))

        if layer_type == "mamba":
            for suffix in (
                "dt_bias",
                "A_log",
                "D",
                "conv1d.weight",
                "conv1d.bias",
                "in_proj.weight",
                "norm.weight",
                "out_proj.weight",
            ):
                assign(
                    f"{prefix_olmo}.attention.{suffix}",
                    require(f"{prefix_hf}.mixer.{suffix}"),
                )
        elif layer_type == "attention":
            for olmo_suffix, hf_suffix in (
                ("w_q.weight", "q_proj.weight"),
                ("w_k.weight", "k_proj.weight"),
                ("w_v.weight", "v_proj.weight"),
                ("w_out.weight", "o_proj.weight"),
            ):
                assign(
                    f"{prefix_olmo}.attention.{olmo_suffix}",
                    require(f"{prefix_hf}.mixer.{hf_suffix}"),
                )
        elif layer_type == "moe":
            gate_weight = require(f"{prefix_hf}.mixer.gate.weight")
            assign(
                f"{prefix_olmo}.attention.routed_experts_router.weight",
                gate_weight.reshape(-1) if gate_weight is not None else None,
            )
            assign(
                f"{prefix_olmo}.attention.routed_experts_router.score_bias",
                require(f"{prefix_hf}.mixer.gate.e_score_correction_bias"),
            )
            if dry_run:
                assigned_olmo_keys.add(f"{prefix_olmo}.attention.routed_experts.w_up_gate")
                assigned_olmo_keys.add(f"{prefix_olmo}.attention.routed_experts.w_down")
                for expert_idx in range(num_experts):
                    require(f"{prefix_hf}.mixer.experts.{expert_idx}.up_proj.weight")
                    require(f"{prefix_hf}.mixer.experts.{expert_idx}.down_proj.weight")
            else:
                up_target = model_state[f"{prefix_olmo}.attention.routed_experts.w_up_gate"]
                down_target = model_state[f"{prefix_olmo}.attention.routed_experts.w_down"]
                up_parts = []
                down_parts = []
                for expert_idx in range(num_experts):
                    up_parts.append(require(f"{prefix_hf}.mixer.experts.{expert_idx}.up_proj.weight"))
                    down_parts.append(require(f"{prefix_hf}.mixer.experts.{expert_idx}.down_proj.weight"))
                assert all(part is not None for part in up_parts)
                assert all(part is not None for part in down_parts)
                _copy_param(
                    up_target,
                    torch.stack(up_parts),
                    f"{prefix_olmo}.attention.routed_experts.w_up_gate",
                )
                _copy_param(
                    down_target,
                    torch.stack(down_parts).transpose(1, 2).contiguous(),
                    f"{prefix_olmo}.attention.routed_experts.w_down",
                )
                assigned_olmo_keys.add(f"{prefix_olmo}.attention.routed_experts.w_up_gate")
                assigned_olmo_keys.add(f"{prefix_olmo}.attention.routed_experts.w_down")

            shared_up = require(f"{prefix_hf}.mixer.shared_experts.up_proj.weight")
            assign(
                f"{prefix_olmo}.attention.shared_experts.w_up_gate",
                (
                    shared_up.transpose(0, 1).contiguous()
                    if shared_up is not None
                    else None
                ),
            )
            shared_down = require(f"{prefix_hf}.mixer.shared_experts.down_proj.weight")
            assign(
                f"{prefix_olmo}.attention.shared_experts.w_down",
                (
                    shared_down.transpose(0, 1).contiguous().unsqueeze(0)
                    if shared_down is not None
                    else None
                ),
            )
        else:
            raise ValueError(f"Unsupported Nemotron layer type: {layer_type!r}")

    return required_hf_keys, assigned_olmo_keys


def convert(
    *,
    hf_model: str,
    revision: str,
    output_path: str,
    cache_dir: str | None,
    dtype: DType,
    device: str,
    attention_backend: AttentionBackendName,
    dry_run: bool,
    save_overwrite: bool,
) -> None:
    hf_config = _load_hf_config_dict(hf_model, revision)
    model_config = build_nemotron3_nano_config_from_hf_config(
        hf_config,
        dtype=dtype,
        attention_backend=attention_backend,
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
    if dry_run:
        loader = SafeTensorKeyLoader(index_file.parent, weight_map)
    else:
        checkpoint_dir = _snapshot_or_path(hf_model, revision, cache_dir)
        loader = SafeTensorKeyLoader(checkpoint_dir, weight_map)

    try:
        required_hf, assigned_olmo = _assign_nemotron_weights(
            model_state,
            loader,
            hf_config,
            dry_run=dry_run,
        )
    finally:
        loader.close()

    missing_hf = sorted(required_hf - set(weight_map))
    missing_olmo -= assigned_olmo
    if missing_hf:
        raise RuntimeError(f"Missing required HF weights: {missing_hf[:20]}")
    if missing_olmo:
        raise RuntimeError(f"Unassigned OLMo weights: {sorted(missing_olmo)[:20]}")

    output_path_obj = Path(output_path)
    if dry_run:
        log.info(
            "Dry run successful: %d HF keys cover %d OLMo tensors",
            len(required_hf),
            len(assigned_olmo),
        )
        return

    model_and_optim_dir = join_path(output_path_obj, "model_and_optim")
    log.info("Saving converted OLMo-core checkpoint to %s", model_and_optim_dir)
    save_model_and_optim_state(model_and_optim_dir, model, save_overwrite=save_overwrite)

    experiment_config_dict = {
        "model": model_config.as_config_dict(),
        "dataset": {
            "tokenizer": _tokenizer_config_from_hf(hf_model, hf_config).as_config_dict(),
        },
        "conversion": {
            "source": hf_model,
            "revision": revision,
            "converted": "Nemotron-H text backbone + lm_head",
        },
    }
    config_path = join_path(output_path_obj, "config.json")
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
    parser.add_argument("--hf-model", default=NEMOTRON3_NANO_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--cache-dir", default="/workspace/checkpoint")
    parser.add_argument("--dtype", choices=[d.value for d in DType], default=DType.bfloat16.value)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--attention-backend",
        choices=[b.value for b in AttentionBackendName],
        default=AttentionBackendName.torch.value,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-overwrite", action="store_true")
    args = parser.parse_args()

    prepare_cli_environment()
    convert(
        hf_model=args.hf_model,
        revision=args.revision,
        output_path=args.output_path,
        cache_dir=args.cache_dir,
        dtype=DType(args.dtype),
        device=args.device,
        attention_backend=AttentionBackendName(args.attention_backend),
        dry_run=args.dry_run,
        save_overwrite=args.save_overwrite,
    )


if __name__ == "__main__":
    main()
