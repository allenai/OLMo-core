"""
Convert OpenAI gpt-oss-20b Hugging Face checkpoints to OLMo-core.

The released HF checkpoint stores routed expert projection weights as MXFP4
blocks/scales. This converter dequantizes those expert weights to the requested
OLMo dtype and writes a regular OLMo distributed checkpoint.
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
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import save_model_and_optim_state
from olmo_core.io import copy_file, join_path
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.moe.v2.gpt_oss import build_gpt_oss_20b_config_from_hf_config
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def _is_local_dir(path_or_repo: str) -> bool:
    return Path(path_or_repo).is_dir()


def _load_hf_config_dict(model_name_or_path: str, revision: str) -> dict[str, Any]:
    cfg = AutoConfig.from_pretrained(model_name_or_path, revision=revision, trust_remote_code=False)
    return cfg.to_dict()


def _tokenizer_config_from_hf(hf_model: str, hf_config: Mapping[str, Any]) -> TokenizerConfig:
    eos_token_id = hf_config.get("eos_token_id")
    if eos_token_id is None:
        raise ValueError("gpt-oss config is missing eos_token_id")
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    pad_token_id = hf_config.get("pad_token_id")
    if pad_token_id is None:
        pad_token_id = eos_token_id
    return TokenizerConfig(
        vocab_size=int(hf_config["vocab_size"]),
        eos_token_id=int(eos_token_id),
        pad_token_id=int(pad_token_id),
        bos_token_id=hf_config.get("bos_token_id"),
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


def _make_config_overrides_for_layer_limit(
    hf_config: Mapping[str, Any],
    max_layers: int | None,
) -> dict[str, Any]:
    if max_layers is None:
        return {}
    total_layers = int(hf_config["num_hidden_layers"])
    if max_layers < 1 or max_layers > total_layers:
        raise ValueError(f"--max-layers must be in [1, {total_layers}], got {max_layers}")
    overrides: dict[str, Any] = {"n_layers": max_layers}
    overrides["layer_types"] = tuple(hf_config["layer_types"][:max_layers])
    return overrides


def _decode_mxfp4_expert_weight(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype,
    rows_per_chunk: int,
) -> torch.Tensor:
    return convert_moe_packed_tensors(
        blocks,
        scales,
        dtype=dtype,
        rows_per_chunk=rows_per_chunk,
    )


def _assign_gpt_oss_weights(
    model_state: dict[str, torch.Tensor],
    loader: SafeTensorKeyLoader,
    hf_config: Mapping[str, Any],
    *,
    n_layers: int,
    dtype: DType,
    decode_rows_per_chunk: int,
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

    assign("embeddings.weight", require("model.embed_tokens.weight"))
    assign("lm_head.norm.weight", require("model.norm.weight"))
    assign("lm_head.w_out.weight", require("lm_head.weight"))

    for layer_idx in range(n_layers):
        prefix_hf = f"model.layers.{layer_idx}"
        prefix_olmo = f"blocks.{layer_idx}"

        assign(f"{prefix_olmo}.attention_norm.weight", require(f"{prefix_hf}.input_layernorm.weight"))
        assign(
            f"{prefix_olmo}.feed_forward_norm.weight",
            require(f"{prefix_hf}.post_attention_layernorm.weight"),
        )
        for proj_hf, proj_olmo in (
            ("q_proj", "w_q"),
            ("k_proj", "w_k"),
            ("v_proj", "w_v"),
            ("o_proj", "w_out"),
        ):
            assign(
                f"{prefix_olmo}.attention.{proj_olmo}.weight",
                require(f"{prefix_hf}.self_attn.{proj_hf}.weight"),
            )
            assign(
                f"{prefix_olmo}.attention.{proj_olmo}.bias",
                require(f"{prefix_hf}.self_attn.{proj_hf}.bias"),
            )
        assign(f"{prefix_olmo}.attention.sinks", require(f"{prefix_hf}.self_attn.sinks"))

        assign_flat(
            f"{prefix_olmo}.routed_experts_router.weight",
            require(f"{prefix_hf}.mlp.router.weight"),
        )
        assign(f"{prefix_olmo}.routed_experts_router.bias", require(f"{prefix_hf}.mlp.router.bias"))

        gate_up_blocks = require(f"{prefix_hf}.mlp.experts.gate_up_proj_blocks")
        gate_up_scales = require(f"{prefix_hf}.mlp.experts.gate_up_proj_scales")
        assigned_olmo_keys.add(f"{prefix_olmo}.routed_experts.w_up_gate")
        if not dry_run:
            assert gate_up_blocks is not None and gate_up_scales is not None
            gate_up = _decode_mxfp4_expert_weight(
                gate_up_blocks,
                gate_up_scales,
                dtype=dtype.as_pt(),
                rows_per_chunk=decode_rows_per_chunk,
            )
            _copy_param(
                model_state[f"{prefix_olmo}.routed_experts.w_up_gate"],
                gate_up.transpose(1, 2).contiguous(),
                f"{prefix_olmo}.routed_experts.w_up_gate",
            )
        assign(
            f"{prefix_olmo}.routed_experts.b_up_gate",
            require(f"{prefix_hf}.mlp.experts.gate_up_proj_bias"),
        )

        down_blocks = require(f"{prefix_hf}.mlp.experts.down_proj_blocks")
        down_scales = require(f"{prefix_hf}.mlp.experts.down_proj_scales")
        assigned_olmo_keys.add(f"{prefix_olmo}.routed_experts.w_down")
        if not dry_run:
            assert down_blocks is not None and down_scales is not None
            down = _decode_mxfp4_expert_weight(
                down_blocks,
                down_scales,
                dtype=dtype.as_pt(),
                rows_per_chunk=decode_rows_per_chunk,
            )
            _copy_param(model_state[f"{prefix_olmo}.routed_experts.w_down"], down, f"{prefix_olmo}.routed_experts.w_down")
        assign(
            f"{prefix_olmo}.routed_experts.b_down",
            require(f"{prefix_hf}.mlp.experts.down_proj_bias"),
        )

    return required_hf_keys, assigned_olmo_keys


def convert_gpt_oss_20b_hf_to_olmo(
    *,
    hf_model: str,
    output_path: str | Path,
    revision: str = "main",
    cache_dir: str | None = "/workspace/checkpoint",
    dtype: DType = DType.bfloat16,
    device: torch.device = torch.device("cpu"),
    attention_backend: AttentionBackendName = AttentionBackendName.torch,
    max_layers: int | None = None,
    dry_run: bool = False,
    save_overwrite: bool = False,
    decode_rows_per_chunk: int = 32768 * 1024,
) -> None:
    hf_config = _load_hf_config_dict(hf_model, revision)
    layer_overrides = _make_config_overrides_for_layer_limit(hf_config, max_layers)
    model_config = build_gpt_oss_20b_config_from_hf_config(
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
        required_hf, assigned_olmo = _assign_gpt_oss_weights(
            model_state,
            loader,
            hf_config,
            n_layers=model_config.n_layers,
            dtype=dtype,
            decode_rows_per_chunk=decode_rows_per_chunk,
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
        log.info(
            "Dry run successful: %d HF keys cover %d OLMo tensors",
            len(required_hf),
            len(assigned_olmo),
        )
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
        required_hf_actual, assigned_olmo_actual = _assign_gpt_oss_weights(
            model_state,
            loader,
            hf_config,
            n_layers=model_config.n_layers,
            dtype=dtype,
            decode_rows_per_chunk=decode_rows_per_chunk,
            dry_run=False,
        )
    finally:
        loader.close()
    if required_hf_actual != required_hf or assigned_olmo_actual != assigned_olmo:
        raise RuntimeError("gpt-oss conversion dry-run and materialized assignment coverage diverged")

    model_and_optim_dir = join_path(output_path, "model_and_optim")
    log.info("Saving converted OLMo-core checkpoint to %s", model_and_optim_dir)
    save_model_and_optim_state(model_and_optim_dir, model, save_overwrite=save_overwrite)

    tokenizer_config = _tokenizer_config_from_hf(hf_model, hf_config)
    experiment_config_dict = {
        "model": model_config.as_config_dict(),
        "dataset": {
            "tokenizer": tokenizer_config.as_config_dict(),
        },
        "conversion": {
            "source": hf_model,
            "revision": revision,
            "converted": "model text tower + lm_head",
            "max_layers": max_layers,
            "expert_weight_source_dtype": "mxfp4",
            "expert_weight_checkpoint_dtype": dtype.value,
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
    parser.add_argument("--hf-model", default="openai/gpt-oss-20b")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--cache-dir", default="/workspace/checkpoint")
    parser.add_argument("--dtype", choices=[d.value for d in DType], default=DType.bfloat16.value)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--attention-backend",
        choices=[b.value for b in AttentionBackendName],
        default=AttentionBackendName.torch.value,
        help="Use torch for exact gpt-oss attention-sink semantics.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Convert only the first N transformer layers, useful for OOM-safe smoke tests.",
    )
    parser.add_argument(
        "--decode-rows-per-chunk",
        type=int,
        default=32768 * 1024,
        help="Rows per MXFP4 decode chunk; lower this if CPU/GPU memory is tight.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-overwrite", action="store_true")
    args = parser.parse_args()

    prepare_cli_environment()
    convert_gpt_oss_20b_hf_to_olmo(
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
        decode_rows_per_chunk=args.decode_rows_per_chunk,
    )


if __name__ == "__main__":
    main()
