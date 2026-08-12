"""Convert OLMo-core Qwen3 and Qwen3.5 MoE checkpoints to native HF format."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dist_cp
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from olmo_core.distributed.checkpoint import RemoteFileSystemReader
from olmo_core.io import normalize_path
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)

_TOKEN_ID_ATTRIBUTES = ("bos_token_id", "eos_token_id", "pad_token_id")
_DEFAULT_MAX_POSITION_EMBEDDINGS = 65_536


def _tokenizer_metadata(tokenizer: Any) -> dict[str, Any]:
    return {
        **{name: getattr(tokenizer, name, None) for name in _TOKEN_ID_ATTRIBUTES},
        "vocab_size": len(tokenizer),
        "chat_template": getattr(tokenizer, "chat_template", None),
    }


def _sync_tokenizer_metadata(config: Any, tokenizer: Any) -> dict[str, Any]:
    metadata = _tokenizer_metadata(tokenizer)
    for name in _TOKEN_ID_ATTRIBUTES:
        setattr(config, name, metadata[name])
    return metadata


def _set_max_position_embeddings(config: Any, max_position_embeddings: int) -> None:
    if max_position_embeddings <= 0:
        raise ValueError("max_position_embeddings must be positive")
    config.max_position_embeddings = max_position_embeddings


def _load_generation_config(config: Any, generation_config_name: str | None) -> GenerationConfig:
    if generation_config_name is not None:
        try:
            return GenerationConfig.from_pretrained(generation_config_name)
        except OSError:
            log.warning(
                "No generation_config.json found at %s; deriving it from the model config",
                generation_config_name,
            )
    return GenerationConfig.from_model_config(config)


def _verify_tokenizer_metadata(
    *,
    output_path: Path,
    tokenizer_name: str,
    generation_config_name: str | None = None,
    max_position_embeddings: int,
) -> dict[str, Any]:
    expected_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=False,
    )
    exported_tokenizer = AutoTokenizer.from_pretrained(
        output_path,
        trust_remote_code=False,
    )
    expected = _tokenizer_metadata(expected_tokenizer)
    actual = _tokenizer_metadata(exported_tokenizer)
    if actual != expected:
        mismatches = {
            name: {"expected": expected[name], "actual": actual[name]}
            for name in expected
            if actual[name] != expected[name]
        }
        raise ValueError(f"Exported tokenizer metadata differs from the source: {mismatches}")

    model_config = AutoConfig.from_pretrained(output_path, trust_remote_code=False)
    if model_config.max_position_embeddings != max_position_embeddings:
        raise ValueError(
            "Exported model config has "
            f"max_position_embeddings={model_config.max_position_embeddings!r}; "
            f"expected {max_position_embeddings!r}"
        )
    generation_config = GenerationConfig.from_pretrained(output_path)
    expected_generation_config = _load_generation_config(model_config, generation_config_name)
    for name in _TOKEN_ID_ATTRIBUTES:
        model_value = getattr(model_config, name, None)
        if model_value != expected[name]:
            raise ValueError(
                f"Exported model config has {name}={model_value!r}; "
                f"expected {expected[name]!r} from {tokenizer_name}"
            )
        generation_value = getattr(generation_config, name, None)
        expected_generation_value = getattr(expected_generation_config, name, None)
        if generation_value != expected_generation_value:
            raise ValueError(
                f"Exported generation config has {name}={generation_value!r}; "
                f"expected {expected_generation_value!r} from "
                f"{generation_config_name or 'the model config'}"
            )

    return {name: expected[name] for name in (*_TOKEN_ID_ATTRIBUTES, "vocab_size")} | {
        "chat_template_matches": True,
        "max_position_embeddings": max_position_embeddings,
    }


def _num_experts(config: Any) -> int:
    for name in ("num_experts", "num_local_experts", "num_routed_experts", "n_routed_experts"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Qwen config does not define the number of routed experts")


def _is_qwen35_text_config(config: Any) -> bool:
    return getattr(config, "model_type", None) == "qwen3_5_moe_text" or hasattr(
        config, "linear_num_key_heads"
    )


def _text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def _load_olmo_state(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load either an OLMo DDP training checkpoint or a plain model checkpoint."""
    checkpoint_dir = normalize_path(checkpoint_dir)
    reader = RemoteFileSystemReader(checkpoint_dir, thread_count=32, pre_download=False)
    metadata = reader.read_metadata().state_dict_metadata
    ddp_metadata = {name: value for name, value in metadata.items() if name.endswith(".main")}
    generic_metadata = {name: value for name, value in metadata.items() if name.startswith("model.")}

    if ddp_metadata:
        tensors_metadata = ddp_metadata
        is_ddp = True
        layout = "OLMo DDP"
    elif generic_metadata:
        tensors_metadata = generic_metadata
        is_ddp = False
        layout = "plain OLMo model"
    else:
        raise ValueError(
            f"Could not identify model tensors in checkpoint metadata at {checkpoint_dir}"
        )

    checkpoint_state: dict[str, torch.Tensor] = {}
    for name, tensor_metadata in tensors_metadata.items():
        properties = getattr(tensor_metadata, "properties", None)
        dtype = getattr(properties, "dtype", torch.float32)
        checkpoint_state[name] = torch.empty(tensor_metadata.size, dtype=dtype)
    dist_cp.state_dict_loader.load(
        checkpoint_state,
        checkpoint_id=checkpoint_dir,
        storage_reader=reader,
        process_group=None,
    )
    log.info("Loaded %d tensors from %s checkpoint layout", len(checkpoint_state), layout)
    if is_ddp:
        return checkpoint_state
    return {
        f"module.{name.removeprefix('model.')}.main": value
        for name, value in checkpoint_state.items()
    }


def _copy_tensor(
    hf_state: dict[str, torch.Tensor],
    remaining_hf: set[str],
    hf_name: str,
    value: torch.Tensor,
    *,
    verify_only: bool = False,
) -> None:
    if hf_name not in hf_state:
        raise KeyError(f"Hugging Face model is missing expected parameter {hf_name!r}")
    target = hf_state[hf_name]
    if target.numel() != value.numel():
        raise ValueError(
            f"{hf_name}: target shape {tuple(target.shape)} does not match "
            f"source shape {tuple(value.shape)}"
        )
    expected = value.reshape(target.shape).to(dtype=target.dtype)
    if verify_only:
        if not torch.equal(target, expected):
            diff = (target.float() - expected.float()).abs()
            raise ValueError(
                f"{hf_name}: exported tensor differs from OLMo source "
                f"(mismatched={torch.count_nonzero(diff).item()}, max_abs_diff={diff.max().item()})"
            )
    else:
        target.copy_(expected)
    remaining_hf.remove(hf_name)


@torch.no_grad()
def _load_qwen3_moe_from_olmo_state(
    hf_model: Any,
    olmo_state: dict[str, torch.Tensor],
    *,
    verify_only: bool = False,
) -> None:
    """Load OLMo DDP Qwen3 MoE tensors into a native HF Qwen3 MoE model."""

    hf_state = hf_model.state_dict()
    remaining_hf = set(hf_state)
    remaining_olmo = set(olmo_state)

    def assign(hf_name: str, value: torch.Tensor) -> None:
        _copy_tensor(
            hf_state,
            remaining_hf,
            hf_name,
            value,
            verify_only=verify_only,
        )

    def take(name: str) -> torch.Tensor:
        try:
            value = olmo_state[name]
        except KeyError as exc:
            raise KeyError(f"OLMo checkpoint is missing expected tensor {name!r}") from exc
        remaining_olmo.discard(name)
        return value

    direct_mapping = {
        "model.embed_tokens.weight": "module.embeddings.weight.main",
        "model.norm.weight": "module.lm_head.norm.weight.main",
        "lm_head.weight": "module.lm_head.w_out.weight.main",
    }
    for hf_name, olmo_name in direct_mapping.items():
        assign(hf_name, take(olmo_name))

    config = hf_model.config
    num_experts = _num_experts(config)
    expert_hidden_size = int(config.moe_intermediate_size)
    d_model = int(config.hidden_size)

    for layer_idx in range(int(config.num_hidden_layers)):
        hf_prefix = f"model.layers.{layer_idx}"
        olmo_prefix = f"module.blocks.{layer_idx}"
        layer_mapping = {
            f"{hf_prefix}.input_layernorm.weight": f"{olmo_prefix}.attention_norm.weight.main",
            f"{hf_prefix}.post_attention_layernorm.weight": (
                f"{olmo_prefix}.feed_forward_norm.weight.main"
            ),
            f"{hf_prefix}.self_attn.q_proj.weight": f"{olmo_prefix}.attention.w_q.weight.main",
            f"{hf_prefix}.self_attn.k_proj.weight": f"{olmo_prefix}.attention.w_k.weight.main",
            f"{hf_prefix}.self_attn.v_proj.weight": f"{olmo_prefix}.attention.w_v.weight.main",
            f"{hf_prefix}.self_attn.o_proj.weight": f"{olmo_prefix}.attention.w_out.weight.main",
            f"{hf_prefix}.self_attn.q_norm.weight": f"{olmo_prefix}.attention.q_norm.weight.main",
            f"{hf_prefix}.self_attn.k_norm.weight": f"{olmo_prefix}.attention.k_norm.weight.main",
            f"{hf_prefix}.mlp.gate.weight": f"{olmo_prefix}.routed_experts_router.weight.main",
        }
        for hf_name, olmo_name in layer_mapping.items():
            assign(hf_name, take(olmo_name))

        up_gate_name = f"{olmo_prefix}.routed_experts.w_up_gate.main"
        down_name = f"{olmo_prefix}.routed_experts.w_down.main"
        up_gate = take(up_gate_name).reshape(num_experts, 2 * expert_hidden_size, d_model)
        down = take(down_name).reshape(num_experts, expert_hidden_size, d_model)
        packed_up_gate_name = f"{hf_prefix}.mlp.experts.gate_up_proj"
        packed_down_name = f"{hf_prefix}.mlp.experts.down_proj"
        if packed_up_gate_name in hf_state:
            # OLMo stores [up, gate], while native Qwen stores [gate, up].
            assign(
                packed_up_gate_name,
                torch.cat(
                    (up_gate[:, expert_hidden_size:], up_gate[:, :expert_hidden_size]),
                    dim=1,
                ),
            )
            assign(
                packed_down_name,
                down.transpose(1, 2),
            )
        else:
            for expert_idx in range(num_experts):
                expert_prefix = f"{hf_prefix}.mlp.experts.{expert_idx}"
                assign(
                    f"{expert_prefix}.up_proj.weight",
                    up_gate[expert_idx, :expert_hidden_size],
                )
                assign(
                    f"{expert_prefix}.gate_proj.weight",
                    up_gate[expert_idx, expert_hidden_size:],
                )
                assign(
                    f"{expert_prefix}.down_proj.weight",
                    down[expert_idx].transpose(0, 1),
                )

    if remaining_hf:
        raise RuntimeError(f"Unassigned Hugging Face parameters: {sorted(remaining_hf)[:20]}")
    if remaining_olmo:
        raise RuntimeError(f"Unconsumed OLMo checkpoint tensors: {sorted(remaining_olmo)[:20]}")

    log.info(
        "%s all %d Hugging Face parameters against %d OLMo checkpoint tensors",
        "Verified" if verify_only else "Mapped",
        len(hf_state),
        len(olmo_state),
    )


@torch.no_grad()
def _load_qwen35_moe_from_olmo_state(
    hf_model: Any,
    olmo_state: dict[str, torch.Tensor],
    *,
    verify_only: bool = False,
) -> None:
    """Load OLMo DDP Qwen3.5 text tensors into a native HF text model."""

    hf_state = hf_model.state_dict()
    remaining_hf = set(hf_state)
    remaining_olmo = set(olmo_state)

    def assign(hf_name: str, value: torch.Tensor) -> None:
        _copy_tensor(hf_state, remaining_hf, hf_name, value, verify_only=verify_only)

    def take(name: str) -> torch.Tensor:
        try:
            value = olmo_state[name]
        except KeyError as exc:
            raise KeyError(f"OLMo checkpoint is missing expected tensor {name!r}") from exc
        remaining_olmo.discard(name)
        return value

    direct_mapping = {
        "model.embed_tokens.weight": "module.embeddings.weight.main",
        "model.norm.weight": "module.lm_head.norm.weight.main",
        "lm_head.weight": "module.lm_head.w_out.weight.main",
    }
    for hf_name, olmo_name in direct_mapping.items():
        assign(hf_name, take(olmo_name))

    config = hf_model.config
    layer_types = list(config.layer_types)
    num_experts = _num_experts(config)
    moe_hidden = int(config.moe_intermediate_size)
    shared_hidden = int(config.shared_expert_intermediate_size)
    hidden_size = int(config.hidden_size)

    for layer_idx in range(int(config.num_hidden_layers)):
        hf_prefix = f"model.layers.{layer_idx}"
        olmo_prefix = f"module.blocks.{layer_idx}"
        assign(
            f"{hf_prefix}.input_layernorm.weight",
            take(f"{olmo_prefix}.attention_norm.weight.main"),
        )
        assign(
            f"{hf_prefix}.post_attention_layernorm.weight",
            take(f"{olmo_prefix}.feed_forward_norm.weight.main"),
        )

        if layer_types[layer_idx] == "linear_attention":
            attention_prefix = f"{olmo_prefix}.attention"
            assign(
                f"{hf_prefix}.linear_attn.in_proj_qkv.weight",
                torch.cat(
                    (
                        take(f"{attention_prefix}.w_q.weight.main"),
                        take(f"{attention_prefix}.w_k.weight.main"),
                        take(f"{attention_prefix}.w_v.weight.main"),
                    ),
                    dim=0,
                ),
            )
            assign(
                f"{hf_prefix}.linear_attn.conv1d.weight",
                torch.cat(
                    (
                        take(f"{attention_prefix}.q_conv1d.weight.main"),
                        take(f"{attention_prefix}.k_conv1d.weight.main"),
                        take(f"{attention_prefix}.v_conv1d.weight.main"),
                    ),
                    dim=0,
                ),
            )
            linear_mapping = {
                "in_proj_a.weight": "w_a.weight.main",
                "in_proj_b.weight": "w_b.weight.main",
                "in_proj_z.weight": "w_g.weight.main",
                "out_proj.weight": "w_out.weight.main",
                "norm.weight": "o_norm.weight.main",
                "A_log": "A_log.main",
                "dt_bias": "dt_bias.main",
            }
            for hf_suffix, olmo_suffix in linear_mapping.items():
                assign(
                    f"{hf_prefix}.linear_attn.{hf_suffix}",
                    take(f"{attention_prefix}.{olmo_suffix}"),
                )
        elif layer_types[layer_idx] == "full_attention":
            attention_prefix = f"{olmo_prefix}.attention"
            head_dim = int(config.head_dim)
            num_heads = int(config.num_attention_heads)
            q = take(f"{attention_prefix}.w_q.weight.main").reshape(
                num_heads, head_dim, hidden_size
            )
            gate = take(f"{attention_prefix}.w_g.weight.main").reshape(
                num_heads, head_dim, hidden_size
            )
            assign(
                f"{hf_prefix}.self_attn.q_proj.weight",
                torch.cat((q, gate), dim=1).reshape(num_heads * 2 * head_dim, hidden_size),
            )
            attention_mapping = {
                "k_proj.weight": "w_k.weight.main",
                "v_proj.weight": "w_v.weight.main",
                "o_proj.weight": "w_out.weight.main",
                "q_norm.weight": "q_norm.weight.main",
                "k_norm.weight": "k_norm.weight.main",
            }
            for hf_suffix, olmo_suffix in attention_mapping.items():
                assign(
                    f"{hf_prefix}.self_attn.{hf_suffix}",
                    take(f"{attention_prefix}.{olmo_suffix}"),
                )
        else:
            raise ValueError(f"Unsupported layer type at {layer_idx}: {layer_types[layer_idx]!r}")

        assign(
            f"{hf_prefix}.mlp.gate.weight",
            take(f"{olmo_prefix}.routed_experts_router.weight.main"),
        )
        up_gate = take(f"{olmo_prefix}.routed_experts.w_up_gate.main").reshape(
            num_experts, 2 * moe_hidden, hidden_size
        )
        up, gate = up_gate.split(moe_hidden, dim=1)
        assign(f"{hf_prefix}.mlp.experts.gate_up_proj", torch.cat((gate, up), dim=1))
        down = take(f"{olmo_prefix}.routed_experts.w_down.main").reshape(
            num_experts, moe_hidden, hidden_size
        )
        assign(f"{hf_prefix}.mlp.experts.down_proj", down.transpose(1, 2))

        shared_up_gate = take(f"{olmo_prefix}.shared_experts.w_up_gate.main").reshape(
            hidden_size, 2 * shared_hidden
        )
        shared_up, shared_gate = shared_up_gate.split(shared_hidden, dim=1)
        assign(f"{hf_prefix}.mlp.shared_expert.up_proj.weight", shared_up.t())
        assign(f"{hf_prefix}.mlp.shared_expert.gate_proj.weight", shared_gate.t())
        shared_down = take(f"{olmo_prefix}.shared_experts.w_down.main").reshape(
            shared_hidden, hidden_size
        )
        assign(f"{hf_prefix}.mlp.shared_expert.down_proj.weight", shared_down.t())
        assign(
            f"{hf_prefix}.mlp.shared_expert_gate.weight",
            take(f"{olmo_prefix}.shared_experts_router.weight.main"),
        )

    if remaining_hf:
        raise RuntimeError(f"Unassigned Hugging Face parameters: {sorted(remaining_hf)[:20]}")
    if remaining_olmo:
        raise RuntimeError(f"Unconsumed OLMo checkpoint tensors: {sorted(remaining_olmo)[:20]}")

    log.info(
        "%s all %d Hugging Face parameters against %d OLMo checkpoint tensors",
        "Verified" if verify_only else "Mapped",
        len(hf_state),
        len(olmo_state),
    )


def load_qwen3_moe_from_olmo_state(
    hf_model: Any,
    olmo_state: dict[str, torch.Tensor],
    *,
    verify_only: bool = False,
) -> None:
    if _is_qwen35_text_config(hf_model.config):
        _load_qwen35_moe_from_olmo_state(hf_model, olmo_state, verify_only=verify_only)
    else:
        _load_qwen3_moe_from_olmo_state(hf_model, olmo_state, verify_only=verify_only)


def verify_export(
    *,
    checkpoint_path: Path,
    output_path: Path,
    tokenizer_name: str,
    generation_config_name: str | None = None,
    dtype: torch.dtype,
    max_position_embeddings: int,
) -> None:
    log.info("Loading exported Hugging Face checkpoint from %s", output_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        output_path,
        dtype=dtype,
        trust_remote_code=False,
    )
    model_and_optim_path = checkpoint_path / "model_and_optim"
    log.info("Loading OLMo checkpoint tensors from %s", model_and_optim_path)
    olmo_state = _load_olmo_state(model_and_optim_path)
    load_qwen3_moe_from_olmo_state(hf_model, olmo_state, verify_only=True)
    tokenizer_metadata = _verify_tokenizer_metadata(
        output_path=output_path,
        tokenizer_name=tokenizer_name,
        generation_config_name=generation_config_name,
        max_position_embeddings=max_position_embeddings,
    )
    result = {
        "source_checkpoint": str(checkpoint_path),
        "hf_export": str(output_path),
        "dtype": str(dtype).removeprefix("torch."),
        "exact_match": True,
        "source_tensor_count": len(olmo_state),
        "hf_parameter_count": len(hf_model.state_dict()),
        "tokenizer_metadata": tokenizer_metadata,
    }
    (output_path / "weight-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    log.info("Exact weight and tokenizer metadata verification complete")


def convert_checkpoint(
    *,
    checkpoint_path: Path,
    output_path: Path,
    hf_model_name: str,
    tokenizer_name: str,
    generation_config_name: str | None = None,
    dtype: torch.dtype,
    max_shard_size: str,
    max_position_embeddings: int,
    save_overwrite: bool,
) -> None:
    if output_path.exists():
        if not save_overwrite:
            raise FileExistsError(f"Output path already exists: {output_path}")
        shutil.rmtree(output_path)

    log.info("Loading tokenizer from %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)

    log.info("Building native Hugging Face model skeleton from %s", hf_model_name)
    source_config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=False)
    config = _text_config(source_config)
    config.torch_dtype = dtype
    _set_max_position_embeddings(config, max_position_embeddings)
    tokenizer_metadata = _sync_tokenizer_metadata(config, tokenizer)
    with torch.device("meta"):
        hf_model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    generation_config = _load_generation_config(config, generation_config_name)
    hf_model.generation_config = generation_config
    hf_model.to(dtype=dtype)
    hf_model.to_empty(device="cpu")

    model_and_optim_path = checkpoint_path / "model_and_optim"
    log.info("Loading OLMo checkpoint tensors from %s", model_and_optim_path)
    olmo_state = _load_olmo_state(model_and_optim_path)
    load_qwen3_moe_from_olmo_state(hf_model, olmo_state)
    source_tensor_count = len(olmo_state)
    del olmo_state
    gc.collect()

    output_path.mkdir(parents=True)
    log.info("Saving native Hugging Face checkpoint to %s", output_path)
    hf_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(output_path)
    generation_config.save_pretrained(output_path)

    provenance = {
        "source_checkpoint": str(checkpoint_path),
        "hf_model_config": hf_model_name,
        "tokenizer": tokenizer_name,
        "generation_config": generation_config_name,
        "dtype": str(dtype).removeprefix("torch."),
        "max_position_embeddings": max_position_embeddings,
        "source_tensor_count": source_tensor_count,
        "hf_parameter_count": len(hf_model.state_dict()),
        "tokenizer_metadata": {
            name: tokenizer_metadata[name] for name in (*_TOKEN_ID_ATTRIBUTES, "vocab_size")
        },
        "chat_template_preserved": tokenizer_metadata["chat_template"] is not None,
    }
    (output_path / "conversion.json").write_text(json.dumps(provenance, indent=2) + "\n")
    log.info("Conversion complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--hf-model-name", default="Qwen/Qwen3-30B-A3B-Base")
    parser.add_argument(
        "--tokenizer-name",
        required=True,
        help="Exact tokenizer directory or HF identifier to package with the export.",
    )
    parser.add_argument(
        "--generation-config-name",
        help="Optional HF identifier or directory supplying generation_config.json.",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=_DEFAULT_MAX_POSITION_EMBEDDINGS,
        help="Maximum sequence length written to the exported Hugging Face config.",
    )
    parser.add_argument("--save-overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    prepare_cli_environment()
    if args.verify_only:
        verify_export(
            checkpoint_path=args.checkpoint_path,
            output_path=args.output_path,
            tokenizer_name=args.tokenizer_name,
            generation_config_name=args.generation_config_name,
            dtype=getattr(torch, args.dtype),
            max_position_embeddings=args.max_position_embeddings,
        )
        return
    convert_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        hf_model_name=args.hf_model_name,
        tokenizer_name=args.tokenizer_name,
        generation_config_name=args.generation_config_name,
        dtype=getattr(torch, args.dtype),
        max_shard_size=args.max_shard_size,
        max_position_embeddings=args.max_position_embeddings,
        save_overwrite=args.save_overwrite,
    )


if __name__ == "__main__":
    main()
