"""Convert an OLMo-core Qwen3 MoE checkpoint to native Hugging Face format."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from olmo_core.nn.moe.v2.hf.convert_checkpoint import load_state_dict_direct
from olmo_core.utils import prepare_cli_environment


log = logging.getLogger(__name__)


def _num_experts(config: Any) -> int:
    for name in ("num_experts", "num_local_experts", "num_routed_experts", "n_routed_experts"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Qwen config does not define the number of routed experts")


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
def load_qwen3_moe_from_olmo_state(
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


def verify_export(
    *,
    checkpoint_path: Path,
    output_path: Path,
    dtype: torch.dtype,
) -> None:
    log.info("Loading exported Hugging Face checkpoint from %s", output_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        output_path,
        dtype=dtype,
        trust_remote_code=False,
    )
    model_and_optim_path = checkpoint_path / "model_and_optim"
    log.info("Loading OLMo checkpoint tensors from %s", model_and_optim_path)
    olmo_state = load_state_dict_direct(
        model_and_optim_path,
        process_group=None,
        pre_download=False,
        thread_count=32,
    )
    load_qwen3_moe_from_olmo_state(hf_model, olmo_state, verify_only=True)
    result = {
        "source_checkpoint": str(checkpoint_path),
        "hf_export": str(output_path),
        "dtype": str(dtype).removeprefix("torch."),
        "exact_match": True,
        "source_tensor_count": len(olmo_state),
        "hf_parameter_count": len(hf_model.state_dict()),
    }
    (output_path / "weight-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    log.info("Exact weight verification complete")


def convert_checkpoint(
    *,
    checkpoint_path: Path,
    output_path: Path,
    hf_model_name: str,
    tokenizer_name: str,
    dtype: torch.dtype,
    max_shard_size: str,
    save_overwrite: bool,
) -> None:
    if output_path.exists():
        if not save_overwrite:
            raise FileExistsError(f"Output path already exists: {output_path}")
        shutil.rmtree(output_path)

    log.info("Building native Hugging Face model skeleton from %s", hf_model_name)
    config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=False)
    config.torch_dtype = dtype
    with torch.device("meta"):
        hf_model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    hf_model.to(dtype=dtype)
    hf_model.to_empty(device="cpu")

    model_and_optim_path = checkpoint_path / "model_and_optim"
    log.info("Loading OLMo checkpoint tensors from %s", model_and_optim_path)
    olmo_state = load_state_dict_direct(
        model_and_optim_path,
        process_group=None,
        pre_download=False,
        thread_count=32,
    )
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)
    tokenizer.save_pretrained(output_path)

    provenance = {
        "source_checkpoint": str(checkpoint_path),
        "hf_model_config": hf_model_name,
        "tokenizer": tokenizer_name,
        "dtype": str(dtype).removeprefix("torch."),
        "source_tensor_count": source_tensor_count,
        "hf_parameter_count": len(hf_model.state_dict()),
    }
    (output_path / "conversion.json").write_text(json.dumps(provenance, indent=2) + "\n")
    log.info("Conversion complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--hf-model-name", default="Qwen/Qwen3-30B-A3B-Base")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--save-overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    prepare_cli_environment()
    if args.verify_only:
        verify_export(
            checkpoint_path=args.checkpoint_path,
            output_path=args.output_path,
            dtype=getattr(torch, args.dtype),
        )
        return
    convert_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        hf_model_name=args.hf_model_name,
        tokenizer_name=args.tokenizer_name,
        dtype=getattr(torch, args.dtype),
        max_shard_size=args.max_shard_size,
        save_overwrite=args.save_overwrite,
    )


if __name__ == "__main__":
    main()
