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
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from olmo_core.distributed.checkpoint import load_olmo_ddp_checkpoint_state
from olmo_core.nn.hf.convert import convert_state_from_hf, convert_state_to_hf
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)

_TOKEN_ID_ATTRIBUTES = ("bos_token_id", "eos_token_id", "pad_token_id")


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


def _verify_tokenizer_metadata(
    *,
    output_path: Path,
    tokenizer_name: str,
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
    generation_config = GenerationConfig.from_pretrained(output_path)
    for config_name, config in (
        ("model config", model_config),
        ("generation config", generation_config),
    ):
        for name in _TOKEN_ID_ATTRIBUTES:
            value = getattr(config, name, None)
            if value != expected[name]:
                raise ValueError(
                    f"Exported {config_name} has {name}={value!r}; "
                    f"expected {expected[name]!r} from {tokenizer_name}"
                )

    return {name: expected[name] for name in (*_TOKEN_ID_ATTRIBUTES, "vocab_size")} | {
        "chat_template_matches": True
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
def load_qwen3_moe_from_olmo_state(
    hf_model: Any,
    olmo_state: dict[str, torch.Tensor],
    *,
    verify_only: bool = False,
) -> None:
    """Load OLMo DDP Qwen3 MoE tensors into a native HF Qwen3 MoE model."""

    hf_state = hf_model.state_dict()
    prefix = "module."
    suffix = ".main"
    native_state = {
        name.removeprefix(prefix).removesuffix(suffix): value
        for name, value in olmo_state.items()
        if name.startswith(prefix) and name.endswith(suffix)
    }
    if len(native_state) != len(olmo_state):
        valid_checkpoint_names = {f"{prefix}{name}{suffix}" for name in native_state}
        invalid_names = sorted(set(olmo_state) - valid_checkpoint_names)
        raise RuntimeError(
            "Unexpected OLMoDDP checkpoint tensor names; expected "
            f"'module.<model parameter>.main': {invalid_names[:20]}"
        )

    expected_native_state = convert_state_from_hf(
        hf_model.config,
        hf_state,
        model_type=hf_model.config.model_type,
    )
    missing_native = set(expected_native_state) - set(native_state)
    unexpected_native = set(native_state) - set(expected_native_state)
    if missing_native:
        raise RuntimeError(f"OLMo checkpoint is missing parameters: {sorted(missing_native)[:20]}")
    if unexpected_native:
        raise RuntimeError(
            f"OLMo checkpoint has unexpected parameters: {sorted(unexpected_native)[:20]}"
        )

    converted_state = convert_state_to_hf(hf_model.config, native_state)
    missing_hf = set(hf_state) - set(converted_state)
    unexpected_hf = set(converted_state) - set(hf_state)
    if missing_hf:
        raise RuntimeError(f"Converted state is missing HF parameters: {sorted(missing_hf)[:20]}")
    if unexpected_hf:
        raise RuntimeError(
            f"Converted state has unexpected HF parameters: {sorted(unexpected_hf)[:20]}"
        )

    remaining_hf = set(hf_state)
    for hf_name, value in converted_state.items():
        _copy_tensor(
            hf_state,
            remaining_hf,
            hf_name,
            value,
            verify_only=verify_only,
        )

    if remaining_hf:
        raise RuntimeError(f"Unassigned Hugging Face parameters: {sorted(remaining_hf)[:20]}")

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
    tokenizer_name: str,
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
    olmo_state = load_olmo_ddp_checkpoint_state(
        model_and_optim_path,
        pre_download=False,
    )
    load_qwen3_moe_from_olmo_state(hf_model, olmo_state, verify_only=True)
    tokenizer_metadata = _verify_tokenizer_metadata(
        output_path=output_path,
        tokenizer_name=tokenizer_name,
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
    dtype: torch.dtype,
    max_shard_size: str,
    save_overwrite: bool,
) -> None:
    if output_path.exists():
        if not save_overwrite:
            raise FileExistsError(f"Output path already exists: {output_path}")
        shutil.rmtree(output_path)

    log.info("Loading tokenizer from %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)

    log.info("Building native Hugging Face model skeleton from %s", hf_model_name)
    config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=False)
    config.torch_dtype = dtype
    tokenizer_metadata = _sync_tokenizer_metadata(config, tokenizer)
    with torch.device("meta"):
        hf_model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    generation_config = GenerationConfig.from_model_config(config)
    hf_model.generation_config = generation_config
    hf_model.to(dtype=dtype)
    hf_model.to_empty(device="cpu")

    model_and_optim_path = checkpoint_path / "model_and_optim"
    log.info("Loading OLMo checkpoint tensors from %s", model_and_optim_path)
    olmo_state = load_olmo_ddp_checkpoint_state(
        model_and_optim_path,
        pre_download=False,
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
    tokenizer.save_pretrained(output_path)
    generation_config.save_pretrained(output_path)

    provenance = {
        "source_checkpoint": str(checkpoint_path),
        "hf_model_config": hf_model_name,
        "tokenizer": tokenizer_name,
        "dtype": str(dtype).removeprefix("torch."),
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
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--save-overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-after-export", action="store_true")
    args = parser.parse_args()

    prepare_cli_environment()
    if args.verify_only:
        verify_export(
            checkpoint_path=args.checkpoint_path,
            output_path=args.output_path,
            tokenizer_name=args.tokenizer_name,
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
    if args.verify_after_export:
        verify_export(
            checkpoint_path=args.checkpoint_path,
            output_path=args.output_path,
            tokenizer_name=args.tokenizer_name,
            dtype=getattr(torch, args.dtype),
        )


if __name__ == "__main__":
    main()
