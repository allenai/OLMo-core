#!/usr/bin/env python3
"""
Convert a MoEV2TransformerTrainModule direct checkpoint from one GQA KV-head count to a larger one.

This rewrites the flattened model+optimizer checkpoint saved under ``model_and_optim/`` and copies the
rest of the checkpoint scaffold. The converted checkpoint can then be resumed normally with the new
``n_kv_heads`` in the training config.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from olmo_core.distributed.checkpoint import (
    get_checkpoint_metadata,
    load_keys,
    save_state_dict,
)

MODEL_AND_OPTIM_DIRNAME = "model_and_optim"
ATTN_BLOCK_PREFIX = re.compile(r"^module\.blocks\.\d+\.attention\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_checkpoint", type=Path, help="Path to the input step checkpoint directory"
    )
    parser.add_argument("target_kv_heads", type=int, help="Target KV head count")
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=None,
        help="Optional output checkpoint directory. Defaults to '<input>-GQA-<n_heads>-<target_kv>'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output checkpoint directory if it already exists.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def infer_attention_params(step_dir: Path) -> Tuple[int, int, int, int, bool]:
    config = load_json(step_dir / "config.json")
    model_cfg = config["model"]
    attn_cfg = model_cfg["block"]["attention"]

    n_heads = int(attn_cfg["n_heads"])
    n_kv_heads = int(attn_cfg.get("n_kv_heads", n_heads))
    d_model = int(model_cfg["d_model"])
    d_attn = int(attn_cfg.get("d_attn", d_model))
    use_head_qk_norm = bool(attn_cfg.get("use_head_qk_norm", False))

    if d_attn % n_heads != 0:
        raise ValueError(f"d_attn={d_attn} must be divisible by n_heads={n_heads}")
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}")

    return n_heads, n_kv_heads, d_model, d_attn, use_head_qk_norm


def derive_output_dir(input_checkpoint: Path, n_heads: int, target_kv_heads: int) -> Path:
    return input_checkpoint.with_name(f"{input_checkpoint.name}-GQA-{n_heads}-{target_kv_heads}")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output checkpoint already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_scaffold(input_checkpoint: Path, output_checkpoint: Path) -> None:
    for entry in input_checkpoint.iterdir():
        if entry.name == MODEL_AND_OPTIM_DIRNAME:
            continue

        target = output_checkpoint / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)


def patch_config(config: Dict[str, Any], old_kv_heads: int, new_kv_heads: int) -> int:
    updated = 0

    def walk(obj: Any) -> None:
        nonlocal updated
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "n_kv_heads" and value == old_kv_heads:
                    obj[key] = new_kv_heads
                    updated += 1
                else:
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(config)
    return updated


def load_flat_state_dict(checkpoint_dir: Path) -> Dict[str, Any]:
    metadata = get_checkpoint_metadata(str(checkpoint_dir))
    keys = list(metadata.state_dict_metadata.keys())
    values = list(load_keys(str(checkpoint_dir), keys))
    return dict(zip(keys, values, strict=True))


def expand_head_axis(
    flat_tensor: torch.Tensor, old_heads: int, head_dim: int, trailing_size: int, repeat_factor: int
) -> torch.Tensor:
    expected_numel = old_heads * head_dim * trailing_size
    if flat_tensor.numel() != expected_numel:
        raise ValueError(f"Expected {expected_numel} elements, found {flat_tensor.numel()}")
    return (
        flat_tensor.reshape(old_heads, head_dim, trailing_size)
        .repeat_interleave(repeat_factor, dim=0)
        .reshape(-1)
    )


def expand_head_vector(
    flat_tensor: torch.Tensor, old_heads: int, head_dim: int, repeat_factor: int
) -> torch.Tensor:
    expected_numel = old_heads * head_dim
    if flat_tensor.numel() != expected_numel:
        raise ValueError(f"Expected {expected_numel} elements, found {flat_tensor.numel()}")
    return (
        flat_tensor.reshape(old_heads, head_dim).repeat_interleave(repeat_factor, dim=0).reshape(-1)
    )


def should_expand_attn_key(key: str) -> bool:
    if not ATTN_BLOCK_PREFIX.match(key):
        return False
    if any(
        marker in key
        for marker in (
            ".w_k.weight.main",
            ".w_k.weight.exp_avg",
            ".w_k.weight.exp_avg_sq",
            ".w_k.bias.main",
            ".w_k.bias.exp_avg",
            ".w_k.bias.exp_avg_sq",
            ".w_v.weight.main",
            ".w_v.weight.exp_avg",
            ".w_v.weight.exp_avg_sq",
            ".w_v.bias.main",
            ".w_v.bias.exp_avg",
            ".w_v.bias.exp_avg_sq",
        )
    ):
        return True
    if any(
        marker in key
        for marker in (
            ".k_norm.weight.main",
            ".k_norm.weight.exp_avg",
            ".k_norm.weight.exp_avg_sq",
        )
    ):
        return True
    return False


def convert_state_dict(
    state_dict: Dict[str, Any],
    *,
    n_heads: int,
    old_kv_heads: int,
    target_kv_heads: int,
    d_model: int,
    d_attn: int,
    use_head_qk_norm: bool,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    head_dim = d_attn // n_heads
    repeat_factor = target_kv_heads // old_kv_heads
    counts = {"w_k": 0, "w_v": 0, "k_norm": 0}
    converted: Dict[str, Any] = {}

    for key, value in state_dict.items():
        if not should_expand_attn_key(key) or not isinstance(value, torch.Tensor):
            converted[key] = value
            continue

        if ".w_k.weight." in key or ".w_v.weight." in key:
            converted[key] = expand_head_axis(
                value,
                old_heads=old_kv_heads,
                head_dim=head_dim,
                trailing_size=d_model,
                repeat_factor=repeat_factor,
            )
            counts["w_k" if ".w_k." in key else "w_v"] += 1
        elif ".w_k.bias." in key or ".w_v.bias." in key:
            converted[key] = expand_head_vector(
                value,
                old_heads=old_kv_heads,
                head_dim=head_dim,
                repeat_factor=repeat_factor,
            )
            counts["w_k" if ".w_k." in key else "w_v"] += 1
        elif (
            ".k_norm.weight." in key
            and not use_head_qk_norm
            and value.numel() == old_kv_heads * head_dim
        ):
            converted[key] = expand_head_vector(
                value,
                old_heads=old_kv_heads,
                head_dim=head_dim,
                repeat_factor=repeat_factor,
            )
            counts["k_norm"] += 1
        else:
            converted[key] = value

    return converted, counts


def main() -> None:
    args = parse_args()
    input_checkpoint = args.input_checkpoint.resolve()
    if not input_checkpoint.is_dir():
        raise FileNotFoundError(f"Input checkpoint directory not found: {input_checkpoint}")

    model_and_optim_dir = input_checkpoint / MODEL_AND_OPTIM_DIRNAME
    if not model_and_optim_dir.is_dir():
        raise FileNotFoundError(f"Missing '{MODEL_AND_OPTIM_DIRNAME}' in {input_checkpoint}")

    n_heads, old_kv_heads, d_model, d_attn, use_head_qk_norm = infer_attention_params(
        input_checkpoint
    )
    target_kv_heads = int(args.target_kv_heads)

    if target_kv_heads < old_kv_heads:
        raise ValueError(
            f"Target KV heads ({target_kv_heads}) must be >= source KV heads ({old_kv_heads})"
        )
    if target_kv_heads == old_kv_heads:
        raise ValueError("Target KV heads is the same as the source KV heads")
    if n_heads % target_kv_heads != 0:
        raise ValueError(f"Target KV heads ({target_kv_heads}) must divide n_heads ({n_heads})")
    if target_kv_heads % old_kv_heads != 0:
        raise ValueError(
            f"Target KV heads ({target_kv_heads}) must be a multiple of source KV heads ({old_kv_heads})"
        )

    output_checkpoint = (
        args.output_checkpoint.resolve()
        if args.output_checkpoint is not None
        else derive_output_dir(input_checkpoint, n_heads, target_kv_heads)
    )
    prepare_output_dir(output_checkpoint, overwrite=args.overwrite)
    copy_scaffold(input_checkpoint, output_checkpoint)

    config = load_json(output_checkpoint / "config.json")
    updated_fields = patch_config(config, old_kv_heads=old_kv_heads, new_kv_heads=target_kv_heads)
    save_json(output_checkpoint / "config.json", config)

    print(f"Loading flat checkpoint from {model_and_optim_dir} ...")
    state_dict = load_flat_state_dict(model_and_optim_dir)

    print(
        f"Converting KV heads {old_kv_heads} -> {target_kv_heads} "
        f"(n_heads={n_heads}, d_model={d_model}, d_attn={d_attn}, use_head_qk_norm={use_head_qk_norm}) ..."
    )
    converted_state_dict, counts = convert_state_dict(
        state_dict,
        n_heads=n_heads,
        old_kv_heads=old_kv_heads,
        target_kv_heads=target_kv_heads,
        d_model=d_model,
        d_attn=d_attn,
        use_head_qk_norm=use_head_qk_norm,
    )

    output_model_and_optim = output_checkpoint / MODEL_AND_OPTIM_DIRNAME
    print(f"Saving converted checkpoint to {output_model_and_optim} ...")
    save_state_dict(str(output_model_and_optim), converted_state_dict, save_overwrite=True)

    print("Conversion complete.")
    print(f"Output checkpoint: {output_checkpoint}")
    print(f"Updated config fields: {updated_fields}")
    print(
        "Expanded tensors: " f"w_k={counts['w_k']}, w_v={counts['w_v']}, k_norm={counts['k_norm']}"
    )


if __name__ == "__main__":
    main()
