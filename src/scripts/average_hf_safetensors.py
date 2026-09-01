#!/usr/bin/env python3
"""Average sharded Hugging Face safetensors checkpoints.

This is intentionally simpler than loading full HF models: it reads tensors from
the input ``model.safetensors.index.json`` files, averages matching floating
point tensors in fp32, writes them back with the first checkpoint's shard layout,
and copies non-weight HF files from the first checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

log = logging.getLogger(__name__)

INDEX_NAME = "model.safetensors.index.json"


@dataclass(frozen=True)
class HFCheckpoint:
    path: Path
    index: dict[str, Any]
    weight_map: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average matching weights from sharded HF safetensors checkpoints."
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        type=Path,
        help="Input HF checkpoint directories.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output HF checkpoint directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory first if it already exists.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=250,
        help="Log progress every N tensors within a shard.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> HFCheckpoint:
    path = path.expanduser().resolve()
    index_path = path / INDEX_NAME
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing HF safetensors index: {index_path}")

    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid or empty weight_map in {index_path}")

    missing_shards = sorted({name for name in weight_map.values() if not (path / name).is_file()})
    if missing_shards:
        raise FileNotFoundError(
            f"{path} is missing {len(missing_shards)} shard(s): {missing_shards[:5]}"
        )

    return HFCheckpoint(path=path, index=index, weight_map=weight_map)


def prepare_output_dir(output: Path, *, overwrite: bool) -> None:
    output = output.expanduser().resolve()
    if output.exists():
        if not overwrite and any(output.iterdir()):
            raise FileExistsError(f"Output directory exists and is not empty: {output}")
        if overwrite:
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def copy_hf_sidecar_files(source: Path, output: Path, shard_names: set[str]) -> None:
    for item in source.iterdir():
        if item.name == INDEX_NAME or item.name in shard_names or item.name.startswith(".tmp"):
            continue
        if item.suffix == ".safetensors":
            continue

        dest = output / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        elif item.is_file():
            shutil.copy2(item, dest)


def build_keys_by_output_shard(weight_map: dict[str, str]) -> dict[str, list[str]]:
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard_name in weight_map.items():
        keys_by_shard[shard_name].append(key)
    return {shard: sorted(keys) for shard, keys in keys_by_shard.items()}


def validate_weight_maps(checkpoints: list[HFCheckpoint]) -> None:
    reference_keys = set(checkpoints[0].weight_map)
    for checkpoint in checkpoints[1:]:
        keys = set(checkpoint.weight_map)
        only_ref = sorted(reference_keys - keys)
        only_other = sorted(keys - reference_keys)
        if only_ref or only_other:
            raise ValueError(
                f"Weight key mismatch for {checkpoint.path}; "
                f"missing={only_ref[:5]}, extra={only_other[:5]}"
            )


def average_tensor(
    key: str,
    checkpoints: list[HFCheckpoint],
    handles: list[dict[str, Any]],
) -> torch.Tensor:
    first_shard = checkpoints[0].weight_map[key]
    first = handles[0][first_shard].get_tensor(key)
    first_shape = tuple(first.shape)
    first_dtype = first.dtype

    if first.is_floating_point():
        averaged = first.to(torch.float32)
        for checkpoint, checkpoint_handles in zip(checkpoints[1:], handles[1:]):
            shard = checkpoint.weight_map[key]
            tensor = checkpoint_handles[shard].get_tensor(key)
            if tuple(tensor.shape) != first_shape or tensor.dtype != first_dtype:
                raise ValueError(
                    f"Tensor mismatch for {key} in {checkpoint.path}: "
                    f"expected shape={first_shape}, dtype={first_dtype}, "
                    f"got shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
                )
            averaged.add_(tensor.to(torch.float32))
            del tensor

        averaged.div_(len(checkpoints))
        result = averaged.to(first_dtype).contiguous()
        del averaged
        del first
        return result

    result = first.contiguous()
    for checkpoint, checkpoint_handles in zip(checkpoints[1:], handles[1:]):
        shard = checkpoint.weight_map[key]
        tensor = checkpoint_handles[shard].get_tensor(key)
        if tuple(tensor.shape) != first_shape or tensor.dtype != first_dtype:
            raise ValueError(
                f"Tensor mismatch for {key} in {checkpoint.path}: "
                f"expected shape={first_shape}, dtype={first_dtype}, "
                f"got shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
            )
        if not torch.equal(result, tensor):
            raise ValueError(f"Non-floating tensor differs across checkpoints: {key}")
        del tensor
    return result


def average_checkpoints(
    checkpoint_paths: list[Path],
    output: Path,
    *,
    overwrite: bool = False,
    log_every: int = 250,
) -> None:
    if len(checkpoint_paths) < 1:
        raise ValueError("At least one checkpoint is required")

    checkpoints = [load_checkpoint(path) for path in checkpoint_paths]
    validate_weight_maps(checkpoints)

    output = output.expanduser().resolve()
    prepare_output_dir(output, overwrite=overwrite)

    first = checkpoints[0]
    shard_names = set(first.weight_map.values())
    copy_hf_sidecar_files(first.path, output, shard_names)

    keys_by_shard = build_keys_by_output_shard(first.weight_map)
    log.info("Averaging %d checkpoints into %s", len(checkpoints), output)
    log.info(
        "Matched %d tensors across %d output shards", len(first.weight_map), len(keys_by_shard)
    )

    with ExitStack() as stack:
        handles: list[dict[str, Any]] = []
        for checkpoint in checkpoints:
            unique_shards = sorted(set(checkpoint.weight_map.values()))
            handles.append(
                {
                    shard_name: stack.enter_context(
                        safe_open(
                            str(checkpoint.path / shard_name),
                            framework="pt",
                            device="cpu",
                        )
                    )
                    for shard_name in unique_shards
                }
            )

        for shard_idx, (shard_name, keys) in enumerate(sorted(keys_by_shard.items()), start=1):
            log.info(
                "Writing shard %d/%d: %s (%d tensors)",
                shard_idx,
                len(keys_by_shard),
                shard_name,
                len(keys),
            )
            shard_tensors: dict[str, torch.Tensor] = {}
            for tensor_idx, key in enumerate(keys, start=1):
                if log_every > 0 and (tensor_idx == 1 or tensor_idx % log_every == 0):
                    log.info("  %s: tensor %d/%d", shard_name, tensor_idx, len(keys))
                shard_tensors[key] = average_tensor(key, checkpoints, handles)

            tmp_path = output / f".tmp-{shard_name}"
            save_file(shard_tensors, str(tmp_path))
            tmp_path.replace(output / shard_name)
            del shard_tensors
            gc.collect()

    with (output / INDEX_NAME).open("w", encoding="utf-8") as f:
        json.dump(first.index, f, indent=2, sort_keys=True)
        f.write("\n")

    log.info("Saved averaged HF checkpoint to %s", output)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    average_checkpoints(
        checkpoint_paths=args.checkpoints,
        output=args.output,
        overwrite=args.overwrite,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
