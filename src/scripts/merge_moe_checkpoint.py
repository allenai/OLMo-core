#!/usr/bin/env python3
"""
Average the main weights from multiple OLMo MoE checkpoints.

The output checkpoint keeps the wrapper structure from the last input
checkpoint, but ``model_and_optim`` is reconstructed from values rather than
copied chunk-for-chunk. Only ``*.main`` tensors are averaged across inputs.
All other entries come from the last input checkpoint by default.
"""

from __future__ import annotations

import argparse
import io
import logging
import pickle
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorStorageMetadata,
)

from olmo_core.distributed.checkpoint import RemoteFileSystemReader
from olmo_core.distributed.checkpoint.filesystem import _StorageInfo
from olmo_core.io import file_exists, join_path, normalize_path

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average the main weights from multiple OLMo MoE checkpoints."
    )
    parser.add_argument(
        "input_checkpoints",
        nargs="+",
        help="Input checkpoint roots, e.g. step153000 step153250 step153500",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output checkpoint root. Defaults to '<last-input>-avg'.",
    )
    parser.add_argument(
        "--save-overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument(
        "--load-thread-count",
        type=int,
        default=2,
        help="Thread count for reading distributed checkpoint data.",
    )
    parser.add_argument(
        "--reset-optimizer-moments",
        action="store_true",
        help="Reset '*.exp_avg' and '*.exp_avg_sq' tensors to zeros in the output checkpoint.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def resolve_checkpoint_dirs(path: str) -> Tuple[str, str]:
    checkpoint_dir = normalize_path(path)
    model_and_optim_dir = normalize_path(join_path(checkpoint_dir, "model_and_optim"))

    if file_exists(join_path(model_and_optim_dir, ".metadata")):
        return checkpoint_dir, model_and_optim_dir

    if file_exists(join_path(checkpoint_dir, ".metadata")):
        checkpoint_path = Path(checkpoint_dir)
        if checkpoint_path.name != "model_and_optim":
            raise FileNotFoundError(
                f"Expected either '{checkpoint_dir}/model_and_optim/.metadata' or a model_and_optim "
                f"directory path, but found only '{checkpoint_dir}/.metadata'."
            )
        return normalize_path(checkpoint_path.parent), checkpoint_dir

    raise FileNotFoundError(
        f"Checkpoint '{checkpoint_dir}' is missing 'model_and_optim/.metadata'."
    )


def read_metadata(reader: RemoteFileSystemReader) -> Metadata:
    metadata = reader.read_metadata()
    if not isinstance(metadata, Metadata):
        raise TypeError(f"Unexpected metadata type: {type(metadata)}")
    return metadata


def main_tensor_keys(metadata: Metadata) -> List[str]:
    keys = []
    for key, meta in metadata.state_dict_metadata.items():
        if key.endswith(".main"):
            if not isinstance(meta, TensorStorageMetadata):
                raise TypeError(f"Expected tensor metadata for '{key}', found {type(meta)}")
            keys.append(key)
    return sorted(keys)


def tensor_signature(meta: TensorStorageMetadata) -> Tuple[torch.dtype, Tuple[int, ...]]:
    return meta.properties.dtype, tuple(meta.size)


def validate_inputs(checkpoint_dirs: Sequence[str], metadatas: Sequence[Metadata]) -> List[str]:
    if not checkpoint_dirs:
        raise ValueError("No input checkpoints provided")

    reference_keys = main_tensor_keys(metadatas[0])
    reference_key_set = set(reference_keys)

    for checkpoint_dir, metadata in zip(checkpoint_dirs[1:], metadatas[1:]):
        current_keys = main_tensor_keys(metadata)
        current_key_set = set(current_keys)
        if current_key_set != reference_key_set:
            missing = sorted(reference_key_set - current_key_set)[:10]
            extra = sorted(current_key_set - reference_key_set)[:10]
            raise ValueError(
                f"Checkpoint '{checkpoint_dir}' has a different set of main tensors. "
                f"Missing sample={missing}, extra sample={extra}"
            )

    for key in reference_keys:
        ref_meta = metadatas[0].state_dict_metadata[key]
        assert isinstance(ref_meta, TensorStorageMetadata)
        ref_signature = tensor_signature(ref_meta)

        for checkpoint_dir, metadata in zip(checkpoint_dirs[1:], metadatas[1:]):
            current_meta = metadata.state_dict_metadata[key]
            if not isinstance(current_meta, TensorStorageMetadata):
                raise TypeError(f"Expected tensor metadata for '{key}' in '{checkpoint_dir}'")
            if tensor_signature(current_meta) != ref_signature:
                raise ValueError(
                    f"Checkpoint '{checkpoint_dir}' has incompatible global tensor metadata for '{key}': "
                    f"expected {ref_signature}, got {tensor_signature(current_meta)}"
                )

    return reference_keys


def copy_checkpoint_wrapper(src_checkpoint_dir: str, output_dir: Path, save_overwrite: bool) -> None:
    src_dir = Path(src_checkpoint_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {src_dir}")

    if output_dir.exists():
        if not save_overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --save-overwrite to replace it."
            )
        remove_path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        if src.name == "model_and_optim":
            continue
        dst = output_dir / src.name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def load_full_tensor(
    reader: RemoteFileSystemReader,
    checkpoint_dir: str,
    metadata: Metadata,
    key: str,
) -> torch.Tensor:
    meta = metadata.state_dict_metadata[key]
    if not isinstance(meta, TensorStorageMetadata):
        raise TypeError(f"Expected tensor metadata for '{key}', found {type(meta)}")

    tensor = torch.empty(meta.size, dtype=meta.properties.dtype, device="cpu")
    dist_cp.state_dict_loader.load(
        {key: tensor},
        checkpoint_id=checkpoint_dir,
        storage_reader=reader,
        process_group=None,
    )
    return tensor


def load_object_entry(
    reader: RemoteFileSystemReader,
    checkpoint_dir: str,
    key: str,
) -> Any:
    state_dict = {key: io.BytesIO()}
    dist_cp.state_dict_loader.load(
        state_dict,
        checkpoint_id=checkpoint_dir,
        storage_reader=reader,
        process_group=None,
    )
    return state_dict[key]


def average_full_tensor(
    *,
    key: str,
    readers: Sequence[RemoteFileSystemReader],
    checkpoint_dirs: Sequence[str],
    metadatas: Sequence[Metadata],
) -> torch.Tensor:
    avg_tensor: torch.Tensor | None = None
    target_dtype: torch.dtype | None = None

    for reader, checkpoint_dir, metadata in zip(readers, checkpoint_dirs, metadatas):
        tensor = load_full_tensor(reader, checkpoint_dir, metadata, key)

        if avg_tensor is None:
            target_dtype = tensor.dtype
            acc_dtype = torch.float64 if tensor.dtype.is_floating_point else tensor.dtype
            avg_tensor = tensor.to(dtype=acc_dtype)
        else:
            avg_tensor.add_(tensor.to(dtype=avg_tensor.dtype))

    assert avg_tensor is not None
    assert target_dtype is not None
    if len(checkpoint_dirs) > 1:
        avg_tensor.div_(len(checkpoint_dirs))
    if avg_tensor.dtype != target_dtype:
        avg_tensor = avg_tensor.to(dtype=target_dtype)

    return avg_tensor.contiguous()


def zero_tensor_from_meta(meta: TensorStorageMetadata) -> torch.Tensor:
    return torch.zeros(meta.size, dtype=meta.properties.dtype, device="cpu")


def should_reset_optimizer_moment(key: str, reset_optimizer_moments: bool) -> bool:
    return reset_optimizer_moments and (
        key.endswith(".exp_avg") or key.endswith(".exp_avg_sq")
    )


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor.clone(), buffer)
    return buffer.getvalue()


def serialize_object(obj: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def full_tensor_chunk_meta(meta: TensorStorageMetadata) -> ChunkStorageMetadata:
    zeros = torch.Size([0] * len(meta.size))
    return ChunkStorageMetadata(offsets=zeros, sizes=meta.size)


def identity_planner_data(metadata: Metadata) -> Dict[str, Tuple[str]]:
    if isinstance(metadata.planner_data, dict):
        return {key: tuple(value) for key, value in metadata.planner_data.items()}
    return {key: (key,) for key in metadata.state_dict_metadata.keys()}


def build_output_entry(
    *,
    key: str,
    meta: TensorStorageMetadata | BytesStorageMetadata,
    readers: Sequence[RemoteFileSystemReader],
    checkpoint_dirs: Sequence[str],
    metadatas: Sequence[Metadata],
    reset_optimizer_moments: bool,
) -> bytes:
    if isinstance(meta, TensorStorageMetadata):
        if key.endswith(".main"):
            tensor = average_full_tensor(
                key=key,
                readers=readers,
                checkpoint_dirs=checkpoint_dirs,
                metadatas=metadatas,
            )
        elif should_reset_optimizer_moment(key, reset_optimizer_moments):
            tensor = zero_tensor_from_meta(meta)
        else:
            tensor = load_full_tensor(readers[-1], checkpoint_dirs[-1], metadatas[-1], key)
        return serialize_tensor(tensor)

    obj = load_object_entry(readers[-1], checkpoint_dirs[-1], key)
    return serialize_object(obj)


def write_reconstructed_model_and_optim(
    *,
    input_model_and_optim_dirs: Sequence[str],
    readers: Sequence[RemoteFileSystemReader],
    input_metadatas: Sequence[Metadata],
    output_dir: Path,
    reset_optimizer_moments: bool,
) -> None:
    template_metadata = input_metadatas[-1]
    output_model_dir = output_dir / "model_and_optim"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    output_state_metadata: Dict[str, TensorStorageMetadata | BytesStorageMetadata] = {}
    output_storage_data: Dict[MetadataIndex, _StorageInfo] = {}
    planner_data = identity_planner_data(template_metadata)

    items = list(template_metadata.state_dict_metadata.items())
    total = len(items)

    for idx, (key, meta) in enumerate(items):
        log.info("Writing %s (%d/%d)", key, idx + 1, total)

        entry_bytes = build_output_entry(
            key=key,
            meta=meta,
            readers=readers,
            checkpoint_dirs=input_model_and_optim_dirs,
            metadatas=input_metadatas,
            reset_optimizer_moments=reset_optimizer_moments,
        )

        relative_path = f"__{idx}.distcp"
        with (output_model_dir / relative_path).open("wb") as f:
            f.write(entry_bytes)

        if isinstance(meta, TensorStorageMetadata):
            chunk = full_tensor_chunk_meta(meta)
            output_state_metadata[key] = TensorStorageMetadata(
                properties=meta.properties,
                size=meta.size,
                chunks=[chunk],
            )
            output_storage_data[
                MetadataIndex(fqn=key, offset=chunk.offsets, index=0)
            ] = _StorageInfo(relative_path=relative_path, offset=0, length=len(entry_bytes))
        else:
            output_state_metadata[key] = BytesStorageMetadata()
            output_storage_data[
                MetadataIndex(fqn=key, offset=None, index=None)
            ] = _StorageInfo(relative_path=relative_path, offset=0, length=len(entry_bytes))

    output_metadata = Metadata(
        state_dict_metadata=output_state_metadata,
        planner_data=planner_data,
        storage_data=output_storage_data,
        storage_meta=StorageMeta(
            checkpoint_id=str(output_model_dir),
            save_id=uuid.uuid4().hex,
            modules=[],
        ),
        version=template_metadata.version,
    )

    with (output_model_dir / ".metadata").open("wb") as f:
        pickle.dump(output_metadata, f)


def write_input_tracking_file(output_dir: Path, input_checkpoint_dirs: Sequence[str]) -> None:
    tracking_path = output_dir / "averaged_from_checkpoints.txt"
    with tracking_path.open("w", encoding="utf-8") as f:
        for checkpoint_dir in input_checkpoint_dirs:
            f.write(f"{checkpoint_dir}\n")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    resolved_inputs = [resolve_checkpoint_dirs(path) for path in args.input_checkpoints]
    input_checkpoint_dirs = [checkpoint_dir for checkpoint_dir, _ in resolved_inputs]
    input_model_and_optim_dirs = [model_dir for _, model_dir in resolved_inputs]

    output_dir = (
        Path(normalize_path(args.output_dir))
        if args.output_dir is not None
        else Path(f"{input_checkpoint_dirs[-1]}-avg")
    )

    if normalize_path(output_dir) in {normalize_path(path) for path in input_checkpoint_dirs}:
        raise ValueError("Output directory must be different from all input checkpoint directories")

    readers = [
        RemoteFileSystemReader(model_dir, thread_count=args.load_thread_count)
        for model_dir in input_model_and_optim_dirs
    ]
    input_metadatas = [read_metadata(reader) for reader in readers]
    validate_inputs(input_checkpoint_dirs, input_metadatas)

    copy_checkpoint_wrapper(
        input_checkpoint_dirs[-1],
        output_dir,
        save_overwrite=args.save_overwrite,
    )
    write_reconstructed_model_and_optim(
        input_model_and_optim_dirs=input_model_and_optim_dirs,
        readers=readers,
        input_metadatas=input_metadatas,
        output_dir=output_dir,
        reset_optimizer_moments=args.reset_optimizer_moments,
    )
    write_input_tracking_file(output_dir, input_checkpoint_dirs)

    log.info("Wrote averaged checkpoint to %s", output_dir)


if __name__ == "__main__":
    main()
