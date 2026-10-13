#!/usr/bin/env python3
"""Flatten a sealed composite subset without changing its documents or order.

Dynamic repacking requires one materialized token file and one document-offset
file.  Nested pool manifests deliberately retain their independently
materialized components, so this utility concatenates only the real document
tokens, shifts the component document offsets, and moves all alignment padding
to the end.  The resulting manifest retains the composite selection metadata
and records the exact source-manifest digest.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import io
import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Iterator, TextIO

import numpy as np

COPY_BYTES = 64 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entries_sha256(entries: list[dict[str, Any]]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def resolve(base_dir: Path, relative_path: str) -> Path:
    return (base_dir / relative_path.lstrip("/")).resolve()


@contextmanager
def deterministic_gzip_writer(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed) as text:
                yield text


def copy_prefix(
    source: Path,
    output: BinaryIO,
    digests: tuple[Any, ...],
    byte_count: int,
) -> None:
    remaining = byte_count
    with source.open("rb") as handle:
        while remaining:
            payload = handle.read(min(COPY_BYTES, remaining))
            if not payload:
                raise EOFError(f"{source} ended with {remaining:,} requested bytes remaining")
            output.write(payload)
            for digest in digests:
                digest.update(payload)
            remaining -= len(payload)


def component_info(
    composite: dict[str, Any], composite_path: Path, base_dir: Path, dtype: np.dtype
) -> list[dict[str, Any]]:
    declared = composite.get("components")
    if not isinstance(declared, list) or len(declared) < 2:
        raise ValueError("source must be a composite manifest with at least two components")
    result: list[dict[str, Any]] = []
    for index, component_ref in enumerate(declared):
        manifest_path = resolve(base_dir, str(component_ref["manifest"]))
        expected_digest = str(component_ref["manifest_sha256"])
        actual_digest = sha256_file(manifest_path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"component {index} manifest digest mismatch: {actual_digest} != {expected_digest}"
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != "olmo-token-subset-v1":
            raise ValueError(f"component {index} is not an OLMo token subset")
        entries = manifest.get("entries") or []
        if len(entries) != 1 or int(entries[0].get("start_instance", -1)) != 0:
            raise ValueError(f"component {index} is not fully materialized from instance zero")
        selection = manifest["selection"]
        materialized = manifest.get("materialized") or {}
        token_path = resolve(base_dir, str(materialized["path"]))
        metadata_path = resolve(base_dir, str(materialized["document_metadata_path"]))
        real_tokens = int(selection["selected_real_document_tokens"])
        selected_tokens = int(selection["selected_tokens"])
        documents = int(selection["selected_documents"])
        if token_path.stat().st_size != selected_tokens * dtype.itemsize:
            raise ValueError(f"component {index} token-file size does not match its manifest")
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata_sha256 = sha256_file(metadata_path)
        if metadata_sha256 != materialized.get("document_metadata_sha256"):
            raise ValueError(f"component {index} document-metadata digest mismatch")
        result.append(
            {
                "manifest": manifest,
                "manifest_path": manifest_path,
                "manifest_sha256": actual_digest,
                "token_path": token_path,
                "metadata_path": metadata_path,
                "token_sha256": str(materialized["token_sha256"]),
                "metadata_sha256": metadata_sha256,
                "real_tokens": real_tokens,
                "selected_tokens": selected_tokens,
                "documents": documents,
            }
        )
    if sum(item["real_tokens"] for item in result) != int(
        composite["selection"]["selected_real_document_tokens"]
    ):
        raise ValueError("component real-token counts do not sum to the composite selection")
    if sum(item["documents"] for item in result) != int(
        composite["selection"]["selected_documents"]
    ):
        raise ValueError("component document counts do not sum to the composite selection")
    return result


def copy_metadata(
    source: Path,
    output: TextIO,
    *,
    documents: int,
    real_tokens: int,
    shift: int,
) -> None:
    expected_start = 0
    rows = 0
    with gzip.open(source, "rt") as handle:
        for raw in handle:
            if rows >= documents:
                break
            fields = raw.rstrip("\n").split(",")
            if len(fields) < 2:
                raise ValueError(f"invalid metadata row {rows + 1} in {source}")
            start, end = int(fields[0]), int(fields[1])
            if start != expected_start or end <= start or end > real_tokens:
                raise ValueError(
                    f"non-contiguous document [{start}, {end}) at row {rows + 1} in {source}"
                )
            output.write(f"{shift + start},{shift + end}\n")
            expected_start = end
            rows += 1
    if rows != documents or expected_start != real_tokens:
        raise ValueError(
            f"metadata {source} covers {rows:,}/{documents:,} documents and "
            f"{expected_start:,}/{real_tokens:,} real tokens"
        )


def validate_existing(
    output_manifest: Path,
    *,
    source_sha256: str,
    base_dir: Path,
    dtype: np.dtype,
) -> dict[str, Any]:
    manifest = json.loads(output_manifest.read_text())
    flattened = manifest.get("flattened_from") or {}
    if flattened.get("manifest_sha256") != source_sha256:
        raise ValueError("existing flat manifest was produced from a different composite source")
    entries = manifest.get("entries") or []
    if len(entries) != 1 or int(entries[0].get("start_instance", -1)) != 0:
        raise ValueError("existing flat manifest is not a one-entry materialization")
    materialized = manifest.get("materialized") or {}
    token_path = resolve(base_dir, str(materialized["path"]))
    metadata_path = resolve(base_dir, str(materialized["document_metadata_path"]))
    expected_tokens = int(manifest["selection"]["selected_tokens"])
    if token_path.stat().st_size != expected_tokens * dtype.itemsize:
        raise ValueError("existing flattened token file has an unexpected size")
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    return manifest


def ensure_flattened_pool(
    *,
    composite_manifest: Path,
    output_manifest: Path,
    output_tokens: Path,
    manifest_base_dir: Path,
    dtype_name: str = "uint32",
) -> dict[str, Any]:
    dtype = np.dtype(dtype_name)
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f"token dtype must be integral, got {dtype}")
    source_sha256 = sha256_file(composite_manifest)
    composite = json.loads(composite_manifest.read_text())
    if composite.get("format") != "olmo-token-subset-v1":
        raise ValueError("source is not an OLMo token subset manifest")
    sequence_length = int(composite["selection"]["sequence_length"])
    selected_tokens = int(composite["selection"]["selected_tokens"])
    selected_instances = int(composite["selection"]["selected_instances"])
    if selected_instances * sequence_length != selected_tokens:
        raise ValueError("composite instance and token counts are inconsistent")

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_tokens.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_manifest.with_name(output_manifest.name + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output_manifest.is_file():
            manifest = validate_existing(
                output_manifest,
                source_sha256=source_sha256,
                base_dir=manifest_base_dir,
                dtype=dtype,
            )
            print(
                f"POOL3B_FLAT_READY manifest={output_manifest} source_sha256={source_sha256}",
                flush=True,
            )
            return manifest

        components = component_info(composite, composite_manifest, manifest_base_dir, dtype)
        token_tmp = output_tokens.with_name(output_tokens.name + f".tmp.{os.getpid()}")
        metadata_path = output_tokens.with_suffix(".csv.gz")
        metadata_tmp = metadata_path.with_name(metadata_path.name + f".tmp.{os.getpid()}")
        manifest_tmp = output_manifest.with_name(output_manifest.name + f".tmp.{os.getpid()}")
        for path in (token_tmp, metadata_tmp, manifest_tmp):
            path.unlink(missing_ok=True)
        digest = hashlib.sha256()
        real_offset = 0
        eos_token: int | None = None
        try:
            with token_tmp.open("wb") as token_output, deterministic_gzip_writer(
                metadata_tmp
            ) as metadata_output:
                for component in components:
                    real_tokens = int(component["real_tokens"])
                    component_digest = hashlib.sha256()
                    copy_prefix(
                        component["token_path"],
                        token_output,
                        (digest, component_digest),
                        real_tokens * dtype.itemsize,
                    )
                    with component["token_path"].open("rb") as source:
                        source.seek((real_tokens - 1) * dtype.itemsize)
                        value = int(np.frombuffer(source.read(dtype.itemsize), dtype=dtype)[0])
                    if eos_token is None:
                        eos_token = value
                    elif value != eos_token:
                        raise ValueError("component documents do not share one EOS token")
                    padding_bytes = (
                        int(component["selected_tokens"]) - real_tokens
                    ) * dtype.itemsize
                    with component["token_path"].open("rb") as source:
                        source.seek(real_tokens * dtype.itemsize)
                        remaining_padding = padding_bytes
                        while remaining_padding:
                            payload = source.read(min(COPY_BYTES, remaining_padding))
                            if not payload:
                                raise EOFError(
                                    "component token file ended inside alignment padding"
                                )
                            values = np.frombuffer(payload, dtype=dtype)
                            if np.any(values != eos_token):
                                raise ValueError("component alignment padding is not all EOS")
                            component_digest.update(payload)
                            remaining_padding -= len(payload)
                    if component_digest.hexdigest() != component["token_sha256"]:
                        raise ValueError("component token-file digest mismatch")
                    copy_metadata(
                        component["metadata_path"],
                        metadata_output,
                        documents=int(component["documents"]),
                        real_tokens=real_tokens,
                        shift=real_offset,
                    )
                    real_offset += real_tokens

                padding_tokens = selected_tokens - real_offset
                if padding_tokens < 0 or eos_token is None:
                    raise ValueError("invalid composite padding or missing EOS token")
                chunk_tokens = max(1, COPY_BYTES // dtype.itemsize)
                padding_chunk = np.full(min(chunk_tokens, padding_tokens), eos_token, dtype=dtype)
                remaining = padding_tokens
                while remaining:
                    payload = padding_chunk[: min(remaining, len(padding_chunk))].tobytes()
                    token_output.write(payload)
                    digest.update(payload)
                    remaining -= len(payload) // dtype.itemsize

            if token_tmp.stat().st_size != selected_tokens * dtype.itemsize:
                raise ValueError("flattened token output has an unexpected size")
            metadata_sha256 = sha256_file(metadata_tmp)
            token_relative = os.path.relpath(output_tokens, manifest_base_dir)
            metadata_relative = os.path.relpath(metadata_path, manifest_base_dir)
            entry = {
                "path": token_relative,
                "start_instance": 0,
                "num_instances": selected_instances,
                "available_instances": selected_instances,
                "available_tokens": selected_tokens,
            }
            manifest = dict(composite)
            manifest.update(
                {
                    "created_at": datetime.now(tz=UTC).isoformat(),
                    "name": f"{composite.get('name', composite_manifest.stem)}_flat_dynamic_repacking",
                    "materialized": {
                        "path": token_relative,
                        "token_sha256": digest.hexdigest(),
                        "document_metadata_path": metadata_relative,
                        "document_metadata_sha256": metadata_sha256,
                    },
                    "flattened_from": {
                        "manifest": os.path.relpath(composite_manifest, manifest_base_dir),
                        "manifest_sha256": source_sha256,
                        "method": "concatenate_real_document_tokens_then_terminal_alignment_padding",
                        "document_order_preserved": True,
                        "component_manifest_sha256": [
                            item["manifest_sha256"] for item in components
                        ],
                    },
                    "entries_sha256": entries_sha256([entry]),
                    "entries": [entry],
                }
            )
            manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
            os.replace(token_tmp, output_tokens)
            os.replace(metadata_tmp, metadata_path)
            os.replace(manifest_tmp, output_manifest)
        except BaseException:
            for path in (token_tmp, metadata_tmp, manifest_tmp):
                path.unlink(missing_ok=True)
            raise

        print(
            f"POOL3B_FLAT_CREATED manifest={output_manifest} tokens={selected_tokens} "
            f"documents={composite['selection']['selected_documents']} "
            f"source_sha256={source_sha256}",
            flush=True,
        )
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composite-manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--output-tokens", required=True, type=Path)
    parser.add_argument(
        "--manifest-base-dir",
        type=Path,
        default=Path("/weka/oe-training-default/ai2-llm"),
    )
    parser.add_argument("--dtype", default="uint32")
    args = parser.parse_args()
    ensure_flattened_pool(
        composite_manifest=args.composite_manifest,
        output_manifest=args.output_manifest,
        output_tokens=args.output_tokens,
        manifest_base_dir=args.manifest_base_dir,
        dtype_name=args.dtype,
    )


if __name__ == "__main__":
    main()
