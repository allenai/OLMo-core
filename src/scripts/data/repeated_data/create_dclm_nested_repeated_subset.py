#!/usr/bin/env python3
"""Materialize a whole-document nested prefix of a sealed DCLM pool.

The base pool's source-document ledger is in global SHA-256 selection order. This
script takes the shortest leading document prefix that reaches ``--target-tokens``
and rematerializes those documents from the already-sealed base token file.
It therefore does not resample DCLM and does not depend on the original source
shards: every selected document is provably a member of the existing base pool.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import numpy as np

DEFAULT_BASE_MANIFEST = Path("src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json")
DEFAULT_MANIFEST_BASE_DIR = Path("/weka/oe-training-default/ai2-llm")
DEFAULT_OUTPUT_PARENT = Path("/weka/oe-training-default/sewonm/icsl/data")
DEFAULT_OUTPUT_MANIFEST_PARENT = Path("src/olmo_core/data/subsets/0802")
DEFAULT_TARGET_TOKENS = 333_000_000
DEFAULT_ALIGNMENT_TOKENS = 4_194_304
SEQUENCE_LENGTH = 4096
EOS_TOKEN_ID = 100257
DTYPE = np.dtype("uint32")


def pool_label(tokens: int) -> str:
    if tokens <= 0:
        raise ValueError("pool token count must be positive")
    for divisor, suffix in ((1_000_000_000, "b"), (1_000_000, "m")):
        if tokens % divisor == 0:
            return f"{tokens // divisor}{suffix}"
    return str(tokens)


def default_output_root(target_tokens: int, base_tokens: int) -> Path:
    return DEFAULT_OUTPUT_PARENT / (
        f"dclm_0802_nested_{pool_label(target_tokens)}_from_{pool_label(base_tokens)}"
    )


def default_output_manifest(target_tokens: int) -> Path:
    return DEFAULT_OUTPUT_MANIFEST_PARENT / (
        f"dclm_0802_repeated_train_{pool_label(target_tokens)}.json"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(entries: list[dict]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def deterministic_gzip_text_writer(path: Path) -> Iterator[TextIO]:
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.TextIOWrapper(compressed, newline="") as text,
    ):
        yield text


def resolve_artifact(base_dir: Path, value: str) -> Path:
    return (base_dir / value).resolve()


def load_ledger(path: Path, target_tokens: int) -> tuple[list[tuple], int, int]:
    rows: list[tuple] = []
    selected_rows: list[tuple] = []
    real_tokens = 0
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {
            "shard_path",
            "document_index",
            "start",
            "end",
            "selection_key",
            "partition_key",
        }
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"unexpected base-pool ledger columns: {reader.fieldnames}")
        for rank, item in enumerate(reader):
            start = int(item["start"])
            end = int(item["end"])
            if end <= start:
                raise ValueError(f"invalid document interval at ledger rank {rank}")
            row = (
                sys.intern(item["shard_path"]),
                int(item["document_index"]),
                start,
                end,
                item["selection_key"],
                item["partition_key"],
                rank,
            )
            rows.append(row)
            if real_tokens < target_tokens:
                selected_rows.append(row)
                real_tokens += end - start
    if real_tokens < target_tokens:
        raise RuntimeError("base-pool ledger is too small for the requested nested prefix")
    return rows, len(selected_rows), real_tokens


def source_spans(
    rows: list[tuple], metadata_path: Path, selected_documents: int
) -> tuple[list[tuple[int, int, int]], int]:
    """Map selected ledger documents to their offsets in the base-pool token file."""
    rows.sort(key=lambda item: (item[0], item[2], item[3], item[1]))
    selected: list[tuple[int, int, int]] = []
    with gzip.open(metadata_path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        for materialized_index, row in enumerate(rows):
            try:
                raw_start, raw_end = next(reader)[:2]
            except StopIteration as error:
                raise RuntimeError("base-pool metadata ended before its document ledger") from error
            source_start, source_end = int(raw_start), int(raw_end)
            if source_end - source_start != row[3] - row[2]:
                raise RuntimeError(
                    "base-pool ledger/materialization length mismatch at "
                    f"materialized document {materialized_index}"
                )
            if row[6] < selected_documents:
                selected.append((source_start, source_end, row[6]))
        trailing = list(reader)
    if len(trailing) > 1:
        raise RuntimeError("base-pool metadata has unexpected trailing document rows")
    if len(selected) != selected_documents:
        raise AssertionError((len(selected), selected_documents))
    padding = 0
    if trailing:
        padding = int(trailing[0][1]) - int(trailing[0][0])
        if padding < 0:
            raise RuntimeError("base-pool metadata has negative trailing padding")
    return selected, padding


def copy_documents(
    source_path: Path,
    output_path: Path,
    metadata_path: Path,
    spans: list[tuple[int, int, int]],
    *,
    real_tokens: int,
    alignment_tokens: int,
    eos_token_id: int,
) -> dict:
    output_tokens = math.ceil(real_tokens / alignment_tokens) * alignment_tokens
    padding_tokens = output_tokens - real_tokens
    token_tmp = output_path.with_name(output_path.name + ".tmp")
    metadata_tmp = metadata_path.with_name(metadata_path.name + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(output_path if output_path.exists() else metadata_path)

    offset = 0
    try:
        with (
            source_path.open("rb") as source,
            token_tmp.open("wb") as output,
            deterministic_gzip_text_writer(metadata_tmp) as metadata,
        ):
            writer = csv.writer(metadata, lineterminator="\n")
            for source_start, source_end, _ in spans:
                length = source_end - source_start
                source.seek(source_start * DTYPE.itemsize)
                remaining = length * DTYPE.itemsize
                while remaining:
                    payload = source.read(min(16 * 1024 * 1024, remaining))
                    if not payload:
                        raise RuntimeError("short read from base-pool materialization")
                    output.write(payload)
                    remaining -= len(payload)
                writer.writerow((offset, offset + length))
                offset += length
            if padding_tokens:
                padding = np.full(padding_tokens, eos_token_id, dtype=DTYPE)
                output.write(padding.tobytes(order="C"))
                writer.writerow((offset, offset + padding_tokens))
                offset += padding_tokens
        if offset != output_tokens:
            raise AssertionError((offset, output_tokens))
        os.replace(token_tmp, output_path)
        os.replace(metadata_tmp, metadata_path)
    except BaseException:
        token_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
        raise
    return {
        "path": str(output_path),
        "token_sha256": sha256_file(output_path),
        "document_metadata_path": str(metadata_path),
        "document_metadata_sha256": sha256_file(metadata_path),
        "real_document_tokens": real_tokens,
        "padding_eos_tokens": padding_tokens,
        "materialized_tokens": output_tokens,
        "instances": output_tokens // SEQUENCE_LENGTH,
    }


def write_ledger(path: Path, rows: list[tuple]) -> dict:
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + ".tmp")
    with deterministic_gzip_text_writer(tmp) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "shard_path",
                "document_index",
                "start",
                "end",
                "selection_key",
                "partition_key",
            ]
        )
        for row in rows:
            writer.writerow(row[:6])
    os.replace(tmp, path)
    return {"path": str(path), "sha256": sha256_file(path), "documents": len(rows)}


def validate_base(base: dict, base_manifest: Path, base_dir: Path, verify_tokens: bool) -> dict:
    if base.get("format") != "olmo-token-subset-v1":
        raise ValueError("base manifest is not an olmo-token-subset-v1 manifest")
    selection = base.get("selection", {})
    requested_tokens = int(selection.get("requested_tokens", 0))
    if requested_tokens <= 0:
        raise ValueError("base manifest has an invalid requested-token count")
    if selection.get("method") != "global-sha256-document-order-prefix":
        raise ValueError("base pool does not use the expected global document order")
    if selection.get("domain") != "dclm-train-repeated-sample-v1":
        raise ValueError("base pool has the wrong selection domain")
    if int(selection.get("seed", -1)) != 1:
        raise ValueError("base pool has the wrong selection seed")
    entries = base.get("entries", [])
    if manifest_digest(entries) != base.get("entries_sha256"):
        raise RuntimeError("base pool entry checksum mismatch")
    nestedness_audit = base.get("nestedness_audit")
    if nestedness_audit is not None and nestedness_audit.get("passed") is not True:
        raise RuntimeError("base pool nestedness audit did not pass")
    token_path = resolve_artifact(base_dir, base["materialized"]["path"])
    metadata_path = resolve_artifact(base_dir, base["materialized"]["document_metadata_path"])
    ledger_path = resolve_artifact(base_dir, base["source_document_ledger"]["path"])
    for path in (token_path, metadata_path, ledger_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(metadata_path) != base["materialized"]["document_metadata_sha256"]:
        raise RuntimeError("base-pool document metadata hash mismatch")
    if sha256_file(ledger_path) != base["source_document_ledger"]["sha256"]:
        raise RuntimeError("base-pool source-document ledger hash mismatch")
    if verify_tokens and sha256_file(token_path) != base["materialized"]["token_sha256"]:
        raise RuntimeError("base-pool token materialization hash mismatch")
    return {
        "manifest": str(base_manifest),
        "manifest_sha256": sha256_file(base_manifest),
        "requested_tokens": requested_tokens,
        "token_path": token_path,
        "metadata_path": metadata_path,
        "ledger_path": ledger_path,
    }


def build(args: argparse.Namespace) -> dict:
    base = json.loads(args.base_manifest.read_text())
    base_artifacts = validate_base(
        base, args.base_manifest, args.base_manifest_base_dir, not args.skip_base_token_hash
    )
    base_requested_tokens = int(base_artifacts["requested_tokens"])
    if args.target_tokens >= base_requested_tokens:
        raise ValueError(
            "nested target must be strictly below the base pool's requested-token count"
        )
    target_label = pool_label(args.target_tokens)
    base_label = pool_label(base_requested_tokens)
    output_root = args.output_root or default_output_root(args.target_tokens, base_requested_tokens)
    output_manifest = args.output_manifest or default_output_manifest(args.target_tokens)
    base_manifest_reference = args.base_manifest_reference or str(args.base_manifest)
    rows, selected_documents, real_tokens = load_ledger(
        base_artifacts["ledger_path"], args.target_tokens
    )
    base_document_count = len(rows)
    if base_document_count != int(base["selection"]["selected_documents"]):
        raise RuntimeError("base-pool manifest/ledger document count mismatch")
    selected_rows = list(rows[:selected_documents])
    if selected_rows[-1][4] >= base["selection"]["boundary_key"]:
        raise RuntimeError("nested prefix boundary is not strictly inside the base pool")
    spans, base_padding = source_spans(rows, base_artifacts["metadata_path"], selected_documents)
    spans.sort(key=lambda item: item[2])
    # Restore the base-pool materialization order, not SHA selection order.
    by_rank = {rank: (start, end) for start, end, rank in spans}
    materialized_spans = [
        by_rank[row[6]] for row in sorted(selected_rows, key=lambda x: (x[0], x[2], x[3], x[1]))
    ]

    output_path = output_root / f"dclm_0802_repeated_train_{target_label}_uint32.npy"
    metadata_path = output_path.with_suffix(".csv.gz")
    ledger_path = output_root / f"dclm_0802_repeated_train_{target_label}.documents.csv.gz"
    output_root.mkdir(parents=True, exist_ok=True)
    materialized = copy_documents(
        base_artifacts["token_path"],
        output_path,
        metadata_path,
        [(start, end, 0) for start, end in materialized_spans],
        real_tokens=real_tokens,
        alignment_tokens=args.alignment_tokens,
        eos_token_id=args.eos_token_id,
    )
    ledger = write_ledger(ledger_path, selected_rows)
    relative_data = os.path.relpath(output_path, args.manifest_base_dir)
    relative_metadata = os.path.relpath(metadata_path, args.manifest_base_dir)
    relative_ledger = os.path.relpath(ledger_path, args.manifest_base_dir)
    entry = {
        "path": relative_data,
        "start_instance": 0,
        "num_instances": materialized["instances"],
        "available_instances": materialized["instances"],
        "available_tokens": materialized["materialized_tokens"],
    }
    manifest = {
        "format": "olmo-token-subset-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": f"repeated_train_{target_label}_nested_in_{base_label}",
        "source": {
            **base.get("source", {}),
            # Keep this as a repository-relative provenance pointer so a manifest
            # materialized from a Weka-mounted session is byte-for-byte portable.
            "nested_base_manifest": base_manifest_reference,
            "nested_base_manifest_sha256": base_artifacts["manifest_sha256"],
        },
        "selection": {
            "method": "global-sha256-document-order-prefix",
            "identity": base["selection"]["identity"],
            "domain": base["selection"]["domain"],
            "seed": int(base["selection"]["seed"]),
            "requested_tokens": args.target_tokens,
            "sequence_length": SEQUENCE_LENGTH,
            "selected_documents": selected_documents,
            "selected_real_document_tokens": real_tokens,
            "padding_eos_tokens": materialized["padding_eos_tokens"],
            "selected_tokens": materialized["materialized_tokens"],
            "selected_instances": materialized["instances"],
            "boundary_key": selected_rows[-1][4],
        },
        "materialized": {
            "path": relative_data,
            "token_sha256": materialized["token_sha256"],
            "document_metadata_path": relative_metadata,
            "document_metadata_sha256": materialized["document_metadata_sha256"],
        },
        "source_document_ledger": {
            **ledger,
            "path": relative_ledger,
        },
        "entries_sha256": manifest_digest([entry]),
        "entries": [entry],
        "nestedness_audit": {
            "base_requested_tokens": base_requested_tokens,
            "base_selected_documents": base_document_count,
            "base_padding_eos_tokens": base_padding,
            "selection_is_exact_leading_ledger_prefix": True,
            "all_selected_documents_are_in_base": True,
            "selected_document_intersection_with_base": selected_documents,
            "document_boundary_preserving": True,
            "boundary_is_strictly_before_base_boundary": True,
            "passed": True,
        },
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if output_manifest.exists():
        raise FileExistsError(output_manifest)
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["_output_manifest"] = str(output_manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--base-manifest-reference")
    parser.add_argument("--base-manifest-base-dir", type=Path, default=DEFAULT_MANIFEST_BASE_DIR)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--manifest-base-dir", type=Path, default=DEFAULT_MANIFEST_BASE_DIR)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--alignment-tokens", type=int, default=DEFAULT_ALIGNMENT_TOKENS)
    parser.add_argument("--eos-token-id", type=int, default=EOS_TOKEN_ID)
    parser.add_argument("--skip-base-token-hash", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_tokens <= 0:
        raise ValueError("nested target must be positive")
    if args.alignment_tokens <= 0 or args.alignment_tokens % SEQUENCE_LENGTH:
        raise ValueError("alignment must be a positive multiple of sequence length")
    manifest = build(args)
    print(
        json.dumps(
            {
                "manifest": manifest.pop("_output_manifest"),
                "output": manifest["materialized"]["path"],
                "selection": manifest["selection"],
                "nestedness_audit": manifest["nestedness_audit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
