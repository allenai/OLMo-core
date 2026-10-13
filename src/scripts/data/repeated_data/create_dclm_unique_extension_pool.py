#!/usr/bin/env python3
"""Materialize an appendable, document-disjoint continuation of the 0802 unique pool.

The logical pool starts strictly after the SHA-256 selection boundary of the original
5B unique sample and excludes the sealed validation/test partition. Only the requested
chunk is materialized; the pool manifest records a much larger logical token budget so
later chunks can continue from the last emitted boundary without reusing a document.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from create_dclm_document_split import (
    KEY_WORDS,
    PARTITION_DOMAIN,
    SAMPLE_DTYPE,
    UNIQUE_DOMAIN,
    _key_prefix,
    canonical_shard_path,
    document_key,
    key_hex,
    materialize,
    prefix_selection,
    sha256_file,
    sort_order,
    threshold_for,
    write_ledger,
    write_subset_manifest,
)
from create_dclm_subset_manifest import (
    DEFAULT_SOURCE_MIX,
    inspect_source,
    parse_token_count,
    resolve_source_paths,
)

IDENTITY_DTYPE = np.dtype([("shard", "<u2"), ("doc_index", "<u8")])


def parse_key(value: str) -> tuple[int, int, int, int]:
    if len(value) != 64:
        raise ValueError(f"expected a 64-character SHA-256 key, got {value!r}")
    return tuple(int(value[index : index + 16], 16) for index in range(0, 64, 16))


def key_greater(key: tuple[int, ...], boundary: tuple[int, ...]) -> bool:
    return key > boundary


def scan_shard(args: tuple) -> dict:
    (
        source,
        partition_seed,
        unique_seed,
        test_boundary,
        start_boundary,
        threshold,
        candidate_dir_str,
    ) = args
    shard = int(source["index"])
    shard_path = canonical_shard_path(str(source["path"]))
    available_tokens = int(source["available_tokens"])
    candidate_dir = Path(candidate_dir_str)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    output_path = candidate_dir / f"extension-{shard:04d}.bin"
    partition_base = hashlib.sha256(_key_prefix(PARTITION_DOMAIN, partition_seed, shard_path))
    unique_base = hashlib.sha256(_key_prefix(UNIQUE_DOMAIN, unique_seed, shard_path))
    buffer: list[tuple] = []
    documents = 0
    tokens = 0
    previous_end = 0
    candidate_documents = 0
    candidate_tokens = 0
    with output_path.open("wb") as output, gzip.open(
        str(source["metadata_path"]), "rt", newline=""
    ) as metadata:
        for doc_index, row in enumerate(csv.reader(metadata)):
            if len(row) < 2:
                raise ValueError(f"invalid metadata row {doc_index} in {source['metadata_path']}")
            start, end = int(row[0]), int(row[1])
            if start != previous_end or end <= start or end > available_tokens:
                raise ValueError(
                    f"non-contiguous/invalid document [{start}, {end}) at "
                    f"{source['metadata_path']}:{doc_index}"
                )
            partition_key = document_key(partition_base, doc_index)
            unique_key = document_key(unique_base, doc_index)
            length = end - start
            if (
                unique_key[0] < threshold
                and key_greater(partition_key, test_boundary)
                and key_greater(unique_key, start_boundary)
            ):
                buffer.append((*partition_key, *unique_key, shard, doc_index, start, end))
                candidate_documents += 1
                candidate_tokens += length
                if len(buffer) >= 100_000:
                    np.asarray(buffer, dtype=SAMPLE_DTYPE).tofile(output)
                    buffer.clear()
            documents += 1
            tokens += length
            previous_end = end
        if buffer:
            np.asarray(buffer, dtype=SAMPLE_DTYPE).tofile(output)
    if previous_end != available_tokens or tokens != available_tokens:
        raise ValueError(
            f"metadata covers {previous_end:,}/{available_tokens:,} tokens in "
            f"{source['metadata_path']}"
        )
    return {
        "index": shard,
        "documents": documents,
        "tokens": tokens,
        "candidate_path": str(output_path),
        "candidate_documents": candidate_documents,
        "candidate_tokens": candidate_tokens,
    }


def load_candidates(paths: list[str]) -> np.ndarray:
    counts = []
    for path_str in paths:
        path = Path(path_str)
        if path.stat().st_size % SAMPLE_DTYPE.itemsize:
            raise ValueError(f"candidate file has invalid size: {path}")
        counts.append(path.stat().st_size // SAMPLE_DTYPE.itemsize)
    records = np.empty(sum(counts), dtype=SAMPLE_DTYPE)
    offset = 0
    for path_str, count in zip(paths, counts):
        records[offset : offset + count] = np.fromfile(path_str, dtype=SAMPLE_DTYPE, count=count)
        offset += count
    return records


def ledger_identities(
    ledger_path: Path, *, expected_documents: int, shard_by_path: dict[str, int]
) -> np.ndarray:
    identities = np.empty(expected_documents, dtype=IDENTITY_DTYPE)
    count = 0
    with gzip.open(ledger_path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if count >= expected_documents:
                raise RuntimeError(f"ledger contains more rows than declared: {ledger_path}")
            identities[count] = (
                shard_by_path[canonical_shard_path(row["shard_path"])],
                int(row["document_index"]),
            )
            count += 1
    if count != expected_documents:
        raise RuntimeError(
            f"ledger contains {count:,} rows, expected {expected_documents:,}: {ledger_path}"
        )
    return identities


def identity_overlap(left: np.ndarray, records: np.ndarray) -> int:
    right = np.empty(len(records), dtype=IDENTITY_DTYPE)
    right["shard"] = records["shard"]
    right["doc_index"] = records["doc_index"]
    return int(np.intersect1d(left, right, assume_unique=True).size)


def resolve_manifest_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def build(args: argparse.Namespace) -> dict:
    original = json.loads(args.original_unique_manifest.read_text())
    partition = json.loads(args.partition_manifest.read_text())
    if original.get("format") != "olmo-token-subset-v1":
        raise ValueError("unexpected original unique manifest format")
    if partition.get("format") != "dclm-document-partition-v1":
        raise ValueError("unexpected partition manifest format")
    selection = original["selection"]
    if selection.get("domain") != UNIQUE_DOMAIN:
        raise ValueError("original unique manifest uses a different selection domain")
    unique_seed = int(selection["seed"])
    original_boundary = parse_key(selection["boundary_key"])
    test_boundary = parse_key(partition["key"]["test_boundary_inclusive"])
    partition_seed = int(partition["key"]["partition_seed"])

    existing_pool = None
    if args.pool_manifest.exists():
        if not args.append:
            raise FileExistsError(
                f"pool manifest already exists; pass --append to extend it: {args.pool_manifest}"
            )
        existing_pool = json.loads(args.pool_manifest.read_text())
        if existing_pool.get("format") != "dclm-unique-extension-pool-v1":
            raise ValueError("unexpected extension pool manifest format")
        logical_tokens = int(existing_pool["logical_pool"]["requested_tokens"])
        if logical_tokens != args.logical_pool_tokens:
            raise ValueError("logical pool token budget differs from existing manifest")
        if existing_pool["selection"]["original_boundary_inclusive"] != key_hex(original_boundary):
            raise ValueError("existing pool starts from a different original boundary")
        start_boundary = parse_key(existing_pool["next_boundary_exclusive"])
        cumulative_requested = sum(
            int(chunk["requested_tokens"]) for chunk in existing_pool["chunks"]
        )
        cumulative_real = sum(
            int(chunk["real_document_tokens"]) for chunk in existing_pool["chunks"]
        )
    else:
        if args.append:
            raise FileNotFoundError(args.pool_manifest)
        start_boundary = original_boundary
        cumulative_requested = 0
        cumulative_real = 0

    if cumulative_requested + args.chunk_tokens > args.logical_pool_tokens:
        raise ValueError("chunk would exceed the logical extension-pool token budget")
    if args.chunk_manifest.exists() or args.materialized_output.exists():
        raise FileExistsError("chunk manifest or materialized output already exists")
    if args.candidate_dir.exists():
        raise FileExistsError(args.candidate_dir)

    dtype = np.dtype(args.dtype)
    relative_paths = [
        canonical_shard_path(path)
        for path in resolve_source_paths(
            args.source_mix, label=args.label, tokenizer=args.tokenizer
        )
    ]
    inspect_args = [
        (index, path, str(args.data_root), dtype.itemsize)
        for index, path in enumerate(relative_paths)
    ]
    if args.workers == 1:
        sources = list(map(inspect_source, inspect_args))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            sources = list(pool.map(inspect_source, inspect_args))
    sources.sort(key=lambda source: int(source["index"]))
    total_tokens = sum(int(source["available_tokens"]) for source in sources)
    train_tokens = int(partition["partition"]["train"]["real_tokens"])
    original_real_tokens = int(selection["selected_real_document_tokens"])
    if args.logical_pool_tokens > train_tokens - original_real_tokens:
        raise ValueError("logical pool token budget exceeds the available training partition")
    threshold = threshold_for(
        target_tokens=original_real_tokens + cumulative_real + args.chunk_tokens,
        total_tokens=total_tokens,
        multiplier=args.candidate_multiplier,
    )
    args.candidate_dir.mkdir(parents=True, exist_ok=False)
    scan_args = [
        (
            source,
            partition_seed,
            unique_seed,
            test_boundary,
            start_boundary,
            threshold,
            str(args.candidate_dir),
        )
        for source in sources
    ]
    if args.workers == 1:
        audits = list(map(scan_shard, scan_args))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            audits = list(pool.map(scan_shard, scan_args))
    audits.sort(key=lambda audit: int(audit["index"]))
    records = load_candidates([str(audit["candidate_path"]) for audit in audits])
    records = records[sort_order(records)]
    chunk = prefix_selection(records, args.chunk_tokens)
    end_boundary = tuple(int(chunk.records[name][-1]) for name in KEY_WORDS)
    if not key_greater(tuple(int(chunk.records[name][0]) for name in KEY_WORDS), start_boundary):
        raise RuntimeError("extension selection did not start after the declared boundary")

    shard_by_path = {
        canonical_shard_path(str(source["path"])): int(source["index"]) for source in sources
    }
    original_ledger_path = resolve_manifest_path(
        original["source_document_ledger"]["path"], args.manifest_base_dir
    )
    original_ids = ledger_identities(
        original_ledger_path,
        expected_documents=int(original["source_document_ledger"]["documents"]),
        shard_by_path=shard_by_path,
    )
    original_overlap = identity_overlap(original_ids, chunk.records)
    if original_overlap:
        raise RuntimeError(f"extension overlaps {original_overlap:,} original-pool documents")
    validation_test_overlap = int(
        np.count_nonzero(
            [
                not key_greater(tuple(int(record[f"p{i}"]) for i in range(4)), test_boundary)
                for record in chunk.records
            ]
        )
    )
    if validation_test_overlap:
        raise RuntimeError(
            f"extension overlaps {validation_test_overlap:,} validation/test documents"
        )

    materialized = materialize(
        chunk.records,
        sources,
        args.materialized_output,
        dtype=dtype,
        eos_token_id=args.eos_token_id,
        alignment_tokens=args.alignment_tokens,
        workers=args.workers,
    )
    ledger_path = args.materialized_output.with_name(
        args.materialized_output.stem + ".documents.csv.gz"
    )
    ledger = write_ledger(ledger_path, chunk.records, sources, include_partition_key=True)
    source_summary = dict(partition["source"])
    args.chunk_manifest.parent.mkdir(parents=True, exist_ok=True)
    chunk_manifest = write_subset_manifest(
        args.chunk_manifest,
        name=args.chunk_name,
        seed=unique_seed,
        domain=UNIQUE_DOMAIN,
        requested_tokens=args.chunk_tokens,
        records=chunk.records,
        materialized=materialized,
        ledger=ledger,
        source_summary=source_summary,
        manifest_base_dir=args.manifest_base_dir,
    )
    chunk_manifest["selection"].update(
        {
            "method": "global-sha256-document-order-range",
            "start_boundary_exclusive": key_hex(start_boundary),
            "original_5b_boundary_inclusive": key_hex(original_boundary),
            "logical_pool_requested_tokens": args.logical_pool_tokens,
            "chunk_index": 0 if existing_pool is None else len(existing_pool["chunks"]),
            "cumulative_extension_requested_tokens": cumulative_requested + args.chunk_tokens,
            "cumulative_extension_real_document_tokens": cumulative_real
            + materialized["real_document_tokens"],
        }
    )
    chunk_manifest["disjointness_audit"] = {
        "original_unique_manifest": os.path.relpath(
            args.original_unique_manifest, args.manifest_base_dir
        ),
        "original_unique_manifest_sha256": sha256_file(args.original_unique_manifest),
        "original_document_overlap": original_overlap,
        "validation_test_document_overlap": validation_test_overlap,
        "prior_chunk_document_overlap": 0,
        "prior_chunk_basis": "strictly increasing non-overlapping SHA-256 selection-key ranges",
        "passed": True,
    }
    args.chunk_manifest.write_text(json.dumps(chunk_manifest, indent=2) + "\n")

    chunk_record = {
        "index": chunk_manifest["selection"]["chunk_index"],
        "manifest": os.path.relpath(args.chunk_manifest, args.manifest_base_dir),
        "manifest_sha256": sha256_file(args.chunk_manifest),
        "requested_tokens": args.chunk_tokens,
        "real_document_tokens": materialized["real_document_tokens"],
        "materialized_tokens": materialized["materialized_tokens"],
        "documents": len(chunk.records),
        "start_boundary_exclusive": key_hex(start_boundary),
        "end_boundary_inclusive": key_hex(end_boundary),
        "token_sha256": materialized["token_sha256"],
        "ledger_sha256": ledger["sha256"],
        "disjointness_audit_passed": True,
    }
    now = datetime.now(timezone.utc).isoformat()
    if existing_pool is None:
        pool_manifest = {
            "format": "dclm-unique-extension-pool-v1",
            "created_at": now,
            "updated_at": now,
            "source": source_summary,
            "logical_pool": {
                "requested_tokens": args.logical_pool_tokens,
                "materialization": "appendable deterministic chunks",
                "materialized_tokens": materialized["materialized_tokens"],
            },
            "selection": {
                "method": "global-sha256-document-order-range",
                "identity": ["canonical_shard_path", "zero_based_metadata_row_index"],
                "domain": UNIQUE_DOMAIN,
                "seed": unique_seed,
                "original_boundary_inclusive": key_hex(original_boundary),
                "partition_test_boundary_inclusive": key_hex(test_boundary),
                "validation_and_test_excluded": True,
            },
            "chunks": [chunk_record],
            "next_boundary_exclusive": key_hex(end_boundary),
            "audit": {
                "original_document_overlap": 0,
                "validation_test_document_overlap": 0,
                "all_chunks_pairwise_disjoint": True,
                "passed": True,
            },
        }
    else:
        pool_manifest = existing_pool
        pool_manifest["updated_at"] = now
        pool_manifest["chunks"].append(chunk_record)
        pool_manifest["next_boundary_exclusive"] = key_hex(end_boundary)
        pool_manifest["logical_pool"]["materialized_tokens"] = sum(
            int(item["materialized_tokens"]) for item in pool_manifest["chunks"]
        )
    args.pool_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.pool_manifest.with_name(args.pool_manifest.name + ".tmp")
    tmp.write_text(json.dumps(pool_manifest, indent=2) + "\n")
    os.replace(tmp, args.pool_manifest)
    return pool_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-unique-manifest", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--chunk-manifest", type=Path, required=True)
    parser.add_argument("--materialized-output", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--chunk-name", required=True)
    parser.add_argument("--logical-pool-tokens", type=parse_token_count, default=1_019_000_000_000)
    parser.add_argument("--chunk-tokens", type=parse_token_count, default=11_500_781_568)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--candidate-multiplier", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--source-mix", type=Path, default=DEFAULT_SOURCE_MIX)
    parser.add_argument("--label", default="dclm")
    parser.add_argument("--tokenizer", default="allenai/dolma2-tokenizer")
    parser.add_argument("--dtype", default="uint32")
    parser.add_argument("--eos-token-id", type=int, default=100257)
    parser.add_argument("--alignment-tokens", type=parse_token_count, default=4_194_304)
    parser.add_argument("--data-root", type=Path, default=Path("/weka/oe-training-default/ai2-llm"))
    parser.add_argument(
        "--manifest-base-dir", type=Path, default=Path("/weka/oe-training-default/ai2-llm")
    )
    args = parser.parse_args()
    if args.candidate_multiplier <= 1:
        parser.error("--candidate-multiplier must exceed 1")
    return args


def main() -> None:
    args = parse_args()
    manifest = build(args)
    print(
        json.dumps(
            {
                "pool_manifest": str(args.pool_manifest),
                "logical_pool": manifest["logical_pool"],
                "latest_chunk": manifest["chunks"][-1],
                "audit": manifest["audit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
