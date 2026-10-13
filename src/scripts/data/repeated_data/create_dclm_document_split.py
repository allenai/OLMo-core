#!/usr/bin/env python3
"""Create an exhaustive document-random DCLM split and Step-2 training samples.

Every source document is identified by its canonical shard path and zero-based
row index in that shard's ``.csv.gz`` metadata.  SHA-256 keys derived from
``(domain, seed, shard path, document index)`` define independent global random
orders for:

* the exhaustive train / validation / test partition;
* the repeated-data 1B sample from the training partition; and
* the unique-data 5B sample from the training partition.

Validation and test are consecutive whole-document prefixes of the partition
order.  All other source documents are training documents, so the three sets
are mutually exclusive and exhaustive by construction.  The two training
samples are independent whole-document prefixes after filtering out validation
and test documents.

Only selected documents are materialized.  The full training partition is
represented exactly by the two partition cut keys and the audited source
document universe, avoiding a multi-terabyte copy of the remaining DCLM pool.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, NamedTuple

import numpy as np
from create_dclm_subset_manifest import (
    DEFAULT_SOURCE_MIX,
    deterministic_gzip_text_writer,
    inspect_source,
    manifest_digest,
    parse_token_count,
    resolve_source_paths,
)

KEY_WORDS = ("k0", "k1", "k2", "k3")
PARTITION_DOMAIN = "dclm-document-partition-v1"
REPEATED_DOMAIN = "dclm-train-repeated-sample-v1"
UNIQUE_DOMAIN = "dclm-train-unique-sample-v1"
UINT64_SPACE = 1 << 64

PARTITION_DTYPE = np.dtype(
    [(name, ">u8") for name in KEY_WORDS]
    + [("shard", "<u2"), ("doc_index", "<u8"), ("start", "<u8"), ("end", "<u8")]
)
SAMPLE_DTYPE = np.dtype(
    [(f"p{i}", ">u8") for i in range(4)]
    + [(name, ">u8") for name in KEY_WORDS]
    + [("shard", "<u2"), ("doc_index", "<u8"), ("start", "<u8"), ("end", "<u8")]
)


class Selection(NamedTuple):
    records: np.ndarray
    real_tokens: int
    boundary: tuple[int, int, int, int]


def canonical_shard_path(path: str) -> str:
    normalized = str(PurePosixPath("/" + path.lstrip("/")))
    if "/../" in normalized or normalized.endswith("/.."):
        raise ValueError(f"unsafe shard path: {path!r}")
    return normalized.lstrip("/")


def _key_prefix(domain: str, seed: int, shard_path: str) -> bytes:
    domain_bytes = domain.encode("utf-8")
    path_bytes = canonical_shard_path(shard_path).encode("utf-8")
    return b"".join(
        (
            struct.pack(">H", len(domain_bytes)),
            domain_bytes,
            struct.pack(">q", seed),
            struct.pack(">I", len(path_bytes)),
            path_bytes,
        )
    )


def document_key(base: "hashlib._Hash", doc_index: int) -> tuple[int, int, int, int]:
    digest = base.copy()
    digest.update(struct.pack(">Q", doc_index))
    return struct.unpack(">QQQQ", digest.digest())


def key_hex(words: Iterable[int]) -> str:
    return "".join(f"{int(word):016x}" for word in words)


def threshold_for(*, target_tokens: int, total_tokens: int, multiplier: float) -> int:
    probability = min(1.0, multiplier * target_tokens / total_tokens)
    return min(UINT64_SPACE, math.ceil(probability * UINT64_SPACE))


def _flush_records(handle, records: list[tuple], dtype: np.dtype) -> None:
    if records:
        np.asarray(records, dtype=dtype).tofile(handle)
        records.clear()


def scan_source(args: tuple[dict, dict, str, int]) -> dict:
    source, config, candidate_dir_str, bucket_bits = args
    shard = int(source["index"])
    shard_path = canonical_shard_path(str(source["path"]))
    available_tokens = int(source["available_tokens"])
    metadata_path = str(source["metadata_path"])
    candidate_dir = Path(candidate_dir_str)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    partition_base = hashlib.sha256(
        _key_prefix(PARTITION_DOMAIN, int(config["partition_seed"]), shard_path)
    )
    repeated_base = hashlib.sha256(
        _key_prefix(REPEATED_DOMAIN, int(config["repeated_seed"]), shard_path)
    )
    unique_base = hashlib.sha256(_key_prefix(UNIQUE_DOMAIN, int(config["unique_seed"]), shard_path))

    partition_path = candidate_dir / f"partition-{shard:04d}.bin"
    repeated_path = candidate_dir / f"repeated-{shard:04d}.bin"
    unique_path = candidate_dir / f"unique-{shard:04d}.bin"
    partition_records: list[tuple] = []
    repeated_records: list[tuple] = []
    unique_records: list[tuple] = []
    bucket_counts = np.zeros(1 << bucket_bits, dtype=np.uint64)
    doc_count = 0
    token_count = 0
    previous_end = 0
    max_doc_tokens = 0
    sum_squared_doc_tokens = 0
    candidate_counts = {"partition": 0, "repeated": 0, "unique": 0}
    candidate_tokens = {"partition": 0, "repeated": 0, "unique": 0}

    with (
        partition_path.open("wb") as partition_out,
        repeated_path.open("wb") as repeated_out,
        unique_path.open("wb") as unique_out,
        gzip.open(metadata_path, "rt", newline="") as metadata,
    ):
        for doc_index, row in enumerate(csv.reader(metadata)):
            if len(row) < 2:
                raise ValueError(f"invalid metadata row {doc_index} in {metadata_path}")
            start, end = int(row[0]), int(row[1])
            if start != previous_end:
                raise ValueError(
                    f"metadata gap/overlap in {metadata_path} at document {doc_index}: "
                    f"expected start {previous_end}, got {start}"
                )
            if end <= start or end > available_tokens:
                raise ValueError(
                    f"invalid document [{start}, {end}) in {metadata_path} at row {doc_index}"
                )
            length = end - start
            partition_key = document_key(partition_base, doc_index)
            bucket_counts[partition_key[0] >> (64 - bucket_bits)] += 1
            repeated_key = document_key(repeated_base, doc_index)
            unique_key = document_key(unique_base, doc_index)

            if partition_key[0] < int(config["partition_threshold"]):
                partition_records.append((*partition_key, shard, doc_index, start, end))
                candidate_counts["partition"] += 1
                candidate_tokens["partition"] += length
            if repeated_key[0] < int(config["repeated_threshold"]):
                repeated_records.append(
                    (*partition_key, *repeated_key, shard, doc_index, start, end)
                )
                candidate_counts["repeated"] += 1
                candidate_tokens["repeated"] += length
            if unique_key[0] < int(config["unique_threshold"]):
                unique_records.append((*partition_key, *unique_key, shard, doc_index, start, end))
                candidate_counts["unique"] += 1
                candidate_tokens["unique"] += length

            if len(partition_records) >= 100_000:
                _flush_records(partition_out, partition_records, PARTITION_DTYPE)
            if len(repeated_records) >= 100_000:
                _flush_records(repeated_out, repeated_records, SAMPLE_DTYPE)
            if len(unique_records) >= 100_000:
                _flush_records(unique_out, unique_records, SAMPLE_DTYPE)

            doc_count += 1
            token_count += length
            max_doc_tokens = max(max_doc_tokens, length)
            sum_squared_doc_tokens += length * length
            previous_end = end

        _flush_records(partition_out, partition_records, PARTITION_DTYPE)
        _flush_records(repeated_out, repeated_records, SAMPLE_DTYPE)
        _flush_records(unique_out, unique_records, SAMPLE_DTYPE)

    if previous_end != available_tokens or token_count != available_tokens:
        raise ValueError(
            f"metadata for {metadata_path} covers {previous_end:,}/{available_tokens:,} tokens"
        )
    return {
        "index": shard,
        "path": shard_path,
        "available_tokens": available_tokens,
        "documents": doc_count,
        "max_document_tokens": max_doc_tokens,
        "sum_squared_document_tokens": sum_squared_doc_tokens,
        "partition_candidates": candidate_counts["partition"],
        "partition_candidate_tokens": candidate_tokens["partition"],
        "repeated_candidates": candidate_counts["repeated"],
        "repeated_candidate_tokens": candidate_tokens["repeated"],
        "unique_candidates": candidate_counts["unique"],
        "unique_candidate_tokens": candidate_tokens["unique"],
        "partition_buckets": bucket_counts.tolist(),
        "partition_candidate_path": str(partition_path),
        "repeated_candidate_path": str(repeated_path),
        "unique_candidate_path": str(unique_path),
    }


def load_candidates(paths: list[str], dtype: np.dtype) -> np.ndarray:
    counts = []
    for path_str in paths:
        size = Path(path_str).stat().st_size
        if size % dtype.itemsize:
            raise ValueError(f"candidate file has invalid size: {path_str}")
        counts.append(size // dtype.itemsize)
    output = np.empty(sum(counts), dtype=dtype)
    offset = 0
    for path_str, count in zip(paths, counts):
        output[offset : offset + count] = np.fromfile(path_str, dtype=dtype, count=count)
        offset += count
    return output


def sort_order(records: np.ndarray) -> np.ndarray:
    order = np.lexsort(tuple(records[name] for name in reversed(KEY_WORDS)))
    if len(order) > 1:
        ordered = records[order]
        duplicate = np.ones(len(ordered) - 1, dtype=bool)
        for name in KEY_WORDS:
            duplicate &= ordered[name][:-1] == ordered[name][1:]
        if duplicate.any():
            raise RuntimeError("cryptographic key collision detected")
    return order


def prefix_selection(records: np.ndarray, target_tokens: int, *, start: int = 0) -> Selection:
    if start >= len(records):
        raise RuntimeError("candidate pool exhausted before selection")
    lengths = records["end"].astype(np.uint64) - records["start"].astype(np.uint64)
    cumulative = np.cumsum(lengths[start:], dtype=np.uint64)
    crossing = int(np.searchsorted(cumulative, target_tokens, side="left"))
    end = start + crossing + 1
    if end > len(records):
        raise RuntimeError("not enough candidate document tokens")
    selected = records[start:end].copy()
    boundary = tuple(int(selected[name][-1]) for name in KEY_WORDS)
    return Selection(selected, int(cumulative[crossing]), boundary)


def partition_key_is_train(records: np.ndarray, test_boundary: tuple[int, ...]) -> np.ndarray:
    greater = np.zeros(len(records), dtype=bool)
    equal = np.ones(len(records), dtype=bool)
    for index, boundary_word in enumerate(test_boundary):
        values = records[f"p{index}"]
        greater |= equal & (values > boundary_word)
        equal &= values == boundary_word
    return greater


def selected_shard_stats(records: np.ndarray, num_shards: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(records["shard"], minlength=num_shards).astype(np.uint64)
    lengths = records["end"].astype(np.uint64) - records["start"].astype(np.uint64)
    tokens = np.bincount(records["shard"], weights=lengths, minlength=num_shards).astype(np.uint64)
    return counts, tokens


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_ledger(
    path: Path,
    records: np.ndarray,
    sources: list[dict],
    *,
    include_partition_key: bool,
) -> dict:
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + ".tmp")
    with deterministic_gzip_text_writer(tmp) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        columns = ["shard_path", "document_index", "start", "end", "selection_key"]
        if include_partition_key:
            columns.append("partition_key")
        writer.writerow(columns)
        for record in records:
            row = [
                canonical_shard_path(str(sources[int(record["shard"])]["path"])),
                int(record["doc_index"]),
                int(record["start"]),
                int(record["end"]),
                key_hex(int(record[name]) for name in KEY_WORDS),
            ]
            if include_partition_key:
                row.append(key_hex(int(record[f"p{i}"]) for i in range(4)))
            writer.writerow(row)
    os.replace(tmp, path)
    return {"path": str(path), "sha256": sha256_file(path), "documents": len(records)}


def materialize_shard(args: tuple) -> dict:
    (
        shard,
        records,
        source,
        fragment_dir_str,
        dtype_str,
        eos_token_id,
    ) = args
    dtype = np.dtype(dtype_str)
    fragment_dir = Path(fragment_dir_str)
    token_path = fragment_dir / f"shard-{shard:04d}.tokens.bin"
    lengths_path = fragment_dir / f"shard-{shard:04d}.lengths.bin"
    marker_path = fragment_dir / f"shard-{shard:04d}.complete.json"
    expected_lengths = records["end"].astype(np.uint64) - records["start"].astype(np.uint64)
    expected_tokens = int(expected_lengths.sum(dtype=np.uint64))
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        if (
            int(marker["documents"]) != len(records)
            or int(marker["tokens"]) != expected_tokens
            or token_path.stat().st_size != expected_tokens * dtype.itemsize
            or lengths_path.stat().st_size != len(records) * np.dtype("<u8").itemsize
        ):
            raise RuntimeError(f"invalid resumable shard fragment: {marker_path}")
        return {
            "shard": shard,
            "token_path": str(token_path),
            "lengths_path": str(lengths_path),
            "documents": len(records),
            "tokens": expected_tokens,
            "resumed": True,
        }

    token_tmp = token_path.with_name(token_path.name + ".tmp")
    lengths_tmp = lengths_path.with_name(lengths_path.name + ".tmp")
    array = np.memmap(str(source["absolute_path"]), mode="r", dtype=dtype)
    try:
        with token_tmp.open("wb") as token_out:
            for record in records:
                start, end = int(record["start"]), int(record["end"])
                values = np.asarray(array[start:end])
                if int(values[-1]) != eos_token_id:
                    raise ValueError(
                        f"document {int(record['doc_index'])} in {source['path']} does not end in EOS"
                    )
                token_out.write(values.tobytes(order="C"))
        expected_lengths.astype("<u8", copy=False).tofile(lengths_tmp)
        if token_tmp.stat().st_size != expected_tokens * dtype.itemsize:
            raise RuntimeError(f"short shard token fragment: {token_tmp}")
        os.replace(token_tmp, token_path)
        os.replace(lengths_tmp, lengths_path)
        marker_path.write_text(
            json.dumps(
                {"shard": shard, "documents": len(records), "tokens": expected_tokens},
                sort_keys=True,
            )
            + "\n"
        )
    except BaseException:
        token_tmp.unlink(missing_ok=True)
        lengths_tmp.unlink(missing_ok=True)
        raise
    finally:
        del array
    return {
        "shard": shard,
        "token_path": str(token_path),
        "lengths_path": str(lengths_path),
        "documents": len(records),
        "tokens": expected_tokens,
        "resumed": False,
    }


def materialize(
    records: np.ndarray,
    sources: list[dict],
    output_path: Path,
    *,
    dtype: np.dtype,
    eos_token_id: int,
    alignment_tokens: int,
    workers: int,
) -> dict:
    metadata_path = output_path.with_suffix(".csv.gz")
    for path in (output_path, metadata_path):
        if path.exists():
            raise FileExistsError(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_tmp = output_path.with_name(output_path.name + ".tmp")
    metadata_tmp = metadata_path.with_name(metadata_path.name + ".tmp")
    source_order = np.lexsort((records["start"], records["shard"]))
    ordered = records[source_order]
    real_tokens = int(np.sum(ordered["end"].astype(np.uint64) - ordered["start"].astype(np.uint64)))
    materialized_tokens = math.ceil(real_tokens / alignment_tokens) * alignment_tokens
    padding_tokens = materialized_tokens - real_tokens
    digest = hashlib.sha256()
    offset = 0
    fragment_dir = output_path.with_name(output_path.name + ".fragments")
    fragment_dir.mkdir(parents=True, exist_ok=True)
    shard_args = []
    for shard_value in np.unique(ordered["shard"]):
        shard = int(shard_value)
        shard_args.append(
            (
                shard,
                ordered[ordered["shard"] == shard].copy(),
                sources[shard],
                str(fragment_dir),
                dtype.str,
                eos_token_id,
            )
        )
    fragments = []
    if workers == 1:
        for index, shard_arg in enumerate(shard_args, start=1):
            fragments.append(materialize_shard(shard_arg))
            print(
                f"materialized {output_path.name}: {index}/{len(shard_args)} shards",
                flush=True,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(materialize_shard, shard_arg) for shard_arg in shard_args]
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                fragments.append(future.result())
                if index == 1 or index % 25 == 0 or index == len(futures):
                    print(
                        f"materialized {output_path.name}: {index}/{len(futures)} shards",
                        flush=True,
                    )
    fragments.sort(key=lambda fragment: int(fragment["shard"]))
    try:
        with token_tmp.open("wb") as token_out, deterministic_gzip_text_writer(
            metadata_tmp
        ) as metadata_out:
            for fragment in fragments:
                with Path(fragment["token_path"]).open("rb") as token_in:
                    while payload := token_in.read(16 * 1024 * 1024):
                        token_out.write(payload)
                        digest.update(payload)
                lengths = np.fromfile(fragment["lengths_path"], dtype="<u8")
                for length_value in lengths:
                    length = int(length_value)
                    metadata_out.write(f"{offset},{offset + length}\n")
                    offset += length
            if padding_tokens:
                padding = np.full(padding_tokens, eos_token_id, dtype=dtype)
                payload = padding.tobytes(order="C")
                token_out.write(payload)
                digest.update(payload)
                metadata_out.write(f"{offset},{offset + padding_tokens}\n")
                offset += padding_tokens
        if offset != materialized_tokens:
            raise AssertionError((offset, materialized_tokens))
        os.replace(token_tmp, output_path)
        os.replace(metadata_tmp, metadata_path)
        shutil.rmtree(fragment_dir)
    except BaseException:
        token_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
        raise
    return {
        "path": str(output_path),
        "token_sha256": digest.hexdigest(),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "real_document_tokens": real_tokens,
        "padding_eos_tokens": padding_tokens,
        "materialized_tokens": materialized_tokens,
        "instances": materialized_tokens // 4096,
    }


def write_subset_manifest(
    path: Path,
    *,
    name: str,
    seed: int,
    domain: str,
    requested_tokens: int,
    records: np.ndarray,
    materialized: dict,
    ledger: dict,
    source_summary: dict,
    manifest_base_dir: Path,
) -> dict:
    relative_data_path = os.path.relpath(materialized["path"], manifest_base_dir)
    entry = {
        "path": relative_data_path,
        "start_instance": 0,
        "num_instances": materialized["instances"],
        "available_instances": materialized["instances"],
        "available_tokens": materialized["materialized_tokens"],
    }
    manifest = {
        "format": "olmo-token-subset-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "source": source_summary,
        "selection": {
            "method": "global-sha256-document-order-prefix",
            "identity": ["canonical_shard_path", "zero_based_metadata_row_index"],
            "domain": domain,
            "seed": seed,
            "requested_tokens": requested_tokens,
            "sequence_length": 4096,
            "selected_documents": len(records),
            "selected_real_document_tokens": materialized["real_document_tokens"],
            "padding_eos_tokens": materialized["padding_eos_tokens"],
            "selected_tokens": materialized["materialized_tokens"],
            "selected_instances": materialized["instances"],
            "boundary_key": key_hex(int(records[-1][word]) for word in KEY_WORDS),
        },
        "materialized": {
            "path": relative_data_path,
            "token_sha256": materialized["token_sha256"],
            "document_metadata_path": os.path.relpath(
                materialized["metadata_path"], manifest_base_dir
            ),
            "document_metadata_sha256": materialized["metadata_sha256"],
        },
        "source_document_ledger": {
            **ledger,
            "path": os.path.relpath(ledger["path"], manifest_base_dir),
        },
        "entries_sha256": manifest_digest([entry]),
        "entries": [entry],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def write_json_gzip(path: Path, value: object) -> dict:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(payload)
    os.replace(tmp, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "contents_sha256": hashlib.sha256(payload).hexdigest(),
    }


def uniformity_summary(bucket_counts: np.ndarray) -> dict:
    total = int(bucket_counts.sum())
    expected = total / len(bucket_counts)
    chi_square = float(np.sum((bucket_counts - expected) ** 2 / expected))
    degrees = len(bucket_counts) - 1
    normal_z = (chi_square - degrees) / math.sqrt(2 * degrees)
    maximum_z = float(np.max(np.abs(bucket_counts - expected) / math.sqrt(expected)))
    return {
        "bucket_bits": int(round(math.log2(len(bucket_counts)))),
        "buckets": len(bucket_counts),
        "documents": total,
        "expected_documents_per_bucket": expected,
        "chi_square": chi_square,
        "degrees_of_freedom": degrees,
        "chi_square_normal_z": normal_z,
        "maximum_bucket_standardized_deviation": maximum_z,
        "sanity_check_abs_chi_square_z_below_5": abs(normal_z) < 5,
    }


def build(args: argparse.Namespace) -> dict:
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
    partition_target = args.validation_tokens + args.test_tokens
    scan_config = {
        "partition_seed": args.partition_seed,
        "repeated_seed": args.repeated_seed,
        "unique_seed": args.unique_seed,
        "partition_threshold": threshold_for(
            target_tokens=partition_target,
            total_tokens=total_tokens,
            multiplier=args.candidate_multiplier,
        ),
        "repeated_threshold": threshold_for(
            target_tokens=args.repeated_tokens,
            total_tokens=total_tokens,
            multiplier=args.candidate_multiplier,
        ),
        "unique_threshold": threshold_for(
            target_tokens=args.unique_tokens,
            total_tokens=total_tokens,
            multiplier=args.candidate_multiplier,
        ),
    }
    args.candidate_dir.mkdir(parents=True, exist_ok=False)
    scan_args = [
        (source, scan_config, str(args.candidate_dir), args.uniformity_bucket_bits)
        for source in sources
    ]
    if args.workers == 1:
        audits = list(map(scan_source, scan_args))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            audits = list(pool.map(scan_source, scan_args))
    audits.sort(key=lambda audit: int(audit["index"]))

    partition_candidates = load_candidates(
        [str(audit["partition_candidate_path"]) for audit in audits], PARTITION_DTYPE
    )
    partition_order = sort_order(partition_candidates)
    partition_candidates = partition_candidates[partition_order]
    validation = prefix_selection(partition_candidates, args.validation_tokens)
    test = prefix_selection(
        partition_candidates,
        args.test_tokens,
        start=len(validation.records),
    )

    def select_training_sample(kind: str, target: int) -> Selection:
        records = load_candidates(
            [str(audit[f"{kind}_candidate_path"]) for audit in audits], SAMPLE_DTYPE
        )
        records = records[partition_key_is_train(records, test.boundary)]
        records = records[sort_order(records)]
        return prefix_selection(records, target)

    repeated = select_training_sample("repeated", args.repeated_tokens)
    unique = select_training_sample("unique", args.unique_tokens)

    num_shards = len(sources)
    validation_counts, validation_by_shard = selected_shard_stats(validation.records, num_shards)
    test_counts, test_by_shard = selected_shard_stats(test.records, num_shards)
    repeated_counts, repeated_by_shard = selected_shard_stats(repeated.records, num_shards)
    unique_counts, unique_by_shard = selected_shard_stats(unique.records, num_shards)
    shard_audit_rows = []
    for source, audit in zip(sources, audits):
        shard = int(source["index"])
        shard_audit_rows.append(
            {
                "path": canonical_shard_path(str(source["path"])),
                "tokens": int(audit["available_tokens"]),
                "documents": int(audit["documents"]),
                "validation_tokens": int(validation_by_shard[shard]),
                "validation_documents": int(validation_counts[shard]),
                "test_tokens": int(test_by_shard[shard]),
                "test_documents": int(test_counts[shard]),
                "train_tokens": int(audit["available_tokens"])
                - int(validation_by_shard[shard])
                - int(test_by_shard[shard]),
                "train_documents": int(audit["documents"])
                - int(validation_counts[shard])
                - int(test_counts[shard]),
                "repeated_sample_tokens": int(repeated_by_shard[shard]),
                "repeated_sample_documents": int(repeated_counts[shard]),
                "unique_sample_tokens": int(unique_by_shard[shard]),
                "unique_sample_documents": int(unique_counts[shard]),
            }
        )

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    selection_specs = {
        "validation": (
            validation,
            args.validation_tokens,
            args.partition_seed,
            PARTITION_DOMAIN,
            4096,
        ),
        "test": (test, args.test_tokens, args.partition_seed, PARTITION_DOMAIN, 4096),
        "repeated_train_1b": (
            repeated,
            args.repeated_tokens,
            args.repeated_seed,
            REPEATED_DOMAIN,
            args.training_alignment_tokens,
        ),
        "unique_train_5b": (
            unique,
            args.unique_tokens,
            args.unique_seed,
            UNIQUE_DOMAIN,
            args.training_alignment_tokens,
        ),
    }
    source_summary = {
        "mix": "dclm-full",
        "source_mix_file": args.source_mix.name,
        "source_mix_sha256": hashlib.sha256(args.source_mix.read_bytes()).hexdigest(),
        "label": args.label,
        "tokenizer": args.tokenizer,
        "dtype": dtype.name,
        "original_num_source_paths": len(sources),
        "original_available_tokens": total_tokens,
        "original_documents": sum(int(audit["documents"]) for audit in audits),
    }
    manifest_dir = args.manifest_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, (selection, requested, seed, domain, alignment) in selection_specs.items():
        data_path = output_root / f"dclm_0802_{name}_uint32.npy"
        ledger_path = output_root / f"dclm_0802_{name}.documents.csv.gz"
        materialized = materialize(
            selection.records,
            sources,
            data_path,
            dtype=dtype,
            eos_token_id=args.eos_token_id,
            alignment_tokens=alignment,
            workers=args.workers,
        )
        ledger = write_ledger(
            ledger_path,
            selection.records,
            sources,
            include_partition_key=name.startswith("repeated") or name.startswith("unique"),
        )
        manifest_path = manifest_dir / f"dclm_0802_{name}.json"
        manifest = write_subset_manifest(
            manifest_path,
            name=name,
            seed=seed,
            domain=domain,
            requested_tokens=requested,
            records=selection.records,
            materialized=materialized,
            ledger=ledger,
            source_summary=source_summary,
            manifest_base_dir=args.manifest_base_dir,
        )
        artifacts[name] = {
            "manifest": os.path.relpath(manifest_path, args.manifest_base_dir),
            "manifest_sha256": sha256_file(manifest_path),
            "selection": manifest["selection"],
            "materialized": manifest["materialized"],
            "ledger": manifest["source_document_ledger"],
        }

    repeated_ids = {(int(record["shard"]), int(record["doc_index"])) for record in repeated.records}
    overlap_documents = sum(
        (int(record["shard"]), int(record["doc_index"])) in repeated_ids
        for record in unique.records
    )
    bucket_counts = np.sum(
        np.asarray([audit["partition_buckets"] for audit in audits], dtype=np.uint64), axis=0
    )
    shard_audit_path = output_root / "dclm_0802_partition.shards.json.gz"
    shard_audit = write_json_gzip(shard_audit_path, shard_audit_rows)
    partition_manifest = {
        "format": "dclm-document-partition-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": ["canonical_shard_path", "zero_based_metadata_row_index"],
        "key": {
            "algorithm": "SHA-256",
            "encoding": "length-prefixed domain and path; signed big-endian seed; unsigned big-endian document index",
            "partition_domain": PARTITION_DOMAIN,
            "partition_seed": args.partition_seed,
            "validation_boundary_inclusive": key_hex(validation.boundary),
            "test_boundary_inclusive": key_hex(test.boundary),
            "assignment": "key <= validation boundary: validation; key <= test boundary: test; otherwise train",
        },
        "source": source_summary,
        "partition": {
            "validation": {
                "documents": len(validation.records),
                "real_tokens": validation.real_tokens,
            },
            "test": {"documents": len(test.records), "real_tokens": test.real_tokens},
            "train": {
                "documents": source_summary["original_documents"]
                - len(validation.records)
                - len(test.records),
                "real_tokens": total_tokens - validation.real_tokens - test.real_tokens,
            },
            "document_count_sum": source_summary["original_documents"],
            "token_count_sum": total_tokens,
            "pairwise_document_intersections": {
                "train_validation": 0,
                "train_test": 0,
                "validation_test": 0,
            },
            "exhaustive_by_boundary_rule": True,
        },
        "training_samples": {
            "repeated": artifacts["repeated_train_1b"],
            "unique": artifacts["unique_train_5b"],
            "repeated_unique_document_overlap": overlap_documents,
            "both_exclude_validation_and_test": True,
        },
        "uniformity": uniformity_summary(bucket_counts),
        "coverage": {
            "every_shard_starts_at_token_zero": True,
            "every_shard_metadata_is_contiguous": True,
            "every_shard_ends_at_available_tokens": True,
            "all_selected_documents_end_in_eos": True,
        },
        "shard_audit": {
            **shard_audit,
            "path": os.path.relpath(shard_audit["path"], args.manifest_base_dir),
        },
        "artifacts": artifacts,
    }
    args.partition_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.partition_manifest.write_text(json.dumps(partition_manifest, indent=2) + "\n")
    return partition_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--validation-tokens", type=parse_token_count, default=100_000_000)
    parser.add_argument("--test-tokens", type=parse_token_count, default=100_000_000)
    parser.add_argument("--repeated-tokens", type=parse_token_count, default=1_000_000_000)
    parser.add_argument("--unique-tokens", type=parse_token_count, default=5_000_000_000)
    parser.add_argument("--partition-seed", type=int, default=0)
    parser.add_argument("--repeated-seed", type=int, default=1)
    parser.add_argument("--unique-seed", type=int, default=2)
    parser.add_argument("--candidate-multiplier", type=float, default=1.5)
    parser.add_argument("--uniformity-bucket-bits", type=int, default=12)
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--source-mix", type=Path, default=DEFAULT_SOURCE_MIX)
    parser.add_argument("--label", default="dclm")
    parser.add_argument("--tokenizer", default="allenai/dolma2-tokenizer")
    parser.add_argument("--dtype", default="uint32")
    parser.add_argument("--eos-token-id", type=int, default=100257)
    parser.add_argument("--training-alignment-tokens", type=parse_token_count, default=4_194_304)
    parser.add_argument("--data-root", type=Path, default=Path("/weka/oe-training-default/ai2-llm"))
    parser.add_argument(
        "--manifest-base-dir", type=Path, default=Path("/weka/oe-training-default/ai2-llm")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.partition_manifest.exists():
        raise FileExistsError(args.partition_manifest)
    if args.candidate_multiplier <= 1:
        raise ValueError("candidate multiplier must exceed 1")
    if not 8 <= args.uniformity_bucket_bits <= 16:
        raise ValueError("uniformity bucket bits must be between 8 and 16")
    manifest = build(args)
    print(
        json.dumps(
            {
                "partition_manifest": str(args.partition_manifest),
                "partition": manifest["partition"],
                "uniformity": manifest["uniformity"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
