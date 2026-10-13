#!/usr/bin/env python3
"""Materialize nested 3B and 9B DCLM pools extending the sealed 0802 1B pool.

The existing 1B pool is preserved byte-for-byte. New documents are selected from
the immediately following range of the same global SHA-256 document ordering:

    3B = existing 1B + new 2B
    9B = existing 1B + new 2B + new 6B

All selections exclude the sealed validation/test partition. The two extension
chunks are materialized independently so an E1 checkpoint on the 1B pool can be
continued on the audited, disjoint 2B chunk without replaying old documents.
Composite 3B/9B manifests are also emitted for later full-pool epochs.
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
from create_dclm_unique_extension_pool import (
    IDENTITY_DTYPE,
    identity_overlap,
    key_greater,
    ledger_identities,
    parse_key,
    resolve_manifest_path,
)

POOL_FORMAT = "dclm-nested-pool-v1"


def scan_shard(args: tuple) -> dict:
    (
        source,
        partition_seed,
        selection_domain,
        selection_seed,
        test_boundary,
        base_boundary,
        threshold,
        candidate_dir_str,
    ) = args
    shard = int(source["index"])
    shard_path = canonical_shard_path(str(source["path"]))
    available_tokens = int(source["available_tokens"])
    candidate_dir = Path(candidate_dir_str)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    output_path = candidate_dir / f"nested-{shard:04d}.bin"
    partition_base = hashlib.sha256(_key_prefix(PARTITION_DOMAIN, partition_seed, shard_path))
    selection_base = hashlib.sha256(_key_prefix(selection_domain, selection_seed, shard_path))
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
            selection_key = document_key(selection_base, doc_index)
            length = end - start
            if (
                selection_key[0] < threshold
                and key_greater(partition_key, test_boundary)
                and key_greater(selection_key, base_boundary)
            ):
                buffer.append((*partition_key, *selection_key, shard, doc_index, start, end))
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
    counts: list[int] = []
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


def entries_sha256(entries: list[dict]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def relative(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path, base_dir)


def composite_manifest(
    *,
    output_path: Path,
    name: str,
    base_manifest_path: Path,
    component_paths: list[Path],
    source_summary: dict,
    selection_domain: str,
    selection_seed: int,
    manifest_base_dir: Path,
) -> dict:
    paths = [base_manifest_path, *component_paths]
    components = [json.loads(path.read_text()) for path in paths]
    for component in components:
        if component.get("format") != "olmo-token-subset-v1":
            raise ValueError("nested-pool component has an unexpected manifest format")
        selection = component["selection"]
        if (
            selection.get("domain") != selection_domain
            or int(selection.get("seed")) != selection_seed
        ):
            raise ValueError("nested-pool components do not share one selection ordering")
    entries = [entry for component in components for entry in component["entries"]]
    selections = [component["selection"] for component in components]
    requested_tokens = sum(int(selection["requested_tokens"]) for selection in selections)
    selected_real_tokens = sum(
        int(selection["selected_real_document_tokens"]) for selection in selections
    )
    padding_tokens = sum(int(selection["padding_eos_tokens"]) for selection in selections)
    selected_tokens = sum(int(selection["selected_tokens"]) for selection in selections)
    selected_instances = sum(int(selection["selected_instances"]) for selection in selections)
    selected_documents = sum(int(selection["selected_documents"]) for selection in selections)
    manifest = {
        "format": "olmo-token-subset-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "source": source_summary,
        "selection": {
            "method": "nested-global-sha256-document-order-prefix",
            "identity": ["canonical_shard_path", "zero_based_metadata_row_index"],
            "domain": selection_domain,
            "seed": selection_seed,
            "requested_tokens": requested_tokens,
            "sequence_length": int(selections[0]["sequence_length"]),
            "selected_documents": selected_documents,
            "selected_real_document_tokens": selected_real_tokens,
            "padding_eos_tokens": padding_tokens,
            "selected_tokens": selected_tokens,
            "selected_instances": selected_instances,
            "boundary_key": selections[-1]["boundary_key"],
            "component_requested_tokens": [int(item["requested_tokens"]) for item in selections],
        },
        "components": [
            {
                "manifest": relative(path, manifest_base_dir),
                "manifest_sha256": sha256_file(path),
                "requested_tokens": int(component["selection"]["requested_tokens"]),
                "selected_real_document_tokens": int(
                    component["selection"]["selected_real_document_tokens"]
                ),
                "selected_instances": int(component["selection"]["selected_instances"]),
            }
            for path, component in zip(paths, components)
        ],
        "entries_sha256": entries_sha256(entries),
        "entries": entries,
        "nestedness_audit": {
            "base_preserved_byte_for_byte": True,
            "component_key_ranges_strictly_increasing": True,
            "validation_and_test_excluded": True,
            "pairwise_document_overlap": 0,
            "passed": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build(args: argparse.Namespace) -> dict:
    base = json.loads(args.base_manifest.read_text())
    partition = json.loads(args.partition_manifest.read_text())
    if base.get("format") != "olmo-token-subset-v1":
        raise ValueError("unexpected base manifest format")
    if partition.get("format") != "dclm-document-partition-v1":
        raise ValueError("unexpected partition manifest format")
    selection = base["selection"]
    selection_domain = str(selection["domain"])
    selection_seed = int(selection["seed"])
    base_boundary = parse_key(selection["boundary_key"])
    test_boundary = parse_key(partition["key"]["test_boundary_inclusive"])
    partition_seed = int(partition["key"]["partition_seed"])
    if int(selection["requested_tokens"]) != args.base_tokens:
        raise ValueError("base manifest requested-token count does not match --base-tokens")
    if args.first_extension_tokens + args.second_extension_tokens <= 0:
        raise ValueError("extension token counts must be positive")
    outputs = (
        args.pool_manifest,
        args.first_chunk_manifest,
        args.second_chunk_manifest,
        args.pool_3b_manifest,
        args.pool_9b_manifest,
        args.first_chunk_output,
        args.second_chunk_output,
    )
    if any(path.exists() for path in outputs) or args.candidate_dir.exists():
        raise FileExistsError("one or more nested-pool outputs already exist")

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
    total_source_tokens = sum(int(source["available_tokens"]) for source in sources)
    extension_tokens = args.first_extension_tokens + args.second_extension_tokens
    threshold = threshold_for(
        target_tokens=args.base_tokens + extension_tokens,
        total_tokens=total_source_tokens,
        multiplier=args.candidate_multiplier,
    )
    args.candidate_dir.mkdir(parents=True, exist_ok=False)
    scan_args = [
        (
            source,
            partition_seed,
            selection_domain,
            selection_seed,
            test_boundary,
            base_boundary,
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
    first = prefix_selection(records, args.first_extension_tokens)
    second = prefix_selection(records[len(first.records) :], args.second_extension_tokens)
    first_start = tuple(int(first.records[name][0]) for name in KEY_WORDS)
    first_end = tuple(int(first.records[name][-1]) for name in KEY_WORDS)
    second_start = tuple(int(second.records[name][0]) for name in KEY_WORDS)
    second_end = tuple(int(second.records[name][-1]) for name in KEY_WORDS)
    if not (base_boundary < first_start <= first_end < second_start <= second_end):
        raise RuntimeError("nested extension ranges are not strictly increasing")

    shard_by_path = {
        canonical_shard_path(str(source["path"])): int(source["index"]) for source in sources
    }
    base_ledger_path = resolve_manifest_path(
        base["source_document_ledger"]["path"], args.manifest_base_dir
    )
    base_ids = ledger_identities(
        base_ledger_path,
        expected_documents=int(base["source_document_ledger"]["documents"]),
        shard_by_path=shard_by_path,
    )
    if identity_overlap(base_ids, first.records) or identity_overlap(base_ids, second.records):
        raise RuntimeError("extension overlaps the base pool")
    first_ids = np.empty(len(first.records), dtype=IDENTITY_DTYPE)
    first_ids["shard"], first_ids["doc_index"] = first.records["shard"], first.records["doc_index"]
    if identity_overlap(first_ids, second.records):
        raise RuntimeError("extension chunks overlap")

    source_summary = dict(partition["source"])
    chunk_specs = (
        (
            "nested_extension_1b_to_3b",
            first.records,
            args.first_extension_tokens,
            base_boundary,
            first_end,
            args.first_chunk_output,
            args.first_chunk_manifest,
        ),
        (
            "nested_extension_3b_to_9b",
            second.records,
            args.second_extension_tokens,
            first_end,
            second_end,
            args.second_chunk_output,
            args.second_chunk_manifest,
        ),
    )
    chunk_records: list[dict] = []
    for index, (
        name,
        chunk_records_array,
        requested,
        start,
        end,
        output,
        manifest_path,
    ) in enumerate(chunk_specs):
        materialized = materialize(
            chunk_records_array,
            sources,
            output,
            dtype=dtype,
            eos_token_id=args.eos_token_id,
            alignment_tokens=args.alignment_tokens,
            workers=args.workers,
        )
        ledger_path = output.with_name(output.stem + ".documents.csv.gz")
        ledger = write_ledger(ledger_path, chunk_records_array, sources, include_partition_key=True)
        manifest = write_subset_manifest(
            manifest_path,
            name=name,
            seed=selection_seed,
            domain=selection_domain,
            requested_tokens=requested,
            records=chunk_records_array,
            materialized=materialized,
            ledger=ledger,
            source_summary=source_summary,
            manifest_base_dir=args.manifest_base_dir,
        )
        manifest["selection"].update(
            {
                "method": "global-sha256-document-order-range",
                "start_boundary_exclusive": key_hex(start),
                "base_boundary_inclusive": key_hex(base_boundary),
                "chunk_index": index,
                "cumulative_pool_requested_tokens": args.base_tokens
                + sum(spec[2] for spec in chunk_specs[: index + 1]),
            }
        )
        manifest["disjointness_audit"] = {
            "base_manifest": relative(args.base_manifest, args.manifest_base_dir),
            "base_document_overlap": 0,
            "prior_chunk_document_overlap": 0,
            "validation_test_document_overlap": 0,
            "passed": True,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        chunk_records.append(
            {
                "index": index,
                "manifest": relative(manifest_path, args.manifest_base_dir),
                "manifest_sha256": sha256_file(manifest_path),
                "requested_tokens": requested,
                "real_document_tokens": materialized["real_document_tokens"],
                "materialized_tokens": materialized["materialized_tokens"],
                "documents": len(chunk_records_array),
                "start_boundary_exclusive": key_hex(start),
                "end_boundary_inclusive": key_hex(end),
                "token_sha256": materialized["token_sha256"],
                "ledger_sha256": ledger["sha256"],
            }
        )

    pool_3b = composite_manifest(
        output_path=args.pool_3b_manifest,
        name="nested_train_3b",
        base_manifest_path=args.base_manifest,
        component_paths=[args.first_chunk_manifest],
        source_summary=source_summary,
        selection_domain=selection_domain,
        selection_seed=selection_seed,
        manifest_base_dir=args.manifest_base_dir,
    )
    pool_9b = composite_manifest(
        output_path=args.pool_9b_manifest,
        name="nested_train_9b",
        base_manifest_path=args.base_manifest,
        component_paths=[args.first_chunk_manifest, args.second_chunk_manifest],
        source_summary=source_summary,
        selection_domain=selection_domain,
        selection_seed=selection_seed,
        manifest_base_dir=args.manifest_base_dir,
    )
    pool_manifest = {
        "format": POOL_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source_summary,
        "base": {
            "manifest": relative(args.base_manifest, args.manifest_base_dir),
            "manifest_sha256": sha256_file(args.base_manifest),
            "requested_tokens": args.base_tokens,
            "boundary_key": key_hex(base_boundary),
        },
        "selection": {
            "method": "global-sha256-document-order-prefix-with-materialized-ranges",
            "identity": ["canonical_shard_path", "zero_based_metadata_row_index"],
            "domain": selection_domain,
            "seed": selection_seed,
            "partition_test_boundary_inclusive": key_hex(test_boundary),
        },
        "chunks": chunk_records,
        "pools": {
            "3b": {
                "manifest": relative(args.pool_3b_manifest, args.manifest_base_dir),
                "manifest_sha256": sha256_file(args.pool_3b_manifest),
                "requested_tokens": int(pool_3b["selection"]["requested_tokens"]),
            },
            "9b": {
                "manifest": relative(args.pool_9b_manifest, args.manifest_base_dir),
                "manifest_sha256": sha256_file(args.pool_9b_manifest),
                "requested_tokens": int(pool_9b["selection"]["requested_tokens"]),
            },
        },
        "audit": {
            "base_document_overlap": 0,
            "chunk_document_overlap": 0,
            "validation_test_document_overlap": 0,
            "strictly_increasing_selection_ranges": True,
            "base_preserved_byte_for_byte": True,
            "passed": True,
        },
    }
    args.pool_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.pool_manifest.write_text(json.dumps(pool_manifest, indent=2) + "\n")
    return pool_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--first-chunk-manifest", type=Path, required=True)
    parser.add_argument("--second-chunk-manifest", type=Path, required=True)
    parser.add_argument("--pool-3b-manifest", type=Path, required=True)
    parser.add_argument("--pool-9b-manifest", type=Path, required=True)
    parser.add_argument("--first-chunk-output", type=Path, required=True)
    parser.add_argument("--second-chunk-output", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--base-tokens", type=parse_token_count, default=1_000_000_000)
    parser.add_argument("--first-extension-tokens", type=parse_token_count, default=2_000_000_000)
    parser.add_argument("--second-extension-tokens", type=parse_token_count, default=6_000_000_000)
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
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
