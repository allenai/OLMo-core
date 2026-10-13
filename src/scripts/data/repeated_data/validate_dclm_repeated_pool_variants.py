#!/usr/bin/env python3
"""Independently prove additional repeated pools are exact train-only key prefixes."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from create_dclm_document_split import (
    PARTITION_DOMAIN,
    _key_prefix,
    canonical_shard_path,
    document_key,
    sha256_file,
)
from create_dclm_subset_manifest import (
    DEFAULT_SOURCE_MIX,
    inspect_source,
    resolve_source_paths,
)
from validate_dclm_document_split import validate_ledger, validate_materialized


def scan_expected(args: tuple[dict, int, list[tuple[str, str, int, str]]]) -> dict:
    source, partition_seed, selections = args
    shard_path = canonical_shard_path(str(source["path"]))
    partition_base = hashlib.sha256(_key_prefix(PARTITION_DOMAIN, partition_seed, shard_path))
    selection_specs = {
        name: (
            hashlib.sha256(_key_prefix(domain, selection_seed, shard_path)),
            bytes.fromhex(boundary_hex),
        )
        for name, domain, selection_seed, boundary_hex in selections
    }
    selected: dict[str, list[tuple[int, int, int]]] = {name: [] for name, _, _, _ in selections}
    previous_end = 0
    available_tokens = int(source["available_tokens"])
    with gzip.open(str(source["metadata_path"]), "rt", newline="") as metadata:
        for doc_index, row in enumerate(csv.reader(metadata)):
            start, end = int(row[0]), int(row[1])
            if start != previous_end or end <= start or end > available_tokens:
                raise ValueError(f"invalid metadata at {source['metadata_path']}:{doc_index}")
            partition_key_int: int | None = None
            for name, (selection_base, boundary) in selection_specs.items():
                selection_key = b"".join(
                    word.to_bytes(8, "big") for word in document_key(selection_base, doc_index)
                )
                if selection_key <= boundary:
                    if partition_key_int is None:
                        partition_key = b"".join(
                            word.to_bytes(8, "big")
                            for word in document_key(partition_base, doc_index)
                        )
                        partition_key_int = int.from_bytes(partition_key, "big")
                    selected[name].append((doc_index, end - start, partition_key_int))
            previous_end = end
    if previous_end != available_tokens:
        raise ValueError(f"metadata does not exhaust {source['metadata_path']}")
    return {"path": shard_path, "selected": selected}


def ledger_by_shard(path: Path) -> dict[str, list[tuple[int, int]]]:
    output: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            output[row["shard_path"]].append(
                (int(row["document_index"]), int(row["end"]) - int(row["start"]))
            )
    for rows in output.values():
        rows.sort()
    return output


def resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / value.lstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, action="append", required=True)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--manifest-base-dir", type=Path, required=True)
    parser.add_argument("--source-mix", type=Path, default=DEFAULT_SOURCE_MIX)
    parser.add_argument("--label", default="dclm")
    parser.add_argument("--tokenizer", default="allenai/dolma2-tokenizer")
    parser.add_argument("--data-root", type=Path, default=Path("/weka/oe-training-default/ai2-llm"))
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    parser.add_argument("--skip-token-hashes", action="store_true")
    args = parser.parse_args()

    partition = json.loads(args.partition_manifest.read_text())
    partition_seed = int(partition["key"]["partition_seed"])
    test_boundary = partition["key"]["test_boundary_inclusive"]
    relative_paths = [
        canonical_shard_path(path)
        for path in resolve_source_paths(
            args.source_mix, label=args.label, tokenizer=args.tokenizer
        )
    ]
    inspect_args = [(i, path, str(args.data_root), 4) for i, path in enumerate(relative_paths)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        sources = list(pool.map(inspect_source, inspect_args))
    sources.sort(key=lambda source: int(source["index"]))

    identity_sets: dict[str, set[tuple[str, int]]] = {}
    loaded: dict[str, dict] = {}
    actual_ledgers: dict[str, dict[str, list[tuple[int, int]]]] = {}
    manifests = list(args.sample_manifest)
    if args.reference_manifest is not None:
        manifests.insert(0, args.reference_manifest)
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        selection = manifest["selection"]
        ledger_path = resolve(args.manifest_base_dir, manifest["source_document_ledger"]["path"])
        if sha256_file(ledger_path) != manifest["source_document_ledger"]["sha256"]:
            raise ValueError(f"ledger SHA-256 mismatch: {ledger_path}")
        identities = validate_ledger(
            ledger_path,
            domain=selection["domain"],
            seed=int(selection["seed"]),
            expected_documents=int(selection["selected_documents"]),
            expected_tokens=int(selection["selected_real_document_tokens"]),
            expected_partition_domain=PARTITION_DOMAIN,
            partition_seed=partition_seed,
            minimum_partition_key=test_boundary,
        )
        if not args.skip_token_hashes:
            validate_materialized(args.manifest_base_dir, manifest)
        name = manifest["name"]
        loaded[name] = manifest
        actual_ledgers[name] = ledger_by_shard(ledger_path)
        identity_sets[name] = identities

    selection_specs = [
        (
            name,
            manifest["selection"]["domain"],
            int(manifest["selection"]["seed"]),
            manifest["selection"]["boundary_key"],
        )
        for name, manifest in loaded.items()
    ]
    scan_args = [(source, partition_seed, selection_specs) for source in sources]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        expected = list(pool.map(scan_expected, scan_args))

    for name, manifest in loaded.items():
        selection = manifest["selection"]
        expected_count = 0
        expected_tokens = 0
        for source_result in expected:
            rows = [
                (doc_index, length)
                for doc_index, length, partition_key in source_result["selected"][name]
                if partition_key > int(test_boundary, 16)
            ]
            rows.sort()
            if actual_ledgers[name].get(source_result["path"], []) != rows:
                raise ValueError(
                    f"ledger is not the exact global key prefix in {source_result['path']}"
                )
            expected_count += len(rows)
            expected_tokens += sum(length for _, length in rows)
        if expected_count != len(identity_sets[name]) or expected_tokens != int(
            selection["selected_real_document_tokens"]
        ):
            raise ValueError("independent prefix recount disagrees with manifest")
        requested = int(selection["requested_tokens"])
        if expected_tokens < requested:
            raise ValueError("selection does not reach requested token budget")
        print(
            f"VALID {name}: exact train-only SHA-256 prefix; "
            f"{expected_count:,} whole documents; {expected_tokens:,} real tokens"
        )

    names = list(identity_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            print(
                f"OVERLAP {left} / {right}: {len(identity_sets[left] & identity_sets[right]):,} documents"
            )


if __name__ == "__main__":
    main()
