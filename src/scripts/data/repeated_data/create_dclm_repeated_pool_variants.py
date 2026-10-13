#!/usr/bin/env python3
"""Create additional independent 1B repeated-data pools from the 0802 train split.

Pool A was created by ``create_dclm_document_split.py`` with selection seed 1.
This script applies the *same* cryptographic document ordering with new seeds to
create pools B and C.  A document is eligible iff its existing partition key is
strictly above the sealed test boundary, so validation and test remain excluded.

Documents are selected as the shortest global SHA-256-key prefix whose real
token count reaches the requested budget.  Source documents are copied whole;
the only non-source tokens are the manifest-declared EOS alignment tail.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from create_dclm_document_split import (
    PARTITION_DOMAIN,
    REPEATED_DOMAIN,
    SAMPLE_DTYPE,
    _key_prefix,
    canonical_shard_path,
    document_key,
    load_candidates,
    materialize,
    partition_key_is_train,
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

DEFAULT_VARIANTS = ("b:3", "c:4")


def parse_variant(value: str) -> tuple[str, int]:
    try:
        name, seed_text = value.lower().split(":", 1)
        seed = int(seed_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("variant must have the form NAME:SEED") from exc
    if not name.replace("_", "").isalnum() or seed < 0:
        raise argparse.ArgumentTypeError("variant name must be alphanumeric and seed non-negative")
    return name, seed


def scan_shard(args: tuple[dict, int, dict[str, int], dict[str, int], str]) -> dict:
    source, partition_seed, variants, thresholds, candidate_dir_str = args
    shard = int(source["index"])
    shard_path = canonical_shard_path(str(source["path"]))
    available_tokens = int(source["available_tokens"])
    candidate_dir = Path(candidate_dir_str)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    partition_base = hashlib.sha256(_key_prefix(PARTITION_DOMAIN, partition_seed, shard_path))
    sample_bases = {
        name: hashlib.sha256(_key_prefix(REPEATED_DOMAIN, seed, shard_path))
        for name, seed in variants.items()
    }
    paths = {name: candidate_dir / f"{name}-{shard:04d}.bin" for name in variants}
    handles = {name: path.open("wb") for name, path in paths.items()}
    buffers: dict[str, list[tuple]] = {name: [] for name in variants}
    doc_count = 0
    token_count = 0
    previous_end = 0
    counts = {name: 0 for name in sample_bases}
    candidate_tokens = {name: 0 for name in sample_bases}
    try:
        with gzip.open(str(source["metadata_path"]), "rt", newline="") as metadata:
            for doc_index, row in enumerate(csv.reader(metadata)):
                if len(row) < 2:
                    raise ValueError(
                        f"invalid metadata row {doc_index} in {source['metadata_path']}"
                    )
                start, end = int(row[0]), int(row[1])
                if start != previous_end or end <= start or end > available_tokens:
                    raise ValueError(
                        f"non-contiguous/invalid document [{start}, {end}) at "
                        f"{source['metadata_path']}:{doc_index}"
                    )
                partition_key = document_key(partition_base, doc_index)
                length = end - start
                for name, base in sample_bases.items():
                    sample_key = document_key(base, doc_index)
                    if sample_key[0] < int(thresholds[name]):
                        buffers[name].append(
                            (*partition_key, *sample_key, shard, doc_index, start, end)
                        )
                        counts[name] += 1
                        candidate_tokens[name] += length
                        if len(buffers[name]) >= 100_000:
                            np.asarray(buffers[name], dtype=SAMPLE_DTYPE).tofile(handles[name])
                            buffers[name].clear()
                doc_count += 1
                token_count += length
                previous_end = end
        if previous_end != available_tokens or token_count != available_tokens:
            raise ValueError(
                f"metadata covers {previous_end:,}/{available_tokens:,} tokens in "
                f"{source['metadata_path']}"
            )
        for name in sample_bases:
            if buffers[name]:
                np.asarray(buffers[name], dtype=SAMPLE_DTYPE).tofile(handles[name])
    finally:
        for handle in handles.values():
            handle.close()
    return {
        "index": shard,
        "documents": doc_count,
        "tokens": token_count,
        "candidate_paths": {name: str(path) for name, path in paths.items()},
        "candidate_documents": counts,
        "candidate_tokens": candidate_tokens,
    }


def read_identities(path: Path) -> set[tuple[str, int]]:
    identities: set[tuple[str, int]] = set()
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            identities.add((row["shard_path"], int(row["document_index"])))
    return identities


def build(args: argparse.Namespace) -> dict:
    partition = json.loads(args.partition_manifest.read_text())
    if partition.get("format") != "dclm-document-partition-v1":
        raise ValueError("unexpected partition manifest format")
    partition_seed = int(partition["key"]["partition_seed"])
    test_boundary = tuple(
        int(partition["key"]["test_boundary_inclusive"][i : i + 16], 16) for i in range(0, 64, 16)
    )
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
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        sources = list(pool.map(inspect_source, inspect_args))
    sources.sort(key=lambda source: int(source["index"]))
    total_tokens = sum(int(source["available_tokens"]) for source in sources)
    variants = dict(args.variant)
    if len(variants) != len(args.variant):
        raise ValueError("variant names must be unique")
    thresholds = {
        name: threshold_for(
            target_tokens=args.tokens,
            total_tokens=total_tokens,
            multiplier=args.candidate_multiplier,
        )
        for name in variants
    }
    args.candidate_dir.mkdir(parents=True, exist_ok=False)
    scan_args = [
        (source, partition_seed, variants, thresholds, str(args.candidate_dir))
        for source in sources
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        audits = list(pool.map(scan_shard, scan_args))
    audits.sort(key=lambda row: int(row["index"]))

    source_summary = dict(partition["source"])
    artifacts: dict[str, dict] = {}
    identities: dict[str, set[tuple[str, int]]] = {}
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, seed in variants.items():
        records = load_candidates(
            [audit["candidate_paths"][name] for audit in audits], SAMPLE_DTYPE
        )
        records = records[partition_key_is_train(records, test_boundary)]
        records = records[sort_order(records)]
        selection = prefix_selection(records, args.tokens)
        artifact_name = f"repeated_train_1b_{name}"
        data_path = args.output_root / f"dclm_0802_{artifact_name}_uint32.npy"
        ledger_path = args.output_root / f"dclm_0802_{artifact_name}.documents.csv.gz"
        materialized = materialize(
            selection.records,
            sources,
            data_path,
            dtype=dtype,
            eos_token_id=args.eos_token_id,
            alignment_tokens=args.training_alignment_tokens,
            workers=args.workers,
        )
        ledger = write_ledger(ledger_path, selection.records, sources, include_partition_key=True)
        manifest_path = args.manifest_dir / f"dclm_0802_{artifact_name}.json"
        manifest = write_subset_manifest(
            manifest_path,
            name=artifact_name,
            seed=seed,
            domain=REPEATED_DOMAIN,
            requested_tokens=args.tokens,
            records=selection.records,
            materialized=materialized,
            ledger=ledger,
            source_summary=source_summary,
            manifest_base_dir=args.manifest_base_dir,
        )
        identities[name] = {
            (
                canonical_shard_path(str(sources[int(record["shard"])]["path"])),
                int(record["doc_index"]),
            )
            for record in selection.records
        }
        artifacts[name] = {
            "manifest": os.path.relpath(manifest_path, args.manifest_base_dir),
            "manifest_sha256": sha256_file(manifest_path),
            "selection": manifest["selection"],
            "materialized": manifest["materialized"],
            "ledger": manifest["source_document_ledger"],
        }

    reference_ids: set[tuple[str, int]] | None = None
    if args.reference_manifest is not None:
        reference = json.loads(args.reference_manifest.read_text())
        reference_ledger = Path(reference["source_document_ledger"]["path"])
        if not reference_ledger.is_absolute():
            reference_ledger = args.manifest_base_dir / str(reference_ledger).lstrip("/")
        reference_ids = read_identities(reference_ledger)
    overlaps: dict[str, int] = {}
    names = list(variants)
    for index, left in enumerate(names):
        if reference_ids is not None:
            overlaps[f"a_{left}"] = len(reference_ids & identities[left])
        for right in names[index + 1 :]:
            overlaps[f"{left}_{right}"] = len(identities[left] & identities[right])

    summary = {
        "format": "dclm-repeated-pool-variants-v1",
        "partition_manifest": str(args.partition_manifest),
        "partition_manifest_sha256": sha256_file(args.partition_manifest),
        "partition_seed": partition_seed,
        "test_boundary_inclusive": partition["key"]["test_boundary_inclusive"],
        "selection_domain": REPEATED_DOMAIN,
        "variants": artifacts,
        "pairwise_document_overlaps": overlaps,
        "overlap_is_allowed_for_independent_uniform_samples": True,
        "all_samples_train_only_by_partition_boundary": True,
    }
    args.summary_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.summary_manifest.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--summary-manifest", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--variant", action="append", type=parse_variant)
    parser.add_argument("--tokens", type=parse_token_count, default=1_000_000_000)
    parser.add_argument("--candidate-multiplier", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
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
    args = parser.parse_args()
    if args.variant is None:
        args.variant = [parse_variant(value) for value in DEFAULT_VARIANTS]
    if args.candidate_multiplier <= 1:
        parser.error("candidate multiplier must exceed 1")
    return args


def main() -> None:
    args = parse_args()
    if args.summary_manifest.exists():
        raise FileExistsError(args.summary_manifest)
    summary = build(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
