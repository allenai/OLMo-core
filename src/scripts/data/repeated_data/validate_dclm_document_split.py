#!/usr/bin/env python3
"""Independently validate generated 0802 DCLM partition artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from create_dclm_document_split import (
    PARTITION_DOMAIN,
    REPEATED_DOMAIN,
    UNIQUE_DOMAIN,
    _key_prefix,
    document_key,
    key_hex,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(base_dir: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_dir / path.lstrip("/")


def validate_ledger(
    path: Path,
    *,
    domain: str,
    seed: int,
    expected_documents: int,
    expected_tokens: int,
    holdout_ids: set[tuple[str, int]] | None = None,
    expected_partition_domain: str | None = None,
    partition_seed: int | None = None,
    minimum_partition_key: str | None = None,
) -> set[tuple[str, int]]:
    identities: set[tuple[str, int]] = set()
    previous_key = ""
    real_tokens = 0
    with gzip.open(path, "rt", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=1):
            shard_path = row["shard_path"]
            doc_index = int(row["document_index"])
            start, end = int(row["start"]), int(row["end"])
            identity = (shard_path, doc_index)
            if end <= start:
                raise ValueError(f"invalid range in {path}:{row_number}")
            expected_key = key_hex(
                document_key(hashlib.sha256(_key_prefix(domain, seed, shard_path)), doc_index)
            )
            if row["selection_key"] != expected_key:
                raise ValueError(f"selection-key mismatch in {path}:{row_number}")
            if previous_key and row["selection_key"] <= previous_key:
                raise ValueError(f"ledger is not strictly key-sorted in {path}:{row_number}")
            previous_key = row["selection_key"]
            if identity in identities:
                raise ValueError(f"duplicate document identity in {path}:{row_number}")
            if holdout_ids is not None and identity in holdout_ids:
                raise ValueError(f"training sample intersects holdout in {path}:{row_number}")
            identities.add(identity)
            real_tokens += end - start
            if expected_partition_domain is not None:
                partition_key = key_hex(
                    document_key(
                        hashlib.sha256(
                            _key_prefix(expected_partition_domain, int(partition_seed), shard_path)
                        ),
                        doc_index,
                    )
                )
                if row["partition_key"] != partition_key:
                    raise ValueError(f"partition-key mismatch in {path}:{row_number}")
                if minimum_partition_key is not None and partition_key <= minimum_partition_key:
                    raise ValueError(
                        f"training document is not in training partition: {path}:{row_number}"
                    )
    if len(identities) != expected_documents:
        raise ValueError(f"{path}: {len(identities):,} docs != {expected_documents:,}")
    if real_tokens != expected_tokens:
        raise ValueError(f"{path}: {real_tokens:,} tokens != {expected_tokens:,}")
    return identities


def validate_materialized(base_dir: Path, manifest: dict) -> None:
    entry = manifest["entries"][0]
    token_path = resolve(base_dir, entry["path"])
    expected_tokens = int(manifest["selection"]["selected_tokens"])
    if (
        token_path.stat().st_size
        != expected_tokens * np.dtype(manifest["source"]["dtype"]).itemsize
    ):
        raise ValueError(f"unexpected token file size: {token_path}")
    if sha256_file(token_path) != manifest["materialized"]["token_sha256"]:
        raise ValueError(f"token SHA-256 mismatch: {token_path}")
    metadata_path = resolve(base_dir, manifest["materialized"]["document_metadata_path"])
    if sha256_file(metadata_path) != manifest["materialized"]["document_metadata_sha256"]:
        raise ValueError(f"metadata SHA-256 mismatch: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--manifest-base-dir", type=Path, required=True)
    parser.add_argument("--skip-token-hashes", action="store_true")
    args = parser.parse_args()
    partition = json.loads(args.partition_manifest.read_text())
    if partition.get("format") != "dclm-document-partition-v1":
        raise ValueError("unexpected partition manifest format")
    if not partition["uniformity"]["sanity_check_abs_chi_square_z_below_5"]:
        raise ValueError("partition hash buckets failed uniformity sanity check")
    split = partition["partition"]
    if (
        sum(split[name]["real_tokens"] for name in ("train", "validation", "test"))
        != split["token_count_sum"]
    ):
        raise ValueError("partition token counts are not exhaustive")
    if (
        sum(split[name]["documents"] for name in ("train", "validation", "test"))
        != split["document_count_sum"]
    ):
        raise ValueError("partition document counts are not exhaustive")

    domains = {
        "validation": PARTITION_DOMAIN,
        "test": PARTITION_DOMAIN,
        "repeated_train_1b": REPEATED_DOMAIN,
        "unique_train_5b": UNIQUE_DOMAIN,
    }
    holdout_ids: set[tuple[str, int]] = set()
    for name in ("validation", "test", "repeated_train_1b", "unique_train_5b"):
        artifact = partition["artifacts"][name]
        manifest_path = resolve(args.manifest_base_dir, artifact["manifest"])
        manifest = json.loads(manifest_path.read_text())
        if sha256_file(manifest_path) != artifact["manifest_sha256"]:
            raise ValueError(f"manifest SHA-256 mismatch: {manifest_path}")
        ledger_path = resolve(args.manifest_base_dir, manifest["source_document_ledger"]["path"])
        if sha256_file(ledger_path) != manifest["source_document_ledger"]["sha256"]:
            raise ValueError(f"ledger SHA-256 mismatch: {ledger_path}")
        is_training = name.startswith("repeated") or name.startswith("unique")
        identities = validate_ledger(
            ledger_path,
            domain=domains[name],
            seed=int(manifest["selection"]["seed"]),
            expected_documents=int(manifest["selection"]["selected_documents"]),
            expected_tokens=int(manifest["selection"]["selected_real_document_tokens"]),
            holdout_ids=holdout_ids if is_training else None,
            expected_partition_domain=PARTITION_DOMAIN if is_training else None,
            partition_seed=int(partition["key"]["partition_seed"]) if is_training else None,
            minimum_partition_key=partition["key"]["test_boundary_inclusive"]
            if is_training
            else None,
        )
        if not is_training:
            if holdout_ids.intersection(identities):
                raise ValueError("validation and test ledgers intersect")
            holdout_ids.update(identities)
        if not args.skip_token_hashes:
            validate_materialized(args.manifest_base_dir, manifest)

    print(
        "VALID partition: "
        f"{split['document_count_sum']:,} documents, {split['token_count_sum']:,} tokens; "
        "validation/test disjoint; both training samples are training-only; hashes verified"
    )


if __name__ == "__main__":
    main()
