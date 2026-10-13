#!/usr/bin/env python3
"""Materialize a proportional, uniformly randomized, document-safe DCLM subset.

The source DCLM arrays are split into shards of different sizes. This script
allocates the requested token budget across shards in proportion to their
usable token counts, chooses a uniform random token anchor in each shard, and
copies a document-aligned cluster following that anchor. This is proportional
uniform cluster sampling: every token position has the same marginal chance of
anchoring its shard's sample, while large contiguous reads keep materialization
practical on remote object storage.

The materialized array is padded with EOS tokens to the requested training
alignment. Padding happens only after a document boundary, so no source
document is ever truncated. The emitted ``olmo-token-subset-v1`` manifest can
be consumed by the existing fixed-sequence-length dataset reader.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np

DEFAULT_SOURCE_MIX = Path("src/olmo_core/data/mixes/OLMoE-mix-0824.txt")


class Document(NamedTuple):
    shard: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def parse_token_count(value: str) -> int:
    value = value.strip().lower().replace("_", "").replace(",", "")
    multipliers = {"k": 10**3, "m": 10**6, "b": 10**9, "t": 10**12}
    multiplier = multipliers.get(value[-1], 1)
    number = value[:-1] if multiplier != 1 else value
    parsed = float(number) * multiplier
    if not parsed.is_integer() or parsed <= 0:
        raise argparse.ArgumentTypeError(f"invalid positive token count: {value!r}")
    return int(parsed)


def resolve_source_paths(source_mix: Path, *, label: str, tokenizer: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    with source_mix.open() as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row_label, path = line.split(",", 1)
            except ValueError as exc:
                raise ValueError(f"invalid row at {source_mix}:{line_number}") from exc
            if row_label != label:
                continue
            path = path.replace("{TOKENIZER}", tokenizer)
            if path not in seen:
                seen.add(path)
                paths.append(path)
    if not paths:
        raise ValueError(f"no paths with label {label!r} in {source_mix}")
    return paths


def inspect_source(args: tuple[int, str, str, int]) -> dict[str, int | str]:
    index, relative_path, data_root_str, item_size = args
    absolute_path = Path(data_root_str) / relative_path.lstrip("/")
    metadata_path = absolute_path.with_suffix(".csv.gz")
    if not absolute_path.is_file():
        raise FileNotFoundError(absolute_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    file_size = absolute_path.stat().st_size
    if file_size % item_size:
        raise ValueError(f"{absolute_path} is not divisible by dtype item size {item_size}")
    return {
        "index": index,
        "path": relative_path,
        "absolute_path": str(absolute_path),
        "metadata_path": str(metadata_path),
        "available_tokens": file_size // item_size,
    }


def proportional_allocation(weights: list[int], total: int) -> list[int]:
    """Largest-remainder allocation whose sum is exactly ``total``."""
    weight_sum = sum(weights)
    if total < 0 or not weights or weight_sum <= 0:
        raise ValueError("weights must be non-empty and positive, and total non-negative")
    numerators = [total * weight for weight in weights]
    result = [numerator // weight_sum for numerator in numerators]
    remainder = total - sum(result)
    order = sorted(
        range(len(weights)),
        key=lambda idx: (numerators[idx] % weight_sum, -idx),
        reverse=True,
    )
    for idx in order[:remainder]:
        result[idx] += 1
    return result


def _shard_seed(seed: int, relative_path: str) -> int:
    payload = f"{seed}\0{relative_path}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def sample_candidates(args: tuple[dict, int, int, float]) -> tuple[int, list[Document]]:
    source, quota, seed, candidate_multiplier = args
    shard = int(source["index"])
    available_tokens = int(source["available_tokens"])
    rng = np.random.default_rng(_shard_seed(seed, str(source["path"])))
    anchor = int(rng.integers(0, available_tokens))
    candidate_token_target = min(available_tokens, math.ceil(candidate_multiplier * quota))
    candidates: list[Document] = []

    def collect(*, before_anchor: bool) -> None:
        collected = sum(doc.length for doc in candidates)
        with gzip.open(str(source["metadata_path"]), "rt") as f:
            for line in f:
                if collected >= candidate_token_target:
                    break
                start_str, end_str, *_ = line.split(",")
                start, end = int(start_str), int(end_str)
                if before_anchor:
                    if start >= anchor:
                        break
                elif start < anchor:
                    continue
                if not 0 <= start < end <= available_tokens:
                    raise ValueError(
                        f"invalid document [{start}, {end}) in {source['metadata_path']}"
                    )
                candidates.append(Document(shard, start, end))
                collected += end - start

    # Start at the first complete document whose beginning is at or after the
    # uniform token anchor. Wrap to the beginning only when the anchor falls
    # too close to the end to provide the requested candidate mass.
    collect(before_anchor=False)
    if sum(doc.length for doc in candidates) < candidate_token_target:
        collect(before_anchor=True)
    if sum(doc.length for doc in candidates) < min(quota, available_tokens):
        raise RuntimeError(
            f"unable to collect enough complete documents around token anchor {anchor:,} "
            f"in {source['metadata_path']}"
        )
    return shard, candidates


def choose_documents(
    candidates_by_shard: list[list[Document]], quotas: list[int], *, seed: int
) -> tuple[list[Document], list[int]]:
    """Fill proportional shard quotas, then use unused candidates to fill globally."""
    chosen: list[Document] = []
    unused: list[Document] = []
    selected_tokens = [0] * len(quotas)
    for shard, (candidates, quota) in enumerate(zip(candidates_by_shard, quotas)):
        for doc in candidates:
            if doc.length <= quota - selected_tokens[shard]:
                chosen.append(doc)
                selected_tokens[shard] += doc.length
            else:
                unused.append(doc)

    remaining = sum(quotas) - sum(selected_tokens)
    rng = np.random.default_rng(seed ^ 0xD1C1_5A1E)
    rng.shuffle(unused)
    for doc in unused:
        if doc.length <= remaining:
            chosen.append(doc)
            selected_tokens[doc.shard] += doc.length
            remaining -= doc.length
            if remaining == 0:
                break
    if not chosen:
        raise RuntimeError("sampling selected no documents")
    return chosen, selected_tokens


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_gzip_text_writer(path: Path):
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return io.TextIOWrapper(compressed)


def write_shard_audit(
    *, shards: list[dict], output_path: Path, manifest_base_dir: Path
) -> dict[str, int | float | str]:
    audit_path = output_path.with_suffix(".shards.json.gz")
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite existing shard audit: {audit_path}")
    payload = json.dumps(shards, sort_keys=True, separators=(",", ":")).encode()
    tmp_path = audit_path.with_name(audit_path.name + ".tmp")
    try:
        with tmp_path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
                f.write(payload)
        os.replace(tmp_path, audit_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    differences = [shard["selected_tokens"] - shard["token_quota"] for shard in shards]
    return {
        "path": os.path.relpath(audit_path, manifest_base_dir),
        "sha256": _sha256_file(audit_path),
        "contents_sha256": hashlib.sha256(payload).hexdigest(),
        "num_shards": len(shards),
        "sum_available_tokens": sum(shard["available_tokens"] for shard in shards),
        "sum_token_quotas": sum(shard["token_quota"] for shard in shards),
        "sum_selected_document_tokens": sum(shard["selected_tokens"] for shard in shards),
        "max_absolute_token_deviation_from_quota": max(map(abs, differences)),
        "mean_absolute_token_deviation_from_quota": sum(map(abs, differences)) / len(shards),
        "max_relative_deviation_from_quota": max(
            abs(difference) / shard["token_quota"] for difference, shard in zip(differences, shards)
        ),
    }


def materialize(
    *,
    documents: list[Document],
    sources: list[dict],
    output_path: Path,
    target_tokens: int,
    dtype: np.dtype,
    eos_token_id: int,
) -> tuple[int, int, str, str]:
    metadata_path = output_path.with_suffix(".csv.gz")
    for path in (output_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_tmp = output_path.with_name(output_path.name + ".tmp")
    metadata_tmp = metadata_path.with_name(metadata_path.name + ".tmp")

    by_shard: dict[int, list[Document]] = defaultdict(list)
    for doc in documents:
        by_shard[doc.shard].append(doc)

    offset = 0
    token_digest = hashlib.sha256()
    try:
        with token_tmp.open("wb") as token_out, deterministic_gzip_text_writer(
            metadata_tmp
        ) as metadata_out:
            for shard in sorted(by_shard):
                source = sources[shard]
                array = np.memmap(source["absolute_path"], mode="r", dtype=dtype)
                shard_documents = sorted(by_shard[shard], key=lambda doc: doc.start)
                spans: list[list] = []
                for doc in shard_documents:
                    if spans and spans[-1][1] == doc.start:
                        spans[-1][1] = doc.end
                        spans[-1][2].append(doc)
                    else:
                        spans.append([doc.start, doc.end, [doc]])
                for span_start, span_end, span_docs in spans:
                    values = np.asarray(array[span_start:span_end])
                    payload = values.tobytes(order="C")
                    token_out.write(payload)
                    token_digest.update(payload)
                    local_offset = 0
                    for doc in span_docs:
                        local_offset += doc.length
                        if int(values[local_offset - 1]) != eos_token_id:
                            raise ValueError(
                                f"document [{doc.start}, {doc.end}) in {source['path']} does "
                                f"not end with EOS token {eos_token_id}"
                            )
                        metadata_out.write(f"{offset},{offset + doc.length}\n")
                        offset += doc.length
                del array

            padding_tokens = target_tokens - offset
            if padding_tokens < 0:
                raise AssertionError("selected documents exceed materialized target")
            if padding_tokens:
                padding = np.full(padding_tokens, eos_token_id, dtype=dtype)
                payload = padding.tobytes(order="C")
                token_out.write(payload)
                token_digest.update(payload)
                for _ in range(padding_tokens):
                    metadata_out.write(f"{offset},{offset + 1}\n")
                    offset += 1
        if offset != target_tokens:
            raise AssertionError(f"wrote {offset:,} tokens, expected {target_tokens:,}")
        os.replace(token_tmp, output_path)
        os.replace(metadata_tmp, metadata_path)
    except BaseException:
        token_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
        raise

    return (
        target_tokens - padding_tokens,
        padding_tokens,
        token_digest.hexdigest(),
        _sha256_file(metadata_path),
    )


def manifest_digest(entries: list[dict[str, int | str]]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_and_materialize(args: argparse.Namespace) -> dict:
    if args.alignment_tokens % args.sequence_length:
        raise ValueError("alignment tokens must be divisible by sequence length")
    dtype = np.dtype(args.dtype)
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f"token dtype must be an integer type, got {dtype.name}")
    aligned_tokens = math.ceil(args.target_tokens / args.alignment_tokens) * args.alignment_tokens

    relative_paths = resolve_source_paths(
        args.source_mix, label=args.label, tokenizer=args.tokenizer
    )
    inspect_args = [
        (idx, path, str(args.data_root), dtype.itemsize) for idx, path in enumerate(relative_paths)
    ]
    if args.workers == 1:
        sources = list(map(inspect_source, inspect_args))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            sources = list(pool.map(inspect_source, inspect_args))
    sources.sort(key=lambda source: int(source["index"]))
    quotas = proportional_allocation(
        [int(source["available_tokens"]) for source in sources], aligned_tokens
    )

    candidate_args = [
        (source, quota, args.seed, args.candidate_multiplier)
        for source, quota in zip(sources, quotas)
    ]
    candidates_by_shard: list[list[Document]] = [[] for _ in sources]
    if args.workers == 1:
        sampled_candidates = map(sample_candidates, candidate_args)
        for shard, candidates in sampled_candidates:
            candidates_by_shard[shard] = candidates
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for shard, candidates in pool.map(sample_candidates, candidate_args):
                candidates_by_shard[shard] = candidates
    documents, selected_tokens_by_shard = choose_documents(
        candidates_by_shard, quotas, seed=args.seed
    )
    padding_tokens = aligned_tokens - sum(selected_tokens_by_shard)
    if padding_tokens > args.max_padding_tokens:
        raise RuntimeError(
            f"sampling left {padding_tokens:,} unfilled tokens, above the configured maximum "
            f"of {args.max_padding_tokens:,}; rerun with a larger --candidate-multiplier"
        )

    real_tokens, padding_tokens, token_sha256, metadata_sha256 = materialize(
        documents=documents,
        sources=sources,
        output_path=args.materialized_output,
        target_tokens=aligned_tokens,
        dtype=dtype,
        eos_token_id=args.eos_token_id,
    )
    relative_output = os.path.relpath(args.materialized_output, args.manifest_base_dir)
    entry = {
        "path": relative_output,
        "start_instance": 0,
        "num_instances": aligned_tokens // args.sequence_length,
        "available_instances": aligned_tokens // args.sequence_length,
        "available_tokens": aligned_tokens,
    }
    entries = [entry]
    shard_stats = []
    doc_counts = defaultdict(int)
    for doc in documents:
        doc_counts[doc.shard] += 1
    for source, quota, selected_tokens in zip(sources, quotas, selected_tokens_by_shard):
        shard_stats.append(
            {
                "path": source["path"],
                "available_tokens": source["available_tokens"],
                "uniform_token_anchor_seed": _shard_seed(args.seed, str(source["path"])),
                "token_quota": quota,
                "selected_documents": doc_counts[int(source["index"])],
                "selected_tokens": selected_tokens,
            }
        )
    shard_audit = write_shard_audit(
        shards=shard_stats,
        output_path=args.materialized_output,
        manifest_base_dir=args.manifest_base_dir,
    )

    source_mix_sha256 = hashlib.sha256(args.source_mix.read_bytes()).hexdigest()
    return {
        "format": "olmo-token-subset-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "mix": "dclm-full",
            "source_mix_file": args.source_mix.name,
            "source_mix_sha256": source_mix_sha256,
            "label": args.label,
            "tokenizer": args.tokenizer,
            "dtype": dtype.name,
            "data_root_hint": str(args.manifest_base_dir),
            "original_data_root": str(args.data_root),
            "original_num_source_paths": len(sources),
            "original_available_tokens": sum(int(s["available_tokens"]) for s in sources),
        },
        "selection": {
            "method": "proportional-shard-uniform-token-anchor-document-clusters",
            "seed": args.seed,
            "requested_tokens": args.target_tokens,
            "alignment_tokens": args.alignment_tokens,
            "sequence_length": args.sequence_length,
            "selected_tokens": aligned_tokens,
            "selected_instances": aligned_tokens // args.sequence_length,
            "selected_real_document_tokens": real_tokens,
            "padding_eos_tokens": padding_tokens,
            "selected_documents": len(documents),
            "num_source_paths": len(entries),
            "candidate_multiplier": args.candidate_multiplier,
            "allocation": "largest-remainder proportional to source shard token count",
        },
        "materialized": {
            "path": relative_output,
            "token_sha256": token_sha256,
            "document_metadata_path": os.path.relpath(
                args.materialized_output.with_suffix(".csv.gz"), args.manifest_base_dir
            ),
            "document_metadata_sha256": metadata_sha256,
        },
        "shard_audit": shard_audit,
        "entries_sha256": manifest_digest(entries),
        "entries": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-tokens", required=True, type=parse_token_count)
    parser.add_argument("--output", required=True, type=Path, help="Manifest JSON path")
    parser.add_argument("--materialized-output", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("/weka/oe-training-default/ai2-llm"))
    parser.add_argument(
        "--manifest-base-dir",
        type=Path,
        default=Path("/weka/oe-training-default/ai2-llm"),
        help="Base directory used by NumpyFSLDatasetConfig.mix_base_dir",
    )
    parser.add_argument("--source-mix", type=Path, default=DEFAULT_SOURCE_MIX)
    parser.add_argument("--label", default="dclm")
    parser.add_argument("--tokenizer", default="allenai/dolma2-tokenizer")
    parser.add_argument("--dtype", default="uint32")
    parser.add_argument("--eos-token-id", type=int, default=100257)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--alignment-tokens", type=parse_token_count, default=4_194_304)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--candidate-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--max-padding-tokens",
        type=int,
        default=4095,
        help="Fail rather than pad more than this many final EOS tokens",
    )
    parser.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Allow replacing the manifest JSON; materialized data are never overwritten",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite_manifest:
        raise FileExistsError(
            f"refusing to overwrite manifest without --overwrite-manifest: {args.output}"
        )
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.candidate_multiplier <= 1:
        raise ValueError("candidate multiplier must be greater than 1")
    if args.max_padding_tokens < 0:
        raise ValueError("max padding tokens must be non-negative")
    manifest = build_and_materialize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = args.output.with_name(args.output.name + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(manifest_tmp, args.output)
    selection = manifest["selection"]
    print(
        f"Wrote {args.output}: {selection['selected_tokens']:,} tokens, "
        f"{selection['selected_documents']:,} complete source documents, "
        f"{selection['padding_eos_tokens']:,} EOS padding tokens; entries sha256 "
        f"{manifest['entries_sha256']}"
    )


if __name__ == "__main__":
    main()
