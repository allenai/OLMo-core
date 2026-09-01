"""
Stage 1a: sample N_PER_BUCKET real documents per (task, context-length bucket) from each distinct
SFT training-data group in ``models.DATA_GROUPS``, ONCE, and materialize them (token ids + label
mask) to weka so every model that shares a data group scores the identical documents.

Sampling is PER TASK, not pooled across a group's tasks: each data group has several constituent
tasks (contradiction/nq/oolong/rerank/outlier, +dolci_instruct_sft for some groups -- see
``models.DATA_GROUPS``), and every (task, bucket) pair gets its own independent 50-example sample.
This mirrors the val manifest's per-task granularity and lets compute_loss.py report train loss
broken out by task, not just by bucket.

Samples individual documents straight out of the same raw per-task ``token_ids_part_*.npy`` /
``labels_mask_*.npy`` shards the training launchers mix and pack from (see ``models.py`` for exactly
which roots, taken verbatim from each launcher). This is a faithful proxy for "the SFT training
data" -- it does NOT replay the packer's mixing ratios or windowing (each task is sampled at the
SAME rate here regardless of its training mixing weight). It also does not exercise any
document-chunk / summary-token / landmark structural augmentation beyond what is ALREADY baked into
the stored tokens (those augmentations happen at the conversion step, before these shards are
written, so the raw shards for docchunk/summtoken groups already contain box markers / summary-token
runs verbatim).

Uses NumpyDocumentSource.get_document_offsets() (cheap: EOS-boundary scan, no big reads) to find
every document's length, buckets it against models.LENGTH_LADDER, and reservoir-samples
N_PER_BUCKET per (task, bucket) -- so document identity is fixed independent of how many documents
exist upstream. Only the sampled documents' tokens are then materialized via get_token_range().

Needs weka mounted (run on a CPU gantry job, e.g. the one-off template in beaker.md). No GPU
required.

Usage:
    PYTHONPATH=src python build_train_manifest.py [--groups pair1_5task_dolci25_32k_qwen35,...]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
from models import (
    DATA_GROUPS,
    LENGTH_LADDER,
    LENGTH_LADDER_LABELS,
    N_PER_BUCKET,
    SEED,
    TRAIN_MANIFEST_DIR,
)


def assign_bucket(length: int) -> Optional[str]:
    for threshold, label in zip(LENGTH_LADDER, LENGTH_LADDER_LABELS):
        if length <= threshold:
            return label
    return None  # longer than the largest bucket (262144) -- drop


def build_group_manifest(group_name: str, work_dir: str) -> None:
    from olmo_core.data import TokenizerConfig
    from olmo_core.data.composable import NumpyDocumentSourceConfig

    group = DATA_GROUPS[group_name]
    tok_cfg = getattr(TokenizerConfig, group.tokenizer_family)()
    doc_tok_cfg = replace(tok_cfg, bos_token_id=None)

    rng = random.Random(SEED)
    tasks = list(group.sources.keys())
    # reservoir[(task, bucket)] = list of (src_idx, doc_start, doc_end, length)
    reservoir: Dict[Tuple[str, str], List[Tuple[int, int, int, int]]] = {
        (task, lab): [] for task in tasks for lab in LENGTH_LADDER_LABELS
    }
    seen_count: Dict[Tuple[str, str], int] = {k: 0 for k in reservoir}

    # Keep the built NumpyDocumentSource objects around (indexed by (task, source_idx)) so the
    # second pass can call get_token_range() on exactly the same objects the offsets came from.
    built_sources: Dict[Tuple[str, int], object] = {}

    t0 = time.time()
    for task, root in group.sources.items():
        cfg = NumpyDocumentSourceConfig(
            source_paths=[f"{root}/token_ids_part_*.npy"],
            tokenizer=doc_tok_cfg,
            label_mask_paths=[f"{root}/labels_mask_*.npy"],
            expand_glob=True,
        )
        sources = cfg.build(work_dir=work_dir)
        print(
            f"[{group_name}] task={task} root={root} -> {len(sources)} shard source(s)", flush=True
        )
        for src_idx, src in enumerate(sources):
            built_sources[(task, src_idx)] = src
            n_docs = 0
            for doc_start, doc_end in src.get_document_offsets():
                n_docs += 1
                length = doc_end - doc_start
                bucket = assign_bucket(length)
                if bucket is None:
                    continue
                key = (task, bucket)
                seen_count[key] += 1
                item = (src_idx, doc_start, doc_end, length)
                if len(reservoir[key]) < N_PER_BUCKET:
                    reservoir[key].append(item)
                else:
                    j = rng.randint(0, seen_count[key] - 1)
                    if j < N_PER_BUCKET:
                        reservoir[key][j] = item
            print(f"    shard {src_idx}: {n_docs} documents scanned", flush=True)

    print(f"[{group_name}] offsets scan done in {time.time() - t0:.1f}s.", flush=True)
    for task in tasks:
        print(
            f"    {task}: "
            + ", ".join(
                f"{lab}={seen_count[(task, lab)]}->{len(reservoir[(task, lab)])}"
                for lab in LENGTH_LADDER_LABELS
            ),
            flush=True,
        )

    # Materialize sampled documents' tokens + label mask.
    npz_payload: Dict[str, np.ndarray] = {}
    index: List[dict] = []
    for task in tasks:
        for bucket in LENGTH_LADDER_LABELS:
            for i, (src_idx, doc_start, doc_end, length) in enumerate(reservoir[(task, bucket)]):
                src = built_sources[(task, src_idx)]
                tr = src.get_token_range(
                    doc_start, doc_end
                )  # {"input_ids": ..., "label_mask": ...}
                key = f"{task}__{bucket}__{i}"
                npz_payload[f"{key}__ids"] = np.asarray(tr["input_ids"], dtype=np.int64)
                label_mask = tr.get("label_mask")
                if label_mask is None:
                    raise RuntimeError(
                        f"{group_name}/{task}: source has no label_mask -- every training source "
                        f"in models.py DATA_GROUPS is expected to carry label_mask_paths."
                    )
                npz_payload[f"{key}__mask"] = np.asarray(label_mask, dtype=bool)
                index.append({"task": task, "bucket": bucket, "i": i, "length": int(length)})

    os.makedirs(TRAIN_MANIFEST_DIR, exist_ok=True)
    npz_path = f"{TRAIN_MANIFEST_DIR}/{group_name}.npz"
    idx_path = f"{TRAIN_MANIFEST_DIR}/{group_name}.index.json"
    np.savez_compressed(npz_path, **npz_payload)
    with open(idx_path, "w") as f:
        json.dump(
            {
                "group": group_name,
                "seed": SEED,
                "n_per_bucket": N_PER_BUCKET,
                "population_by_task_bucket": {
                    f"{task}@{bucket}": seen_count[(task, bucket)]
                    for task in tasks
                    for bucket in LENGTH_LADDER_LABELS
                },
                "sampled_by_task_bucket": {
                    f"{task}@{bucket}": len(reservoir[(task, bucket)])
                    for task in tasks
                    for bucket in LENGTH_LADDER_LABELS
                },
                "examples": index,
            },
            f,
            indent=2,
        )
    print(f"[{group_name}] wrote {npz_path} and {idx_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--groups",
        default=None,
        help="comma list restricting which data groups to build (default: all)",
    )
    ap.add_argument("--work-dir", default=f"{TRAIN_MANIFEST_DIR}/_work")
    args = ap.parse_args()

    groups = args.groups.split(",") if args.groups else list(DATA_GROUPS.keys())
    os.makedirs(args.work_dir, exist_ok=True)
    for g in groups:
        if g not in DATA_GROUPS:
            raise SystemExit(f"unknown data group {g!r}; known: {sorted(DATA_GROUPS)}")
        build_group_manifest(g, args.work_dir)


if __name__ == "__main__":
    main()
