"""
Scan the tokenized 5-task SFT shards and report the document-length distribution, so we can pick a
power-of-2 packing window (:class:`PackingInstanceSource` requires ``sequence_length`` to be a power
of 2). Documents are single-EOS-separated (id 151643); a document's length includes its trailing EOS.

Reports, per task and overall: doc count, max, mean, p50/p90/p99/p99.9, and how many documents exceed
each candidate window (32768, 40960, 65536) -- i.e. how many would be truncated at that window.

Numpy-only (no torch / olmo_core needed). Run as a small gantry job with the training weka mounted::

    /opt/conda/bin/python src/scripts/train/sft/scan_doc_lengths.py \\
        --data-root /weka/oe-training-default/ai2-llm/checkpoints/prasanns/cptmix_data_ladder40k
"""

import argparse
import glob
from pathlib import Path
from typing import List

import numpy as np

EOS_ID = 151643  # qwen3 <|endoftext|> document separator
TASKS = ["contradiction", "nq", "oolong", "rerank", "outlier"]
WINDOWS = [32768, 40960, 65536]


def _doc_lengths(token_path: str) -> np.ndarray:
    """Document lengths (each includes its trailing EOS) from one raw uint32 token shard."""
    tokens = np.memmap(token_path, mode="r", dtype=np.uint32)
    eos = np.flatnonzero(tokens == EOS_ID)
    bounds = np.concatenate(([-1], eos)).astype(np.int64)
    lengths = np.diff(bounds)
    # Trailing tokens with no final EOS also count as one (truncated/unterminated) document.
    if eos.size == 0 or int(eos[-1]) != len(tokens) - 1:
        lengths = np.append(lengths, (len(tokens) - 1) - int(bounds[-1]))
    return lengths[lengths > 0]


def _report(name: str, lengths: np.ndarray) -> None:
    n = lengths.size
    print(f"\n[{name}]  docs={n:,}")
    if n == 0:
        print("  (no documents found -- check paths)")
        return
    print(
        f"  max={lengths.max():,}  mean={lengths.mean():,.0f}  "
        f"p50={np.percentile(lengths, 50):,.0f}  p90={np.percentile(lengths, 90):,.0f}  "
        f"p99={np.percentile(lengths, 99):,.0f}  p99.9={np.percentile(lengths, 99.9):,.0f}"
    )
    for w in WINDOWS:
        over = int((lengths > w).sum())
        print(
            f"  > {w:>6,} : {over:>8,} docs ({100 * over / n:6.3f}%)  <- truncated at window {w:,}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="dir containing the 5 task subdirs")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    args = ap.parse_args()

    root = Path(args.data_root)
    all_lengths: List[np.ndarray] = []
    for task in args.tasks:
        paths = sorted(glob.glob(str(root / task / "token_ids_part_*.npy")))
        if not paths:
            print(f"\n[{task}]  WARNING: no shards under {root / task}")
            continue
        task_lengths = np.concatenate([_doc_lengths(p) for p in paths])
        all_lengths.append(task_lengths)
        _report(f"{task}  ({len(paths)} shards)", task_lengths)

    if all_lengths:
        _report("ALL TASKS", np.concatenate(all_lengths))
    print("\nDone.")


if __name__ == "__main__":
    main()
