"""
Sanity check for the packing + masking fixes in ``Qwen3-4B-dense-5task-32k-nocpt-SFT.py``.

It builds a small synthetic 5-task mix in the *exact on-disk format* the real shards use (raw
headerless ``uint32`` token ids, each document terminated by the qwen3 EOS id ``151643``, plus a
parallel raw ``bool`` ``labels_mask``), then runs it through the *real* ``PackingInstanceSource`` +
``ComposableDataLoader`` with the same settings as the training script:

  * ``PackingInstanceSource`` with ``LongDocStrategy.truncate`` -- whole-document bin-packing.
  * loader tokenizer with ``bos_token_id=None`` so the EOS-based ``doc_lens`` detection splits on
    every EOS (qwen3 has ``bos==eos==151643``; the default would only split on EOS->BOS adjacency,
    which never occurs in single-EOS-separated SFT data).

For every packed window it asserts:
  1. ``doc_lens.sum() == sequence_length``                    (boundaries tile the whole window)
  2. every internal boundary lands immediately after an EOS   (no boundary mid-document)
  3. no whole document is split across the window boundary    (packing kept documents intact)
and reports (does not assert):
  * the number of all-masked windows (these would give a NaN loss),
  * the average padding fraction,
  * a negative control: the same data with the BUGGY ``bos==eos`` tokenizer, showing ``doc_lens``
    collapses so documents would attend across each other.

Run locally (no GPU / no weka needed)::

    PYTHONPATH=src python src/scripts/train/sft/sanity_check_packing.py
    PYTHONPATH=src python src/scripts/train/sft/sanity_check_packing.py --seq-len 256

Optionally point it at real shards (a directory with the 5 task subdirs)::

    PYTHONPATH=src python src/scripts/train/sft/sanity_check_packing.py \\
        --data-root /weka/oe-training-default/ai2-llm/checkpoints/prasanns/cptmix_data_ladder40k \\
        --seq-len 40960
"""

import argparse
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from olmo_core.data import TokenizerConfig
from olmo_core.data.collator import DataCollator
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LongDocStrategy,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
    PackingInstanceSourceConfig,
)
from olmo_core.data.types import NumpyDatasetDType

EOS_ID = 151643  # qwen3 <|endoftext|> == eos == bos == pad
TASKS = ["contradiction", "nq", "oolong", "rerank", "outlier"]
WEIGHTS = {"contradiction": 2.0, "nq": 1.0, "oolong": 1.0, "rerank": 1.5, "outlier": 1.5}


def _write_synthetic_task(
    out_dir: Path, *, n_docs: int, seq_len: int, rng: np.random.Generator
) -> None:
    """Write one task's ``token_ids_part_000000.npy`` + ``labels_mask_000000.npy`` (raw, headerless)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens: List[int] = []
    mask: List[bool] = []
    for i in range(n_docs):
        # A few deliberately over-length docs to exercise LongDocStrategy.truncate. Keep their
        # first chunk partly unmasked so truncation doesn't produce an all-masked window.
        if i % 17 == 0:
            body_len = int(seq_len * 1.5)
        else:
            body_len = int(rng.integers(8, max(9, seq_len - 8)))
        body = rng.integers(10, 1000, size=body_len).tolist()  # never 151643 inside a body
        # SFT shape: a masked "prompt" prefix then an unmasked "answer", then the EOS separator.
        prompt_len = max(1, body_len // 2)
        doc_tokens = body + [EOS_ID]
        doc_mask = [False] * prompt_len + [True] * (body_len - prompt_len) + [True]  # eos -> answer
        tokens.extend(doc_tokens)
        mask.extend(doc_mask)

    tok = np.asarray(tokens, dtype=np.uint32)
    msk = np.asarray(mask, dtype=np.bool_)
    np.memmap(out_dir / "token_ids_part_000000.npy", mode="w+", dtype=np.uint32, shape=tok.shape)[
        :
    ] = tok
    np.memmap(out_dir / "labels_mask_000000.npy", mode="w+", dtype=np.bool_, shape=msk.shape)[
        :
    ] = msk


def _build_specs(data_root: Path) -> List[MixingDocumentSourceSpecConfig]:
    doc_tok = replace(TokenizerConfig.qwen3(), bos_token_id=None)
    wsum = sum(WEIGHTS.values())
    specs = []
    for task in TASKS:
        r = data_root / task
        specs.append(
            MixingDocumentSourceSpecConfig(
                source=NumpyDocumentSourceConfig(
                    source_paths=[f"{r}/token_ids_part_*.npy"],
                    tokenizer=doc_tok,
                    label_mask_paths=[f"{r}/labels_mask_*.npy"],
                    dtype=NumpyDatasetDType.uint32,
                    expand_glob=True,
                ),
                ratio=WEIGHTS[task] / wsum,
                max_repetition_factor=8.0,
                label=task,
            )
        )
    return specs


def _iter_instances(
    specs, *, seq_len: int, work_dir: Path, bos_token_id: Optional[int], gbs_instances: int = 4
):
    """Build the real loader and yield (input_ids, label_mask, doc_lens) per instance."""
    loader_tok = replace(TokenizerConfig.qwen3(), bos_token_id=bos_token_id)
    instance_source = PackingInstanceSourceConfig(
        sources=[MixingDocumentSourceConfig(source_specs=specs)],
        sequence_length=seq_len,
        tokenizer=replace(TokenizerConfig.qwen3(), bos_token_id=None),
        long_doc_strategy=LongDocStrategy.truncate,
    ).build(work_dir)

    # ComposableDataLoaderConfig.build wires the collator + generate_doc_lengths exactly as training.
    loader = ComposableDataLoaderConfig(
        tokenizer=loader_tok,
        work_dir=str(work_dir),
        global_batch_size=seq_len * gbs_instances,
        seed=34521,
        num_workers=0,
        generate_doc_lengths=True,
    ).build(instance_source, collator=DataCollator(pad_token_id=EOS_ID))

    loader.reshuffle()
    for batch in loader:
        input_ids = batch["input_ids"]
        label_mask = batch["label_mask"]
        doc_lens = batch["doc_lens"]
        for i in range(input_ids.shape[0]):
            yield input_ids[i], label_mask[i], doc_lens[i]


def _pad_count(input_ids: torch.Tensor) -> int:
    """Trailing run of pad tokens (pad==EOS==151643), minus the one EOS the last real doc owns."""
    trailing = 0
    for t in reversed(input_ids.tolist()):
        if t == EOS_ID:
            trailing += 1
        else:
            break
    return max(0, trailing - 1)


def _real_doc_count(input_ids: torch.Tensor, doc_lens: torch.Tensor, seq_len: int) -> int:
    """Number of doc_lens segments lying in the real-token region (excludes pad pseudo-docs)."""
    real_len = seq_len - _pad_count(input_ids)
    bounds = torch.cumsum(doc_lens[doc_lens != 0], 0).tolist()
    return sum(1 for b in bounds if b <= real_len)


def _check(input_ids: torch.Tensor, label_mask: torch.Tensor, doc_lens: torch.Tensor, seq_len: int):
    """Returns (pad_count, all_masked, n_truncated) for one packed window."""
    lens = doc_lens[doc_lens != 0]
    assert int(lens.sum()) == seq_len, f"doc_lens sum {int(lens.sum())} != seq_len {seq_len}"

    # INVARIANT (correct masking): every internal boundary lands immediately after an EOS token, so
    # no attention block ever straddles a document boundary. (With the buggy bos==eos tokenizer this
    # is trivially true only because there are almost no internal boundaries -- see docs/window below.)
    bounds = torch.cumsum(lens, 0).tolist()
    for b in bounds[:-1]:
        assert int(input_ids[b - 1]) == EOS_ID, f"boundary at {b} not preceded by EOS"

    # Trailing run of pad tokens (pad==EOS==151643): the last real doc contributes exactly one EOS,
    # the rest of the run is padding.
    pad_count = _pad_count(input_ids)
    real_len = seq_len - pad_count

    # INVARIANT (documents kept whole): every segment in the real region either ends in EOS (a whole
    # document) or is a single over-long document truncated to fill the window (LongDocStrategy
    # .truncate) -- never a fragment continued in another window. Count the truncated ones.
    n_truncated = 0
    prev = 0
    for b in bounds:
        if b > real_len:
            break
        seg_end_tok = int(input_ids[b - 1])
        if seg_end_tok != EOS_ID:
            # only legal if this single segment fills the whole window (a truncated over-long doc)
            assert (
                prev == 0 and b == seq_len
            ), f"mid-window segment [{prev}:{b}] does not end in EOS"
            n_truncated += 1
        prev = b

    all_masked = not bool(label_mask[:real_len].any())
    return pad_count, all_masked, n_truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--data-root", type=str, default=None, help="real shards (5 task subdirs)")
    ap.add_argument("--n-docs", type=int, default=60, help="synthetic docs per task")
    args = ap.parse_args()

    tmp = tempfile.TemporaryDirectory()
    work_dir = Path(tmp.name) / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.data_root is not None:
        data_root = Path(args.data_root)
        print(f"Using real shards under {data_root}")
    else:
        data_root = Path(tmp.name) / "data"
        rng = np.random.default_rng(0)
        for task in TASKS:
            _write_synthetic_task(
                data_root / task, n_docs=args.n_docs, seq_len=args.seq_len, rng=rng
            )
        print(f"Wrote synthetic 5-task data ({args.n_docs} docs/task) under {data_root}")

    specs = _build_specs(data_root)

    # ---- Fixed path (bos=None): the real training configuration ----
    print(f"\n=== FIXED (bos=None), seq_len={args.seq_len}, truncate ===")
    n, total_pad, n_all_masked, n_trunc, real_docs = 0, 0, 0, 0, []
    for input_ids, label_mask, doc_lens in _iter_instances(
        specs, seq_len=args.seq_len, work_dir=work_dir, bos_token_id=None
    ):
        pad_count, all_masked, n_truncated = _check(input_ids, label_mask, doc_lens, args.seq_len)
        n += 1
        total_pad += pad_count
        n_all_masked += int(all_masked)
        n_trunc += n_truncated
        real_docs.append(_real_doc_count(input_ids, doc_lens, args.seq_len))

    print(f"  instances checked            : {n}")
    print("  doc_lens.sum()==seq_len      : OK (all)")
    print("  boundaries land on EOS       : OK (all)")
    print("  documents kept whole         : OK (all)")
    print(f"  avg real docs/window         : {np.mean(real_docs):.2f}")
    print(f"  truncated over-long docs     : {n_trunc}")
    print(f"  avg padding fraction         : {total_pad / (n * args.seq_len):.1%}")
    print(
        f"  all-masked (NaN-loss) windows: {n_all_masked}"
        + (
            "  <-- WARNING: over-long prompt-only docs got truncated"
            if n_all_masked
            else "  (none)"
        )
    )

    # ---- Negative control (bos==eos==151643): the bug ----
    work_dir2 = Path(tmp.name) / "work2"
    work_dir2.mkdir(parents=True, exist_ok=True)
    buggy_real_docs = []
    for input_ids, label_mask, doc_lens in _iter_instances(
        specs, seq_len=args.seq_len, work_dir=work_dir2, bos_token_id=EOS_ID
    ):
        buggy_real_docs.append(_real_doc_count(input_ids, doc_lens, args.seq_len))
    print("\n=== NEGATIVE CONTROL (buggy bos==eos) ===")
    print(
        f"  avg real docs/window         : {np.mean(buggy_real_docs):.2f}"
        f"  (vs {np.mean(real_docs):.2f} fixed)"
    )
    print(
        "  -> with bos==eos the real-token region collapses toward a SINGLE segment: multiple "
        "documents (and tasks) share one attention block and attend across each other. The bos=None "
        "fix restores one block per document."
    )

    tmp.cleanup()
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
