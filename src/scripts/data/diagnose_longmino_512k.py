"""
Diagnose a built longmino-512k tokenized tree when training on it produces an implausible loss.

Perplexity ~1.0 means the cross-entropy is ~0, i.e. the next token is essentially always already
known. Only a few things do that, and this script checks each of them directly against the bytes
on disk:

1. **Mix sizing / epoch count** -- :func:`calculate_sample_sizes` shrinks the *whole* mix to what
   the scarcest stratum supports (``adjustment_factor``), and the mix is three levels deep, so one
   undersized leaf collapses everything. A mix far smaller than ``max_duration`` means the run is
   doing many epochs over the same instances, which memorizes and drives loss to ~0.
2. **Degenerate token content** -- all-zero / single-token / long-run arrays, which a pretrained
   model predicts perfectly. Reported as distinct-token fraction, top-1 token share and longest
   identical run over random windows.
3. **Wrong tokenizer for the tree** -- the qwen3 and qwen35 trees are tokenized by separate
   invocations, so a swapped ``--tokenizer`` is possible. Checked via the observed max token id
   against the tokenizer config's vocab size, and by decoding a window.
4. **Document separator** -- the writer emits a single EOS per document, but ``TokenizerConfig``
   sets ``bos_token_id == eos_token_id`` for both Qwen families, and
   :func:`~olmo_core.data.utils.iter_document_indices` then only counts a boundary where *two*
   EOS tokens are adjacent. Reports EOS density and adjacent-EOS pairs so the effective document
   segmentation is visible.
5. **Duplicated content** -- the pool buckets are pulled raw and are not deduplicated against
   themselves. Reports the repeat rate of hashed fixed-size token windows within each stratum.

Run (via gantry, from the repo root)::

    gantry run --workspace ai2/flex2 --budget ai2/oe-other \\
        --cluster ai2/jupiter-cirrascale-2 \\
        --weka oe-training-default:/weka/oe-training-default \\
        --cpus 8 --gpus 0 --priority urgent \\
        --python-manager conda --system-python \\
        --install "pip install 'transformers>=4.40' numpy" \\
        --yes \\
        -- python src/scripts/data/diagnose_longmino_512k.py --tree qwen35
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import zlib
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from longmino_512k_common import WEKA_ROOT  # noqa: E402
from longmino_512k_mix import (  # noqa: E402
    STRATUM_GLOBS,
    measured_available,
    nominal_ratios,
    plan_tokens,
)

DTYPE = np.uint32
ITEMSIZE = 4

#: Tokenizer identity per tree: (HF id, vocab size, ``<|endoftext|>`` id) from TokenizerConfig.
TREE_TOKENIZER = {
    "qwen3": ("Qwen/Qwen3-0.6B", 151936, 151643),
    "qwen35": ("Qwen/Qwen3.5-0.8B", 248320, 248044),
}


def stratum_parts(base: str, stratum_glob: str) -> List[str]:
    return sorted(glob.glob(os.path.join(base, stratum_glob), recursive=True))


def on_disk_tokens(parts: List[str]) -> int:
    return sum(os.path.getsize(p) for p in parts) // ITEMSIZE


# --------------------------------------------------------------------------------------- sizing


def report_sizing(base: str, root: str, tree: str, max_duration: int, sequence_length: int) -> None:
    """
    Compare per-stratum tokens actually on disk against what the mix ratios ask for, and report how
    many epochs the run would do.

    The mix is sized by ``adjustment_factor = min(available_s / ideal_s)`` over strata, so the
    binding stratum -- the one with the smallest ``available/ratio`` -- sets the size of the entire
    mix. That stratum is called out explicitly.
    """
    print("\n=== 0. tokenization completeness ===")
    # token_counts.json is only written after the *last* stratum finishes, so its absence means the
    # tokenizer job never completed. That matters more than it looks: build_longmino_512k_mix falls
    # back to PUBLISHED_AVAILABLE when it is missing, so the mix is planned against the dataset
    # card's token counts while the tree on disk holds far fewer -- and calculate_sample_sizes then
    # shrinks the whole mix to fit what is really there.
    counts_path = os.path.join(base, "token_counts.json")
    progress_path = os.path.join(base, "progress.json")
    if not os.path.exists(counts_path):
        print(f"  !! {counts_path} MISSING -- the tokenizer job never finished.")
        print("     build_longmino_512k_mix is therefore planning against PUBLISHED_AVAILABLE.")
    else:
        print(f"  {counts_path} present")
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            prog = json.load(f)
        print(
            f"  progress.json: tokenizer={prog.get('tokenizer')!r} "
            f"eos={prog.get('eos_token_id')} dtype={prog.get('dtype')} "
            f"{len(prog.get('processed', [])):,} shards processed"
        )
        expected_hf, _, expected_eos = TREE_TOKENIZER[tree]
        if prog.get("tokenizer") != expected_hf:
            print(
                f"  !! TOKENIZER MISMATCH: tree '{tree}' was written with "
                f"{prog.get('tokenizer')!r}, expected {expected_hf!r}"
            )
        if prog.get("eos_token_id") != expected_eos:
            print(
                f"  !! EOS MISMATCH: recorded {prog.get('eos_token_id')}, "
                f"TokenizerConfig expects {expected_eos}"
            )
    else:
        print(f"  !! {progress_path} missing")

    print("\n=== 1. mix sizing ===")
    disk: Dict[str, int] = {}
    for label, g in STRATUM_GLOBS.items():
        parts = stratum_parts(base, g)
        disk[label] = on_disk_tokens(parts)
        print(f"  {label:18s} {len(parts):4d} parts  {disk[label] / 1e9:8.3f}B tok on disk")

    recorded = measured_available(root, tree)
    for label in STRATUM_GLOBS:
        rec, dsk = recorded.get(label, 0), disk[label]
        if rec != dsk:
            print(f"  MISMATCH {label}: token_counts.json {rec:,} vs on-disk {dsk:,}")

    empty = [k for k, v in disk.items() if v == 0]
    if empty:
        print(f"\n  !! EMPTY STRATA (glob matched nothing): {empty}")

    nominal = nominal_ratios()
    print(f"\n  {'stratum':18s} {'ratio':>8s} {'available':>11s} {'implied mix':>13s}")
    caps: Dict[str, float] = {}
    for label, ratio in nominal.items():
        avail = disk.get(label, 0)
        if avail == 0:
            print(f"  {label:18s} {ratio:8.4f} {0.0:10.3f}B  {'-- EMPTY --':>13s}")
            continue
        caps[label] = avail / ratio
        print(f"  {label:18s} {ratio:8.4f} {avail / 1e9:10.3f}B {caps[label] / 1e9:12.3f}B")

    if not caps:
        print("  no usable strata at all")
        return

    binder = min(caps, key=lambda k: caps[k])
    target = plan_tokens(disk)
    total = sum(target.values())
    print(f"\n  binding stratum: {binder} -> mix capped at {caps[binder] / 1e9:.3f}B tokens")
    print(f"  planned mix total (with FULL_USE top-ups): {total / 1e9:.3f}B tokens")
    instances = int(total // sequence_length)
    print(f"  ~{instances:,} instances at sequence_length={sequence_length:,}")
    if total > 0:
        epochs = max_duration / total
        print(
            f"  max_duration={max_duration / 1e9:.1f}B tokens -> {epochs:.2f} epochs over the mix"
        )
        if epochs > 2:
            print(
                "  !! MANY EPOCHS: the run repeats the same instances; a pretrained model "
                "memorizes them and loss collapses toward 0 (ppl -> 1.0)."
            )


# ------------------------------------------------------------------------------------- content


def sample_windows(
    parts: List[str], n_windows: int, window: int, rng: np.random.Generator
) -> List[np.ndarray]:
    """Read ``n_windows`` random contiguous token windows spread over a stratum's part files."""
    out = []
    usable = [(p, os.path.getsize(p) // ITEMSIZE) for p in parts]
    usable = [(p, n) for p, n in usable if n > window]
    if not usable:
        return out
    for _ in range(n_windows):
        p, n = usable[int(rng.integers(len(usable)))]
        start = int(rng.integers(0, n - window))
        mm = np.memmap(p, mode="r", dtype=DTYPE, offset=start * ITEMSIZE, shape=(window,))
        out.append(np.asarray(mm))
    return out


def longest_run(a: np.ndarray) -> int:
    if a.size == 0:
        return 0
    change = np.flatnonzero(a[1:] != a[:-1])
    bounds = np.concatenate(([-1], change, [a.size - 1]))
    return int(np.diff(bounds).max())


def report_content(
    base: str, tree: str, n_windows: int, window: int, seed: int, decode: bool
) -> None:
    """
    Per-stratum degeneracy statistics over random windows, plus optional decoded previews.

    ``distinct`` is the fraction of the window that is unique token ids; ``top1`` the share held by
    the single most common id; ``maxrun`` the longest stretch of one repeated id. Healthy prose sits
    around distinct 0.3-0.6, top1 < 0.1, maxrun < 10. Anything near distinct ~0 / top1 ~1 is
    trivially predictable and would on its own explain ppl ~1.0.
    """
    print("\n=== 2/3. token content ===")
    hf_id, vocab_size, eos_id = TREE_TOKENIZER[tree]
    tok = None
    if decode:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(hf_id)
        except Exception as e:  # noqa: BLE001
            print(f"  (decode disabled: {e})")

    rng = np.random.default_rng(seed)
    print(f"  expecting vocab_size={vocab_size:,} (eos {eos_id}) for tree '{tree}'\n")
    print(
        f"  {'stratum':18s} {'maxid':>8s} {'distinct':>9s} {'top1':>7s} {'maxrun':>7s} "
        f"{'zeros':>7s} {'bits/tok':>9s}"
    )
    for label, g in STRATUM_GLOBS.items():
        parts = stratum_parts(base, g)
        wins = sample_windows(parts, n_windows, window, rng)
        if not wins:
            print(f"  {label:18s} {'-- no data --':>8s}")
            continue
        cat = np.concatenate(wins)
        vals, counts = np.unique(cat, return_counts=True)
        distinct = vals.size / cat.size
        top1 = counts.max() / cat.size
        maxrun = max(longest_run(w) for w in wins)
        zeros = float((cat == 0).mean())
        # gzip bits per token: a cheap upper bound on the entropy the model has to model. Raw text
        # of this kind sits near 8-9 bits/token; anything far below that is intrinsically
        # low-entropy content that will pull training loss down on its own.
        bits = (
            8
            * len(zlib.compress(np.concatenate(wins[:20]).tobytes(), 9))
            / (sum(w.size for w in wins[:20]))
        )
        flag = ""
        if int(vals.max()) >= vocab_size:
            flag += "  !! TOKEN ID >= VOCAB (wrong tokenizer for this tree)"
        if int(vals.max()) < 65536:
            flag += "  !! all ids < 65536 (suspiciously small vocab)"
        if distinct < 0.02 or top1 > 0.5 or maxrun > 1000:
            flag += "  !! DEGENERATE"
        print(
            f"  {label:18s} {int(vals.max()):8d} {distinct:9.3f} {top1:7.3f} "
            f"{maxrun:7d} {zeros:7.3f} {bits:9.2f}{flag}"
        )

    if tok is not None:
        print("\n  --- decoded previews (first 300 chars of one window per stratum) ---")
        rng = np.random.default_rng(seed + 1)
        for label, g in STRATUM_GLOBS.items():
            wins = sample_windows(stratum_parts(base, g), 1, min(window, 512), rng)
            if not wins:
                continue
            text = tok.decode(wins[0].astype(np.int64).tolist())
            print(f"\n  [{label}] {text[:300]!r}")


# ------------------------------------------------------------------------------------ separator


def report_separator(base: str, tree: str, n_files: int, seed: int) -> None:
    """
    EOS density and adjacent-EOS pairs.

    ``iter_document_indices`` is called with ``bos_token_id`` set (both Qwen configs set
    ``bos == eos``), so it only yields a boundary where two EOS tokens are *adjacent*. The writer
    emits one EOS per document and never prepends a BOS, so ``adjacent`` is expected to be ~0 --
    meaning each part file is treated as a single document. That does not by itself break
    concat-and-chunk training, but it does mean no intra-document masking and it makes any
    ``max_document_length`` logic meaningless.
    """
    print("\n=== 4. document separator ===")
    _, _, eos_id = TREE_TOKENIZER[tree]
    rng = np.random.default_rng(seed)
    print(f"  eos id {eos_id}")
    for label, g in STRATUM_GLOBS.items():
        parts = stratum_parts(base, g)
        if not parts:
            continue
        picks = [
            parts[int(i)]
            for i in rng.choice(len(parts), size=min(n_files, len(parts)), replace=False)
        ]
        eos_total = adj_total = tok_total = 0
        for p in picks:
            mm = np.memmap(p, mode="r", dtype=DTYPE)
            head = np.asarray(mm[:20_000_000])
            eos = head == eos_id
            eos_total += int(eos.sum())
            adj_total += int(np.logical_and(eos[:-1], eos[1:]).sum())
            tok_total += head.size
        mean_doc = tok_total / eos_total if eos_total else float("inf")
        print(
            f"  {label:18s} {eos_total:9,d} eos / {tok_total / 1e6:7.1f}M tok "
            f"-> mean doc {mean_doc:11,.0f} tok; adjacent-eos pairs {adj_total}"
        )


# ------------------------------------------------------------------------------------ duplicates


def report_duplicates(base: str, n_windows: int, window: int, seed: int) -> None:
    """Repeat rate of hashed fixed-size token windows sampled within each stratum."""
    print("\n=== 5. intra-stratum duplication ===")
    rng = np.random.default_rng(seed)
    for label, g in STRATUM_GLOBS.items():
        parts = stratum_parts(base, g)
        wins = sample_windows(parts, n_windows, window, rng)
        if not wins:
            continue
        seen: Dict[str, int] = {}
        for w in wins:
            h = hashlib.sha1(w.tobytes()).hexdigest()
            seen[h] = seen.get(h, 0) + 1
        dupes = sum(c - 1 for c in seen.values())
        print(
            f"  {label:18s} {len(wins)} windows of {window} tok -> "
            f"{len(seen)} distinct, {dupes} repeats ({100 * dupes / len(wins):.1f}%)"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=WEKA_ROOT)
    parser.add_argument("--tree", default="qwen35", choices=["qwen3", "qwen35"])
    parser.add_argument("--sequence-length", type=int, default=262144)
    parser.add_argument("--max-duration", type=int, default=10_000_000_000)
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--files-per-stratum", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-decode", action="store_true")
    args = parser.parse_args()

    base = os.path.join(args.root, args.tree)
    print(f"diagnosing {base}")
    if not os.path.isdir(base):
        raise SystemExit(f"{base} does not exist")

    report_sizing(base, args.root, args.tree, args.max_duration, args.sequence_length)
    report_content(base, args.tree, args.windows, args.window_size, args.seed, not args.no_decode)
    report_separator(base, args.tree, args.files_per_stratum, args.seed)
    report_duplicates(base, args.windows, args.window_size, args.seed)


if __name__ == "__main__":
    main()
