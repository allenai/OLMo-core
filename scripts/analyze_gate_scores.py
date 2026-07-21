#!/usr/bin/env python3
"""
Analyze landmark gate-score logs (olmo_core.nn.attention.landmark_gate_analysis) and plot how peaky
the gate-score distribution is.

Reads the JSONL gate logs written by an eval run with OLMO_LANDMARK_GATE_LOG + OLMO_GATE_LOG_ALL set
(each worker writes a "<base>.rank<N>" file). Every record is one decoded token; under
``layers.<layer>.<head>`` it carries:

    all_scores : every candidate landmark block's gate score (raw q.k logit), descending by score
    blocks     : the top-k blocks hard top-k retrieval KEPT (so k = len(blocks) is the selection cutoff)

Because the kept blocks are exactly the top-k of ``all_scores``, the selected gates occupy ranks
``0..k-1`` of the descending ``all_scores`` and the cutoff sits at rank ``k``. From each
(layer, head, decode-step) sample we derive three rank-aligned curves and aggregate them across
samples (mean + inter-quartile band), producing the three requested plots:

    (1) softmax over ALL gates      -- the attention mass each block would get if every candidate
        competed (softmax over all_scores); a vertical dotted line marks the selection cutoff (rank k).
    (2) softmax over SELECTED gates -- the mass across only the kept blocks (softmax over the top-k
        scores), i.e. what the current gating actually allocates among retrieved blocks.
    (3) raw pre-softmax scores      -- the landmark logits themselves, descending, with the same cutoff.

The gap between (1) and (2) is the effect of the hard cutoff + renormalization; how much of (1)'s mass
already sits left of the cutoff line (printed as "mass captured by selected") says how much the
gate weighting concentrates on the retrieved blocks.

Usage::

    python scripts/analyze_gate_scores.py '/weka/.../gate_scores/<label>/ruler/gate.ruler8k.rank*' \
        --out gate_ruler8k.png

    # filter to one layer/head, or a context length, and cap samples for speed:
    python scripts/analyze_gate_scores.py 'gate.*' --layer 12 --head 0 --context-len 8192 \
        --max-records 20000 --out fig.png

Notes:
- Pass one or more path globs. Non-glob paths are used directly.
- Pool with care: the selection count k grows with context (fraction-of-blocks retrieval), so the
  cutoff line is the MEDIAN k over the sampled records -- filter with --context-len for a crisp line.
"""
import argparse
import glob
import json
import os
import sys
from typing import List, Optional

import numpy as np

# Headless backend so this runs on a login/CPU node without a display.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _iter_paths(patterns: List[str]):
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            yield from sorted(hits)
        elif os.path.exists(pat):
            yield pat
        else:
            print(f"WARNING: no files match {pat!r}", file=sys.stderr)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def load_samples(
    patterns: List[str],
    *,
    layer: Optional[int],
    head: Optional[int],
    context_len: Optional[int],
    dataset: Optional[str],
    subtask: Optional[str],
    max_records: Optional[int],
    rng: np.random.Generator,
):
    """Return (all_scores_list, k_list, meta). Each list element is one (layer, head, step) sample:
    ``all_scores_list[i]`` a descending np.ndarray of every candidate gate's raw score, ``k_list[i]``
    the number selected (the cutoff rank)."""
    all_scores: List[np.ndarray] = []
    ks: List[int] = []
    n_records = 0
    n_missing_all = 0
    layers_seen: set = set()
    ctx_seen: set = set()
    layer_key = f"layer{layer}" if layer is not None else None
    head_key = f"head{head}" if head is not None else None

    for path in _iter_paths(patterns):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if dataset is not None and rec.get("dataset") != dataset:
                    continue
                if subtask is not None and rec.get("subtask") != subtask:
                    continue
                if context_len is not None and int(rec.get("context_len", -1)) != context_len:
                    continue
                ctx_seen.add(rec.get("context_len"))
                n_records += 1
                layers = rec.get("layers", {})
                for lname, heads in layers.items():
                    if layer_key is not None and lname != layer_key:
                        continue
                    layers_seen.add(lname)
                    for hname, entry in heads.items():
                        if head_key is not None and hname != head_key:
                            continue
                        a = entry.get("all_scores")
                        if a is None:
                            # No OLMO_GATE_LOG_ALL: only the kept scores were logged. Fall back to
                            # them (can't show the non-selected tail); flag it.
                            a = entry.get("scores", [])
                            n_missing_all += 1
                        if len(a) == 0:
                            continue
                        arr = np.asarray(a, dtype=np.float64)
                        # all_scores is already descending, but sort defensively.
                        arr = np.sort(arr)[::-1]
                        k = len(entry.get("blocks", []))
                        k = max(1, min(k, len(arr)))
                        all_scores.append(arr)
                        ks.append(k)

    # Optional subsample of (layer, head, step) samples for speed / memory.
    if max_records is not None and len(all_scores) > max_records:
        idx = rng.choice(len(all_scores), size=max_records, replace=False)
        all_scores = [all_scores[i] for i in idx]
        ks = [ks[i] for i in idx]

    meta = dict(
        n_records=n_records,
        n_samples=len(all_scores),
        n_missing_all=n_missing_all,
        layers_seen=sorted(layers_seen),
        ctx_seen=sorted(c for c in ctx_seen if c is not None),
    )
    return all_scores, ks, meta


def _rank_matrix(rows: List[np.ndarray], max_rank: int) -> np.ndarray:
    """Stack ragged descending rows into an (n_samples x max_rank) array, nan-padding short rows."""
    m = np.full((len(rows), max_rank), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        w = min(len(r), max_rank)
        m[i, :w] = r[:w]
    return m


def _agg(mat: np.ndarray):
    """Per-rank mean and inter-quartile band, ignoring nan padding."""
    with np.errstate(all="ignore"):
        mean = np.nanmean(mat, axis=0)
        p25 = np.nanpercentile(mat, 25, axis=0)
        p75 = np.nanpercentile(mat, 75, axis=0)
    return mean, p25, p75


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="gate-log JSONL path(s) or glob(s) (e.g. 'gate.ruler8k.rank*').")
    ap.add_argument("--out", default="gate_scores.png", help="output figure path (PNG).")
    ap.add_argument("--layer", type=int, default=None, help="restrict to one layer index.")
    ap.add_argument("--head", type=int, default=None, help="restrict to one head index.")
    ap.add_argument("--context-len", type=int, default=None,
                    help="restrict to one context_len (keeps the cutoff line crisp).")
    ap.add_argument("--dataset", default=None, help="restrict to one 'dataset' field.")
    ap.add_argument("--subtask", default=None, help="restrict to one 'subtask' field.")
    ap.add_argument("--max-records", type=int, default=50000,
                    help="cap the number of (layer,head,step) samples aggregated (subsampled).")
    ap.add_argument("--max-rank", type=int, default=None,
                    help="x-axis cap (gate rank). Default: the 99th percentile of candidate counts.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default=None, help="figure suptitle (default: derived from filters).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    all_scores, ks, meta = load_samples(
        args.paths, layer=args.layer, head=args.head, context_len=args.context_len,
        dataset=args.dataset, subtask=args.subtask, max_records=args.max_records, rng=rng,
    )
    if not all_scores:
        print("ERROR: no gate samples matched the given paths/filters.", file=sys.stderr)
        sys.exit(2)

    ns = np.array([len(a) for a in all_scores])
    ks_arr = np.array(ks)
    if args.max_rank is not None:
        max_rank = args.max_rank
    else:
        max_rank = int(np.percentile(ns, 99))
    max_rank = max(2, max_rank)
    median_k = int(np.median(ks_arr))

    # Per-sample curves.
    softmax_all_rows = [_softmax(a) for a in all_scores]                 # (1) over all gates
    softmax_sel_rows = [_softmax(a[:k]) for a, k in zip(all_scores, ks)]  # (2) over selected gates
    raw_rows = all_scores                                                # (3) raw scores

    A = _rank_matrix(softmax_all_rows, max_rank)
    S = _rank_matrix(softmax_sel_rows, min(max_rank, int(ks_arr.max())))
    R = _rank_matrix(raw_rows, max_rank)
    a_mean, a_lo, a_hi = _agg(A)
    s_mean, s_lo, s_hi = _agg(S)
    r_mean, r_lo, r_hi = _agg(R)

    # Summary stats.
    captured = np.array([np.sum(_softmax(a)[:k]) for a, k in zip(all_scores, ks)])  # mass left of cutoff
    top1_all = np.array([_softmax(a)[0] for a in all_scores])
    entropy_all = np.array([-np.sum((p := _softmax(a)) * np.log(p + 1e-12)) for a in all_scores])
    entropy_sel = np.array(
        [-np.sum((p := _softmax(a[:k])) * np.log(p + 1e-12)) for a, k in zip(all_scores, ks)]
    )

    def stat(x):
        return f"mean={x.mean():.4f}  median={np.median(x):.4f}  p10={np.percentile(x,10):.4f}  p90={np.percentile(x,90):.4f}"

    print(f"records matched         : {meta['n_records']}")
    print(f"(layer,head,step) samples: {meta['n_samples']}  (aggregated: {len(all_scores)})")
    if meta["n_missing_all"]:
        print(f"  NOTE: {meta['n_missing_all']} samples lacked all_scores (no OLMO_GATE_LOG_ALL) -> "
              f"non-selected tail missing for those.")
    print(f"context_lens present    : {meta['ctx_seen']}")
    print(f"layers present          : {meta['layers_seen']}")
    print(f"# candidate gates (n)   : {stat(ns.astype(float))}")
    print(f"# selected gates (k)    : {stat(ks_arr.astype(float))}   -> cutoff line at median k={median_k}")
    print(f"top-1 gate mass (all)   : {stat(top1_all)}")
    print(f"mass captured by selected: {stat(captured)}   (sum of softmax-over-all mass at ranks < k)")
    print(f"entropy softmax(all)    : {stat(entropy_all)}  (nats; 0=one-hot, log(n)=uniform)")
    print(f"entropy softmax(selected): {stat(entropy_sel)}")

    # ---- plots ----
    C_ALL, C_SEL, C_RAW, C_CUT = "#2b6cb0", "#c05621", "#2f855a", "#718096"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    def _cut(ax, x):
        ax.axvline(x, ls=":", lw=1.6, color=C_CUT, label=f"selection cutoff (median k={x})")

    # (1) softmax over all gates
    ax = axes[0]
    x = np.arange(A.shape[1])
    ax.fill_between(x, a_lo, a_hi, color=C_ALL, alpha=0.18, linewidth=0)
    ax.plot(x, a_mean, color=C_ALL, lw=2, label="mean")
    _cut(ax, median_k)
    ax.set_yscale("log")
    ax.set_title("(1) softmax over ALL gates")
    ax.set_xlabel("gate rank (desc by score)")
    ax.set_ylabel("attention mass (log)")
    ax.legend(fontsize=8)

    # (2) softmax over selected gates
    ax = axes[1]
    xs = np.arange(S.shape[1])
    ax.fill_between(xs, s_lo, s_hi, color=C_SEL, alpha=0.18, linewidth=0)
    ax.plot(xs, s_mean, color=C_SEL, lw=2, label="mean")
    ax.set_yscale("log")
    ax.set_title("(2) softmax over SELECTED gates (top-k)")
    ax.set_xlabel("selected-gate rank (0..k-1)")
    ax.set_ylabel("attention mass (log)")
    ax.legend(fontsize=8)

    # (3) raw pre-softmax scores over all gates
    ax = axes[2]
    ax.fill_between(x, r_lo, r_hi, color=C_RAW, alpha=0.18, linewidth=0)
    ax.plot(x, r_mean, color=C_RAW, lw=2, label="mean")
    _cut(ax, median_k)
    ax.set_title("(3) raw scores over ALL gates")
    ax.set_xlabel("gate rank (desc by score)")
    ax.set_ylabel("landmark logit (q·k·scale)")
    ax.legend(fontsize=8)

    filt = []
    if args.dataset:
        filt.append(f"dataset={args.dataset}")
    if args.context_len:
        filt.append(f"ctx={args.context_len}")
    if args.layer is not None:
        filt.append(f"layer={args.layer}")
    if args.head is not None:
        filt.append(f"head={args.head}")
    sup = args.title or ("landmark gate scores" + (f"  [{', '.join(filt)}]" if filt else ""))
    fig.suptitle(f"{sup}   (n={len(all_scores)} head/layer/step samples; band = IQR)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote figure -> {args.out}")


if __name__ == "__main__":
    main()
