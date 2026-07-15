#!/usr/bin/env python3
"""Reconstructs the Q1-Q5 landmark gate-similarity (Jaccard) figures from compact gate-set dumps.

Landmark attention keeps, per (layer, head, decode step), a hard top-k *set* of landmark blocks. These
figures ask how similar those opened-gate sets are along different axes, using Jaccard similarity
``|A n B| / |A u B|``:

  * **Q1** gate similarity across LAYERS (same head), vs layer gap        -> ``q1_layers.png``
           + head-pooled layer x layer Jaccard matrix                     -> ``q1_layer_matrix.png``
  * **Q2** gate similarity across DECODED TOKENS (same example), vs gap   -> ``q2_tokens.png``
  * **Q3/Q4** cross-example (positional bias) vs cross-model, per subtask -> ``q3q4_example_model.png``
  * **Q5** within-layer agreement across HEADS (first/middle/final layer) -> ``q5_cross_head.png``
           + head x head Jaccard matrix within a layer (64k)              -> ``q5_head_matrix.png``

Input is the compact per-record dump produced by ``extract_gate_sets.py`` (one JSONL line per decoded
token: ``{"len","doc","sub","tok","n","g":{layer:{head:[blocks]}}}``). This script never touches the
raw ~TB logs -- do the extraction on weka, plot here.

The original figures compared ``compressive`` vs ``fast`` (both base). This version is generalized to
compare any two labeled models; for the pre/post-SFT study pass the compressive base dump as model A
and the compressive SFT dump as model B.

The ``random`` baseline (dotted) is the exact expected Jaccard of two independent uniform ``k``-subsets
of ``n`` candidate blocks (hypergeometric overlap), with ``n``/``k`` read from the data per length --
see ``../landmark_gate_jaccard_baseline.py`` for the derivation.

Usage (dumps are named ``<label>_ruler_<K>k.jsonl``)::

    python plot_gate_jaccard.py \
        --a-label pre-SFT  --a-dumps 'dumps/q4b-base-fastcomplm-s2385_ruler_*.jsonl' \
        --b-label post-SFT --b-dumps 'dumps/q4b-comp-5task-s8550_ruler_*.jsonl' \
        --lengths 8192 16384 32768 65536 --outdir plots_pre_post
"""
import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict
from math import comb
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

A_COLOR = "tab:blue"
B_COLOR = "tab:orange"


# --------------------------------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------------------------------
class Record:
    __slots__ = ("length", "doc", "sub", "tok", "n", "gates")

    def __init__(self, d):
        self.length = int(d["len"])
        self.doc = int(d["doc"])
        self.sub = d.get("sub", "") or ""
        self.tok = int(d["tok"])
        self.n = int(d["n"]) if "n" in d else None
        # gates[layer_int][head_int] = frozenset(block ids)
        self.gates: Dict[int, Dict[int, frozenset]] = {
            int(l): {int(h): frozenset(bs) for h, bs in heads.items()}
            for l, heads in d["g"].items()
        }


def load_dumps(patterns) -> Dict[int, List[Record]]:
    """Load compact dumps, grouping records by context length."""
    by_len: Dict[int, List[Record]] = defaultdict(list)
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else []))
    if not files:
        print(f"WARNING: no dump files match {patterns}", file=sys.stderr)
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = Record(json.loads(line))
                by_len[r.length].append(r)
    return by_len


def jac(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 1.0


def exact_jaccard(N: int, k: int) -> float:
    """Exact E[Jaccard] of two independent uniform ``k``-subsets of ``{0..N-1}`` (hypergeometric)."""
    if k <= 0 or N <= 0:
        return float("nan")
    if k >= N:
        return 1.0
    denom = comb(N, k)
    total = 0.0
    for c in range(max(1, 2 * k - N), k + 1):
        total += comb(k, c) * comb(N - k, k - c) / denom * c / (2 * k - c)
    return total


def baseline_for(records: List[Record]) -> float:
    """Chance Jaccard for a length: ``n`` = median candidate blocks, ``k`` = median kept-set size."""
    if not records:
        return float("nan")
    ns, ks = [], []
    for r in records:
        if r.n is not None:
            ns.append(r.n)
        for heads in r.gates.values():
            for s in heads.values():
                ks.append(len(s))
    if not ks:
        return float("nan")
    k = int(round(float(np.median(ks))))
    if ns:
        N = int(round(float(np.median(ns))))
    else:
        N = max(k + 1, records[0].length // 64)  # one landmark per 64-token block
    return exact_jaccard(N, k)


def subsample(records: List[Record], cap: int, seed: int = 0) -> List[Record]:
    if cap is None or len(records) <= cap:
        return records
    rng = random.Random(seed)
    return rng.sample(records, cap)


def lighten(color, amount):
    """amount in [0,1]: 0 = original, 1 = white."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def len_label(L):
    return f"{L // 1024}k"


def norm_sub(sub):
    """``ruler_cwe__8192`` -> ``cwe`` (strip the ``ruler_`` prefix and ``__<len>`` suffix)."""
    if not sub:
        return sub
    s = sub[len("ruler_"):] if sub.startswith("ruler_") else sub
    i = s.rfind("__")
    return s[:i] if i > 0 else s


# --------------------------------------------------------------------------------------------------
# Q1 -- cross-layer, same head
# --------------------------------------------------------------------------------------------------
def q1_gap_curve(records, cap):
    sums, cnts = defaultdict(float), defaultdict(int)
    for r in subsample(records, cap):
        head_layers = defaultdict(list)  # head -> [(layer, set)]
        for l, heads in r.gates.items():
            for h, s in heads.items():
                head_layers[h].append((l, s))
        for lst in head_layers.values():
            lst.sort()
            for i in range(len(lst)):
                li, si = lst[i]
                for j in range(i + 1, len(lst)):
                    lj, sj = lst[j]
                    gap = lj - li
                    sums[gap] += jac(si, sj)
                    cnts[gap] += 1
    return {g: sums[g] / cnts[g] for g in sorted(sums)}


def q1_matrix(records, n_layers, cap):
    M = np.zeros((n_layers, n_layers))
    C = np.zeros((n_layers, n_layers))
    for r in subsample(records, cap):
        head_layers = defaultdict(list)
        for l, heads in r.gates.items():
            for h, s in heads.items():
                head_layers[h].append((l, s))
        for lst in head_layers.values():
            for li, si in lst:
                for lj, sj in lst:
                    M[li, lj] += jac(si, sj)
                    C[li, lj] += 1
    with np.errstate(invalid="ignore"):
        return M / np.where(C == 0, np.nan, C)


# --------------------------------------------------------------------------------------------------
# Q2 -- cross decoded-token, same example
# --------------------------------------------------------------------------------------------------
def q2_token_curve(records, cap, max_gap=7):
    groups = defaultdict(list)  # (doc, sub) -> [record]
    for r in records:
        groups[(r.doc, r.sub)].append(r)
    sums, cnts = defaultdict(float), defaultdict(int)
    items = list(groups.values())
    for grp in subsample(items, cap):
        th = defaultdict(list)  # (layer, head) -> [(tok, set)]
        for r in grp:
            for l, heads in r.gates.items():
                for h, s in heads.items():
                    th[(l, h)].append((r.tok, s))
        for lst in th.values():
            lst.sort()
            for i in range(len(lst)):
                ti, si = lst[i]
                for j in range(i + 1, len(lst)):
                    tj, sj = lst[j]
                    gap = abs(tj - ti)
                    if gap == 0 or gap > max_gap:
                        continue
                    sums[gap] += jac(si, sj)
                    cnts[gap] += 1
    return {g: sums[g] / cnts[g] for g in sorted(sums)}


# --------------------------------------------------------------------------------------------------
# Q3 -- cross-example (same model, different doc) ; Q4 -- cross-model (same doc)
# --------------------------------------------------------------------------------------------------
def q3_cross_example(records, subs_of, cap_docs=24, head_stride=1):
    """mean pairwise Jaccard across docs, for fixed (sub, layer, head, tok) -> per-subtask mean."""
    # bucket: (sub, tok, layer, head) -> list of (doc, set)
    buckets = defaultdict(list)
    for r in records:
        sub = subs_of(r)
        for l, heads in r.gates.items():
            for h, s in heads.items():
                if h % head_stride:
                    continue
                buckets[(sub, r.tok, l, h)].append((r.doc, s))
    per_sub_sum, per_sub_cnt = defaultdict(float), defaultdict(int)
    for (sub, _tok, _l, _h), lst in buckets.items():
        docs = {}
        for d, s in lst:
            docs.setdefault(d, s)  # first occurrence per doc
        vals = list(docs.values())
        if len(vals) < 2:
            continue
        if len(vals) > cap_docs:
            vals = vals[:cap_docs]
        tot, cnt = 0.0, 0
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                tot += jac(vals[i], vals[j])
                cnt += 1
        if cnt:
            per_sub_sum[sub] += tot / cnt
            per_sub_cnt[sub] += 1
    return {s: per_sub_sum[s] / per_sub_cnt[s] for s in per_sub_sum}


def q4_cross_model(recs_a, recs_b, subs_of, head_stride=1):
    """Jaccard between model A and B gate sets at the same (doc, tok, layer, head) -> per-subtask."""
    def index(recs):
        idx = {}
        for r in recs:
            idx.setdefault((r.doc, r.tok), r)  # first record per (doc, tok)
        return idx

    ia, ib = index(recs_a), index(recs_b)
    per_sub_sum, per_sub_cnt = defaultdict(float), defaultdict(int)
    for key in ia.keys() & ib.keys():
        ra, rb = ia[key], ib[key]
        sub = subs_of(ra)
        for l, heads_a in ra.gates.items():
            heads_b = rb.gates.get(l)
            if not heads_b:
                continue
            for h, sa in heads_a.items():
                if h % head_stride:
                    continue
                sb = heads_b.get(h)
                if sb is None:
                    continue
                per_sub_sum[sub] += jac(sa, sb)
                per_sub_cnt[sub] += 1
    return {s: per_sub_sum[s] / per_sub_cnt[s] for s in per_sub_sum}


# --------------------------------------------------------------------------------------------------
# Q5 -- within-layer, across heads
# --------------------------------------------------------------------------------------------------
def q5_cross_head_at_layer(records, layer, cap):
    tot, cnt = 0.0, 0
    for r in subsample(records, cap):
        heads = r.gates.get(layer)
        if not heads or len(heads) < 2:
            continue
        items = list(heads.values())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                tot += jac(items[i], items[j])
                cnt += 1
    return tot / cnt if cnt else float("nan")


def q5_head_matrix(records, layer, n_heads, cap):
    M = np.zeros((n_heads, n_heads))
    C = np.zeros((n_heads, n_heads))
    for r in subsample(records, cap):
        heads = r.gates.get(layer)
        if not heads:
            continue
        items = list(heads.items())
        for hi, si in items:
            for hj, sj in items:
                M[hi, hj] += jac(si, sj)
                C[hi, hj] += 1
    with np.errstate(invalid="ignore"):
        return M / np.where(C == 0, np.nan, C)


# --------------------------------------------------------------------------------------------------
# dimensions
# --------------------------------------------------------------------------------------------------
def dims(by_len_a, by_len_b):
    n_layers, n_heads = 0, 0
    for by_len in (by_len_a, by_len_b):
        for recs in by_len.values():
            for r in recs:
                for l, heads in r.gates.items():
                    n_layers = max(n_layers, l + 1)
                    for h in heads:
                        n_heads = max(n_heads, h + 1)
    return n_layers, n_heads


def subs_present(by_len_a, by_len_b):
    subs = set()
    for by_len in (by_len_a, by_len_b):
        for recs in by_len.values():
            for r in recs:
                subs.add(r.sub)
    return subs


# --------------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------------
def plot_q1_layers(A, B, lengths, cap, outdir):
    fig, axes = plt.subplots(1, len(lengths), figsize=(4.2 * len(lengths), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, L in zip(axes, lengths):
        for model, color in ((A, A_COLOR), (B, B_COLOR)):
            recs = model["by_len"].get(L, [])
            curve = q1_gap_curve(recs, cap)
            if curve:
                ax.plot(list(curve), list(curve.values()), color=color, label=model["label"])
        base = baseline_for(A["by_len"].get(L, []) + B["by_len"].get(L, []))
        if base == base:
            ax.axhline(base, ls=":", color="0.4", lw=1, label="random")
        ax.set_title(len_label(L))
        ax.set_xlabel("layer gap")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean Jaccard (same head)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Q1: gate similarity across layers (vs layer distance)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, outdir, "q1_layers.png")


def plot_q1_matrix(A, B, L, n_layers, cap, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model in zip(axes, (A, B)):
        M = q1_matrix(model["by_len"].get(L, []), n_layers, cap)
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"{model['label']} {len_label(L)} (head-pooled)")
        ax.set_xlabel("layer")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Q1: head-pooled layer x layer gate Jaccard")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, outdir, "q1_layer_matrix.png")


def plot_q2_tokens(A, B, lengths, cap, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    handles = []
    for model, base_color in ((A, A_COLOR), (B, B_COLOR)):
        for i, L in enumerate(lengths):
            recs = model["by_len"].get(L, [])
            curve = q2_token_curve(recs, cap)
            if not curve:
                continue
            color = lighten(base_color, 0.55 * (len(lengths) - 1 - i) / max(1, len(lengths) - 1))
            (h,) = ax.plot(list(curve), list(curve.values()), "o-", color=color, ms=4,
                           label=f"{model['label']} {len_label(L)}")
            handles.append(h)
    ax.set_xlabel("decoded-token gap")
    ax.set_ylabel("mean Jaccard")
    ax.set_title("Q2: gate similarity across decoded tokens (same example)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, outdir, "q2_tokens.png")


def plot_q3q4(A, B, lengths, subs_order, cap_docs, head_stride, outdir):
    fig, axes = plt.subplots(1, len(lengths), figsize=(4.8 * len(lengths), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, L in zip(axes, lengths):
        ra = A["by_len"].get(L, [])
        rb = B["by_len"].get(L, [])
        subs_of = _subs_of

        q3a = q3_cross_example(ra, subs_of, cap_docs, head_stride)
        q3b = q3_cross_example(rb, subs_of, cap_docs, head_stride)
        q4 = q4_cross_model(ra, rb, subs_of, head_stride)
        subs = [s for s in subs_order if (s in q3a or s in q3b or s in q4)]
        if not subs:
            subs = sorted(set(q3a) | set(q3b) | set(q4))
        x = np.arange(len(subs))
        w = 0.27
        ax.bar(x - w, [q3a.get(s, np.nan) for s in subs], w, color=A_COLOR,
               label=f"{A['label']} (cross-ex)")
        ax.bar(x, [q3b.get(s, np.nan) for s in subs], w, color=B_COLOR,
               label=f"{B['label']} (cross-ex)")
        ax.bar(x + w, [q4.get(s, np.nan) for s in subs], w, color="tab:green", label="cross-model")
        base = baseline_for(ra + rb)
        if base == base:
            ax.axhline(base, ls=":", color="0.4", lw=1)
        ax.set_title(len_label(L))
        ax.set_xticks(x)
        ax.set_xticklabels([s or "all" for s in subs], rotation=90, fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("mean Jaccard")
    axes[0].legend(fontsize=8)
    fig.suptitle("Q3 (cross-example = positional bias) vs Q4 (cross-model), per subtask")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, outdir, "q3q4_example_model.png")


def plot_q5_cross_head(A, B, lengths, layers3, cap, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    xlabels = [f"first (L{layers3[0]})", "middle", "final"]
    for model, base_color in ((A, A_COLOR), (B, B_COLOR)):
        for i, L in enumerate(lengths):
            recs = model["by_len"].get(L, [])
            ys = [q5_cross_head_at_layer(recs, l, cap) for l in layers3]
            if all(y != y for y in ys):
                continue
            color = lighten(base_color, 0.55 * (len(lengths) - 1 - i) / max(1, len(lengths) - 1))
            marker = "o" if model is A else "s"
            ax.plot(range(3), ys, marker + "-", color=color, ms=5,
                    label=f"{model['label']} {len_label(L)}")
    ax.set_xticks(range(3))
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("mean pairwise head Jaccard")
    ax.set_title("Q5: within-layer agreement across heads")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, outdir, "q5_cross_head.png")


def plot_q5_head_matrix(A, B, L, layers3, n_heads, cap, outdir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    names = ["first", "middle", "final"]
    for row, model in enumerate((A, B)):
        recs = model["by_len"].get(L, [])
        for col, (lname, layer) in enumerate(zip(names, layers3)):
            ax = axes[row, col]
            M = q5_head_matrix(recs, layer, n_heads, cap)
            im = ax.imshow(M, vmin=0, vmax=1, cmap="magma")
            ax.set_title(f"{model['label']}: {lname} layer (L{layer})")
            ax.set_xlabel("head")
            ax.set_ylabel("head")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Q5: head x head gate Jaccard within a layer ({len_label(L)})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, outdir, "q5_head_matrix.png")


def _save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


# subtask resolver installed at runtime (normalizes ``ruler_cwe__8192`` -> ``cwe``; --infer-subtask
# overrides it for logs that store the subtask empty)
_subs_of = lambda r: norm_sub(r.sub)  # noqa: E731


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a-label", required=True)
    ap.add_argument("--a-dumps", nargs="+", required=True)
    ap.add_argument("--b-label", required=True)
    ap.add_argument("--b-dumps", nargs="+", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384, 32768, 65536])
    ap.add_argument("--matrix-length", type=int, default=65536, help="length for the Q1/Q5 matrices")
    ap.add_argument("--outdir", default="plots_pre_post")
    ap.add_argument("--max-records", type=int, default=200, help="cap records per (len,model) for curves")
    ap.add_argument("--matrix-records", type=int, default=80, help="cap records for the heatmaps")
    ap.add_argument("--head-stride", type=int, default=1, help="use every Nth head in Q3/Q4 (speed)")
    ap.add_argument("--cap-docs", type=int, default=24, help="max docs per (sub,layer,head,tok) in Q3")
    ap.add_argument("--infer-subtask", metavar="ORDER:SIZE", default=None,
                    help="derive subtask from doc_id when logs store it empty, e.g. "
                         "'cwe,fwe,...,vt:50' -> sub = order[doc//50].")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    A = {"label": args.a_label, "by_len": load_dumps(args.a_dumps)}
    B = {"label": args.b_label, "by_len": load_dumps(args.b_dumps)}
    for m in (A, B):
        counts = {len_label(L): len(m["by_len"].get(L, [])) for L in args.lengths}
        print(f"{m['label']}: records/len = {counts}")

    n_layers, n_heads = dims(A["by_len"], B["by_len"])
    print(f"dims: {n_layers} layers, {n_heads} heads")
    layers3 = [0, n_layers // 2, n_layers - 1]

    # subtask resolver
    global _subs_of
    if args.infer_subtask:
        order_str, size_str = args.infer_subtask.rsplit(":", 1)
        order = order_str.split(",")
        size = int(size_str)

        def _subs_of(r):  # noqa: F811
            idx = r.doc // size
            return order[idx] if 0 <= idx < len(order) else "?"

        subs_order = order
    else:
        present = sorted(norm_sub(s) for s in subs_present(A["by_len"], B["by_len"]))
        subs_order = present
        if present == [""]:
            print("NOTE: subtask field empty in logs -> Q3/Q4 pooled into a single 'all' bar. "
                  "Pass --infer-subtask ORDER:SIZE to split by subtask.", file=sys.stderr)

    plot_q1_layers(A, B, args.lengths, args.max_records, args.outdir)
    plot_q1_matrix(A, B, args.matrix_length, n_layers, args.matrix_records, args.outdir)
    plot_q2_tokens(A, B, args.lengths, args.max_records, args.outdir)
    plot_q3q4(A, B, args.lengths, subs_order, args.cap_docs, args.head_stride, args.outdir)
    plot_q5_cross_head(A, B, args.lengths, layers3, args.max_records, args.outdir)
    plot_q5_head_matrix(A, B, args.matrix_length, layers3, n_heads, args.matrix_records, args.outdir)
    print(f"done -> {args.outdir}")


if __name__ == "__main__":
    main()
