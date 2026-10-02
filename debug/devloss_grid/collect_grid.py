"""Collect the dev-loss grid: every ``<results>/<task>_<rung>.json`` written by ``ctc_devloss_grid.py``
-> ``debug/devloss_grid/grid.csv`` + ``grid.json`` (one record per task x rung x scheme).

    python debug/devloss_grid/collect_grid.py [--results /net/sneetches/data/prasann/devloss_grid/results]

Fails loudly on schema surprises (missing ``summary``/``per_row``, a scheme without a ``full`` twin,
per-row list length mismatches) instead of skipping a file.

Columns: task (manifest row name), driver_task, rung, scheme, ce, ce_se, dce (= ce - ce_full),
dce_se (paired: std of per-row differences / sqrt(n) when both per-row lists are present and equal
length; otherwise sqrt(ce_se^2 + ce_full_se^2), flagged in ``dce_se_kind``), ce_digit, ce_digit_se,
top1, kl, compaction, kept_docs, n_docs, eval_size, rows_requested (from argv --rows),
gold_degenerate (all rows fell back to the gold-blind twin), gold_degenerate_rows, ckpt, ckpt_note,
rung_note (manifest stand-in note), git_commit, file.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifest.json")
# every results dir the grid artifact is built from (earlier roots win a duplicate cell); results_router =
# the learned token router (debug/learned_router/, 2026-09-23)
DEFAULT_RESULTS = ",".join(os.path.join(HERE, d) for d in (
    "results", "results_fl20", "results_fl20p8", "results_goldk0", "results_screen", "results_first20",
    "results_skip", "results_rand20p8", "results_router", "results_router_diff", "results_router_e2e", "results_router_xtask", "results_router_xtask17", "results_router_recipe", "results_attntop",
    "results_layerskip"))
# per-task label-free cutoffs carry a task-specific suffix (`_c0.338`): one canonical grid row per router
import re as _re
SCHEME_CANON = [(_re.compile(r"^(router_xtask_[A-Za-z0-9]+)_c[0-9.]+$"), r"\1_cal")]


def canon(s: str) -> str:
    for pat, rep in SCHEME_CANON:
        if pat.match(s):
            return pat.sub(rep, s)
    return s
SCHEME_ORDER = ["full", "k0", "first16", "first64", "fl16", "fl64", "idf16", "idf64", "fl20", "fl20p8", "rand33", "gold_rand33", "gold_fl64", "gold_fl20", "gold_fl20p8", "gold_k0", "first20", "gold_first20", "fl20p8_noslot", "gold_fl20p8_noslot", "rand20", "first128", "rule20", "grad20", "attnrow20", "gold_fl20p8_skipodd", "gold_fl20p8_skipeven", "gold_fl20p8_skipgdn2", "gold_rand20p8_noslot"]
SCHEME_ORDER += ["gold_only_noslot"] + [f"router_{v}l{lam}{sfx}" for lam in ("0.05", "0.2", "0.5", "1.0", "2.0") for v in ("", "nogold_", "noemb_") for sfx in ("", "_samp")] + ["router_l0.2_nomark"]
SCHEME_ORDER += [f"router_diff_tau{t}" for t in ("0.05", "0.1", "0.2", "0.4")]
SCHEME_ORDER += [f"router_e2e_rho{r}" for r in ("0.05", "0.1", "0.2", "0.3", "0.5")] + [f"router_e2e_tau{t}" for t in ("0.05", "0.1", "0.2")]
RUNGS = ["2k", "8k", "32k"]
COLUMNS = ["task", "driver_task", "rung", "scheme", "ce", "ce_se", "dce", "dce_se", "dce_se_kind", "ce_digit",
           "ce_digit_se", "top1", "kl", "compaction", "layer_frac", "sel_cost", "flops", "kept_docs", "n_docs", "eval_size", "rows_requested",
           "gold_degenerate", "gold_degenerate_rows", "ckpt", "ckpt_note", "rung_note", "git_commit", "file"]


# ---- forward-FLOP model for Qwen3.5-4B (TransformerConfig.qwen3_5_4B, 32 blocks gdn,gdn,gdn,attn) ----
# Per token per block: 2 x block params (projections + FFN). GatedDeltaNet adds its delta-rule state
# work, ~8 * d_k * d_v per value head (read S.k, rank-1 update, decay, read S.q). Softmax attention
# adds the causal quadratic term, QK^T + AV = 2 * n_heads * head_dim * T^2 per sequence. Embedding
# and the LM head (answer positions only, identical for every scheme) are left out.
N_LAYERS = 32
ATTN_LAYERS = frozenset(range(3, 32, 4))
GDN_LIN = 2 * 112_923_840 + 8 * 128 * 128 * 32  # per token
ATTN_LIN = 2 * 107_484_672  # per token
ATTN_QUAD = 2 * 16 * 256  # x T^2 per sequence
# layers the driver's skip schemes bypass for non-kept-document columns (ctc_devloss_grid.SCHEMES)
SKIP_LAYERS = {
    "gold_fl20p8_skipodd": list(range(1, 32, 2)),
    "gold_fl20p8_skipeven": list(range(0, 32, 2)),
    "gold_fl20p8_skipgdn2": [i for j, i in enumerate([b for b in range(32) if (b + 1) % 4 != 0]) if j % 2 == 0],
}


def forward_flops(T: float, skipped_cols: float = 0.0, skip_layers=()) -> float:
    """FLOPs of one forward over ``T`` columns, of which ``skipped_cols`` bypass ``skip_layers``
    (they are neither queries nor keys there -- a gather implementation, not the eval's
    compute-then-discard)."""
    skip = set(skip_layers)
    f = 0.0
    for layer in range(N_LAYERS):
        t = T - skipped_cols if layer in skip else T
        f += (ATTN_LIN * t + ATTN_QUAD * t * t) if layer in ATTN_LAYERS else GDN_LIN * t
    return f


def flops_ratio(scheme: str, lens, comps, layer_fracs, sel_cost: float) -> float | None:
    """Mean over rows of (construction FLOPs / full-row FLOPs) + selector cost in full-forward units.
    ``layer_fracs`` is the driver's masked-columns x skipped-layers / (columns x layers) on the
    compacted row, so the skipped column share is layer_frac * N_LAYERS / len(skip_layers)."""
    if scheme == "full":
        return 1.0
    if not lens or not comps or len(lens) != len(comps):
        return None
    sl = SKIP_LAYERS.get(scheme, [])
    out = []
    for i, (T, c) in enumerate(zip(lens, comps)):
        T2 = c * T
        lf = (layer_fracs[i] if layer_fracs and i < len(layer_fracs) else 0.0) or 0.0
        share = lf * N_LAYERS / len(sl) if sl else 0.0
        out.append(forward_flops(T2, share * T2, sl) / forward_flops(T))
    return float(np.mean(out)) + float(sel_cost or 0.0)


def task_key(driver_task: str, manifest: dict) -> str:
    """``ctc_<row>`` -> manifest row name; ``cpt80`` stays. Aliases the driver may use are checked
    against the manifest's hf_config too."""
    if driver_task in manifest["tasks"]:
        return driver_task
    stripped = driver_task[4:] if driver_task.startswith("ctc_") else driver_task
    if stripped in manifest["tasks"]:
        return stripped
    for name, t in manifest["tasks"].items():
        if t.get("hf_config") and t["hf_config"].lower() == stripped.lower():
            return name
    raise KeyError(f"driver task {driver_task!r} matches no manifest row (tried {stripped!r})")


def rows_requested(argv: list) -> int | None:
    for i, a in enumerate(argv):
        if a == "--rows" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--rows="):
            return int(a.split("=", 1)[1])
    return None


def collect(results_root: str, manifest: dict) -> list[dict]:
    # comma-separated roots merge: a later root only ADDS (task, rung, scheme) cells absent from an
    # earlier one (scheme-only follow-up runs like fl20/gold_fl20 carry their own `full` for pairing)
    files = [f for root in results_root.split(",") for f in sorted(glob.glob(os.path.join(root, "*.json")))]
    if not files:
        print(f"[collect] no JSON under {results_root}", file=sys.stderr)
    recs = []
    seen = set()
    for f in files:
        d = json.load(open(f))
        for key in ("task", "rung", "eval_size", "summary", "per_row", "argv"):
            if key not in d:
                raise SystemExit(f"{f}: missing top-level key {key!r} (schema surprise)")
        if "full" not in d["summary"]:
            raise SystemExit(f"{f}: no 'full' scheme -- every file must carry the reference")
        if d["rung"] not in RUNGS:
            raise SystemExit(f"{f}: unexpected rung {d['rung']!r}")
        task = task_key(d["task"], manifest)
        t = manifest["tasks"][task]
        rung_note = (t.get("rungs", {}).get(d["rung"]) or {}).get("note", "")
        full = d["summary"]["full"]
        full_rows = d["per_row"]["full"].get("ce")
        degenerate = d.get("gold_degenerate_rows", {}) or {}
        for s_raw, m in d["summary"].items():
            s = canon(s_raw)
            if (task, d["rung"], s) in seen:
                continue
            seen.add((task, d["rung"], s))
            for k in ("ce", "ce_se", "compaction"):
                if k not in m:
                    raise SystemExit(f"{f}: summary[{s}] lacks {k!r}")
            rows = d["per_row"].get(s_raw, {}).get("ce")
            if s == "full":
                dce, dce_se, kind = 0.0, 0.0, "reference"
            elif rows is not None and full_rows is not None and len(rows) == len(full_rows) and len(rows) > 1:
                diff = np.asarray(rows, dtype=float) - np.asarray(full_rows, dtype=float)
                diff = diff[~np.isnan(diff)]
                dce, dce_se, kind = float(diff.mean()), float(diff.std(ddof=1) / math.sqrt(diff.size)), "paired"
            else:
                if rows is not None and full_rows is not None and len(rows) != len(full_rows):
                    raise SystemExit(f"{f}: per_row length mismatch {s}={len(rows)} vs full={len(full_rows)}")
                dce = m["ce"] - full["ce"]
                dce_se = math.sqrt((m["ce_se"] or 0.0) ** 2 + (full["ce_se"] or 0.0) ** 2)
                kind = "independent"
            deg_rows = degenerate.get(s_raw, [])
            recs.append({
                "task": task, "driver_task": d["task"], "rung": d["rung"], "scheme": s,
                "ce": m["ce"], "ce_se": m["ce_se"], "dce": dce, "dce_se": dce_se, "dce_se_kind": kind,
                "ce_digit": m.get("ce_digit"), "ce_digit_se": m.get("ce_digit_se"), "top1": m.get("top1"),
                "kl": m.get("kl"), "compaction": m["compaction"], "layer_frac": m.get("layer_frac"), "sel_cost": m.get("sel_cost"),
                # a file may carry its own per-scheme FLOP ratio (layer-skip routers: per-(token, layer) skips that
                # compaction + layer_frac cannot express); otherwise the compaction/layer_frac model
                "flops": m["flops"] if m.get("flops") is not None else flops_ratio(
                    s, d.get("row_lens"), d["per_row"].get(s_raw, {}).get("compaction"),
                    d["per_row"].get(s_raw, {}).get("layer_frac"), m.get("sel_cost") or 0.0), "kept_docs": m.get("kept_docs"),
                "n_docs": m.get("n_docs"), "eval_size": d["eval_size"], "rows_requested": rows_requested(d["argv"]),
                "gold_degenerate": bool(deg_rows) and len(deg_rows) >= d["eval_size"],
                "gold_degenerate_rows": len(deg_rows), "ckpt": d.get("ckpt"), "ckpt_note": t.get("ckpt_note", ""),
                "rung_note": rung_note, "git_commit": d.get("git_commit"), "file": os.path.basename(f),
            })
    order = {s: i for i, s in enumerate(SCHEME_ORDER)}
    recs.sort(key=lambda r: (r["task"], RUNGS.index(r["rung"]), order.get(r["scheme"], 99)))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--out-csv", default=os.path.join(HERE, "grid.csv"))
    ap.add_argument("--out-json", default=os.path.join(HERE, "grid.json"))
    a = ap.parse_args()
    manifest = json.load(open(MANIFEST))
    recs = collect(a.results, manifest)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(recs)
    json.dump({"results_root": a.results, "records": recs}, open(a.out_json, "w"), indent=1)
    cells = sorted({(r["task"], r["rung"]) for r in recs})
    print(f"[collect] {len(recs)} records, {len(cells)} task x rung cells -> {a.out_csv}")
    for task, rung in cells:
        rr = [r for r in recs if r["task"] == task and r["rung"] == rung]
        full = next(r for r in rr if r["scheme"] == "full")
        line = "  ".join(f"{r['scheme']}={r['ce']:.3f}({r['dce']:+.3f})@{r['compaction']:.2f}" for r in rr if r["scheme"] != "full")
        print(f"  {task:14} {rung:>3} eval_size={full['eval_size']:3d} full={full['ce']:.3f} | {line}")


if __name__ == "__main__":
    main()
