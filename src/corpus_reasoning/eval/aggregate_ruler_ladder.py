#!/usr/bin/env python3
"""Aggregate per-length RULER-ladder eval JSONs into one per-row ``*_rulerladder.json``.

For each headline core row we run RULER (7 subtasks) at L8192/L16384/L32768 as separate
single-length jobs. We also pull the existing 1-2k (L1024/L2048) RULER cells from the row's main
ladder JSON, so each row has a full RULER@{2k,8k,16k,32k} rung set for the viz rung selector.

Per-rung avg recall is RE-COMPUTED here from the per-subtask cells (grouped by length), not read
from a possibly-partial ``ruler_avg_recall`` -- the landmark/compressive drivers flush incrementally,
so a length is only trustworthy when all 7 subtasks are present (``n`` reported per rung).

Writes ``outputs/eval_results/<row>_rulerladder.json`` with flat keys ``ruler_<rung>_avg_recall``
and ``ruler_<rung>_n`` plus the merged per-subtask ``ruler`` block.
"""
import json
import os
import re

EVDIR = "/scratch/users/prasann/corpus-reasoning/outputs/eval_results"
# map a RULER length token -> headline rung label
LEN_RUNG = {"L1024": "2k", "L2048": "2k", "L8192": "8k", "L16384": "16k", "L32768": "32k"}
EXPECT_N = {"2k": 14, "8k": 7, "16k": 7, "32k": 7}  # 7 subtasks/length (2k = L1024+L2048)

# row -> (existing main-ladder JSON for the 1-2k cells, per-length file prefix for 8k/16k/32k)
ROWS = {
    "q4b-dense-nocpt":   ("q4b-dense-32k_ladder",                        "q4b-dense-nocpt_rulerladder_"),
    "q4b-dense-cptmix":  ("q4b-dense-cptmix-32k-native_ladder",          "q4b-dense-cptmix_rulerladder_"),
    "q4b-lm-nocpt":      ("q4b-lm-32k_ladder",                           "q4b-lm-nocpt_rulerladder_"),
    "q4b-lm-cptmix":     ("q4b-lm-cptmix-5task-40k-v3_ladder_native500", "q4b-lm-cptmix_rulerladder_"),
    "q4b-comp-cptmix":   ("q4b-comp-cptmix-32k_ladder",                  "q4b-comp-cptmix-32k_ruler"),
    "q4b-comp-nocpt":    ("q4b-comp-nocpt-32k_ladder",                   "q4b-comp-nocpt-32k_ruler"),
}
# 2k cells come from the row's main ladder JSON; for dense+CPT it was run separately (Fix 1).
FIX2K = {
    "q4b-dense-cptmix": "q4b-dense-cptmix_ruler_L1024L2048",
    # lm-cptmix's existing ladder has only a partial (4-subtask) 2k RULER block; re-run full 7.
    "q4b-lm-cptmix": "q4b-lm-cptmix_ruler_L1024L2048",
}


def _load(name):
    p = os.path.join(EVDIR, name if name.endswith(".json") else name + ".json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None


def _len_of(cell_key):
    m = re.search(r"_(L\d+)$", cell_key)
    return m.group(1) if m else None


def main():
    summary = {}
    for row, (existing, prefix) in ROWS.items():
        cells = {}  # subtask_Llen -> recall
        model_path = None
        # 2k cells: from Fix-1 file if present else the main ladder JSON's ruler block
        src2k = _load(FIX2K[row]) if row in FIX2K else None
        if src2k is None:
            src2k = _load(existing)
        if src2k and isinstance(src2k.get("ruler"), dict):
            for k, v in src2k["ruler"].items():
                if _len_of(k) in ("L1024", "L2048") and isinstance(v, dict) and "recall" in v:
                    cells[k] = v["recall"]
        # 8k/16k/32k cells: from the per-length jobs
        for L in ("L8192", "L16384", "L32768"):
            d = _load(prefix + L)
            if d is None:
                continue
            model_path = model_path or d.get("model_path")
            if isinstance(d.get("ruler"), dict):
                for k, v in d["ruler"].items():
                    if _len_of(k) == L and isinstance(v, dict) and "recall" in v:
                        cells[k] = v["recall"]
        # group cells -> per-rung avg
        out = {"row": row, "model_path": model_path, "ruler_cells": cells}
        bucket = {}
        for k, rec in cells.items():
            rung = LEN_RUNG.get(_len_of(k))
            if rung:
                bucket.setdefault(rung, []).append(rec)
        for rung in ("2k", "8k", "16k", "32k"):
            vals = bucket.get(rung, [])
            out[f"ruler_{rung}_n"] = len(vals)
            out[f"ruler_{rung}_avg_recall"] = round(sum(vals) / len(vals), 4) if vals else None
            out[f"ruler_{rung}_complete"] = len(vals) >= EXPECT_N[rung]
        op = os.path.join(EVDIR, row + "_rulerladder.json")
        json.dump(out, open(op, "w"), indent=2)
        summary[row] = {
            r: (out[f"ruler_{r}_avg_recall"], f"n={out[f'ruler_{r}_n']}"
                + ("" if out[f"ruler_{r}_complete"] else " PARTIAL"))
            for r in ("2k", "8k", "16k", "32k")
        }
        print(f"[agg] {row}:")
        for r in ("2k", "8k", "16k", "32k"):
            v, tag = summary[row][r]
            print(f"        @{r:>3}: {v}  ({tag})")
    json.dump(summary, open(os.path.join(EVDIR, "_rulerladder_summary.json"), "w"), indent=2,
              default=str)


if __name__ == "__main__":
    main()
