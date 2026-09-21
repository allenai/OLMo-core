#!/usr/bin/env python
"""Shrink a CTC rung by DROPPING non-gold documents, the mirror of expand_ctc_rung.py.

`expand_ctc_rung.py` can only grow a rung (it skips any target needing `need <= 0` fillers), so a
ladder whose smallest shipped file already overshoots the bottom rung has no way to reach it. That
is contra_fever's situation: its smallest build (n=100 docs) measures p50 2958 through the real
prompt path, so calling it the 2k rung would put a +44% label error on the x-axis -- exactly the
mislabelling documented in the ctc-rung-labels-not-tokens record.

Dropping distractors is sound wherever ADDING them is not, and vice versa, so the safety argument
is the opposite of expand's and has to be made separately:

* Gold defined by a relation to something specific (query_match, pairwise) stays valid -- removing
  a non-gold document cannot destroy a contradiction pair or a query's answer, and every gold
  document is kept by construction.
* Gold defined by ABSENCE or by STRUCTURE over the corpus is NOT safe to shrink either, for the
  same reason it is not safe to grow: removing documents can turn a non-gold document into an
  orphan/outlier. Those tasks must be regenerated at the smaller n by their own generator.

So this script accepts the same `query_match`/`pairwise` tasks expand accepts and refuses the rest,
reusing expand's own TASKS table rather than restating it. Calibration is expand's too: fit
`prefill = a + b * n_docs` from two real probes through the evaluator's prompt path, solve, build,
re-measure, refine.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPAND = os.path.join(os.path.dirname(_HERE), "ctc_modelscale", "expand_ctc_rung.py")


def _load_expand():
    spec = importlib.util.spec_from_file_location("_expand", _EXPAND)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shrink_rows(rows, drop, cfg, seed):
    """Drop `drop` non-gold documents from every row, remapping every index field.

    Gold documents are never candidates. Index fields are rebuilt from an old->new position map,
    and any index that pointed at a dropped document is removed from the auxiliary fields (it can
    never be a gold one, since gold is protected).
    """
    base = cfg["base"]
    out = []
    for i, row in enumerate(rows):
        rng = random.Random(seed + i)
        docs = row["documents"]
        gold_positions = set()
        raw_gold = row.get("gold_doc_indices") or []
        for g in raw_gold:
            for idx in g if cfg["pairs"] else [g]:
                gold_positions.add(idx - base)
        droppable = [p for p in range(len(docs)) if p not in gold_positions]
        if drop > len(droppable):
            raise ValueError(f"row {i}: cannot drop {drop} of {len(droppable)} non-gold docs")
        removed = set(rng.sample(droppable, drop))
        keep = [p for p in range(len(docs)) if p not in removed]
        remap = {old: new for new, old in enumerate(keep)}

        new_row = dict(row)
        new_row["documents"] = [docs[p] for p in keep]
        if cfg["pairs"]:
            new_row["gold_doc_indices"] = [
                [remap[a - base] + base, remap[b - base] + base] for a, b in raw_gold
            ]
        else:
            new_row["gold_doc_indices"] = [remap[g - base] + base for g in raw_gold]
        for field in cfg.get("index_fields", []):
            if field in row and row[field]:
                new_row[field] = [
                    remap[v - base] + base for v in row[field] if (v - base) in remap
                ]
        for field in (cfg.get("per_doc_fields") or {}):
            if field in row and isinstance(row[field], list) and len(row[field]) == len(docs):
                new_row[field] = [row[field][p] for p in keep]
        out.append(new_row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    expand = _load_expand()
    ap.add_argument("--task", required=True, choices=sorted(expand.TASKS))
    ap.add_argument("--src", required=True)
    ap.add_argument("--targets", required=True, help="comma-separated target MEDIAN prefill tokens")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--calib", type=int, default=16)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--max-refine", type=int, default=2)
    ap.add_argument("--tokenizer", default=expand.TOKENIZER)
    ap.add_argument("--query-position", default="both")
    ap.add_argument("--cot-mode", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    cfg = dict(expand.TASKS[args.task])
    semantics = cfg.get("gold_semantics")
    if semantics not in ("query_match", "pairwise"):
        print(
            f"REFUSED: {args.task} gold is '{semantics}'. Removing documents can turn a non-gold "
            "document into a true positive that the label does not list, exactly as injection can. "
            "Regenerate at the smaller n with the task's own generator instead.",
            file=sys.stderr,
        )
        return 2

    rows = [json.loads(line) for line in open(args.src)]
    print(f"[shrink] {args.task}: {len(rows)} examples from {args.src}", flush=True)
    src_docs = len(rows[0]["documents"])

    meter = expand.PrefillMeter(args.task, args.tokenizer, args.query_position, args.cot_mode)
    calib_rows = rows[: args.calib]

    def probe(drop):
        return meter.lengths(shrink_rows(calib_rows, drop, cfg, args.seed))

    base_lens = probe(0)
    base_med = statistics.median(base_lens)
    lo_drop = max(1, src_docs // 4)
    lo_med = statistics.median(probe(lo_drop))
    b = (base_med - lo_med) / lo_drop  # tokens per document
    a = base_med - b * src_docs
    print(
        f"[shrink] source: {src_docs} docs -> REAL prefill median={base_med:.0f}; "
        f"fit prefill ~ {a:.0f} + {b:.2f}*n_docs",
        flush=True,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    for target in [int(t) for t in args.targets.split(",")]:
        drop = src_docs - int(round((target - a) / b))
        built = None
        for attempt in range(args.max_refine + 1):
            if drop <= 0:
                print(f"[shrink] {target}: SKIP -- source is already at or below it", flush=True)
                drop = None
                break
            lens = probe(drop)
            med = statistics.median(lens)
            print(
                f"[shrink] {target}: attempt {attempt} drop={drop} -> realized median={med:.0f} "
                f"({(med - target) / target:+.1%})",
                flush=True,
            )
            if abs(med - target) / target <= args.tol:
                built = drop
                break
            drop = int(round(drop - (target - med) / b))
        if built is None and drop:
            built = drop
        if not built:
            continue

        out_rows = shrink_rows(rows, built, cfg, args.seed)
        lens = meter.lengths(out_rows)
        lens_sorted = sorted(lens)
        p50 = lens_sorted[len(lens_sorted) // 2]
        p90 = lens_sorted[int(len(lens_sorted) * 0.9)]
        dst = os.path.join(args.out_dir, f"rung_{target}.jsonl")
        with open(dst, "w") as f:
            for row in out_rows:
                f.write(json.dumps(row) + "\n")
        print(
            f"[shrink] WROTE {dst}\n"
            f"           realized prefill p50={p50} p90={p90} max={lens_sorted[-1]} | "
            f"{src_docs - built} docs (-{built}/ex) | gold preserved on all {len(out_rows)} examples",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
