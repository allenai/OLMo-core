"""
Is the setA training data IID with what the CTC suite actually grades -- at every context length?

This is the check that has cost this project the most: a train/eval mismatch does not error, it
scores. Realistic-mode contradiction graded on the ``both`` ladder read 0.559 when the true number
was 0.946; a train n-max of 697 against an eval of 1,423 documents read as a long-context collapse.
Both were data bugs wearing a capability result's clothes.

The authority is **olmo-eval's CTC suite** -- its ROSTER (which rows exist, which spec grades
them, which rungs they have), its vendored ``ctc`` (the parser, scorer and prompt builder that
will grade the run) and its data (``PrasannSinghal/ctc-suite-eval`` on HF). The first attempt at
this audit compared against the ladder *formula* and the spec's default rung list instead, and
half its verdicts were about the audit, not the data. So every comparison here is against a real
eval row at the same rung:

1. **Rung coverage** (per task): rungs the eval grades that training lacks (a train gap) and
   rungs training has that nothing grades (unevaluated, reported not failed).
2. **Spec**: the spec our renderer uses must be the spec the ROSTER grades with.
3. **Distribution** (per rung): document count, per-document length and gold cardinality of the
   train rows against the eval rows. oolong is one long document, so it compares total length.
4. **Fields**: every top-level field an eval row carries must exist on the train rows (rerank's
   scorer reads ``ce_scores``; a train row without them cannot be graded the way eval rows are).
5. **Prompt**: on the SAME eval row, the training renderer (``olmo_core`` ``build_prompt``, what
   the converter tokenizes) and the eval's ``spec.build_prompt`` must produce identical text.
6. **Target**: the training target, parsed and scored by the EVAL's parser and scorer with gold
   resolved exactly as ``CTCScorer`` resolves it, must score 1.0 -- on train rows and on eval rows.

Runs where both weka (train data) and HF (eval data) are reachable::

    python debug/ctc_sft_setA/audit_train_eval_iid.py \\
        --root /weka/.../ctc_sft_sets/setA_max20 --vendor <olmo-eval>/.../ctc_suite/_vendor
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))

#: train task -> olmo-eval ROSTER key. The ROSTER row then names the eval subset and grading spec.
TASK_TO_ROW = {
    "nq": "ctc_nq",
    "hotpotqa": "ctc_hpqa",
    "qdmatch_nq": "ctc_qdmatch_nq",
    "outlier": "ctc_outlier",
    "oolong": "ctc_oolong",
    "contradiction": "ctc_contradiction",
    "xabsence": "ctc_xabsence",
    "reorder": "ctc_reorder",
    "rerank": "ctc_rerank",
    "strmatch": "ctc_strmatch",
    "grouping": "ctc_grouping",
}
BUCKETS = ["2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k"]
#: The training window: eval rungs above it are out of scope, not train gaps.
MAX_TRAIN_RUNG = "256k"


def _load_builder_roster(repo: str) -> Dict[str, dict]:
    """The spec / chunk_by each train task is RENDERED with, from the builder that made the data."""
    sys.path.insert(0, os.path.join(repo, "src", "scripts", "data", "ctc_sft"))
    import build_ctc_sft as B  # noqa: E402

    return {t.name: {"spec": t.spec, "chunk_by": t.chunk_by} for t in B.SET_A}


def _train_rows(root: str, task: str, bucket: str, k: int) -> List[dict]:
    path = os.path.join(root, "per_task", f"_b{bucket}", task, "train.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if len(rows) >= k:
                break
            ex = json.loads(line)
            rows.append(_normalize(ex["ex"] if "ex" in ex and "documents" not in ex else ex))
    return rows


def _eval_rows(ds: str, subset: str, rung: str, k: int) -> Optional[List[dict]]:
    """First ``k`` rows of one rung's parquet, read one batch at a time (the 256k files are ~90MB)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    path = f"datasets/{ds}/data/{subset}/{rung}.parquet"
    if not fs.exists(path):
        return None
    pf = pq.ParquetFile(fs.open(path))
    out: List[dict] = []
    for batch in pf.iter_batches(batch_size=k):
        out.extend(batch.to_pylist())
        if len(out) >= k:
            break
    return [_normalize(r) for r in out[:k]]


def _normalize(ex: dict) -> dict:
    """The HF parquet stores dict-valued fields (outlier ``meta``, ``_meta``) as JSON strings."""
    for key in ("meta", "_meta"):
        if isinstance(ex.get(key), str):
            try:
                ex[key] = json.loads(ex[key])
            except ValueError:
                pass
    return ex


def _gold(sp, ex: dict):
    """Exactly ``CTCScorer.score``'s gold resolution."""
    if sp.extra.get("score_takes_example"):
        return ex
    return ex.get(sp.extra.get("gold_field", "gold_doc_indices")) or ex


def _gold_card(sp, ex: dict) -> Optional[int]:
    if sp.extra.get("score_takes_example"):
        return None
    g = ex.get(sp.extra.get("gold_field", "gold_doc_indices"))
    return len(g) if isinstance(g, (list, tuple)) else None


def _doc_text(d) -> str:
    return str(d.get("text", "")) if isinstance(d, dict) else str(d)


def _stats(xs: List[float]) -> Optional[dict]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"min": min(xs), "med": statistics.median(xs), "max": max(xs)}


def _first_diff(a: str, b: str) -> str:
    i = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
    return (f"first diff at char {i} (train len {len(a)}, eval len {len(b)}): "
            f"train={a[max(0, i - 40):i + 60]!r} | eval={b[max(0, i - 40):i + 60]!r}")


def _render_train(ex: dict, spec: str, qpos: str, alpaca: bool = False):
    """What the converter tokenizes (segment_prompt_to_chunks' build_prompt call, minus markers).

    ``alpaca=True`` adds the Alpaca preamble the eval wraps every prompt in, so the BODY can be
    compared byte-for-byte; the converter itself renders ``use_alpaca=False`` inside the chat
    template, a wrapper difference reported once, globally, rather than per cell.
    """
    from olmo_core.data.corpus_reasoning_prompts import build_prompt

    return build_prompt(ex, task=spec, query_position=qpos, use_alpaca=alpaca, cot_mode="none",
                        use_titles=False)


def _self_score(sp, ex: dict, answer: str) -> float:
    parsed = sp.parse(answer, len(ex.get("documents") or []))
    return float(sp.score(parsed, _gold(sp, ex)).get(sp.primary_metric, 0.0))


def audit_cell(task, bucket, train, evalr, sp, train_spec, qpos, cal=None) -> dict:
    rec: dict = {"task": task, "bucket": bucket, "train_rows": len(train), "eval_rows": len(evalr)}
    fails: List[str] = []
    line_mode = task == "oolong"

    def dist(rows):
        nd = [len(r.get("documents") or []) for r in rows]
        dl = [statistics.median([len(_doc_text(d)) for d in r["documents"]]) for r in rows
              if r.get("documents")]
        tot = [sum(len(_doc_text(d)) for d in r.get("documents") or []) for r in rows]
        gc = [_gold_card(sp, r) for r in rows]
        return {"n_docs": _stats(nd), "doc_chars": _stats(dl), "total_chars": _stats(tot),
                "gold_card": _stats(gc)}

    rec["train"], rec["eval"] = dist(train), dist(evalr)

    # 3. distributions: medians within tolerance. (An earlier version also accepted any eval median
    # inside the train RANGE; that let oolong's 2x-short training contexts through.) doc_chars is
    # skipped for qdmatch, whose "documents" mix short queries and long passages, so its median
    # flips between the two modes on the count alone.
    keys = ["total_chars"] if line_mode else ["n_docs", "doc_chars", "gold_card"]
    if task.startswith("qdmatch"):
        keys = [k for k in keys if k != "doc_chars"]
    tol = {"n_docs": 0.06, "doc_chars": 0.15, "total_chars": 0.15, "gold_card": 0.25}
    for key in keys:
        t, e = rec["train"][key], rec["eval"][key]
        if not t or not e:
            continue
        if key == "n_docs" and cal and ("capped" in cal or "capped_note" in cal):
            # a deliberate, recorded cap (the eval's own row overflows the training window):
            # compare against the calibrated count instead of the eval's
            want = cal.get("fill_to", cal.get("n_docs"))
            rec["n_docs_capped_to"] = want
            if abs(t["med"] - want) > tol[key] * want:
                fails.append(f"n_docs: train median {t['med']} vs calibrated cap {want}")
            continue
        if abs(t["med"] - e["med"]) > tol[key] * max(e["med"], 1):
            fails.append(f"{key}: train {t['min']}/{t['med']}/{t['max']} vs eval "
                         f"{e['min']}/{e['med']}/{e['max']} (min/med/max)")

    # 3b. mixes that are part of the task definition, not its size.
    def mix(rows, fn):
        from collections import Counter

        c = Counter(fn(r) for r in rows)
        return {k: round(v / max(len(rows), 1), 2) for k, v in c.most_common()}

    if task == "oolong":
        tg = lambda r: (r.get("_meta") or {}).get("task_group")  # noqa: E731
        rec["mix_train"], rec["mix_eval"] = mix(train, tg), mix(evalr, tg)
        if set(rec["mix_train"]) != set(rec["mix_eval"]):
            fails.append(f"question-type mix: train {rec['mix_train']} vs eval {rec['mix_eval']}")
    if task == "rerank":
        def null_frac(r):
            ce = r.get("ce_scores") or []
            return round(sum(x is None for x in ce) / max(len(ce), 1), 1)

        rec["null_ce_train"] = statistics.mean(null_frac(r) for r in train) if train else None
        rec["null_ce_eval"] = statistics.mean(null_frac(r) for r in evalr) if evalr else None
        if rec["null_ce_train"] is not None and abs(rec["null_ce_train"] - rec["null_ce_eval"]) > 0.1:
            fails.append(f"unscored (foreign-fill) document share: train {rec['null_ce_train']:.2f} "
                         f"vs eval {rec['null_ce_eval']:.2f}")

    # 4. fields the eval row carries that the train rows do not.
    ek = set().union(*(r.keys() for r in evalr))
    tk = set().union(*(r.keys() for r in train))
    missing = sorted(k for k in ek - tk if not k.startswith("_"))
    rec["fields_missing_in_train"] = missing
    # A field matters only if something reads it. The prompt and target checks below already prove
    # the renderer and grader agree, EXCEPT for a spec whose scorer reads the whole example (rerank
    # reads ce_scores); for everything else a missing field is provenance (outlier `meta` vs
    # `_meta`, qdmatch counts, reorder offsets) and is recorded, not failed.
    if missing and sp.extra.get("score_takes_example"):
        fails.append(f"eval rows carry fields the train rows lack: {missing}")

    # 5. prompt parity on the SAME eval rows.
    mism, err = 0, None
    for ex in evalr[:5]:
        try:
            tp, _ = _render_train(ex, train_spec, qpos, alpaca=True)
            ep = sp.build_prompt(ex, query_position=qpos)
            if tp != ep:
                mism += 1
                rec.setdefault("prompt_diff", _first_diff(tp, ep))
        except Exception as e:  # noqa: BLE001 -- report, never abort the sweep
            err = err or f"{type(e).__name__}: {e}"
    rec["prompt_mismatch"] = mism
    if err:
        fails.append(f"prompt render: {err}")
    if mism:
        fails.append(f"prompt BODY differs from the eval's on {mism}/{min(5, len(evalr))} eval rows; "
                     f"{rec['prompt_diff']}")

    # 6. training target graded by the eval, on train rows and on eval rows.
    for name, rows in (("train", train), ("eval", evalr)):
        scores, err = [], None
        for ex in rows:
            try:
                _, ans = _render_train(ex, train_spec, qpos)
                scores.append(_self_score(sp, ex, ans))
                if scores[-1] < 1.0 and f"worst_{name}" not in rec:
                    parsed = sp.parse(ans, len(ex.get("documents") or []))
                    rec[f"worst_{name}"] = {
                        "target": ans[:300], "parsed": str(parsed)[:300],
                        "metrics": sp.score(parsed, _gold(sp, ex)),
                        "gold": str(ex.get(sp.extra.get("gold_field", "gold_doc_indices")))[:200],
                        "ce_scores": str(ex.get("ce_scores"))[:300],
                    }
            except Exception as e:  # noqa: BLE001
                err = err or f"{type(e).__name__}: {e}"
        rec[f"self_score_{name}"] = round(statistics.mean(scores), 4) if scores else None
        if err:
            fails.append(f"target on {name} rows: {err}")
        elif scores and statistics.mean(scores) < 1.0:
            fails.append(f"training target scores {statistics.mean(scores):.3f} (not 1.0) under "
                         f"the eval's grader on {name} rows; first: {rec.get(f'worst_{name}')}")
    rec["fails"] = fails
    rec["status"] = "FAIL" if fails else "ok"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the set root holding per_task/")
    ap.add_argument("--vendor", required=True,
                    help="olmo-eval .../ctc_suite/_vendor -- the code that will GRADE the run")
    ap.add_argument("--roster", default=os.path.join(HERE, "olmo_eval_roster.json"),
                    help="frozen olmo-eval ROSTER (dump_olmo_eval_roster.py)")
    ap.add_argument("--repo", default=os.path.abspath(os.path.join(HERE, os.pardir, os.pardir)))
    ap.add_argument("--sample", type=int, default=40, help="rows per (task, rung) on each side")
    ap.add_argument("--query-position", default="both",
                    help="the eval pins 'both' (olmo-eval QUERY_POSITION)")
    ap.add_argument("--tasks", nargs="*", default=list(TASK_TO_ROW))
    ap.add_argument("--calibration", default="",
                    help="eval_calibration.json the set was built with: 256k cells it caps are "
                         "checked against the cap, not the eval's larger count")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    R = json.load(open(args.roster))
    sys.path.insert(0, args.vendor)
    sys.path.insert(0, os.path.join(args.repo, "src"))
    import ctc.tasks as T
    from ctc.format import registry

    T.load_all()
    builder = _load_builder_roster(args.repo)
    calib = json.load(open(args.calibration))["tasks"] if args.calibration else {}
    print(f"grading through the VENDORED spec at {args.vendor}; ROSTER from olmo-eval "
          f"{R['olmo_eval_commit']}; eval data {R['hf_dataset']}\n", flush=True)

    cells, coverage, task_fails = [], {}, {}
    top = BUCKETS.index(MAX_TRAIN_RUNG)
    for task in args.tasks:
        row = R["roster"][TASK_TO_ROW[task]]
        train_spec = builder[task]["spec"]
        tf = []
        if train_spec != row["spec"]:
            tf.append(f"rendered as spec {train_spec!r} but the eval grades {TASK_TO_ROW[task]} "
                      f"with {row['spec']!r}")
        if not row.get("spec_registered"):
            tf.append(f"the eval's spec {row['spec']!r} is not registered in its own vendored "
                      "registry -- this row cannot be graded at all")
        eval_rungs = [r[1:] for r in row["rungs"]]
        train_b = [b for b in BUCKETS if _train_rows(args.root, task, b, 1)]
        in_window = [r for r in eval_rungs if r in BUCKETS[: top + 1]]
        coverage[task] = {
            "eval_rungs": eval_rungs, "train_buckets": train_b,
            "train_gap": [r for r in in_window if r not in train_b],
            "unevaluated": [b for b in train_b if b not in eval_rungs],
        }
        if coverage[task]["train_gap"]:
            tf.append(f"the eval grades {coverage[task]['train_gap']} but training has no rows there")
        task_fails[task] = tf
        try:
            sp = registry.get(row["spec"] if row.get("spec_registered") else train_spec)
        except KeyError:
            continue
        for b in train_b:
            if b not in eval_rungs:
                continue
            evalr = _eval_rows(R["hf_dataset"], row["subset"], "r" + b, args.sample)
            if not evalr:
                cells.append({"task": task, "bucket": b, "status": "FAIL",
                              "fails": [f"eval parquet {row['subset']}/r{b} missing on HF"]})
                continue
            train = _train_rows(args.root, task, b, args.sample)
            rec = audit_cell(task, b, train, evalr, sp, train_spec, args.query_position,
                             cal=calib.get(task, {}).get(b))
            cells.append(rec)
            t, e = rec.get("train", {}), rec.get("eval", {})
            nd = lambda d, k: (f"{d[k]['med']:g}" if d.get(k) else "-")  # noqa: E731
            print(f"{task:<17}{b:>5}  n_docs {nd(t,'n_docs'):>6}/{nd(e,'n_docs'):<6}"
                  f" doc_chars {nd(t,'doc_chars'):>6}/{nd(e,'doc_chars'):<6}"
                  f" gold {nd(t,'gold_card'):>3}/{nd(e,'gold_card'):<3}"
                  f" self {rec.get('self_score_train')}/{rec.get('self_score_eval')}"
                  f" prompt_mismatch {rec.get('prompt_mismatch')}  {rec['status']}", flush=True)

    print("\n=== coverage (train buckets vs eval rungs, window <= "
          f"{MAX_TRAIN_RUNG}) ===")
    for task, c in coverage.items():
        print(f"  {task:<17} eval {c['eval_rungs'][0]}..{c['eval_rungs'][-1]:<5} "
              f"train {c['train_buckets'][0] if c['train_buckets'] else '-'}.."
              f"{c['train_buckets'][-1] if c['train_buckets'] else '-':<5}"
              f" gap={c['train_gap'] or '-'}  unevaluated={c['unevaluated'] or '-'}")

    bad_cells = [c for c in cells if c["status"] == "FAIL"]
    bad_tasks = {t: f for t, f in task_fails.items() if f}
    if bad_tasks or bad_cells:
        print("\n=== FAILURES ===")
        for t, fs in bad_tasks.items():
            for f in fs:
                print(f"  {t}: {f}")
        for c in bad_cells:
            for f in c["fails"]:
                print(f"  {c['task']}@{c['bucket']}: {f}")
    print("\n=== wrapper (global) ===\n  the eval sends spec.build_prompt as a RAW COMPLETION: Alpaca "
          "preamble, no chat template. The converter renders use_alpaca=False inside the Qwen chat "
          "template. Bodies are compared with the Alpaca preamble applied; the wrapper itself "
          "differs for every task.")
    n_ok = len(cells) - len(bad_cells)
    print(f"\n{n_ok}/{len(cells)} graded cells IID-clean; {len(bad_tasks)} task-level issue(s)"
          + ("" if not (bad_cells or bad_tasks) else "  -- DO NOT TRAIN until understood"))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"vendor": args.vendor, "roster_commit": R["olmo_eval_commit"],
                       "coverage": coverage, "task_fails": task_fails, "cells": cells}, f,
                      indent=2, default=str)
        print(f"wrote {args.out}")
    raise SystemExit(1 if (bad_cells or bad_tasks) else 0)


if __name__ == "__main__":
    main()
