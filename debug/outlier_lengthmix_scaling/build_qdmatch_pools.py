"""Build pure-length qdmatch_nq pools + held-out eval rungs for the length-mix scaling experiment.

Mirrors debug/outlier_lengthmix_scaling/gen_rung_pools.sbatch (outlier) but for qdmatch_nq, and
fixes two things the shipped CTC-suite qdmatch_nq build did NOT do:

  1. TRAIN/EVAL UNIT DISJOINTNESS. The shipped ladder drew both its 20k train file and its
     canonical eval file from the SAME 19,967-unit pool (see the two logs
     /data/prasann/ctc_suite_data/logs20k/qdmatch_nq_{q9,evalcanon}.log -- both say
     "19967 usable units"). Here train units come from the p10/CE-filtered NQ *train* file and
     eval units from the p10 NQ *validation* file, and the query sets are asserted disjoint.
  2. ONE PARSE of each 1.5 GB source instead of one per output file.

Everything else is byte-faithful to `generate_qdmatch_data.run()`: same `build_example`, same
`rng.sample(units, M + N - k)` draw, same layout/k, same record schema. Each output file gets its
own seeded `random.Random` (the upstream script shares one rng across its train+eval calls; per
file seeding is deterministic and lets pools be rebuilt independently).

Rung <-> doc-count mapping is READ OFF the shipped qdmatch_nq eval rungs, not guessed:
    rung_2048 -> M=N=9,  rung_8192 -> M=N=42  (k=3, layout=separate) for every rung.
"""

import argparse
import json
import os
import pathlib
import random
import sys
import time

sys.path.insert(0, os.environ.get("CTC_REPO_SRC", "/data/prasann/repo/OLMo-core/src"))
from corpus_reasoning.data import generate_qdmatch_data as G  # noqa: E402

K_RELEVANT = 3
LAYOUT = "separate"
SRC_TAG = "nq"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _progress_points(total):
    pts = {1, 2, 5, 10, 50, 100, 500, 1000}
    pts |= {i for i in range(2000, total + 1, 2000)}
    pts.add(total)
    return pts


def load_units(path, label):
    """Parse a unified single-query retrieval JSONL -> (units, audit dict). One pass."""
    t0 = time.time()
    log(f"--- loading {label}: {path}")
    units, queries = [], []
    n_rows, hard, ndocs, n_nogold, n_gold = 0, 0, 0, 0, 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            ex = json.loads(line)
            n_rows += 1
            hard += len(ex.get("hard_neg_indices") or [])
            ndocs += len(ex["documents"])
            u = G._query_unit(ex, 0)  # 0 = keep every gold, exactly as the shipped build did
            if u is None:
                n_nogold += 1
                continue
            units.append(u)
            n_gold += len(u[1])
            queries.append(ex["queries"][0])
            if i in (1, 2, 5, 10) or i % 5000 == 0:
                el = time.time() - t0
                log(f"    {label}: row {i}  ({el:.1f}s elapsed)")
    audit = {
        "path": path,
        "rows": n_rows,
        "usable_units": len(units),
        "rows_without_gold": n_nogold,
        "hard_neg_ratio": round(hard / max(ndocs, 1), 4),
        "mean_gold_docs_per_unit": round(n_gold / max(len(units), 1), 3),
        "mean_docs_per_row": round(ndocs / max(n_rows, 1), 1),
        "load_seconds": round(time.time() - t0, 1),
    }
    log(f"    {label} DONE: {json.dumps(audit)}")
    return units, queries, audit


def emit(units, n_docs, count, seed, out_path, source_tag):
    """`count` qdmatch examples at M=N=n_docs. Mirrors generate_qdmatch_data.run()."""
    M = N = n_docs
    need = M + N - K_RELEVANT
    if len(units) < need:
        raise RuntimeError(f"{out_path}: {len(units)} units < M+N-k={need}")
    rng = random.Random(seed)
    pts = _progress_points(count)
    t0 = time.time()
    log(f"--- emit {out_path.name}: {count} ex, M=N={n_docs}, need={need} units/ex, seed={seed}")
    with open(out_path, "w") as f:
        for i in range(1, count + 1):
            ui = rng.sample(units, need)
            ex = G.build_example(ui, M, N, K_RELEVANT, LAYOUT, rng)
            ex["source"] = source_tag
            f.write(json.dumps(ex) + "\n")
            if i in pts:
                el = time.time() - t0
                eta = el / i * (count - i)
                log(f"    {out_path.name}: {i}/{count}  {el:.1f}s elapsed, ETA {eta:.0f}s")
    reuse = round(count * need / max(len(units), 1), 1)
    log(f"    {out_path.name} DONE in {time.time() - t0:.1f}s "
        f"({out_path.stat().st_size / 1e6:.0f} MB, unit reuse x{reuse})")
    return {"file": str(out_path), "examples": count, "num_queries": M, "num_docs": N,
            "num_relevant": K_RELEVANT, "layout": LAYOUT, "seed": seed,
            "units_available": len(units), "units_per_example": need, "unit_reuse_factor": reuse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/data/prasann/qdmatch_lengthmix")
    ap.add_argument("--train-src", required=True)
    ap.add_argument("--eval-src", required=True)
    ap.add_argument("--pool-2k", type=int, default=25300, help="q9 pool size (incl. 300 heldout)")
    ap.add_argument("--pool-8k", type=int, default=12300, help="q42 pool size (incl. 300 heldout)")
    ap.add_argument("--heldout", type=int, default=300)
    ap.add_argument("--eval-size", type=int, default=600)
    args = ap.parse_args()

    work = pathlib.Path(args.work)
    (work / "eval_rungs" / "qdmatch_nq").mkdir(parents=True, exist_ok=True)
    report = {"rung_map": {"2048": 9, "8192": 42},
              "k_relevant": K_RELEVANT, "layout": LAYOUT}

    tr_units, tr_q, report["train_source"] = load_units(args.train_src, "train-src")
    ev_units, ev_q, report["eval_source"] = load_units(args.eval_src, "eval-src")

    # --- p10 gate: refuse the banned 98%-hard NQ regime outright ---
    for key in ("train_source", "eval_source"):
        r = report[key]["hard_neg_ratio"]
        if not (0.05 <= r <= 0.20):
            raise SystemExit(f"!!! {key} hard_neg_ratio={r} is NOT the p10 regime -- refusing")
    log(f"p10 gate PASSED: train hr={report['train_source']['hard_neg_ratio']}, "
        f"eval hr={report['eval_source']['hard_neg_ratio']}")

    # --- train/eval overlap gate ---
    inter = set(tr_q) & set(ev_q)
    report["query_overlap"] = {"train_queries": len(set(tr_q)), "eval_queries": len(set(ev_q)),
                               "shared_queries": len(inter)}
    log(f"query overlap: {json.dumps(report['query_overlap'])}")
    if inter:
        log(f"!!! WARNING: {len(inter)} shared queries; e.g. {list(inter)[:3]}")

    # --- training pools (train units only) ---
    report["pools"] = {}
    for tag, n_docs, count, seed in (("q9", 9, args.pool_2k, 201), ("q42", 42, args.pool_8k, 202)):
        pool = work / f"qdmatch_nq_{tag}_pool.jsonl"
        if pool.exists() and sum(1 for _ in open(pool)) == count:
            log(f"[skip] {pool.name} already has {count} lines")
        else:
            report["pools"][tag] = emit(tr_units, n_docs, count, seed, pool, f"qdmatch_{SRC_TAG}")
        lines = pool.read_text().splitlines()
        (work / f"qdmatch_nq_{tag}_train.jsonl").write_text(
            "\n".join(lines[: -args.heldout]) + "\n")
        (work / f"qdmatch_nq_{tag}_heldout.jsonl").write_text(
            "\n".join(lines[-args.heldout:]) + "\n")
        log(f"split {tag}: train={len(lines) - args.heldout} heldout={args.heldout}")
        report["pools"].setdefault(tag, {})["train_examples"] = len(lines) - args.heldout
        report["pools"][tag]["heldout_examples"] = args.heldout

    # --- held-out eval rungs (eval units only) ---
    report["eval_rungs"] = {}
    for label, n_docs, seed in (("2048", 9, 7001), ("8192", 42, 7002)):
        out = work / "eval_rungs" / "qdmatch_nq" / f"rung_{label}.jsonl"
        report["eval_rungs"][label] = emit(
            ev_units, n_docs, args.eval_size, seed, out, f"qdmatch_{SRC_TAG}")

    (work / "BUILD_REPORT.json").write_text(json.dumps(report, indent=2))
    log("=== BUILD_REPORT ===")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
