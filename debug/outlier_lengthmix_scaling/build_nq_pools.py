"""Build pure-length **nq** (retrieval-graded) pools + held-out eval rungs for the length-mix
data-scaling framework.

Sibling of `build_qdmatch_pools.py` (qdmatch_nq) and `gen_rung_pools.sbatch` (outlier); same
contract, different task:

  * SOURCE = the p10 + CE-filtered NQ pipeline ONLY ([[nq-pipeline-10pct-hardneg-cefilter]]).
    Train units come from `nq_train_k25-202_clean.jsonl` (19,967 rows, hard-ratio ~0.097), eval
    units from `nq_validation_k25-202_600.jsonl`. The hard-negative ratio of BOTH files is
    re-measured here and the build hard-fails outside [0.05, 0.20], so the retired 98%-hard NQ
    cannot slip in.
  * TRAIN/EVAL DISJOINTNESS is asserted on the query strings (train file vs validation file).
  * LENGTH is set by `k` = documents per example. nq rows carry 25-202 docs, so a fixed-length
    example is produced by SHRINKING a row: keep every gold doc, keep a random `k - |gold|`
    subset of its distractors (preserving the `hard_neg_indices` flags), reshuffle, and rewrite
    the 0-indexed `gold_doc_indices` ([[gold-doc-indices-per-task-base]]: nq is 0-indexed).
    This is byte-for-byte the same operation `build_v2_eval_ladders.py` uses in "shrink" mode for
    the shipped nq eval ladder -- including the CLAMP (a row with fewer than `k` docs keeps all of
    them rather than being dropped), so train and eval see the same length distribution.
  * ONE PARSE of the 1.5 GB train source. Both pools (and the mild reuse variants) are emitted in
    that single streaming pass.

Reuse: nq's unit is a QUERY, and there are only ~19,967 of them, so a 20,000-example pool needs a
reuse factor of ~1.02 (a handful of queries appear twice with *different* distractor subsets).
Variant-1 examples are written first and variant-2 examples are appended at the END of the pool,
so every arm except the largest is duplicate-free, and the 300-example held-out split is carved
out BEFORE reuse (its queries never appear in the train pool at all).

  # calibrate k -> tokens (needs a tokenizer)
  python build_nq_pools.py --calibrate 9,11,13,44,48,52 --eval-src <validation.jsonl>
  # build
  python build_nq_pools.py --work /data/prasann/nq_lengthmix --train-src ... --eval-src ... \
      --k2k 11 --k8k 48
"""

import argparse
import json
import os
import pathlib
import random
import sys
import time

sys.path.insert(0, os.environ.get("CTC_REPO_SRC", "/data/prasann/repo/OLMo-core/src"))

TASK = "retrieval"          # nq's build_prompt task name (there is NO "nq" task in build_prompt:
                            # it would silently fall through to the generic QA instruction)
SRC_TAG = "nq"
SHRINK_SEED = 4242          # variant 1
SHRINK_SEED2 = 909091       # variant 2 (reuse)
DUP_MOD, DUP_REM = 10, 3    # rows eligible for a second variant: i % 10 == 3 (~10%)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def shrink(ex, k, rng):
    """Return a copy of `ex` holding exactly min(k, len(docs)) documents.

    All gold docs are kept; the remaining slots are a random subset of the distractors, with the
    `hard_neg_indices` flags carried along. Everything is reshuffled and the 0-indexed
    `gold_doc_indices` / `hard_neg_indices` are rewritten to the new positions.
    """
    docs = ex["documents"]
    gold = sorted({i for i in (ex.get("gold_doc_indices") or []) if 0 <= i < len(docs)})
    if not gold:
        return None
    gset = set(gold)
    hard_old = set(ex.get("hard_neg_indices") or [])
    distr = [(docs[i], i in hard_old) for i in range(len(docs)) if i not in gset]
    rng.shuffle(distr)
    keep = min(max(k - len(gold), 0), len(distr))
    chosen = distr[:keep]
    combined = [docs[g] for g in gold] + [d for d, _ in chosen]
    perm = list(range(len(combined)))
    rng.shuffle(perm)
    newpos = {old: new for new, old in enumerate(perm)}
    out = dict(ex)
    out["documents"] = [combined[p] for p in perm]
    out["gold_doc_indices"] = sorted(newpos[j] for j in range(len(gold)))
    out["hard_neg_indices"] = sorted(
        newpos[len(gold) + j] for j, (_, h) in enumerate(chosen) if h
    )
    out["source"] = SRC_TAG
    return out


# --------------------------------------------------------------------------- calibration
def calibrate(path, ks, tokenizer, sample, query_position):
    """Median rendered TOKEN length per candidate k. Labels lie ([[ctc-rung-labels-not-tokens]]),
    so the k for each rung is measured, never read off a table."""
    from transformers import AutoTokenizer

    from corpus_reasoning.lib.data_format import build_prompt

    tok = AutoTokenizer.from_pretrained(tokenizer)
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= sample:
                break
    log(f"calibrating on {len(rows)} rows from {os.path.basename(path)}; "
        f"docs/row min={min(len(r['documents']) for r in rows)} "
        f"max={max(len(r['documents']) for r in rows)}")
    print(f"{'k':>5} {'p10':>7} {'median':>7} {'p90':>7} {'clamped':>8}")
    out = {}
    for k in ks:
        lens, clamped = [], 0
        for i, ex in enumerate(rows):
            if len(ex["documents"]) < k:
                clamped += 1
            s = shrink(ex, k, random.Random(SHRINK_SEED + i))
            if s is None:
                continue
            prompt, answer = build_prompt(s, task=TASK, use_alpaca=False,
                                          query_position=query_position)
            msgs = [{"role": "user", "content": prompt}]
            full = tok.apply_chat_template(
                msgs + [{"role": "assistant", "content": answer}], tokenize=False,
                add_generation_prompt=False)
            lens.append(len(tok(full, add_special_tokens=False)["input_ids"]) + 1)  # +1 EOS
        lens.sort()
        q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]  # noqa: E731
        out[k] = {"p10": q(.10), "median": q(.5), "p90": q(.90),
                  "clamped_frac": round(clamped / len(rows), 3)}
        print(f"{k:>5} {q(.10):>7} {q(.5):>7} {q(.90):>7} {out[k]['clamped_frac']:>8}")
    return out


# --------------------------------------------------------------------------- build
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/data/prasann/nq_lengthmix")
    ap.add_argument("--train-src")
    ap.add_argument("--eval-src", required=True)
    ap.add_argument("--k2k", type=int, default=11)
    ap.add_argument("--k8k", type=int, default=48)
    ap.add_argument("--pool-2k", type=int, default=20300, help="incl. the held-out split")
    ap.add_argument("--pool-8k", type=int, default=8300)
    ap.add_argument("--heldout", type=int, default=300)
    ap.add_argument("--eval-size", type=int, default=600)
    ap.add_argument("--query-position", default="after")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--calibrate", default="", help="comma-separated k values; calibrate and exit")
    ap.add_argument("--calibrate-sample", type=int, default=60)
    args = ap.parse_args()

    if args.calibrate:
        ks = [int(x) for x in args.calibrate.split(",")]
        res = calibrate(args.eval_src, ks, args.tokenizer, args.calibrate_sample,
                        args.query_position)
        print(json.dumps(res, indent=2))
        return

    assert args.train_src, "--train-src is required for a build"
    work = pathlib.Path(args.work)
    (work / "eval_rungs" / "nq").mkdir(parents=True, exist_ok=True)
    (work / "tmp").mkdir(exist_ok=True)
    report = {"task": TASK, "source_tag": SRC_TAG, "k_map": {"2048": args.k2k, "8192": args.k8k},
              "query_position": args.query_position,
              "shrink": "keep all gold + random distractor subset, clamp if row has < k docs "
                        "(identical to build_v2_eval_ladders.py shrink mode)"}
    specs = [("n2k", args.k2k, args.pool_2k, "2048"), ("n8k", args.k8k, args.pool_8k, "8192")]

    # ---------------- pass 1: stream the train source ONCE, emit v1 (+ v2 for ~10% of rows)
    t0 = time.time()
    fh = {t: {"v1": open(work / "tmp" / f"{t}_v1.jsonl", "w"),
              "v2": open(work / "tmp" / f"{t}_v2.jsonl", "w")} for t, *_ in specs}
    n_rows = hard = ndocs = n_nogold = 0
    counts = {t: {"v1": 0, "v2": 0, "clamped": 0} for t, *_ in specs}
    train_queries = set()
    log(f"--- pass 1 over {args.train_src}")
    with open(args.train_src) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            ex = json.loads(line)
            n_rows += 1
            hard += len(ex.get("hard_neg_indices") or [])
            ndocs += len(ex["documents"])
            if not [g for g in (ex.get("gold_doc_indices") or []) if 0 <= g < len(ex["documents"])]:
                n_nogold += 1
                continue
            train_queries.add(ex["queries"][0])
            for tag, k, _, _ in specs:
                if len(ex["documents"]) < k:
                    counts[tag]["clamped"] += 1
                s = shrink(ex, k, random.Random(SHRINK_SEED + k * 1_000_003 + i))
                fh[tag]["v1"].write(f"{i}\t{json.dumps(s)}\n")
                counts[tag]["v1"] += 1
                if i % DUP_MOD == DUP_REM:
                    s2 = shrink(ex, k, random.Random(SHRINK_SEED2 + k * 7919 + i))
                    fh[tag]["v2"].write(f"{i}\t{json.dumps(s2)}\n")
                    counts[tag]["v2"] += 1
            if n_rows in (1, 2, 5, 10) or n_rows % 2000 == 0:
                el = time.time() - t0
                log(f"    row {n_rows}  ({el:.1f}s elapsed, "
                    f"{el / n_rows * max(0, 20000 - n_rows):.0f}s ETA to ~20k)")
    for d in fh.values():
        for h in d.values():
            h.close()
    report["train_source"] = {
        "path": args.train_src, "rows": n_rows, "usable_rows": len(train_queries),
        "rows_without_gold": n_nogold, "hard_neg_ratio": round(hard / max(ndocs, 1), 4),
        "mean_docs_per_row": round(ndocs / max(n_rows, 1), 1),
        "unique_queries": len(train_queries), "pass_seconds": round(time.time() - t0, 1)}
    log(f"    train source: {json.dumps(report['train_source'])}")

    # ---------------- eval source (small) ----------------
    ev_rows, ev_queries = [], set()
    e_hard = e_ndocs = 0
    with open(args.eval_src) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            e_hard += len(ex.get("hard_neg_indices") or [])
            e_ndocs += len(ex["documents"])
            if not [g for g in (ex.get("gold_doc_indices") or []) if 0 <= g < len(ex["documents"])]:
                continue
            ev_rows.append(ex)
            ev_queries.add(ex["queries"][0])
    report["eval_source"] = {"path": args.eval_src, "rows": len(ev_rows),
                             "hard_neg_ratio": round(e_hard / max(e_ndocs, 1), 4),
                             "unique_queries": len(ev_queries),
                             "mean_docs_per_row": round(e_ndocs / max(len(ev_rows), 1), 1)}
    log(f"    eval source: {json.dumps(report['eval_source'])}")

    # ---------------- gates ----------------
    for key in ("train_source", "eval_source"):
        r = report[key]["hard_neg_ratio"]
        if not (0.05 <= r <= 0.20):
            raise SystemExit(f"!!! {key} hard_neg_ratio={r} is NOT the p10 regime -- refusing")
    log(f"p10 gate PASSED: train hr={report['train_source']['hard_neg_ratio']}, "
        f"eval hr={report['eval_source']['hard_neg_ratio']}")
    inter = train_queries & ev_queries
    report["query_overlap"] = {"train_queries": len(train_queries),
                               "eval_queries": len(ev_queries), "shared_queries": len(inter)}
    log(f"query overlap: {json.dumps(report['query_overlap'])}")
    if inter:
        raise SystemExit(f"!!! {len(inter)} queries shared between train and eval -- refusing "
                         f"(e.g. {list(inter)[:2]})")

    # ---------------- pools: shuffle v1, carve held-out, top up from v2 ----------------
    report["pools"] = {}
    for tag, k, target, _ in specs:
        v1 = (work / "tmp" / f"{tag}_v1.jsonl").read_text().splitlines()
        random.Random(31337 + k).shuffle(v1)
        heldout = v1[-args.heldout:]
        held_rows = {ln.split("\t", 1)[0] for ln in heldout}
        train_lines = v1[: -args.heldout]
        want = target - args.heldout
        reused = 0
        if len(train_lines) < want:
            v2 = [ln for ln in (work / "tmp" / f"{tag}_v2.jsonl").read_text().splitlines()
                  if ln.split("\t", 1)[0] not in held_rows]
            random.Random(51337 + k).shuffle(v2)
            take = min(want - len(train_lines), len(v2))
            train_lines += v2[:take]
            reused = take
        ceiling = len(train_lines)
        if ceiling < want:
            log(f"!!! {tag}: ceiling {ceiling} < requested {want} (reported, not faked)")
        strip = lambda ls: "\n".join(ln.split("\t", 1)[1] for ln in ls) + "\n"  # noqa: E731
        (work / f"nq_{tag}_train.jsonl").write_text(strip(train_lines))
        (work / f"nq_{tag}_heldout.jsonl").write_text(strip(heldout))
        report["pools"][tag] = {
            "k": k, "train_examples": len(train_lines), "heldout_examples": len(heldout),
            "unique_queries_in_train": len(train_lines) - reused,
            "reused_queries": reused,
            "unit_reuse_factor": round(len(train_lines) / max(len(train_lines) - reused, 1), 4),
            "rows_clamped_below_k": counts[tag]["clamped"],
            "clamped_frac": round(counts[tag]["clamped"] / max(counts[tag]["v1"], 1), 4),
            "ceiling_unique": counts[tag]["v1"] - args.heldout,
            "heldout_disjoint_from_train": True}
        log(f"pool {tag} (k={k}): train={len(train_lines)} (reused {reused}) "
            f"heldout={len(heldout)} clamped={report['pools'][tag]['clamped_frac']}")

    # ---------------- held-out eval rungs (validation split only) ----------------
    report["eval_rungs"] = {}
    for tag, k, _, label in specs:
        out = work / "eval_rungs" / "nq" / f"rung_{label}.jsonl"
        rows = ev_rows[: args.eval_size]
        clamped = sum(1 for ex in rows if len(ex["documents"]) < k)
        with open(out, "w") as f:
            for i, ex in enumerate(rows):
                s = shrink(ex, k, random.Random(SHRINK_SEED + k * 1_000_003 + 900000 + i))
                f.write(json.dumps(s) + "\n")
        report["eval_rungs"][label] = {"file": str(out), "k": k, "eval_size": len(rows),
                                       "rows_clamped_below_k": clamped,
                                       "disjoint_from_train": True}
        log(f"eval rung {label} (k={k}): {len(rows)} examples, {clamped} clamped -> {out}")

    (work / "BUILD_REPORT.json").write_text(json.dumps(report, indent=2))
    log("=== BUILD_REPORT ===")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
