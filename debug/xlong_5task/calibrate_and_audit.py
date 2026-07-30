"""Qwen3.5 token calibration + known-bad-variant audit for the 5-task 2k->256k build.

Two jobs in one pass over each source pool:

1. **Calibration.** The existing eval rungs' ``n`` values were fitted with the *Qwen3* tokenizer
   (``eval500_v2/_xlong_build.log``: contra 40.91 tok/doc, nq 157.15, outlier 146.25, rerank 84.39).
   Qwen3.5 has a 248k vocab and compresses differently, so those ``n`` do **not** land on their
   labeled token lengths under Qwen3.5. Re-fit ``tokens ~ a + b * n_docs`` against the real rendered
   prompt (the same ``build_prompt`` the converter and the native eval harness call) and emit the
   ``n`` needed for every target band edge.

2. **Audit.** Refuse to build on a known-bad pool variant. Checks, per the trap list:
   - nq: hard-negative ratio must be ~0.10 (the 98%-hard hn49/hn99/hn199/ladder64k build is banned).
   - contradiction: gold indices 1-indexed and in range (the ``build_v2`` off-by-one that scored 0).
   - outlier: gold indices in range (the shrink-breaks-scale-K bug).
   - rerank: ``ce_scores`` present (CE-filtered pool, not a raw BM25 dump).

Writes ``calibration.json`` next to this file. Run on a build node (node-local conda + HF cache):

    srun -p jsteinhardt --qos=preemptive_high --nodelist=cubbins --cpus-per-task=8 --mem=64G \\
      --time=00:40:00 bash -c '... python debug/xlong_5task/calibrate_and_audit.py'
"""

import argparse
import json
import os
import random
import sys

TOKENIZER = "/scratch/users/prasann/hf_models/Qwen3.5-4B-Base"

#: task key -> (build_prompt task name, pool path)
POOLS = {
    "contradiction": (
        "contradiction",
        "/data/prasann/ctc_suite_data/contradiction_pool/"
        "contradiction_train_pubmed_realistic_n50-950_k3.jsonl",
    ),
    "outlier": (
        "outlier",
        "/data/prasann/ctc_suite_data/outlier_pool/outlier_wiki100w_contin_n14-220_k3_20000.jsonl",
    ),
    "rerank": (
        "rerank",
        "/data/prasann/ctc_suite_data/rerank_pool/msmarco_trainhn_train_k20-315_20000.jsonl",
    ),
    "nq": ("retrieval", "/scratch/users/prasann/nq_p10_20k/nq_train_k25-202_clean.jsonl"),
}

#: The context-length ladder we are building data for.
BANDS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

#: Index base of ``gold_doc_indices``, PER TASK. Not uniform across the suite -- see the audit note.
GOLD_INDEX_BASE = {"contradiction": 1, "outlier": 0, "rerank": 0, "nq": 0}


def reservoir_sample(path: str, k: int, seed: int = 0) -> list:
    """Uniform sample of ``k`` JSON lines without loading the (multi-GB) file."""
    rng = random.Random(seed)
    out: list = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if len(out) < k:
                out.append(line)
            else:
                j = rng.randint(0, i)
                if j < k:
                    out[j] = line
    return [json.loads(x) for x in out]


def audit(task: str, examples: list) -> dict:
    """Known-bad-variant checks. Returns {check: verdict}; 'FAIL' anywhere blocks the build."""
    res: dict = {}

    if task == "nq":
        ratios = []
        for ex in examples:
            k = len(ex.get("documents", []))
            hn = len(ex.get("hard_neg_indices", []) or [])
            if k:
                ratios.append(hn / k)
        if not ratios:
            res["hard_neg_ratio"] = "FAIL: no documents/hard_neg_indices"
        else:
            mean = sum(ratios) / len(ratios)
            # p10 pipeline ~= 0.10. The banned 98%-hard build sits at ~0.99.
            ok = 0.02 <= mean <= 0.25
            res["hard_neg_ratio"] = (
                f"{'PASS' if ok else 'FAIL'}: mean={mean:.3f} "
                f"min={min(ratios):.3f} max={max(ratios):.3f} (want ~0.10; 98%-hard build is BANNED)"
            )

    if task in GOLD_INDEX_BASE:
        # ⚠ The index base is PER TASK, verified against both the pools and the renderer:
        #   * outlier / rerank / nq  -> 0-indexed; the rendered answer adds 1
        #     (outlier golds [43,64,72] <-> answer "44; 65; 73").
        #   * contradiction          -> 1-indexed; data_format.py:1083 says "These are already
        #     1-indexed claim IDs" and json.dumps()es the pairs verbatim.
        # Applying the wrong base is exactly the off-by-one that scored contradiction at 0 in the
        # v2 eval build, so check each against its own convention.
        base = GOLD_INDEX_BASE[task]
        lo, hi_off = (base, base - 1)  # valid range is [base, n - 1 + base]
        bad_range = 0
        checked = 0
        observed_min, observed_max = 10**9, -1
        for ex in examples:
            n = len(ex.get("documents", []))
            golds = ex.get("gold_doc_indices")
            if not golds or not n:
                continue
            flat: list = []
            for g in golds:
                flat.extend(g if isinstance(g, (list, tuple)) else [g])
            if not flat:
                continue
            checked += 1
            observed_min = min(observed_min, min(flat))
            observed_max = max(observed_max, max(flat))
            if min(flat) < lo or max(flat) > n + hi_off:
                bad_range += 1
        res["gold_indices"] = (
            f"{'PASS' if bad_range == 0 and checked else 'FAIL'}: checked={checked} "
            f"out_of_range={bad_range} observed=[{observed_min},{observed_max}] "
            f"-- {base}-indexed, valid [{lo}, n_docs{hi_off:+d}]"
        )

    if task == "rerank":
        with_ce = sum(1 for ex in examples if ex.get("ce_scores"))
        res["ce_scores"] = (
            f"{'PASS' if with_ce > 0.9 * len(examples) else 'FAIL'}: "
            f"{with_ce}/{len(examples)} examples carry ce_scores (CE-filtered pool required)"
        )

    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400, help="examples sampled per task")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "calibration.json"))
    ap.add_argument("--tasks", nargs="*", default=sorted(POOLS))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from corpus_reasoning.lib.data_format import build_prompt

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    print(f"tokenizer={TOKENIZER} vocab={len(tok)}", flush=True)

    report: dict = {"tokenizer": TOKENIZER, "bands": BANDS, "tasks": {}}
    blocked = False

    for key in args.tasks:
        prompt_task, path = POOLS[key]
        if not os.path.exists(path):
            print(f"[{key}] MISSING POOL {path}", flush=True)
            report["tasks"][key] = {"error": f"missing pool {path}"}
            blocked = True
            continue

        print(f"\n=== [{key}] {path} ===", flush=True)
        examples = reservoir_sample(path, args.sample, seed=17)
        print(f"  sampled {len(examples)}", flush=True)

        checks = audit(key, examples)
        for name, verdict in checks.items():
            print(f"  AUDIT {name}: {verdict}", flush=True)
            if verdict.startswith("FAIL"):
                blocked = True

        pts = []
        for ex in examples:
            n = len(ex.get("documents", []))
            if not n:
                continue
            try:
                prompt, _ans = build_prompt(
                    ex,
                    task=prompt_task,
                    query_position="both",
                    use_alpaca=False,
                    cot_mode="none",
                    use_titles=False,
                )
            except Exception as e:  # noqa: BLE001 - surface, don't crash the whole sweep
                print(f"  build_prompt failed on an example: {e}", flush=True)
                continue
            pts.append((n, len(tok(prompt, add_special_tokens=False)["input_ids"])))

        if len(pts) < 10:
            report["tasks"][key] = {"error": f"only {len(pts)} usable points", "audit": checks}
            blocked = True
            continue

        # Least-squares fit tokens = a + b * n_docs.
        m = len(pts)
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        sxx = sum(p[0] * p[0] for p in pts)
        sxy = sum(p[0] * p[1] for p in pts)
        denom = m * sxx - sx * sx
        b = (m * sxy - sx * sy) / denom
        a = (sy - b * sx) / m
        resid = [abs(y - (a + b * x)) for x, y in pts]
        mape = sum(r / y for r, (_x, y) in zip(resid, pts)) / m

        n_for_band = {
            str(t): max(1, int(round((t - a) / b))) for t in BANDS if (t - a) / b >= 1
        }
        ns = [p[0] for p in pts]
        report["tasks"][key] = {
            "pool": path,
            "audit": checks,
            "fit": {"intercept": round(a, 2), "tok_per_doc": round(b, 4), "mape": round(mape, 4)},
            "pool_n_docs": {"min": min(ns), "max": max(ns), "median": sorted(ns)[len(ns) // 2]},
            "n_for_band": n_for_band,
        }
        print(
            f"  FIT tokens = {a:.1f} + {b:.3f} * n_docs   (MAPE {mape * 100:.1f}%, {m} pts)",
            flush=True,
        )
        print(f"  pool n_docs: min={min(ns)} med={sorted(ns)[len(ns) // 2]} max={max(ns)}", flush=True)
        print(f"  n per band: {n_for_band}", flush=True)

    report["blocked"] = blocked
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}  blocked={blocked}", flush=True)
    sys.exit(1 if blocked else 0)


if __name__ == "__main__":
    main()
