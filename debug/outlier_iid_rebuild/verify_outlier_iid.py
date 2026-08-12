"""Check a rebuilt outlier training file against the v2 scale-K eval ladder.

The eval targets below were measured off
`s3://.../_eval_bundle_eval500_v2/outlier/outlier_wiki100w_n{22,55,110,220}_k3_eval_600.jsonl`
(truncated reads: 208/83/41/20 examples per rung). K is `meta.num_categories`; docs-per-majority
is derived as (n - num_outliers) / (K - 1).

The old training file measured K = 2-10 (median 6.5 at n~220) with ~40 docs per majority topic,
against the eval's K = 25 and ~9 docs. A rebuild is only iid if BOTH move onto the eval's values.
"""

import argparse
import json
import statistics

# rung -> (K median, K min, K max) measured on the eval ladder
EVAL_K = {22: (3, 3, 5), 55: (7, 5, 11), 110: (13, 10, 17), 220: (25, 23, 28)}
BANDS = {22: (14, 30), 55: (45, 70), 110: (95, 125), 220: (190, 220)}


def load(path, limit=None):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="rebuilt outlier training jsonl")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = load(args.train, limit=args.limit)
    print(f"[verify] {len(recs)} examples from {args.train}", flush=True)

    report = {"train_file": args.train, "examples_sampled": len(recs), "rungs": {}}
    ok = True
    print(f"{'rung':<6} {'TRAIN K med (min-max)':<26} {'EVAL K med (min-max)':<24} "
          f"{'train docs/maj':<15} {'eval docs/maj':<14} verdict")
    for n, (lo, hi) in BANDS.items():
        ks, dpm = [], []
        for r in recs:
            m = r.get("meta") or {}
            nd = len(r["documents"])
            k = m.get("num_categories")
            if not (lo <= nd <= hi) or not k:
                continue
            ks.append(k)
            if k > 1:
                dpm.append((nd - m.get("num_outliers", 3)) / (k - 1))
        ek, ekmin, ekmax = EVAL_K[n]
        eval_dpm = (n - 3) / (ek - 1)
        if not ks:
            print(f"{n:<6} {'(no samples in band)':<26}")
            report["rungs"][n] = {"samples": 0}
            ok = False
            continue
        km = statistics.median(ks)
        dm = statistics.median(dpm) if dpm else float("nan")
        # in-band if the train median K falls inside the eval's observed K range
        verdict = "OK" if ekmin <= km <= ekmax else "MISMATCH"
        if verdict != "OK":
            ok = False
        print(f"{n:<6} {f'{km} ({min(ks)}-{max(ks)})':<26} {f'{ek} ({ekmin}-{ekmax})':<24} "
              f"{dm:<15.1f} {eval_dpm:<14.1f} {verdict}")
        report["rungs"][n] = {
            "samples": len(ks), "train_k_median": km,
            "train_k_min": min(ks), "train_k_max": max(ks),
            "eval_k_median": ek, "eval_k_range": [ekmin, ekmax],
            "train_docs_per_majority": round(dm, 2),
            "eval_docs_per_majority": round(eval_dpm, 2),
            "verdict": verdict,
        }

    report["all_rungs_in_eval_range"] = ok
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[verify] all rungs in eval K range: {ok}", flush=True)
    print(f"[verify] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
