#!/usr/bin/env python
"""Test whether the 32k 'collapse' is a 4-digit ID truncation rather than a retrieval failure.

The sampled collapsed generations all show the same shape: the right document is found and then
the LAST DIGIT of a 4-digit id is dropped (gold [83, 1170] -> emitted [83, 117]; [692, 1409] ->
[692, 140]; [54, 1136] -> [54, 113]). If that is systematic it reframes the whole result, because
the 32k rung is the ONLY one whose corpus exceeds 1000 documents -- so it is the only rung where
ids are 4 digits, which is exactly why 2k and 8k look healthy on a broken checkpoint.

Six hand-picked examples cannot establish that, so this measures it on all 500:

  1. DIGIT HISTOGRAM. Fraction of emitted ids with 1/2/3/4 digits vs gold. A model that cannot
     emit 4-digit ids shows a deficit at 4 and a surplus at 3, with no reference to any hypothesis
     about which ids are correct.
  2. PREFIX RATE. Of the gold ids the model MISSED, how many were emitted with their last digit
     removed? A retrieval failure has no reason to produce prefixes of the right answer; a digit
     bug produces almost nothing else. Compared against a null: how often a prefix relationship
     arises by chance between a missed gold id and an unrelated emitted id.
  3. REPAIR TEST. Re-score the eval after mapping every emitted 3-digit id to the gold 4-digit id
     it is a prefix of, when that is unambiguous. If f1 jumps from 0.25 to near the healthy run's,
     the retrieval was always there and only the rendering was broken.

(3) is the decisive one: it says whether the model's long-context ability was ever actually lost.
"""
import argparse
import json
import os
import random
from collections import Counter


def load(responses_path, rung_jsonl, max_samples=100000):
    from corpus_reasoning.eval.evaluate import _eval_contradiction, load_unified_examples

    pack = json.load(open(responses_path))
    examples = load_unified_examples(
        rung_jsonl, max_samples, task="contradiction",
        query_position="both", use_alpaca=True,
    )
    responses = [pack["responses"][str(i)] for i in range(len(examples))]
    metrics, details = _eval_contradiction(examples, responses)
    return metrics, details


def digits(n):
    return len(str(int(n)))


def analyze(tag, details, rng):
    pred_digits, gold_digits = Counter(), Counter()
    n_missed = n_prefix = 0
    n_null = 0                      # chance-level prefix hits, same counts, shuffled ids
    all_pred_ids = []
    for d in details:
        p_ids = [i for p in d["predicted_pairs"] for i in p]
        g_ids = [i for g in d["gold_pairs"] for i in g]
        all_pred_ids += p_ids
        for i in p_ids:
            pred_digits[digits(i)] += 1
        for i in g_ids:
            gold_digits[digits(i)] += 1
        # A gold id is "missed" if it never appears verbatim in the prediction.
        missed = [g for g in g_ids if g not in p_ids]
        n_missed += len(missed)
        pset = set(p_ids)
        for g in missed:
            if g >= 10 and (g // 10) in pset:
                n_prefix += 1
    # null model: are prefix hits just a consequence of many small ids being emitted?
    pool = list({i for i in all_pred_ids})
    for d in details:
        g_ids = [i for g in d["gold_pairs"] for i in g]
        p_ids = [i for p in d["predicted_pairs"] for i in p]
        fake = set(rng.sample(pool, min(len(p_ids), len(pool))))
        for g in g_ids:
            if g not in p_ids and g >= 10 and (g // 10) in fake:
                n_null += 1

    tp, tg = sum(pred_digits.values()), sum(gold_digits.values())
    print(f"\n  [{tag}]")
    print("     digit histogram  " + "  ".join(
        f"{k}d pred {pred_digits[k]/tp:5.1%} gold {gold_digits[k]/tg:5.1%}"
        for k in sorted(set(pred_digits) | set(gold_digits))))
    if n_missed:
        print(f"     missed gold ids: {n_missed}    of which emitted as last-digit-dropped "
              f"prefix: {n_prefix} ({n_prefix/n_missed:.1%})   chance level: {n_null/n_missed:.1%}")
    return {"pred_digits": dict(pred_digits), "gold_digits": dict(gold_digits),
            "n_missed": n_missed, "n_prefix": n_prefix, "n_null": n_null}


def repair_score(details):
    """Re-score after undoing the truncation, to see if the retrieval was there all along.

    A predicted id is repaired to a gold id when the gold id is 4-digit, the prediction is its
    3-digit prefix, and NO other gold id in the example shares that prefix (so the repair is
    forced, not chosen to be flattering).
    """
    tot_p = tot_r = tot_f = 0.0
    for d in details:
        g_ids = [i for g in d["gold_pairs"] for i in g]
        fix = {}
        for g in g_ids:
            if g >= 1000:
                pre = g // 10
                if sum(1 for o in g_ids if o >= 1000 and o // 10 == pre) == 1:
                    fix[pre] = g
        pred = [tuple(sorted((fix.get(a, a), fix.get(b, b)))) for a, b in d["predicted_pairs"]]
        gold = {tuple(sorted(g)) for g in d["gold_pairs"]}
        pred = set(pred)
        inter = len(pred & gold)
        p = inter / len(pred) if pred else 0.0
        r = inter / len(gold) if gold else 0.0
        tot_p += p
        tot_r += r
        tot_f += (2 * p * r / (p + r)) if (p + r) else 0.0
    n = len(details)
    return tot_p / n, tot_r / n, tot_f / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-results", required=True)
    ap.add_argument("--rungs-dir", required=True)
    ap.add_argument("--arms", default="A4rr,A4s2,C3rr,C3s2,A4e")
    ap.add_argument("--rung", type=int, default=32768)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(0)
    jl = os.path.join(args.rungs_dir, f"rung_{args.rung}.jsonl")
    report = {}
    for arm in args.arms.split(","):
        p = os.path.join(args.eval_results, f"{arm}_rung{args.rung}.responses.json")
        if not os.path.exists(p):
            print(f"\n  [{arm}] SKIP -- no responses at {p}")
            continue
        m, d = load(p, jl)
        r = analyze(f"{arm}  f1={m['f1']:.3f}", d, rng)
        pp, rr, ff = repair_score(d)
        print(f"     REPAIRED (undo last-digit truncation): f1 {m['f1']:.3f} -> {ff:.3f}   "
              f"(P {pp:.3f} R {rr:.3f})")
        r["f1"] = m["f1"]
        r["f1_repaired"] = ff
        report[arm] = r

    print("\n=== summary: does undoing the truncation recover the collapse? ===")
    for arm, r in report.items():
        print(f"   {arm:6s} f1 {r['f1']:.3f} -> repaired {r['f1_repaired']:.3f}  "
              f"(+{r['f1_repaired']-r['f1']:+.3f})")

    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
