#!/usr/bin/env python
"""Diagnose the 32k seed collapse in the length-mix experiment.

Two of five replicated points had one seed collapse at 32k while its twin was healthy
(A4: 0.249 vs 0.585; C3: 0.257 vs 0.528). Both seeds pass 2k and 8k at essentially the
same f1, and BOTH have parse_rate 1.0 -- so this is not a truncation or format failure.
The model emits a well-formed list of pairs and the pairs are simply wrong, but only when
the context is long.

That leaves a small number of distinguishable failure modes, and the point of this script
is to tell them apart rather than guess:

  (a) PRECISION collapse -- it emits many more pairs than gold, carpet-bombing the answer.
  (b) RECALL collapse -- it emits too few pairs, or the right count but misses the gold.
  (c) POSITIONAL truncation -- predicted doc ids concentrate in the early part of the
      context, i.e. the run lost the ability to retrieve from far back. This is the
      hypothesis the 32k-only signature most suggests, and it is testable: compare the
      normalized position (id-1)/(n_docs-1) of predicted ids against gold.
  (d) DEGENERATION -- the same answer repeated across examples (a constant predictor),
      which would show up as very few distinct predictions.

It also reports per-example agreement between the collapsed and healthy seed: if the
collapsed seed is wrong on a random subset, that is noise; if it is wrong specifically on
the examples whose gold pairs sit late in the context, that is (c) again and confirms it
from the other direction.

Run on Beaker where /weka is mounted (the responses live there and are multi-GB).
"""
import argparse
import json
import os
from collections import Counter


def load_details(responses_path, rung_jsonl, max_samples=100000):
    """Grade one responses file and return (metrics, per-example details)."""
    from corpus_reasoning.eval.evaluate import _eval_contradiction, load_unified_examples

    with open(responses_path) as f:
        pack = json.load(f)
    examples = load_unified_examples(
        rung_jsonl, max_samples, task="contradiction",
        query_position="both", use_alpaca=True,
    )
    if len(examples) != pack["eval_size"]:
        raise SystemExit(f"size mismatch {len(examples)} vs {pack['eval_size']} for {responses_path}")
    responses = [pack["responses"][str(i)] for i in range(len(examples))]
    metrics, details = _eval_contradiction(examples, responses)
    # n_docs is what turns a raw doc id into a POSITION in the context; without it the
    # positional hypothesis cannot be tested at all.
    for d, ex in zip(details, examples):
        d["n_docs"] = ex["ex"].get("n_docs") or ex["ex"].get("num_docs")
    return metrics, details


def norm_positions(pairs, n_docs):
    """Normalized [0,1] context positions of every doc id in a list of pairs."""
    if not n_docs or n_docs < 2:
        return []
    return [(i - 1) / (n_docs - 1) for p in pairs for i in p if 1 <= i <= n_docs]


def quartiles(positions):
    """Fraction of positions falling in each context quarter."""
    if not positions:
        return [0.0] * 4
    c = Counter(min(3, int(p * 4)) for p in positions)
    return [c[i] / len(positions) for i in range(4)]


def summarize(tag, metrics, details):
    n = len(details)
    pred_counts = [len(d["predicted_pairs"]) for d in details]
    gold_counts = [len(d["gold_pairs"]) for d in details]
    pred_pos, gold_pos = [], []
    for d in details:
        pred_pos += norm_positions(d["predicted_pairs"], d["n_docs"])
        gold_pos += norm_positions(d["gold_pairs"], d["n_docs"])
    # A constant predictor is the one failure mode that fakes a real f1 distribution.
    distinct = len({json.dumps(sorted(map(sorted, d["predicted_pairs"]))) for d in details})
    print(f"\n  [{tag}]  f1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  "
          f"R={metrics['recall']:.3f}  EM={metrics['exact_match']:.3f}  "
          f"parse={metrics['parse_rate']:.3f}")
    print(f"     pairs/example: pred {sum(pred_counts)/n:.2f} vs gold {sum(gold_counts)/n:.2f}"
          f"   distinct predictions {distinct}/{n}")
    print(f"     pred position quartiles {['%.2f' % q for q in quartiles(pred_pos)]}"
          f"   gold {['%.2f' % q for q in quartiles(gold_pos)]}")
    if pred_pos:
        print(f"     mean pred position {sum(pred_pos)/len(pred_pos):.3f}  "
              f"mean gold position {sum(gold_pos)/len(gold_pos):.3f}"
              if gold_pos else "")
    return {"metrics": metrics, "mean_pred_pairs": sum(pred_counts) / n,
            "mean_gold_pairs": sum(gold_counts) / n, "distinct_predictions": distinct,
            "pred_quartiles": quartiles(pred_pos), "gold_quartiles": quartiles(gold_pos),
            "mean_pred_pos": (sum(pred_pos) / len(pred_pos)) if pred_pos else None,
            "mean_gold_pos": (sum(gold_pos) / len(gold_pos)) if gold_pos else None}


def compare(bad, good):
    """Is the collapsed seed wrong on the LATE-gold examples specifically?

    Split examples by where their gold pairs sit in the context and report each seed's f1
    per half. Positional truncation predicts the collapsed seed's deficit is concentrated
    in the late half; noise predicts it is flat.
    """
    rows = []
    for d_bad, d_good in zip(bad, good):
        gp = norm_positions(d_bad["gold_pairs"], d_bad["n_docs"])
        if not gp:
            continue
        rows.append((sum(gp) / len(gp), d_bad["f1"], d_good["f1"]))
    if not rows:
        print("     (no positional split possible -- n_docs missing)")
        return None
    rows.sort()
    half = len(rows) // 2
    out = {}
    for name, part in (("early-gold", rows[:half]), ("late-gold", rows[half:])):
        fb = sum(r[1] for r in part) / len(part)
        fg = sum(r[2] for r in part) / len(part)
        out[name] = {"collapsed_f1": fb, "healthy_f1": fg, "gap": fg - fb, "n": len(part)}
        print(f"     {name:11s} (eval_size={len(part)}): collapsed f1={fb:.3f}  "
              f"healthy f1={fg:.3f}  gap={fg-fb:+.3f}")
    e, l = out["early-gold"]["gap"], out["late-gold"]["gap"]
    if l > 2 * max(e, 1e-6):
        print("     => gap is concentrated in LATE-gold examples: POSITIONAL failure "
              "(the collapsed run cannot retrieve from far back)")
    elif abs(l - e) < 0.05:
        print("     => gap is FLAT across context position: not positional; the collapsed "
              "run is uniformly worse (a bad optimum, not a lost long-range ability)")
    else:
        print("     => gap is larger for late gold but not decisively (see numbers above)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-results", required=True, help="dir holding <arm>_rung<R>.responses.json")
    ap.add_argument("--rungs-dir", required=True, help="dir holding rung_<R>.jsonl")
    ap.add_argument("--pairs", default="A4:A4s2,C3:C3s2",
                    help="comma list of collapsed:healthy arm pairs")
    ap.add_argument("--rungs", default="2048,8192,32768")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-examples", type=int, default=4,
                    help="print this many collapsed-vs-healthy generations per pair at 32k")
    args = ap.parse_args()

    rungs = [int(r) for r in args.rungs.split(",")]
    report = {}
    for spec in args.pairs.split(","):
        bad_arm, good_arm = spec.split(":")
        print(f"\n{'='*78}\n=== {bad_arm} (collapsed) vs {good_arm} (healthy) ===")
        report[spec] = {}
        for rung in rungs:
            jl = os.path.join(args.rungs_dir, f"rung_{rung}.jsonl")
            pb = os.path.join(args.eval_results, f"{bad_arm}_rung{rung}.responses.json")
            pg = os.path.join(args.eval_results, f"{good_arm}_rung{rung}.responses.json")
            missing = [p for p in (jl, pb, pg) if not os.path.exists(p)]
            if missing:
                print(f"\n--- rung {rung}: SKIP, missing {missing}")
                continue
            print(f"\n--- rung {rung} ---")
            mb, db = load_details(pb, jl)
            mg, dg = load_details(pg, jl)
            r = {"collapsed": summarize(bad_arm, mb, db),
                 "healthy": summarize(good_arm, mg, dg)}
            print(f"\n     positional split (gold-position halves):")
            r["positional_split"] = compare(db, dg)
            report[spec][str(rung)] = r
            if rung == max(rungs) and args.dump_examples:
                print(f"\n     --- sample generations at {rung} "
                      f"(examples where {bad_arm} scored 0 and {good_arm} did not) ---")
                shown = 0
                for i, (x, y) in enumerate(zip(db, dg)):
                    if shown >= args.dump_examples:
                        break
                    if x["f1"] == 0.0 and y["f1"] > 0.5:
                        print(f"     ex{i} n_docs={x['n_docs']} gold={x['gold_pairs']}")
                        print(f"       {bad_arm:6s}: {x['prediction'][:220]!r}")
                        print(f"       {good_arm:6s}: {y['prediction'][:220]!r}")
                        shown += 1
                if shown == 0:
                    print("     (no example where collapsed=0 and healthy>0.5)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
