"""Build gold-preserving token-budget eval rungs for OBLIQ (obliq_retrieval).

Why a custom builder (not build_v2_eval_ladders): OBLIQ's subsets have wildly
different per-doc lengths (a tweet ~30 tok, a congressional record ~1000 tok) AND
it is multi-gold (1-64 golds/example, gold footprints span ~290 to ~47k tokens).
A uniform doc-COUNT rung is therefore not a uniform token budget, and small rungs
cannot hold high-gold examples. So we ladder by a PER-EXAMPLE TOKEN BUDGET:

  per (example, budget T): keep ALL golds (never dropped), then add distractors
  in a fixed shuffled order until the next distractor would exceed T tokens;
  remap gold_doc_indices / hard_neg_indices; shuffle the kept set.

Nested-prefix: the distractor order is fixed per example across rungs, so a
shorter rung's distractors are a subset of a longer rung's; golds/query are
byte-identical across rungs. An example whose golds ALONE exceed T is
GOLD-SATURATED -> it is DROPPED from that rung (keeping it would truncate a gold
and make it unanswerable). Rungs where <500 examples survive are flagged.

Viability (measured): only 16k (99%) and 32k (100%) preserve golds for ~all
examples; 2k/4k/8k drop 11-51% and are ill-posed, so this builder emits ONLY the
rungs you pass that clear a survivor threshold. Token budget is a word*1.3 + per-
doc marker-overhead estimate (matches the calibration used by the other rungs).

Usage:
    python build_obliq_token_ladder.py \\
        --eval /scratch/users/prasann/corpus-reasoning/data/obliq_mix4_test_488.jsonl \\
        --rungs 16384 32768 --out-dir /scratch/.../eval_rungs/obliq_retrieval
"""
import argparse
import json
import random
from pathlib import Path

MARKER_OVER = 8  # ~per-doc marker/wrap overhead tokens (doc_start/doc_end/etc.)


def wtok(t):
    return int(len(t.split()) * 1.3) + MARKER_OVER


def build_rung(rows, T, seed, keep_fitting=False):
    """Return (examples, n_dropped) for token budget T. Distractor order is fixed
    per example (seeded by index) so rungs nest.

    keep_fitting=False (strict, for 16k/32k): keep ALL golds or drop the example.
    keep_fitting=True (for small rungs 2k/4k where a multi-gold example's golds
    exceed T): keep the SUBSET of golds that fit (smallest-first) + distractors;
    the eval gold set becomes that fitting subset. An example is dropped only if
    ZERO golds fit."""
    out, dropped = [], 0
    for ei, r in enumerate(rows):
        docs = r["documents"]
        gi = set(r.get("gold_doc_indices", []))
        hn = set(r.get("hard_neg_indices", []))
        qtok = wtok(r["queries"][0])
        gold_tok = sum(wtok(docs[i]["text"]) for i in gi)
        if qtok + gold_tok > T:
            if not keep_fitting:
                dropped += 1  # gold-saturated: cannot keep all golds under budget
                continue
            # keep the golds that fit (smallest first); drop only if none fit
            budget = qtok
            kept_g = []
            for i in sorted(gi, key=lambda i: wtok(docs[i]["text"])):
                if budget + wtok(docs[i]["text"]) <= T:
                    budget += wtok(docs[i]["text"]); kept_g.append(i)
            if not kept_g:
                dropped += 1
                continue
            gi = set(kept_g)
            gold_tok = sum(wtok(docs[i]["text"]) for i in gi)
        # fixed distractor order (hard negs first, then others), seeded per example
        rng = random.Random(seed * 100003 + ei)
        hard = [i for i in range(len(docs)) if i in hn and i not in gi]
        other = [i for i in range(len(docs)) if i not in hn and i not in gi]
        rng.shuffle(hard); rng.shuffle(other)
        order = hard + other
        budget = qtok + gold_tok
        keep_dist = []
        for i in order:
            dt = wtok(docs[i]["text"])
            if budget + dt > T:
                continue  # skip this one, try smaller later ones (nested prefix)
            budget += dt
            keep_dist.append(i)
        chosen = [(i, "g") for i in gi] + [(i, "d") for i in keep_dist]
        rng.shuffle(chosen)
        new = dict(r)
        new["documents"] = [docs[i] for i, _ in chosen]
        new["gold_doc_indices"] = [j for j, (_, t) in enumerate(chosen) if t == "g"]
        new["hard_neg_indices"] = [j for j, (_, t) in enumerate(chosen) if t == "d"
                                   and chosen[j][0] in hn]
        out.append(new)
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--rungs", nargs="+", type=int, default=[16384, 32768])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-survivors", type=int, default=460,
                    help="skip a rung if fewer examples survive (ill-posed)")
    ap.add_argument("--keep-fitting", action="store_true",
                    help="for small rungs: keep the SUBSET of golds that fit "
                         "(not all-or-nothing) so multi-gold examples stay answerable")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.eval)]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"source: {len(rows)} examples  (keep_fitting={args.keep_fitting})")
    for T in sorted(args.rungs):
        ex, dropped = build_rung(rows, T, args.seed, keep_fitting=args.keep_fitting)
        tag = "OK" if len(ex) >= args.min_survivors else "SKIP (ill-posed, <min)"
        print(f"rung {T}: {len(ex)} survivors, {dropped} gold-saturated dropped  -> {tag}")
        if len(ex) < args.min_survivors:
            continue
        p = out / f"rung_{T}.jsonl"
        with open(p, "w") as f:
            for e in ex:
                f.write(json.dumps(e) + "\n")
        print(f"   wrote {p}")


if __name__ == "__main__":
    main()
