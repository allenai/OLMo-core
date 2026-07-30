"""Tune the CE gold-quality filter on the EXISTING nq_validation_k20 file.

For each example, CE-score (question, gold) and (question, each hard neg).
Report: gold-is-top rate, score distributions, keep-rate under several
(min, margin) settings, and the decision on a few known-bad golds.
"""
import json, sys
# sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from corpus_reasoning.lib.cross_encoder import CrossEncoderScorer

PATH = "data/nq_validation_k20_hn19_500.jsonl"
N = 200  # subset for speed
KNOWN_BAD = [
    "who played bat masterson in the movie tombstone",
    "what is the salary of us secretary of state",
    "who was the winner of the first indianapolis 500",
    "how many times have the winter olympics been in the usa since 1924",
]
SETTINGS = [(0.0, 0.0), (0.0, 1.0), (2.0, 0.0), (0.0, 2.0), (-2.0, 0.0)]

rows = [json.loads(l) for l in open(PATH)][:N]
scorer = CrossEncoderScorer()

pairs, spans = [], []
for ex in rows:
    docs = ex["documents"]; q = ex["queries"][0]
    gi = ex["gold_doc_indices"][0]
    hns = ex.get("hard_neg_indices", [])
    s = len(pairs); pairs.append((q, docs[gi]["text"]))
    for h in hns:
        pairs.append((q, docs[h]["text"]))
    spans.append((s, len(hns)))
scores = scorer.score(pairs)

gold_s, best_hn_s, is_top = [], [], []
for (s, nh) in spans:
    g = scores[s]; hn = scores[s+1:s+1+nh]
    bh = max(hn) if hn else float("-inf")
    gold_s.append(g); best_hn_s.append(bh); is_top.append(g >= bh)

def pct(xs, p):
    xs = sorted(xs); return xs[int(p/100*(len(xs)-1))]

print(f"=== CE tune on {PATH} (first {len(rows)}) ===")
print(f"gold-is-top-vs-hardnegs: {sum(is_top)}/{len(rows)} ({100*sum(is_top)/len(rows):.1f}%)")
print(f"gold score    p10/p50/p90: {pct(gold_s,10):.2f}/{pct(gold_s,50):.2f}/{pct(gold_s,90):.2f}")
print(f"best-hardneg  p10/p50/p90: {pct(best_hn_s,10):.2f}/{pct(best_hn_s,50):.2f}/{pct(best_hn_s,90):.2f}")
print("\nkeep-rate by (ce_gold_min, ce_margin):")
for mn, mg in SETTINGS:
    keep = sum(1 for g, bh in zip(gold_s, best_hn_s) if g >= mn and g >= bh + mg)
    print(f"  min={mn:+.1f} margin={mg:.1f} -> keep {keep}/{len(rows)} ({100*keep/len(rows):.1f}%)")

print("\nknown-bad golds (expect DROP under min=0,margin=0):")
qmap = {ex["queries"][0]: i for i, ex in enumerate(rows)}
for q in KNOWN_BAD:
    if q not in qmap:
        print(f"  [not in subset] {q!r}"); continue
    i = qmap[q]; g, bh = gold_s[i], best_hn_s[i]
    print(f"  gold={g:+.2f} best_hn={bh:+.2f} keep@(0,0)={g>=0 and g>=bh}  {q!r}")
