"""Sanity-check the mathmatch template CoT: render targets on real data and
confirm the eval pair-parser recovers exactly the gold pairs from the CoT."""
import json

from corpus_reasoning.lib.data_format import build_prompt
from corpus_reasoning.eval.evaluate import parse_pairs

SRC = "data/mathmatch_n100_k3_x4_r-500to500_numsonly_eval.jsonl"

rows = [json.loads(l) for l in open(SRC)][:5]
ok = 0
for i, ex in enumerate(rows):
    prompt, output = build_prompt(ex, task="mathmatch", query_position="both",
                                  cot_mode="template")
    gold = sorted(sorted(p) for p in ex["gold_doc_indices"])
    parsed = parse_pairs(output)
    parsed_sorted = sorted(sorted(p) for p in parsed) if parsed else parsed
    match = parsed_sorted == gold
    ok += match
    if i < 2:
        print("=" * 70)
        print("TARGET OUTPUT:\n" + output)
        print(f"gold={gold}")
        print(f"parser recovered={parsed_sorted}  MATCH={match}")
print("=" * 70)
print(f"parse-safe on {ok}/{len(rows)} examples")
