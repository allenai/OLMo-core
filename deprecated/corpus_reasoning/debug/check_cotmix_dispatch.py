"""Confirm the mixed file dispatches per-row exactly as the olmo tokenizer would:
auto-detect `_task` -> multitask -> per-row `_cot_mode` -> build_prompt target."""
import json

from corpus_reasoning.lib.data_format import build_prompt
from corpus_reasoning.data.tokenize_unified_for_olmo import _is_multitask
from pathlib import Path

SRC = "data/mathmatch_n100_k3_x4_r-500to500_numsonly_cotmix50_train.jsonl"

print(f"_is_multitask -> {_is_multitask(Path(SRC))}")
rows = [json.loads(l) for l in open(SRC)]

cot_ex = next(r for r in rows if r.get("_cot_mode") == "template")
plain_ex = next(r for r in rows if r.get("_cot_mode") == "label")

for label, ex in [("CoT row (_cot_mode=template)", cot_ex),
                  ("PLAIN row (_cot_mode=label)", plain_ex)]:
    # mirror tokenizer multitask dispatch: task/cot from the row tags
    _, output = build_prompt(ex, task=ex["_task"], query_position="both",
                             cot_mode=ex["_cot_mode"])
    print("=" * 70)
    print(label)
    print("TARGET:\n" + output)

n_cot = sum(r.get("_cot_mode") == "template" for r in rows)
print("=" * 70)
print(f"{n_cot}/{len(rows)} rows -> CoT target, {len(rows) - n_cot} -> plain answer")
