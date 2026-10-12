import json
import re

EVAL = "/data/prasann/ctc_suite_staged/eval_rungs/niah/rung_2048.jsonl"
RESP = "/data/prasann/ctc_suite_vllm_results/_thinkfix_audit/responses_niah_2048.json"

gold_rows = [json.loads(l) for l in open(EVAL)]
resp = json.load(open(RESP)).get("responses", {})

OLD = re.compile(r"\[(\d+)\]")
NEW = re.compile(r"\[?\s*(\d+)\s*\]")


def score(pat):
    f1s = []
    for i, e in enumerate(gold_rows):
        gi = e["gold_doc_indices"]
        gi = gi[0] if gi and isinstance(gi[0], list) else gi
        gold = set(int(g) + 1 for g in gi)  # 0-idx gold -> 1-idx (as _eval_retrieval does)
        pred = set(int(m) for m in pat.findall(resp.get(str(i), "")))
        inter = len(pred & gold)
        p = inter / len(pred) if pred else 0.0
        r = inter / len(gold) if gold else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return sum(f1s) / len(f1s)


print(f"niah gold_id_f1  OLD regex \\[(\\d+)\\]  = {score(OLD):.3f}  (the broken 0.16)")
print(f"niah gold_id_f1  NEW regex \\[?(\\d+)\\] = {score(NEW):.3f}  (with the fix)")
