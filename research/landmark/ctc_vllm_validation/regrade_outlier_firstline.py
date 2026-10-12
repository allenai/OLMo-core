import json
import re

EVAL = "/data/prasann/ctc_suite_staged/eval_rungs/outlier/rung_2048.jsonl"
RESP = "/data/prasann/ctc_suite_vllm_results/_thinkfix_audit/responses_outlier_2048.json"
PAT = re.compile(r"\[?\s*(\d+)\s*\]")  # the fixed parser

gold_rows = [json.loads(l) for l in open(EVAL)]
resp = json.load(open(RESP)).get("responses", {})


def set_f1(pred, gold):
    inter = len(pred & gold)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(gold) if gold else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def score(first_line_only):
    fs = []
    for i, e in enumerate(gold_rows):
        gold = set(int(g) + 1 for g in e["gold_doc_indices"])  # 0->1 indexed
        txt = resp.get(str(i), "")
        if first_line_only:
            # take only the first "Outliers:" answer line (kill the ramble)
            txt = txt.split("\n", 1)[0]
        pred = set(int(m) for m in PAT.findall(txt))
        fs.append(set_f1(pred, gold))
    return sum(fs) / len(fs)


print(f"outlier set_f1  FULL (all rambled lines) = {score(False):.3f}  (the 0.66)")
print(f"outlier set_f1  FIRST-LINE only          = {score(True):.3f}")

# cycle sanity: does first-line regress it? (cycle emits [[...]] one line)
CEV = "/data/prasann/ctc_suite_staged/eval_rungs/cycle/rung_2048.jsonl"
CRE = "/data/prasann/ctc_suite_vllm_results/_thinkfix_audit/responses_cycle_2048.json"
try:
    crows = [json.loads(l) for l in open(CEV)]
    cresp = json.load(open(CRE)).get("responses", {})
    print(f"\n(cycle responses present: {len(cresp)}) sample: {list(cresp.values())[0][:60]!r}")
except Exception as e:
    print("cycle responses not preserved:", e)
