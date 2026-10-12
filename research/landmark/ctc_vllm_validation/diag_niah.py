import json
import re
import sys

EVAL = "/data/prasann/ctc_suite_staged/eval_rungs/niah/rung_2048.jsonl"
RESP = "/data/prasann/ctc_suite_vllm_results/_thinkfix_audit/responses_niah_2048.json"

ex = [json.loads(l) for i, l in enumerate(open(EVAL)) if i < 3]
resp = json.load(open(RESP)).get("responses", {})
for i in range(3):
    e = ex[i]
    docs = e["documents"]
    gi = e["gold_doc_indices"]
    qs = e.get("queries", [])
    q = qs[0] if qs else e.get("query", "")
    pred = resp.get(str(i), "")
    print(f"=== ex{i}: gold_doc_indices={gi}  n_docs={len(docs)}  MODEL_PRED={pred!r}")
    print(f"  query: {str(q)[:140]}")
    golds = gi if isinstance(gi, list) else [gi]
    for g in golds:
        gt = docs[g] if isinstance(g, int) and 0 <= g < len(docs) else "?"
        print(f"  GOLD doc[{g}] (0-idx): {str(gt)[:150]}")
    m = re.findall(r"\d+", str(pred))
    if m:
        pj = int(m[0]) - 1
        pd = docs[pj] if 0 <= pj < len(docs) else "OOR"
        print(f"  MODEL picked doc[{pj}] (0-idx, pred {m[0]} 1-idx): {str(pd)[:150]}")
    print()
