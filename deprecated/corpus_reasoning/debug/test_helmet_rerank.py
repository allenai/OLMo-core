"""Logic test for the HELMET-format rerank: converter + prompt + parse + NDCG."""
import math

from corpus_reasoning.data.generate_msmarco_helmet_rerank_data import (
    ce_to_grade, to_helmet_record)
from corpus_reasoning.lib.data_format import build_prompt, build_helmet_rerank
from corpus_reasoning.lib.prompts import HELMET_RERANK_USER_TEMPLATE

THR = [0.1, 0.5, 0.8]

# ── CE -> grade bucketing ──
assert ce_to_grade(10.0, THR) == 3      # p~1.0
assert ce_to_grade(3.0, THR) == 3       # p~0.95
assert ce_to_grade(0.5, THR) == 2       # p~0.62
assert ce_to_grade(-1.0, THR) == 1      # p~0.27
assert ce_to_grade(-10.0, THR) == 0     # p~0.0
assert ce_to_grade(None, THR) == 0
print("ce_to_grade OK")

# ── Unified CE-scored example -> HELMET record ──
ex = {
    "documents": [{"title": None, "text": f"passage {i}"} for i in range(5)],
    "queries": ["what is x?"],
    "answers": [""],
    "gold_doc_indices": [0],
    "hard_neg_indices": [1, 3],
    "ce_scores": [10.0, -10.0, 0.5, 3.0, -1.0],   # -> labels [3,0,2,3,1]
    "source": "msmarco_trainhn",
}
rec = to_helmet_record(ex, idx=0, thresholds=THR, seed=42)
assert rec["query"] == "what is x?"
assert len(rec["qid"]) == 12
labels = [c["label"] for c in rec["ctxs"]]
assert labels == [3, 0, 2, 3, 1], labels
ids = [c["id"] for c in rec["ctxs"]]
assert sorted(int(i) for i in ids) == [1, 2, 3, 4, 5], ids   # perm of 1..K
assert "title" not in rec["ctxs"][0]                          # untitled passages
print("to_helmet_record OK:", labels, ids)

# ── Prompt + target (build_prompt task=rerank_helmet) ──
prompt, target = build_prompt(rec, task="rerank_helmet")
# gold order = ids sorted by label desc, stable (pos0,pos3,pos2,pos4,pos1)
gold_ids = [ids[0], ids[3], ids[2], ids[4], ids[1]]
# K=5 pool < default top-10 -> full order
assert target == " " + " > ".join(gold_ids), (target, gold_ids)
assert prompt.endswith("Ranking:"), prompt[-40:]
assert "relelvant" in prompt                                  # verbatim HELMET typo
assert f"[ID: {ids[0]}] Document: passage 0" in prompt
assert "Query: what is x?" in prompt
assert "{demos}" not in prompt and "{context}" not in prompt  # template filled
# zero-shot: nothing between the instruction blank line and first passage
assert build_helmet_rerank(rec)[0] == prompt
print("prompt/target OK")

# top-10 truncation on a >10 pool: target lists exactly 10 ids (highest labels),
# but the prompt still shows ALL docs (HELMET instruction verbatim).
big = {
    "documents": [{"title": None, "text": f"p{i}"} for i in range(14)],
    "queries": ["q?"], "answers": [""], "gold_doc_indices": [0],
    "hard_neg_indices": [], "source": "msmarco_trainhn",
    # descending CE so label/order is unambiguous
    "ce_scores": [12.0 - i for i in range(14)],
}
brec = to_helmet_record(big, idx=1, thresholds=THR, seed=7)
bprompt, btarget = build_prompt(brec, task="rerank_helmet")
assert len(btarget.strip().split(" > ")) == 10, btarget
assert bprompt.count("[ID:") == 14, "prompt must still show all docs"
# explicit override to full ranking
_, bfull = build_prompt(brec, task="rerank_helmet", output_top_k=14)
assert len(bfull.strip().split(" > ")) == 14, bfull
print("top-10 truncation OK")

# tie-break: two docs bucket to the SAME grade but have different CE -> the
# target must lead with the higher-CE one (the gold), not the display-order one.
tie = {
    "documents": [{"title": None, "text": "distractor"},
                  {"title": None, "text": "gold"}],
    "queries": ["q?"], "answers": [""], "gold_doc_indices": [1],
    "hard_neg_indices": [0], "source": "msmarco_trainhn",
    "ce_scores": [5.0, 9.0],   # both sigmoid>=0.8 -> grade 3; gold (idx1) higher
}
trec = to_helmet_record(tie, idx=2, thresholds=THR, seed=1)
assert [c["label"] for c in trec["ctxs"]] == [3, 3], trec
gold_id = trec["ctxs"][1]["id"]
_, ttgt = build_prompt(trec, task="rerank_helmet")
assert ttgt.strip().split(" > ")[0] == gold_id, (ttgt, gold_id)
print("tie-break by CE OK (gold leads despite equal grade)")

# few-shot demo injection
rec2 = dict(rec); rec2["_demos"] = "DEMOBLOCK\n\n"
p2, _ = build_prompt(rec2, task="rerank_helmet")
assert "DEMOBLOCK\n\n[ID:" in p2, "demos must directly precede the context"
print("demo injection OK")

# ── Parser + graded NDCG ──
from corpus_reasoning.eval.evaluate import _parse_helmet_rankings, _eval_rerank_helmet
valid = [c["id"] for c in rec["ctxs"]]
assert _parse_helmet_rankings("Ranking:" + target, valid) == gold_ids
# tolerate "ID"-prefixed echoes and trailing newline
assert _parse_helmet_rankings(f"Ranking: ID{gold_ids[0]} > ID{gold_ids[1]}\njunk",
                              valid) == gold_ids[:2]
print("parse OK")

examples = [{"ex": rec}]
perfect = ["Ranking:" + target]
res, _ = _eval_rerank_helmet(examples, perfect)
assert abs(res["ndcg@10"] - 1.0) < 1e-9, res
assert abs(res["parse_rate"] - 1.0) < 1e-9, res
print("ndcg perfect OK:", {k: round(v, 4) for k, v in res.items()
                           if isinstance(v, float)})

reverse = ["Ranking: " + " > ".join(reversed(gold_ids))]
res2, _ = _eval_rerank_helmet(examples, reverse)
assert res2["ndcg@10"] < 1.0, res2
print("ndcg reversed OK:", round(res2["ndcg@10"], 4))

# empty / unparseable -> parse_rate 0, ndcg 0
res3, _ = _eval_rerank_helmet(examples, ["i refuse"])
assert res3["parse_rate"] == 0.0 and res3["ndcg@10"] == 0.0, res3
print("ndcg empty OK")

# discrimination: raw-CE-rank NDCG must score the true CE order ABOVE a ranking
# that swaps the top two (the bucketed/graded metric could not tell these apart).
ce_order = [ids[0], ids[3], ids[2], ids[4], ids[1]]          # idx0 ce=8 leads
swap_top = [ids[3], ids[0], ids[2], ids[4], ids[1]]          # idx3 ce=3 first
rg, _ = _eval_rerank_helmet(examples, ["Ranking: " + " > ".join(ce_order)])
rb, _ = _eval_rerank_helmet(examples, ["Ranking: " + " > ".join(swap_top)])
assert abs(rg["ndcg@10"] - 1.0) < 1e-9 and rb["ndcg@10"] < 1.0, (rg, rb)
assert rg["ndcg@10"] > rb["ndcg@10"], (rg, rb)
print(f"discrimination OK: CE-order ndcg@10={rg['ndcg@10']:.4f} > "
      f"swapped-top {rb['ndcg@10']:.4f}")

print("\nALL PASS")
