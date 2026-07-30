"""Logic test for CE-ranked rerank: build_prompt ordering + _eval_rerank NDCG/tau."""
import math

from corpus_reasoning.lib.data_format import build_prompt, _build_output, _rerank_reference_order
from corpus_reasoning.lib.prompts import rerank_instruction

# Pool of 5 docs. CE scores: doc indices 0..4 (1-indexed 1..5).
#   idx0 gold ce=8.0, idx1 hard ce=-2.0, idx2 random None,
#   idx3 hard ce=1.0, idx4 random None
ex = {
    "documents": [{"title": None, "text": f"d{i}"} for i in range(5)],
    "queries": ["q?"],
    "answers": [""],
    "gold_doc_indices": [0],
    "hard_neg_indices": [1, 3],
    "ce_scores": [8.0, -2.0, None, 1.0, None],
    "source": "msmarco_trainhn",
}

# Reference order = scored by CE desc (1,4,2) then unscored in display order (3,5).
order = _rerank_reference_order(ex)
assert order == [1, 4, 2, 3, 5], order
print("ref order OK:", order)

# Target uses CE order; output_top_k truncates.
full = _build_output(ex, "rerank")
assert full == "Ranking: [1], [4], [2], [3], [5]", full
top3 = _build_output(ex, "rerank", output_top_k=3)
assert top3 == "Ranking: [1], [4], [2]", top3
print("target OK:", top3)

# Instruction wording switches with top_k.
assert "in ranked order" in rerank_instruction(-1)
assert "10 most relevant" in rerank_instruction(10)
print("instruction OK")

# build_prompt threads output_top_k into both instruction and target.
p10, o10 = build_prompt(ex, task="rerank", output_top_k=10, use_alpaca=False)
assert "10 most relevant" in p10, p10[:200]
pall, oall = build_prompt(ex, task="rerank", output_top_k=-1, use_alpaca=False)
assert oall == full
print("build_prompt OK")

# Fallback: no ce_scores -> gold-first then display order.
ex2 = {k: v for k, v in ex.items() if k != "ce_scores"}
ex2["gold_doc_indices"] = [2]
assert _rerank_reference_order(ex2) == [3, 1, 2, 4, 5]
print("fallback OK")

# _eval_rerank: perfect prediction -> ndcg 1.0, tau 1.0; gold MRR 1.0.
from corpus_reasoning.eval.evaluate import _eval_rerank
examples = [{"ex": ex}]
perfect = ["Ranking: [1], [4], [2], [3], [5]"]
res, det = _eval_rerank(examples, perfect)
assert abs(res["mrr@10"] - 1.0) < 1e-9, res
assert abs(res["ndcg@10"] - 1.0) < 1e-9, res
assert abs(res["kendall_tau"] - 1.0) < 1e-9, res
print("eval perfect OK:", {k: round(v, 4) for k, v in res.items() if isinstance(v, float)})

# Reversed scored order -> tau = -1 over scored docs {1,4,2}; ndcg < 1.
rev = ["Ranking: [2], [4], [1], [5], [3]"]
res2, _ = _eval_rerank(examples, rev)
assert abs(res2["kendall_tau"] + 1.0) < 1e-9, res2
assert res2["ndcg@10"] < 1.0, res2
print("eval reversed OK:", {k: round(v, 4) for k, v in res2.items() if isinstance(v, float)})

# Old-format example (no ce_scores) -> no ndcg/tau keys, mrr still computed.
res3, _ = _eval_rerank([{"ex": ex2}], ["Ranking: [3], [1]"])
assert "ndcg@10" not in res3 and abs(res3["mrr@10"] - 1.0) < 1e-9, res3
print("eval legacy OK:", res3)

print("\nALL PASS")
