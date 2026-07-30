"""Smoke-test qdmatch end-to-end (CPU, no model): generate -> build_prompt ->
parse gold -> score round-trip. Covers NQ (1 gold/query) and HotpotQA (2 golds/
query -> 2k pairs), both layouts. Asserts F1==1.0 on the gold output and that
gold pairs are (query_idx -> doc_idx) consistent with the item tags."""
import json, subprocess, sys, os
# sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from corpus_reasoning.lib.data_format import build_prompt
from corpus_reasoning.eval.evaluate import parse_qd_pairs, pair_metrics

# (tag, eval-source, golds-per-query, expected #pairs for k=3)
SOURCES = [
    ("nq",   "data/nq_validation_k20_hn19_500.jsonl",        1, 3),
    ("hpqa", "data/hotpotqa_eval_k100_bridge_500.jsonl",     2, 6),
]
M = N = 14   # N must be >= G (=2k=6 for hpqa)

for tag, src, gpq, exp_pairs in SOURCES:
    for layout in ["separate", "shuffled"]:
        print(f"\n===== {tag} layout={layout} (golds/query={gpq}) =====")
        subprocess.run([sys.executable, "scripts/data/generate_qdmatch_data.py",
                        "--from-eval", src, "--num-docs", str(N), "--num-relevant", "3",
                        "--layout", layout, "--num-eval", "5", "--output-dir", "/tmp"],
                       check=True)
        path = f"/tmp/qdmatch_eval_{tag}_q{M}_n{N}_k3_{layout}.jsonl"
        ex = [json.loads(l) for l in open(path)][0]
        items = ex["documents"]
        assert sum(i["type"] == "query" for i in items) == M, "M queries"
        assert sum(i["type"] == "document" for i in items) == N, "N docs"
        assert len(ex["gold_pairs"]) == exp_pairs, f"expected {exp_pairs} pairs, got {len(ex['gold_pairs'])}"
        # each gold pair: first idx -> query item, second idx -> document item
        for q, d in ex["gold_pairs"]:
            assert items[q-1]["type"] == "query", f"pair[0]={q} not a query"
            assert items[d-1]["type"] == "document", f"pair[1]={d} not a doc"
        # hpqa: exactly 3 distinct relevant queries, each appearing in 2 pairs
        relq = {q for q, _ in ex["gold_pairs"]}
        assert len(relq) == 3, f"expected 3 relevant queries, got {len(relq)}"
        prompt, output = build_prompt(ex, task="qdmatch", query_position="after", use_alpaca=True)
        print("GOLD OUTPUT:", output)
        m = pair_metrics(parse_qd_pairs(output), ex["gold_pairs"])
        print("round-trip:", m)
        assert m["f1"] == 1.0 and m["exact_match"] == 1.0, f"round-trip FAILED: {m}"
        swapped = [[d, q] for q, d in ex["gold_pairs"]]
        assert pair_metrics(parse_qd_pairs(json.dumps(swapped)), ex["gold_pairs"])["f1"] == 0.0, "order should matter"
        nblocks = prompt.count("] Query:") + prompt.count("] Document:")
        assert nblocks == M + N, f"expected {M+N} tagged lines, got {nblocks}"
        print(f"  OK: {exp_pairs} pairs, order-sensitive, {nblocks} tagged items")

        # enumerate CoT: walks every query, final JSON still parses to gold.
        _, cot_out = build_prompt(ex, task="qdmatch", query_position="after",
                                  use_alpaca=True, cot_mode="enumerate")
        assert cot_out.count(" -> ") == M, f"CoT should walk all {M} queries, got {cot_out.count(' -> ')}"
        assert "-> none" in cot_out, "CoT should mark unanswerable queries 'none'"
        mc = pair_metrics(parse_qd_pairs(cot_out), ex["gold_pairs"])
        assert mc["f1"] == 1.0, f"CoT round-trip FAILED: {mc}"
        print(f"  CoT OK: walks {M} queries, final answer round-trips f1=1.0")

print("\nALL QDMATCH SMOKE CHECKS PASSED (nq + hpqa, both layouts)")
