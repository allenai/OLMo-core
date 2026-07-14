"""Smoke-test xabsence end-to-end (CPU, no LLM): stub-pool -> assemble ->
build_prompt -> parse -> score round-trip via the absence ID-set scorer."""
import json, subprocess, sys, os, glob
# sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from corpus_reasoning.lib.data_format import build_prompt
from corpus_reasoning.eval.evaluate import _parse_id_set

SRC = next((p for p in [
    "data/contradiction_train_pubmed_both_n100_k3.jsonl",
    "data/absence_train_pubmed_n100_p01.jsonl",
    "data/absence_train_pubmed_n20_p01.jsonl",
] if os.path.exists(p)), None)
assert SRC, "no source corpus found"
print("source:", SRC)

P, k = 8, 3
# phase 1: stub pool
subprocess.run([sys.executable, "scripts/data/generate_xabsence_data.py", "--build-pool",
                "--from", SRC, "--pool-out", "/tmp/xab_pool.jsonl", "--pool-size", "60",
                "--stub-paraphrase"], check=True)
# phase 2: assemble
subprocess.run([sys.executable, "scripts/data/generate_xabsence_data.py",
                "--pool", "/tmp/xab_pool.jsonl", "--num-pairs", str(P), "--num-unmatched", str(k),
                "--num-train", "0", "--num-eval", "5", "--src-tag", "pubmed",
                "--output-dir", "/tmp"], check=True)
path = glob.glob("/tmp/xabsence_eval_pubmed_*.jsonl")[0]
exs = [json.loads(l) for l in open(path)]

for ex in exs[:5]:
    items = ex["documents"]
    nA = sum(it["corpus"] == "A" for it in items)
    nB = sum(it["corpus"] == "B" for it in items)
    gold0 = ex["gold_doc_indices"]
    assert len(items) == 2 * P + k, f"expected {2*P+k} items, got {len(items)}"
    assert nA + nB == len(items)
    assert len(gold0) == k, f"expected {k} unmatched, got {len(gold0)}"
    # A block must come before B block (contiguous), shared index
    assert all(items[i]["corpus"] == "A" for i in range(nA)), "A block not contiguous-first"
    prompt, output = build_prompt(ex, task="xabsence", query_position="both", use_alpaca=True)
    assert output.startswith("Unmatched: ["), f"bad output: {output!r}"
    # round-trip: gold output parses back to the gold id set
    gold_ids = {g + 1 for g in gold0}
    pred = _parse_id_set(output, len(items))
    assert pred == gold_ids, f"round-trip FAILED: pred={pred} gold={gold_ids}"
    # parser must anchor on 'Unmatched:' even with distractor brackets before it
    pred2 = _parse_id_set("[99] some reasoning [1]\nUnmatched: " + ", ".join(f"[{g}]" for g in sorted(gold_ids)), len(items))
    assert pred2 == gold_ids, f"anchor FAILED: {pred2}"
    ntag = prompt.count("] A:") + prompt.count("] B:")
    assert ntag == len(items), f"expected {len(items)} tagged lines, got {ntag}"
print(f"\nGOLD OUTPUT example: {output}")
print(f"items/ctx={2*P+k} (A={nA} B={nB}), {k} unmatched, round-trip f1=1.0, anchor OK")
print("ALL XABSENCE SMOKE CHECKS PASSED")
