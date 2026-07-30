"""Smoke test for the RULER pipeline: generate -> build_prompt -> _eval_ruler.

Run via sbatch on a compute node (login node has no python). Generates a tiny
split for every subtask in num_docs mode, builds a training prompt for one
example, checks the gold strings actually appear in the rendered context, then
runs the ruler eval metric against (a) a perfect prediction and (b) an empty
prediction to confirm recall/all_correct behave.
"""
import json
import subprocess
import sys
import tempfile
import os
 # sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from corpus_reasoning.lib.data_format import build_prompt
from corpus_reasoning.eval.evaluate import _eval_ruler

SUBTASKS = ["niah_single", "niah_multikey", "niah_multivalue",
            "niah_multiquery", "vt", "cwe", "fwe"]

tmp = tempfile.mkdtemp()
ok = True
for sub in SUBTASKS:
    subprocess.run([
        sys.executable, "scripts/data/generate_ruler_data.py",
        "--subtask", sub, "--length-mode", "num_docs", "--num-docs", "12",
        "--num-train", "8", "--num-eval", "5", "--disjoint-vocab",
        "--output-dir", tmp, "--seed", "42",
    ], check=True)

    # find the generated train file
    train = [f for f in os.listdir(tmp)
             if f.startswith(f"ruler_{sub}_") and f.endswith("_train.jsonl")][0]
    rows = [json.loads(l) for l in open(os.path.join(tmp, train))]
    ex = rows[0]

    # 1) prompt builds and gold strings are present in the rendered context
    prompt, output = build_prompt(ex, task="ruler", use_alpaca=True)
    missing = [a for a in ex["answers"] if str(a) not in prompt]
    ctx_ok = not missing
    out_ok = output.startswith("Answer:") and all(str(a) in output for a in ex["answers"])

    # 2) eval metric: perfect vs empty prediction
    entries = [{"answers": r["answers"]} for r in rows]
    perfect = ["Answer: " + ", ".join(map(str, r["answers"])) for r in rows]
    empty = ["Answer: nothing" for _ in rows]
    m_perfect, _ = _eval_ruler(entries, perfect)
    m_empty, _ = _eval_ruler(entries, empty)

    good = (ctx_ok and out_ok
            and abs(m_perfect["recall"] - 1.0) < 1e-9
            and m_perfect["all_correct"] == 1.0
            and m_empty["recall"] < 1.0)
    ok = ok and good
    print(f"[{sub}] n_docs={len(ex['documents'])} n_ans={len(ex['answers'])} "
          f"ctx_ok={ctx_ok} out_ok={out_ok} "
          f"perfect_recall={m_perfect['recall']:.2f} "
          f"empty_recall={m_empty['recall']:.2f} -> {'OK' if good else 'FAIL'}")
    if missing:
        print(f"    MISSING gold in context: {missing}")

print("\nPROMPT SAMPLE (niah_single, truncated):")
sub = "niah_single"
train = [f for f in os.listdir(tmp)
         if f.startswith(f"ruler_{sub}_") and f.endswith("_train.jsonl")][0]
ex = [json.loads(l) for l in open(os.path.join(tmp, train))][0]
prompt, output = build_prompt(ex, task="ruler", use_alpaca=True)
print(prompt[-600:])
print("---OUTPUT---")
print(output)

print("\nALL OK" if ok else "\nSMOKE TEST FAILED")
sys.exit(0 if ok else 1)
