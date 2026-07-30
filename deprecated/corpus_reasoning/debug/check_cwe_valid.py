import json, subprocess, sys, tempfile, os
from collections import Counter
# sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
tmp = tempfile.mkdtemp()
subprocess.run([sys.executable, "scripts/data/generate_ruler_data.py",
    "--subtask", "cwe", "--length-mode", "target", "--target-length", "32768",
    "--tokenizer", "Qwen/Qwen2.5-0.5B", "--num-train", "1", "--num-eval", "1",
    "--disjoint-vocab", "--output-dir", tmp, "--seed", "7"],
    check=True, stdout=subprocess.DEVNULL)
f = [x for x in os.listdir(tmp) if x.endswith("_train.jsonl")][0]
ex = json.loads(open(os.path.join(tmp, f)).readline())
words = " ".join(d["text"] for d in ex["documents"]).split()
c = Counter(words)
common = ex["answers"]
cf = [c[w] for w in common]
unc = [v for w, v in c.items() if w not in set(common)]
print(f"docs={len(ex['documents'])} total_words={len(words)} distinct={len(c)}")
print(f"common freq: min={min(cf)} max={max(cf)}  | uncommon freq: max={max(unc)}")
ok = min(cf) > max(unc)
print("VALID (every common word strictly more frequent than every uncommon)" if ok
      else "INVALID — ambiguous gold")
sys.exit(0 if ok else 1)
