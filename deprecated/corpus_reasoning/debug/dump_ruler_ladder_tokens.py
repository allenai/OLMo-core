"""Dump a token-calibrated RULER ladder (4k -> 32k) to a txt file.

For each subtask and each target token length, generate one example in
--length-mode target, build the real training prompt, MEASURE its token length,
then write an abbreviated view: needles and structural pieces are kept verbatim
while long runs of haystack noise / word-list filler are collapsed to a marker.
"""
import json
import os
import subprocess
import sys
import tempfile
 # sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from corpus_reasoning.lib.data_format import build_prompt
from corpus_reasoning.data.generate_ruler_data import NOISE_SENTENCE

TOKENIZER = os.environ.get("RULER_TOKENIZER", "Qwen/Qwen2.5-0.5B")
SUBTASKS = ["niah_single", "niah_multikey", "niah_multivalue",
            "niah_multiquery", "vt", "cwe", "fwe"]
RUNGS = [4096, 8192, 16384, 32768]
OUT = "examples/ruler_ladder_tokens.txt"
# cwe/fwe length is governed by word-frequency params (heuristic from
# target_length), not tokenizer-calibrated like the needle-family haystack.
HEURISTIC = set()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TOKENIZER)


def n_tokens(text):
    return len(tok(text, add_special_tokens=False).input_ids)


def is_noise(p):
    return p.replace(NOISE_SENTENCE, "").strip() == "" and NOISE_SENTENCE in p


def is_wordlist(p):
    toks = p.split()
    return len(toks) >= 8 and all(t.isalpha() for t in toks)


def abbreviate(prompt):
    """Collapse consecutive noise / word-list paragraphs; keep the rest verbatim."""
    paras = prompt.split("\n\n")
    out, run = [], []

    def flush():
        if run:
            ntok = sum(n_tokens(p) for p in run)
            out.append(f"        [... {len(run)} filler paragraph(s), "
                       f"~{ntok} tokens, omitted ...]")
            run.clear()

    for p in paras:
        if is_noise(p) or is_wordlist(p):
            run.append(p)
        else:
            flush()
            out.append(p)
    flush()
    return "\n\n".join(out)


os.makedirs("examples", exist_ok=True)
tmp = tempfile.mkdtemp()
lines = [
    "=" * 80,
    "RULER token-calibrated task ladder — sample TRAIN examples",
    f"tokenizer: {TOKENIZER}   |   --length-mode target",
    "Needles + structure shown verbatim; long noise / word-list runs collapsed.",
    "all subtasks are tokenizer-calibrated to the target length.",
    "=" * 80,
]

for sub in SUBTASKS:
    for L in RUNGS:
        subprocess.run([
            sys.executable, "scripts/data/generate_ruler_data.py",
            "--subtask", sub, "--length-mode", "target", "--target-length", str(L),
            "--tokenizer", TOKENIZER, "--num-train", "1", "--num-eval", "1",
            "--disjoint-vocab", "--output-dir", tmp, "--seed", "7",
        ], check=True, stdout=subprocess.DEVNULL)
        train = [f for f in os.listdir(tmp)
                 if f.startswith(f"ruler_{sub}_") and f.endswith("_train.jsonl")][0]
        path = os.path.join(tmp, train)
        ex = json.loads(open(path).readline())
        os.remove(path)
        prompt, output = build_prompt(ex, task="ruler", use_alpaca=True)
        measured = n_tokens(prompt)
        note = "  (heuristic length)" if sub in HEURISTIC else ""
        lines += [
            "",
            "=" * 80,
            f"SUBTASK: {sub}   |   target={L} tokens   |   MEASURED={measured} "
            f"prompt tokens{note}",
            f"docs={len(ex['documents'])}   gold answers={len(ex['answers'])}   "
            f"needle positions={ex['gold_doc_indices']}",
            f"answers (recall target): {ex['answers']}",
            "-" * 80,
            "[PROMPT — abbreviated]",
            abbreviate(prompt),
            "[EXPECTED OUTPUT]",
            output,
        ]

open(OUT, "w").write("\n".join(lines) + "\n")
print(f"wrote {OUT}")
