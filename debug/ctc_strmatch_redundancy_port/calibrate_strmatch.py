"""Measure strmatch tokens/document against the Qwen3 tokenizer.

BUILD_MATRIX's row 20 is an ESTIMATE (``~45 (synth x1.5-3)``) and flags itself
``calibrate before freezing n values``. This renders shipped strmatch examples through the real
prompt assembler and reports the median prompt length, so the ported ladder row can say
``measured`` rather than inheriting a guess.
"""

import json
import statistics
import sys

from transformers import AutoTokenizer

from ctc.format import registry
from ctc.tasks import load_all

load_all()
spec = registry.get("strmatch")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

path = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 40
lengths, per_doc = [], []
with open(path, encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if i >= limit:
            break
        row = json.loads(line)
        prompt = spec.build_prompt(row)
        n = len(tok(prompt)["input_ids"])
        lengths.append(n)
        per_doc.append(n / len(row["documents"]))
print(f"{path}")
print(f"  docs/example      : {len(row['documents'])}")
print(f"  median prompt tok : {statistics.median(lengths):.0f}")
print(f"  median tok/doc    : {statistics.median(per_doc):.2f}")
