"""
Scan an oolong training shard for the ``--item-regex '||'`` defect.

Two independent signatures of the bug, both read straight off the raw ``token_ids_part_*.npy``
uint32 stream (no tokenizer needed for the first, one needed for the second):

1. **Inter-chunk FREE tokens.** Every token between ``<|box_end|>`` and the next ``<|box_start|>``
   that is not an example boundary (no EOS in the gap) is a global bridge between two chunks that
   the mask is supposed to isolate. The correct escaped regex leaves 0; the bare ``'||'`` wraps the
   instruction / question / header lines as their own chunks and leaves the blank lines between
   them FREE, which measures as ~5 per example.

2. **Chunks per example vs ITEM lines per example.** oolong is line-based: exactly the item lines
   (``Date: ... || User: ... || Instance: ...``) should be wrapped. Under the bug, chunks/example
   overshoots the item count by the number of preamble/question lines.

Usage::

    python scan_oolong_shard.py --shard /data/prasann/ctc_suite/shards/oolong_train --parts 1
    python scan_oolong_shard.py --shard ... --tokenizer /data/prasann/hf_models/Qwen3.5-4B-Base
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List

import numpy as np

#: Qwen3.5 marker ids, as recorded in every oolong shard's ``metadata.json``.
DOC_START, DOC_END, EOS = 248049, 248050, 248044

#: An oolong item line, per ``ctc.data.sources.oolong._ITEM_RE``.
ITEM_RE = re.compile(r"^Date: (.*?) \|\| User: (.*?) \|\| Instance: ")


def split_examples(ids: np.ndarray) -> List[np.ndarray]:
    """
    :param ids: A flat uint32 token stream.

    :returns: One array per EOS-terminated example.
    """
    eos = np.flatnonzero(ids == EOS)
    out, lo = [], 0
    for e in eos.tolist():
        out.append(ids[lo : e + 1])
        lo = e + 1
    return out


def scan_example(ex: np.ndarray) -> Dict[str, int]:
    """
    :param ex: One example's token ids, EOS-terminated.

    :returns: ``chunks`` and ``leak_gaps`` / ``leak_tokens`` (FREE tokens strictly between chunks).
    """
    starts = np.flatnonzero(ex == DOC_START)
    ends = np.flatnonzero(ex == DOC_END)
    n = min(len(starts), len(ends))
    leak_gaps = leak_tokens = 0
    for k in range(n - 1):
        gap = int(starts[k + 1]) - int(ends[k]) - 1
        if gap > 0:
            leak_gaps += 1
            leak_tokens += gap
    return {"chunks": n, "leak_gaps": leak_gaps, "leak_tokens": leak_tokens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="a *_train shard directory")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--max-examples", type=int, default=0, help="0 = every example in the parts")
    ap.add_argument("--tokenizer", default=None, help="enables the chunks-vs-item-lines check")
    ap.add_argument("--decode-examples", type=int, default=3)
    ap.add_argument("--out", default=None, help="write the summary here as JSON")
    args = ap.parse_args()

    parts = sorted(glob.glob(os.path.join(args.shard, "token_ids_part_*.npy")))[: args.parts]
    if not parts:
        raise SystemExit(f"no token_ids_part_*.npy under {args.shard}")

    examples: List[np.ndarray] = []
    for p in parts:
        examples.extend(split_examples(np.fromfile(p, dtype=np.uint32)))
    if args.max_examples:
        examples = examples[: args.max_examples]

    per = [scan_example(e) for e in examples]
    total = {k: sum(d[k] for d in per) for k in ("chunks", "leak_gaps", "leak_tokens")}
    n_ex = len(examples)
    summary: Dict[str, object] = {
        "shard": args.shard,
        "parts_scanned": len(parts),
        "examples": n_ex,
        "chunks_total": total["chunks"],
        "chunks_per_example": round(total["chunks"] / n_ex, 2) if n_ex else 0,
        "inter_chunk_free_gaps": total["leak_gaps"],
        "inter_chunk_free_tokens": total["leak_tokens"],
        "leak_gaps_per_example": round(total["leak_gaps"] / n_ex, 3) if n_ex else 0,
        "verdict": "CLEAN" if total["leak_tokens"] == 0 else "LEAK",
    }

    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        rows = []
        for ex in examples[: args.decode_examples]:
            text = tok.decode(ex.astype(np.int64).tolist(), skip_special_tokens=False)
            stripped = text.replace("<|box_start|>", "").replace("<|box_end|>", "")
            lines = stripped.split("\n")
            item_lines = sum(1 for ln in lines if ITEM_RE.match(ln))
            chunks = int((ex == DOC_START).sum())
            rows.append(
                {
                    "chunks": chunks,
                    "item_lines": item_lines,
                    "total_lines": len(lines),
                    "excess_chunks_over_items": chunks - item_lines,
                    "first_chunk_text": (
                        text.split("<|box_start|>")[1].split("<|box_end|>")[0][:160]
                        if "<|box_start|>" in text
                        else ""
                    ),
                }
            )
        summary["decoded"] = rows

    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
