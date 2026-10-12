"""
Decode the real outlier training layout to pick ``--st-header-stop-count`` for the header-real arms.

Runs the SAME code the ds64 shards were tokenized with -- ``segment_prompt_to_chunks`` with
``chunk_by="document"`` and the qwen3_5 marker ids -- on a real unified outlier example, then prints
each document's first tokens so the header (``Document [N]:``) and the ``":"`` (id 25) that ends it
are visible, and reports how ``mark_doc_headers_free`` labels them at a given stop count.

CPU only. Needs a local Qwen3.5 tokenizer directory (``--tokenizer``).

    python research/token_routing/ds64/probe_outlier_header.py --tokenizer <dir> \
        --input /scratch/users/prasann/cpt_data/eval500_v2/outlier/outlier_wiki100w_n110_k3_eval_600.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "src")

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from olmo_core.data.document_chunk_landmark import segment_prompt_to_chunks  # noqa: E402
from olmo_core.nn.attention.chunked_mask import (  # noqa: E402
    build_chunk_ids_from_tokens,
    mark_doc_headers_free,
)
from olmo_core.data.document_chunk_landmark import RESERVED_IDS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--stop-counts", default="1,2")
    ap.add_argument("--stop-id", type=int, default=25)
    ap.add_argument("--docs", type=int, default=4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if getattr(tok, "chat_template", None) is None:
        # the exported serving tokenizer carries no template; the Qwen3 one only wraps the prompt
        # and cannot change how a document header inside the user content tokenizes.
        tok.chat_template = (
            "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
            "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
    print("token id 25 ->", repr(tok.decode([25])))
    for s in ("Document [7]: ", "]:", " [7]:", "Document [1234]:",
              "\n\nDocument [7]: body", "Claim 269: text", "Date: 2024 || User: x || Instance: y"):
        print(f"  {s!r} -> {tok(s, add_special_tokens=False)['input_ids']}")

    ids_set = RESERVED_IDS["qwen3_5"]
    print(f"\nmarker ids: doc_start={ids_set.doc_start} doc_end={ids_set.doc_end} eos={ids_set.eos}")

    ex = json.loads(open(args.input).readline())
    print(f"example: {len(ex['documents'])} documents, titles={{{ex['documents'][0].get('title')!r}}}")
    _segs, ids, _m = segment_prompt_to_chunks(
        tok, ex, "outlier", query_position="after", cot_mode="none", chunk_by="document",
        item_regex=r"\|\|", include_answer=True, use_titles=False,
        doc_start_id=ids_set.doc_start, doc_end_id=ids_set.doc_end,
    )
    t = torch.tensor(ids).unsqueeze(0)
    chunk = build_chunk_ids_from_tokens(
        t, doc_start_id=ids_set.doc_start, doc_end_id=ids_set.doc_end, eos_id=ids_set.eos,
        mode="chunked",
    )
    starts = [i for i, x in enumerate(ids) if x == ids_set.doc_start]
    print(f"\n{len(starts)} wrapped documents; first {args.docs} headers (12 tokens after <|box_start|>):")
    for s in starts[: args.docs]:
        toks = ids[s : s + 13]
        print(f"  @{s}: " + " | ".join(repr(tok.decode([x])) for x in toks))

    for sc in [int(x) for x in args.stop_counts.split(",")]:
        free = mark_doc_headers_free(
            chunk, t, doc_start_id=ids_set.doc_start, doc_end_id=ids_set.doc_end,
            stop_id=args.stop_id, stop_count=sc, cap=32,
        )
        n_free = int(((free < 0) & (chunk >= 0)).sum())
        per = []
        for s in starts[:3]:
            k = 0
            while s + 1 + k < len(ids) and free[0, s + 1 + k].item() < 0:
                k += 1
            per.append(repr(tok.decode(ids[s + 1 : s + 1 + k])))
        print(f"\nstop_id={args.stop_id} stop_count={sc}: {n_free} body tokens re-labelled FREE (header-real); "
              f"per-doc header = {'; '.join(per)}")


if __name__ == "__main__":
    main()
