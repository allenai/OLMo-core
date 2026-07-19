"""
Build the **eval-side** ``gold_pairs.json`` for the ``gold_hop_controlled`` mask, keyed by the
fingerprint of the eval's **prompt-only prefill**.

Why a separate builder from ``build_gold_sidecar_from_shard.py``
----------------------------------------------------------------
The gold-hop mask looks each example up by a SHA1 of the token ids the model is about to read (so gold
identity never enters the token stream). Training rows and eval prefills are **different token
sequences for the same example**:

===================  ===========================================================
training shard row   prompt + ``<think></think>[[a, b], ...]`` + EOS + right-padding
eval prefill         prompt only -- **no answer, no EOS, no padding**
===================  ===========================================================

So the training shard's ``gold_pairs.json`` **cannot** hit at eval -- not "mostly", not "for some
examples": every fingerprint misses. And a miss degrades to an all-True graph = plain causal over the
context, which would score every arm as unrestricted ``standard`` (near the 0.943 ceiling) and look
like a triumphant result. Hence this builder, and hence the eval's hard hit-rate assert.

⚠ **Keyed off the eval's own renderer, not a copy of it.** This calls
``eval_lc_native_docchunk_contra.build_eval_prefill`` -- the exact function the eval uses to build the
ids it feeds the model. A fingerprint is a hash: a one-token drift between two copies of that rendering
would silently produce a 0% hit rate. There is one implementation, so there is nothing to drift.

⚠ **Every layout flag must match the eval invocation**, for the same reason: ``--cot-mode``,
``--free-pad-repeat``, ``--repeat-doc-text``, ``--summary-every-k`` and the boundary/EOS ids all change
the token stream and therefore the key. The defaults here mirror the eval's.

Usage::

    python src/scripts/data/build_gold_pairs_for_eval.py \\
        --contra-data /scratch/users/prasann/corpus-reasoning/data/contradiction_eval_pubmed_both_n50_k3.jsonl \\
        --out /scratch/users/prasann/longctx_sft_qwen/contra_n50_v2_orig/gold_pairs_eval_n50.json \\
        --tokenizer Qwen/Qwen3-0.6B --cot-mode none
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from olmo_core.data.document_chunk_landmark import (  # canonical ids -- never retype
    DOC_END_ID,
    DOC_START_ID,
    EOS_TOKEN_ID,
)
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens
from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contra-data", required=True, help="the eval JSONL (unified format)")
    ap.add_argument("--out", required=True, help="destination gold_pairs.json")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--variant", default="dense", choices=["dense", "landmark", "full"])
    # ⚠ Every flag below changes the prefill token stream and therefore the fingerprint. They MUST
    # equal the eval invocation's values, which must in turn equal the training shard's.
    ap.add_argument("--cot-mode", default="none")
    ap.add_argument("--free-pad-repeat", type=int, default=0)
    ap.add_argument("--repeat-doc-text", type=int, default=1)
    ap.add_argument("--summary-every-k", type=int, default=0)
    ap.add_argument("--mem-freq", type=int, default=63)
    ap.add_argument("--doc-start-id", type=int, default=DOC_START_ID)
    ap.add_argument("--doc-end-id", type=int, default=DOC_END_ID)
    ap.add_argument("--eos-token-id", type=int, default=EOS_TOKEN_ID)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    from corpus_reasoning.eval.evaluate import load_unified_examples
    from corpus_reasoning.eval.eval_lc_native_docchunk_contra import build_eval_prefill

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # max_samples=0 -> NO subsampling, so the sidecar covers the whole file. The eval may then run any
    # --max-test-samples subset and still hit every row. (load_unified_examples subsamples with a fixed
    # seed when max_samples is truthy; covering everything makes that irrelevant.)
    examples = load_unified_examples(
        args.contra_data, 0, task="contradiction", query_position="both", use_alpaca=True
    )

    table: dict = {}
    n_docs_seen: list = []
    pair_dists: list = []
    collisions = 0

    for ex in examples:
        raw = ex.get("ex", ex)
        prefill = build_eval_prefill(
            tok,
            raw,
            variant=args.variant,
            cot_mode=args.cot_mode,
            doc_start_id=args.doc_start_id,
            doc_end_id=args.doc_end_id,
            free_pad_repeat=args.free_pad_repeat,
            repeat_doc_text=args.repeat_doc_text,
            summary_every_k=args.summary_every_k,
            mem_freq=args.mem_freq,
        )
        # The eval's live hook fingerprints its input_ids with exactly this call, so this is the key
        # the model will look up. (There is no EOS in a prompt-only prefill, so the "slice to the first
        # EOS" is a no-op here -- but it is the same function, not a re-derivation of it.)
        fp = content_fingerprint_from_row(prefill, args.eos_token_id)

        # gold_doc_indices are 1-indexed "Claim N" display ids grouped in contradicting PAIRS; document
        # order survives rendering + wrapping, so chunk index == Claim id - 1.
        gold = raw["gold_doc_indices"]
        pairs = []
        for p in gold:
            if len(p) != 2:
                raise SystemExit(
                    f"gold_doc_indices entry {p!r} is not a pair -- the hop ladder is pair-critical "
                    "and cannot use a flat gold set."
                )
            pairs.append(sorted(int(x) - 1 for x in p))

        # Cross-check against the document structure ACTUALLY present in the prefill tokens: a gold
        # index must name a real chunk in this row, or the mask would edit the wrong document.
        roles = build_chunk_ids_from_tokens(
            torch.tensor([prefill]),
            doc_start_id=args.doc_start_id,
            doc_end_id=args.doc_end_id,
            eos_id=args.eos_token_id,
        )[0]
        present = {int(d) for d in torch.unique(roles[roles >= 0]).tolist()}
        flat = {d for p in pairs for d in p}
        if not flat or not flat.issubset(present):
            raise SystemExit(
                f"FATAL: gold ids {sorted(flat)} are not all real chunks in this prefill "
                f"(present: {min(present) if present else '-'}..{max(present) if present else '-'}). "
                "The prefill layout and gold_doc_indices disagree."
            )

        if fp in table and table[fp] != pairs:
            collisions += 1
        table[fp] = pairs
        n_docs_seen.append(len(present))
        pair_dists.extend(b - a for a, b in pairs)

    if collisions:
        raise SystemExit(
            f"FATAL: {collisions} fingerprint collisions with DIFFERENT gold pairs -- two distinct "
            "examples render to identical prefills. The mask would use the wrong graph."
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table, f)

    d = np.asarray(pair_dists)
    print(f"eval file         : {args.contra_data}")
    print(f"examples          : {len(examples)}  unique fingerprints: {len(table)}")
    print(f"docs/example      : mean {np.mean(n_docs_seen):.1f}  min {min(n_docs_seen)}")
    print(f"pairs total       : {int(d.size)}")
    print(
        f"gold-pair distance: mean {d.mean():.2f}  median {int(np.median(d))}  min {int(d.min())}  "
        f"max {int(d.max())}"
    )
    print(f"  dist==1 (no 2-hop possible): {int((d == 1).sum())} / {d.size} ({(d == 1).mean():.2%})")
    print(f"  dist<=2 (no 3-hop possible): {int((d <= 2).sum())} / {d.size} ({(d <= 2).mean():.2%})")
    print(f"wrote             : {args.out}")

    if len(table) != len(examples):
        print(
            f"WARNING: {len(examples) - len(table)} examples share a prefill with another (identical "
            "gold pairs, so the graph is well-defined, but the eval set has duplicates).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
