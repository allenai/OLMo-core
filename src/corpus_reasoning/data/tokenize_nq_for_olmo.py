"""Tokenize NQ-k50 retrieval data into olmo-core NumpyPaddedFSLDataset format.

Reuses scripts.lib.data_format.build_prompt so the prompt is byte-identical to what
train_chunked_fast / evaluate.py produce (standard-attention form, i.e. no wrap_documents).
Emits a flat uint32 token array (n_examples x SEQ, pad-filled) + a parallel bool label-mask
(True only on the 'Relevant Document: [id]' completion tokens).
"""

import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer

from corpus_reasoning.lib.data_format import build_prompt


def main():
    raise NotImplementedError(
        "tokenize_nq_for_olmo.py is DEPRECATED — superseded by "
        "scripts/data/tokenize_unified_for_olmo.py (any task/format, multitask-aware). "
        "The olmo-core dispatcher (train.py, `backend: olmo-core`) calls it automatically."
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/nq_train_k50_hn49_2500_cot.jsonl")
    ap.add_argument("--out-dir", default="data/olmo_nq")
    ap.add_argument("--prefix", default="nq_k50_cot_train")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--task", default="cot_retrieval")
    ap.add_argument("--query-position", default="both")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--pad-token-id", type=int, default=151643)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tok.eos_token_id
    os.makedirs(args.out_dir, exist_ok=True)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    examples = [json.loads(l) for l in open(args.input)]
    # olmo-core's NumpyPaddedFSLDataset wants a RAW concat of EOS-terminated documents (no
    # manual padding) — it segments on eos at prepare() time and pads each doc to seq_len at
    # read time. So emit variable-length docs ending in the real tokenizer eos.
    all_ids, all_mask = [], []
    kept = skipped = 0
    for ex in examples:
        prompt, output = build_prompt(
            ex, task=args.task, query_position=args.query_position,
            use_alpaca=True, cot_mode="label",
        )
        # Tokenize prompt and completion separately so the loss-mask boundary is exact and
        # the prompt tokenization matches inference (evaluate.py tokenizes the prompt alone).
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        o_ids = tok(output, add_special_tokens=False).input_ids + [eos_id]
        if len(p_ids) + len(o_ids) > args.seq_len:  # docs >seq_len get truncated (would lose output)
            skipped += 1
            continue
        all_ids.extend(p_ids + o_ids)
        all_mask.extend([False] * len(p_ids) + [True] * len(o_ids))
        kept += 1

    tokens = np.asarray(all_ids, dtype=np.uint32)
    masks = np.asarray(all_mask, dtype=np.bool_)
    tok_path = os.path.join(args.out_dir, f"{args.prefix}_tokens.npy")
    mask_path = os.path.join(args.out_dir, f"{args.prefix}_label_mask.npy")
    meta_path = os.path.join(args.out_dir, f"{args.prefix}_meta.json")
    # RAW C-order bytes (olmo-core memmaps these; a .npy header would be miscounted as data).
    tokens.tofile(tok_path)
    masks.tofile(mask_path)
    # Sidecar so the SFT launcher uses the SAME eos/pad ids the data was written with.
    json.dump({"eos_token_id": int(eos_id), "pad_token_id": int(pad_id),
               "n_examples": kept, "seq_len": args.seq_len}, open(meta_path, "w"))
    print(f"kept={kept} skipped(>{args.seq_len})={skipped} loss_tokens={int(masks.sum())} "
          f"eos_id={eos_id} pad_id={pad_id}")
    print(f"tokens: {tok_path} shape={tokens.shape} dtype={tokens.dtype}")
    print(f"mask:   {mask_path} shape={masks.shape} dtype={masks.dtype}")
    print(f"per-example loss tokens (avg)={masks.sum()/max(kept,1):.1f}")


if __name__ == "__main__":
    main()
