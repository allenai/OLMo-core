"""Convert a unified-format JSONL -> olmo-core SFT shards matching
convert_longctx_tasks_to_sft.py's format (token_ids_part_*.npy uint32 + labels_mask_*.npy bool,
EOS-separated, completion-only loss), so it can be mixed via NumpyDocumentSource.

Generalized from convert_nq_to_sft.py: pass --task to pick the build_prompt task
(e.g. retrieval, rerank, outlier). Prompt comes from
scripts.lib.data_format.build_prompt(task=<TASK>) — byte-identical to the eval —
rendered through the Qwen3 chat template. Loss mask = assistant turn only (offset-based).

    python scripts/data/convert_unified_to_sft.py \
      --task rerank \
      --input data/msmarco_train_rerank_k20_5000.jsonl \
      --out-dir /scratch/users/prasann/cpt_data/rerank_qwen --max-seq-len 8192
"""
import argparse
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer
 # sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from corpus_reasoning.lib.data_format import build_prompt  # noqa: E402

EOS = 151643


def render_chat(tok, messages, add_generation_prompt):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def tokenize_instance(tok, user_content, answer, max_seq_len):
    """(token_ids, labels_mask) with loss True only on the assistant turn; None if too long/bad."""
    messages = [{"role": "user", "content": user_content}]
    prompt_str = render_chat(tok, messages, True)
    full_str = render_chat(tok, messages + [{"role": "assistant", "content": answer}], False)
    if not full_str.startswith(prompt_str):
        return None
    enc = tok(full_str, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    if len(ids) > max_seq_len:
        return None
    boundary = len(prompt_str)
    mask = [off[0] >= boundary and off[1] > off[0] for off in enc["offset_mapping"]]
    if not any(mask):
        return None
    ids = ids + [EOS]
    mask = mask + [True]  # learn to emit EOS (matches contradiction converter)
    return np.array(ids, dtype=np.uint32), np.array(mask, dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    help="build_prompt task name (e.g. retrieval, rerank, outlier)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--query-position", default="both")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out_dir, exist_ok=True)
    tok_chunks, mask_chunks = [], []
    lengths = []
    n, n_skip, n_loss = 0, 0, 0
    for i, line in enumerate(open(args.input)):
        if args.limit and n >= args.limit:
            break
        ex = json.loads(line)
        prompt, answer = build_prompt(ex, task=args.task, use_alpaca=False,
                                      query_position=args.query_position)
        r = tokenize_instance(tok, prompt, answer, args.max_seq_len)
        if r is None:
            n_skip += 1
            continue
        ids, mask = r
        tok_chunks.append(ids); mask_chunks.append(mask)
        lengths.append(int(ids.size))
        n += 1; n_loss += int(mask.sum())
        if n % 500 == 0:
            print(f"{n} instances...", flush=True)
    toks = np.concatenate(tok_chunks); masks = np.concatenate(mask_chunks)
    toks.tofile(os.path.join(args.out_dir, "token_ids_part_000000.npy"))
    masks.tofile(os.path.join(args.out_dir, "labels_mask_000000.npy"))
    lengths = np.array(lengths)
    meta = {"input": args.input, "num_instances": n, "num_skipped": n_skip,
            "num_tokens": int(toks.size), "num_loss_tokens": int(n_loss),
            "median_len": int(np.median(lengths)), "max_len": int(lengths.max()),
            "tokenizer": args.tokenizer, "eos": EOS, "task": args.task}
    json.dump(meta, open(os.path.join(args.out_dir, "metadata.json"), "w"), indent=2)
    print("DONE:", json.dumps(meta))


if __name__ == "__main__":
    main()
