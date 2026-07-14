"""Build random_token masks on a real n20 contradiction training example
and dump them to a text file for inspection.

Mirrors the trainer pipeline:
  - load Qwen2.5-3B tokenizer + add doc_start/doc_end specials
  - run example 0 through ChunkedDataset (alpaca template, wrap_documents,
    tokenize, label-mask)
  - pad to FLEX_BLOCK boundary like FlexChunkedCollator does
  - build chunk_ids
  - build random_keep at p in {0.0, 0.5, 1.0}
  - build the dense bool mask via the production code path

Output: outputs/debug/random_token_masks_real.txt
"""

import os
import sys
import torch
from transformers import AutoTokenizer
 # sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from corpus_reasoning.train.train_chunked_fast import (
    ChunkedDataset, FLEX_BLOCK, _round_up, build_chunk_ids,
)
from corpus_reasoning.lib.chunked_attention import (
    setup_tokenizer, AttentionPattern, build_dense_bool_mask,
    build_random_token_keep, PAD_CHUNK_ID, FREE_CHUNK_ID,
)


def role_label(cid: int) -> str:
    if cid == PAD_CHUNK_ID:
        return "P"
    if cid == FREE_CHUNK_ID:
        return "F"
    return f"D{cid:02d}"


def main():
    BASE = "Qwen/Qwen2.5-3B"
    DATA = "data/contradiction_train_pubmed_both_n20_k3.jsonl"
    EX_IDX = 0
    SEQ_LEN_CFG = 2048

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    doc_start_id, doc_end_id = setup_tokenizer(tokenizer)

    ds = ChunkedDataset(
        data_path=DATA,
        tokenizer=tokenizer,
        max_len=SEQ_LEN_CFG,
        doc_start_id=doc_start_id, doc_end_id=doc_end_id,
        query_position="both",
        train_on_inputs=False,
        task="contradiction",
        use_titles=True,
        before_dummy=0, after_dummy=0,
        unified_prompt=False,
        cot_mode="label",
    )
    ex = ds[EX_IDX]
    raw_ids = ex["input_ids"]
    raw_len = raw_ids.size(0)
    max_len = _round_up(raw_len, FLEX_BLOCK)

    # Pad like the collator.
    pad_len = max_len - raw_len
    if pad_len > 0:
        ids = torch.cat([raw_ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=raw_ids.dtype)])
    else:
        ids = raw_ids
    chunk_id_orig = build_chunk_ids(raw_ids, doc_start_id, doc_end_id)
    chunk_id = torch.full((max_len,), PAD_CHUNK_ID, dtype=torch.int32)
    chunk_id[:raw_len] = chunk_id_orig
    chunk_ids = chunk_id.unsqueeze(0)

    n_docs = int((chunk_id_orig[chunk_id_orig >= 0]).unique().numel())

    # Build masks.
    masks = {}
    for p in (0.0, 0.5, 1.0):
        rk = build_random_token_keep(max_len, p, seed=42).unsqueeze(0)
        m = build_dense_bool_mask(
            AttentionPattern(name="random_token", keep_prob=p, random_seed=42),
            chunk_ids, random_keep=rk,
        )[0]  # (S, S) bool
        masks[p] = m

    # Write to file.
    out_dir = "outputs/debug"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/random_token_masks_real.txt"

    with open(out_path, "w") as f:
        # ---- Header
        f.write("Random-token attention masks on real training data\n")
        f.write("=" * 72 + "\n")
        f.write(f"data:       {DATA}\n")
        f.write(f"example:    {EX_IDX}\n")
        f.write(f"base model: {BASE}\n")
        f.write(f"raw_len:    {raw_len}  (FLEX_BLOCK-padded to {max_len})\n")
        f.write(f"n_docs:     {n_docs}\n")
        f.write(f"doc_start_id={doc_start_id}, doc_end_id={doc_end_id}, pad_id={tokenizer.pad_token_id}\n")
        f.write(f"random_seed=42\n\n")

        # ---- Token-role layout: compact per-position role string.
        f.write("Token-role layout\n")
        f.write("-" * 72 + "\n")
        f.write("Each position labeled with role: F=free, Dnn=doc nn, P=pad.\n")
        f.write("(Role from chunk_ids.) Doc spans include doc_start/doc_end tokens.\n\n")
        roles = [role_label(int(c)) for c in chunk_id]
        # Row-of-10 grouping with index header.
        for i in range(0, max_len, 20):
            f.write(f"  {i:>5}: ")
            f.write("  ".join(f"{r:<3}" for r in roles[i:i + 20]))
            f.write("\n")
        f.write("\n")

        # ---- Doc-pair edge density per p.
        f.write("Doc-pair edge density (fraction of allowed edges per (q-role, kv-role) block)\n")
        f.write("-" * 72 + "\n")
        f.write("Cells are mean of mask within block. Roles ordered: F, D00..D{n-1}, P.\n")
        f.write("Reads as: of all (q in row-role, kv in col-role) pairs, what fraction\n")
        f.write("are attendable. Should be 1.0 along free rows/cols (modulo causal).\n\n")
        # Build role index per token: free = -1, doc i = i, pad = -2 (already in chunk_id).
        # Map to a column index: F = 0, doc i = 1 + i, P = 1 + n_docs.
        role_idx = chunk_id.clone()
        role_idx[chunk_id == FREE_CHUNK_ID] = 0
        role_idx[chunk_id == PAD_CHUNK_ID] = 1 + n_docs
        for d in range(n_docs):
            role_idx[chunk_id == d] = 1 + d
        n_roles = 2 + n_docs  # F, D00..D{n-1}, P
        role_names = ["F"] + [f"D{i:02d}" for i in range(n_docs)] + ["P"]

        for p, m in masks.items():
            f.write(f"\np = {p:.2f}   (total allowed edges: {int(m.sum())})\n")
            # Build (n_roles x n_roles) mean matrix.
            density = torch.zeros((n_roles, n_roles))
            counts = torch.zeros((n_roles, n_roles))
            for q_role in range(n_roles):
                q_mask = role_idx == q_role
                if not q_mask.any():
                    continue
                for kv_role in range(n_roles):
                    kv_mask = role_idx == kv_role
                    if not kv_mask.any():
                        continue
                    sub = m[q_mask][:, kv_mask]
                    density[q_role, kv_role] = sub.float().mean()
                    counts[q_role, kv_role] = sub.numel()
            # Print as a table.
            f.write("        " + "  ".join(f"{r:>4}" for r in role_names) + "\n")
            for q_role in range(n_roles):
                row = "  ".join(
                    f"{density[q_role, kv_role].item():>4.2f}" for kv_role in range(n_roles)
                )
                f.write(f"  {role_names[q_role]:>4}  {row}\n")

        # ---- Zoomed full mask: first 200 tokens.
        ZOOM = min(200, max_len)
        f.write("\n\n")
        f.write(f"Mask zoom: first {ZOOM} tokens (full content-doc block visible)\n")
        f.write("-" * 72 + "\n")
        f.write("# = attendable, . = masked. Row label = role of q token, col labels in header.\n")
        for p, m in masks.items():
            f.write(f"\np = {p:.2f}\n")
            # Header: role per kv col.
            f.write("        " + "".join(roles[i][0] for i in range(ZOOM)) + "\n")
            for i in range(ZOOM):
                line = "".join("#" if m[i, j].item() else "." for j in range(ZOOM))
                f.write(f"  {roles[i]:>3} {i:>4}  {line}\n")

        # ---- Edge counts summary.
        f.write("\n\n")
        f.write("Edge-count summary\n")
        f.write("-" * 72 + "\n")
        for p, m in masks.items():
            f.write(f"  p = {p:.2f}   edges = {int(m.sum()):>8d}\n")

    print(f"wrote {out_path}")
    print(f"  raw_len={raw_len}  padded={max_len}  n_docs={n_docs}")
    for p, m in masks.items():
        print(f"  p={p:.2f}  edges={int(m.sum())}")


if __name__ == "__main__":
    main()
