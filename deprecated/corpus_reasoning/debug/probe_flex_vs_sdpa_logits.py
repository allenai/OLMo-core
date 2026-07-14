"""Probe whether HF native flex_attention forward produces logits that match
sdpa/eager when fed the SAME logical chunked mask on Qwen2.5-3B.

Builds a synthetic prompt with three docs + a query, derives chunk_ids, and
runs three forwards through fresh model copies:

  A) attn_implementation="flex_attention" + BlockMask via attention_mask kwarg
     (= the training-time path used by train_chunked_fast.py for Qwen2.5)
  B) attn_implementation="sdpa"           + dense 4D float mask via attention_mask
     (= the train_chunked_fast.py SDPA collator path)
  C) attn_implementation="eager"          + same dense 4D float mask
     (= reference, no kernel optimizations)

If A diverges from B/C by more than bf16 noise, that confirms flex training
sees materially different logits from sdpa training despite identical masks.
"""

import torch
import sys
import os
 # sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.attention.flex_attention import create_block_mask

from corpus_reasoning.lib.chunked_attention import (
    DOC_START, DOC_END, setup_tokenizer, wrap_documents,
    AttentionPattern, build_flex_mask_mod, build_dense_bool_mask,
    PAD_CHUNK_ID, FREE_CHUNK_ID,
)
from corpus_reasoning.train.train_chunked_fast import build_chunk_ids, FLEX_BLOCK


def load_model(attn_impl, base_model, tokenizer):
    m = AutoModelForCausalLM.from_pretrained(
        base_model, attn_implementation=attn_impl, torch_dtype=torch.bfloat16,
    )
    m.resize_token_embeddings(len(tokenizer))
    # Match the training init: doc_start/doc_end embeddings = mean of others.
    with torch.no_grad():
        mean_emb = m.get_input_embeddings().weight[:-2].mean(dim=0)
        m.get_input_embeddings().weight[-2] = mean_emb
        m.get_input_embeddings().weight[-1] = mean_emb
    m = m.cuda().eval()
    return m


def main():
    BASE = "Qwen/Qwen2.5-3B"
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    doc_start_id, doc_end_id = setup_tokenizer(tok)

    # Hand-build a doc-wrapped sequence with explicit doc_start/doc_end tokens
    # so chunk_ids actually has document spans (not just FREE+PAD). This lets
    # the chunked mask diverge from plain causal — which is the case we want
    # to compare flex vs sdpa on.
    pre_ids  = tok.encode("### Instruction:\nFind contradiction pairs.\n\n### Documents:\n\n",
                          add_special_tokens=False)
    doc_a    = tok.encode("Document 1\n\nThe sky is blue.\n\n", add_special_tokens=False)
    doc_b    = tok.encode("Document 2\n\nThe grass is green.\n\n", add_special_tokens=False)
    doc_c    = tok.encode("Document 3\n\nWater flows downhill.\n\n", add_special_tokens=False)
    post_ids = tok.encode("### Query:\nWhich pair contradicts?\n\n### Response:\n",
                          add_special_tokens=False)

    seq = []
    chunk_id_list = []
    seq += pre_ids;        chunk_id_list += [FREE_CHUNK_ID] * len(pre_ids)
    for di, doc in enumerate([doc_a, doc_b, doc_c]):
        seq.append(doc_start_id);  chunk_id_list.append(di)
        seq += doc;                chunk_id_list += [di] * len(doc)
        seq.append(doc_end_id);    chunk_id_list.append(di)
    seq += post_ids;       chunk_id_list += [FREE_CHUNK_ID] * len(post_ids)

    raw_ids = torch.tensor(seq, dtype=torch.long).unsqueeze(0).cuda()
    B, S_orig = raw_ids.shape
    pad_len = (FLEX_BLOCK - S_orig % FLEX_BLOCK) % FLEX_BLOCK
    if pad_len > 0:
        input_ids = torch.cat([
            raw_ids,
            torch.full((B, pad_len), tok.pad_token_id, dtype=raw_ids.dtype, device=raw_ids.device),
        ], dim=1)
    else:
        input_ids = raw_ids
    S = input_ids.shape[1]

    chunk_ids = torch.full((B, S), PAD_CHUNK_ID, dtype=torch.int32)
    chunk_ids[0, :S_orig] = torch.tensor(chunk_id_list, dtype=torch.int32)
    chunk_ids = chunk_ids.cuda()

    pattern = AttentionPattern(name="chunked")

    # --- Build BlockMask for flex path
    mask_mod = build_flex_mask_mod(pattern, chunk_ids)
    block_mask = create_block_mask(
        mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device=input_ids.device,
    )

    # --- Build equivalent dense bool/float mask for sdpa+eager paths
    bool_mask = build_dense_bool_mask(pattern, chunk_ids)  # (B, S, S)
    dtype = torch.bfloat16
    min_val = torch.finfo(dtype).min
    dense_mask = torch.where(
        bool_mask, torch.zeros(1, dtype=dtype, device=input_ids.device),
        torch.full((1,), min_val, dtype=dtype, device=input_ids.device),
    )
    dense_mask_4d = dense_mask.unsqueeze(1)  # (B, 1, S, S) — broadcast over heads

    print(f"[setup] B={B} S_orig={S_orig} S(padded)={S}  vocab_size_after_resize={len(tok)}")
    print(f"[setup] doc_start_id={doc_start_id} doc_end_id={doc_end_id}")
    print(f"[setup] chunk_ids unique values: {chunk_ids.unique().tolist()}")
    print()

    response_marker = tok.encode("### Response:\n", add_special_tokens=False)
    ids = input_ids[0].tolist()
    response_start = None
    for i in range(len(ids) - len(response_marker) + 1):
        if ids[i:i + len(response_marker)] == response_marker:
            response_start = i + len(response_marker)
            break
    print(f"[setup] response_start_idx={response_start}")
    print()

    results = {}
    for label, attn_impl, mask in [
        ("flex (BlockMask)", "flex_attention", block_mask),
        ("sdpa (dense 4D)",  "sdpa",           dense_mask_4d),
        ("eager (dense 4D)", "eager",          dense_mask_4d),
    ]:
        torch.manual_seed(0)
        model = load_model(attn_impl, BASE, tok)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=mask, use_cache=False)
        logits = out.logits  # (B, S, V)
        # Focus on the first response position (predicting first response token).
        if response_start is not None and response_start - 1 < S:
            resp_logits = logits[0, response_start - 1, :].float()
        else:
            resp_logits = logits[0, S_orig - 1, :].float()
        results[label] = resp_logits
        top5 = torch.topk(resp_logits, 5)
        top_strs = [tok.decode([i.item()]) for i in top5.indices]
        print(f"[{label}] top5 next-token (at response_start-1):")
        for i, (s, p) in enumerate(zip(top_strs, top5.values.tolist())):
            print(f"   #{i+1}: logit={p:+.3f} token={s!r}")
        # Free memory before loading next model.
        del model
        torch.cuda.empty_cache()
        print()

    a = results["flex (BlockMask)"]
    b = results["sdpa (dense 4D)"]
    c = results["eager (dense 4D)"]

    def cmp(x, y, tag):
        d = (x - y).abs()
        cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
        print(f"[diff {tag}] max-abs={d.max().item():.4f}  mean-abs={d.mean().item():.4f}  "
              f"cos_sim={cos:.6f}  argmax_match={(x.argmax()==y.argmax()).item()}")

    print("=== Logit divergence at first response position ===")
    cmp(a, b, "flex vs sdpa ")
    cmp(a, c, "flex vs eager")
    cmp(b, c, "sdpa vs eager")
    print()

    # bf16 noise floor for reference: same model, same mask, just dtype.
    # Sdpa-vs-eager gives the lower-bound for "what's just kernel noise."


if __name__ == "__main__":
    main()
