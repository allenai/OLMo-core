"""Smoke test: end-to-end eval-style generation with the hierarchical
wrapper.

Loads Qwen2.5-0.5B with the hierarchical wrapper installed via the eval
helper, runs a short prefill+decode on a 4-doc prompt, and verifies:
  1. Generation completes without shape mismatches.
  2. Output is non-empty.
  3. Pre-hook ran (chunk_ids populated for the prefill batch).
  4. The pre-hook is installed exactly once even across multiple calls.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from corpus_reasoning.lib.chunked_attention import (
    AttentionPattern, setup_tokenizer, wrap_documents,
)
from corpus_reasoning.eval.evaluate import _generate_hierarchical


MODEL = "Qwen/Qwen2.5-0.5B"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    doc_start_id, doc_end_id = setup_tokenizer(tok)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
    )
    orig_vocab = model.get_input_embeddings().weight.shape[0]
    new_vocab = len(tok)
    if new_vocab > orig_vocab:
        model.resize_token_embeddings(new_vocab)
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            mean_emb = emb[:orig_vocab].mean(dim=0)
            for i in range(orig_vocab, new_vocab):
                emb[i] = mean_emb
    model = model.eval()

    pattern = AttentionPattern(name="hierarchical_anchor", num_anchors=2, stride_base=2)

    docs = "\n\n".join(
        f"Document [{i}] (Title: T{i}): body of document {i}." for i in range(4)
    )
    text = wrap_documents(docs + "\n\nQuestion: which doc is special?")
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(next(model.parameters()).device)

    # Snapshot the state captured during the prefill; the pre-hook fires
    # again on every decoding step and would otherwise overwrite chunk_ids
    # with the (1, 1) length-1 input.
    prefill_snapshot = {}
    real_register = model.register_forward_pre_hook
    orig_state = None

    out1 = _generate_hierarchical(
        model, tok, input_ids,
        doc_start_id=doc_start_id, doc_end_id=doc_end_id,
        max_new_tokens=10, stop_token_ids={tok.eos_token_id},
        attention_pattern=pattern,
    )
    print(f"call 1 output: {out1!r}")

    state = model._hierarchical_state
    # State now reflects the *last* forward (a length-1 decode step);
    # to verify the prefill hook fired correctly, redo just the prefill.
    with torch.no_grad():
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["sdpa_hierarchical"] = model._hierarchical_fn
        _ = model(input_ids=input_ids, use_cache=False)
    cu = sorted(set(state["chunk_ids"].flatten().tolist()))
    print(f"prefill chunk_ids unique = {cu}  (expect [-1, 0, 1, 2, 3])")
    print(f"prefill layer_strides    = {state['layer_strides']}")

    n_pre_hooks_1 = len(model._forward_pre_hooks)

    # Second call shouldn't accumulate pre-hooks.
    out2 = _generate_hierarchical(
        model, tok, input_ids,
        doc_start_id=doc_start_id, doc_end_id=doc_end_id,
        max_new_tokens=10, stop_token_ids={tok.eos_token_id},
        attention_pattern=pattern,
    )
    print(f"call 2 output: {out2!r}")
    n_pre_hooks_2 = len(model._forward_pre_hooks)

    assert out1 and out2, "generation produced empty output"
    assert n_pre_hooks_1 == n_pre_hooks_2, (
        f"pre-hooks leaked: {n_pre_hooks_1} -> {n_pre_hooks_2}"
    )
    assert cu == [-1, 0, 1, 2, 3], f"unexpected chunk_ids: {cu}"
    print(f"\nALL CHECKS PASSED  (pre-hooks: {n_pre_hooks_1})")


if __name__ == "__main__":
    main()
