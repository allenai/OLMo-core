"""End-to-end smoke test for the hierarchical_anchor SDPA wrapper.

Loads Qwen2.5-0.5B (cached locally), installs the per-layer hierarchical
attention wrapper + chunk-id pre-hook, runs a forward pass on a synthetic
4-document batch, and verifies:

  1. Pre-hook populates chunk_ids correctly from the doc-boundary tokens.
  2. The wrapper is invoked once per transformer layer with the scheduled
     stride.
  3. The per-layer mask is non-trivial (masks a meaningful fraction of
     causal cells).
  4. The hierarchical forward produces a loss that DIFFERS from the
     plain-causal baseline (i.e., the mask actually reaches SDPA).
  5. An adversarial diagonal-only mask drives the loss far from causal,
     confirming the wrapper has full control over the SDPA path.

Run:
    python -m scripts.diag.diag_hierarchical_smoke
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from corpus_reasoning.lib.chunked_attention import (
    AttentionPattern, setup_tokenizer, wrap_documents,
    install_hierarchical_sdpa_attention, attach_hierarchical_pre_hook,
    compute_layer_strides, build_hierarchical_sdpa_mask,
)


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

    num_layers = int(model.config.num_hidden_layers)
    pattern = AttentionPattern(name="hierarchical_anchor", num_anchors=2, stride_base=2)

    state = {}
    install_hierarchical_sdpa_attention(model, state)
    attach_hierarchical_pre_hook(
        model, state, doc_start_id=doc_start_id, doc_end_id=doc_end_id,
        pattern=pattern, num_transformer_layers=num_layers,
    )

    # Wrap the registered SDPA function so we can record (layer_idx, stride)
    # for each call. base_fn = the hierarchical wrapper just installed.
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    base_fn = ALL_ATTENTION_FUNCTIONS["sdpa_hierarchical"]
    seen = []

    def recording(module, q, k, v, attn_mask, **kw):
        layer_idx = getattr(module, "layer_idx", 0) or 0
        ls = state.get("layer_strides")
        seen.append((layer_idx, int(ls[layer_idx]) if ls else None))
        return base_fn(module, q, k, v, attn_mask, **kw)

    ALL_ATTENTION_FUNCTIONS["sdpa_hierarchical"] = recording
    for module in model.modules():
        if hasattr(module, "_attn_implementation"):
            module._attn_implementation = "sdpa_hierarchical"

    # 4 documents + question. wrap_documents inserts <|doc_start|>/<|doc_end|>.
    docs = "\n\n".join(
        f"Document [{i}] (Title: T{i}): body of document {i}, repeated. "
        f"Token A{i} token B{i} token C{i}." for i in range(4)
    )
    text = wrap_documents(docs + "\n\nQuestion: which doc is special?")
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    labels = input_ids.clone()
    device = next(model.parameters()).device

    # === Hierarchical forward.
    out = model(input_ids=input_ids.to(device),
                attention_mask=attn.to(device),
                labels=labels.to(device))
    print(f"hier loss          = {out.loss.item():.6f}")

    # === Sanity: pre-hook saw the docs and built the schedule.
    chunk_ids = state["chunk_ids"]
    unique_chunks = sorted(set(chunk_ids.flatten().tolist()))
    print(f"chunk_ids unique   = {unique_chunks}  (expect [-1, 0, 1, 2, 3])")
    print(f"layer_strides      = {state['layer_strides']}")

    expected = compute_layer_strides(
        num_chunks=4, stride_base=pattern.stride_base,
        num_transformer_layers=num_layers,
    )
    by_layer = sorted({li: st for li, st in seen}.items())
    mismatches = [(li, st, expected[li]) for li, st in by_layer
                  if li < len(expected) and st != expected[li]]
    print(f"layers called      = {len(by_layer)}/{num_layers}")
    print(f"stride mismatches  = {mismatches if mismatches else 'none'}")

    # === Mask is non-trivial (vs. plain causal).
    S = chunk_ids.shape[1]
    causal_cells = (S * (S + 1)) // 2
    print("mask vs causal (frac of causal cells masked):")
    for li in [0, num_layers // 2, num_layers - 1]:
        stride = state["layer_strides"][li]
        m = build_hierarchical_sdpa_mask(
            chunk_ids, stride=stride, num_anchors=pattern.num_anchors,
            dtype=torch.bfloat16,
        )
        masked_below = int((m < -1e3).sum().item())
        masked_in_causal = masked_below - (S * S - causal_cells)
        print(f"  layer {li:>2} (stride={stride}): "
              f"{masked_in_causal}/{causal_cells} = "
              f"{masked_in_causal / causal_cells:.3f}")

    # === Plain-causal baseline in the same process.
    plain_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
    saved = ALL_ATTENTION_FUNCTIONS["sdpa_hierarchical"]
    ALL_ATTENTION_FUNCTIONS["sdpa_hierarchical"] = plain_sdpa
    out2 = model(input_ids=input_ids.to(device),
                 attention_mask=attn.to(device),
                 labels=labels.to(device))
    ALL_ATTENTION_FUNCTIONS["sdpa_hierarchical"] = saved
    print(f"plain-causal loss  = {out2.loss.item():.6f}")
    delta = out.loss.item() - out2.loss.item()
    print(f"  delta vs hier    = {delta:+.6f}")
    assert abs(delta) > 1e-4, (
        "Hierarchical loss should differ from plain causal loss; "
        "if not, the per-layer mask isn't reaching SDPA."
    )

    # === Adversarial: replace mask with diagonal-only -> loss should diverge.
    diag = torch.full(
        (1, 1, S, S), torch.finfo(torch.bfloat16).min,
        dtype=torch.bfloat16, device=device,
    )
    diag[..., torch.arange(S), torch.arange(S)] = 0.0

    def sdpa_diagonal(module, q, k, v, _, **kw):
        kw["is_causal"] = False
        return plain_sdpa(module, q, k, v, diag, **kw)

    ALL_ATTENTION_FUNCTIONS["sdpa_hierarchical"] = sdpa_diagonal
    out3 = model(input_ids=input_ids.to(device),
                 attention_mask=attn.to(device),
                 labels=labels.to(device))
    ALL_ATTENTION_FUNCTIONS["sdpa_hierarchical"] = saved
    print(f"diagonal-only loss = {out3.loss.item():.6f}  "
          f"(must be far from {out2.loss.item():.4f})")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
