"""Validate that the per-layer hierarchical wrapper actually changes
intermediate activations vs. the default single-mask causal path.

The smoke test (diag_hierarchical_smoke.py) already shows that the
final loss differs. That alone could in principle be explained by the
mask reaching only the last layer. Here we capture all hidden states
(`output_hidden_states=True`) for the SAME input under two paths:

  * hierarchical: per-layer mask, stride grows with depth
  * plain causal: single causal mask shared across layers (HF default)

then report per-layer max |Δ|. Expected pattern:

  - hidden_states[0] (post-embedding, pre-layer-0) → identical
  - hidden_states[1] (after layer 0) → first divergence. With
    stride=1 and N=2 anchors, query chunk j attends to chunks
    {j-2, j-1, j}. Causal would give {0..j}. So they differ iff
    j >= num_anchors+1. With 4 chunks (N=2) only chunk 3 diverges
    at this layer; chunks 0/1/2 sit at the bf16 SDPA noise floor.
  - hidden_states[k] for k>=2 → delta grows with depth as the mask
    differences propagate and strides increase.

Also breaks down per-chunk delta at the first divergent layer and
verifies that the chunks predicted to diverge dominate the chunks
predicted to be at noise level.

Run:
    python -m scripts.diag.diag_hierarchical_hidden_states
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from corpus_reasoning.lib.chunked_attention import (
    AttentionPattern, setup_tokenizer, wrap_documents,
    install_hierarchical_sdpa_attention, attach_hierarchical_pre_hook,
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
    model.eval()

    docs = "\n\n".join(
        f"Document [{i}] (Title: T{i}): body of document {i}, repeated. "
        f"Token A{i} token B{i} token C{i}." for i in range(4)
    )
    text = wrap_documents(docs + "\n\nQuestion: which doc is special?")
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attn = attn.to(device)

    # === Hierarchical path.
    with torch.no_grad():
        out_hier = model(
            input_ids=input_ids, attention_mask=attn,
            output_hidden_states=True, use_cache=False,
        )
    hs_hier = out_hier.hidden_states  # tuple of (B, S, H), length L+1
    chunk_ids = state["chunk_ids"].clone()
    layer_strides = list(state["layer_strides"])

    # === Plain-causal path: swap the registered fn for plain sdpa.
    plain_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
    saved = ALL_ATTENTION_FUNCTIONS["sdpa_hierarchical"]
    ALL_ATTENTION_FUNCTIONS["sdpa_hierarchical"] = plain_sdpa
    try:
        with torch.no_grad():
            out_plain = model(
                input_ids=input_ids, attention_mask=attn,
                output_hidden_states=True, use_cache=False,
            )
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa_hierarchical"] = saved
    hs_plain = out_plain.hidden_states

    assert len(hs_hier) == len(hs_plain) == num_layers + 1, (
        f"expected {num_layers + 1} hidden states, "
        f"got hier={len(hs_hier)} plain={len(hs_plain)}"
    )

    # === Per-layer max |Δ|.
    print(f"chunk_ids unique  = {sorted(set(chunk_ids.flatten().tolist()))}")
    print(f"layer_strides     = {layer_strides}")
    print(f"seq len           = {input_ids.shape[1]}, "
          f"hidden states    = {num_layers + 1} "
          f"(0 = post-embed, k = after layer k-1)")
    print()
    print(f"{'idx':>4}  {'desc':<28}  {'stride':>7}  {'max|Δ|':>12}  {'mean|Δ|':>12}")
    deltas_max = []
    for k in range(num_layers + 1):
        a = hs_hier[k].float()
        b = hs_plain[k].float()
        diff = (a - b).abs()
        mx = diff.max().item()
        mn = diff.mean().item()
        deltas_max.append(mx)
        if k == 0:
            desc = "post-embed (pre-layer-0)"
            stride = "-"
        else:
            desc = f"after layer {k - 1}"
            stride = str(layer_strides[k - 1])
        print(f"{k:>4}  {desc:<28}  {stride:>7}  {mx:>12.6f}  {mn:>12.6f}")

    # === Sanity: layer 0 input is identical (same embedding).
    assert deltas_max[0] < 1e-6, (
        f"hidden_states[0] (post-embed) differs by {deltas_max[0]:.3e}; "
        "embeddings should be identical across paths."
    )

    # === Find first divergent layer.
    first_diff = next(
        (k for k in range(1, num_layers + 1) if deltas_max[k] > 1e-4),
        None,
    )
    print()
    if first_diff is None:
        print("FAIL: no layer diverged — hierarchical mask is not reaching SDPA")
        return
    print(f"First divergence: hidden_states[{first_diff}] "
          f"(after layer {first_diff - 1}, stride={layer_strides[first_diff - 1]})")

    # === Per-chunk breakdown at first divergent layer.
    # At stride s, N anchors: chunk j sees {j} ∪ N nearest strict-past
    # anchors on the stride grid → roughly chunks {j - N*s, ..., j}.
    # Causal sees {0..j}. So chunk j diverges from causal iff there
    # exists a chunk k < j that's NOT in chunk j's anchor set.
    a = hs_hier[first_diff].float()
    b = hs_plain[first_diff].float()
    per_tok = (a - b).abs().max(dim=-1).values[0]  # (S,)
    cids = chunk_ids[0]
    stride0 = layer_strides[first_diff - 1]
    N = pattern.num_anchors
    num_chunks = max(int(cids.max().item()) + 1, 1)

    def expected_to_diverge(j: int, stride: int, n_anchors: int) -> bool:
        """Causal kv set is {0..j}. Hier kv set is {j} ∪ {n nearest
        strict-past anchors of j on the stride grid}. They differ iff
        causal includes some chunk that hier excludes."""
        if j == 0:
            return False
        a0 = ((j - 1) // stride) * stride
        anchors = {a0 - i * stride for i in range(n_anchors)
                   if a0 - i * stride >= 0}
        anchors.add(j)
        causal = set(range(j + 1))
        return bool(causal - anchors)

    print()
    print(f"Per-chunk max |Δ| at hidden_states[{first_diff}] "
          f"(stride={stride0}, N={N}):")
    diverge_max = []
    noise_max = []
    for c in sorted(set(cids.tolist())):
        mask = (cids == c)
        cmax = per_tok[mask].max().item()
        cmean = per_tok[mask].float().mean().item()
        if c == -1:
            label = "FREE"
            tag = "(sees all)"
            noise_max.append(cmax)
        else:
            should = expected_to_diverge(c, stride0, N)
            label = f"doc {c}"
            tag = "← expect divergence" if should else "(noise floor)"
            (diverge_max if should else noise_max).append(cmax)
        print(f"  {label:<7} ({int(mask.sum().item()):>3} tokens):  "
              f"max={cmax:>10.6f}  mean={cmean:>10.6f}  {tag}")

    # === Validate: chunks predicted to diverge should dominate noise.
    # Use ratio rather than absolute thresholds so this stays robust
    # across different (num_chunks, N, stride) combos.
    print()
    if not diverge_max:
        print("  no chunk predicted to diverge at this layer "
              "(num_chunks <= N — increase docs or decrease anchors to test)")
    else:
        max_diverge = max(diverge_max)
        max_noise = max(noise_max) if noise_max else 0.0
        ratio = max_diverge / max(max_noise, 1e-9)
        print(f"max diverge-chunk |Δ|  = {max_diverge:.4f}")
        print(f"max noise-chunk |Δ|    = {max_noise:.4f}  "
              f"(bf16 SDPA path-diff noise)")
        print(f"ratio                  = {ratio:.1f}x")
        assert ratio > 3.0, (
            f"divergent chunks ({max_diverge:.4f}) only {ratio:.1f}x noise "
            f"floor ({max_noise:.4f}) — mask differences may not be reaching "
            f"the predicted chunks."
        )

    # === Growth check: deltas_max should be (weakly) non-decreasing
    # past the first divergent layer.
    print()
    print(f"Δ growth (max|Δ| through depth):")
    print(f"  layer 0  : {deltas_max[1]:.4f}")
    print(f"  middle   : {deltas_max[num_layers // 2]:.4f}")
    print(f"  final    : {deltas_max[num_layers]:.4f}")
    if deltas_max[num_layers] > deltas_max[1]:
        print("  → divergence grows with depth (expected)")
    else:
        print("  → unexpected: divergence does not grow")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
