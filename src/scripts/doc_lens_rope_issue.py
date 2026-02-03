"""Minimal reproduction: doc_lens does not reset RoPE positions per document.

This script demonstrates that when using doc_lens for intra-document masking,
RoPE positions are NOT reset per document. The second document in a packed
sequence gets RoPE positions that continue from the first document instead
of starting from 0.

Expected behavior:
    Both documents should have identical logits since they contain identical
    tokens and doc_lens should create independent attention contexts.

Actual behavior:
    The second document has different logits because it uses RoPE positions
    [seq_len, seq_len+1, ...] instead of [0, 1, ...].

Impact:
    This affects use cases like DPO training where chosen and rejected
    sequences are packed together. The rejected sequence gets incorrect
    RoPE positions, causing its log probabilities to differ from what
    they would be if processed separately.

To run:
    uv run python src/scripts/doc_lens_rope_issue.py

Requires: CUDA and flash attention (flash-attn package).
"""

import torch

from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.transformer import TransformerConfig


def main():
    print("=" * 70)
    print("OLMo-core doc_lens RoPE Issue Reproduction")
    print("=" * 70)

    print("\nCreating OLMo-core model with flash attention (2 layers)...")
    config = TransformerConfig.olmo2_1B(vocab_size=100352)
    config.n_layers = 2
    config.block.attention = AttentionConfig(
        n_heads=config.block.attention.n_heads,
        n_kv_heads=config.block.attention.n_kv_heads,
        bias=config.block.attention.bias,
        rope=config.block.attention.rope,
        qk_norm=config.block.attention.qk_norm,
        backend="flash_2",
    )
    model = config.build().cuda().to(torch.bfloat16).eval()

    seq_len = 10
    torch.manual_seed(42)
    seq = torch.randint(1, 100352, (1, seq_len), device="cuda")

    packed = torch.cat([seq, seq], dim=1)
    doc_lens = torch.tensor([seq_len, seq_len], device="cuda")

    print(f"\nSetup:")
    print(f"  - Single sequence of length {seq_len}: {seq[0, :5].tolist()}...")
    print(f"  - Packed two identical copies: [seq | seq], total length {2*seq_len}")
    print(f"  - doc_lens: {doc_lens.tolist()}")

    print("\nRunning forward passes...")
    with torch.no_grad():
        logits_packed = model(packed, doc_lens=doc_lens, max_doc_lens=[seq_len])
        logits_separate = model(seq)

    print("\n" + "=" * 70)
    print("Results: Position 0 logits (first 5 vocab entries)")
    print("=" * 70)
    print(f"  Packed doc1[0]:  {logits_packed[0, 0, :5].float().tolist()}")
    print(f"  Packed doc2[0]:  {logits_packed[0, seq_len, :5].float().tolist()}")
    print(f"  Separate[0]:     {logits_separate[0, 0, :5].float().tolist()}")

    print("\n" + "=" * 70)
    print("Results: Position 1 logits (first 5 vocab entries)")
    print("=" * 70)
    print(f"  Packed doc1[1]:  {logits_packed[0, 1, :5].float().tolist()}")
    print(f"  Packed doc2[1]:  {logits_packed[0, seq_len + 1, :5].float().tolist()}")
    print(f"  Separate[1]:     {logits_separate[0, 1, :5].float().tolist()}")

    pos0_doc1_matches = torch.allclose(
        logits_packed[0, 0, :], logits_separate[0, 0, :], atol=1e-3, rtol=1e-3
    )
    pos0_doc2_matches = torch.allclose(
        logits_packed[0, seq_len, :], logits_separate[0, 0, :], atol=1e-3, rtol=1e-3
    )
    pos1_doc1_matches = torch.allclose(
        logits_packed[0, 1, :], logits_separate[0, 1, :], atol=1e-3, rtol=1e-3
    )
    pos1_doc2_matches = torch.allclose(
        logits_packed[0, seq_len + 1, :], logits_separate[0, 1, :], atol=1e-3, rtol=1e-3
    )

    print("\n" + "=" * 70)
    print("Comparison (should all be True if doc_lens resets RoPE)")
    print("=" * 70)
    print(f"  pos0 doc1 matches separate: {pos0_doc1_matches}")
    print(f"  pos0 doc2 matches separate: {pos0_doc2_matches}")
    print(f"  pos1 doc1 matches separate: {pos1_doc1_matches}")
    print(f"  pos1 doc2 matches separate: {pos1_doc2_matches}  <-- FAILS")

    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)
    if pos1_doc1_matches and not pos1_doc2_matches:
        print("CONFIRMED: doc_lens does NOT reset RoPE positions per document.")
        print("")
        print("The first document (doc1) matches separate forward because it gets")
        print(f"RoPE positions [0, 1, ..., {seq_len-1}] which is correct.")
        print("")
        print("The second document (doc2) does NOT match because it gets")
        print(f"RoPE positions [{seq_len}, {seq_len+1}, ..., {2*seq_len-1}]")
        print(f"instead of [0, 1, ..., {seq_len-1}].")
        print("")
        print("doc_lens correctly masks attention (documents can't attend to each")
        print("other), but RoPE positions are applied globally across the packed")
        print("sequence instead of resetting per document.")
        return True
    elif pos1_doc1_matches and pos1_doc2_matches:
        print("FIXED: doc_lens now correctly resets RoPE positions per document!")
        return False
    else:
        print("Unexpected result - please investigate further.")
        print(f"  pos0_doc1_matches: {pos0_doc1_matches}")
        print(f"  pos0_doc2_matches: {pos0_doc2_matches}")
        print(f"  pos1_doc1_matches: {pos1_doc1_matches}")
        print(f"  pos1_doc2_matches: {pos1_doc2_matches}")
        return True


if __name__ == "__main__":
    has_issue = main()
    exit(0 if not has_issue else 1)
