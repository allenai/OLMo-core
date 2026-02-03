#!/usr/bin/env python3
"""Minimal reproduction of doc_lens issue in OLMo-core.

This script demonstrates that when two identical sequences are packed together
with doc_lens, they produce DIFFERENT logits - even though they should be
processed independently and produce identical results.

To run:
    python scripts/debug/minimal_doc_lens_repro.py

Expected behavior:
    Two identical documents packed with doc_lens should produce identical logits.

Actual behavior:
    The logits differ, indicating doc_lens is not correctly isolating documents.
"""

import torch

from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.transformer import TransformerConfig


def main():
    print("Building OLMo-core model with Flash Attention...")
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
    model = config.build().cuda().to(torch.bfloat16)
    model.eval()

    seq_len = 20
    seq = torch.randint(1, 100352, (1, seq_len)).cuda()
    print(f"Input sequence shape: {seq.shape}")
    print(f"Input tokens: {seq[0, :10].tolist()}...")

    batched_input = seq.repeat(2, 1)
    packed_input = seq.repeat(1, 2)
    doc_lens = torch.tensor([seq_len, seq_len], device="cuda")

    print(f"\nBatched input shape: {batched_input.shape}")
    print(f"Packed input shape: {packed_input.shape}")
    print(f"doc_lens: {doc_lens.tolist()}")

    print("\nRunning forward passes...")
    with torch.no_grad():
        logits_batched = model(batched_input)
        logits_packed = model(packed_input, doc_lens=doc_lens, max_doc_lens=[seq_len])

    logits_batched_seq0 = logits_batched[0]
    logits_batched_seq1 = logits_batched[1]
    logits_packed_doc0 = logits_packed[0, :seq_len]
    logits_packed_doc1 = logits_packed[0, seq_len:]

    batched_seq0_matches_seq1 = torch.allclose(logits_batched_seq0, logits_batched_seq1, atol=1e-5)
    packed_doc0_matches_doc1 = torch.allclose(logits_packed_doc0, logits_packed_doc1, atol=1e-5)
    batched_matches_packed = torch.allclose(logits_batched_seq0, logits_packed_doc0, atol=1e-3)

    max_diff_batched = (logits_batched_seq0 - logits_batched_seq1).abs().max().item()
    max_diff_packed = (logits_packed_doc0 - logits_packed_doc1).abs().max().item()
    max_diff_batched_vs_packed = (logits_batched_seq0 - logits_packed_doc0).abs().max().item()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(
        f"Batched: seq0 matches seq1: {batched_seq0_matches_seq1} (max diff: {max_diff_batched:.6f})"
    )
    print(
        f"Packed:  doc0 matches doc1: {packed_doc0_matches_doc1} (max diff: {max_diff_packed:.6f})"
    )
    print(
        f"Batched vs Packed doc0:     {batched_matches_packed} (max diff: {max_diff_batched_vs_packed:.6f})"
    )

    print("\nSample logits at position 5:")
    print(f"  Batched seq0[5, :5]: {logits_batched_seq0[5, :5].tolist()}")
    print(f"  Batched seq1[5, :5]: {logits_batched_seq1[5, :5].tolist()}")
    print(f"  Packed doc0[5, :5]:  {logits_packed_doc0[5, :5].tolist()}")
    print(f"  Packed doc1[5, :5]:  {logits_packed_doc1[5, :5].tolist()}")

    print("\n" + "=" * 60)
    if not packed_doc0_matches_doc1:
        print("BUG CONFIRMED: Packed forward with doc_lens produces different")
        print("logits for identical documents. This should not happen.")
        print("=" * 60)
        return 1
    else:
        print("No bug detected - packed documents match.")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    exit(main())
