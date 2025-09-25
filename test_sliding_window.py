#!/usr/bin/env python
"""Debug sliding window attention."""

import torch
from src.olmo_core.nn.attention import Attention, get_flex_attn_causal_block_mask

def test_sliding_window():
    device = torch.device("cpu")
    dtype = torch.float32

    torch.manual_seed(42)

    d_model = 128
    seq_len = 32
    batch_size = 2
    n_heads = 8
    window_size = 16

    # Create attention modules
    # SDPA doesn't support window_size, so we don't set it
    sdpa_attn = Attention(
        d_model=d_model,
        n_heads=n_heads,
        init_device=device.type,
        use_flash=False,
        use_flex=False,
    )

    flex_attn = Attention(
        d_model=d_model,
        n_heads=n_heads,
        init_device=device.type,
        window_size=window_size,
        use_flash=False,
        use_flex=True,
    )

    print(f"Window size: {window_size}")
    print(f"Attention.window_size: {flex_attn.window_size}")

    # Copy weights
    with torch.no_grad():
        flex_attn.w_q.load_state_dict(sdpa_attn.w_q.state_dict())
        flex_attn.w_k.load_state_dict(sdpa_attn.w_k.state_dict())
        flex_attn.w_v.load_state_dict(sdpa_attn.w_v.state_dict())
        flex_attn.w_out.load_state_dict(sdpa_attn.w_out.state_dict())

    # Create block mask with sliding window
    block_mask = get_flex_attn_causal_block_mask(
        seq_len,
        device,
        flex_attn.window_size,  # (15, 0)
        block_size=8
    )

    # Create inputs in the same format as test_sdpa
    q = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads, dtype=dtype, device=device)

    # Create attention mask for SDPA with sliding window
    attn_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril(diagonal=0)
    attn_mask = torch.logical_and(
        attn_mask,
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(diagonal=-window_size+1)
    )

    print(f"Attention mask shape: {attn_mask.shape}")
    print(f"Attention mask (first 8x8):")
    print(attn_mask[:8, :8].int())

    with torch.no_grad():
        # Run SDPA version with attention mask (since it doesn't support window_size)
        # Convert to the format SDPA expects
        q_sdpa = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)

        # Use math backend for consistency with flex attention
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            y_sdpa = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=attn_mask,
                is_causal=False,
                scale=1.0/torch.sqrt(torch.tensor(d_model // n_heads, dtype=torch.float32))
            )
        y_sdpa = y_sdpa.transpose(1, 2)  # Back to (batch, seq_len, n_heads, head_dim)

        # Run flex attention version
        y_flex = flex_attn.sdpa(q, k, v, block_mask=block_mask)

    # Compare outputs
    try:
        torch.testing.assert_close(y_sdpa, y_flex, rtol=1e-3, atol=1e-4)
        print("✓ Test passed!")
    except AssertionError as e:
        print(f"✗ Test failed!")
        diff = (y_sdpa - y_flex).abs()
        print(f"Max absolute difference: {diff.max().item():.6f}")
        print(f"Mean absolute difference: {diff.mean().item():.6f}")

if __name__ == "__main__":
    test_sliding_window()