import pytest
import torch

from olmo_core.nn.rope import (
    ComplexRotaryEmbedding,
    FusedRotaryEmbedding,
    RotaryEmbedding,
)

from ..utils import DEVICES, requires_flash_attn, requires_gpu


@pytest.mark.parametrize("device", DEVICES)
def test_rope_head_first_vs_seq_first(device):
    B, T, d_model, n_heads = 2, 12, 16, 4
    rope = RotaryEmbedding(head_size=d_model // n_heads)

    with torch.no_grad():
        q = torch.rand(B, n_heads, T, d_model // n_heads, device=device)
        k = torch.rand(B, n_heads, T, d_model // n_heads, device=device)

        q1, k1 = rope(q, k, head_first=True)

        q2, k2 = rope(q.transpose(1, 2), k.transpose(1, 2), head_first=False)
        q2 = q2.transpose(1, 2)
        k2 = k2.transpose(1, 2)

        torch.testing.assert_close(q1, q2)
        torch.testing.assert_close(k1, k2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "head_first",
    [
        pytest.param(True, id="head_first"),
        pytest.param(False, id="seq_first"),
    ],
)
def test_rope_with_past_key_values(device, head_first):
    B, T, d_model, n_heads = 2, 12, 16, 4
    rope = RotaryEmbedding(head_size=d_model // n_heads)

    with torch.no_grad():
        q = torch.rand(B, n_heads, T, d_model // n_heads, device=device)
        k = torch.rand(B, n_heads, T, d_model // n_heads, device=device)
        q_last = q[:, :, -1:, :]

        if not head_first:
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            q_last = q_last.transpose(1, 2)

        q_full, k_full = rope(q, k, head_first=head_first)
        q_part, k_part = rope(q_last, k, head_first=head_first)

        if not head_first:
            q_full, k_full = q_full.transpose(1, 2), k_full.transpose(1, 2)
            q_part, k_part = q_part.transpose(1, 2), k_part.transpose(1, 2)

        torch.testing.assert_close(q_full[:, :, -1:, :], q_part)
        torch.testing.assert_close(k_full, k_part)


@requires_gpu
@requires_flash_attn
@pytest.mark.parametrize(
    "dtype", [pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float32, id="fp32")]
)
def test_fused_rope(dtype):
    B, T, d_model, n_heads = 2, 12, 32, 4
    fused_rope = FusedRotaryEmbedding(head_size=d_model // n_heads)
    rope = RotaryEmbedding(head_size=d_model // n_heads)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        qkv = torch.rand(B, T, 3, n_heads, d_model // n_heads, device="cuda", dtype=dtype)
        q, k, _ = qkv.split(1, dim=2)
        q, k = q.squeeze(2), k.squeeze(2)
        qkv = fused_rope(qkv.clone())
        q, k = rope(q, k, head_first=False)
        torch.testing.assert_close(q, qkv[:, :, 0, :])
        torch.testing.assert_close(k, qkv[:, :, 1, :])


@pytest.mark.parametrize("device", DEVICES)
def test_complex_rope_head_first_vs_seq_first(device):
    B, T, d_model, n_heads = 2, 12, 16, 4
    rope = ComplexRotaryEmbedding(head_size=d_model // n_heads)

    with torch.no_grad():
        q = torch.rand(B, n_heads, T, d_model // n_heads, device=device)
        k = torch.rand(B, n_heads, T, d_model // n_heads, device=device)

        q1, k1 = rope(q, k, head_first=True)

        q2, k2 = rope(q.transpose(1, 2), k.transpose(1, 2), head_first=False)
        q2 = q2.transpose(1, 2)
        k2 = k2.transpose(1, 2)

        torch.testing.assert_close(q1, q2)
        torch.testing.assert_close(k1, k2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "head_first",
    [
        pytest.param(True, id="head_first"),
        pytest.param(False, id="seq_first"),
    ],
)
def test_complex_rope_with_past_key_values(device, head_first):
    B, T, d_model, n_heads = 2, 12, 16, 4
    rope = ComplexRotaryEmbedding(head_size=d_model // n_heads)

    with torch.no_grad():
        q = torch.rand(B, n_heads, T, d_model // n_heads, device=device)
        k = torch.rand(B, n_heads, T, d_model // n_heads, device=device)
        q_last = q[:, :, -1:, :]

        if not head_first:
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            q_last = q_last.transpose(1, 2)

        q_full, k_full = rope(q, k, head_first=head_first)
        q_part, k_part = rope(q_last, k, head_first=head_first)

        if not head_first:
            q_full, k_full = q_full.transpose(1, 2), k_full.transpose(1, 2)
            q_part, k_part = q_part.transpose(1, 2), k_part.transpose(1, 2)

        torch.testing.assert_close(q_full[:, :, -1:, :], q_part)
        torch.testing.assert_close(k_full, k_part)
