import torch

from olmo_core.nn.complex_rope import ComplexRotaryEmbedding


def test_complex_rope_head_first_vs_seq_first():
    B, T, d_model, n_heads = 2, 12, 16, 4
    c_rope = ComplexRotaryEmbedding(head_shape=d_model // n_heads)

    with torch.no_grad():
        q = torch.rand(B, n_heads, T, d_model // n_heads)
        k = torch.rand(B, n_heads, T, d_model // n_heads)

        q1, k1 = c_rope(q, k, head_first=True)

        q2, k2 = c_rope(q.transpose(1, 2), k.transpose(1, 2), head_first=False)
        q2 = q2.transpose(1, 2)
        k2 = k2.transpose(1, 2)

        torch.testing.assert_close(q1, q2)
        torch.testing.assert_close(k1, k2)
