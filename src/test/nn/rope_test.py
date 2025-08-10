import pytest
import torch

from olmo_core.nn.rope import (
    ABFRoPEScalingConfig,
    ComplexRotaryEmbedding,
    FusedRotaryEmbedding,
    PIRoPEScalingConfig,
    RotaryEmbedding,
    StepwiseRoPEScalingConfig,
    YaRNRoPEScalingConfig,
)
from olmo_core.testing import DEVICES, requires_flash_attn, requires_gpu


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


@pytest.mark.parametrize("device", DEVICES)
def test_abf_rope_scaling(device):
    B, T, d_model, n_heads = 2, 12, 16, 4
    head_size = d_model // n_heads

    # Test case where new_theta == theta, output should be identical
    rope_vanilla = RotaryEmbedding(head_size=head_size, theta=500_000)
    rope_abf = RotaryEmbedding(head_size=head_size, scaling=ABFRoPEScalingConfig(new_theta=500_000))

    with torch.no_grad():
        q = torch.rand(B, n_heads, T, head_size, device=device)
        k = torch.rand(B, n_heads, T, head_size, device=device)

        q1, k1 = rope_vanilla(q, k)
        q2, k2 = rope_abf(q.clone(), k.clone())

        torch.testing.assert_close(q1, q2, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(k1, k2, rtol=1e-5, atol=1e-5)

    # Test case where new_theta != theta, output should be different
    rope_abf_new_theta = RotaryEmbedding(
        head_size=head_size, scaling=ABFRoPEScalingConfig(new_theta=16_000_000)
    )

    with torch.no_grad():
        q3, k3 = rope_abf_new_theta(q.clone(), k.clone())

        assert not torch.allclose(q1, q3, rtol=1e-5, atol=1e-5)
        assert not torch.allclose(k1, k3, rtol=1e-5, atol=1e-5)

        assert q1.shape == q3.shape
        assert k1.shape == k3.shape


@pytest.mark.parametrize("device", DEVICES)
def test_pi_rope_scaling(device):
    B, T, d_model, n_heads = 2, 12, 16, 4
    head_size = d_model // n_heads

    # Test vanilla RoPE vs PI scaling with factor=1 (should be identical)
    rope_vanilla = RotaryEmbedding(head_size=head_size)
    rope_pi_factor1 = RotaryEmbedding(head_size=head_size, scaling=PIRoPEScalingConfig(factor=1.0))

    with torch.no_grad():
        q = torch.rand(B, n_heads, T, head_size, device=device)
        k = torch.rand(B, n_heads, T, head_size, device=device)

        q1, k1 = rope_vanilla(q, k)
        q2, k2 = rope_pi_factor1(q.clone(), k.clone())

        torch.testing.assert_close(q1, q2, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(k1, k2, rtol=1e-5, atol=1e-5)

    # Test PI scaling with factor=2 (should compress positions)
    rope_pi_factor2 = RotaryEmbedding(head_size=head_size, scaling=PIRoPEScalingConfig(factor=2.0))

    with torch.no_grad():
        q3, k3 = rope_pi_factor2(q.clone(), k.clone())

        # With factor=2, the embeddings should be different from vanilla
        assert not torch.allclose(q1, q3, rtol=1e-5, atol=1e-5)
        assert not torch.allclose(k1, k3, rtol=1e-5, atol=1e-5)

        # But the shapes should be the same
        assert q1.shape == q3.shape
        assert k1.shape == k3.shape


@pytest.mark.parametrize("device", DEVICES)
def test_per_frequency_rope_scaling(device):
    B, T, d_model, n_heads = 2, 12, 512, 4  # head_size = 128
    head_size = d_model // n_heads

    rope_vanilla = RotaryEmbedding(head_size=head_size)
    rope_per_freq = RotaryEmbedding(head_size=head_size, scaling=StepwiseRoPEScalingConfig())

    with torch.no_grad():
        q = torch.rand(B, n_heads, T, head_size, device=device)
        k = torch.rand(B, n_heads, T, head_size, device=device)

        q_van, k_van = rope_vanilla(q, k)
        q_pf, k_pf = rope_per_freq(q.clone(), k.clone())

        # Per-freq scaling keeps the very first (highest-frequency) rotary pair unchanged.
        torch.testing.assert_close(
            q_van[..., :1],  # first rotary pair → highest frequency
            q_pf[..., :1],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            k_van[..., :1],
            k_pf[..., :1],
            rtol=1e-5,
            atol=1e-5,
        )

        # Lower-frequency components *should* differ from vanilla RoPE.
        assert not torch.allclose(q_van[..., -1:], q_pf[..., -1:], rtol=1e-5, atol=1e-5)
        assert not torch.allclose(k_van[..., -1:], k_pf[..., -1:], rtol=1e-5, atol=1e-5)

        # Shapes stay identical.
        assert q_van.shape == q_pf.shape
        assert k_van.shape == k_pf.shape


@pytest.mark.parametrize("device", DEVICES)
def test_yarn_rope_scaling(device):
    B, T, d_model, n_heads = 2, 12, 512, 4  # head_size = 128
    head_size = d_model // n_heads

    rope_vanilla = RotaryEmbedding(head_size=head_size)
    rope_yarn = RotaryEmbedding(head_size=head_size, scaling=YaRNRoPEScalingConfig())

    with torch.no_grad():
        q = torch.rand(B, n_heads, T, head_size, device=device)
        k = torch.rand(B, n_heads, T, head_size, device=device)

        q_van, k_van = rope_vanilla(q, k)
        q_yarn, k_yarn = rope_yarn(q.clone(), k.clone())

        assert not torch.allclose(q_van, q_yarn, rtol=1e-5, atol=1e-5)
        assert not torch.allclose(k_van, k_yarn, rtol=1e-5, atol=1e-5)

        # Shapes stay identical.
        assert q_van.shape == q_yarn.shape
        assert k_van.shape == k_yarn.shape


@pytest.mark.parametrize("seq_len", [4, 8, 16, 32], ids=lambda t: f"T={t}")
def test_rope_scaling_with_different_seq_lengths(seq_len):
    B, d_model, n_heads = 2, 16, 4
    head_size = d_model // n_heads
    device = torch.device("cpu")

    rope = RotaryEmbedding(head_size=head_size, scaling=PIRoPEScalingConfig(factor=2.0))

    with torch.no_grad():
        q = torch.rand(B, n_heads, seq_len, head_size, device=device)
        k = torch.rand(B, n_heads, seq_len, head_size, device=device)

        q_out, k_out = rope(q, k)

        # Shapes must remain unchanged.
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

        # Ensure no NaN or Inf values are present.
        assert torch.isfinite(q_out).all()
        assert torch.isfinite(k_out).all()


@pytest.mark.parametrize(
    "scaling_config",
    [
        pytest.param(ABFRoPEScalingConfig(new_theta=8_000_000), id="abf"),
        pytest.param(PIRoPEScalingConfig(factor=2.0), id="pi"),
        pytest.param(
            PIRoPEScalingConfig(factor=4.0, attention_rescale_factor=1.2), id="pi_rescale"
        ),
        pytest.param(StepwiseRoPEScalingConfig(factor=16.0), id="perfreq"),
        pytest.param(YaRNRoPEScalingConfig(factor=8.0), id="yarn"),
    ],
)
def test_rope_scaling_consistency(scaling_config):
    B, T, d_model, n_heads = 2, 8, 16, 4
    head_size = d_model // n_heads
    device = torch.device("cpu")

    rope = RotaryEmbedding(head_size=head_size, scaling=scaling_config)

    with torch.no_grad():
        q = torch.rand(B, n_heads, T, head_size, device=device)
        k = torch.rand(B, n_heads, T, head_size, device=device)

        # Multiple calls should produce identical results
        # (1st call populates cache and 2nd call uses cache)
        q1, k1 = rope(q.clone(), k.clone())
        q2, k2 = rope(q.clone(), k.clone())

        torch.testing.assert_close(q1, q2)
        torch.testing.assert_close(k1, k2)


def test_rope_scaling_attention_rescale_factor():
    """Test attention rescale factor functionality."""
    B, T, d_model, n_heads = 2, 8, 16, 4
    head_size = d_model // n_heads
    device = torch.device("cpu")

    # Test with custom attention rescale factor
    rope_rescaled = RotaryEmbedding(
        head_size=head_size, scaling=PIRoPEScalingConfig(factor=2.0, attention_rescale_factor=1.5)
    )

    rope_normal = RotaryEmbedding(
        head_size=head_size, scaling=PIRoPEScalingConfig(factor=2.0, attention_rescale_factor=1.0)
    )

    with torch.no_grad():
        q = torch.rand(B, n_heads, T, head_size, device=device)
        k = torch.rand(B, n_heads, T, head_size, device=device)

        q1, k1 = rope_normal(q.clone(), k.clone())
        q2, k2 = rope_rescaled(q.clone(), k.clone())

        # With attention rescaling, the embeddings should be scaled
        expected_q2 = q1 * 1.5
        expected_k2 = k1 * 1.5

        torch.testing.assert_close(q2, expected_q2, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(k2, expected_k2, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "head_first",
    [pytest.param(True, id="head_first"), pytest.param(False, id="seq_first")],
)
@pytest.mark.parametrize(
    "rope_cls",
    [
        pytest.param(RotaryEmbedding, id="default"),
        pytest.param(ComplexRotaryEmbedding, id="complex"),
    ],
)
def test_rope_start_pos_zero_matches_default(device, head_first, rope_cls):
    B, T, d_model, n_heads = 2, 12, 16, 4
    head_size = d_model // n_heads
    rope = rope_cls(head_size=head_size)

    with torch.no_grad():
        q = torch.rand(B, n_heads, T, head_size, device=device)
        k = torch.rand(B, n_heads, T, head_size, device=device)
        if not head_first:
            q, k = q.transpose(1, 2), k.transpose(1, 2)

        # Default behavior
        q_def, k_def = rope(q.clone(), k.clone(), head_first=head_first)

        # Explicit start_pos = 0 should match default
        q_zero, k_zero = rope(q.clone(), k.clone(), head_first=head_first, start_pos=0)

        torch.testing.assert_close(q_def, q_zero)
        torch.testing.assert_close(k_def, k_zero)


# TODO: better test for start positions
