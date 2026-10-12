from test.nn.attention.attention_test import BF16_ATOL, BF16_RTOL

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.attention import (
    AttentionConfig,
    GatedDeltaNetConfig,
    NemotronMamba2Config,
)
from olmo_core.testing import requires_gpu
from olmo_core.testing.utils import requires_fla
from olmo_core.utils import seed_all


@requires_fla
@pytest.mark.parametrize(
    "recurrent_config",
    [
        pytest.param(GatedDeltaNetConfig(n_heads=8), id="default"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, n_v_heads=16), id="GVA"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, head_dim=32), id="head_dim=32"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, expand_v=1.0), id="expand_v=1.0"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, conv_size=8, conv_bias=True), id="conv_bias"),
        pytest.param(
            GatedDeltaNetConfig(n_heads=8, allow_neg_eigval=False), id="allow_neg_eigval=False"
        ),
    ],
)
def test_gated_delta_net_config_num_params(recurrent_config: GatedDeltaNetConfig):
    d_model = 512
    module = recurrent_config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    # Make sure the estimated number of params matches the actual number of params.
    n_params = sum(p.numel() for p in module.parameters())
    assert recurrent_config.num_params(d_model) == n_params


@requires_fla
@requires_gpu
def test_gated_delta_net_fwd_bwd():
    device = "cuda"
    dtype = torch.bfloat16

    d_model, seq_len, batch_size = 256, 32, 2

    config = GatedDeltaNetConfig(n_heads=8)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

    with torch.autocast(device_type=device, dtype=dtype):
        y = module(x)
        assert y.shape == x.shape

        loss = y.sum()
        loss.backward()
    assert x.grad is not None


@requires_fla
@requires_gpu
@pytest.mark.parametrize(
    "recurrent_config",
    [
        pytest.param(GatedDeltaNetConfig(n_heads=8), id="default"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, n_v_heads=16), id="GVA"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, expand_v=1.0), id="expand_v=1.0"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, conv_size=8, conv_bias=True), id="conv_bias"),
    ],
)
def test_gated_delta_net_cached_decode_matches_full_forward(
    recurrent_config: GatedDeltaNetConfig,
):
    """
    Cached decoding (chunk prefill -> single-step recurrent decode, with conv + recurrent state
    caches) must reproduce a single full-sequence forward at the decoded positions.
    """
    device = "cuda"
    dtype = torch.bfloat16
    d_model, seq_len, batch_size, prefill_len = 256, 24, 2, 16

    module = recurrent_config.build(d_model, layer_idx=0, n_layers=12, init_device=device)
    seed_all(0)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(mean=0.0, std=0.02)

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

    module.eval()
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        # Reference: one full-sequence forward, no cache.
        y_full = module(x)

        # Cached: prefill the prompt, then decode the remaining tokens one at a time.
        module.init_state_cache(batch_size, seq_len)
        y_prefill = module(x[:, :prefill_len, :])
        decoded = [module(x[:, t : t + 1, :]) for t in range(prefill_len, seq_len)]
        y_decoded = torch.cat(decoded, dim=1)
        module.state_cache = None

    torch.testing.assert_close(
        y_prefill, y_full[:, :prefill_len, :], atol=BF16_ATOL, rtol=BF16_RTOL
    )
    torch.testing.assert_close(
        y_decoded, y_full[:, prefill_len:, :], atol=BF16_ATOL, rtol=BF16_RTOL
    )


@requires_fla
@requires_gpu
def test_gated_delta_net_cache_leftpad_matches_unpadded():
    """
    ``cache_leftpad`` (added for batched eval decode -- see
    ``corpus_reasoning.eval.batched_native_decode``) must make a left-padded row's REAL tokens
    bit-identical to a bs=1 forward of just those tokens with no padding at all.

    Unlike attention (causally + mask-excluded from ever seeing the pad prefix), GatedDeltaNet's
    causal-conv + delta-rule recurrent scan is not automatically invariant to whatever garbage sits
    in a left-padded prefix -- without the fix, the pad tokens' conv/decay contributions would leak
    into the real tokens' outputs. The batch mixes a zero-padded row with a row that needs NO
    padding at all, so a pass here also rules out an off-by-one in the per-row leftpad index.
    """
    device = "cuda"
    dtype = torch.bfloat16
    d_model = 256
    config = GatedDeltaNetConfig(n_heads=8)

    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)
    seed_all(0)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(mean=0.0, std=0.02)
    module.eval()

    real_lens = [20, 13]
    B = len(real_lens)
    W = max(real_lens)
    leftpads = [W - L for L in real_lens]

    torch.manual_seed(1)
    x_reals = [torch.randn(1, L, d_model, device=device, dtype=dtype) for L in real_lens]
    # Left-pad prefix filled with large-magnitude garbage (not zeros) -- if the mask were a no-op,
    # this would visibly perturb the real tokens' outputs.
    x_padded = torch.full((B, W, d_model), 0.0, device=device, dtype=dtype)
    for b, (lp, x_real) in enumerate(zip(leftpads, x_reals)):
        if lp > 0:
            x_padded[b, :lp] = torch.randn(lp, d_model, device=device, dtype=dtype) * 50.0
        x_padded[b, lp:] = x_real[0]
    leftpad = torch.tensor(leftpads, dtype=torch.int32, device=device)

    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        y_refs = [module(x_real) for x_real in x_reals]  # bs=1, no padding at all
        y_padded = module(x_padded, cache_leftpad=leftpad)

    for b, (lp, y_ref) in enumerate(zip(leftpads, y_refs)):
        torch.testing.assert_close(
            y_padded[b : b + 1, lp:], y_ref, atol=BF16_ATOL, rtol=BF16_RTOL
        )


@requires_fla
def test_gated_delta_net_num_flops_per_token():
    d_model, n_heads, seq_len = 256, 2, 8192

    gdn = GatedDeltaNetConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )
    attn = AttentionConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )

    # At long sequence lengths, recurrent layers use fewer FLOPs than quadratic attention.
    gdn_flops = gdn.num_flops_per_token(seq_len)
    attn_flops = attn.num_flops_per_token(seq_len)  # type: ignore
    # Training FLOPs apply a factor of 3 (forward + dgrad + wgrad): the base
    # 2-ops-per-MAC counts become 6, 6, and 24 respectively.
    linear_flops = 6 * sum(
        m.weight.numel() for m in (gdn.w_q, gdn.w_k, gdn.w_v, gdn.w_a, gdn.w_b, gdn.w_g, gdn.w_out)
    )
    conv_flops = 6 * gdn.conv_size * (gdn.key_dim + gdn.key_dim + gdn.value_dim)
    recurrent_flops = 24 * gdn.n_v_heads * gdn.head_k_dim * gdn.head_v_dim

    assert gdn_flops == linear_flops + conv_flops + recurrent_flops
    assert 0 < gdn_flops < attn_flops


def _small_nemotron_mamba2_config() -> NemotronMamba2Config:
    return NemotronMamba2Config(
        mamba_num_heads=4,
        mamba_head_dim=16,
        n_groups=1,
        ssm_state_size=16,
        chunk_size=8,
        dtype=DType.float32,
    )


def test_nemotron_mamba2_num_params():
    d_model = 64
    config = _small_nemotron_mamba2_config()
    module = config.build(d_model, layer_idx=0, n_layers=2, init_device="meta")
    actual = sum(p.numel() for p in module.parameters())
    assert config.num_params(d_model) == actual


def test_nemotron_mamba2_forward_runs_on_cpu():
    seed_all(0)
    d_model = 64
    module = _small_nemotron_mamba2_config().build(d_model, layer_idx=0, n_layers=2)
    x = torch.randn(2, 12, d_model)
    with torch.no_grad():
        out = module(x)
    assert out.shape == (2, 12, d_model)


def test_nemotron_mamba2_dt_bias_initialized_within_configured_range():
    seed_all(0)
    config = NemotronMamba2Config(
        mamba_num_heads=512,
        mamba_head_dim=8,
        n_groups=1,
        ssm_state_size=16,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=0.0001,
        dtype=DType.float32,
    )
    # init_weights drives the seeded generator; without it the constructor's reset_parameters uses
    # the global RNG (seeded by seed_all above), which is fine for the range check.
    module = config.build(64, layer_idx=0, n_layers=1)
    # softplus(dt_bias) must recover effective timesteps within the configured range, not the
    # softplus(1) == 1.31 that a plain ones-init would give.
    dt = torch.nn.functional.softplus(module.dt_bias.float())
    assert dt.min() >= config.time_step_floor - 1e-6
    assert dt.max() <= config.time_step_max + 1e-4


def test_nemotron_mamba2_dt_bias_uses_supplied_generator_not_global_rng():
    from olmo_core.nn.transformer.init import InitMethod

    config = _small_nemotron_mamba2_config()

    def init_dt_bias(global_seed: int) -> torch.Tensor:
        # Perturb the ambient global RNG differently each time.
        torch.manual_seed(global_seed)
        module = config.build(64, layer_idx=0, n_layers=2)
        generator = torch.Generator()
        generator.manual_seed(2026)
        module.init_weights(
            init_method=InitMethod.normal,
            d_model=64,
            block_idx=0,
            num_blocks=2,
            generator=generator,
        )
        return module.dt_bias.detach().clone()

    # An identical supplied generator must produce identical dt_bias regardless of global RNG state.
    torch.testing.assert_close(init_dt_bias(1), init_dt_bias(999))
