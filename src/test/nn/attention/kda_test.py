import pytest
import torch

from olmo_core.nn.attention import KimiDeltaAttentionConfig
from olmo_core.testing import requires_gpu
from olmo_core.testing.utils import requires_fla


def _init_real_weights(module, d_model: int, seed: int = 777) -> None:
    """Apply the real init: exp(A_log) in [1, 16] gives per-step decays of ~16 log2
    units per channel — the strong-decay regime where unbounded exp2 factorizations
    in the kernels overflow. config.build() alone leaves A_log as torch.empty garbage
    (typically tiny, i.e. weak decay), which masks that whole failure class."""
    from olmo_core.nn.transformer.init import InitMethod

    generator = torch.Generator(device=module.A_log.device).manual_seed(seed)
    with torch.no_grad():
        module.init_weights(
            init_method=InitMethod.normal,
            d_model=d_model,
            block_idx=0,
            num_blocks=12,
            generator=generator,
        )


@requires_fla
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(KimiDeltaAttentionConfig(n_heads=8), id="default"),
        pytest.param(KimiDeltaAttentionConfig(n_heads=8, head_dim=32), id="head_dim=32"),
        pytest.param(KimiDeltaAttentionConfig(n_heads=8, conv_size=8), id="conv_size=8"),
    ],
)
def test_kimi_delta_attention_config_num_params(config: KimiDeltaAttentionConfig):
    d_model = 512
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")
    assert config.num_params(d_model) == sum(p.numel() for p in module.parameters())


@requires_fla
@requires_gpu
def test_kimi_delta_attention_fwd_bwd():
    device = "cuda"
    dtype = torch.bfloat16
    d_model, seq_len, batch_size = 256, 64, 2
    config = KimiDeltaAttentionConfig(n_heads=2, head_dim=128)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    cu_doc_lens = torch.tensor([0, 32, 64, 96, 128], dtype=torch.int32, device=device)

    with torch.autocast(device_type=device, dtype=dtype):
        y = module(x, cu_doc_lens=cu_doc_lens)
        assert y.shape == x.shape
        y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@requires_fla
@requires_gpu
def test_kimi_delta_attention_cute_under_torch_compile():
    """The train module compiles each block; the cute host path must graph-break cleanly
    (dynamo tracing it used to die on torch.cuda.current_stream().cuda_stream)."""
    from olmo_core.nn.attention.kda_cute.chunk import _has_cute

    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("the CuTe KDA kernels require Blackwell (sm100+)")
    if not _has_cute():
        pytest.skip("CUTLASS CuTe DSL is not installed")

    device = "cuda"
    dtype = torch.bfloat16
    d_model, seq_len, batch_size = 256, 128, 2
    torch.manual_seed(0)
    config = KimiDeltaAttentionConfig(n_heads=2, head_dim=128, use_cute_kernel=True)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)
    _init_real_weights(module, d_model)
    compiled = torch.compile(module)
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

    with torch.autocast(device_type=device, dtype=dtype):
        y = compiled(x)
        assert y.shape == x.shape
        y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@requires_fla
@requires_gpu
def test_kimi_delta_attention_cute_matches_fla():
    from olmo_core.nn.attention.kda_cute.chunk import _has_cute

    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("the CuTe KDA kernels require Blackwell (sm100+)")
    if not _has_cute():
        pytest.skip("CUTLASS CuTe DSL is not installed")

    device = "cuda"
    dtype = torch.bfloat16
    # B * HV * (V / 64) = 4 * 16 * 4 = 256 CTAs, enough that the CuTe backward scans
    # engage instead of taking their small-grid fla fallback (_MIN_CTAS = 256).
    d_model, seq_len, batch_size = 256, 256, 4
    torch.manual_seed(0)
    config = KimiDeltaAttentionConfig(n_heads=16, head_dim=128, expand_v=2.0, allow_neg_eigval=True)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)
    _init_real_weights(module, d_model)
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

    results = {}
    for use_cute in (False, True):
        module.use_cute_kernel = use_cute
        module.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_(True)
        with torch.autocast(device_type=device, dtype=dtype):
            y = module(xi)
            y.square().mean().backward()
        assert xi.grad is not None
        results[use_cute] = (y.detach(), xi.grad, module.A_log.grad.clone())

    y_fla, dx_fla, dA_fla = results[False]
    y_cute, dx_cute, dA_cute = results[True]
    torch.testing.assert_close(y_cute, y_fla, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(dx_cute, dx_fla, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(dA_cute, dA_fla, atol=1e-2, rtol=1e-2)
