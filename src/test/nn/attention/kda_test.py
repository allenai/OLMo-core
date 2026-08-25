import pytest
import torch

from olmo_core.nn.attention import KimiDeltaAttentionConfig
from olmo_core.testing import requires_gpu
from olmo_core.testing.utils import requires_fla

# The vendored kernels carry two independent CTA floors and quietly hand the work back to
# FLA below either one: the chain needs B * HV * (V // 64) >= 256, and the MMA intra
# backward needs B * (T / 64) * HV >= 1024 of its own. Every cute arm below is sized to
# clear both — B=4, T=1024, HV=16, V=256 gives exactly 256 and 1024 — because a test run
# under a floor measures FLA against FLA and reports it as a pass.
_CUTE_B, _CUTE_T, _CUTE_HEADS, _CUTE_DMODEL = 4, 1024, 16, 256


def _skip_unless_cute() -> None:
    """Skip unless these kernels would actually engage on this box.

    Asks the kernels' own predicate rather than re-deriving it here: it owns the arch gate
    (sm100 exactly — sm_120 is consumer Blackwell with no tcgen05), the DSL probe and the
    shape floors, and a copy of that logic here would drift from the one that decides.
    """
    import torch

    from olmo_core.nn.attention.kda_cute import is_supported

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    kw = {"device": "cuda", "dtype": torch.bfloat16}
    q = torch.empty(_CUTE_B, _CUTE_T, _CUTE_HEADS, 128, **kw)
    v = torch.empty(_CUTE_B, _CUTE_T, _CUTE_HEADS, 256, **kw)
    ok, reason = is_supported(q, v, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True)
    if not ok:
        pytest.skip(f"the cute KDA kernels decline this box: {reason}")


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
    _skip_unless_cute()

    device = "cuda"
    dtype = torch.bfloat16
    d_model, seq_len, batch_size = _CUTE_DMODEL, _CUTE_T, _CUTE_B
    torch.manual_seed(0)
    config = KimiDeltaAttentionConfig(
        n_heads=_CUTE_HEADS, head_dim=128, expand_v=2.0, use_cute_kernel=True
    )
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
def test_kimi_delta_attention_cute_extreme_decay():
    """Deterministic worst case for the gate exponents: A_log at its init maximum
    (log 16) and large softplus inputs give per-step decays of ~90 log2 units per
    channel. Any two-sided exp2 factorization in a kernel overflows fp32 here and
    NaNs — this regime is what the 30m ladder hit on its first optimizer step.

    Sized to clear both CTA floors so the MMA intra backward — the stage whose diagonal
    blocks carry the one-sided-exp2 contract — actually runs. Under a floor this arm would
    exercise the Triton fallback and pass without ever reaching the kernel it guards.
    This now also drives the gate activation through FLA's fused cumsum rather than eager
    fp32 torch ops, which is where the exponent is formed in the first place.
    """
    from olmo_core.nn.attention.flash_linear_attn_api import dispatch_chunk_kda

    _skip_unless_cute()

    device, dtype = "cuda", torch.bfloat16
    B, T, H, K, V = _CUTE_B, _CUTE_T, _CUTE_HEADS, 128, 256
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g_raw = torch.randn(B, T, H, K, device=device, dtype=torch.float32) + 4.0
    beta = torch.rand(B, T, H, device=device, dtype=torch.float32) * 2
    A_log = torch.full((H,), 16.0, device=device).log()
    dt_bias = torch.zeros(H * K, device=device)
    do = torch.randn(B, T, H, V, device=device, dtype=dtype)

    results = {}
    for use_cute in (False, True):
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g_raw, beta, A_log)]
        o, _ = dispatch_chunk_kda(
            q=leaves[0],
            k=leaves[1],
            v=leaves[2],
            g=leaves[3],
            beta=leaves[4],
            A_log=leaves[5],
            dt_bias=dt_bias,
            scale=K**-0.5,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_cute_kernel=use_cute,
        )
        (o.float() * do.float()).sum().backward()
        grads = [t.grad for t in leaves]
        for gr in [o, *grads]:
            assert gr is not None and torch.isfinite(gr).all()
        results[use_cute] = (o.detach(), grads)

    o_fla, g_fla = results[False]
    o_cute, g_cute = results[True]
    torch.testing.assert_close(o_cute, o_fla, atol=5e-3, rtol=5e-3)
    for gc, gf in zip(g_cute, g_fla):
        torch.testing.assert_close(gc.float(), gf.float(), atol=2e-2, rtol=2e-2)


@requires_fla
@requires_gpu
def test_kimi_delta_attention_cute_matches_fla():
    _skip_unless_cute()

    device = "cuda"
    dtype = torch.bfloat16
    d_model, seq_len, batch_size = _CUTE_DMODEL, _CUTE_T, _CUTE_B
    torch.manual_seed(0)
    config = KimiDeltaAttentionConfig(
        n_heads=_CUTE_HEADS, head_dim=128, expand_v=2.0, allow_neg_eigval=True
    )
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
