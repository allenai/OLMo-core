"""
Parity tests for :mod:`olmo_core.kernels.fused_rms_norm_gated` against the original fla
implementation (``fla.modules.fused_norm_gate``), which serves as the numerical oracle.

The vendored kernel keeps fla's fp32 accumulation and full-row reduction semantics, so with an
*identical launch config* the outputs match fla *bitwise*. Left to itself, ``@triton.autotune``
may pick different tile shapes for the two implementations (selection depends on timing noise),
which changes the fp32 reduction order by ~1 ulp and makes bitwise comparison flaky — so the
parity tests pin both autotuners to one shared config, and a separate unpinned test checks the
live-autotune path with numeric tolerances instead.
"""

import pytest
import torch

from olmo_core.testing.utils import requires_fla, requires_gpu
from olmo_core.utils import seed_all

# The single launch config used for bitwise parity runs (applied to both implementations).
PARITY_CONFIG = {"BT": 32, "num_warps": 4}


def _autotuner_of(kernel):
    obj = kernel
    while not hasattr(obj, "configs"):
        obj = obj.fn
    return obj


@pytest.fixture()
def pinned_configs():
    import fla.modules.fused_norm_gate as fla_fng
    import triton

    import olmo_core.kernels.fused_rms_norm_gated as vendored

    kernels = (
        vendored.rms_norm_gated_fwd_kernel,
        vendored.rms_norm_gated_bwd_kernel,
        fla_fng.layer_norm_gated_fwd_kernel,
        fla_fng.layer_norm_gated_bwd_kernel,
    )
    saved = []
    for kernel in kernels:
        tuner = _autotuner_of(kernel)
        saved.append((tuner, tuner.configs))
        tuner.configs = [
            triton.Config({"BT": PARITY_CONFIG["BT"]}, num_warps=PARITY_CONFIG["num_warps"])
        ]
        tuner.cache.clear()
    yield
    for tuner, configs in saved:
        tuner.configs = configs
        tuner.cache.clear()


# The row counts cover: a single row, tiny non-multiples of every candidate block size (BT),
# exact block-size boundaries, and a couple of large sizes including a non-multiple tail that
# exercises the persistent backward's tail-tile masking (T > SM count * BT).
ROW_COUNTS = [1, 5, 127, 128, 129, 2048, 8191, 262144]
D = 128  # head_v_dim in every GDN2 ladder config


def _run_fla(x, g, weight, activation, eps):
    from fla.modules.fused_norm_gate import rms_norm_gated as fla_rms_norm_gated

    return fla_rms_norm_gated(x, g, weight, None, activation=activation, eps=eps)


def _run_vendored(x, g, weight, activation, eps):
    from olmo_core.kernels.fused_rms_norm_gated import rms_norm_gated

    return rms_norm_gated(x, g, weight, activation=activation, eps=eps)


def _make_inputs(shape, dtype, device, affine=True):
    x = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    g = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    weight = None
    if affine:
        weight = torch.randn(shape[-1], dtype=dtype, device=device, requires_grad=True)
    return x, g, weight


def _clone_inputs(x, g, weight):
    x = x.detach().clone().requires_grad_(True)
    g = g.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True) if weight is not None else None
    return x, g, weight


def _assert_parity(shape, dtype, activation, eps=1e-5, affine=True):
    seed_all(42)
    x, g, weight = _make_inputs(shape, dtype, "cuda", affine=affine)
    x_ref, g_ref, weight_ref = _clone_inputs(x, g, weight)
    dy = torch.randn(shape, dtype=dtype, device="cuda")

    y = _run_vendored(x, g, weight, activation, eps)
    y_ref = _run_fla(x_ref, g_ref, weight_ref, activation, eps)
    assert torch.equal(y, y_ref), "forward output mismatch"

    y.backward(dy)
    y_ref.backward(dy)
    assert x.grad is not None and x_ref.grad is not None
    assert g.grad is not None and g_ref.grad is not None
    assert torch.equal(x.grad, x_ref.grad), "dx mismatch"
    assert torch.equal(g.grad, g_ref.grad), "dg mismatch"
    if affine:
        assert weight is not None and weight_ref is not None
        assert weight.grad is not None and weight_ref.grad is not None
        assert torch.equal(weight.grad, weight_ref.grad), "dw mismatch"


@requires_fla
@requires_gpu
@pytest.mark.parametrize("T", ROW_COUNTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("activation", ["sigmoid", "swish"])
def test_parity_2d(T: int, dtype: torch.dtype, activation: str, pinned_configs):
    _assert_parity((T, D), dtype, activation)


@requires_fla
@requires_gpu
@pytest.mark.parametrize("activation", ["sigmoid", "swish"])
def test_parity_4d(activation: str, pinned_configs):
    # The GDN2 call-site shape: (batch, seq, n_v_heads, head_v_dim).
    _assert_parity((2, 32, 4, D), torch.bfloat16, activation)


@requires_fla
@requires_gpu
@pytest.mark.parametrize("d", [64, 100, 512], ids=["d=64", "d=100", "d=512"])
def test_parity_other_dims(d: int, pinned_configs):
    # d=100 exercises the non-power-of-2 column masking; d=512 is the largest supported dim.
    _assert_parity((2048, d), torch.bfloat16, "sigmoid")


@requires_fla
@requires_gpu
def test_parity_no_affine(pinned_configs):
    _assert_parity((2048, D), torch.bfloat16, "sigmoid", affine=False)


@requires_fla
@requires_gpu
def test_live_autotune_close_to_fla():
    # With autotune unpinned the two implementations may pick different launch configs, which
    # perturbs the fp32 reduction order by ~1 ulp — so this checks numeric closeness, not
    # bitwise equality. It exists to exercise the real autotune path end to end.
    seed_all(42)
    shape = (8191, D)
    x, g, weight = _make_inputs(shape, torch.bfloat16, "cuda")
    x_ref, g_ref, weight_ref = _clone_inputs(x, g, weight)
    dy = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    y = _run_vendored(x, g, weight, "sigmoid", 1e-5)
    y_ref = _run_fla(x_ref, g_ref, weight_ref, "sigmoid", 1e-5)
    torch.testing.assert_close(y, y_ref)
    y.backward(dy)
    y_ref.backward(dy)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(g.grad, g_ref.grad)
    torch.testing.assert_close(weight.grad, weight_ref.grad)


@requires_fla
@requires_gpu
def test_parity_noncontiguous_inputs(pinned_configs):
    seed_all(42)
    # Build non-contiguous x/g by slicing a wider tensor along the last dim.
    x_full = torch.randn(2048, 2 * D, dtype=torch.bfloat16, device="cuda")
    g_full = torch.randn(2048, 2 * D, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(D, dtype=torch.bfloat16, device="cuda")

    x, g = x_full[:, :D], g_full[:, :D]
    assert not x.is_contiguous()
    y = _run_vendored(x, g, weight, "sigmoid", 1e-5)
    y_ref = _run_fla(x.contiguous(), g.contiguous(), weight, "sigmoid", 1e-5)
    assert torch.equal(y, y_ref)


@requires_fla
@requires_gpu
def test_module_matches_fla_module(pinned_configs):
    import fla.modules

    from olmo_core.kernels.fused_rms_norm_gated import FusedRMSNormGated

    seed_all(42)
    ours = FusedRMSNormGated(D, eps=1e-5, activation="sigmoid", device="cuda")
    theirs = fla.modules.FusedRMSNormGated(D, eps=1e-5, activation="sigmoid", device="cuda")
    with torch.no_grad():
        ours.weight.normal_()
    # State dicts are interchangeable (checkpoint compatibility).
    theirs.load_state_dict(ours.state_dict())

    x = torch.randn(2, 32, 4, D, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(2, 32, 4, D, dtype=torch.bfloat16, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
        assert torch.equal(ours(x, g), theirs(x, g))


@requires_gpu
def test_invalid_args_raise():
    from olmo_core.kernels.fused_rms_norm_gated import FusedRMSNormGated, rms_norm_gated

    with pytest.raises(ValueError, match="activation"):
        FusedRMSNormGated(D, activation="gelu")

    # Only the tiled D <= 512 kernel path is vendored.
    x = torch.randn(4, 1024, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(4, 1024, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="D <= 512"):
        rms_norm_gated(x, g, None, activation="sigmoid")
