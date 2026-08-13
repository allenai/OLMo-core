import logging

import pytest
import torch

import olmo_core.ops.gnorm as gnorm
from olmo_core.ops.gnorm import (
    GNORM_CUTE_MAX_DIM,
    GNormBackend,
    gnorm_cute_unsupported_reason,
    rms_norm_gated,
)
from olmo_core.testing import requires_gnorm_cute, requires_gpu
from olmo_core.testing.utils import requires_fla

# The budgets the kernels were developed against (kernel-fun-2, kernels/gnorm/spec.py), as
# max-abs-difference relative to the reference's scale. Both arms here are real kernels rather
# than an fp32 oracle, and the recorded runs sit 40-80x inside these, so a failure means
# something is wrong rather than the tolerance being tight.
TOL = {
    "y": 0.005,
    "dx": 0.008,
    "dg": 0.008,
    # dw sums a product over every row in the tensor, so its error is about summation order.
    "dw": 0.02,
}


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max absolute difference relative to the reference's scale — fla's own error metric."""
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-6)).item()


def _make_inputs(
    B: int,
    T: int,
    HV: int,
    D: int,
    dtype: torch.dtype = torch.bfloat16,
    weight_dtype: torch.dtype = torch.float32,
    device: str = "cuda",
):
    """
    Deterministic inputs in the layer's layout: ``x`` and ``g`` are ``[B, T, HV, D]``, exactly
    as :class:`~olmo_core.nn.attention.recurrent.GatedDeltaNet` hands them over.

    Gaussian rather than uniform: ``x`` is an attention output and ``g`` a linear projection of
    the hidden state, and both are centered in training. A uniform-positive gate would keep
    ``sigmoid(g)`` away from its flat regions and hide exactly the rounding this op can get
    wrong. The weight is perturbed off ones because against ones an implementation that ignored
    it entirely would pass.
    """
    gen = torch.Generator(device=device).manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)

    return {
        "x": randn(B, T, HV, D).to(dtype),
        "g": randn(B, T, HV, D).to(dtype),
        "w": (1.0 + 0.1 * randn(D)).to(weight_dtype).contiguous(),
        "dy": randn(B, T, HV, D).to(dtype),
    }


def _fwd_bwd(inputs: dict, backend: GNormBackend, activation: str = "swish", eps: float = 1e-5):
    x = inputs["x"].detach().clone().requires_grad_()
    g = inputs["g"].detach().clone().requires_grad_()
    w = inputs["w"].detach().clone().requires_grad_()

    y = rms_norm_gated(x, g, w, eps=eps, activation=activation, backend=backend)
    y.backward(inputs["dy"])

    assert x.grad is not None and g.grad is not None and w.grad is not None
    return {"y": y.detach(), "dx": x.grad, "dg": g.grad, "dw": w.grad}


@requires_gnorm_cute
@pytest.mark.parametrize(
    "D, activation, dtype",
    [
        # The production configuration: head_v_dim=256, swish, bf16.
        pytest.param(256, "swish", torch.bfloat16, id="prod"),
        # The other activation fla supports on this path. Its gradient is a different
        # expression, not a scaled version of swish's, so it's a real second code path.
        pytest.param(256, "sigmoid", torch.bfloat16, id="sigmoid"),
        # D sets the per-lane vector width (D/32), so smaller values catch anything that
        # hardcoded 256.
        pytest.param(128, "swish", torch.bfloat16, id="D=128"),
        pytest.param(64, "swish", torch.bfloat16, id="D=64"),
        # Catches accumulation assumptions that only hold for bf16's exponent range.
        pytest.param(256, "swish", torch.float16, id="fp16"),
    ],
)
def test_rms_norm_gated_cute_matches_fla(D: int, activation: str, dtype: torch.dtype):
    inputs = _make_inputs(2, 256, 4, D, dtype=dtype)
    assert gnorm_cute_unsupported_reason(inputs["x"], inputs["g"], inputs["w"]) is None

    cute = _fwd_bwd(inputs, GNormBackend.cute, activation=activation)
    fla = _fwd_bwd(inputs, GNormBackend.fla, activation=activation)

    errs = {name: _rel_err(cute[name], fla[name]) for name in TOL}
    assert all(errs[name] <= TOL[name] for name in TOL), errs


@requires_gnorm_cute
def test_rms_norm_gated_cute_matches_fla_with_low_precision_weight():
    """
    A bf16 norm weight has to work too, and ``dw`` has to come back in the parameter's dtype.

    The kernels only take an fp32 weight, so this exercises the cast in
    :class:`~olmo_core.ops.gnorm.RMSNormGatedCute` — the case a model that was moved to bf16
    wholesale, rather than trained with an fp32 master weight, would hit.
    """
    inputs = _make_inputs(2, 256, 4, 256, weight_dtype=torch.bfloat16)

    cute = _fwd_bwd(inputs, GNormBackend.cute)
    fla = _fwd_bwd(inputs, GNormBackend.fla)

    assert cute["dw"].dtype == torch.bfloat16
    errs = {name: _rel_err(cute[name], fla[name]) for name in TOL}
    assert all(errs[name] <= TOL[name] for name in TOL), errs


@requires_gnorm_cute
def test_rms_norm_gated_cute_does_not_alias_outputs():
    """
    Two calls with identical input layouts must not share an output buffer.

    The kernels cache their marshaled CuTe views keyed on input shape/stride/dtype. Every GDN
    layer in a stack norms the same shape, so an output owned by that cache would mean layer
    N+1's forward overwriting the ``y`` layer N saved for its backward. The kernels allocate
    outputs per call to avoid exactly this; the regression is invisible to a single call.
    """
    inputs = _make_inputs(2, 256, 4, 256)
    x, g, w = inputs["x"], inputs["g"], inputs["w"]

    first = rms_norm_gated(x, g, w, backend=GNormBackend.cute)
    snapshot = first.detach().clone()

    # Same layouts, different values, so a shared buffer would show up as a changed `first`.
    rms_norm_gated(g, x, w * 2, backend=GNormBackend.cute)

    torch.testing.assert_close(first, snapshot, rtol=0, atol=0)


@requires_gnorm_cute
def test_rms_norm_gated_cute_is_deterministic():
    """
    ``dw`` reduces over every row in the tensor. It does so through a fixed split and
    fixed-order folds rather than atomics, so repeated calls must agree bit for bit.
    """
    inputs = _make_inputs(2, 256, 4, 256)
    first = _fwd_bwd(inputs, GNormBackend.cute)
    second = _fwd_bwd(inputs, GNormBackend.cute)
    for name in ("y", "dx", "dg", "dw"):
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


@requires_gnorm_cute
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        pytest.param({"activation": "gelu"}, "activation", id="activation"),
        pytest.param({"dtype": torch.float32}, "bf16 or fp16", id="dtype"),
    ],
)
def test_rms_norm_gated_cute_rejects_unsupported(kwargs: dict, expected: str):
    kwargs = dict(kwargs)  # parametrize args are shared between runs; don't mutate in place
    inputs = _make_inputs(2, 256, 4, 256, dtype=kwargs.pop("dtype", torch.bfloat16))

    with pytest.raises(RuntimeError, match=expected):
        rms_norm_gated(inputs["x"], inputs["g"], inputs["w"], backend=GNormBackend.cute, **kwargs)


# requires_fla only tags the test `gpu`; the actual CUDA skip comes from requires_gpu.
@requires_fla
@requires_gpu
def test_rms_norm_gated_auto_falls_back_and_says_so(caplog):
    """
    Under ``auto``, an unsupported shape must still run — on fla — and log why.

    Too few rows for the backward's ``dw`` fold is outside the envelope on every GPU, so this
    covers the fallback on hardware with no CuTe path at all as well as on hardware that has
    one.
    """
    inputs = _make_inputs(1, 8, 1, 256)
    x, g, w = (t.detach().clone().requires_grad_() for t in (inputs["x"], inputs["g"], inputs["w"]))
    assert gnorm_cute_unsupported_reason(x, g, w) is not None

    gnorm._LOGGED.clear()  # the log fires once per process, so don't let test order decide it
    with caplog.at_level(logging.WARNING, logger="olmo_core.ops.gnorm"):
        y = rms_norm_gated(x, g, w, backend=GNormBackend.auto)

    assert y.shape == x.shape
    y.backward(inputs["dy"])
    assert x.grad is not None and w.grad is not None

    assert any("Falling back to the fla" in r.message for r in caplog.records), caplog.text


@requires_gpu
@pytest.mark.parametrize(
    "dims, expected",
    [
        pytest.param({"D": GNORM_CUTE_MAX_DIM + 32}, "row length", id="row-length"),
        pytest.param({"D": 48}, "row length", id="row-length-not-a-multiple-of-32"),
        # 4 rows short of a full block, and then a full block short of the dw fold's minimum.
        pytest.param({"B": 1, "T": 65, "HV": 4}, "multiple of", id="row-count"),
        pytest.param({"B": 1, "T": 8, "HV": 1}, "at least", id="too-few-rows"),
    ],
)
def test_gnorm_cute_unsupported_reason(dims: dict, expected: str):
    shape = {"B": 2, "T": 256, "HV": 4, "D": 256}
    shape.update(dims)
    B, T, HV, D = (shape[n] for n in ("B", "T", "HV", "D"))
    opts = {"device": "cuda", "dtype": torch.bfloat16}

    reason = gnorm_cute_unsupported_reason(
        torch.empty(B, T, HV, D, **opts),  # type: ignore[arg-type]
        torch.empty(B, T, HV, D, **opts),  # type: ignore[arg-type]
        torch.empty(D, device="cuda", dtype=torch.float32),
    )
    assert reason is not None
    # On a non-Blackwell GPU every shape is unsupported for the same reason, which is a true
    # answer but not the one under test.
    if "cutlass" not in reason:
        assert expected in reason
