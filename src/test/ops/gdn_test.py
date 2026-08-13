import logging

import pytest
import torch
import torch.nn.functional as F

import olmo_core.ops.gdn as gdn
from olmo_core.ops.gdn import (
    GDN_CUTE_CHUNK_SIZE,
    GDNBackend,
    chunk_gated_delta_rule,
    gdn_cute_unsupported_reason,
)
from olmo_core.testing import requires_gdn_cute, requires_gpu
from olmo_core.testing.utils import requires_fla

# Lifted from fla's own tests/ops/test_gdn.py, which compares its chunked kernel against its
# eager recurrence. These compare two chunked kernels against each other, so they should be
# comfortably inside an upstream-blessed budget rather than at the edge of it.
TOL = {
    "o": 0.005,
    "dq": 0.008,
    "dk": 0.008,
    "dv": 0.008,
    "dbeta": 0.02,
    "dg": 0.02,
}


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max absolute difference relative to the reference's scale — fla's own error metric."""
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-6)).item()


def _make_inputs(B: int, T: int, HV: int, K: int, V: int, device: str = "cuda"):
    """Deterministic inputs, following the distributions in fla's tests/ops/test_gdn.py."""
    gen = torch.Generator(device=device).manual_seed(0)
    dtype = torch.bfloat16

    def rand(*shape, dt=torch.float32):
        return torch.rand(*shape, generator=gen, device=device, dtype=dt)

    return {
        "q": F.normalize(rand(B, T, HV, K), p=2, dim=-1).to(dtype).requires_grad_(),
        "k": F.normalize(rand(B, T, HV, K), p=2, dim=-1).to(dtype).requires_grad_(),
        "v": rand(B, T, HV, V).to(dtype).requires_grad_(),
        "g": F.logsigmoid(rand(B, T, HV)).requires_grad_(),
        "beta": rand(B, T, HV).sigmoid().requires_grad_(),
        "do": torch.randn(B, T, HV, V, generator=gen, device=device, dtype=dtype),
    }


def _fwd_bwd(inputs: dict, backend: GDNBackend):
    leaves = {n: inputs[n].detach().clone().requires_grad_() for n in ("q", "k", "v", "g", "beta")}
    o, _ = chunk_gated_delta_rule(**leaves, use_qk_l2norm_in_kernel=True, backend=backend)
    o.backward(inputs["do"])
    grads = {f"d{n}": t.grad for n, t in leaves.items()}
    assert all(g is not None for g in grads.values())
    return {"o": o.detach(), **{n: g.detach() for n, g in grads.items()}}  # type: ignore[union-attr]


@requires_gdn_cute
@pytest.mark.parametrize(
    "B, T, HV, V",
    [
        # 256+ CTAs (B * HV * V/64), which is where the CuTe serial scans stop delegating to
        # fla and the whole backward is actually exercised.
        pytest.param(4, 256, 16, 256, id="full-grid"),
        # Below that threshold two backward stages fall back to fla mid-op. That mixed path is
        # what small batches really run, so check it too.
        pytest.param(2, 128, 4, 256, id="small-grid"),
        pytest.param(2, 128, 4, 128, id="head_v_dim=128"),
    ],
)
def test_chunk_gated_delta_rule_cute_matches_fla(B: int, T: int, HV: int, V: int):
    inputs = _make_inputs(B, T, HV, 128, V)
    assert gdn_cute_unsupported_reason(inputs["q"], inputs["k"], inputs["v"]) is None

    cute = _fwd_bwd(inputs, GDNBackend.cute)
    fla = _fwd_bwd(inputs, GDNBackend.fla)

    errs = {name: _rel_err(cute[name], fla[name]) for name in TOL}
    assert all(errs[name] <= TOL[name] for name in TOL), errs


@requires_gdn_cute
def test_chunk_gated_delta_rule_cute_does_not_alias_outputs():
    """
    Two calls with identical input layouts must not share an output buffer.

    The kernels cache their marshaled CuTe views keyed on input shape/stride/dtype. In the repo
    these came from, the output tensors were part of that cache entry, which in a stack of
    same-shaped GDN layers means layer N+1's forward overwrites the activation layer N saved
    for its backward.
    """
    inputs = _make_inputs(2, 128, 4, 128, 256)
    q, k, v, g, beta = (inputs[n] for n in ("q", "k", "v", "g", "beta"))

    first, _ = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, use_qk_l2norm_in_kernel=True, backend=GDNBackend.cute
    )
    snapshot = first.detach().clone()

    # Same layouts, different values, so a shared buffer would show up as a changed `first`.
    chunk_gated_delta_rule(
        q=k, k=q, v=v * 2, g=g, beta=beta, use_qk_l2norm_in_kernel=True, backend=GDNBackend.cute
    )

    torch.testing.assert_close(first.detach(), snapshot, rtol=0, atol=0)


@requires_gdn_cute
def test_chunk_gated_delta_rule_cute_matches_fla_with_many_chunks_per_cta(monkeypatch):
    """
    ``dg`` must survive a CTA owning more than one chunk.

    ``prepare_wy_repr_bwd``'s CTA count is chosen for occupancy — it targets ~1024 CTAs, so at
    any shape a test can afford each CTA gets exactly one chunk, while at a production shape
    (B=16, T=8192) it gets 32. Only in the multi-chunk case can chunk c+1's ``dA_m`` store land
    on the ``sP`` staging that chunk c's ``dg`` half is still reading, and the resulting ``dg``
    is both wrong and nondeterministic. Forcing the CTA target down reproduces that at a shape
    small enough for CI: with the bug present ``dg`` misses its budget by orders of magnitude.
    """
    from olmo_core.kernels.gdn_cute import kernel_wy_bwd

    # nseg is baked into the marshaling cache entry when it is built, so the override only
    # takes effect on a miss. Clear on the way in, and again on the way out so no later test
    # inherits a one-CTA entry.
    monkeypatch.setenv("GDN_WYBWD_CTAS", "1")
    kernel_wy_bwd._CALL_CACHE.clear()
    try:
        inputs = _make_inputs(4, 256, 16, 128, 256)
        cute = _fwd_bwd(inputs, GDNBackend.cute)
        fla = _fwd_bwd(inputs, GDNBackend.fla)
        errs = {name: _rel_err(cute[name], fla[name]) for name in TOL}
        assert all(errs[name] <= TOL[name] for name in TOL), errs
    finally:
        kernel_wy_bwd._CALL_CACHE.clear()


@requires_gdn_cute
def test_chunk_gated_delta_rule_cute_initial_and_final_state():
    B, T, HV, K, V = 2, 128, 4, 128, 256
    inputs = _make_inputs(B, T, HV, K, V)
    h0 = torch.randn(B, HV, K, V, device="cuda", dtype=torch.float32, requires_grad=True)

    out = {}
    for backend in (GDNBackend.cute, GDNBackend.fla):
        leaves = {
            n: inputs[n].detach().clone().requires_grad_() for n in ("q", "k", "v", "g", "beta")
        }
        state = h0.detach().clone().requires_grad_()
        o, ht = chunk_gated_delta_rule(
            **leaves,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            backend=backend,
        )
        assert ht is not None
        (o.float().sum() + ht.sum()).backward()
        assert state.grad is not None
        out[backend] = (o.detach(), ht.detach(), state.grad.detach())

    for i, name in enumerate(("o", "ht", "dh0")):
        assert _rel_err(out[GDNBackend.cute][i], out[GDNBackend.fla][i]) <= 0.008, name


@requires_gdn_cute
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        pytest.param({"chunk_size": 32}, "chunk_size", id="chunk_size"),
        pytest.param({"cu_seqlens": True}, "cu_seqlens", id="cu_seqlens"),
    ],
)
def test_chunk_gated_delta_rule_cute_rejects_unsupported(kwargs: dict, expected: str):
    inputs = _make_inputs(2, 128, 4, 128, 256)
    kwargs = dict(kwargs)  # parametrize args are shared between runs; don't mutate in place
    if kwargs.pop("cu_seqlens", False):
        kwargs["cu_seqlens"] = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)

    with pytest.raises(RuntimeError, match=expected):
        chunk_gated_delta_rule(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            g=inputs["g"],
            beta=inputs["beta"],
            backend=GDNBackend.cute,
            **kwargs,
        )


# requires_fla only tags the test `gpu`; the actual CUDA skip comes from requires_gpu.
@requires_fla
@requires_gpu
def test_chunk_gated_delta_rule_auto_falls_back_and_says_so(caplog):
    """
    Under ``auto``, an unsupported shape must still run — on fla — and log why.

    head_k_dim=64 is outside the CuTe envelope on every GPU, so this covers the fallback on
    hardware that has no CuTe path at all as well as on hardware that does.
    """
    inputs = _make_inputs(2, 128, 4, 64, 128)
    q, k, v, g, beta = (inputs[n] for n in ("q", "k", "v", "g", "beta"))
    assert gdn_cute_unsupported_reason(q, k, v) is not None

    gdn._LOGGED.clear()  # the log fires once per process, so don't let test order decide it
    with caplog.at_level(logging.WARNING, logger="olmo_core.ops.gdn"):
        o, _ = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, use_qk_l2norm_in_kernel=True, backend=GDNBackend.auto
        )

    assert o.shape == v.shape
    o.backward(inputs["do"])
    assert q.grad is not None

    assert any("Falling back to the fla" in r.message for r in caplog.records), caplog.text


@requires_gdn_cute
@pytest.mark.parametrize(
    "shape_kwargs, expected",
    [
        pytest.param({"K": 64}, "head_k_dim", id="head_k_dim"),
        pytest.param({"V": 32}, "head_v_dim", id="head_v_dim"),
        pytest.param({"T": GDN_CUTE_CHUNK_SIZE + 1}, "seq_len", id="seq_len"),
    ],
)
def test_gdn_cute_unsupported_reason(shape_kwargs: dict, expected: str):
    dims = {"B": 2, "T": 128, "HV": 4, "K": 128, "V": 256}
    dims.update(shape_kwargs)
    B, T, HV, K, V = (dims[n] for n in ("B", "T", "HV", "K", "V"))
    opts = {"device": "cuda", "dtype": torch.bfloat16}
    reason = gdn_cute_unsupported_reason(
        torch.empty(B, T, HV, K, **opts),  # type: ignore[arg-type]
        torch.empty(B, T, HV, K, **opts),  # type: ignore[arg-type]
        torch.empty(B, T, HV, V, **opts),  # type: ignore[arg-type]
    )
    assert reason is not None and expected in reason
