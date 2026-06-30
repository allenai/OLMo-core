"""
Unit tests for the rowwise expert-parallel comm autograd functions.

The rowwise all-to-all has four characteristic failure modes; these cover the ones that are
exercisable in a single process (the symmetric-memory kernels are monkeypatched or the guards run
before them):

- **buffer lifetime / lease** — symmetric buffers must be released exactly once after use
  (``test_rowwise_bf16_combine_releases_lifetime_leases_and_grads_probs``).
- **routing-map correctness** — malformed routing maps (``src_ranks`` / ``src_rows`` / ``probs``
  shapes) must be rejected, not silently mis-scattered
  (``test_rowwise_combine_rejects_*``).
- **on-the-wire FP8 / SwiGLU math** — the hand-written rowwise SwiGLU forward/backward must match
  a reference / autograd (``test_rowwise_fp8_swiglu_*``).

The fourth mode — cross-rank barrier / race correctness — is inherently multi-process and is
covered end-to-end by the no-EP-vs-EP parity test that lands with the expert-parallel forward.
"""

import pytest
import torch

from olmo_core.nn.moe.v2 import comm


class _FakeLease:
    def __init__(self):
        self.released = 0

    def release(self):
        self.released += 1


def test_rowwise_bf16_combine_releases_lifetime_leases_and_grads_probs(monkeypatch):
    def _stub_rowwise_combine_get(
        expert_out,
        combine_out,
        src_ranks,
        src_rows,
        group_name,
        *,
        probs=None,
        nblocks,
        gathered_out=None,
        pre_barrier,
        post_barrier,
    ):
        del expert_out, src_ranks, src_rows, group_name, probs, nblocks, pre_barrier, post_barrier
        combine_out.fill_(1.0)
        if gathered_out is not None:
            gathered_out.fill_(1.0)

    def _stub_rowwise_dispatch_put(
        dispatch_source,
        symm_out,
        dst_ranks,
        dst_rows,
        group_name,
        *,
        probs=None,
        nblocks,
    ):
        del dispatch_source, dst_ranks, dst_rows, group_name, probs, nblocks
        symm_out.zero_()

    monkeypatch.setattr(
        comm.symm_mem_vdev2d_kernels,
        "rowwise_combine_get",
        _stub_rowwise_combine_get,
    )
    monkeypatch.setattr(
        comm.symm_mem_vdev2d_kernels,
        "rowwise_dispatch_put",
        _stub_rowwise_dispatch_put,
    )

    expert_out = torch.zeros(4, 8, dtype=torch.float32, requires_grad=True)
    symm_expert_out = torch.zeros_like(expert_out)
    symm_combine_out = torch.empty(2, 8, dtype=torch.float32)
    symm_gathered_routes = torch.empty(2, 2, 8, dtype=torch.float32)
    src_ranks = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
    src_rows = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    probs = torch.ones(2, 2, dtype=torch.float32, requires_grad=True)
    combine_out_lease = _FakeLease()
    gather_lease = _FakeLease()

    out = comm._RowwiseCombineWeightedAutograd.apply(
        expert_out,
        symm_expert_out,
        symm_combine_out,
        combine_out_lease,
        symm_gathered_routes,
        gather_lease,
        src_ranks,
        src_rows,
        probs,
        "test_group",
        None,  # group
        1,  # nblocks
        False,  # expert_out_aliases_symm_expert_out
        False,  # pre_barrier
        False,  # post_barrier
    )
    out.sum().backward()

    assert combine_out_lease.released == 1
    assert gather_lease.released == 1
    assert probs.grad is not None
    torch.testing.assert_close(probs.grad, torch.full_like(probs, 8.0))


def test_rowwise_fp8_combine_backward_returns_grad_probs(monkeypatch):
    def _stub_quantize_rows_to_mxfp8(x, *, block_size, out, scales_out):
        del x, block_size
        out.zero_()
        scales_out.fill_(1)

    def _stub_rowwise_combine_get_scaled(
        expert_q,
        expert_scales,
        combine_out,
        src_ranks,
        src_rows,
        group_name,
        *,
        probs=None,
        block_size,
        nblocks,
        gathered_q_out=None,
        gathered_scales_out=None,
    ):
        del expert_q, expert_scales, src_ranks, src_rows, group_name, probs, block_size, nblocks
        combine_out.zero_()
        if gathered_q_out is not None:
            gathered_q_out.zero_()
        if gathered_scales_out is not None:
            gathered_scales_out.fill_(1)

    def _stub_dot_gathered_rows_mxfp8_with_grad(
        gathered_q,
        gathered_scales,
        grad_out,
        *,
        valid_mask,
        block_size,
        out_dtype,
    ):
        del gathered_q, gathered_scales, grad_out, block_size
        return valid_mask.to(dtype=out_dtype) * 3.0

    monkeypatch.setattr(comm, "quantize_rows_to_mxfp8", _stub_quantize_rows_to_mxfp8)
    monkeypatch.setattr(
        comm.symm_mem_vdev2d_kernels,
        "rowwise_combine_get_scaled",
        _stub_rowwise_combine_get_scaled,
    )
    monkeypatch.setattr(
        comm,
        "dot_gathered_rows_mxfp8_with_grad",
        _stub_dot_gathered_rows_mxfp8_with_grad,
    )

    expert_out = torch.zeros(4, 64, dtype=torch.bfloat16)
    src_ranks = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
    src_rows = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    probs = torch.ones(2, 2, dtype=torch.float32, requires_grad=True)
    q = torch.empty_like(expert_out, dtype=torch.float8_e4m3fn)
    scales = torch.empty((expert_out.shape[0], 2), dtype=torch.float8_e8m0fnu)

    out = comm._RowwiseCombineWeightedFP8Autograd.apply(
        expert_out,
        src_ranks,
        src_rows,
        probs,
        q,
        scales,
        32,
        "test_group",
        None,
        1,
    )
    out.sum().backward()

    assert probs.grad is not None
    torch.testing.assert_close(probs.grad, torch.full_like(probs, 3.0))


# --- routing-map correctness: malformed maps must be rejected before the kernel ---


def test_rowwise_combine_rejects_mismatched_probs():
    expert_out = torch.zeros(4, 8)
    src = torch.zeros(2, 2, dtype=torch.long)
    probs = torch.ones(2, 3)  # shape disagrees with src_ranks/src_rows
    with pytest.raises(RuntimeError, match="probs shape mismatch"):
        comm._RowwiseCombineWeightedAutograd.apply(
            expert_out,
            expert_out.clone(),
            None,  # symm_combine_out
            None,  # symm_combine_out_lease
            None,  # symm_gathered_routes
            None,  # symm_gathered_routes_lease
            src,
            src.clone(),
            probs,
            "g",  # group_name
            None,  # group
            1,  # nblocks
            False,  # expert_out_aliases_symm_expert_out
            False,  # pre_barrier
            False,  # post_barrier
        )


def test_rowwise_combine_rejects_negative_nblocks():
    expert_out = torch.zeros(4, 8)
    src = torch.zeros(2, 2, dtype=torch.long)
    probs = torch.ones(2, 2)
    with pytest.raises(RuntimeError, match="nblocks must be >= 0"):
        comm._RowwiseCombineWeightedAutograd.apply(
            expert_out,
            expert_out.clone(),
            None,
            None,
            None,
            None,
            src,
            src.clone(),
            probs,
            "g",
            None,
            -1,  # nblocks
            False,
            False,
            False,
        )


# --- on-the-wire FP8 / SwiGLU math ---


def test_rowwise_fp8_swiglu_forward_matches_reference():
    torch.manual_seed(0)
    up_gate = torch.randn(6, 16)
    hidden = up_gate.shape[-1] // 2
    up, gate = up_gate[:, :hidden], up_gate[:, hidden:]
    expected = up * (gate * torch.sigmoid(gate))
    torch.testing.assert_close(comm._rowwise_fp8_swiglu_forward_impl(up_gate), expected)


def test_rowwise_fp8_swiglu_backward_matches_autograd():
    torch.manual_seed(0)
    up_gate = torch.randn(6, 16, dtype=torch.float64, requires_grad=True)
    grad_h = torch.randn(6, 8, dtype=torch.float64)
    comm._rowwise_fp8_swiglu_forward_impl(up_gate).backward(grad_h)
    manual = comm._rowwise_fp8_swiglu_backward_impl(up_gate.detach(), grad_h)
    torch.testing.assert_close(up_gate.grad, manual)
