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
        None,
        1,
        1,
        1,
        False,
        False,
        False,
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
        post_barrier=False,
    ):
        del (
            expert_q,
            expert_scales,
            src_ranks,
            src_rows,
            group_name,
            probs,
            block_size,
            nblocks,
            post_barrier,
        )
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
        None,
        None,
        32,
        "test_group",
        None,
        1,
        1,
    )
    out.sum().backward()

    assert probs.grad is not None
    torch.testing.assert_close(probs.grad, torch.full_like(probs, 3.0))


def test_rowwise_dispatch_put_weighted_mxfp8_uses_weighted_quantize(monkeypatch):
    q = torch.empty(6, 64, dtype=torch.float8_e4m3fn)
    scales = torch.empty(6, 2, dtype=torch.float8_e8m0fnu)
    seen: dict = {"quantize": 0, "puts": []}

    def _stub_weighted_quantize_rows_to_mxfp8(
        input_hp, probs, *, block_size, out=None, scales_out=None
    ):
        seen["quantize"] += 1
        assert input_hp.shape == (2, 64)
        assert probs.shape == (2, 3)
        assert block_size == 32
        assert out is None
        assert scales_out is None
        return q, scales

    def _stub_rowwise_dispatch_put(
        input_tensor,
        out,
        dst_ranks,
        dst_rows,
        group_name,
        *,
        nblocks,
        pre_barrier=False,
        post_barrier=True,
    ):
        seen["puts"].append(
            {
                "input": input_tensor,
                "out": out,
                "dst_ranks": dst_ranks.clone(),
                "dst_rows": dst_rows.clone(),
                "group_name": group_name,
                "nblocks": nblocks,
                "pre_barrier": pre_barrier,
                "post_barrier": post_barrier,
            }
        )

    monkeypatch.setattr(
        comm,
        "weighted_quantize_rows_to_mxfp8",
        _stub_weighted_quantize_rows_to_mxfp8,
    )
    monkeypatch.setattr(
        comm.symm_mem_vdev2d_kernels,
        "rowwise_dispatch_put",
        _stub_rowwise_dispatch_put,
    )

    input_hp = torch.randn(2, 64, dtype=torch.bfloat16)
    probs = torch.rand(2, 3)
    out_q = torch.empty(4, 64, dtype=torch.float8_e4m3fn)
    out_scales = torch.empty(4, 2, dtype=torch.float8_e8m0fnu)
    dst_ranks = torch.tensor([[0, 1, -1], [1, 0, 0]], dtype=torch.long)
    dst_rows = torch.tensor([[0, 2, -1], [1, 3, 0]], dtype=torch.long)

    comm._rowwise_dispatch_put_weighted_mxfp8(
        input_hp,
        probs,
        out_q,
        out_scales,
        dst_ranks,
        dst_rows,
        "test_group",
        block_size=32,
        nblocks=128,
    )

    assert seen["quantize"] == 1
    assert len(seen["puts"]) == 2
    assert seen["puts"][0]["input"] is q
    assert seen["puts"][0]["out"] is out_q
    assert seen["puts"][0]["group_name"] == "test_group"
    assert seen["puts"][0]["nblocks"] == 128
    assert seen["puts"][0]["pre_barrier"] is False
    assert seen["puts"][0]["post_barrier"] is False
    assert seen["puts"][1]["input"] is scales
    assert seen["puts"][1]["out"] is out_scales
    assert seen["puts"][1]["pre_barrier"] is False
    assert seen["puts"][1]["post_barrier"] is True
    expected_ranks = dst_ranks.reshape(-1, 1)
    expected_rows = dst_rows.reshape(-1, 1)
    assert torch.equal(seen["puts"][0]["dst_ranks"], expected_ranks)
    assert torch.equal(seen["puts"][0]["dst_rows"], expected_rows)
    assert torch.equal(seen["puts"][1]["dst_ranks"], expected_ranks)
    assert torch.equal(seen["puts"][1]["dst_rows"], expected_rows)
