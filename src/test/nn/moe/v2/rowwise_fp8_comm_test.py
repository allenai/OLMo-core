import torch

from olmo_core.kernels import symm_mem_vdev2d
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
        del expert_q, expert_scales, src_ranks, src_rows, group_name, probs, block_size, nblocks, post_barrier
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
    )
    out.sum().backward()

    assert probs.grad is not None
    torch.testing.assert_close(probs.grad, torch.full_like(probs, 3.0))


def test_rowwise_dispatch_put_weighted_mxfp8_uses_weighted_quantize(monkeypatch):
    q = torch.empty(6, 64, dtype=torch.float8_e4m3fn)
    scales = torch.empty(6, 2, dtype=torch.float8_e8m0fnu)
    seen = {"quantize": 0, "puts": []}

    def _stub_weighted_quantize_rows_to_mxfp8(input_hp, probs, *, block_size, out=None, scales_out=None):
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


def test_rowwise_dispatch_put_scaled_pair_uses_paired_put(monkeypatch):
    q = torch.empty(2, 64, dtype=torch.float8_e4m3fn)
    scales = torch.empty(2, 2, dtype=torch.float8_e8m0fnu)
    seen = {}

    def _stub_quantize_rows_to_mxfp8(input_hp, *, block_size, out=None, scales_out=None):
        seen["quantize"] = {
            "input": input_hp,
            "block_size": block_size,
            "out": out,
            "scales_out": scales_out,
        }
        return q, scales

    def _stub_rowwise_dispatch_put_pair(
        input_q,
        input_scales,
        out_q,
        out_scales,
        dst_ranks,
        dst_rows,
        group_name,
        *,
        nblocks,
        pre_barrier=False,
        post_barrier=True,
    ):
        seen["pair"] = {
            "input_q": input_q,
            "input_scales": input_scales,
            "out_q": out_q,
            "out_scales": out_scales,
            "dst_ranks": dst_ranks,
            "dst_rows": dst_rows,
            "group_name": group_name,
            "nblocks": nblocks,
            "pre_barrier": pre_barrier,
            "post_barrier": post_barrier,
        }

    monkeypatch.setattr(symm_mem_vdev2d, "quantize_rows_to_mxfp8", _stub_quantize_rows_to_mxfp8)
    monkeypatch.setattr(symm_mem_vdev2d, "rowwise_dispatch_put_pair", _stub_rowwise_dispatch_put_pair)

    input_hp = torch.randn(2, 64, dtype=torch.bfloat16)
    out_q = torch.empty(4, 64, dtype=torch.float8_e4m3fn)
    out_scales = torch.empty(4, 2, dtype=torch.float8_e8m0fnu)
    dst_ranks = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    dst_rows = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    input_q = torch.empty_like(q)
    input_scales = torch.empty_like(scales)

    symm_mem_vdev2d.rowwise_dispatch_put_scaled_pair(
        input_hp,
        out_q,
        out_scales,
        dst_ranks,
        dst_rows,
        "test_group",
        block_size=32,
        nblocks=128,
        pre_barrier=True,
        post_barrier=False,
        input_q=input_q,
        input_scales=input_scales,
    )

    assert seen["quantize"]["input"] is input_hp
    assert seen["quantize"]["block_size"] == 32
    assert seen["quantize"]["out"] is input_q
    assert seen["quantize"]["scales_out"] is input_scales
    assert seen["pair"]["input_q"] is q
    assert seen["pair"]["input_scales"] is scales
    assert seen["pair"]["out_q"] is out_q
    assert seen["pair"]["out_scales"] is out_scales
    assert seen["pair"]["dst_ranks"] is dst_ranks
    assert seen["pair"]["dst_rows"] is dst_rows
    assert seen["pair"]["group_name"] == "test_group"
    assert seen["pair"]["nblocks"] == 128
    assert seen["pair"]["pre_barrier"] is True
    assert seen["pair"]["post_barrier"] is False


def test_rowwise_dispatch_put_scaled_packed_uses_one_packed_put(monkeypatch):
    q = torch.empty(2, 64, dtype=torch.float8_e4m3fn)
    scales = torch.empty(2, 2, dtype=torch.float8_e8m0fnu)
    packed_input = torch.empty(2, 128, dtype=torch.uint8)
    packed_out = torch.empty(4, 128, dtype=torch.uint8)
    seen = {}

    def _stub_quantize_rows_to_mxfp8(input_hp, *, block_size, out=None, scales_out=None):
        seen["quantize"] = {
            "input": input_hp,
            "block_size": block_size,
            "out": out,
            "scales_out": scales_out,
        }
        return q, scales

    def _stub_pack_rowwise_mxfp8_rows(input_q, input_scales, packed, *, alignment):
        seen["pack"] = {
            "input_q": input_q,
            "input_scales": input_scales,
            "packed": packed,
            "alignment": alignment,
        }

    def _stub_rowwise_dispatch_put(
        input,
        out,
        dst_ranks,
        dst_rows,
        group_name,
        *,
        probs=None,
        nblocks,
        pre_barrier=False,
        post_barrier=True,
    ):
        seen["put"] = {
            "input": input,
            "out": out,
            "dst_ranks": dst_ranks,
            "dst_rows": dst_rows,
            "group_name": group_name,
            "probs": probs,
            "nblocks": nblocks,
            "pre_barrier": pre_barrier,
            "post_barrier": post_barrier,
        }

    def _stub_unpack_rowwise_mxfp8_rows(packed, out_q, out_scales, *, alignment):
        seen["unpack"] = {
            "packed": packed,
            "out_q": out_q,
            "out_scales": out_scales,
            "alignment": alignment,
        }

    monkeypatch.setattr(symm_mem_vdev2d, "quantize_rows_to_mxfp8", _stub_quantize_rows_to_mxfp8)
    monkeypatch.setattr(symm_mem_vdev2d, "pack_rowwise_mxfp8_rows", _stub_pack_rowwise_mxfp8_rows)
    monkeypatch.setattr(symm_mem_vdev2d, "rowwise_dispatch_put", _stub_rowwise_dispatch_put)
    monkeypatch.setattr(symm_mem_vdev2d, "unpack_rowwise_mxfp8_rows", _stub_unpack_rowwise_mxfp8_rows)

    input_hp = torch.randn(2, 64, dtype=torch.bfloat16)
    out_q = torch.empty(4, 64, dtype=torch.float8_e4m3fn)
    out_scales = torch.empty(4, 2, dtype=torch.float8_e8m0fnu)
    dst_ranks = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    dst_rows = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    input_q = torch.empty_like(q)
    input_scales = torch.empty_like(scales)

    symm_mem_vdev2d.rowwise_dispatch_put_scaled_packed(
        input_hp,
        out_q,
        out_scales,
        dst_ranks,
        dst_rows,
        "test_group",
        block_size=32,
        nblocks=128,
        pre_barrier=True,
        post_barrier=False,
        input_q=input_q,
        input_scales=input_scales,
        packed_input=packed_input,
        packed_out=packed_out,
        pack_alignment=128,
    )

    assert seen["quantize"]["input"] is input_hp
    assert seen["quantize"]["out"] is input_q
    assert seen["quantize"]["scales_out"] is input_scales
    assert seen["pack"]["input_q"] is q
    assert seen["pack"]["input_scales"] is scales
    assert seen["pack"]["packed"] is packed_input
    assert seen["pack"]["alignment"] == 128
    assert seen["put"]["input"] is packed_input
    assert seen["put"]["out"] is packed_out
    assert seen["put"]["dst_ranks"] is dst_ranks
    assert seen["put"]["dst_rows"] is dst_rows
    assert seen["put"]["group_name"] == "test_group"
    assert seen["put"]["probs"] is None
    assert seen["put"]["nblocks"] == 128
    assert seen["put"]["pre_barrier"] is True
    assert seen["put"]["post_barrier"] is False
    assert seen["unpack"]["packed"] is packed_out
    assert seen["unpack"]["out_q"] is out_q
    assert seen["unpack"]["out_scales"] is out_scales
    assert seen["unpack"]["alignment"] == 128


def test_pack_unpack_rowwise_mxfp8_rows_preserves_bytes_and_padding():
    q = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [11, 12, 13, 14, 15],
        ],
        dtype=torch.uint8,
    )
    scales = torch.tensor(
        [
            [101, 102, 103],
            [111, 112, 113],
        ],
        dtype=torch.uint8,
    )
    packed = torch.full((2, 12), 255, dtype=torch.uint8)

    symm_mem_vdev2d.pack_rowwise_mxfp8_rows(q, scales, packed, alignment=4)

    expected = torch.tensor(
        [
            [1, 2, 3, 4, 5, 0, 0, 0, 101, 102, 103, 0],
            [11, 12, 13, 14, 15, 0, 0, 0, 111, 112, 113, 0],
        ],
        dtype=torch.uint8,
    )
    torch.testing.assert_close(packed, expected, atol=0, rtol=0)

    unpacked_q = torch.empty_like(q)
    unpacked_scales = torch.empty_like(scales)
    symm_mem_vdev2d.unpack_rowwise_mxfp8_rows(
        packed,
        unpacked_q,
        unpacked_scales,
        alignment=4,
    )

    torch.testing.assert_close(unpacked_q, q, atol=0, rtol=0)
    torch.testing.assert_close(unpacked_scales, scales, atol=0, rtol=0)


def test_rowwise_combine_get_scaled_pair_uses_paired_gather_and_reduce(monkeypatch):
    seen = {}

    def _stub_rowwise_gather_get_pair(
        expert_q,
        expert_scales,
        gathered_q,
        gathered_scales,
        src_ranks,
        src_rows,
        group_name,
        *,
        nblocks,
        pre_barrier=True,
        post_barrier=False,
    ):
        seen["gather"] = {
            "expert_q": expert_q,
            "expert_scales": expert_scales,
            "gathered_q": gathered_q,
            "gathered_scales": gathered_scales,
            "src_ranks": src_ranks.clone(),
            "src_rows": src_rows.clone(),
            "group_name": group_name,
            "nblocks": nblocks,
            "pre_barrier": pre_barrier,
            "post_barrier": post_barrier,
        }
        gathered_q.zero_()
        gathered_scales.fill_(1)

    def _stub_reduce_gathered_rows_from_mxfp8(
        gathered_q,
        gathered_scales,
        out,
        *,
        probs=None,
        valid_mask,
        block_size,
        gathered_out=None,
    ):
        seen["reduce"] = {
            "gathered_q_shape": tuple(gathered_q.shape),
            "gathered_scales_shape": tuple(gathered_scales.shape),
            "out": out,
            "probs": probs,
            "valid_mask": valid_mask.clone(),
            "block_size": block_size,
            "gathered_out": gathered_out,
        }
        out.fill_(2.0)

    monkeypatch.setattr(symm_mem_vdev2d, "rowwise_gather_get_pair", _stub_rowwise_gather_get_pair)
    monkeypatch.setattr(
        symm_mem_vdev2d,
        "reduce_gathered_rows_from_mxfp8",
        _stub_reduce_gathered_rows_from_mxfp8,
    )

    expert_q = torch.empty(4, 64, dtype=torch.float8_e4m3fn)
    expert_scales = torch.empty(4, 2, dtype=torch.float8_e8m0fnu)
    out = torch.empty(2, 64, dtype=torch.bfloat16)
    src_ranks = torch.tensor([[0, -1], [1, 0]], dtype=torch.long)
    src_rows = torch.tensor([[0, -1], [3, 2]], dtype=torch.long)
    probs = torch.ones(2, 2, dtype=torch.float32)
    gathered_q = torch.empty(2, 2, 64, dtype=torch.float8_e4m3fn)
    gathered_scales = torch.empty(2, 2, 2, dtype=torch.float8_e8m0fnu)

    symm_mem_vdev2d.rowwise_combine_get_scaled_pair(
        expert_q,
        expert_scales,
        out,
        src_ranks,
        src_rows,
        "test_group",
        probs=probs,
        block_size=32,
        nblocks=64,
        gathered_q_out=gathered_q,
        gathered_scales_out=gathered_scales,
        pre_barrier=False,
        post_barrier=True,
    )

    assert seen["gather"]["expert_q"] is expert_q
    assert seen["gather"]["expert_scales"] is expert_scales
    assert seen["gather"]["gathered_q"].shape == (4, 64)
    assert seen["gather"]["gathered_scales"].shape == (4, 2)
    assert seen["gather"]["group_name"] == "test_group"
    assert seen["gather"]["nblocks"] == 64
    assert seen["gather"]["pre_barrier"] is False
    assert seen["gather"]["post_barrier"] is True
    expected_flat_ranks = torch.tensor([[0], [-1], [1], [0]], dtype=torch.long)
    expected_flat_rows = torch.tensor([[0], [-1], [3], [2]], dtype=torch.long)
    assert torch.equal(seen["gather"]["src_ranks"], expected_flat_ranks)
    assert torch.equal(seen["gather"]["src_rows"], expected_flat_rows)

    assert seen["reduce"]["gathered_q_shape"] == (2, 2, 64)
    assert seen["reduce"]["gathered_scales_shape"] == (2, 2, 2)
    assert seen["reduce"]["out"] is out
    assert seen["reduce"]["probs"] is probs
    assert seen["reduce"]["block_size"] == 32
    torch.testing.assert_close(out, torch.full_like(out, 2.0))
