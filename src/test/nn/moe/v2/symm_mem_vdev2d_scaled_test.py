import torch

import olmo_core.kernels.symm_mem_vdev2d as symm_mod


def _patch_reduce_identity(monkeypatch):
    def _reduce_identity(
        gathered_q,
        gathered_scales,
        out,
        *,
        probs=None,
        valid_mask=None,
        block_size=32,
        gathered_out=None,
    ):
        del gathered_scales, block_size
        gathered = gathered_q.to(torch.float32)
        if valid_mask is not None:
            gathered = gathered * valid_mask.to(torch.float32).unsqueeze(-1)
        if gathered_out is not None:
            gathered_out.copy_(gathered.to(gathered_out.dtype))
        if probs is not None:
            gathered = gathered * probs.to(torch.float32).unsqueeze(-1)
        out.copy_(gathered.sum(dim=1).to(out.dtype))

    monkeypatch.setattr(symm_mod, "reduce_gathered_rows_from_mxfp8", _reduce_identity)


def test_rowwise_combine_get_scaled_masks_invalid_routes(monkeypatch):
    _patch_reduce_identity(monkeypatch)

    def _fake_gather(expert_out, out, src_ranks, src_rows, group_name, *, nblocks=0):
        del expert_out, group_name, nblocks
        if out.shape[1] == 16:
            out.fill_(1.0)
            invalid = (src_ranks.reshape(-1) < 0) | (src_rows.reshape(-1) < 0)
            out[invalid] = 0.0
        else:
            for i in range(out.shape[0]):
                if src_ranks[i, 0] < 0 or src_rows[i, 0] < 0:
                    out[i].zero_()
                else:
                    out[i].fill_(float(i + 1))

    monkeypatch.setattr(symm_mod, "rowwise_gather_get", _fake_gather)

    # N=2, K=2. Route (0,1) is dropped.
    src_ranks = torch.tensor([[0, -1], [1, 0]], dtype=torch.long)
    src_rows = torch.tensor([[0, -1], [2, 3]], dtype=torch.long)

    out = torch.empty((2, 512), dtype=torch.float32)
    expert_out_q = torch.empty((8, 512), dtype=torch.float32)
    expert_out_scales = torch.empty((8, 16), dtype=torch.float32)

    symm_mod.rowwise_combine_get_scaled(
        expert_out_q,
        expert_out_scales,
        out,
        src_ranks,
        src_rows,
        "dummy_group",
        block_size=32,
    )

    # Flat gathered rows are [1,2,3,4], but row 2 is invalid and must be zeroed.
    expected = torch.stack(
        [
            torch.full((512,), 1.0, dtype=torch.float32),  # 1 + 0
            torch.full((512,), 7.0, dtype=torch.float32),  # 3 + 4
        ],
    )
    torch.testing.assert_close(out, expected)


def test_rowwise_combine_get_scaled_weighted(monkeypatch):
    _patch_reduce_identity(monkeypatch)

    def _fake_gather(expert_out, out, src_ranks, src_rows, group_name, *, nblocks=0):
        del expert_out, src_ranks, src_rows, group_name, nblocks
        if out.shape[1] == 16:
            out.fill_(1.0)
        else:
            vals = torch.stack(
                [
                    torch.full((512,), 2.0, dtype=out.dtype),
                    torch.full((512,), 6.0, dtype=out.dtype),
                ],
            )
            out.copy_(vals)

    monkeypatch.setattr(symm_mod, "rowwise_gather_get", _fake_gather)

    src_ranks = torch.tensor([[0, 1]], dtype=torch.long)
    src_rows = torch.tensor([[0, 1]], dtype=torch.long)
    probs = torch.tensor([[0.25, 0.75]], dtype=torch.float32)

    out = torch.empty((1, 512), dtype=torch.float32)
    gathered_out = torch.empty((1, 2, 512), dtype=torch.float32)
    expert_out_q = torch.empty((4, 512), dtype=torch.float32)
    expert_out_scales = torch.ones((4, 16), dtype=torch.float32)

    symm_mod.rowwise_combine_get_scaled(
        expert_out_q,
        expert_out_scales,
        out,
        src_ranks,
        src_rows,
        "dummy_group",
        probs=probs,
        block_size=32,
        gathered_out=gathered_out,
    )

    expected_gather = torch.stack(
        [
            torch.stack(
                [
                    torch.full((512,), 2.0, dtype=torch.float32),
                    torch.full((512,), 6.0, dtype=torch.float32),
                ],
            )
        ],
    )
    expected = torch.full((1, 512), 5.0, dtype=torch.float32)
    torch.testing.assert_close(gathered_out, expected_gather)
    torch.testing.assert_close(out, expected)


def test_rowwise_combine_get_scaled_reuses_gather_output_buffers(monkeypatch):
    _patch_reduce_identity(monkeypatch)

    observed_ptrs = {}

    def _fake_gather(expert_out, out, src_ranks, src_rows, group_name, *, nblocks=0):
        del expert_out, src_ranks, src_rows, group_name, nblocks
        if out.shape[1] == 512:
            observed_ptrs["q"] = out.data_ptr()
            out.copy_(
                torch.stack(
                    [
                        torch.full((512,), 1.0, dtype=out.dtype),
                        torch.full((512,), 3.0, dtype=out.dtype),
                    ],
                )
            )
        else:
            observed_ptrs["s"] = out.data_ptr()
            out.fill_(1.0)

    monkeypatch.setattr(symm_mod, "rowwise_gather_get", _fake_gather)

    src_ranks = torch.tensor([[0, 1]], dtype=torch.long)
    src_rows = torch.tensor([[0, 1]], dtype=torch.long)
    out = torch.empty((1, 512), dtype=torch.float32)
    expert_out_q = torch.empty((4, 512), dtype=torch.float32)
    expert_out_scales = torch.ones((4, 16), dtype=torch.float32)
    gathered_q_out = torch.empty((1, 2, 512), dtype=torch.float32)
    gathered_scales_out = torch.empty((1, 2, 16), dtype=torch.float32)

    symm_mod.rowwise_combine_get_scaled(
        expert_out_q,
        expert_out_scales,
        out,
        src_ranks,
        src_rows,
        "dummy_group",
        block_size=32,
        gathered_q_out=gathered_q_out,
        gathered_scales_out=gathered_scales_out,
    )

    assert observed_ptrs["q"] == gathered_q_out.view(-1, 512).data_ptr()
    assert observed_ptrs["s"] == gathered_scales_out.view(-1, 16).data_ptr()
    torch.testing.assert_close(
        gathered_q_out,
        torch.stack(
            [
                torch.stack(
                    [
                        torch.full((512,), 1.0, dtype=torch.float32),
                        torch.full((512,), 3.0, dtype=torch.float32),
                    ],
                )
            ],
        ),
    )
    torch.testing.assert_close(gathered_scales_out, torch.ones_like(gathered_scales_out))
    torch.testing.assert_close(out, torch.full((1, 512), 4.0, dtype=torch.float32))
