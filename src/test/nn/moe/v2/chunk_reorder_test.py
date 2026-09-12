import torch

from olmo_core.kernels.moe_chunk_reorder import (
    _chunk_unpermute_torch,
    _dispatch_permute_by_row_id_map,
)


def test_chunk_unpermute_torch_zeros_out_of_range_rows():
    inp = torch.arange(3 * 512, dtype=torch.float32).view(3, 512)
    row_id_map = torch.tensor([0, 3, -1, 2], dtype=torch.int32)

    out = _chunk_unpermute_torch(inp, row_id_map, num_tokens=4, out=None)

    expected = torch.zeros(4, 512)
    expected[0].copy_(inp[0])
    expected[3].copy_(inp[2])
    torch.testing.assert_close(out, expected)


def test_chunk_permute_by_row_id_map_torch_honors_out_buffer():
    inp = torch.arange(3 * 512, dtype=torch.float32).view(3, 512)
    row_id_map = torch.tensor([2, -1, 0], dtype=torch.int32)
    out_buffer = torch.empty((3, 512), dtype=inp.dtype)

    out = _dispatch_permute_by_row_id_map(
        inp,
        row_id_map,
        num_out_tokens=3,
        backend="triton",
        out=out_buffer,
    )

    assert out.data_ptr() == out_buffer.data_ptr()
    expected = torch.zeros(3, 512)
    expected[0].copy_(inp[2])
    expected[2].copy_(inp[0])
    torch.testing.assert_close(out, expected)
