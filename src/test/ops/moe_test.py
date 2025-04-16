import numpy as np
import pytest
import torch

from olmo_core.ops import moe as ops

from ..utils import DEVICES, requires_gpu


@requires_gpu
@pytest.mark.parametrize(
    ("sl", "hs", "ne", "top_k"),
    [
        (4, 2, 2, 1),
        (4, 2, 2, 2),
        (4, 2, 2, 4),
        (1024, 1536, 4, 1),
        (1024, 1536, 4, 2),
        (1024, 1536, 4, 4),
        (1024, 1536, 64, 1),
        (1024, 1536, 64, 2),
        (1024, 1536, 64, 4),
        (1024, 1536, 128, 1),
        (1024, 1536, 128, 2),
        (1024, 1536, 128, 4),
        (16384, 768, 4, 1),
        (16384, 768, 4, 2),
        (16384, 768, 4, 4),
        (16384, 768, 64, 1),
        (16384, 768, 64, 2),
        (16384, 768, 64, 4),
        (16384, 768, 128, 1),
        (16384, 768, 128, 2),
        (16384, 768, 128, 4),
    ],
)
def test_binned_gather(sl: int, hs: int, ne: int, top_k: int):
    # NOTE: Capacity factor == 1.
    ec = (sl * top_k) // ne

    # Create the data and indices.
    x = torch.randn((sl, hs)).cuda().half()

    # Randomly assign tokens to experts.
    top_expert = torch.randint(0, ne, (sl * top_k,)).cuda().int()
    _, indices = torch.sort(top_expert)
    indices = indices.int()
    bins = torch.cumsum(torch.histc(top_expert, ne, min=0, max=ne - 1), 0).int()

    def binned_gather(
        x: torch.Tensor,
        indices: torch.Tensor,
        bins: torch.Tensor,
        ec: int,
        top_k: int,
    ):
        x_np = x.cpu().numpy()
        indices_np = indices.cpu().numpy()
        bins_np = bins.cpu().numpy()
        start = 0
        out = np.zeros((ne, ec, hs))
        for i in range(ne):
            end = bins_np[i]
            for j in range(min(ec, end - start)):
                index = indices_np[start + j] // top_k
                out[i, j, :] = x_np[index, :]
            start = end
        return torch.from_numpy(out).cuda().half()

    out = ops.binned_gather(x, indices, bins, ec, top_k)
    expected_out = binned_gather(x, indices, bins, ec, top_k)
    assert torch.all(torch.eq(out, expected_out))


@requires_gpu
@pytest.mark.parametrize(
    ("sl", "hs", "ne", "top_k"),
    [
        (4, 2, 2, 1),
        (4, 2, 2, 2),
        (4, 2, 2, 4),
        (1024, 1536, 4, 1),
        (1024, 1536, 4, 2),
        (1024, 1536, 4, 4),
        (1024, 1536, 64, 1),
        (1024, 1536, 64, 2),
        (1024, 1536, 64, 4),
        (1024, 1536, 128, 1),
        (1024, 1536, 128, 2),
        (1024, 1536, 128, 4),
        (16384, 768, 4, 1),
        (16384, 768, 4, 2),
        (16384, 768, 4, 4),
        (16384, 768, 64, 1),
        (16384, 768, 64, 2),
        (16384, 768, 64, 4),
        (16384, 768, 128, 1),
        (16384, 768, 128, 2),
        (16384, 768, 128, 4),
    ],
)
def test_binned_scatter(sl: int, hs: int, ne: int, top_k: int):
    # NOTE: Capacity factor == 1.
    ec = (sl * top_k) // ne

    # Create the data and indices.
    x = torch.randn((sl, hs)).cuda().half()

    # Randomly assign tokens to experts.
    top_expert = torch.randint(0, ne, (sl * top_k,)).cuda().int()
    _, indices = torch.sort(top_expert)
    indices = indices.int()
    bins = torch.cumsum(torch.histc(top_expert, ne, min=0, max=ne - 1), 0).int()

    # Sample weights for the scatter reduce.
    weights = torch.rand((sl * top_k,)).cuda().half()

    x = ops.binned_gather(x, indices, bins, ec, top_k)

    def binned_scatter(
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ):
        x_np = x.cpu().numpy()
        indices_np = indices.cpu().numpy()
        weights_np = weights.cpu().numpy()
        bins_np = bins.cpu().numpy()
        start = 0
        out = np.zeros((sl, hs))
        for i in range(ne):
            end = bins_np[i]
            for j in range(min(ec, end - start)):
                index = indices_np[start + j]
                scale = weights_np[index]
                index //= top_k

                out[index, :] += scale * x_np[i, j, :]
            start = end
        return torch.from_numpy(out).cuda().half()

    out = ops.binned_scatter(x, indices, weights, bins, top_k)
    expected_out = binned_scatter(x, indices, weights, bins, top_k)

    # NOTE: We need to check approximate equality because the
    # scatter reduce uses atomics.
    assert (
        np.testing.assert_allclose(
            out.cpu(),
            expected_out.cpu(),
            rtol=5e-3,
        )
        is None
    )


@pytest.mark.parametrize("device", DEVICES)
def test_batched_histc(device: torch.device):
    x = torch.tensor([[0, 1, 1], [2, 0, 0]], device=device)
    hist = ops.batched_histc(x, 3)
    torch.testing.assert_close(hist, torch.tensor([[1, 2, 0], [2, 0, 1]], device=device))
