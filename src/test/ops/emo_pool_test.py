import pytest
import torch

from olmo_core.ops.moe import pool_keep_mask, pool_keep_mask_inverse_scatter


@pytest.mark.parametrize("experts", [1, 16, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pool_inverse_scatter_exact(experts, dtype):
    torch.manual_seed(12)
    scores = torch.randint(-3, 4, (2, 31, experts)).to(dtype)
    # Ties and non-finite values exercise precisely the original first-sort ordering.
    scores[0, 0, 0] = float("nan")
    scores[0, 1, 0] = float("inf")
    scores[0, 2, 0] = -float("inf")
    pools = torch.randint(0, experts + 1, (2, 31))
    expected = pool_keep_mask(scores, pools)
    actual = pool_keep_mask_inverse_scatter(scores, pools)
    assert torch.equal(actual, expected)
    assert torch.equal(actual.sum(-1), pools)
