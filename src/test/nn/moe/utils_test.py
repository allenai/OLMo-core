import torch

from olmo_core.nn.moe import utils
from olmo_core.nn.moe.v2 import routed_experts


def test_moe_permutation_torch_fallback(monkeypatch):
    monkeypatch.setattr(utils, "moe_permute", None)
    monkeypatch.setattr(utils, "moe_unpermute", None)

    inp = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        requires_grad=True,
    )
    routing_map = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.int32)

    permuted, row_id_map = utils.moe_permute_no_compile(
        inp=inp,
        routing_map=routing_map,
        num_out_tokens=routing_map.numel(),
        map_type="index",
    )

    torch.testing.assert_close(permuted, inp[[0, 1, 2, 0, 1, 2]])
    torch.testing.assert_close(
        row_id_map,
        torch.tensor([3, 0, 1, 4, 5, 2], dtype=torch.int32),
    )

    merging_probs = torch.tensor([[0.25, 0.75], [0.6, 0.4], [0.1, 0.9]])
    restored = utils.moe_unpermute_no_compile(
        inp=permuted,
        row_id_map=row_id_map,
        merging_probs=merging_probs,
        restore_shape=inp.shape,
        map_type="index",
    )

    torch.testing.assert_close(restored, inp)
    restored.sum().backward()
    torch.testing.assert_close(inp.grad, torch.ones_like(inp))


def test_grouped_mm_torch_fallback(monkeypatch):
    monkeypatch.setattr(routed_experts, "grouped_gemm", None)

    inp = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    weights = torch.tensor(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
        ],
        requires_grad=True,
    )
    batch_sizes = torch.tensor([2, 1])

    actual = routed_experts.gmm_no_compile(inp, weights, batch_sizes)
    expected = torch.cat((inp[:2] @ weights[0], inp[2:] @ weights[1]))
    torch.testing.assert_close(actual, expected)

    actual.sum().backward()
    assert inp.grad is not None
    assert weights.grad is not None
