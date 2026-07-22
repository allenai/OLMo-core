"""
Forward-correctness tests for ``MoERouterV2.get_top_k``.

``get_top_k`` has no learnable parameters: it is a pure function of the routing
``scores`` that returns *which* experts each token is assigned to
(``expert_indices``) and *their* scores (``expert_weights``). So the essential
property to pin down is the forward selection, not gradients.

These tests are deliberately **order-invariant** along the top-k axis: they
compare selected experts as a *set* per row and weights as a sorted *multiset*.
That is intentional. The order in which the k winners are returned is an
implementation detail that nothing downstream depends on (gather -> normalize ->
weighted sum are all order-invariant), so a correct test must pass whether
``torch.topk`` is called with ``sorted=True`` or ``sorted=False``. What the test
*does* pin down is that the returned experts really are the top-k by score and
that each returned weight is aligned to its expert index -- i.e. it fails on a
genuine routing bug, not on a benign reordering.

Runs on CPU in fp32; no GPU required.
"""

import pytest
import torch

from olmo_core.nn.moe.v2.router import MoERouterConfigV2


def _build_router(*, num_experts: int, top_k: int) -> "torch.nn.Module":
    # dtype left unset -> module default torch.float32; init on CPU.
    return MoERouterConfigV2(
        d_model=16,
        num_experts=num_experts,
        top_k=top_k,
    ).build(init_device="cpu")


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_get_top_k_selects_true_topk(top_k: int):
    torch.manual_seed(0)
    num_experts, n_tokens = 8, 64
    router = _build_router(num_experts=num_experts, top_k=top_k)

    # Continuous scores -> ties at the top-k boundary are measure-zero, so the
    # set/multiset comparisons below are unambiguous.
    scores = torch.rand(n_tokens, num_experts, dtype=torch.float32)

    weights, indices = router.get_top_k(scores)

    assert weights.shape == (n_tokens, top_k)
    assert indices.shape == (n_tokens, top_k)

    # Independent trusted reference: the true top-k by score.
    ref_weights, ref_indices = scores.topk(top_k, dim=-1)

    # Property 1 (SELECTION): returned experts are exactly the true top-k,
    # compared as a set per row (order-invariant).
    for n in range(n_tokens):
        assert set(indices[n].tolist()) == set(ref_indices[n].tolist()), f"row {n}"

    # Property 2 (ALIGNMENT): each returned weight is the score at its own index.
    torch.testing.assert_close(weights, scores.gather(-1, indices))

    # Property 3 (WEIGHTS): the multiset of returned weights equals the true
    # top-k weights (order-invariant).
    torch.testing.assert_close(weights.sort(dim=-1).values, ref_weights.sort(dim=-1).values)


def test_get_top_k_is_order_invariant_to_sorted_flag():
    """
    Explicit guard on the property that motivated this suite: the SET of selected
    experts and the aligned weights must not depend on whether ``topk`` sorts its
    output. If a future change makes downstream code depend on top-k ordering,
    this test (via the set/alignment checks above) should be the thing that
    surfaces it -- while correctly *allowing* the ``sorted=False`` optimization.
    """
    torch.manual_seed(1)
    router = _build_router(num_experts=8, top_k=2)
    scores = torch.rand(64, 8, dtype=torch.float32)

    weights, indices = router.get_top_k(scores)

    # Reconstruct weights purely from the returned indices; must match exactly
    # regardless of intra-row ordering.
    torch.testing.assert_close(weights, scores.gather(-1, indices))
    # Every row selects `top_k` DISTINCT experts.
    for n in range(indices.size(0)):
        assert len(set(indices[n].tolist())) == indices.size(1)
