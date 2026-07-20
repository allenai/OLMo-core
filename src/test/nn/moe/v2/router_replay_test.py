"""Router replay (R3) tests for ``MoERouterV2``.

Replay semantics under test: when replay indices are armed via
``set_replay_expert_indices``, the forward pass must

  1. SELECT exactly the injected experts (bypassing live top-k, score_bias,
     grouping, and uniform assignment), and
  2. compute gate weights LIVE from the current scores at those indices, so
     that gradients still flow into the router weight.

Property 2 is the load-bearing design choice (from Rollout Routing Replay):
a full lookup of recorded weights would starve the router of gradient and
freeze it at rollout-time gate values.

Runs on CPU in fp32; no GPU required.
"""

import pytest
import torch

from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.moe.v2.router import MoERouterConfigV2

D_MODEL = 16
NUM_EXPERTS = 8
TOP_K = 2


def _build_router(gating_function=MoERouterGatingFunction.softmax, **kwargs):
    return MoERouterConfigV2(
        d_model=D_MODEL,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        gating_function=gating_function,
        **kwargs,
    ).build(init_device="cpu")


def _anti_topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """The k WORST experts per row — guaranteed to differ from live selection."""
    return torch.topk(scores, k, dim=-1, largest=False).indices


def test_get_top_k_replay_returns_injected_indices():
    torch.manual_seed(0)
    router = _build_router()
    scores = torch.rand(64, NUM_EXPERTS)

    injected = _anti_topk_indices(scores, TOP_K)
    router.set_replay_expert_indices(injected)
    weights, indices = router.get_top_k(scores)

    torch.testing.assert_close(indices, injected)
    # Live gates: exactly the current scores at the injected indices.
    torch.testing.assert_close(weights, scores.gather(-1, injected))
    # Sanity: this is NOT what live selection would have picked.
    live_weights, live_indices = _build_router().get_top_k(scores)
    assert not torch.equal(indices, live_indices)


def test_forward_replay_selection_and_live_gates():
    torch.manual_seed(1)
    router = _build_router(normalize_expert_weights=1.0)
    x = torch.randn(2, 8, D_MODEL)

    # Reference scores computed exactly as forward does for softmax gating.
    with torch.no_grad():
        ref_scores = router.get_expert_logits(x.float()).float().softmax(dim=-1)

    injected = _anti_topk_indices(ref_scores, TOP_K)
    router.set_replay_expert_indices(injected)
    weights, indices, _, _ = router.forward(x, scores_only=False)

    torch.testing.assert_close(indices, injected)
    # Gates = live scores gathered at injected indices, then L1-normalized.
    expected = ref_scores.gather(-1, injected)
    expected = expected / expected.norm(p=1, dim=-1, keepdim=True)
    torch.testing.assert_close(weights, expected)


def test_forward_replay_gradient_flows_to_router_weight():
    torch.manual_seed(2)
    router = _build_router()
    x = torch.randn(2, 8, D_MODEL)

    with torch.no_grad():
        ref_scores = router.get_expert_logits(x.float()).float().softmax(dim=-1)
    router.set_replay_expert_indices(_anti_topk_indices(ref_scores, TOP_K))

    weights, _, _, _ = router.forward(x, scores_only=False)
    weights.sum().backward()

    assert router.weight.grad is not None
    assert router.weight.grad.abs().sum() > 0


def test_gates_track_live_weights_not_recorded_ones():
    """After a (simulated) weight update, replayed selection is unchanged but
    gate values move — the definitional difference between replaying selection
    and replaying the full router output."""
    torch.manual_seed(3)
    router = _build_router()
    x = torch.randn(2, 8, D_MODEL)
    with torch.no_grad():
        ref_scores = router.get_expert_logits(x.float()).float().softmax(dim=-1)
    injected = _anti_topk_indices(ref_scores, TOP_K)
    router.set_replay_expert_indices(injected)

    w_before, i_before, _, _ = router.forward(x, scores_only=False)
    with torch.no_grad():
        router.weight.add_(torch.randn_like(router.weight) * 0.5)
    w_after, i_after, _, _ = router.forward(x, scores_only=False)

    torch.testing.assert_close(i_before, i_after)  # selection pinned
    assert not torch.allclose(w_before, w_after)  # gates live


def test_clear_replay_restores_live_selection():
    torch.manual_seed(4)
    router = _build_router()
    scores = torch.rand(16, NUM_EXPERTS)
    injected = _anti_topk_indices(scores, TOP_K)

    router.set_replay_expert_indices(injected)
    _, replayed = router.get_top_k(scores)
    router.clear_replay_expert_indices()
    _, live = router.get_top_k(scores)

    torch.testing.assert_close(replayed, injected)
    ref = torch.topk(scores, TOP_K, dim=-1).indices
    for n in range(live.size(0)):
        assert set(live[n].tolist()) == set(ref[n].tolist())


def test_replay_topk_softmax_gating_path():
    """The topk_softmax gating branch bypasses get_top_k; replay must work there too."""
    torch.manual_seed(5)
    router = _build_router(gating_function=MoERouterGatingFunction.topk_softmax)
    x = torch.randn(2, 8, D_MODEL)

    with torch.no_grad():
        logits = router.get_expert_logits(x.float()).float()
    injected = _anti_topk_indices(logits, TOP_K)
    router.set_replay_expert_indices(injected)

    weights, indices, _, _ = router.forward(x, scores_only=False)
    torch.testing.assert_close(indices, injected)
    # Gates: softmax over the LIVE logits at the injected indices.
    expected = logits.gather(-1, injected).softmax(dim=-1)
    torch.testing.assert_close(weights, expected)


def test_replay_validation():
    router = _build_router()
    scores = torch.rand(4, NUM_EXPERTS)

    with pytest.raises(ValueError, match="top_k"):
        router.set_replay_expert_indices(torch.zeros(4, TOP_K + 1, dtype=torch.long))
    with pytest.raises(ValueError, match="out of range"):
        router.set_replay_expert_indices(torch.full((4, TOP_K), NUM_EXPERTS, dtype=torch.long))
    router.set_replay_expert_indices(torch.zeros(3, TOP_K, dtype=torch.long))
    with pytest.raises(ValueError, match="leading dims"):
        router.get_top_k(scores)  # 3 rows armed vs 4 rows of scores
