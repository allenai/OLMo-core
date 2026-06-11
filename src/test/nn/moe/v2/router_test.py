import torch

from olmo_core.config import DType
from olmo_core.nn.moe.router import MoERouterGatingFunction
from olmo_core.nn.moe.v2.router import MoERouterConfigV2


def _build(*, top_k=2, num_experts=8, d_model=16, **kwargs):
    return MoERouterConfigV2(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        dtype=DType.float32,
        **kwargs,
    ).build(init_device="cpu")


def test_router_config_num_params():
    cfg = MoERouterConfigV2(d_model=16, num_experts=8, top_k=2)
    # The router weight is (num_experts * d_model).
    assert cfg.num_params() == 16 * 8


def test_router_forward_shapes_and_invariants():
    torch.manual_seed(0)
    B, S, D, E, K = 2, 4, 16, 8, 2
    router = _build(top_k=K, num_experts=E, d_model=D)

    x = torch.randn(B, S, D)
    weights, indices, batch_size_per_expert, aux = router(x, False)

    assert weights.shape == (B, S, K)
    assert indices.shape == (B, S, K)
    assert batch_size_per_expert.shape == (E,)
    # Every selected expert index is valid.
    assert int(indices.min()) >= 0 and int(indices.max()) < E
    # Each of B*S tokens is routed to exactly top_k experts.
    assert int(batch_size_per_expert.sum()) == B * S * K
    # Forward returns the auxiliary-loss inputs (not the reduced losses).
    assert aux is not None


def test_router_scores_only_short_circuits():
    B, S, D, E = 2, 4, 16, 8
    router = _build(num_experts=E, d_model=D)

    scores, indices, batch_size_per_expert, aux = router(torch.randn(B, S, D), True)

    assert scores.shape == (B, S, E)
    assert indices is None and batch_size_per_expert is None and aux is None


def test_router_top1_and_sigmoid_gating():
    router = _build(top_k=1, gating_function=MoERouterGatingFunction.sigmoid)

    weights, indices, _, _ = router(torch.randn(2, 4, 16), False)

    assert weights.shape == (2, 4, 1)
    assert indices.shape == (2, 4, 1)
