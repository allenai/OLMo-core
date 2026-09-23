"""Numerical checks against real Core modules and HF reference models."""

import contextlib

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from olmo_core.nn.moe.v2.replay import replay_routes
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.train.train_module.transformer.objective import train_batch_with_loss


class ObjectiveModule:
    def __init__(self, model):
        self.model = model
        self.contexts = []

    @contextlib.contextmanager
    def _train_microbatch_context(self, index, count):
        self.contexts.append((index, count))
        yield


def test_custom_objective_accumulation_matches_full_batch():
    torch.manual_seed(17)
    model = nn.Linear(4, 3)
    reference = nn.Linear(4, 3)
    reference.load_state_dict(model.state_dict())
    x = torch.randn(5, 4)
    labels = torch.tensor([0, 1, 2, 0, 2])
    module = ObjectiveModule(model)

    def objective(module, batch):
        loss = F.cross_entropy(module.model(batch["x"]), batch["y"], reduction="sum") / 5
        return loss, {"loss": loss}

    metrics = train_batch_with_loss(
        module, [{"x": x[:2], "y": labels[:2]}, {"x": x[2:], "y": labels[2:]}], objective
    )
    F.cross_entropy(reference(x), labels).backward()
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)
    assert module.contexts == [(0, 2), (1, 2)]
    assert all(not metric["loss"].requires_grad for metric in metrics)


@pytest.mark.parametrize("recompute", [False, True])
def test_router_replay_keeps_experts_and_router_gradients(recompute):
    router = MoERouterConfigV2(d_model=8, num_experts=4, top_k=2).build()
    (
        router.reset_parameters()
        if hasattr(router, "reset_parameters")
        else nn.init.normal_(router.weight, std=0.1)
    )
    model = nn.Module()
    model.add_module("routed_experts_router", router)
    indices = torch.tensor([[[0, 2], [1, 3], [2, 3]]])
    x = torch.randn(1, 3, 8, requires_grad=True)
    seen = []

    def forward(x):
        weights, actual, _, _ = router(x, scores_only=False)
        seen.append(actual.detach().clone())
        return weights

    with replay_routes(model, {"routed_experts_router": indices}):
        weights = checkpoint(forward, x, use_reentrant=True) if recompute else forward(x)
        (weights * torch.tensor([1.0, 2.0])).sum().backward()
    assert seen and all(torch.equal(actual, indices) for actual in seen)
    assert torch.isfinite(router.weight.grad).all() and router.weight.grad.abs().sum() > 0
    assert router.replay_expert_indices is None
