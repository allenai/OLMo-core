"""Numerical checks against real Core modules and HF reference models."""

import contextlib

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.moe.v2.replay import replay_routes
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig
from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig
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


def test_replay_rejects_emo_before_setting_router_state():
    from olmo_core.nn.moe.emo import EmoRouterConfig

    model = nn.Module()
    standard = MoERouterConfigV2(d_model=4, num_experts=4, top_k=2).build()
    emo = MoERouterConfigV2(
        d_model=4,
        num_experts=4,
        top_k=2,
        emo=EmoRouterConfig(eos_token_id=0, min_document_expert_pool=4, max_document_expert_pool=4),
    ).build()
    model.add_module("first_routed_experts_router", standard)
    model.add_module("emo_routed_experts_router", emo)
    routes = {name: torch.tensor([[[0, 1]]]) for name, _ in model.named_children()}
    with pytest.raises(ValueError, match="does not support EmoRouterV2"):
        with replay_routes(model, routes):
            pytest.fail("Unsupported replay must fail before entering the context")
    assert getattr(standard, "replay_expert_indices", None) is None
    assert getattr(emo, "replay_expert_indices", None) is None


def test_nested_replay_restores_routes_after_failure():
    model = nn.Module()
    router = MoERouterConfigV2(d_model=4, num_experts=4, top_k=2).build()
    model.add_module("routed_experts_router", router)
    outer, inner = torch.tensor([[[0, 1]]]), torch.tensor([[[2, 3]]])
    with replay_routes(model, {"routed_experts_router": outer}):
        with pytest.raises(RuntimeError, match="injected"):
            with replay_routes(model, {"routed_experts_router": inner}):
                assert router.replay_expert_indices is inner
                raise RuntimeError("injected")
        assert router.replay_expert_indices is outer
    assert router.replay_expert_indices is None


@pytest.mark.parametrize("objective_fails", [False, True])
def test_eval_after_custom_objective_restores_eval_mode(objective_fails):
    model = TransformerConfig.llama_like(
        d_model=16,
        vocab_size=32,
        n_layers=1,
        n_heads=2,
        feed_forward=FeedForwardConfig(hidden_size=32, bias=False),
        attn_backend=AttentionBackendName.torch,
    ).build(init_device="cpu")
    module = TransformerTrainModuleConfig(
        rank_microbatch_size=4,
        max_sequence_length=4,
        optim=AdamWConfig(),
    ).build(model, device=torch.device("cpu"))
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    forward_modes = []
    hook = module.model.register_forward_pre_hook(
        lambda model, _args: forward_modes.append(model.training)
    )

    def objective(module, batch):
        logits = module.model_forward(batch["input_ids"])
        assert isinstance(logits, torch.Tensor)
        if objective_fails:
            raise RuntimeError("injected objective failure")
        loss = logits.square().mean()
        return loss, {"loss": loss}

    try:
        before = module.eval_batch(dict(batch))
        module.zero_grads()
        with (
            pytest.raises(RuntimeError, match="injected objective failure")
            if objective_fails
            else contextlib.nullcontext()
        ):
            module.train_batch_with_loss([batch], objective)
        after = module.eval_batch(dict(batch))
    finally:
        hook.remove()

    # Check the mode at the actual forwards, including after a failed objective.
    assert forward_modes == [False, True, False]
    assert not module.model.training
    # No optimizer step occurred, so evaluation must still give the same logits.
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    if not objective_fails:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
