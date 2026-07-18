from types import SimpleNamespace
from typing import Any

import pytest
import torch

import olmo_core.nn.ddp.model as ddp_model_module
import olmo_core.nn.moe.v2.ep_deepep_v2 as ep_deepep_v2_module
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock
from olmo_core.nn.ddp.model import OLMoDDPModel
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath


def _stub_block(
    path: ExpertParallelPath,
    *,
    ep_enabled: bool = True,
    training: bool = True,
) -> OLMoDDPTransformerBlock:
    block = object.__new__(OLMoDDPTransformerBlock)
    torch.nn.Module.__init__(block)
    block.routed_experts = torch.nn.Identity()
    block._ep_enabled = ep_enabled
    block._ep_no_sync_rowwise_static_checkpoint_state = None
    block.ep = ExpertParallelConfig(path=path)
    block.train(training)
    return block


@pytest.mark.parametrize(
    ("path", "ep_enabled", "training", "expected_reentrant"),
    [
        (ExpertParallelPath.deepep_v2, True, True, True),
        (ExpertParallelPath.sync_1d, True, True, False),
        (ExpertParallelPath.rowwise_nvshmem, True, True, False),
        (ExpertParallelPath.deepep_v2, False, True, False),
        (ExpertParallelPath.deepep_v2, True, False, False),
    ],
)
def test_per_block_checkpoint_uses_reentrant_only_for_deepep(
    monkeypatch: pytest.MonkeyPatch,
    path: ExpertParallelPath,
    ep_enabled: bool,
    training: bool,
    expected_reentrant: bool,
) -> None:
    checkpoint_kwargs = []
    forwarded_kwargs = []
    forwarded_anchors = []

    def fake_checkpoint(function, *args, **kwargs):
        checkpoint_kwargs.append(kwargs)
        return function(*args)

    def forward_one_block(h, _block_key, block_kwargs, grad_anchor=None):
        forwarded_kwargs.append(block_kwargs)
        forwarded_anchors.append(grad_anchor)
        return h + 1

    monkeypatch.setattr(ddp_model_module, "checkpoint", fake_checkpoint)
    model: Any = SimpleNamespace(
        blocks={
            "0": _stub_block(
                path,
                ep_enabled=ep_enabled,
                training=training,
            )
        },
        compile_enabled=False,
        recompute_each_block=True,
        recompute_block_keys=None,
        _forwrad_one_block=forward_one_block,
    )

    out = OLMoDDPModel._forward_blocks(model, torch.zeros(1), {}, {})

    torch.testing.assert_close(out, torch.ones(1))
    assert len(checkpoint_kwargs) == 1
    assert checkpoint_kwargs[0]["use_reentrant"] is expected_reentrant
    if expected_reentrant:
        assert "context_fn" not in checkpoint_kwargs[0]
        assert checkpoint_kwargs[0]["preserve_rng_state"] is True
        assert forwarded_kwargs[0]["deepep_reentrant_checkpoint"] is True
        # h is detached here, so a grad-requiring anchor is passed to keep the
        # reentrant recompute (and DeepEP expert backward) alive.
        assert forwarded_anchors[0] is not None and forwarded_anchors[0].requires_grad
    else:
        assert "context_fn" in checkpoint_kwargs[0]
        assert "deepep_reentrant_checkpoint" not in forwarded_kwargs[0]
        assert forwarded_anchors[0] is None


def test_reentrant_checkpoint_supports_deepep_nested_backward_pattern() -> None:
    class NestedBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, source):
            recv_x = source.detach().requires_grad_(True)
            with torch.enable_grad():
                expert_out = recv_x.square()
            ctx.save_for_backward(recv_x, expert_out)
            return expert_out.detach().clone()

        @staticmethod
        def backward(ctx, grad_out):
            recv_x, expert_out = ctx.saved_tensors
            torch.autograd.backward(expert_out, grad_out)
            assert recv_x.grad is not None
            return recv_x.grad

    torch.manual_seed(1234)
    source = torch.randn(8, requires_grad=True)
    routes = []

    def routed_expert(value: torch.Tensor) -> torch.Tensor:
        route = torch.rand_like(value)
        routes.append(route.detach().clone())
        return NestedBackward.apply(value * route)

    out = torch.utils.checkpoint.checkpoint(
        routed_expert,
        source,
        use_reentrant=True,
    )
    out.sum().backward()

    assert len(routes) == 2
    torch.testing.assert_close(routes[0], routes[1])
    torch.testing.assert_close(source.grad, 2 * source * routes[0].square())


def test_reentrant_checkpoint_accumulates_each_deepep_metric_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_combined_forward(
        _block,
        x,
        *,
        accumulate_routed_aux_loss_metrics,
        accumulate_router_aux_loss_metrics,
        **_kwargs,
    ):
        calls.append(
            (
                torch.is_grad_enabled(),
                accumulate_routed_aux_loss_metrics,
                accumulate_router_aux_loss_metrics,
            )
        )
        return x.square()

    monkeypatch.setattr(
        ep_deepep_v2_module,
        "combined_forward_ep_deepep_v2",
        fake_combined_forward,
    )
    block = _stub_block(ExpertParallelPath.deepep_v2)
    source = torch.randn(8, requires_grad=True)

    out = torch.utils.checkpoint.checkpoint(
        lambda x: OLMoDDPTransformerBlock.combined_forward_ep_deepep_v2(
            block,
            x,
            deepep_reentrant_checkpoint=True,
        ),
        source,
        use_reentrant=True,
    )
    out.sum().backward()

    assert calls == [
        (False, True, True),
        (True, False, True),
    ]
