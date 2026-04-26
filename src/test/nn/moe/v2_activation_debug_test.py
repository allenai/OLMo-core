import torch
from torch import nn

from olmo_core.nn.moe.v2 import activation_debug


class _DebugBlock(nn.Module):
    def __init__(self, *, block_idx: int = 3):
        super().__init__()
        self.block_idx = block_idx
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.calls = 0

    def combined_forward_ep_no_sync(self, x, *, loss_div_factor=None, **kwargs):
        del loss_div_factor, kwargs
        self.calls += 1
        return x * self.weight


def test_ep_no_sync_activation_debug_returns_none_when_gate_is_closed(monkeypatch):
    block = _DebugBlock(block_idx=3)
    x = torch.ones(8, 512, requires_grad=True)

    monkeypatch.setattr(
        activation_debug,
        "_get_train_global_arg",
        lambda key, default=None: False if key == "dry_run_done" else default,
    )

    out = activation_debug.maybe_dump_ep_no_sync_saved_activations(
        block,
        x,
        loss_div_factor=None,
        forward_kwargs={},
    )

    assert out is None
    assert block.calls == 0


def test_ep_no_sync_activation_debug_runs_once_and_sets_global_flag(monkeypatch):
    block = _DebugBlock(block_idx=3)
    x = torch.ones(8, 512, requires_grad=True)
    global_args = {
        "dry_run_done": True,
        "ep_no_sync_saved_activations_dumped_block_3": False,
    }

    monkeypatch.setattr(
        activation_debug,
        "_get_train_global_arg",
        lambda key, default=None: global_args.get(key, default),
    )
    monkeypatch.setattr(
        activation_debug,
        "_set_train_global_arg",
        lambda key, value: global_args.__setitem__(key, value),
    )
    monkeypatch.setattr(activation_debug, "get_rank", lambda: 0)
    monkeypatch.setattr(activation_debug.torch.cuda, "memory_allocated", lambda: 0)

    out = activation_debug.maybe_dump_ep_no_sync_saved_activations(
        block,
        x,
        loss_div_factor=None,
        forward_kwargs={"unused_kwarg": object()},
    )

    torch.testing.assert_close(out, x * 2)
    assert block.calls == 1
    assert global_args["ep_no_sync_saved_activations_dumped_block_3"] is True
