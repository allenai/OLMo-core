from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.train.callbacks import ProfilerAnnotationCallback
from olmo_core.train.callbacks.profiler_annotation import block_annotation_name

VOCAB_SIZE = 128
SEQ_LEN = 8


def _tiny_model(n_layers: int = 3) -> nn.Module:
    return TransformerConfig(
        d_model=64,
        vocab_size=VOCAB_SIZE,
        n_layers=n_layers,
        block=TransformerBlockConfig(
            name=TransformerBlockType.peri_norm,
            sequence_mixer=AttentionConfig(name=AttentionType.default, n_heads=4),
            feed_forward=FeedForwardConfig(hidden_size=128),
            layer_norm=LayerNormConfig(),
        ),
        lm_head=LMHeadConfig(layer_norm=LayerNormConfig()),
    ).build(init_device="cpu")


def _attach(callback: ProfilerAnnotationCallback, model: nn.Module) -> ProfilerAnnotationCallback:
    trainer = Mock()
    trainer.train_module.model = model
    trainer.train_module.optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer.callbacks = {}
    trainer.global_step = 0
    callback._trainer = trainer
    return callback


def _batch():
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    labels = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    return input_ids, labels


def _train_batch(model: nn.Module):
    input_ids, labels = _batch()
    out = model(input_ids, labels=labels, loss_reduction="sum")
    out.loss.backward()


class _FakeMixer(nn.Module):
    pass


class GatedDeltaNet(_FakeMixer):  # noqa: D101 - stands in for the real class by name
    pass


class Attention(_FakeMixer):  # noqa: D101 - stands in for the real class by name
    pass


class KimiDeltaAttention(_FakeMixer):  # noqa: D101 - stands in for the real class by name
    pass


class _FakeBlock(nn.Module):
    def __init__(self, mixer: nn.Module, is_moe: bool = False):
        super().__init__()
        self.attention = mixer
        self._is_moe = is_moe

    @property
    def is_moe(self) -> bool:
        return self._is_moe


def test_block_annotation_names():
    assert block_annotation_name(_FakeBlock(GatedDeltaNet()), 0) == "block00.gdn"
    assert block_annotation_name(_FakeBlock(Attention()), 4) == "block04.attn"
    assert block_annotation_name(_FakeBlock(Attention()), 4, index_width=3) == "block004.attn"
    assert block_annotation_name(_FakeBlock(KimiDeltaAttention()), 2) == "block02.kda"
    assert block_annotation_name(_FakeBlock(GatedDeltaNet(), is_moe=True), 7) == "block07.gdn+moe"


def test_disabled_registers_no_hooks():
    model = _tiny_model()
    callback = _attach(ProfilerAnnotationCallback(enabled=False), model)
    callback.pre_train()
    assert callback._handles is None
    assert len(model.blocks["0"]._forward_pre_hooks) == 0


@pytest.mark.parametrize("depth", [1, 2])
def test_markers_present_and_cleaned_up(depth: int):
    model = _tiny_model()
    callback = _attach(
        ProfilerAnnotationCallback(enabled=True, follow_profiler=False, start=0, depth=depth), model
    )
    callback.pre_train()
    assert callback._handles

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        callback.pre_load_batch()
        callback.pre_step({})
        _train_batch(model)
        callback.pre_optim_step()
        callback.trainer.train_module.optim.step()
        callback.post_step()

    names = {event.key for event in prof.key_averages()}
    expected = {
        "data_loading",
        "fwd",
        "bwd",
        "fwd/block00.attn",
        "bwd/block00.attn",
        "fwd/block02.attn",
        "bwd/block02.attn",
        "fwd/lm_head",
        "bwd/lm_head",
        "optim_step",
        "optim_step/pre",
    }
    if depth == 2:
        expected |= {"fwd/block00.attn/mixer", "fwd/block00.attn/ffn"}
    assert expected <= names, f"missing: {sorted(expected - names)}"

    # Nothing may be left open.
    assert callback._fwd_stack == []
    assert callback._bwd_open == {}


def test_backward_ranges_run_in_reverse_block_order():
    model = _tiny_model()
    callback = _attach(
        ProfilerAnnotationCallback(enabled=True, follow_profiler=False, start=0), model
    )
    callback.pre_train()

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        callback.pre_load_batch()
        callback.pre_step({})
        _train_batch(model)
        callback.post_step()

    starts = {}
    for event in prof.events():
        if event.name.startswith("bwd/block") and event.name not in starts:
            starts[event.name] = event.time_range.start
    assert set(starts) == {"bwd/block00.attn", "bwd/block01.attn", "bwd/block02.attn"}
    assert starts["bwd/block02.attn"] < starts["bwd/block01.attn"] < starts["bwd/block00.attn"]


def test_no_leak_without_grad():
    model = _tiny_model()
    callback = _attach(
        ProfilerAnnotationCallback(enabled=True, follow_profiler=False, start=0), model
    )
    callback.pre_train()
    callback.pre_load_batch()
    callback.pre_step({})
    with torch.no_grad():
        input_ids, _ = _batch()
        model(input_ids)
    callback.post_step()
    assert callback._fwd_stack == []
    assert callback._bwd_open == {}


def test_no_leak_on_exception():
    model = _tiny_model()
    callback = _attach(
        ProfilerAnnotationCallback(enabled=True, follow_profiler=False, start=0), model
    )
    callback.pre_train()
    callback.pre_load_batch()
    callback.pre_step({})

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    model.blocks["1"].feed_forward.forward = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="boom"):
        _train_batch(model)
    callback.on_error(RuntimeError("boom"))
    assert callback._fwd_stack == []
    assert callback._bwd_open == {}


def test_grad_accumulation():
    model = _tiny_model()
    callback = _attach(
        ProfilerAnnotationCallback(enabled=True, follow_profiler=False, start=0), model
    )
    callback.pre_train()

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        callback.pre_load_batch()
        callback.pre_step({})
        _train_batch(model)
        _train_batch(model)
        callback.pre_optim_step()
        callback.post_step()

    fwd = [event for event in prof.key_averages() if event.key == "fwd"]
    assert fwd and fwd[0].count == 2
    assert callback._fwd_stack == []
    assert callback._bwd_open == {}


def test_step_window_gating():
    model = _tiny_model()
    callback = _attach(
        ProfilerAnnotationCallback(enabled=True, follow_profiler=False, start=5, end=6), model
    )
    callback.pre_train()
    observed = {}
    for step in (3, 4, 5, 6):
        callback.trainer.global_step = step - 1  # incremented after 'pre_load_batch'
        callback.pre_load_batch()
        observed[step] = callback._active
        callback.post_step()
    assert observed == {3: False, 4: False, 5: True, 6: True}


def test_compiled_blocks_have_no_graph_breaks():
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()

    model = _tiny_model(n_layers=2)
    for block in model.blocks.values():
        block.compile(fullgraph=False)

    callback = _attach(
        ProfilerAnnotationCallback(enabled=True, follow_profiler=False, start=0), model
    )
    callback.pre_train()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        callback.pre_load_batch()
        callback.pre_step({})
        _train_batch(model)
        callback.post_step()

    breaks = sum(counters["graph_break"].values())
    assert breaks == 0, f"unexpected graph breaks: {dict(counters['graph_break'])}"
    names = {event.key for event in prof.key_averages()}
    assert "fwd/block00.attn" in names
    assert "bwd/block00.attn" in names
