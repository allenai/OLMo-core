import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import seed_all


def _model(enable=True, **kw):
    cfg = TransformerConfig.olmo2_190M(
        vocab_size=1000, n_layers=3, fused_ops=False, dtype=DType.float32
    )
    model = cfg.build(init_device="cpu")
    if enable:
        model.enable_ffn_token_drop(**kw)
    return model


def test_no_new_state_dict_keys():
    seed_all(0)
    base = _model(enable=False)
    dropped = _model(enable=True, max_rate=0.5)
    assert set(base.state_dict()) == set(dropped.state_dict())


def test_eval_mode_is_the_dense_model():
    seed_all(1)
    m = _model(max_rate=0.9, layer_prob=0.5)
    m.eval()
    ff = m.blocks["1"].feed_forward
    x = torch.randn(2, 32, m.d_model)
    torch.testing.assert_close(ff(x), ff._fdrop_orig_forward(x))


def test_zero_rates_are_the_dense_model_in_training():
    seed_all(2)
    m = _model(max_rate=0.0, layer_prob=0.0)
    m.train()
    ff = m.blocks["1"].feed_forward
    x = torch.randn(2, 32, m.d_model)
    torch.testing.assert_close(ff(x), ff._fdrop_orig_forward(x))


def test_start_layer_is_never_dropped():
    m = _model(max_rate=1.0, start_layer=1)
    assert not hasattr(m.blocks["0"].feed_forward, "_fdrop_orig_forward")
    assert hasattr(m.blocks["1"].feed_forward, "_fdrop_orig_forward")


def test_dropped_tokens_are_exactly_zero_and_kept_tokens_exact():
    seed_all(3)
    m = _model(max_rate=1.0)
    m.train()
    holder = m._ffn_token_drop["holder"]
    holder.begin_forward(training=True)
    ff = m.blocks["2"].feed_forward
    x = torch.randn(2, 64, m.d_model)
    out = ff(x)
    dense = ff._fdrop_orig_forward(x)
    zero = (out == 0).all(-1)
    assert 0 < zero.sum() < zero.numel()
    torch.testing.assert_close(out[~zero], dense[~zero])


def test_whole_layer_drop_keeps_zero_grads_for_fsdp():
    seed_all(4)
    m = _model(max_rate=0.0, layer_prob=1.0)
    m.train()
    m._ffn_token_drop["holder"].begin_forward(training=True)
    ff = m.blocks["1"].feed_forward
    x = torch.randn(1, 16, m.d_model, requires_grad=True)
    out = ff(x)
    assert (out == 0).all()
    out.sum().backward()
    for p in ff.parameters():
        assert p.grad is not None and (p.grad == 0).all()


def test_draws_are_deterministic_within_a_forward():
    """Activation checkpointing re-runs the block in backward without a new begin_forward; the
    recompute must drop exactly the same tokens."""
    seed_all(5)
    m = _model(max_rate=0.8, layer_prob=0.2)
    m.train()
    holder = m._ffn_token_drop["holder"]
    ff = m.blocks["1"].feed_forward
    x = torch.randn(2, 64, m.d_model)
    holder.begin_forward(training=True)
    with torch.no_grad():
        first, recompute = ff(x), ff(x)
    torch.testing.assert_close(first, recompute)
    holder.begin_forward(training=True)
    with torch.no_grad():
        later = ff(x)
    assert not torch.equal(first, later)


def test_realized_rate_matches_schedule():
    seed_all(6)
    m = _model(max_rate=0.6, layer_prob=0.0)
    m.train()
    holder = m._ffn_token_drop["holder"]
    ff = m.blocks["1"].feed_forward
    x = torch.randn(4, 256, m.d_model)
    with torch.no_grad():
        for _ in range(50):
            holder.begin_forward(training=True)
            ff(x)
    frac = holder.pop_metrics()["ffn_drop/frac"]
    assert frac == pytest.approx(0.3, abs=0.03)  # E[r] = max_rate / 2


def test_full_model_trains_under_drop():
    seed_all(7)
    m = _model(max_rate=0.5, layer_prob=0.1)
    m.train()
    ids = torch.randint(0, 1000, (2, 32))
    logits = m(ids)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 1000), ids[:, 1:].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert m._ffn_token_drop["holder"].calls == 1
