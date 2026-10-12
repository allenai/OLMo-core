"""Span skipping: one routing decision shared by K consecutive blocks.

See :func:`olmo_core.nn.block_skip.set_block_skip_span`. Every claim the benchmark scripts in
``debug/flexcompute_40x/`` make about spans (keep-all init is exact, the decision is genuinely
shared, the lazy end-of-span mix is exact, the budget still averages over blocks) is pinned here.
"""

import torch

import olmo_core.nn.router_fn as rf
from olmo_core.nn.block_skip import set_block_skip_span
from olmo_core.nn.transformer import TransformerConfig

N_LAYERS = 4
VOCAB = 128


def _tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    cfg = TransformerConfig.llama_like(
        d_model=64, n_layers=N_LAYERS, n_heads=4, n_kv_heads=2, vocab_size=VOCAB
    )
    model = cfg.build(init_device="cpu")
    model.init_weights()
    return model


def _patched_logits(pattern):
    """Force ``block_skip_decide`` to see ``pattern`` (a (B, T) float tensor) as its router logits.

    ``block_skip_decide`` imports ``router_logits`` inside its body, so patching the module
    attribute is picked up on the next call -- the same hook ``block_skip_test.py`` uses.
    """
    orig = rf.router_logits

    def routed(h, router, eps=1e-6):
        out = orig(h, router, eps)
        return out * 0.0 + pattern.to(out.dtype)[..., None]

    return orig, routed


def test_span_reproduces_base_at_init():
    """Keep-all init (router bias +10) must reproduce the base model under every span setting."""
    ids = torch.randint(0, VOCAB, (2, 16))
    model = _tiny_model()
    with torch.no_grad():
        ref = model(ids)
    model.enable_block_skip(target=0.5)
    for span in (1, 2, 4):
        set_block_skip_span(model, span)
        for lazy in (False, True):
            model._block_skip["holder"].lazy_span_mix = lazy
            with torch.no_grad():
                out = model(ids)
            torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_one_router_per_span_and_the_decision_is_shared():
    model = _tiny_model()
    model.enable_block_skip(target=0.5)
    assert len(model.bskip_routers) == N_LAYERS
    set_block_skip_span(model, 2)
    assert sorted(int(k) for k in model.bskip_routers) == [0, 2]
    # The dropped routers must leave the state dict: a span model has fewer parameters.
    assert sorted(k for k in model.state_dict() if k.startswith("bskip_routers")) == [
        "bskip_routers.0.w.bias",
        "bskip_routers.0.w.weight",
        "bskip_routers.2.w.bias",
        "bskip_routers.2.w.weight",
    ]

    ids = torch.randint(0, VOCAB, (1, 12))
    pattern = torch.full((1, 12), 5.0)
    pattern[0, 3] = -5.0  # token 3 skips
    orig, routed = _patched_logits(pattern)
    rf.router_logits = routed
    try:
        with torch.no_grad():
            model(ids)
    finally:
        rf.router_logits = orig
    keep = model._block_skip["holder"].per_layer_keep(last_forward=False)
    assert sorted(keep) == list(range(N_LAYERS))  # every block is accounted for, not just the heads
    assert all(abs(v - 11 / 12) < 1e-9 for v in keep.values())


def test_lazy_end_of_span_mix_is_exact():
    """Deferring the where/straight-through to the end of the span changes nothing numerically."""
    ids = torch.randint(0, VOCAB, (1, 12))
    pattern = torch.full((1, 12), 5.0)
    pattern[0, 2] = pattern[0, 7] = -5.0
    outs = {}
    for lazy in (False, True):
        model = _tiny_model()
        model.enable_block_skip(target=0.5)
        set_block_skip_span(model, 2)
        model._block_skip["holder"].lazy_span_mix = lazy
        orig, routed = _patched_logits(pattern)
        rf.router_logits = routed
        try:
            with torch.no_grad():
                outs[lazy] = model(ids)
        finally:
            rf.router_logits = orig
    torch.testing.assert_close(outs[True], outs[False], atol=1e-6, rtol=1e-6)
    # ...and it is a real skip, not a no-op that trivially agrees.
    assert model._block_skip["holder"].mean_keep(last_forward=False) < 1.0


def test_budget_still_averages_over_blocks_not_decisions():
    """A span of K contributes K identical terms, so `target` keeps meaning 'fraction of block
    passes' and a span model's budget loss matches the per-block model's on the same decision."""
    ids = torch.randint(0, VOCAB, (1, 12))
    pattern = torch.full((1, 12), 5.0)
    pattern[0, 5] = -5.0
    losses, keeps = {}, {}
    for span in (1, 2, 4):
        model = _tiny_model()
        model.enable_block_skip(target=0.5, budget_weight=1.0)
        set_block_skip_span(model, span)
        orig, routed = _patched_logits(pattern)
        rf.router_logits = routed
        try:
            # labels: the holder only collects budget terms when a loss is being computed.
            model(ids, labels=torch.randint(0, VOCAB, ids.shape), loss_reduction="sum")
        finally:
            rf.router_logits = orig
        holder = model._block_skip["holder"]
        losses[span] = float(holder.regularization_loss().detach())
        keeps[span] = holder.mean_keep(last_forward=False)
        assert len(holder._exp_layers) == N_LAYERS
    assert abs(losses[1] - losses[2]) < 1e-6 and abs(losses[1] - losses[4]) < 1e-6
    assert abs(keeps[1] - keeps[2]) < 1e-9 and abs(keeps[1] - keeps[4]) < 1e-9


def test_span_router_gets_gradient_from_every_block_in_its_span():
    ids = torch.randint(0, VOCAB, (1, 12))
    model = _tiny_model()
    model.enable_block_skip(target=0.5, budget_weight=0.0)
    set_block_skip_span(model, 4)
    assert list(model.bskip_routers) == ["0"]
    model(ids).sum().backward()
    g = model.bskip_routers["0"].w.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_span_roundtrip_restores_per_block_routing():
    model = _tiny_model()
    model.enable_block_skip(target=0.5)
    before = {k: v.w.weight.clone() for k, v in model.bskip_routers.items()}
    set_block_skip_span(model, 2)
    assert len(model.bskip_routers) == 2
    assert set_block_skip_span(model, 1) == N_LAYERS
    assert model._block_skip["holder"].span == 1
    for k, v in model.bskip_routers.items():
        torch.testing.assert_close(v.w.weight, before[k])
    ids = torch.randint(0, VOCAB, (1, 8))
    with torch.no_grad():
        model(ids)  # per-block routing works again
