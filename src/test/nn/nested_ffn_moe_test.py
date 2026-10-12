import pytest
import torch

from olmo_core.config import DType
import olmo_core.nn.nested_ffn_moe as nffn
from olmo_core.nn.nested_ffn_moe import (
    FULL_RUNG_INIT_BIAS,
    NestedFFNHolder,
    apply_ffn_permutation,
    ffn_importance_permutation,
    install_nested_ffn_moe,
    resolve_rung_widths,
)
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import seed_all

EOS = 999


def _model(enable=True, start_layer=0, **kw):
    cfg = TransformerConfig.olmo2_190M(
        vocab_size=1000, n_layers=2, fused_ops=False, dtype=DType.float32
    )
    model = cfg.build(init_device="cpu")
    if enable:
        model.enable_nested_ffn_moe(start_layer=start_layer, **kw)
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# rung resolution
# --------------------------------------------------------------------------------------


def test_resolve_rung_widths_qwen3_4b():
    widths, costs = resolve_rung_widths(9728, (1, 4, 16, 64))
    assert widths == [9728, 2432, 608, 152, 0]
    assert costs[0] == 1.0 and costs[-1] == 0.0
    assert costs[3] == pytest.approx(152 / 9728, rel=1e-6)
    # the ladder spans a 64x dynamic range plus a free rung
    assert 1 / costs[3] == pytest.approx(64.0, rel=1e-2)


def test_resolve_rung_widths_no_null():
    widths, costs = resolve_rung_widths(1024, (1, 8), include_null=False)
    assert widths == [1024, 128]
    assert costs == [1.0, 0.125]


@pytest.mark.parametrize(
    "divisors", [(4, 1), (1, 1), (0.5, 2)], ids=["descending", "duplicate", "below-one"]
)
def test_resolve_rung_widths_rejects_bad_divisors(divisors):
    with pytest.raises(ValueError):
        resolve_rung_widths(1024, divisors)


def test_resolve_rung_widths_rejects_collapsing_divisors():
    # 1/500 and 1/600 of 1024 both round to the 8-unit floor
    with pytest.raises(ValueError, match="duplicate widths"):
        resolve_rung_widths(1024, (1, 500, 600))


# --------------------------------------------------------------------------------------
# init is exactly the base model
# --------------------------------------------------------------------------------------


def test_router_init_reproduces_base_model_exactly():
    """The whole point of the zero-init router: enabling the mixture on a trained checkpoint must
    not move a single logit until the router is trained."""
    seed_all(0)
    m_moe = _model(enable=True)
    seed_all(0)
    m_plain = _model(enable=False)
    x = torch.randint(5, 800, (2, 16))
    x[:, -1] = EOS
    with torch.no_grad():
        torch.testing.assert_close(m_moe(x), m_plain(x))


def test_router_init_puts_all_mass_on_full_rung():
    m = _model(enable=True)
    ff = m.blocks["0"].feed_forward
    logits = ff._nffn_router(torch.randn(32, m.d_model))
    probs = torch.softmax(logits, dim=-1)
    assert probs[:, 0].min() > 1.0 - 1e-3
    assert ff._nffn_router.w.bias[0].item() == pytest.approx(FULL_RUNG_INIT_BIAS)


def test_only_router_and_gain_are_new_state_dict_keys():
    seed_all(0)
    m_moe = _model(enable=True)
    seed_all(0)
    m_plain = _model(enable=False)
    new = set(m_moe.state_dict()) - set(m_plain.state_dict())
    assert new, "expected new keys (they must be baked into the base checkpoint)"
    assert all("_nffn_router" in k or "_nffn_gain" in k for k in new), new
    # every pre-existing key is untouched
    assert set(m_plain.state_dict()) <= set(m_moe.state_dict())


# --------------------------------------------------------------------------------------
# routing mechanics
# --------------------------------------------------------------------------------------


def _force_rung(model, rung: int):
    """Bias the router so every token takes ``rung``."""
    for block in model.blocks.values():
        ff = block.feed_forward
        with torch.no_grad():
            ff._nffn_router.w.bias.zero_()
            ff._nffn_router.w.bias[rung] = FULL_RUNG_INIT_BIAS


def test_forced_rung_matches_manual_prefix_slice():
    """A routed FFN on rung g must equal the hand-computed prefix-sliced MLP (times the gain)."""
    seed_all(3)
    m = _model(enable=True, divisors=(1, 4))
    ff = m.blocks["0"].feed_forward
    _force_rung(m, 1)
    width = ff._nffn_widths[1]
    x = torch.randn(1, 6, m.d_model)
    with torch.no_grad():
        got = ff(x)
        h = ff.activation_fn(torch.nn.functional.linear(x, ff.w1.weight[:width], None)) * (
            torch.nn.functional.linear(x, ff.w3.weight[:width], None)
        )
        want = torch.nn.functional.linear(h, ff.w2.weight[:, :width], ff.w2.bias)
    torch.testing.assert_close(got, want)


def test_null_rung_outputs_zero():
    seed_all(4)
    m = _model(enable=True)
    ff = m.blocks["0"].feed_forward
    _force_rung(m, len(ff._nffn_widths) - 1)
    with torch.no_grad():
        out = ff(torch.randn(1, 6, m.d_model))
    assert torch.count_nonzero(out) == 0


def test_mixed_routing_is_per_token():
    """Tokens on different rungs must each get their own rung's output -- i.e. the gather/scatter
    really is per token, not per batch."""
    seed_all(5)
    m = _model(enable=True, divisors=(1, 4))
    ff = m.blocks["0"].feed_forward
    d = m.d_model
    # route on the sign of feature 0: positive -> full, negative -> the 1/4 rung
    with torch.no_grad():
        ff._nffn_router.w.bias.zero_()
        ff._nffn_router.w.weight.zero_()
        ff._nffn_router.w.weight[0, 0] = 10.0
        ff._nffn_router.w.weight[1, 0] = -10.0
    x = torch.randn(1, 8, d)
    x[0, ::2, 0] = 5.0
    x[0, 1::2, 0] = -5.0
    with torch.no_grad():
        out = ff(x)
        full = ff._nffn_orig_forward(x)
    torch.testing.assert_close(out[0, ::2], full[0, ::2])
    assert not torch.allclose(out[0, 1::2], full[0, 1::2])


def test_gain_scales_the_rung_output():
    seed_all(6)
    m = _model(enable=True, divisors=(1, 4))
    ff = m.blocks["0"].feed_forward
    _force_rung(m, 1)
    x = torch.randn(1, 4, m.d_model)
    with torch.no_grad():
        base = ff(x)
        ff._nffn_gain[1] = 2.0
        scaled = ff(x)
    torch.testing.assert_close(scaled, 2.0 * base)


def test_start_layer_leaves_earlier_blocks_dense():
    m = _model(enable=True, start_layer=1)
    assert not hasattr(m.blocks["0"].feed_forward, "_nffn_router")
    assert hasattr(m.blocks["1"].feed_forward, "_nffn_router")


def test_install_rejects_non_gated_ffn():
    m = _model(enable=False)
    holder = NestedFFNHolder([1.0, 0.0])
    del m.blocks["0"].feed_forward.w1
    with pytest.raises(ValueError, match="not a gated MLP"):
        install_nested_ffn_moe(m.blocks, holder, start_layer=0, widths=[8, 0], costs=[1.0, 0.0])


# --------------------------------------------------------------------------------------
# budget loss
# --------------------------------------------------------------------------------------


def test_budget_hinge_is_zero_below_target_and_positive_above():
    holder = NestedFFNHolder([1.0, 0.25, 0.0], target_cost=0.5, budget_weight=2.0)
    holder.begin_forward()
    holder.accumulate(
        exp_cost=torch.tensor(0.3),
        entropy=torch.tensor(0.0),
        hard_cost_sum=0.3,
        n_tokens=1,
        usage=[0, 1, 0],
    )
    assert holder.regularization_loss().item() == pytest.approx(0.0)

    holder.begin_forward()
    holder.accumulate(
        exp_cost=torch.tensor(0.9),
        entropy=torch.tensor(0.0),
        hard_cost_sum=0.9,
        n_tokens=1,
        usage=[1, 0, 0],
    )
    assert holder.regularization_loss().item() == pytest.approx(2.0 * 0.4)


def test_budget_loss_gradient_pushes_router_toward_cheap_rungs():
    """The end-to-end gradient check: with only the budget term, router mass must move off the
    full rung."""
    seed_all(7)
    m = _model(enable=True, target_cost=0.0, budget_weight=1.0)
    m.train()
    ff = m.blocks["0"].feed_forward
    holder = m._nested_ffn_moe["holder"]
    holder.begin_forward()
    ff(torch.randn(2, 8, m.d_model))
    loss = holder.regularization_loss()
    loss.backward()
    # bias gradient on the full rung is positive => a step downhill reduces its logit
    assert ff._nffn_router.w.bias.grad[0].item() > 0
    assert ff._nffn_router.w.bias.grad[-1].item() < 0  # null rung gets pushed up


def test_straight_through_gives_router_gradient_from_the_task_loss():
    """The forward value is unchanged by the ST coefficient (p/p.detach() == 1), but the router
    must still receive gradient from whatever consumes the FFN output."""
    seed_all(8)
    m = _model(enable=True, divisors=(1, 4))
    m.train()
    ff = m.blocks["0"].feed_forward
    holder = m._nested_ffn_moe["holder"]
    holder.begin_forward()
    out = ff(torch.randn(2, 8, m.d_model))
    out.square().mean().backward()
    assert ff._nffn_router.w.weight.grad is not None
    assert ff._nffn_router.w.weight.grad.abs().sum() > 0


def test_straight_through_gradient_is_bounded_for_improbable_rungs():
    """Regression: the ``p / p.detach()`` form of the ST coefficient has gradient ``1/p``, which
    explodes when exploration routes a token to a rung the router scores ~0 -- that NaN'd CE
    within 75 steps in the smoke test. The gradient must stay O(1) for any p."""
    seed_all(18)
    m = _model(enable=True, divisors=(1, 4))
    m.train()
    ff = m.blocks["0"].feed_forward
    # make rung 1 essentially impossible under the router, then force every token onto it
    with torch.no_grad():
        ff._nffn_router.w.bias[0] = 40.0  # p(rung 1) ~ 1e-18
    holder = m._nested_ffn_moe["holder"]
    holder.explore_prob = 1.0
    holder.explore_anneal_calls = 0
    holder.begin_forward()
    out = ff(torch.randn(2, 16, m.d_model))
    out.square().mean().backward()
    grad = ff._nffn_router.w.bias.grad
    assert torch.isfinite(grad).all(), grad
    assert grad.abs().max().item() < 1e3, grad


def test_routing_is_deterministic_within_a_forward():
    """Regression: activation checkpointing re-runs the block in backward. If the exploration draw
    came from the ambient RNG, the recompute would route a different number of tokens to each rung
    and torch aborts with `CheckpointError: Recomputed values ... have different metadata`
    (observed 5653 vs 5643 on a live 4B run). Re-running a routed forward WITHOUT an intervening
    begin_forward must reproduce the routing exactly."""
    seed_all(19)
    m = _model(enable=True, explore_prob=0.5, recon_frac=0.1, recon_weight=1.0)
    m.train()
    ff = m.blocks["0"].feed_forward
    holder = m._nested_ffn_moe["holder"]
    x = torch.randn(2, 64, m.d_model)

    holder.begin_forward()
    with torch.no_grad():
        first = ff(x)
        usage_first = list(holder._usage)
        recompute = ff(x)  # what AC does in backward: same call index, no begin_forward
    torch.testing.assert_close(first, recompute)
    # the recompute's own accumulation must match the forward's, rung for rung
    assert [b - a for a, b in zip(usage_first, holder._usage)] == usage_first

    # ...and a genuinely new forward must be free to route differently
    holder.begin_forward()
    with torch.no_grad():
        later = ff(x)
    assert not torch.allclose(first, later)


def test_exploration_actually_reaches_every_rung():
    """Exploration exists to give the narrow rungs gradient before the router prefers them; if it
    silently did nothing, small rungs would never train."""
    seed_all(20)
    m = _model(enable=True, explore_prob=1.0)
    m.train()
    ff = m.blocks["0"].feed_forward
    holder = m._nested_ffn_moe["holder"]
    holder.begin_forward()
    with torch.no_grad():
        ff(torch.randn(2, 256, m.d_model))
    assert all(c > 0 for c in holder._usage), holder._usage


def test_metrics_report_hard_cost_and_usage():
    seed_all(9)
    m = _model(enable=True)
    holder = m._nested_ffn_moe["holder"]
    holder.begin_forward()
    with torch.no_grad():
        m(torch.randint(5, 800, (1, 12)))
    metrics = holder.metrics()
    # untrained router => everything on the full rung => no savings yet
    assert metrics["ffn_moe/mean_cost"] == pytest.approx(1.0)
    assert metrics["ffn_moe/frac_rung0"] == pytest.approx(1.0)
    assert metrics["ffn_moe/speedup"] == pytest.approx(1.0)


def test_metrics_survive_an_accumulator_reset():
    """Regression: a trainer callback reads metrics in post_step, which is not guaranteed to land
    between the last routed forward and the next begin_forward. Reading the live accumulators
    alone reported NOTHING for 190 steps of a live 4B run while routing was working fine."""
    seed_all(21)
    m = _model(enable=True)
    holder = m._nested_ffn_moe["holder"]
    holder.begin_forward()
    with torch.no_grad():
        m(torch.randint(5, 800, (1, 12)))
    live = holder.metrics()
    holder.begin_forward()  # what the next forward does before the callback ever runs
    after = holder.metrics()
    assert after["ffn_moe/mean_cost"] == pytest.approx(live["ffn_moe/mean_cost"])
    assert after["ffn_moe/frac_rung0"] == pytest.approx(live["ffn_moe/frac_rung0"])


def test_metrics_always_report_schedule_state():
    """Even with no routed forward yet, the monitor must say so rather than return nothing --
    an empty dict makes an inert router look identical to a broken monitor."""
    holder = NestedFFNHolder([1.0, 0.0], target_cost=0.2)
    m = holder.metrics()
    assert m["ffn_moe/tokens"] == 0.0
    assert "ffn_moe/target" in m and "ffn_moe/calls" in m
    assert "ffn_moe/mean_cost" not in m


def test_target_and_explore_anneal():
    holder = NestedFFNHolder(
        [1.0, 0.0],
        target_cost=0.1,
        target_start=1.0,
        target_anneal_calls=10,
        explore_prob=0.5,
        explore_anneal_calls=10,
    )
    holder.begin_forward()  # call 1
    assert holder.current_target() == pytest.approx(1.0 + 0.1 * (0.1 - 1.0))
    assert holder.current_explore() == pytest.approx(0.5 * 0.9)
    for _ in range(19):
        holder.begin_forward()
    assert holder.current_target() == pytest.approx(0.1)
    assert holder.current_explore() == pytest.approx(0.0)


def test_set_calls_makes_the_schedule_a_function_of_the_step():
    """A crash-resume restarts the in-memory clock. Pinning it from the global step must give
    the same target/explore a never-interrupted run would have -- the first wave of routed 4B
    arms ended 3000 steps at target=0.84 because it did not."""
    kw = dict(
        target_cost=0.1,
        target_start=1.0,
        target_anneal_calls=100,
        explore_prob=0.5,
        explore_anneal_calls=100,
    )
    uninterrupted = NestedFFNHolder([1.0, 0.0], **kw)
    for _ in range(60):
        uninterrupted.begin_forward()
    resumed = NestedFFNHolder([1.0, 0.0], **kw)  # fresh process, clock at 0
    resumed.begin_forward()  # a stray forward before the callback's first pre_step
    resumed.set_calls(59)  # (global_step - 1) * calls_per_step, then the step's forward
    resumed.begin_forward()
    assert resumed.calls == uninterrupted.calls == 60
    assert resumed.current_target() == pytest.approx(uninterrupted.current_target())
    assert resumed.current_explore() == pytest.approx(uninterrupted.current_explore())
    assert resumed.current_target() < 0.5  # i.e. NOT restarted from 1.0


def test_cumulative_metrics_cover_every_forward():
    seed_all(12)
    m = _model(enable=True, start_layer=0)
    holder = m._nested_ffn_moe["holder"]
    n_layers = len(m.blocks)
    assert holder.cumulative_metrics() == {"ffn_moe/total_tokens": 0.0}
    with torch.no_grad():
        for _ in range(3):
            holder.begin_forward(collect_loss=False)
            m(torch.randint(5, 800, (2, 7)))
    cum = holder.cumulative_metrics()
    # every routed layer counts every token of every forward, not just the last one
    assert cum["ffn_moe/total_tokens"] == 3 * 2 * 7 * n_layers
    assert cum["ffn_moe/mean_cost"] == pytest.approx(1.0)  # init router: everything full
    assert cum["ffn_moe/frac_rung0"] == pytest.approx(1.0)
    last = holder.metrics()
    assert last["ffn_moe/tokens"] == 2 * 7 * n_layers  # the per-forward view is one forward


def test_mixed_rungs_match_a_per_token_reference():
    """The in-place gather/scatter path (and the single-rung fast path it bypasses) must give
    exactly what routing each token through its rung individually gives."""
    seed_all(13)
    m = _model(enable=True)
    ff = m.blocks["0"].feed_forward
    widths = ff._nffn_widths
    x = torch.randn(1, 16, m.d_model)
    # A router that spreads the 16 tokens over every rung.
    with torch.no_grad():
        ff._nffn_router.w.weight.normal_(std=3.0)
        ff._nffn_router.w.bias.zero_()
    holder = m._nested_ffn_moe["holder"]
    holder.begin_forward()
    with torch.no_grad():
        got = ff(x)
        choice = torch.softmax(ff._nffn_router(x[0]), dim=-1).argmax(-1)
    assert len(set(choice.tolist())) >= 3, "router did not spread the tokens"
    with torch.no_grad():
        ff._nffn_router.w.weight.zero_()  # _force_rung steers by bias alone
    for t in range(16):
        g = int(choice[t])
        _force_rung(m, g)
        holder.begin_forward()
        with torch.no_grad():
            ref = ff(x[:, t : t + 1])
        torch.testing.assert_close(got[:, t : t + 1], ref, rtol=1e-5, atol=1e-5)
        if widths[g] == 0:
            assert got[0, t].abs().sum() == 0


def _spread_router(ff, std=3.0):
    with torch.no_grad():
        ff._nffn_router.w.weight.normal_(std=std)
        ff._nffn_router.w.bias.zero_()


@pytest.mark.parametrize("explore", [0.0, 0.5])
def test_fused_ladder_matches_reference_autograd(monkeypatch, explore):
    """The single-node ladder (_NestedLadderFn) must give the same output AND the same gradients
    for x, w1/w2/w3, the gains and the router as plain autograd over weight slices."""
    seed_all(14)
    grads = {}
    outs = {}
    for fused in (True, False):
        monkeypatch.setattr(nffn, "USE_FUSED_LADDER", fused)
        seed_all(14)
        m = _model(enable=True, explore_prob=explore)
        m.train()
        ff = m.blocks["0"].feed_forward
        assert nffn._fused_ladder_ok(ff) == fused or not fused
        _spread_router(ff)
        holder = m._nested_ffn_moe["holder"]
        holder.begin_forward()
        x = torch.randn(2, 24, m.d_model, requires_grad=True)
        y = ff(x)
        (y.float().pow(2).sum()).backward()
        outs[fused] = y.detach()
        grads[fused] = {
            "x": x.grad.clone(),
            "w1": ff.w1.weight.grad.clone(),
            "w2": ff.w2.weight.grad.clone(),
            "w3": ff.w3.weight.grad.clone(),
            "gain": ff._nffn_gain.grad.clone(),
            "router_w": ff._nffn_router.w.weight.grad.clone(),
        }
        assert len(set(holder._last_usage)) > 0
    torch.testing.assert_close(outs[True], outs[False], rtol=1e-5, atol=1e-5)
    for k in grads[True]:
        torch.testing.assert_close(grads[True][k], grads[False][k], rtol=1e-4, atol=1e-5, msg=k)
    # and the gradient really is non-trivial in the weight prefixes (the narrow rungs were used)
    assert grads[True]["w1"].abs().sum() > 0 and grads[True]["gain"].abs().sum() > 0


def test_fused_ladder_all_null_gives_zero_weight_grads():
    seed_all(16)
    m = _model(enable=True)
    m.train()
    ff = m.blocks["0"].feed_forward
    _force_rung(m, len(ff._nffn_widths) - 1)
    m._nested_ffn_moe["holder"].begin_forward()
    x = torch.randn(1, 8, m.d_model, requires_grad=True)
    y = ff(x)
    assert y.abs().sum() == 0
    y.sum().backward()
    assert ff.w1.weight.grad.abs().sum() == 0
    assert x.grad.abs().sum() == 0


def test_layer_curriculum_opens_from_the_last_layer_down():
    holder = NestedFFNHolder([1.0, 0.0], start_layer=0, n_layers=4, layer_curriculum_calls=6)
    seen = []
    for _ in range(8):
        holder.begin_forward()
        seen.append(holder.current_min_layer())
    # monotone non-increasing, starts at the last layer, ends at start_layer
    assert seen == sorted(seen, reverse=True)
    assert seen[0] == 3 and seen[-1] == 0 and seen[-2] == 0
    # start_layer > 0 is respected as the floor
    h2 = NestedFFNHolder([1.0, 0.0], start_layer=2, n_layers=4, layer_curriculum_calls=4)
    for _ in range(10):
        h2.begin_forward()
    assert h2.current_min_layer() == 2
    # no curriculum: constant
    h3 = NestedFFNHolder([1.0, 0.0], start_layer=1, n_layers=4)
    assert h3.current_min_layer() == 1


def test_curriculum_gated_layer_is_dense_and_outside_the_budget():
    """A layer the curriculum has not opened runs the full FFN, is counted as full-rung usage
    in the cost report, but does not enter the budget mean or get a loss term."""
    seed_all(17)
    m = _model(enable=True, start_layer=0, layer_curriculum_calls=1000, target_cost=0.0)
    m.train()
    holder = m._nested_ffn_moe["holder"]
    n_layers = len(m.blocks)
    # force every router to the null rung: an OPEN layer would output 0 and cost 0
    _force_rung(m, len(m.blocks["0"].feed_forward._nffn_widths) - 1)
    holder.begin_forward()  # call 1: only the last layer is open
    assert holder.current_min_layer() == n_layers - 1
    x = torch.randn(1, 6, m.d_model)
    y0 = m.blocks["0"].feed_forward(x)
    y_last = m.blocks[str(n_layers - 1)].feed_forward(x)
    assert y0.abs().sum() > 0, "closed layer must run the dense FFN"
    assert y_last.abs().sum() == 0, "open layer follows its router (null)"
    per_layer = holder.per_layer_cost(last_forward=False)
    assert per_layer[0] == pytest.approx(1.0) and per_layer[n_layers - 1] == pytest.approx(0.0)
    # the budget mean only saw the open layer (null -> cost 0), so with target 0 the hinge is 0
    assert holder.metrics()["ffn_moe/mean_cost"] == pytest.approx(0.0)
    assert holder.metrics()["ffn_moe/tokens"] == 6  # one open layer x 6 tokens
    cum = holder.cumulative_metrics()
    assert cum["ffn_moe/total_tokens"] == 12  # both layers counted for the cost REPORT
    assert cum["ffn_moe/mean_cost"] == pytest.approx(0.5)


def test_per_layer_usage_is_tracked():
    seed_all(18)
    m = _model(enable=True, start_layer=0)
    holder = m._nested_ffn_moe["holder"]
    with torch.no_grad():
        holder.begin_forward(collect_loss=False)
        m(torch.randint(5, 800, (1, 9)))
    assert sorted(holder.usage_by_layer) == list(range(len(m.blocks)))
    assert all(sum(v) == 9 for v in holder.usage_by_layer.values())
    assert holder.per_layer_cost() == {k: pytest.approx(1.0) for k in holder.usage_by_layer}


def test_single_hidden_unit_rung():
    """width_multiple=1 allows a rung of ONE hidden unit (divisor == hidden size); it must run,
    cost 1/hidden, and match the manual one-column slice."""
    seed_all(19)
    m = _model(enable=False)
    hidden = m.blocks["0"].feed_forward.w1.out_features
    m.enable_nested_ffn_moe(start_layer=0, divisors=(1, hidden), width_multiple=1)
    ff = m.blocks["0"].feed_forward
    assert ff._nffn_widths == [hidden, 1, 0]
    assert ff._nffn_costs[1] == pytest.approx(1 / hidden)
    _force_rung(m, 1)
    m._nested_ffn_moe["holder"].begin_forward()
    x = torch.randn(1, 5, m.d_model)
    with torch.no_grad():
        y = ff(x)
        h = torch.nn.functional.silu(x @ ff.w1.weight[:1].t()) * (x @ ff.w3.weight[:1].t())
        ref = h @ ff.w2.weight[:, :1].t()
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-6)
    y2 = ff(x)
    y2.sum().backward()  # fused backward on a width-1 prefix
    assert ff.w1.weight.grad[1:].abs().sum() == 0 and ff.w1.weight.grad[:1].abs().sum() > 0


def test_recon_term_is_zero_when_every_token_is_on_the_full_rung():
    """With the init router (all tokens full), the chosen output IS the full output, so the
    reconstruction term must vanish -- it only bites once cheap rungs are used."""
    seed_all(10)
    m = _model(enable=True, recon_frac=1.0, recon_weight=1.0, target_cost=1.0)
    m.train()
    ff = m.blocks["0"].feed_forward
    holder = m._nested_ffn_moe["holder"]
    holder.begin_forward()
    ff(torch.randn(1, 8, m.d_model))
    assert holder.regularization_loss().item() == pytest.approx(0.0, abs=1e-6)

    _force_rung(m, len(ff._nffn_widths) - 1)  # null rung: output 0, target nonzero
    holder.begin_forward()
    ff(torch.randn(1, 8, m.d_model))
    assert holder.regularization_loss().item() > 0.5


def test_disabled_holder_falls_back_to_the_dense_ffn():
    seed_all(11)
    m = _model(enable=True)
    ff = m.blocks["0"].feed_forward
    _force_rung(m, len(ff._nffn_widths) - 1)  # would zero everything
    m._nested_ffn_moe["holder"].enabled = False
    x = torch.randn(1, 4, m.d_model)
    with torch.no_grad():
        torch.testing.assert_close(ff(x), ff._nffn_orig_forward(x))


# --------------------------------------------------------------------------------------
# end-to-end through the model
# --------------------------------------------------------------------------------------


def test_budget_term_reaches_the_model_loss():
    """The full path: model forward with labels must return CE + the budget hinge, and the
    difference from a target-1.0 run must be exactly the hinge value."""
    seed_all(15)
    m = _model(enable=True, target_cost=0.0, budget_weight=3.0)
    x = torch.randint(5, 800, (1, 12))
    labels = x.clone()
    out = m(x, labels=labels)
    holder = m._nested_ffn_moe["holder"]
    # untrained router: every token full => expected cost ~1 => hinge = 3.0 * (1 - 0) = 3.0
    assert holder.regularization_loss().item() == pytest.approx(3.0, rel=1e-3)
    assert out.loss.item() > out.ce_loss.item()
    assert (out.loss - out.ce_loss).item() == pytest.approx(3.0, rel=1e-2)


def test_no_budget_term_when_there_are_no_labels():
    """Generation-time forwards carry no loss, so nothing must be accumulated -- but routing (and
    therefore the usage metrics) still happens."""
    seed_all(16)
    m = _model(enable=True)
    with torch.no_grad():
        m(torch.randint(5, 800, (1, 12)))
    holder = m._nested_ffn_moe["holder"]
    assert holder.regularization_loss() is None
    assert holder.metrics()["ffn_moe/mean_cost"] == pytest.approx(1.0)


def test_backward_through_the_full_model_reaches_router_and_backbone():
    seed_all(17)
    m = _model(enable=True, target_cost=0.1, budget_weight=1.0)
    m.train()
    x = torch.randint(5, 800, (1, 12))
    m(x, labels=x.clone()).loss.backward()
    ff = m.blocks["0"].feed_forward
    assert ff._nffn_router.w.weight.grad.abs().sum() > 0
    assert ff.w1.weight.grad.abs().sum() > 0  # the backbone still trains


def test_dense_reference_run_has_no_router_state():
    m = _model(enable=False)
    assert m._nested_ffn_moe is None
    x = torch.randint(5, 800, (1, 8))
    m(x, labels=x.clone())  # must not raise


# --------------------------------------------------------------------------------------
# importance permutation
# --------------------------------------------------------------------------------------


def test_permutation_is_output_preserving():
    """Reordering hidden units is an exact reparameterization -- this is what makes it safe to
    apply to a trained checkpoint."""
    seed_all(12)
    m = _model(enable=False)
    ff = m.blocks["0"].feed_forward
    x = torch.randn(2, 5, m.d_model)
    with torch.no_grad():
        before = ff(x)
        apply_ffn_permutation(ff, ffn_importance_permutation(ff))
        after = ff(x)
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)


def test_permutation_puts_high_mass_units_first():
    seed_all(13)
    m = _model(enable=False)
    ff = m.blocks["0"].feed_forward
    hidden = ff.w1.out_features
    act_stats = torch.zeros(hidden)
    act_stats[hidden // 2] = 100.0  # one dominant unit, in the middle
    perm = ffn_importance_permutation(ff, act_stats)
    assert perm[0].item() == hidden // 2


def test_permutation_improves_the_prefix_slice_approximation():
    """The point of the permutation: after it, a narrow prefix explains more of the full FFN's
    output than an arbitrary prefix did."""
    seed_all(14)
    m = _model(enable=False)
    ff = m.blocks["0"].feed_forward
    hidden = ff.w1.out_features
    width = hidden // 8
    # make importance genuinely non-uniform so ordering can matter
    with torch.no_grad():
        ff.w2.weight.mul_(torch.rand(hidden).pow(3) + 0.01)
    x = torch.randn(64, m.d_model)

    def prefix_err():
        with torch.no_grad():
            full = ff(x)
            h = ff.activation_fn(
                torch.nn.functional.linear(x, ff.w1.weight[:width], None)
            ) * torch.nn.functional.linear(x, ff.w3.weight[:width], None)
            part = torch.nn.functional.linear(h, ff.w2.weight[:, :width], ff.w2.bias)
        return (full - part).pow(2).sum().item() / full.pow(2).sum().item()

    before = prefix_err()
    apply_ffn_permutation(ff, ffn_importance_permutation(ff))
    assert prefix_err() < before
