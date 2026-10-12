"""CPU tests for the *compressive* landmark decode math (no Triton kernel needed).

These validate the eager decode of :class:`FastCompressiveLandmarkAttention`:

* the defining difference from plain landmark attention -- a past block's landmark token now receives
  output weight (it is folded into the block's within-block softmax), instead of being zeroed;
* the compressive grouped softmax matches an independent brute-force reference, both with and without
  hard top-k landmark retrieval;
* with top-k retrieval, the non-selected blocks' landmark tokens collectively keep exactly
  ``nonselected_landmark_mass`` (alpha) of the attention mass (split by a softmax over their scores),
  their content tokens get zero, and the local section + selected blocks share the remaining
  ``1 - alpha``;
* under GQA, ``group_landmark_selection`` ("mean"/"max") makes every query head in a KV group agree on
  the same top-k landmark blocks, instead of each head retrieving independently (the default, ``None``).
"""

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.attention.landmark import repeat_kv
from olmo_core.nn.layer_norm import LayerNormConfig


def _build(*, mem_freq, head_dim, nonselected_landmark_mass=0.1, gate_temperature=False):
    attn = AttentionConfig(
        name=AttentionType.fast_compressive_landmark,
        n_heads=1,
        n_kv_heads=1,
        head_dim=head_dim,
        bias=False,
        mem_freq=mem_freq,
        nonselected_landmark_mass=nonselected_landmark_mass,
        gate_temperature=gate_temperature,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
    ).build(head_dim, layer_idx=0, n_layers=1, init_device="cpu")
    attn.eval()
    return attn


def _build_gqa(
    *,
    mem_freq,
    head_dim,
    n_heads,
    n_kv_heads,
    nonselected_landmark_mass=0.1,
    group_landmark_selection=None,
):
    attn = AttentionConfig(
        name=AttentionType.fast_compressive_landmark,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        bias=False,
        mem_freq=mem_freq,
        nonselected_landmark_mass=nonselected_landmark_mass,
        group_landmark_selection=group_landmark_selection,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
    ).build(head_dim, layer_idx=0, n_layers=1, init_device="cpu")
    attn.eval()
    return attn


def _decode_probs(attn, q, k, total):
    """Read off the per-key probability vector via one-hot values (value[m] = e_m)."""
    v = torch.eye(total).view(1, 1, total, total)
    with torch.no_grad():
        return attn._decode_one(q, k, v, total - 1).view(-1)


def _brute_compressive_probs(s, Lb, section_start, top_k, alpha):
    """Independent reference for the single-query compressive grouped softmax.

    :param s: 1D scaled-score vector over keys ``0..total-1`` (the last key is the query position).
    :param Lb: block size.
    :param section_start: start of the local section (a multiple of ``Lb``).
    """
    total = s.shape[0]
    S = section_start
    probs = torch.zeros(total, dtype=torch.float64)
    sd = s.double()

    lm_pos = [j for j in range(total) if j % Lb == Lb - 1 and j < S]
    local_pos = list(range(S, total))

    if top_k is not None and len(lm_pos) > top_k:
        ranked = sorted(lm_pos, key=lambda j: float(sd[j]), reverse=True)
        selected = set(ranked[:top_k])
        nonselected = [j for j in lm_pos if j not in selected]
        a = alpha
    else:
        selected = set(lm_pos)
        nonselected = []
        a = 0.0

    gate_keys = sorted(list(selected) + local_pos)
    gate_w = torch.softmax(torch.tensor([float(sd[j]) for j in gate_keys], dtype=torch.float64), 0)
    gate_map = {j: float(gate_w[i]) for i, j in enumerate(gate_keys)}

    for j in local_pos:
        probs[j] = gate_map[j]
    for lm in selected:
        b_start = (lm // Lb) * Lb
        block = list(range(b_start, b_start + Lb))  # full block incl landmark at `lm`
        wl = torch.softmax(torch.tensor([float(sd[j]) for j in block], dtype=torch.float64), 0)
        for idx, j in enumerate(block):
            probs[j] += gate_map[lm] * float(wl[idx])

    if nonselected:
        probs *= 1.0 - a
        ns_w = torch.softmax(
            torch.tensor([float(sd[j]) for j in nonselected], dtype=torch.float64), 0
        )
        for i, j in enumerate(nonselected):
            probs[j] += a * float(ns_w[i])

    return probs


def test_compressive_decode_landmark_token_gets_weight():
    # block_size = 16; landmark at j % 16 == 15. Query at 39 -> local block [32..39],
    # past landmark blocks [0..15] (landmark 15) and [16..31] (landmark 31).
    attn = _build(mem_freq=15, head_dim=40)
    total = 40
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)
    probs = _decode_probs(attn, q, k, total)

    assert torch.isfinite(probs).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    # The defining behavior: the past blocks' landmark tokens (15 and 31) DO receive output weight
    # (they are folded into their block's within-block softmax), unlike plain landmark attention.
    assert probs[15].abs() > 1e-6
    assert probs[31].abs() > 1e-6
    # The local block [32..39] is fully attended.
    assert all(probs[m].abs() > 1e-6 for m in range(32, 40))


def test_compressive_decode_matches_brute_reference_no_topk():
    attn = _build(mem_freq=15, head_dim=47)
    Lb = 16
    # query at 46 (not a landmark, so no self-key drop) -> local block [32..46]; past blocks
    # [0..15], [16..31] with landmarks 15, 31.
    total = 47
    section_start = (total - 1) // Lb * Lb  # = 32
    torch.manual_seed(1)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)
    probs = _decode_probs(attn, q, k, total).double()

    s = (q @ k.transpose(-1, -2)).view(-1) * attn.softmax_scale
    ref = _brute_compressive_probs(s, Lb, section_start, top_k=None, alpha=0.0)
    torch.testing.assert_close(probs, ref, rtol=1e-5, atol=1e-6)


def test_compressive_decode_topk_alpha_mass_split():
    alpha = 0.25
    attn = _build(mem_freq=15, head_dim=63, nonselected_landmark_mass=alpha)
    Lb = 16
    # query at 62 (not a landmark) -> local [48..62]; past blocks with landmarks {15, 31, 47}.
    total = 63
    section_start = (total - 1) // Lb * Lb  # = 48
    torch.manual_seed(2)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)

    # prompt_len == total means every key is a prompt position -> per-block decode path with top-k.
    attn.set_landmark_eval_decode(total, "extend_last_block", top_k=1)
    probs = _decode_probs(attn, q, k, total).double()
    attn.clear_landmark_eval_decode()

    assert abs(float(probs.sum()) - 1.0) < 1e-6

    lm_pos = [15, 31, 47]
    s = (q @ k.transpose(-1, -2)).view(-1) * attn.softmax_scale
    selected = max(lm_pos, key=lambda j: float(s[j]))
    nonselected = [j for j in lm_pos if j != selected]

    # Non-selected blocks contribute *only* their landmark token, and those landmarks collectively
    # hold exactly alpha of the mass; their content tokens get zero.
    ns_mass = float(sum(probs[j] for j in nonselected))
    assert abs(ns_mass - alpha) < 1e-6
    for lm in nonselected:
        b_start = (lm // Lb) * Lb
        for j in range(b_start, b_start + Lb - 1):  # content of a non-selected block
            assert probs[j].abs() < 1e-9
    # Everything else (local section + the selected block) holds the remaining (1 - alpha).
    rest = float(probs.sum()) - ns_mass
    assert abs(rest - (1.0 - alpha)) < 1e-6

    ref = _brute_compressive_probs(s, Lb, section_start, top_k=1, alpha=alpha)
    torch.testing.assert_close(probs, ref, rtol=1e-5, atol=1e-6)


def test_compressive_decode_topk_noop_when_few_blocks():
    # With only 2 past landmark blocks and top_k=5, nothing is non-selected -> alpha has no effect
    # and the result equals the plain (no-top-k) compressive decode.
    attn = _build(mem_freq=15, head_dim=48, nonselected_landmark_mass=0.3)
    total = 48
    torch.manual_seed(3)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)

    attn.set_landmark_eval_decode(total, "extend_last_block", top_k=5)
    probs_topk = _decode_probs(attn, q, k, total).double()
    attn.clear_landmark_eval_decode()
    probs_plain = _decode_probs(attn, q, k, total).double()

    torch.testing.assert_close(probs_topk, probs_plain, rtol=1e-6, atol=1e-7)


def test_compressive_eval_decode_generated_token():
    # Generated-token decode ("one long local block"): generated query reaches earlier prompt blocks
    # only through their (now value-contributing) landmarks.
    attn = _build(mem_freq=15, head_dim=40)
    total, P = 40, 32  # section_start (extend) = (32 // 16) * 16 = 32; generated query at 39
    torch.manual_seed(4)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)

    attn.set_landmark_eval_decode(P, "extend_last_block")  # prompt_len 32 < qpos 39 -> eval path
    probs = _decode_probs(attn, q, k, total).double()
    attn.clear_landmark_eval_decode()

    assert abs(float(probs.sum()) - 1.0) < 1e-5
    # Local block [32..39] fully attended; past landmarks 15 and 31 carry their blocks' mass and
    # themselves receive weight (compressive).
    assert all(probs[m].abs() > 1e-6 for m in range(32, 40))
    assert probs[15].abs() > 1e-6 and probs[31].abs() > 1e-6

    s = (q @ k.transpose(-1, -2)).view(-1) * attn.softmax_scale
    ref = _brute_compressive_probs(s, 16, P, top_k=None, alpha=0.0)
    torch.testing.assert_close(probs, ref, rtol=1e-5, atol=1e-6)


def test_compressive_decode_distinct_from_plain_landmark():
    # Same q/k/v fed to fast_landmark vs fast_compressive_landmark must differ: the compressive
    # variant additionally routes mass through the landmark tokens' values.
    head_dim, total = 40, 40
    common = dict(mem_freq=15, head_dim=head_dim)
    compressive = _build(**common)
    plain = AttentionConfig(
        name=AttentionType.fast_landmark,
        n_heads=1,
        n_kv_heads=1,
        head_dim=head_dim,
        bias=False,
        mem_freq=15,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
    ).build(head_dim, layer_idx=0, n_layers=1, init_device="cpu")
    plain.eval()

    torch.manual_seed(5)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)
    v = torch.randn(1, 1, total, total)
    with torch.no_grad():
        o_c = compressive._decode_one(q, k, v, total - 1)
        o_p = plain._decode_one(q, k, v, total - 1)
    assert not torch.allclose(o_c, o_p, atol=1e-5)


def test_group_landmark_scores_mean_and_max_aggregate_within_kv_group():
    # n_heads=4, n_kv_heads=2 -> group 0 = heads {0, 1}, group 1 = heads {2, 3} (contiguous, matching
    # `repeat_kv`'s layout: (B, n_kv_heads, T, D) -> (B, n_kv_heads * n_rep, T, D)).
    lm_scores = torch.tensor(
        [
            [[5.0, 5.0, 0.0]],  # head0 (group 0)
            [[5.0, 0.0, 6.0]],  # head1 (group 0) -- diverges from head0
            [[2.0, 2.0, 2.0]],  # head2 (group 1)
            [[2.0, 2.0, 2.0]],  # head3 (group 1) -- identical to head2
        ]
    ).unsqueeze(
        0
    )  # (1, 4, 1, 3)

    attn_mean = _build_gqa(
        mem_freq=15, head_dim=4, n_heads=4, n_kv_heads=2, group_landmark_selection="mean"
    )
    agg_mean = attn_mean._group_landmark_scores(lm_scores)
    assert agg_mean.shape == lm_scores.shape
    expected_g0_mean = torch.tensor([5.0, 2.5, 3.0])  # mean([5,5,0], [5,0,6])
    torch.testing.assert_close(agg_mean[0, 0, 0], expected_g0_mean)
    torch.testing.assert_close(agg_mean[0, 1, 0], expected_g0_mean)  # both heads see the same
    torch.testing.assert_close(agg_mean[0, 2, 0], torch.tensor([2.0, 2.0, 2.0]))
    torch.testing.assert_close(agg_mean[0, 3, 0], torch.tensor([2.0, 2.0, 2.0]))
    assert (
        int(agg_mean[0, 0, 0].argmax()) == 0
    )  # mean favors landmark 0 (head0 and head1 agree there)

    attn_max = _build_gqa(
        mem_freq=15, head_dim=4, n_heads=4, n_kv_heads=2, group_landmark_selection="max"
    )
    agg_max = attn_max._group_landmark_scores(lm_scores)
    expected_g0_max = torch.tensor([5.0, 5.0, 6.0])  # elementwise max([5,5,0], [5,0,6])
    torch.testing.assert_close(agg_max[0, 0, 0], expected_g0_max)
    torch.testing.assert_close(agg_max[0, 1, 0], expected_g0_max)
    # mean and max disagree on which landmark ranks highest for this group -- the whole point of
    # having two aggregation choices rather than one "obviously correct" one.
    assert int(agg_max[0, 0, 0].argmax()) == 2

    attn_none = _build_gqa(
        mem_freq=15, head_dim=4, n_heads=4, n_kv_heads=2, group_landmark_selection=None
    )
    assert attn_none._group_landmark_scores(lm_scores) is lm_scores  # off -> exact no-op

    # inverse_mean: negated group mean, so a subsequent topk keeps the LEAST-attended block. It must
    # be the exact ranking-reversal of "mean" within each group (argmax(inverse) == argmin(mean)).
    attn_inv = _build_gqa(
        mem_freq=15, head_dim=4, n_heads=4, n_kv_heads=2, group_landmark_selection="inverse_mean"
    )
    agg_inv = attn_inv._group_landmark_scores(lm_scores)
    torch.testing.assert_close(agg_inv[0, 0, 0], -expected_g0_mean)
    torch.testing.assert_close(agg_inv[0, 1, 0], -expected_g0_mean)
    # mean's top block is landmark 0 (score 5.0); inverse_mean's top block is landmark 1 (mean 2.5,
    # the lowest) -- the deliberate opposite pick.
    assert int(agg_inv[0, 0, 0].argmax()) == 1
    assert int(agg_inv[0, 0, 0].argmax()) == int(agg_mean[0, 0, 0].argmin())


def test_group_landmark_scores_noop_for_mha():
    # No GQA grouping to do (n_heads == n_kv_heads) -> always a no-op regardless of the setting, so
    # MHA models are bit-identical whether or not `group_landmark_selection` is configured.
    attn = _build_gqa(
        mem_freq=15, head_dim=4, n_heads=2, n_kv_heads=2, group_landmark_selection="mean"
    )
    lm_scores = torch.randn(1, 2, 1, 5)
    assert attn._group_landmark_scores(lm_scores) is lm_scores


def test_group_landmark_selection_forces_agreement_across_gqa_group():
    # Two query heads sharing one KV head (n_heads=2, n_kv_heads=1, n_rep=2). Past landmark blocks at
    # 15/31/47 (block_size=16); query at 62 with `nonselected_landmark_mass=0.0` so a non-selected
    # block's content *and* landmark get exactly zero weight -- "which block did this head keep"
    # reduces to "which block has nonzero mass".
    Lb, total = 16, 63

    def block_mass(probs: torch.Tensor, block_start: int) -> float:
        return float(probs[block_start : block_start + Lb].sum())

    # Shared (post-repeat_kv) K: landmark dot products (with q=1) would rank 47 > 31 > 15.
    k_kv = torch.zeros(1, 1, total, 1)
    k_kv[0, 0, 15, 0] = 1.0
    k_kv[0, 0, 31, 0] = 2.0
    k_kv[0, 0, 47, 0] = 3.0
    k = repeat_kv(k_kv, 2)  # (1, 2, total, 1), identical for both heads -- as a real KV-cache read
    v = torch.eye(total).view(1, 1, total, total).expand(1, 2, total, total).contiguous()

    # head0's query agrees with the raw K ranking (prefers landmark 47); head1's disagrees (a negative
    # query flips the ranking, preferring landmark 15 -- the *weakest* raw dot product).
    q = torch.zeros(1, 2, 1, 1)
    q[0, 0, 0, 0] = 1.0
    q[0, 1, 0, 0] = -0.9

    for mode in (None, "mean", "max"):
        attn = _build_gqa(
            mem_freq=15,
            head_dim=1,
            n_heads=2,
            n_kv_heads=1,
            nonselected_landmark_mass=0.0,
            group_landmark_selection=mode,
        )
        attn.set_landmark_eval_decode(total, "extend_last_block", top_k=1)
        with torch.no_grad():
            probs = attn._decode_one(q, k, v, total - 1)  # (1, 2, 1, total)
        attn.clear_landmark_eval_decode()

        head0, head1 = probs[0, 0, 0], probs[0, 1, 0]
        # head0 always keeps block 47 (its own top-1 choice, independent or not).
        assert block_mass(head0, 32) > 1e-6  # block [32:48) contains landmark 47
        assert block_mass(head0, 0) < 1e-9  # block [0:16) contains landmark 15

        if mode is None:
            # Independent per-head selection: head1 keeps its OWN top-1 (landmark 15), disagreeing
            # with head0 despite sharing a KV group -- the behavior being fixed.
            assert block_mass(head1, 0) > 1e-6
            assert block_mass(head1, 32) < 1e-9
        else:
            # Grouped selection: head1 is forced onto the group's shared choice (block 47), overriding
            # its own (weaker) preference for landmark 15.
            assert block_mass(head1, 32) > 1e-6
            assert block_mass(head1, 0) < 1e-9


def test_gate_temperature_default_off_decode_gate_scores_is_none():
    attn = _build(mem_freq=15, head_dim=40)
    assert attn.log_gate_temp is None
    q = torch.randn(1, 1, 1, 40)
    k = torch.randn(1, 1, 40, 40)
    assert attn._decode_gate_scores(q, k) is None


def test_gate_temperature_decode_gate_scores_matches_scaled_formula():
    """``_decode_gate_scores`` must equal ``q @ k^T * softmax_scale * exp(-log_gate_temp)`` --
    the same scaling applied to ``q_gate`` at train/prefill (see ``_attn_core``), so decode is
    consistent with training."""
    attn = _build(mem_freq=15, head_dim=40, gate_temperature=True)
    with torch.no_grad():
        attn.log_gate_temp.fill_(0.7)
    torch.manual_seed(2)
    q = torch.randn(1, 1, 1, 40)
    k = torch.randn(1, 1, 40, 40)
    gate_scores = attn._decode_gate_scores(q, k)
    expected = (
        torch.matmul(q, k.transpose(-1, -2)) * attn.softmax_scale * torch.exp(-attn.log_gate_temp)
    )
    torch.testing.assert_close(gate_scores, expected)


def test_gate_temperature_noop_at_init_matches_brute_reference():
    """``gate_temperature=True`` initializes ``log_gate_temp`` to 0 (temp=1) -- decode output must
    exactly match the existing no-temperature brute reference, i.e. this is a no-op at init."""
    attn = _build(mem_freq=15, head_dim=47, gate_temperature=True)
    Lb = 16
    total = 47
    section_start = (total - 1) // Lb * Lb  # = 32
    torch.manual_seed(1)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)
    probs = _decode_probs(attn, q, k, total).double()

    s = (q @ k.transpose(-1, -2)).view(-1) * attn.softmax_scale
    ref = _brute_compressive_probs(s, Lb, section_start, top_k=None, alpha=0.0)
    torch.testing.assert_close(probs, ref, rtol=1e-5, atol=1e-6)


def test_gate_temperature_changes_decode_output():
    """A nonzero ``log_gate_temp`` must change the decode attention distribution relative to
    temp=1 -- the knob actually does something end-to-end through ``_decode_one``."""
    attn = _build(mem_freq=15, head_dim=40, gate_temperature=True)
    total = 40
    torch.manual_seed(3)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)
    probs_temp1 = _decode_probs(attn, q, k, total).clone()

    with torch.no_grad():
        attn.log_gate_temp.fill_(-1.5)  # inv_temp = exp(1.5) > 1 -> sharper gate
    probs_sharper = _decode_probs(attn, q, k, total)

    assert not torch.allclose(probs_temp1, probs_sharper)


def test_gate_temperature_decode_gradient_flows_to_log_gate_temp():
    """Backprop through the eager decode path (``_decode_gate_scores`` -> ``_decode_one``) must
    populate ``log_gate_temp.grad`` -- the parameter is actually in the autograd graph, not just
    inert state."""
    attn = _build(mem_freq=15, head_dim=40, gate_temperature=True)
    with torch.no_grad():
        attn.log_gate_temp.fill_(0.3)
    total = 40
    torch.manual_seed(4)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)
    v = torch.randn(1, 1, total, 8)

    out = attn._decode_one(q, k, v, total - 1)
    out.sum().backward()

    assert attn.log_gate_temp.grad is not None
    assert attn.log_gate_temp.grad.abs().item() > 0


def test_gate_temperature_config_rejected_for_unsupported_types():
    """``gate_temperature`` is only supported for ``fast_compressive_landmark``; requesting it on
    other compressive variants must raise, not silently ignore the flag."""
    for name in (AttentionType.compressive_gqa_grouped, AttentionType.document_compressive_landmark):
        with pytest.raises(OLMoConfigurationError):
            AttentionConfig(
                name=name,
                n_heads=2,
                n_kv_heads=1,
                mem_freq=15,
                gate_temperature=True,
            ).build(16, layer_idx=0, n_layers=1, init_device="cpu")
