"""CPU tests for the *compressive* landmark decode math (no Triton kernel needed).

These validate the eager decode of :class:`FastCompressiveLandmarkAttention`:

* the defining difference from plain landmark attention -- a past block's landmark token now receives
  output weight (it is folded into the block's within-block softmax), instead of being zeroed;
* the compressive grouped softmax matches an independent brute-force reference, both with and without
  hard top-k landmark retrieval;
* with top-k retrieval, the non-selected blocks' landmark tokens collectively keep exactly
  ``nonselected_landmark_mass`` (alpha) of the attention mass (split by a softmax over their scores),
  their content tokens get zero, and the local section + selected blocks share the remaining
  ``1 - alpha``.
"""

import torch

from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.layer_norm import LayerNormConfig


def _build(*, mem_freq, head_dim, nonselected_landmark_mass=0.1):
    attn = AttentionConfig(
        name=AttentionType.fast_compressive_landmark,
        n_heads=1,
        n_kv_heads=1,
        head_dim=head_dim,
        bias=False,
        mem_freq=mem_freq,
        nonselected_landmark_mass=nonselected_landmark_mass,
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
