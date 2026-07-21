"""CPU tests for the inference-only *flat-softmax* landmark decode ablation
(``GenerationConfig.landmark_flat_softmax`` / ``OLMO_LANDMARK_FLAT_SOFTMAX`` ->
``attn._eval_flat_softmax``).

The variant keeps the hard top-k block selection unchanged but replaces the landmark grouped/gated
softmax at decode time with a plain (flat) softmax over exactly the value-carrying support of the
gated scheme:

* **non-compressive** (:class:`FastLandmarkAttention`): selected blocks' *content* + local section;
  every landmark position carries no value, so it stays at exactly zero weight;
* **compressive** (:class:`FastCompressiveLandmarkAttention`): selected blocks' content + their
  landmark tokens + local section; non-selected blocks (and the ``nonselected_landmark_mass`` alpha
  reserved for their landmarks) are excluded entirely.

These exercise the eager decode math only (no Triton kernel), so they run on CPU. See
``analysis/flat_softmax_variant_eval.md``.
"""

import torch

from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.layer_norm import LayerNormConfig


def _build(name, *, mem_freq, head_dim, nonselected_landmark_mass=None):
    kwargs = dict(
        name=name,
        n_heads=1,
        n_kv_heads=1,
        head_dim=head_dim,
        bias=False,
        mem_freq=mem_freq,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
    )
    if nonselected_landmark_mass is not None:
        kwargs["nonselected_landmark_mass"] = nonselected_landmark_mass
    attn = AttentionConfig(**kwargs).build(head_dim, layer_idx=0, n_layers=1, init_device="cpu")
    attn.eval()
    return attn


def _decode_probs(attn, q, k, total):
    """Read off the per-key probability vector via one-hot values (value[m] = e_m)."""
    v = torch.eye(total).view(1, 1, total, total)
    with torch.no_grad():
        return attn._decode_one(q, k, v, total - 1).view(-1)


def _support(probs, tol=1e-6):
    return sorted(m for m in range(probs.numel()) if float(probs[m].abs()) > tol)


def _flat_ref(s, Lb, section_start, top_k, *, include_landmark):
    """Independent reference: a single flat softmax over the selected support.

    :param s: 1D scaled-score vector over keys ``0..total-1`` (last key is the query position).
    :param include_landmark: whether a selected block's landmark token is part of the support
        (True for compressive, False for plain landmark where landmarks carry no value).
    """
    total = s.shape[0]
    S = section_start
    sd = s.double()
    lm_pos = [j for j in range(total) if j % Lb == Lb - 1 and j < S]
    local_pos = list(range(S, total))

    if top_k is not None and len(lm_pos) > top_k:
        selected = set(sorted(lm_pos, key=lambda j: float(sd[j]), reverse=True)[:top_k])
    else:
        selected = set(lm_pos)

    support = set(local_pos)
    for lm in selected:
        b_start = (lm // Lb) * Lb
        for j in range(b_start, min(b_start + Lb, total)):
            if j == lm and not include_landmark:
                continue
            support.add(j)

    support = sorted(support)
    w = torch.softmax(torch.tensor([float(sd[j]) for j in support], dtype=torch.float64), 0)
    probs = torch.zeros(total, dtype=torch.float64)
    for i, j in enumerate(support):
        probs[j] = float(w[i])
    return probs


def test_fast_flat_matches_reference_and_flag_off_unchanged():
    attn = _build(AttentionType.fast_landmark, mem_freq=15, head_dim=53)
    Lb = 16
    # prompt_len == total forces the per-block decode path (with top-k). Past landmark blocks
    # {15, 31, 47}; local section = block containing the query.
    total = 53
    section_start = (total - 1) // Lb * Lb  # = 48
    landmarks = [15, 31, 47]
    top_k = 1

    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)
    s = (q @ k.transpose(-1, -2)).view(-1) * attn.softmax_scale

    # Flag OFF: the gated top-k decode is unchanged (its support is retrieved content + local).
    attn.set_landmark_eval_decode(total, "extend_last_block", top_k=top_k)
    gated = _decode_probs(attn, q, k, total).double()
    best = max(landmarks, key=lambda j: float(s[j]))
    assert _support(gated) == sorted(
        list(range(best - Lb + 1, best)) + list(range(section_start, total))
    )

    # Flag ON: flat softmax over exactly {selected block content + local section}.
    attn._eval_flat_softmax = True
    flat = _decode_probs(attn, q, k, total).double()
    ref = _flat_ref(s, Lb, section_start, top_k, include_landmark=False)
    torch.testing.assert_close(flat, ref, rtol=1e-5, atol=1e-6)

    assert abs(float(flat.sum()) - 1.0) < 1e-6
    # Non-selected blocks' content is hard-zeroed, and ALL landmark positions carry zero value.
    for lm in landmarks:
        assert float(flat[lm].abs()) < 1e-9  # landmark tokens never carry value here
        if lm != best:
            for j in range(lm - Lb + 1, lm):  # non-selected block content
                assert float(flat[j].abs()) < 1e-9
    # The flat weighting genuinely differs from the gated one on the shared support.
    assert not torch.allclose(flat, gated, atol=1e-4)

    # clear() resets the flag along with the rest of the eval-decode state.
    attn.clear_landmark_eval_decode()
    assert attn._eval_flat_softmax is False


def test_compressive_flat_matches_reference_and_flag_off_unchanged():
    alpha = 0.25
    attn = _build(
        AttentionType.fast_compressive_landmark,
        mem_freq=15,
        head_dim=63,
        nonselected_landmark_mass=alpha,
    )
    Lb = 16
    total = 63
    section_start = (total - 1) // Lb * Lb  # = 48
    landmarks = [15, 31, 47]
    top_k = 1

    torch.manual_seed(2)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)
    s = (q @ k.transpose(-1, -2)).view(-1) * attn.softmax_scale
    best = max(landmarks, key=lambda j: float(s[j]))

    # Flag OFF: gated compressive decode reserves alpha mass on non-selected landmarks (unchanged).
    attn.set_landmark_eval_decode(total, "extend_last_block", top_k=top_k)
    gated = _decode_probs(attn, q, k, total).double()
    ns_mass = float(sum(gated[j] for j in landmarks if j != best))
    assert abs(ns_mass - alpha) < 1e-6  # gated path still reserves alpha for non-selected landmarks

    # Flag ON: flat softmax over {selected block content + selected landmark + local section};
    # non-selected blocks (and their reserved alpha) are excluded entirely.
    attn._eval_flat_softmax = True
    flat = _decode_probs(attn, q, k, total).double()
    ref = _flat_ref(s, Lb, section_start, top_k, include_landmark=True)
    torch.testing.assert_close(flat, ref, rtol=1e-5, atol=1e-6)

    assert abs(float(flat.sum()) - 1.0) < 1e-6
    # Selected block's landmark token IS in the support (compressive) and carries weight.
    assert float(flat[best].abs()) > 1e-6
    # Non-selected blocks are fully excluded: content AND landmark get exactly zero (no alpha).
    for lm in landmarks:
        if lm != best:
            for j in range(lm - Lb + 1, lm + 1):  # whole non-selected block incl its landmark
                assert float(flat[j].abs()) < 1e-9
    assert not torch.allclose(flat, gated, atol=1e-4)

    attn.clear_landmark_eval_decode()
    assert attn._eval_flat_softmax is False


def test_compressive_flat_generated_token_path():
    # The "one long local block" generated-token decode path (qpos >= prompt_len) also honors the
    # flag. Here every past block's landmark is present; with top-k culling only selected blocks
    # (content + landmark) plus the local section survive under the flat softmax.
    attn = _build(AttentionType.fast_compressive_landmark, mem_freq=15, head_dim=40)
    Lb = 16
    total, P = 40, 32  # section_start (extend_last_block) = 32; generated query at 39
    landmarks = [15, 31]

    torch.manual_seed(4)
    q = torch.randn(1, 1, 1, total)
    k = torch.randn(1, 1, total, total)
    s = (q @ k.transpose(-1, -2)).view(-1) * attn.softmax_scale

    attn.set_landmark_eval_decode(P, "extend_last_block", top_k=1)
    attn._eval_flat_softmax = True
    flat = _decode_probs(attn, q, k, total).double()
    attn.clear_landmark_eval_decode()

    ref = _flat_ref(s, Lb, P, top_k=1, include_landmark=True)
    torch.testing.assert_close(flat, ref, rtol=1e-5, atol=1e-6)
    assert abs(float(flat.sum()) - 1.0) < 1e-6
    best = max(landmarks, key=lambda j: float(s[j]))
    for lm in landmarks:
        if lm != best:
            for j in range(lm - Lb + 1, lm + 1):
                assert float(flat[j].abs()) < 1e-9
