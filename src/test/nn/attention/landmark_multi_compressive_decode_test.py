"""CPU tests for the multi-landmark compressive *decode* math (no Triton kernel needed).

Validate :meth:`MultiCompressiveLandmarkAttention._compressive_decode_probs`:

* ``num_landmarks == 1`` reproduces the single-landmark
  :meth:`FastCompressiveLandmarkAttention._compressive_decode_probs` exactly (with and without top-k);
* for several landmarks per block it matches an independent brute-force reference (gate = mean/max pool
  over each block's landmark scores; within-block softmax over content + landmarks; ``top_k`` reserves
  ``nonselected_landmark_mass`` for the non-selected blocks' landmark tokens);
* the implied per-key probabilities are valid (non-negative, sum to 1).
"""

import pytest
import torch

from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.layer_norm import LayerNormConfig


def _build(name, *, mem_freq, num_landmarks=1, landmark_gate_pool="mean", head_dim=32):
    kwargs = dict(
        name=name,
        n_heads=1,
        n_kv_heads=1,
        head_dim=head_dim,
        bias=False,
        mem_freq=mem_freq,
        nonselected_landmark_mass=0.1,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
    )
    if name == AttentionType.multi_compressive_landmark:
        kwargs["num_landmarks"] = num_landmarks
        kwargs["landmark_gate_pool"] = landmark_gate_pool
    attn = AttentionConfig(**kwargs).build(head_dim, layer_idx=0, n_layers=1, init_device="cpu")
    attn.eval()
    return attn


def _brute_multi_decode(s, Lb, Lm, agg, section_start, top_k, alpha):
    """Independent single-query reference over a 1D scaled-score vector ``s`` (keys 0..total-1)."""
    total = s.shape[0]
    S = section_start
    n_blocks = S // Lb
    dtype = s.dtype

    g = (
        torch.stack(
            [
                (
                    s[b * Lb + Lb - Lm : b * Lb + Lb].mean()
                    if agg == "mean"
                    else s[b * Lb + Lb - Lm : b * Lb + Lb].max()
                )
                for b in range(n_blocks)
            ]
        )
        if n_blocks > 0
        else torch.zeros(0, dtype=dtype)
    )

    if top_k is not None and n_blocks > top_k:
        keep = set(torch.topk(g, top_k).indices.tolist())
        has_ns = True
        alpha_eff = alpha
    else:
        keep = set(range(n_blocks))
        has_ns = False
        alpha_eff = 0.0

    local_idx = list(range(S, total))
    gate_logits, gate_keys = [], []
    for b in sorted(keep):
        gate_logits.append(g[b])
        gate_keys.append(("rep", b))
    for j in local_idx:
        gate_logits.append(s[j])
        gate_keys.append(("local", j))
    gate_w = torch.softmax(torch.stack(gate_logits), 0)
    gate_map = {key: w for w, key in zip(gate_w, gate_keys)}

    probs = torch.zeros(total, dtype=dtype)
    for j in local_idx:
        probs[j] = gate_map[("local", j)]
    for b in sorted(keep):
        within = torch.softmax(s[b * Lb : (b + 1) * Lb], 0)
        probs[b * Lb : (b + 1) * Lb] = gate_map[("rep", b)] * within

    probs = probs * (1.0 - alpha_eff)
    if has_ns:
        ns_pos = [
            p
            for b in range(n_blocks)
            if b not in keep
            for p in range(b * Lb + Lb - Lm, b * Lb + Lb)
        ]
        if ns_pos:
            ns_logits = torch.full((total,), torch.finfo(dtype).min, dtype=dtype)
            ns_logits[ns_pos] = s[ns_pos]
            probs = probs + alpha_eff * torch.softmax(ns_logits, 0)
    return probs


def _decode_probs(attn, scores, section_start, top_k=None):
    total = scores.shape[-1]
    Lb = attn.block_size
    j = torch.arange(total)
    is_mem = (j % Lb) == (Lb - 1)  # single-landmark mask the caller would build (ignored by multi)
    last_section = j >= section_start
    attn._eval_top_k = top_k
    with torch.no_grad():
        return attn._compressive_decode_probs(scores, is_mem, last_section, section_start)


@pytest.mark.parametrize("top_k", [None, 2])
def test_decode_reduces_to_single_landmark_cpu(top_k):
    """num_landmarks == 1 decode must equal the single-landmark compressive decode."""
    torch.manual_seed(0)
    mem_freq = 31  # block_size 32
    Lb = 32
    total = Lb * 4 + 5  # 4 whole past blocks + a local section
    section_start = Lb * 4
    scores = torch.randn(1, 1, 1, total, dtype=torch.float64)

    multi = _build(AttentionType.multi_compressive_landmark, mem_freq=mem_freq, num_landmarks=1)
    single = _build(AttentionType.fast_compressive_landmark, mem_freq=mem_freq)
    for pool in ("mean", "max"):
        multi.landmark_gate_pool = pool
        multi._agg = 0 if pool == "mean" else 1
        pm = _decode_probs(multi, scores, section_start, top_k=top_k)
        ps = _decode_probs(single, scores, section_start, top_k=top_k)
        torch.testing.assert_close(pm, ps, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("agg", ["mean", "max"])
@pytest.mark.parametrize("top_k", [None, 2])
@pytest.mark.parametrize("num_landmarks", [2, 4])
def test_decode_matches_brute_cpu(num_landmarks, top_k, agg):
    torch.manual_seed(1)
    mem_freq = 32 - num_landmarks  # block_size 32
    Lb = 32
    total = Lb * 5 + 7
    section_start = Lb * 5
    scores = torch.randn(1, 1, 1, total, dtype=torch.float64)

    attn = _build(
        AttentionType.multi_compressive_landmark,
        mem_freq=mem_freq,
        num_landmarks=num_landmarks,
        landmark_gate_pool=agg,
    )
    probs = _decode_probs(attn, scores, section_start, top_k=top_k)[0, 0, 0]
    brute = _brute_multi_decode(
        scores[0, 0, 0], Lb, num_landmarks, agg, section_start, top_k, alpha=0.1
    )
    torch.testing.assert_close(probs, brute, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(probs.sum(), torch.ones((), dtype=torch.float64), rtol=0, atol=1e-9)
