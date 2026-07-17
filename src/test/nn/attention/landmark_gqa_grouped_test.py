"""CPU tests for the *gate-only GQA-grouped* compressive landmark math (no Triton needed).

The GQA-grouped variant computes each past block's cross-block gate weight ``G_b`` from the MEAN of
the KV group's landmark scores for that block (equivalently -- since the landmark key is shared across
the group -- from the group-mean query dotted with the landmark key), while the within-block softmax
``f_n`` (which token inside a block) and the local/diagonal section stay per-head. These tests exercise
the eager primitive (``compressive_landmark_grouped_softmax`` with ``gate_logits=``) against an
independent brute-force reference, and pin the two defining properties:

* gate-only: within-block distribution is unchanged from per-head; only the block rescaling is grouped;
* n_rep==1 (MHA): the grouped gate is a no-op (gate_logits == x).
"""

import math

import torch

from olmo_core.nn.attention.landmark import (
    build_landmark_masks,
    compressive_landmark_grouped_softmax,
)


def _group_mean_gate_logits(x, block_size, n_kv_heads):
    """Replace each landmark COLUMN's per-head logit with the KV-group mean over the n_rep heads.

    ``x``: (B, H, T, T). Heads [g*n_rep, (g+1)*n_rep) share KV group g (repeat_kv layout). Returns a
    copy of ``x`` whose landmark-column entries are the group mean; non-landmark columns untouched.
    Landmark columns are derived from ``block_size`` (the block-end positions), independent of query.
    """
    B, H, T, _ = x.shape
    n_rep = H // n_kv_heads
    grouped = x.view(B, n_kv_heads, n_rep, T, T)
    gmean = grouped.mean(dim=2, keepdim=True).expand_as(grouped).reshape(B, H, T, T)
    is_mem_col = ((torch.arange(T, device=x.device) % block_size) == (block_size - 1)).view(
        1, 1, 1, T
    )
    return torch.where(is_mem_col, gmean, x)


def _brute_grouped_gate(x, gate_x, block_size):
    """Independent dense reference for the gate-only grouped compressive softmax (single doc, causal).

    gate ``G_b`` from ``gate_x`` landmark scores; within-block ``f_n`` and own-section from ``x``.
    Returns probabilities (B, H, T, T).
    """
    B, H, T, _ = x.shape
    Lb = block_size
    device = x.device
    neg = torch.finfo(x.dtype).min
    pos = torch.arange(T, device=device)
    sec = pos // Lb
    is_mem = (pos % Lb) == (Lb - 1)
    causal = pos[None, :] <= pos[:, None]
    same_block = sec[None, :] == sec[:, None]
    past_block = sec[None, :] < sec[:, None]
    kmem = is_mem[None, :]
    local_content = same_block & (~kmem) & causal  # own-section content (never own landmark)
    past_landmark = past_block & kmem
    gate_set = (local_content | past_landmark).view(1, 1, T, T)

    # gate softmax uses gate_x
    gw = torch.softmax(gate_x.masked_fill(~gate_set, neg), dim=-1)  # (B,H,T,T)
    # within-block softmax over each past block's tokens uses x
    within = torch.softmax(x[..., :].reshape(B, H, T, T // Lb, Lb), dim=-1).reshape(B, H, T, T)
    block_gate = gw[..., is_mem]  # (B,H,T,n_blocks)
    block_gate_full = block_gate.repeat_interleave(Lb, dim=-1)  # (B,H,T,T)
    past_mask = past_block.view(1, 1, T, T)
    local_mask = local_content.view(1, 1, T, T)
    final = torch.where(past_mask, block_gate_full * within, torch.zeros_like(within))
    final = torch.where(local_mask, gw, final)
    return final


def _run_eager(x, gate_logits, block_size, B, H, T):
    _, is_mem, lsm = build_landmark_masks(T, block_size, x.device, x.dtype)
    return compressive_landmark_grouped_softmax(
        x,
        dim=-1,
        is_mem=is_mem.expand(B, H, T, T),
        last_section_mask=lsm.expand(B, 1, T, T),
        gate_logits=gate_logits,
    )


def test_grouped_gate_matches_brute_reference():
    torch.manual_seed(0)
    B, H, n_kv, T, d = 2, 8, 2, 16, 8  # n_rep = 4
    block_size = 4
    q = torch.randn(B, H, T, d, dtype=torch.float64)
    k = torch.randn(B, H, T, d, dtype=torch.float64)
    scale = 1.0 / math.sqrt(d)

    attn_mask, is_mem, _ = build_landmark_masks(T, block_size, q.device, q.dtype)
    x = (q @ k.transpose(-1, -2)) * scale + attn_mask  # masked logits (own-landmark etc.)

    gate_logits = _group_mean_gate_logits(x, block_size, n_kv)
    probs = _run_eager(x, gate_logits, block_size, B, H, T)
    ref = _brute_grouped_gate(x, gate_logits, block_size)

    assert torch.isfinite(probs).all()
    torch.testing.assert_close(probs.sum(-1), probs.new_ones(B, H, T), atol=1e-5, rtol=0)
    torch.testing.assert_close(probs, ref.to(probs.dtype), atol=1e-6, rtol=1e-5)


def test_grouped_gate_shares_block_score_ratio_within_group():
    # The defining property. Only the block SCORE (pre-softmax landmark logit) is grouped; the final
    # gate WEIGHT G_b is still normalized per-head against that head's own-section (local) scores, which
    # differ across heads. So the block masses themselves are NOT equal across a group -- but their
    # RATIO is: mass_b / mass_b' = exp(score_b) / exp(score_b'), and the per-head normalizer cancels.
    # A grouped score => a group-shared log-mass DIFFERENCE between any two past blocks.
    torch.manual_seed(1)
    B, H, n_kv, T, d = 1, 6, 2, 16, 8  # n_rep = 3, 4 blocks -> query in block 3, past blocks 0,1,2
    block_size = 4
    q = torch.randn(B, H, T, d, dtype=torch.float64)
    k = torch.randn(B, H, T, d, dtype=torch.float64)
    scale = 1.0 / math.sqrt(d)
    attn_mask, is_mem, _ = build_landmark_masks(T, block_size, q.device, q.dtype)
    x = (q @ k.transpose(-1, -2)) * scale + attn_mask
    gate_logits = _group_mean_gate_logits(x, block_size, n_kv)
    probs = _run_eager(x, gate_logits, block_size, B, H, T)

    Lb = block_size
    n_rep = H // n_kv
    qrow = T - 1
    n_past = qrow // Lb  # blocks strictly before the query's own block
    per_block = probs[0, :, qrow, :].reshape(H, T // Lb, Lb).sum(-1)  # (H, n_blocks); mass_b == G_b
    past = per_block[:, :n_past]  # (H, n_past); all > 0 (every past block gated)
    log_diff = (
        past.log()[:, 1:] - past.log()[:, :1]
    )  # (H, n_past-1): log G_b - log G_0, group-shared
    for g in range(n_kv):
        grp = log_diff[g * n_rep : (g + 1) * n_rep]
        for i in range(1, n_rep):
            torch.testing.assert_close(grp[0], grp[i], atol=1e-6, rtol=1e-6)
    # sanity: the masses themselves are NOT all equal across the group (per-head normalizer differs),
    # otherwise this test would trivially pass for the wrong reason.
    assert (past[0] - past[1]).abs().max() > 1e-4


def test_grouped_gate_noop_when_gate_logits_none():
    # gate_logits=None must reproduce the plain per-head compressive softmax exactly.
    torch.manual_seed(2)
    B, H, T, d = 1, 4, 12, 8
    block_size = 4
    q = torch.randn(B, H, T, d, dtype=torch.float64)
    k = torch.randn(B, H, T, d, dtype=torch.float64)
    attn_mask, is_mem, lsm = build_landmark_masks(T, block_size, q.device, q.dtype)
    x = (q @ k.transpose(-1, -2)) * (d**-0.5) + attn_mask
    kw = dict(dim=-1, is_mem=is_mem.expand(B, H, T, T), last_section_mask=lsm.expand(B, 1, T, T))
    p_none = compressive_landmark_grouped_softmax(x, gate_logits=None, **kw)
    p_default = compressive_landmark_grouped_softmax(x, **kw)
    torch.testing.assert_close(p_none, p_default, atol=0, rtol=0)


def test_grouped_gate_mha_is_identity():
    # n_rep == 1 (H == n_kv): the group mean equals x at landmark columns, so gate_logits == x and the
    # grouped result is bit-identical to the per-head compressive softmax.
    torch.manual_seed(3)
    B, H, n_kv, T, d = 1, 4, 4, 12, 8  # n_rep = 1
    block_size = 4
    q = torch.randn(B, H, T, d, dtype=torch.float64)
    k = torch.randn(B, H, T, d, dtype=torch.float64)
    attn_mask, is_mem, lsm = build_landmark_masks(T, block_size, q.device, q.dtype)
    x = (q @ k.transpose(-1, -2)) * (d**-0.5) + attn_mask
    gate_logits = _group_mean_gate_logits(x, block_size, n_kv)
    torch.testing.assert_close(gate_logits, x, atol=0, rtol=0)  # n_rep=1 -> mean is identity
    kw = dict(dim=-1, is_mem=is_mem.expand(B, H, T, T), last_section_mask=lsm.expand(B, 1, T, T))
    p_grouped = compressive_landmark_grouped_softmax(x, gate_logits=gate_logits, **kw)
    p_plain = compressive_landmark_grouped_softmax(x, **kw)
    torch.testing.assert_close(p_grouped, p_plain, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# CompressiveGQAGroupedAttention module (eager path, CPU)
# ---------------------------------------------------------------------------

from olmo_core.nn.attention import AttentionConfig, AttentionType  # noqa: E402
from olmo_core.nn.attention.landmark_compressive_gqa import (  # noqa: E402
    CompressiveGQAGroupedAttention,
)
from olmo_core.nn.layer_norm import LayerNormConfig  # noqa: E402


def _build(*, n_heads, n_kv_heads, head_dim=16, mem_freq=15, dtype=torch.float64):
    d_model = n_heads * head_dim
    attn = AttentionConfig(
        name=AttentionType.compressive_gqa_grouped,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        bias=False,
        mem_freq=mem_freq,
        landmark_use_kernel=False,  # eager path (no triton on CPU)
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
    ).build(d_model, layer_idx=0, n_layers=1, init_device="cpu")
    attn = attn.to(dtype)
    attn.eval()
    return attn


def test_module_builds_and_is_grouped_type():
    attn = _build(n_heads=8, n_kv_heads=2)
    assert isinstance(attn, CompressiveGQAGroupedAttention)
    assert attn.use_kernel is False
    assert attn.n_heads == 8 and attn.n_kv_heads == 2


def test_module_forward_backward_finite():
    attn = _build(n_heads=8, n_kv_heads=2, mem_freq=15)
    B, T = 2, 32  # T multiple of block_size (=16)
    x = torch.randn(B, T, attn.n_heads * attn.head_dim, dtype=torch.float64, requires_grad=True)
    out = attn(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_attn_core_mha_matches_ungrouped_gate():
    # n_rep == 1: q_gate == q, so the grouped gate logits equal the per-head logits and the eager
    # grouped forward must equal the plain (gate_logits=None) compressive forward on the same q/k/v.
    attn = _build(n_heads=4, n_kv_heads=4, mem_freq=15)  # n_rep = 1
    B, H, T, D = 1, 4, 32, attn.head_dim
    q = torch.randn(B, H, T, D, dtype=torch.float64)
    k = torch.randn(B, H, T, D, dtype=torch.float64)
    v = torch.randn(B, H, T, D, dtype=torch.float64)
    with torch.no_grad():
        q_gate = attn._group_mean_query(q)
        torch.testing.assert_close(q_gate, q, atol=0, rtol=0)  # n_rep=1 -> identity
        grouped = attn._eager_forward(q, q_gate, k, v)
        # plain per-head compressive: gate_logits=None
        attn_mask, is_mem, lsm = build_landmark_masks(T, attn.block_size, q.device, q.dtype)
        xx = torch.matmul(q, k.transpose(-1, -2)) * attn.softmax_scale + attn_mask
        xx = torch.maximum(xx, torch.tensor(torch.finfo(xx.dtype).min))
        p = compressive_landmark_grouped_softmax(
            xx, dim=-1, is_mem=is_mem.expand(B, H, T, T), last_section_mask=lsm.expand(B, 1, T, T)
        )
        plain = torch.matmul(p.to(v.dtype), v)
    torch.testing.assert_close(grouped, plain, atol=1e-9, rtol=1e-8)


def test_attn_core_grouped_differs_from_per_head():
    # n_rep > 1: grouping the gate must actually change the output vs the per-head compressive gate.
    attn = _build(n_heads=6, n_kv_heads=2, mem_freq=15)  # n_rep = 3
    B, H, T, D = 1, 6, 48, attn.head_dim
    q = torch.randn(B, H, T, D, dtype=torch.float64)
    k = torch.randn(B, H, T, D, dtype=torch.float64)
    v = torch.randn(B, H, T, D, dtype=torch.float64)
    with torch.no_grad():
        q_gate = attn._group_mean_query(q)
        grouped = attn._eager_forward(q, q_gate, k, v)
        attn_mask, is_mem, lsm = build_landmark_masks(T, attn.block_size, q.device, q.dtype)
        xx = torch.matmul(q, k.transpose(-1, -2)) * attn.softmax_scale + attn_mask
        xx = torch.maximum(xx, torch.tensor(torch.finfo(xx.dtype).min))
        p = compressive_landmark_grouped_softmax(
            xx, dim=-1, is_mem=is_mem.expand(B, H, T, T), last_section_mask=lsm.expand(B, 1, T, T)
        )
        per_head = torch.matmul(p.to(v.dtype), v)
    assert not torch.allclose(grouped, per_head, atol=1e-4)


def test_gate_couples_gradient_across_group():
    # The defining training-time property: because a group's block gate uses the group-MEAN query, one
    # head's output depends on the OTHER heads' queries in its group -- through the shared gate. So
    # d(out[head_j]) / d(q[head_i]) is NONZERO for two heads i != j in the same group, whereas for the
    # plain per-head compressive gate it is exactly zero (heads are independent). This is what the
    # eager module must route (and what the fused kernel's dq_gate term must reproduce).
    attn = _build(n_heads=4, n_kv_heads=2, mem_freq=15)  # n_rep = 2; group0 = heads {0,1}
    B, H, T, D = 1, 4, 32, attn.head_dim
    torch.manual_seed(0)
    k = torch.randn(B, H, T, D, dtype=torch.float64)
    v = torch.randn(B, H, T, D, dtype=torch.float64)

    def out_head1_grad_wrt_q(use_group_gate: bool):
        q = torch.randn(B, H, T, D, dtype=torch.float64, requires_grad=True)
        torch.manual_seed(1)
        q.data.normal_()
        q_gate = attn._group_mean_query(q) if use_group_gate else q
        out = attn._eager_forward(q, q_gate, k, v)  # (B,H,T,D)
        out[0, 1].sum().backward()  # loss depends only on head 1's output
        return q.grad[0, 0].abs().max().item()  # sensitivity of head-1 output to head-0's query

    grouped_coupling = out_head1_grad_wrt_q(use_group_gate=True)
    perhead_coupling = out_head1_grad_wrt_q(
        use_group_gate=False
    )  # q_gate == q -> no cross-head path
    assert grouped_coupling > 1e-4, grouped_coupling  # shared gate couples heads 0 and 1
    assert perhead_coupling < 1e-12, perhead_coupling  # per-head gate: head 1 independent of head 0


# ---------------------------------------------------------------------------
# GPU: fused grouped kernel vs the eager grouped path (forward + backward parity)
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from olmo_core.nn.attention.landmark_kernel import has_landmark_kernel  # noqa: E402
from olmo_core.testing import requires_gpu  # noqa: E402


def _group_mean_query(q, n_kv_heads):
    B, H, T, D = q.shape
    n_rep = H // n_kv_heads
    if n_rep == 1:
        return q
    g = q.view(B, n_kv_heads, n_rep, T, D)
    return g.mean(dim=2, keepdim=True).expand(B, n_kv_heads, n_rep, T, D).reshape(B, H, T, D)


def _eager_grouped_out(q, q_gate, k, v, block_size, scale):
    """Eager grouped-gate compressive output (B,H,T,D) -- the parity reference for the fused kernel."""
    B, H, T, _ = q.shape
    attn_mask, is_mem, lsm = build_landmark_masks(T, block_size, q.device, q.dtype)
    x = torch.matmul(q, k.transpose(-1, -2)) * scale + attn_mask
    x = torch.maximum(x, torch.tensor(torch.finfo(x.dtype).min, device=x.device, dtype=x.dtype))
    x_gate = torch.matmul(q_gate, k.transpose(-1, -2)) * scale + attn_mask
    is_mem_col = ((torch.arange(T, device=q.device) % block_size) == (block_size - 1)).view(
        1, 1, 1, T
    )
    gate_logits = torch.where(is_mem_col, x_gate, x)
    probs = compressive_landmark_grouped_softmax(
        x,
        dim=-1,
        is_mem=is_mem.expand(B, H, T, T),
        last_section_mask=lsm.expand(B, 1, T, T),
        gate_logits=gate_logits,
    ).to(q.dtype)
    return torch.matmul(probs, v)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize("n_heads, n_kv_heads", [(8, 2), (4, 4)])  # GQA (n_rep=4) and MHA (n_rep=1)
@pytest.mark.parametrize("head_dim, mem_freq", [(64, 15), (128, 63)])
def test_fused_grouped_matches_eager_fwd_bwd(n_heads, n_kv_heads, head_dim, mem_freq):
    from olmo_core.nn.attention.landmark_compressive_gqa_kernel import (
        fused_compressive_gqa_grouped_attention,
    )

    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, T = 2, block_size * 4
    scale = head_dim**-0.5
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)

    def mk():
        g = torch.randn(B, n_heads, T, head_dim, device="cuda", dtype=torch.float32) * 0.5
        return g.clone().requires_grad_(True)

    q_e, k_e, v_e = mk(), mk(), mk()
    q_f = q_e.detach().clone().requires_grad_(True)
    k_f = k_e.detach().clone().requires_grad_(True)
    v_f = v_e.detach().clone().requires_grad_(True)

    out_e = _eager_grouped_out(q_e, _group_mean_query(q_e, n_kv_heads), k_e, v_e, block_size, scale)
    out_f = fused_compressive_gqa_grouped_attention(
        q_f,
        _group_mean_query(q_f, n_kv_heads),
        k_f,
        v_f,
        is_mem,
        sm_scale=scale,
        block_size=block_size,
    )
    torch.testing.assert_close(out_f, out_e, rtol=2e-3, atol=2e-3)

    g = torch.randn_like(out_e)
    out_e.backward(g)
    out_f.backward(g)
    # q.grad includes the gate gradient distributed back through group_mean -> validates dq_gate too.
    torch.testing.assert_close(q_f.grad, q_e.grad, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(k_f.grad, k_e.grad, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(v_f.grad, v_e.grad, rtol=3e-3, atol=3e-3)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
def test_fused_grouped_mha_matches_plain_compressive_kernel():
    # n_rep == 1: q_gate == q, so the grouped kernel must equal the plain compressive kernel exactly.
    from olmo_core.nn.attention.landmark_compressive import fused_compressive_landmark_attention
    from olmo_core.nn.attention.landmark_compressive_gqa_kernel import (
        fused_compressive_gqa_grouped_attention,
    )

    torch.manual_seed(1)
    head_dim, mem_freq = 64, 15
    block_size = mem_freq + 1
    B, H, T = 2, 4, block_size * 4
    scale = head_dim**-0.5
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)
    q = torch.randn(B, H, T, head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, T, head_dim, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, T, head_dim, device="cuda", dtype=torch.float32)

    out_grouped = fused_compressive_gqa_grouped_attention(
        q, q, k, v, is_mem, sm_scale=scale, block_size=block_size  # q_gate == q
    )
    out_plain = fused_compressive_landmark_attention(
        q, k, v, is_mem, sm_scale=scale, block_size=block_size
    )
    torch.testing.assert_close(out_grouped, out_plain, rtol=1e-4, atol=1e-4)
