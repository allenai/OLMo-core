"""
Tests for :mod:`olmo_core.nn.attention.summary_token`.

The load-bearing one is :func:`test_block_mask_implies_exact_token_mask`. The analytic block-mask
builder deliberately computes a *conservative superset* of the true block set, which is only safe
because ``BlockMask.from_kv_blocks`` re-applies ``mask_mod`` inside every **partial** block -- but
blocks it marks **full** skip the predicate entirely, so a block wrongly marked full silently changes
numerics. That test reconstructs exactly what the kernel would compute (full blocks unconditionally,
partial blocks through ``mask_mod``) and compares it against the dense ground truth.

:func:`test_build_is_cheap_at_256k` is the regression test the original code lacked: the stock
``create_block_mask`` path materializes every ``mask_mod`` intermediate at full ``(B, H, T, T)``,
which measures ~3 GiB / 4 s at ``T=16384`` and extrapolates to roughly 760 GiB / 17 minutes at
``T=262144`` -- per microbatch forward.
"""

import resource
import time

import pytest
import torch
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.attention.kv_cache import KVCacheManager
from olmo_core.nn.attention.summary_mask import (
    SummaryMaskSpec,
    build_summary_mask_mod,
    build_summary_roles,
    summary_mask_allowed,
)
from olmo_core.nn.attention.summary_token import SummaryTokenAttention, build_summary_block_mask

DOC_START, DOC_END, SUMM, EOS, PAD = 900, 901, 902, 903, 904
BLOCK = 128

IDS_KW = dict(
    doc_start_id=DOC_START, doc_end_id=DOC_END, summary_token_id=SUMM, eos_id=EOS, pad_id=PAD
)

SPECS = [
    SummaryMaskSpec(n_summary_tokens=5),
    SummaryMaskSpec(n_summary_tokens=5, summary_visible_tokens=2),
    SummaryMaskSpec(n_summary_tokens=5, summary_visible_tokens=0),
    SummaryMaskSpec(n_summary_tokens=5, summaries_read_own_document=False),
    SummaryMaskSpec(n_summary_tokens=5, summaries_read_earlier_summaries=False),
    SummaryMaskSpec(n_summary_tokens=5, query_reads_documents=True),
]


def _example(seq_len: int, doc_len: int = 90, n_summary: int = 5) -> torch.Tensor:
    """``[instruction][<doc><summ>]*[<query>][eos][pad*]`` padded to exactly ``seq_len``."""
    ids = [10] * 17
    doc = 0
    while len(ids) + doc_len + n_summary + 32 < seq_len:
        ids += [DOC_START] + [20 + (doc % 5)] * doc_len + [DOC_END] + [SUMM] * n_summary
        doc += 1
    ids += [DOC_START] + [50] * 12 + [DOC_END] + [EOS]
    ids += [PAD] * (seq_len - len(ids))
    return torch.tensor([ids[:seq_len]])


def _roles(seq_len: int, **kw) -> torch.Tensor:
    return build_summary_roles(_example(seq_len, **kw), **IDS_KW)


def _token_mask_implied_by(block_mask, mask_mod, seq_len: int, block: int = BLOCK) -> torch.Tensor:
    """Reconstruct exactly what the kernel computes from a :class:`BlockMask`.

    Full blocks are taken unconditionally (the predicate is skipped there); partial blocks are
    evaluated through ``mask_mod``; everything else is masked out.
    """
    n_blocks = seq_len // block
    out = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    kv_num, kv_idx = block_mask.kv_num_blocks[0, 0], block_mask.kv_indices[0, 0]
    full_num, full_idx = block_mask.full_kv_num_blocks[0, 0], block_mask.full_kv_indices[0, 0]
    q_off = torch.arange(block).view(-1, 1).expand(block, block)
    k_off = torch.arange(block).view(1, -1).expand(block, block)
    zeros = torch.zeros(block, block, dtype=torch.long)
    for i in range(n_blocks):
        q0 = i * block
        for j in full_idx[i, : full_num[i]].tolist():
            out[q0 : q0 + block, j * block : (j + 1) * block] = True
        for j in kv_idx[i, : kv_num[i]].tolist():
            out[q0 : q0 + block, j * block : (j + 1) * block] = mask_mod(
                zeros, zeros, q_off + q0, k_off + j * block
            )
    return out


# ---------------------------------------------------------------------------------------------
# The analytic block mask
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [512, 1024, 2048])
@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("causal", [False, True])
def test_block_mask_implies_exact_token_mask(seq_len, spec, causal):
    """What the kernel would compute must equal the dense rule, full-block shortcut included."""
    roles = _roles(seq_len)
    ce = torch.tensor([causal])
    mask_mod = build_summary_mask_mod(roles, spec, causal_example=ce)
    block_mask = build_summary_block_mask(roles, spec, causal_example=ce, block_size=BLOCK)
    assert block_mask is not None
    implied = _token_mask_implied_by(block_mask, mask_mod, seq_len)
    expected = summary_mask_allowed(roles, spec, causal_example=ce)[0]
    assert torch.equal(implied, expected)


@pytest.mark.parametrize("spec", SPECS)
def test_analytic_block_set_is_a_superset_of_the_reference(spec):
    """Dropping a block the predicate needs would be silent data loss; extra blocks are only slower."""
    from torch.nn.attention.flex_attention import create_block_mask

    seq_len = 1024
    roles = _roles(seq_len)
    mask_mod = build_summary_mask_mod(roles, spec)
    analytic = build_summary_block_mask(roles, spec, block_size=BLOCK)
    reference = create_block_mask(
        mask_mod, 1, None, seq_len, seq_len, device="cpu", BLOCK_SIZE=(BLOCK, BLOCK)
    )
    a = analytic.to_dense()[0, 0].bool()
    r = reference.to_dense()[0, 0].bool()
    assert bool((r <= a).all()), "analytic block set dropped a block the reference needs"


def test_sequence_length_not_a_multiple_of_the_block_size_declines():
    """The caller must be able to fall back rather than get a wrong mask."""
    roles = _roles(1024)[:, :, :1000]
    assert build_summary_block_mask(roles, SPECS[0], block_size=BLOCK) is None


def test_build_is_cheap_at_256k():
    """
    Regression test for the failure this builder exists to avoid.

    ``create_block_mask`` at this length would need ~760 GiB and ~17 minutes; the budgets below are
    loose enough not to be flaky but tight enough that a regression to that path cannot pass.
    (``ru_maxrss`` is a process high-water mark, so this measures a delta, not an absolute.)
    """
    seq_len = 262144
    roles = _roles(seq_len, doc_len=2000)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.time()
    block_mask = build_summary_block_mask(roles, SPECS[0], block_size=BLOCK)
    elapsed = time.time() - start
    peak_delta = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before

    assert block_mask is not None
    assert elapsed < 60.0, f"analytic block-mask build took {elapsed:.1f}s at T={seq_len}"
    # ru_maxrss is KiB on Linux and bytes on macOS; 16 GiB under either reading still rules out the
    # dense path by two orders of magnitude.
    assert peak_delta < 16 * 2**30, f"peak RSS grew by {peak_delta} at T={seq_len}"

    n_blocks = seq_len // BLOCK
    total = n_blocks * n_blocks
    density = (
        int(block_mask.kv_num_blocks.sum()) + int(block_mask.full_kv_num_blocks.sum())
    ) / total
    assert density < 0.25, f"summary mask is not sparse at 256k (density {density:.3f})"


# ---------------------------------------------------------------------------------------------
# Kernel equivalence: the analytic mask must not change numerics
# ---------------------------------------------------------------------------------------------


def _dense_reference(q, k, v, allowed, scale):
    bias = torch.where(
        allowed.unsqueeze(1),
        torch.zeros((), dtype=q.dtype),
        torch.full((), torch.finfo(q.dtype).min, dtype=q.dtype),
    )
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=False, scale=scale)


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("causal", [False, True])
def test_flex_output_matches_the_dense_mask(spec, causal):
    """End to end: the block-sparse kernel and the materialized mask must agree numerically."""
    from torch.nn.attention.flex_attention import flex_attention

    torch.manual_seed(0)
    seq_len, n_heads, head_dim = 512, 2, 32
    roles = _roles(seq_len)
    ce = torch.tensor([causal])
    q, k, v = (torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.float64) for _ in range(3))
    scale = head_dim**-0.5

    block_mask = build_summary_block_mask(roles, spec, causal_example=ce, block_size=BLOCK)
    got = flex_attention(q, k, v, block_mask=block_mask, scale=scale)
    want = _dense_reference(q, k, v, summary_mask_allowed(roles, spec, causal_example=ce), scale)
    torch.testing.assert_close(got, want, atol=1e-9, rtol=1e-7)


def test_flex_output_matches_the_reference_block_mask():
    """The analytic builder and ``create_block_mask`` must produce identical attention output."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    torch.manual_seed(0)
    seq_len, n_heads, head_dim = 512, 2, 32
    roles = _roles(seq_len)
    spec = SPECS[0]
    mask_mod = build_summary_mask_mod(roles, spec)
    q, k, v = (torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.float64) for _ in range(3))
    scale = head_dim**-0.5

    analytic = flex_attention(
        q, k, v, block_mask=build_summary_block_mask(roles, spec, block_size=BLOCK), scale=scale
    )
    reference = flex_attention(
        q,
        k,
        v,
        block_mask=create_block_mask(
            mask_mod, 1, None, seq_len, seq_len, device="cpu", BLOCK_SIZE=(BLOCK, BLOCK)
        ),
        scale=scale,
    )
    torch.testing.assert_close(analytic, reference, atol=1e-9, rtol=1e-7)


# ---------------------------------------------------------------------------------------------
# The module
# ---------------------------------------------------------------------------------------------


def _build_attention(**cfg_kwargs) -> SummaryTokenAttention:
    cfg = AttentionConfig(name=AttentionType.summary_token, n_heads=4, n_kv_heads=2, **cfg_kwargs)
    return cfg.build(d_model=64, layer_idx=0, n_layers=2, init_device="cpu")


def test_config_builds_the_right_class_and_spec():
    attn = _build_attention(n_summary_tokens=5, summary_visible_tokens=3)
    assert isinstance(attn, SummaryTokenAttention)
    assert attn.spec.n_summary_tokens == 5
    assert attn.spec.summary_visible_tokens == 3
    # Defaults are the treatment, not the floor control.
    assert attn.spec.summaries_read_own_document is True
    assert attn.spec.summaries_read_earlier_summaries is True
    assert attn.spec.query_reads_documents is False


def test_cached_decode_matches_full_sequence_last_row():
    """A generated QUERY token must retain the summary mask while using cached K/V."""
    torch.manual_seed(0)
    attn = _build_attention(n_summary_tokens=5)
    attn.eval()
    prompt_ids = torch.tensor(
        [[10, DOC_START, 20, DOC_END, SUMM, SUMM, SUMM, SUMM, SUMM, 30]]
    )
    full_ids = torch.cat([prompt_ids, torch.tensor([[31]])], dim=1)
    prompt_roles = build_summary_roles(prompt_ids, **IDS_KW)
    full_roles = build_summary_roles(full_ids, **IDS_KW)

    B, T, H, H_kv, D = 1, prompt_ids.shape[1], 4, 2, 16
    q = torch.randn(B, T + 1, H, D)
    k = torch.randn(B, T + 1, H_kv, D)
    v = torch.randn(B, T + 1, H_kv, D)
    with torch.no_grad():
        attn._summary_roles = full_roles
        expected = attn._sdpa_masked(q, k, v)[:, -1:]
        attn._summary_roles = prompt_roles
        attn.kv_cache_manager = KVCacheManager(B, T + 4, H_kv, D, torch.device("cpu"), q.dtype)
        attn._sdpa_cached(q[:, :T], k[:, :T], v[:, :T], cache_leftpad=None)
        attn._summary_roles = full_roles[:, :, -1:]
        actual = attn._sdpa_cached(q[:, -1:], k[:, -1:], v[:, -1:], cache_leftpad=None)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_flex_prefill_pads_arbitrary_prompt_length(monkeypatch):
    """Natural eval prompt lengths must not fall off the block-sparse path."""
    torch.manual_seed(0)
    T, H, H_kv, D = 513, 4, 2, 16
    attn = _build_attention(n_summary_tokens=5)
    roles = _roles(640)[:, :, :T]
    q = torch.randn(1, T, H, D)
    k = torch.randn(1, T, H_kv, D)
    v = torch.randn(1, T, H_kv, D)
    attn._summary_roles = roles
    seen = {}

    def fake_flex(qh, kh, vh, **kwargs):
        seen["shape"] = qh.shape
        return qh

    monkeypatch.setattr("olmo_core.nn.attention.summary_token._flex_attention", fake_flex)
    with torch.no_grad():
        out = attn._run_flex(q, k, v, roles, B=1, T=T)
    assert out is not None
    assert out.shape == q.shape
    assert seen["shape"][2] == 640


def test_levers_rejected_on_other_attention_types():
    with pytest.raises(OLMoConfigurationError, match="summary_token"):
        AttentionConfig(name=AttentionType.document_chunked, n_heads=4, n_summary_tokens=5).build(
            d_model=64, layer_idx=0, n_layers=2, init_device="cpu"
        )


def test_forward_without_roles_is_plain_causal():
    torch.manual_seed(0)
    attn = _build_attention(n_summary_tokens=5)
    attn.eval()
    x = torch.randn(1, 64, 64)
    with torch.no_grad():
        masked = attn(x, summary_roles=None)
        # Same module, roles absent -> the fallback must be ordinary causal attention.
        assert masked.shape == x.shape
        assert torch.isfinite(masked).all()


def test_forward_applies_the_mask():
    """A masked forward must differ from the causal one -- otherwise the mask is not reaching sdpa."""
    torch.manual_seed(0)
    attn = _build_attention(n_summary_tokens=5)
    attn.eval()
    seq_len = 256
    roles = _roles(seq_len, doc_len=40)
    x = torch.randn(1, seq_len, 64)
    with torch.no_grad():
        masked = attn(x, summary_roles=roles, causal_example=torch.tensor([False]))
        causal = attn(x, summary_roles=roles, causal_example=torch.tensor([True]))
    assert not torch.allclose(masked, causal)


def test_rejects_intra_document_packing():
    attn = _build_attention(n_summary_tokens=5)
    q = k = v = torch.randn(1, 8, 4, 16)
    with pytest.raises(NotImplementedError, match="generate_doc_lengths"):
        attn.sdpa(q, k, v, cu_doc_lens=torch.tensor([0, 8], dtype=torch.int32))


def test_roles_must_be_full_length():
    attn = _build_attention(n_summary_tokens=5)
    attn.eval()
    x = torch.randn(1, 256, 64)
    with pytest.raises(OLMoConfigurationError, match="unsharded"):
        attn(x, summary_roles=_roles(512, doc_len=40))
