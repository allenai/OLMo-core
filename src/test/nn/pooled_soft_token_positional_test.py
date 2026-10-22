"""Positional-block mode of the input token-drop (pooled soft token) path: marker-free rows whose
"documents" are fixed 512-token position blocks (``positional_block_size``), as used by the
olmo3_cptdrop arms I-rand (random 20% of blocks kept whole) and I-fl (first+last 25% of every
block kept), both slot-less (``drop_slots``)."""

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.attention.chunked_mask import (
    FREE_CHUNK_ID,
    PAD_CHUNK_ID,
    build_positional_block_chunk_ids,
)
from olmo_core.nn.pooled_soft_token import example_lengths
from olmo_core.nn.transformer import TransformerConfig

EOS = 100257  # the OLMo 3 / dolma2 EOS
VOCAB = 100352
IGN = -100
T = 32768
BLOCK = 512


def _rows(B, T=T, seed=0):
    """EOS + (T - 1) plain-text tokens per row, like build_pool_data.py's train32k rows."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, 50000, (B, T), generator=g)
    ids[:, 0] = EOS
    return ids


def _labels(ids):
    """get_labels-style shifted labels: label[t] = ids[t + 1], last position ignored."""
    lab = torch.full_like(ids, IGN)
    lab[:, :-1] = ids[:, 1:]
    return lab


def _model(**kw):
    cfg = TransformerConfig.olmo2_30M(
        vocab_size=VOCAB, n_layers=kw.pop("n_layers", 1), fused_ops=False, dtype=DType.float32
    )
    model = cfg.build(init_device="cpu")
    model.enable_pooled_soft_tokens(
        -5, -6, EOS, placeholder_id=100300, drop_slots=True, positional_block_size=BLOCK, **kw
    )
    model.train()
    return model


def _compact(model, ids):
    out = model._compact_pooled_soft_tokens(ids, _labels(ids), IGN)
    assert out is not None
    return out[0]


def _kept_positions(cb, b):
    L = int(cb.row_lens[b])
    return cb.position_ids[b, :L]


# I-fl reference: block d covers positions [1 + 512 d, 1 + 512 (d + 1)); k_d = ceil(0.25 |block|)
# = 128 for every block (the last block has 511 tokens, ceil(127.75) = 128), first k_d // 2 = 64 and
# last k_d - k_d // 2 = 64 kept -> 1 + 64 * 128 = 8193 tokens per 32,768-token row (0.25003).
def _ifl_reference(T=T, block=BLOCK, k=0.25):
    import math

    keep = [0]
    start = 1
    while start < T:
        n = min(block, T - start)
        kd = max(1, math.ceil(k * n))
        if kd >= n:
            keep += list(range(start, start + n))
        else:
            a = kd // 2
            keep += list(range(start, start + a)) + list(range(start + n - (kd - a), start + n))
        start += n
    return torch.tensor(keep)


def test_positional_block_chunk_ids_layout():
    ids = _rows(2)
    cid = build_positional_block_chunk_ids(ids, block_size=BLOCK, eos_id=EOS)
    assert cid.dtype == torch.int32
    assert (cid[:, 0] == FREE_CHUNK_ID).all()  # leading EOS: kept, in no block
    assert int(cid.max()) == 63  # 63 full blocks + a 511-token last block
    counts = torch.bincount(cid[0, 1:].long())
    assert counts[:63].eq(512).all() and int(counts[63]) == 511
    assert int(cid[0, 1]) == 0 and int(cid[0, 512]) == 0 and int(cid[0, 513]) == 1
    assert not (cid == PAD_CHUNK_ID).any()

    # No leading EOS: blocks start at position 0.
    ids2 = ids.clone()
    ids2[:, 0] = 7
    cid2 = build_positional_block_chunk_ids(ids2, block_size=BLOCK, eos_id=EOS)
    assert int(cid2[0, 0]) == 0 and int(cid2[0, 511]) == 0 and int(cid2[0, 512]) == 1

    # Trailing EOS padding: the first trailing EOS is content, the rest PAD; interior EOS is content.
    ids3 = _rows(1, T=40)
    ids3[0, 10] = EOS
    ids3[0, 30:] = EOS
    cid3 = build_positional_block_chunk_ids(ids3, block_size=8, eos_id=EOS)
    assert int(cid3[0, 10]) == (10 - 1) // 8
    assert int(cid3[0, 30]) == (30 - 1) // 8
    assert (cid3[0, 31:] == PAD_CHUNK_ID).all()


def test_irand_keeps_whole_blocks_at_twenty_percent():
    B = 8
    ids = _rows(B)
    model = _model(keep_prob=0.2)
    cb = _compact(model, ids)
    assert cb.soft_rows.numel() == 0  # no slot tokens
    kept_sets = []
    total = 0
    for b in range(B):
        pos = _kept_positions(cb, b)
        assert int(pos[0]) == 0 and torch.all(pos[1:] > pos[:-1])  # EOS kept, order preserved
        blk = torch.bincount((pos[1:] - 1) // BLOCK, minlength=64)
        full = torch.tensor([512] * 63 + [511])
        assert torch.all((blk == 0) | (blk == full)), "a block was partially kept"
        kept_sets.append(tuple((blk > 0).tolist()))
        total += pos.numel()
    frac = total / (B * T)
    # 512 Bernoulli(0.2) blocks: SE 0.018
    assert abs(frac - 0.2) < 0.06, frac
    # The per-row salt: rows with different tokens keep different blocks.
    assert len(set(kept_sets)) == B
    # ... and the draw is deterministic given the row.
    cb2 = _compact(model, ids)
    assert torch.equal(cb2.position_ids, cb.position_ids)


def test_ifl_keeps_first_last_quarter_of_every_block():
    B = 2
    ids = _rows(B)
    model = _model(keep_prob=0.0, keep_token_rule="first_last", keep_token_k=0.25)
    cb = _compact(model, ids)
    ref = _ifl_reference()
    assert ref.numel() == 8193
    lab = _labels(ids)
    for b in range(B):
        pos = _kept_positions(cb, b)
        assert torch.equal(pos, ref)
        L = int(cb.row_lens[b])
        # kept tokens and their labels are the originals at their ORIGINAL positions
        assert torch.equal(cb.input_ids[b, :L], ids[b, ref])
        assert torch.equal(cb.labels[b, :L], lab[b, ref])
    assert abs(8193 / T - 0.25) < 1e-3


def test_zero_drop_rows_unchanged():
    ids = _rows(2)
    model = _model(keep_prob=1.0)
    cb = _compact(model, ids)
    assert cb.input_ids.shape == ids.shape
    assert torch.equal(cb.input_ids, ids)
    assert torch.equal(cb.labels, _labels(ids))
    assert torch.equal(cb.position_ids, torch.arange(T).expand(2, T))


def test_header_extra_tokens_and_integer_first_per_block():
    ids = _rows(1, T=1 + 4 * 16)
    for kw in ({"header_extra_tokens": 3}, {"keep_token_rule": "first", "keep_token_k": 3}):
        cfg = TransformerConfig.olmo2_30M(
            vocab_size=VOCAB, n_layers=1, fused_ops=False, dtype=DType.float32
        )
        model = cfg.build(init_device="cpu")
        model.enable_pooled_soft_tokens(
            -5, -6, EOS, placeholder_id=100300, drop_slots=True, positional_block_size=16,
            keep_prob=0.0, **kw,
        )
        model.train()
        cb = _compact(model, ids)
        want = [0] + [1 + 16 * d + j for d in range(4) for j in range(3)]
        assert _kept_positions(cb, 0).tolist() == want, kw


def test_positional_refuses_header_stop_and_distill():
    cfg = TransformerConfig.olmo2_30M(vocab_size=VOCAB, n_layers=1, fused_ops=False)
    from olmo_core.exceptions import OLMoConfigurationError

    for kw in ({"header_stop_id": 25}, {"distill_prob": 0.5}):
        model = cfg.build(init_device="cpu")
        with pytest.raises(OLMoConfigurationError):
            model.enable_pooled_soft_tokens(
                -5, -6, EOS, placeholder_id=100300, positional_block_size=BLOCK, **kw
            )


def test_example_lengths_leading_eos_is_not_a_terminator():
    ids = _rows(2, T=64)
    assert example_lengths(ids, None, EOS) == [64, 64]
    ids[1, 40:] = EOS  # trailing pad: up to and including the first terminating EOS
    assert example_lengths(ids, None, EOS) == [64, 41]


def test_forward_trains_on_kept_labels_and_logs_label_frac():
    """End to end through Transformer.forward: the loss is finite, the FLOP-meter counters see the
    compacted length, and the kept-label fraction (the logged-CE correction) is recorded."""
    Ts, B = 2048, 2
    ids = _rows(B, T=Ts)
    model = _model(keep_prob=0.0, keep_token_rule="first_last", keep_token_k=0.25, n_layers=2)
    model._flop_pad_id = EOS
    out = model(ids, labels=_labels(ids), ignore_index=IGN)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    st = model._soft_token_compaction
    kept = int(_ifl_reference(T=Ts).numel())  # 1 + 4 * 128 = 513
    assert st["tokens_in"] == B * Ts and st["tokens_out"] == B * kept
    # every kept position has a live label except the row's last position (always kept by I-fl)
    assert st["labels_in"] == B * (Ts - 1) and st["labels_out"] == B * (kept - 1)
    lens = model._flop_example_lens
    assert lens["dense"] == [Ts] * B and lens["actual"] == [kept] * B


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("arm", ["irand", "ifl"])
def test_gpu_smoke_olmo3_two_steps(arm):
    """Two AdamW steps of a small OLMo 3-style model (SWA [4096 x3, full], flash-2, YaRN x4 on the
    full-attention layers) on 32k EOS-led rows with each arm's flags."""
    from olmo_core.nn.rope import YaRNRoPEScalingConfig

    cfg = TransformerConfig.olmo3_190M(vocab_size=VOCAB, n_layers=4, dtype=DType.bfloat16)
    cfg = cfg.with_rope_scaling(
        YaRNRoPEScalingConfig(factor=4.0, beta_fast=32, beta_slow=1, old_context_len=8192)
    )
    model = cfg.build(init_device="cuda")
    kw = (
        {"keep_prob": 0.2}
        if arm == "irand"
        else {"keep_prob": 0.0, "keep_token_rule": "first_last", "keep_token_k": 0.25}
    )
    model.enable_pooled_soft_tokens(
        -5, -6, EOS, placeholder_id=100300, drop_slots=True, positional_block_size=BLOCK, **kw
    )
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    for step in range(2):
        ids = _rows(1, seed=step).cuda()
        out = model(ids, labels=_labels(ids), ignore_index=IGN)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss))
    st = model._soft_token_compaction
    frac = st["tokens_out"] / st["tokens_in"]
    print(f"[gpu-smoke] {arm}: losses={losses} kept_frac={frac:.4f}")
    assert all(torch.isfinite(torch.tensor(losses)))
    if arm == "ifl":
        assert st["tokens_out"] == 2 * 8193
    else:
        assert 0.05 < frac < 0.4
