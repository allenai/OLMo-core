"""Online train-time token router (``keep_token_rule="online"``, olmo_core.nn.online_token_router)."""
import math

import pytest
import torch

from olmo_core.nn.attention.chunked_mask import FREE_CHUNK_ID, build_chunk_ids_from_tokens
from olmo_core.nn.online_token_router import (
    N_STRUCT,
    OnlineRouterConfig,
    OnlineTokenRouter,
    query_mask,
    structural_features,
)

DS, DE, EOS, PH = 9001, 9002, 9000, 9003
QTOK = 1005  # a body token of doc 0 that also occurs in the query


def _row(B=1):
    def doc(n, base):
        return [DS] + [base + i for i in range(n)] + [DE]

    one = [1, 2, 3] + doc(10, 1000) + doc(40, 2000) + doc(3, 3000) + [7, QTOK, 9] + [50, 51, EOS]
    ids = torch.tensor([one] * B)
    cid = build_chunk_ids_from_tokens(ids, doc_start_id=DS, doc_end_id=DE, eos_id=EOS, mode="chunked")
    labels = torch.full_like(ids, -100)
    T = ids.shape[1]
    labels[:, T - 4 : T - 1] = ids[:, T - 3 :]  # answer = [50, 51, EOS]
    return ids, cid, labels


class _Emb(torch.nn.Module):
    """Minimal stand-in for the model attributes the router reads at feature_layer 0."""

    def __init__(self, V=9100, D=16):
        super().__init__()
        self.embeddings = torch.nn.Embedding(V, D)
        self.embed_scale = None
        self.embedding_norm = None


def test_structural_features_offsets_and_body():
    ids, cid, _ = _row()
    f, body = structural_features(ids, cid, doc_start_id=DS, doc_end_id=DE)
    assert f.shape == (1, ids.shape[1], N_STRUCT)
    assert int(body.sum()) == 10 + 40 + 3
    # first body token of doc 0 (position 4): start one-hot 0 and end one-hot 9 (10 body tokens)
    assert f[0, 4, 1 + 0] == 1 and f[0, 4, 1 + 27 + 9] == 1
    # markers and free tokens carry no features
    assert f[0, 3].abs().sum() == 0 and f[0, 0].abs().sum() == 0
    # doc 1 offset 20 falls in the first log2 bucket (16..31)
    p = 3 + 12 + 1 + 20
    assert ids[0, p] == 2020 and f[0, p, 1 + 16 + 0] == 1


def test_query_mask_excludes_answer():
    ids, cid, labels = _row()
    qm = query_mask(cid, labels)
    assert ids[0][qm[0]].tolist() == [7, QTOK, 9]


def _router(**kw):
    cfg = OnlineRouterConfig(**{"keep_frac": 0.2, "explore_frac": 0.0, "samples_per_row": 1000, **kw})
    return OnlineTokenRouter(cfg, doc_start_id=DS, doc_end_id=DE, eos_id=EOS, d_model=16)


def test_warmup_keeps_everything_then_learns_lexical_match():
    """Step 1 is keep-all; after a fit on targets that equal the in-query flag, the router keeps the
    query-matching token first (a learned rule, no heuristic init)."""
    torch.manual_seed(0)
    m = _Emb()
    ids, cid, labels = _row(B=2)
    r = _router(use_hidden=False, doc_share=0.0)
    r.begin_step(1, model=m)
    keep = r.select(m, ids, cid, labels)
    body = (cid >= 0) & (ids != DS) & (ids != DE)
    assert torch.equal(keep, body)
    # synthetic backward: target = log(1 + 9 * in_query)
    e = r._current
    e["ok"] = torch.ones_like(e["rows"], dtype=torch.bool)
    tgt = torch.log1p(9.0 * (ids[e["rows"], e["pos"]] == QTOK).float())
    e["target"] = tgt
    r._pending.append(e)
    r._current = None
    r.begin_step(2, model=m)
    assert r.w is not None and not r.dense_now()
    keep2 = r.select(m, ids, cid, labels)
    assert keep2[(ids == QTOK) & body].all()
    n_body = int(body[0].sum())
    assert int(keep2[0].sum()) == math.ceil(0.2 * n_body)
    assert m._flop_extra_actual > 0  # scoring + fit were charged


def test_exploration_share_and_dense_every():
    torch.manual_seed(0)
    m = _Emb()
    ids, cid, labels = _row()
    r = _router(explore_frac=0.5, dense_every=3, doc_share=0.0)
    r.w = torch.zeros(r.n_features)
    r.step = 2
    keep = r.select(m, ids, cid, labels)
    n = math.ceil(0.2 * 53)
    assert int(keep.sum()) == n
    assert int(r._current["explore"].sum()) == round(0.5 * n)
    r.step = 3
    assert r.dense_now()


def test_target_hook_maps_compacted_columns():
    """Targets are read at the compacted embedding output and mapped back via position_ids."""
    torch.manual_seed(0)
    m = _Emb()
    ids, cid, labels = _row()
    r = _router(use_hidden=False)
    r.step = 1
    keep = r.select(m, ids, cid, labels)  # warmup: all body tokens
    # compact: drop the first 3 prompt tokens only (positions shift by 3)
    pos = torch.arange(3, ids.shape[1])[None]
    h = torch.randn(1, pos.shape[1], 16, requires_grad=True)
    out = r.attach_target_hook(m, h, pos)
    wts = torch.arange(pos.shape[1], dtype=torch.float32)[None, :, None]
    (out * wts).sum().backward()
    e = r._pending[-1]
    assert e["ok"].all()
    # grad at compacted column c is wts[c] * ones(16): norm = c * 4
    exp = torch.log((e["pos"] - 3).float() * 4.0 + 1e-12)
    assert torch.allclose(e["target"], exp, atol=1e-4)


def test_ridge_recovers_hidden_direction():
    torch.manual_seed(0)
    m = _Emb()
    ids, cid, labels = _row(B=2)
    r = _router(ridge=1e-3, decay=1.0)
    r.step = 1
    r.select(m, ids, cid, labels)
    e = r._current
    w_true = torch.randn(16)
    h = m.embeddings(ids).detach()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    e["target"] = (h[e["rows"], e["pos"]] @ w_true)
    e["ok"] = torch.ones_like(e["rows"], dtype=torch.bool)
    r._pending.append(e)
    r._current = None
    r.begin_step(2, model=m)
    hd = r.w[r.n_scalar : r.n_scalar + 16]
    cos = torch.nn.functional.cosine_similarity(hd, w_true, dim=0)
    assert cos > 0.9, cos


def test_model_training_step_with_online_rule_cpu():
    """End to end on a tiny CPU model: warmup step keeps every body token, the hook captures targets,
    the refit routes the next step at the budget, and FLOP overhead is recorded."""
    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig

    torch.manual_seed(0)
    V = 9100
    cfg = TransformerConfig.olmo2_190M(vocab_size=V, n_layers=2, fused_ops=False, dtype=DType.float32)
    model = cfg.build(init_device="cpu")
    D = int(model.embeddings.weight.shape[1])
    router = OnlineTokenRouter(
        OnlineRouterConfig(keep_frac=0.25, explore_frac=0.2, samples_per_row=64),
        doc_start_id=DS, doc_end_id=DE, eos_id=EOS, d_model=D,
    )
    model.enable_pooled_soft_tokens(
        DS, DE, EOS, placeholder_id=PH, keep_prob=0.0, keep_seed=0, detach_soft_kv=True,
        drop_slots=True, keep_token_rule="online", keep_token_k=0.25, online_router=router,
    )
    model.train()
    ids, cid, labels = _row(B=2)
    for step in (1, 2):
        router.begin_step(step, model=model)
        out = model(ids, labels=labels)
        loss = out.loss
        assert torch.isfinite(loss)
        loss.backward()
        assert router._pending and router._pending[-1]["target"] is not None
        if step == 1:
            assert router.dense_now()
            comp = model._soft_token_compaction
            # all 53 body tokens + 6 markers + 9 free kept: only the right pad goes
            assert comp["tokens_out"] == comp["tokens_in"]
        model._soft_token_compaction = None
    router.begin_step(3, model=model)
    assert router.w is not None
    cb = model._compact_pooled_soft_tokens(ids, labels, -100)[0]
    # 25% of 53 body tokens (14; exploration tokens can overfill a short document's share by <= 3)
    # + 6 markers + 9 free (prompt, query, answer, EOS)
    n_kept = int(cb.row_lens[0]) - 6 - 9
    assert 14 <= n_kept <= 14 + 3, cb.row_lens
    assert model._flop_extra_actual > 0
    mets = router.metrics()
    assert "online_router/keep_frac" in mets and "online_router/rank_corr_kept" in mets


def test_online_rule_requires_router():
    from olmo_core.config import DType
    from olmo_core.exceptions import OLMoConfigurationError
    from olmo_core.nn.transformer import TransformerConfig

    cfg = TransformerConfig.olmo2_190M(vocab_size=9100, n_layers=1, fused_ops=False, dtype=DType.float32)
    model = cfg.build(init_device="cpu")
    with pytest.raises(OLMoConfigurationError):
        model.enable_pooled_soft_tokens(
            DS, DE, EOS, placeholder_id=PH, drop_slots=True, keep_token_rule="online", keep_token_k=0.3
        )


def _gpu_hybrid():
    """3 blocks: GDN, attention, GDN (the Qwen3.5 hybrid's shape in miniature)."""
    from olmo_core.nn.attention import AttentionConfig, GatedDeltaNetConfig
    from olmo_core.nn.feed_forward import FeedForwardConfig
    from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
    from olmo_core.nn.lm_head import LMHeadConfig
    from olmo_core.nn.rope import RoPEConfig, RoPEType
    from olmo_core.nn.transformer import TransformerBlockConfig, TransformerBlockType, TransformerConfig

    torch.manual_seed(0)
    ln = LayerNormConfig(name=LayerNormType.rms, bias=False)
    ff = FeedForwardConfig(hidden_size=512, bias=False)
    gdn = TransformerBlockConfig(name=TransformerBlockType.reordered_norm, sequence_mixer=GatedDeltaNetConfig(n_heads=8),
                                 layer_norm=ln, feed_forward=ff)
    attn = TransformerBlockConfig(name=TransformerBlockType.reordered_norm,
                                  attention=AttentionConfig(n_heads=8, rope=RoPEConfig(name=RoPEType.default)),
                                  layer_norm=ln, feed_forward=ff)
    cfg = TransformerConfig(d_model=256, vocab_size=9100, n_layers=3, block=gdn, block_overrides={1: attn},
                            lm_head=LMHeadConfig(layer_norm=ln, bias=False))
    model = cfg.build(init_device="cuda")
    model.init_weights()
    return model


@pytest.mark.gpu
def test_online_rule_on_gdn_hybrid_gpu():
    """GPU, GDN + attention: features from a no-grad prefix pass through blocks [0, 2) (GDN and
    attention on the FULL row), compaction to the budget, targets at the compacted embedding output,
    a refit, and the prefix pass charged as forward-only FLOPs."""
    from olmo_core.testing import requires_gpu  # noqa: F401
    from olmo_core.testing.utils import requires_fla  # noqa: F401

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    pytest.importorskip("fla")
    model = _gpu_hybrid()
    router = OnlineTokenRouter(
        OnlineRouterConfig(keep_frac=0.3, explore_frac=0.2, feature_layer=2, samples_per_row=64),
        doc_start_id=DS, doc_end_id=DE, eos_id=EOS, d_model=256,
    )
    model.enable_pooled_soft_tokens(
        DS, DE, EOS, placeholder_id=PH, keep_prob=0.0, keep_seed=0, detach_soft_kv=True,
        drop_slots=True, keep_token_rule="online", keep_token_k=0.3, online_router=router,
    )
    model.train()
    ids, cid, labels = _row(B=2)
    ids, labels = ids.cuda(), labels.cuda()
    prefix_flops = sum(int(b.num_flops_per_token(ids.shape[1])) for b in list(model.blocks.values())[:2]) / 3.0
    for step in (1, 2, 3):
        router.begin_step(step, model=model)
        model._flop_extra_actual = 0.0
        out = model(ids, labels=labels)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        e = router._pending[-1]
        assert e["target"] is not None and e["ok"].all() and torch.isfinite(e["target"]).all()
        # the prefix pass ran on the full (trimmed) row and was charged
        assert model._flop_extra_actual >= prefix_flops * ids.numel() * 0.99
        comp = model._soft_token_compaction
        if step >= 2:
            assert comp["tokens_out"] < comp["tokens_in"]
        model._soft_token_compaction = None
    assert router.w is not None and torch.isfinite(router.w).all()


def test_doc_share_spreads_budget_over_documents():
    """doc_share=1: every document keeps ceil(keep_frac * |body|) of its own top tokens, even when the
    scorer prefers one document; doc_share=0 lets the scorer take the whole budget from it."""
    m = _Emb()
    ids, cid, labels = _row()
    for share, expect in ((1.0, {0: 2, 1: 8, 2: 1}), (0.0, {0: 10, 1: 1, 2: 0})):
        r = _router(use_hidden=False, doc_share=share, keep_frac=0.2)
        r.w = torch.zeros(r.n_features)
        r.w[r.feature_names().index("start_off0")] = 0.0
        r.step = 2
        # scorer: doc 0 tokens high (via the doc-index feature, negative weight = earlier docs first)
        r.w[r.feature_names().index("doc_rel_idx")] = -5.0
        keep = r.select(m, ids, cid, labels)
        got = {d: int((keep[0] & (cid[0] == d)).sum()) for d in range(3)}
        assert sum(got.values()) == math.ceil(0.2 * 53) or share == 1.0
        if share == 1.0:
            assert got == expect, got
        else:
            assert got[0] == 10 and got[2] == 0, got


def test_mid_stack_drop_cpu():
    """drop_layer=1 on a 2-block model: the input row is kept whole, block 0 runs on every token, the
    router keeps its budget at block 1, labels follow the gather, targets come from block 1's input,
    and the FLOP adjustment is negative (block 1 + head ran on fewer tokens)."""
    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig

    torch.manual_seed(0)
    cfg = TransformerConfig.olmo2_190M(vocab_size=9100, n_layers=2, fused_ops=False, dtype=DType.float32)
    model = cfg.build(init_device="cpu")
    D = int(model.embeddings.weight.shape[1])
    router = OnlineTokenRouter(
        OnlineRouterConfig(keep_frac=0.25, explore_frac=0.2, samples_per_row=64, drop_layer=1),
        doc_start_id=DS, doc_end_id=DE, eos_id=EOS, d_model=D,
    )
    model.enable_pooled_soft_tokens(
        DS, DE, EOS, placeholder_id=PH, keep_prob=0.0, keep_seed=0, detach_soft_kv=True,
        drop_slots=True, keep_token_rule="online", keep_token_k=0.25, online_router=router,
    )
    model.train()
    ids, cid, labels = _row(B=2)
    seen = {}
    h1 = model.blocks["1"].register_forward_pre_hook(lambda m, a: seen.__setitem__("w", a[0].shape[1]))
    for step in (1, 2):
        router.begin_step(step, model=model)
        model._flop_extra_actual = 0.0
        out = model(ids, labels=labels)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        assert router._pending[-1]["target"] is not None and router._pending[-1]["ok"].all()
        comp = model._soft_token_compaction
        assert comp["tokens_out"] == comp["tokens_in"]  # nothing dropped at the input
        model._soft_token_compaction = None
        if step == 1:
            assert seen["w"] == ids.shape[1]  # warmup: every token through block 1
        else:
            # routed: block 1 saw 25% of the body + markers + free tokens
            assert seen["w"] < ids.shape[1] and seen["w"] >= 14 + 6 + 9
            assert model._flop_extra_actual < 0
    h1.remove()
    # the answer CE is unchanged by the gather when everything is kept (dense step): same loss as full
    router.cfg.dense_every = 1
    router.begin_step(3, model=model)
    model.eval()
    ref = model(ids, labels=labels).loss
    model.train()
    out = model(ids, labels=labels)
    assert torch.allclose(out.loss, ref, atol=1e-5), (out.loss, ref)
