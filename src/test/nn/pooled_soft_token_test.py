"""Tests for soft-token document pooling ("B1"): compact_pooled_rows + the
Transformer.enable_pooled_soft_tokens training path."""

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens
from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder, install_pooled_doc_keep
from olmo_core.nn.pooled_soft_token import compact_pooled_rows
from olmo_core.nn.transformer import TransformerConfig

DOC_START, DOC_END, EOS, PLACEHOLDER = 900, 901, 999, 902
IGN = -100


def _row(n_docs=3, doc_len=3, pad=0):
    """[free, free] + n_docs marker-wrapped docs + [q, q, EOS] + pad*[EOS]."""
    ids = [11, 12]
    for d in range(n_docs):
        ids += [DOC_START, *(100 + doc_len * d + j for j in range(doc_len)), DOC_END]
    ids += [21, 22, EOS] + [EOS] * pad
    return ids


def _shifted_labels(ids, n_loss=3):
    """Labels shifted like get_labels: label[t] = ids[t+1]; loss only on the answer region (the
    ``n_loss`` positions whose targets are the tokens up to and including the FIRST EOS)."""
    lab = [IGN] * len(ids)
    first_eos = ids.index(EOS)
    for t in range(first_eos - n_loss, first_eos):
        lab[t] = ids[t + 1]
    return lab


def _model(enable=True, keep_prob=0.5):
    cfg = TransformerConfig.olmo2_190M(
        vocab_size=1000, n_layers=2, fused_ops=False, dtype=DType.float32
    )
    model = cfg.build(init_device="cpu")
    if enable:
        model.enable_pooled_soft_tokens(
            DOC_START, DOC_END, EOS, placeholder_id=PLACEHOLDER, keep_prob=keep_prob
        )
    model.train()
    return model


def _chunk_ids(ids):
    return build_chunk_ids_from_tokens(
        torch.tensor([ids]), doc_start_id=DOC_START, doc_end_id=DOC_END, eos_id=EOS
    )


def test_compact_pooled_rows_basic():
    ids = _row(n_docs=3, pad=4)
    x = torch.tensor([ids])
    lab = torch.tensor([_shifted_labels(ids)])
    cids = _chunk_ids(ids)
    keep = torch.tensor([[True, False, True]])
    cb = compact_pooled_rows(
        x, lab, cids, keep, placeholder_id=PLACEHOLDER, pad_token_id=EOS, ignore_index=IGN
    )
    new_ids, new_lab, pos = cb.input_ids, cb.labels, cb.position_ids
    s_rows, s_cols, s_docs = cb.soft_rows, cb.soft_cols, cb.soft_docs
    # doc1 (5 tokens incl markers) -> 1 placeholder; original PAD (4 tokens) dropped.
    assert new_ids.shape[1] == len(ids) - 4 - 4
    assert s_docs.tolist() == [1] and s_rows.tolist() == [0]
    col = int(s_cols[0])
    assert new_ids[0, col] == PLACEHOLDER
    # The soft token sits at doc1's center original position, between doc0 and doc2 tokens.
    doc1_positions = (cids[0] == 1).nonzero(as_tuple=True)[0]
    assert pos[0, col] == (doc1_positions[0] + doc1_positions[-1]) // 2
    # Kept tokens keep their original ids and positions, in order.
    kept_mask = new_ids[0] != PLACEHOLDER
    orig_kept = [(p, i) for p, (i, c) in enumerate(zip(ids, cids[0].tolist())) if c in (-1, 0, 2)]
    assert [int(i) for i in new_ids[0, kept_mask]] == [i for _, i in orig_kept]
    assert [int(p) for p in pos[0, kept_mask]] == [p for p, _ in orig_kept]
    # Counted-label set preserved exactly (labels only in the trailing FREE region).
    assert int((new_lab != IGN).sum()) == int((lab != IGN).sum())
    assert sorted(new_lab[0][new_lab[0] != IGN].tolist()) == sorted(lab[0][lab[0] != IGN].tolist())


def test_forward_no_markers_is_identity_path():
    model = _model()
    x = torch.randint(5, 800, (2, 16))
    x[:, -1] = EOS
    lab = torch.full_like(x, IGN)
    lab[:, -4:-1] = x[:, -3:]
    out_enabled = model(x, labels=lab)
    model._pooled_soft_tokens, saved = None, model._pooled_soft_tokens
    out_disabled = model(x, labels=lab)
    model._pooled_soft_tokens = saved
    assert torch.allclose(out_enabled.loss, out_disabled.loss, atol=1e-6)


def test_keep_all_no_pad_matches_disabled():
    model = _model()
    model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=torch.ones(1, 3, dtype=torch.bool))
    ids = _row(n_docs=3, pad=0)  # EOS is the last token -> no PAD to drop
    x = torch.tensor([ids])
    lab = torch.tensor([_shifted_labels(ids)])
    out_enabled = model(x, labels=lab)
    model._pooled_soft_tokens, saved = None, model._pooled_soft_tokens
    out_disabled = model(x, labels=lab)
    model._pooled_soft_tokens = saved
    assert torch.allclose(out_enabled.loss, out_disabled.loss, atol=1e-6)


def test_pooled_forward_shrinks_and_preserves_label_count():
    model = _model()
    model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=torch.tensor([[True, False, False]]))
    ids = _row(n_docs=3, pad=2)
    x = torch.tensor([ids])
    lab = torch.tensor([_shifted_labels(ids)])
    cb, inject, _ = model._compact_pooled_soft_tokens(x, lab, IGN)
    assert cb.input_ids.shape[1] < x.shape[1]
    assert int((cb.labels != IGN).sum()) == int((lab != IGN).sum())
    assert inject is not None and inject[2].shape == (2, model.d_model)
    out = model(x, labels=lab)
    assert torch.isfinite(out.loss)


def test_grad_reaches_projector_and_embeddings():
    model = _model()
    model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=torch.tensor([[False, False, True]]))
    ids = _row(n_docs=3)
    x = torch.tensor([ids])
    lab = torch.tensor([_shifted_labels(ids)])
    model(x, labels=lab).loss.backward()
    assert model.pooled_projector.w_out.weight.grad is not None
    assert model.pooled_projector.w_out.weight.grad.abs().sum() > 0
    emb_grad = model.embeddings.weight.grad
    assert emb_grad is not None
    # Pooled docs' token embeddings receive gradient through the mean-embed feature.
    assert emb_grad[100:103].abs().sum() > 0  # doc0 content tokens (pooled)


def test_eval_mode_untouched():
    model = _model()
    model.eval()
    ids = _row(n_docs=3, pad=2)
    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model(x)
    assert logits.shape[:2] == (1, len(ids))  # full length: no compaction at eval


def test_projector_residual_init_is_identity():
    model = _model()
    z = torch.randn(5, model.d_model)
    assert torch.allclose(model.pooled_projector(z), z)


def test_installer_attaches_to_soft_token_transformer():
    model = _model()
    holder = install_pooled_doc_keep(model, lambda ids: torch.ones(ids.shape[0], 3).bool())
    assert holder.n_attached == 1
    ids = _row(n_docs=3)
    x = torch.tensor([ids])
    model(x)  # pre-hook fires and sets keep_docs
    assert holder.keep_docs is not None and holder.keep_docs.shape == (1, 3)


def test_mutually_exclusive_with_document_chunk_attention():
    model = _model(enable=False)
    model.enable_document_chunk_attention(DOC_START, DOC_END, EOS)
    with pytest.raises(Exception):
        model.enable_pooled_soft_tokens(DOC_START, DOC_END, EOS, placeholder_id=PLACEHOLDER)


def _aux_model(weight=1.0):
    cfg = TransformerConfig.olmo2_190M(
        vocab_size=1000, n_layers=2, fused_ops=False, dtype=DType.float32
    )
    model = cfg.build(init_device="cpu")
    model.enable_pooled_soft_tokens(
        DOC_START, DOC_END, EOS, placeholder_id=PLACEHOLDER, aux_match_weight=weight,
        aux_queries=4,
    )
    model.train()
    return model


def test_aux_shadows_do_not_perturb_lm_loss():
    # The shadows + masked path must leave the LM computation bit-identical: ce_loss with aux on
    # must equal ce_loss with aux off (same keep set, same weights).
    torch.manual_seed(0)
    m_aux = _aux_model(weight=1.0)
    m_plain = _model()
    m_plain.load_state_dict({k: v for k, v in m_aux.state_dict().items()})
    keep = torch.tensor([[True, False, True]])
    m_aux._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep)
    m_plain._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep)
    ids = _row(n_docs=3, pad=2)
    x = torch.tensor([ids])
    lab = torch.tensor([_shifted_labels(ids)])
    torch.manual_seed(1)
    out_aux = m_aux(x, labels=lab)
    torch.manual_seed(1)
    out_plain = m_plain(x, labels=lab)
    assert torch.allclose(out_aux.ce_loss, out_plain.ce_loss, atol=1e-5), (
        out_aux.ce_loss, out_plain.ce_loss,
    )
    # ...but the TOTAL loss includes a nonzero aux term.
    assert (out_aux.loss - out_plain.loss).abs() > 0


def test_aux_loss_grad_reaches_projector():
    torch.manual_seed(0)
    model = _aux_model(weight=1.0)
    model._pooled_keep_holder = PooledDocKeepHolder(
        keep_docs=torch.tensor([[True, False, True]])
    )
    ids = _row(n_docs=3)
    x = torch.tensor([ids])
    lab = torch.tensor([_shifted_labels(ids)])
    model(x, labels=lab).loss.backward()
    g = model.pooled_projector.w_out.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_aux_zero_weight_skips_shadows():
    torch.manual_seed(0)
    model = _aux_model(weight=0.0)
    model._pooled_keep_holder = PooledDocKeepHolder(
        keep_docs=torch.tensor([[True, False, True]])
    )
    ids = _row(n_docs=3)
    x = torch.tensor([ids])
    cb, _, _ = model._compact_pooled_soft_tokens(x, None, IGN)
    assert cb.shadow_rows.numel() == 0


def _detach_model(aux=0.0):
    cfg = TransformerConfig.olmo2_190M(
        vocab_size=1000, n_layers=2, fused_ops=False, dtype=DType.float32
    )
    model = cfg.build(init_device="cpu")
    model.enable_pooled_soft_tokens(
        DOC_START, DOC_END, EOS, placeholder_id=PLACEHOLDER, aux_match_weight=aux,
        aux_queries=4, detach_soft_kv=True,
    )
    model.train()
    return model


def test_detach_soft_kv_blocks_lm_gradient_to_projector():
    torch.manual_seed(0)
    model = _detach_model(aux=0.0)
    model._pooled_keep_holder = PooledDocKeepHolder(
        keep_docs=torch.tensor([[True, False, False]])
    )
    ids = _row(n_docs=3)
    x = torch.tensor([ids])
    lab = torch.tensor([_shifted_labels(ids)])
    out = model(x, labels=lab)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    g = model.pooled_projector.w_in.weight.grad
    assert g is None or g.abs().sum() == 0  # static KV: no LM gradient into the projector
    # Pooled docs' token embeddings likewise receive no gradient through the mean feature.
    assert model.embeddings.weight.grad[103:106].abs().sum() == 0  # doc1 content (pooled)


def test_detach_soft_kv_forward_identical():
    # Detachment must not change the forward values, only the backward graph.
    torch.manual_seed(0)
    m_det = _detach_model(aux=0.0)
    m_std = _model()
    m_std.load_state_dict(m_det.state_dict())
    keep = torch.tensor([[True, False, False]])
    m_det._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep)
    m_std._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep)
    ids = _row(n_docs=3)
    x = torch.tensor([ids])
    lab = torch.tensor([_shifted_labels(ids)])
    assert torch.allclose(m_det(x, labels=lab).loss, m_std(x, labels=lab).loss, atol=1e-6)


def test_detach_with_aux_trains_projector_only_via_shadows():
    torch.manual_seed(0)
    model = _detach_model(aux=1.0)
    model._pooled_keep_holder = PooledDocKeepHolder(
        keep_docs=torch.tensor([[True, False, True]])
    )
    ids = _row(n_docs=3)
    x = torch.tensor([ids])
    lab = torch.tensor([_shifted_labels(ids)])
    out = model(x, labels=lab)
    out.loss.backward()
    g = model.pooled_projector.w_out.weight.grad
    assert g is not None and g.abs().sum() > 0  # aux (shadow) path still trains P


def _distill_model(prob=1.0, detach=True):
    cfg = TransformerConfig.olmo2_190M(
        vocab_size=1000, n_layers=2, fused_ops=False, dtype=DType.float32
    )
    model = cfg.build(init_device="cpu")
    model.enable_pooled_soft_tokens(
        DOC_START, DOC_END, EOS, placeholder_id=PLACEHOLDER,
        detach_soft_kv=detach, distill_prob=prob, distill_weight=1.0, distill_layer_stride=1,
    )
    model.train()
    return model


def test_paired_distill_forward_runs_and_includes_teacher_loss():
    torch.manual_seed(0)
    m = _distill_model(prob=1.0)
    m._pooled_keep_holder = PooledDocKeepHolder(keep_docs=torch.tensor([[True, False, False]]))
    ids = _row(n_docs=3)
    x = torch.tensor([ids])
    lab = torch.tensor([_shifted_labels(ids)])
    out_paired = m(x, labels=lab)
    m2 = _distill_model(prob=0.0)
    m2.load_state_dict(m.state_dict())
    m2._pooled_keep_holder = PooledDocKeepHolder(keep_docs=torch.tensor([[True, False, False]]))
    out_solo = m2(x, labels=lab)
    assert torch.isfinite(out_paired.loss)
    # Paired loss = comp LM + full LM + distill > solo comp LM alone.
    assert out_paired.loss > out_solo.loss
    # ce_loss reports the COMPRESSED pass only -> identical between the two.
    assert torch.allclose(out_paired.ce_loss, out_solo.ce_loss, atol=1e-6)


def test_paired_distill_backward_flows():
    torch.manual_seed(0)
    m = _distill_model(prob=1.0)
    m._pooled_keep_holder = PooledDocKeepHolder(keep_docs=torch.tensor([[True, False, False]]))
    ids = _row(n_docs=3)
    x = torch.tensor([ids])
    lab = torch.tensor([_shifted_labels(ids)])
    m(x, labels=lab).loss.backward()
    for p in m.blocks["0"].parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        break


def test_distill_coin_is_deterministic():
    m1 = _distill_model(prob=0.5)
    m2 = _distill_model(prob=0.5)
    import random as _r

    seq1 = [_r.Random(f"distill:{m1._pooled_soft_tokens['keep_seed']}:{i}").random() < 0.5 for i in range(20)]
    seq2 = [_r.Random(f"distill:{m2._pooled_soft_tokens['keep_seed']}:{i}").random() < 0.5 for i in range(20)]
    assert seq1 == seq2  # rank-synchronized branch decisions
