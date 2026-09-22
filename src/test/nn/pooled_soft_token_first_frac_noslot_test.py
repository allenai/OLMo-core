"""Fractional ``first`` keep rule and ``drop_slots`` compaction (dev-loss grid schemes
``gold_first20`` / ``gold_fl20p8_noslot``, 2026-09-22)."""
import torch

from olmo_core.nn.attention.chunked_mask import (
    FREE_CHUNK_ID,
    build_chunk_ids_from_tokens,
    mark_doc_topk_tokens_free,
)
from olmo_core.nn.pooled_soft_token import compact_pooled_rows, first_scores

DS, DE, EOS, PH = 9001, 9002, 9000, 9003


def _row():
    def doc(n, base):
        return [DS] + [base + i for i in range(n)] + [DE]

    ids = torch.tensor([[1, 2, 3] + doc(10, 1000) + doc(50, 2000) + doc(3, 3000) + [7, 8, 9, EOS]])
    cid = build_chunk_ids_from_tokens(ids, doc_start_id=DS, doc_end_id=DE, eos_id=EOS, mode="chunked")
    return ids, cid


def test_fractional_first_keeps_prefix_per_document():
    ids, cid = _row()
    sc = first_scores(ids, cid, doc_start_id=DS, doc_end_id=DE)
    out = mark_doc_topk_tokens_free(cid, ids, sc, doc_start_id=DS, doc_end_id=DE, k=0.2)
    freed = ids[(out == FREE_CHUNK_ID) & (cid != FREE_CHUNK_ID)].tolist()
    # ceil(0.2*10)=2, ceil(0.2*50)=10, ceil(0.2*3)=1 -- all contiguous prefixes
    assert freed == [1000, 1001] + [2000 + i for i in range(10)] + [3000], freed


def test_drop_slots_emits_no_placeholder_and_keeps_positions():
    ids, cid = _row()
    keep = torch.tensor([[False, True, False]])  # doc 1 real, docs 0 and 2 pooled
    with_slots = compact_pooled_rows(ids, None, cid, keep, placeholder_id=PH, pad_token_id=EOS)
    no_slots = compact_pooled_rows(ids, None, cid, keep, placeholder_id=PH, pad_token_id=EOS, drop_slots=True)
    assert with_slots.soft_cols.numel() == 2 and (with_slots.input_ids == PH).sum() == 2
    assert no_slots.soft_cols.numel() == 0 and (no_slots.input_ids == PH).sum() == 0
    assert int(no_slots.row_lens[0]) == int(with_slots.row_lens[0]) - 2
    # kept tokens keep their ORIGINAL positions in both modes
    L = int(no_slots.row_lens[0])
    real_pos_with = with_slots.position_ids[0][with_slots.input_ids[0] != PH][: L]
    assert torch.equal(no_slots.position_ids[0][:L], real_pos_with)
    # default path is untouched
    again = compact_pooled_rows(ids, None, cid, keep, placeholder_id=PH, pad_token_id=EOS, drop_slots=False)
    assert torch.equal(again.input_ids, with_slots.input_ids)


def test_drop_slots_training_step_cpu():
    """Slot-less training: with ``drop_slots=True`` the model compacts to real tokens only (no
    placeholder columns), the LM loss on kept labels is finite, and a backward pass produces
    finite gradients while the detach paths (nothing to detach) stay no-ops."""
    from olmo_core.config import DType
    from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder
    from olmo_core.nn.transformer import TransformerConfig

    torch.manual_seed(0)
    V = 9100
    cfg = TransformerConfig.olmo2_190M(vocab_size=V, n_layers=2, fused_ops=False, dtype=DType.float32)
    model = cfg.build(init_device="cpu")
    model.enable_pooled_soft_tokens(
        DS, DE, EOS, placeholder_id=PH, keep_prob=0.0, keep_seed=0, detach_soft_kv=True,
        header_extra_tokens=2, drop_slots=True,
    )
    ids, cid = _row()
    labels = torch.full_like(ids, -100)
    labels[0, -5:-1] = ids[0, -4:]  # loss on the last real tokens before EOS
    model.train()
    model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=torch.tensor([[False, True, False]]))
    cb = model._compact_pooled_soft_tokens(ids, labels, -100)[0]
    assert (cb.input_ids == PH).sum() == 0 and cb.soft_cols.numel() == 0
    assert cb.input_ids.shape[1] < ids.shape[1]
    out = model(ids, labels=labels)
    loss = out[1] if isinstance(out, tuple) else out.loss
    assert torch.isfinite(loss), loss
    loss.backward()
    g = [p.grad for p in model.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(x).all() for x in g)
