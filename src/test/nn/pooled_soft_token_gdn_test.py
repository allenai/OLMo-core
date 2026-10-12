"""Soft-token slots must play NO role in training on the GDN hybrid (Prasann 2026-09-08): with
``detach_soft_kv`` + ``detach_soft_gdn`` the loss gradient at every block's input is exactly zero
at slot positions (attention K/V and GDN writes detached -> the slot's own query/FFN path is a dead
end), and non-zero at real positions. Without the GDN detach the slot leaks gradient through the
recurrent state."""

import torch

from olmo_core.nn.attention import AttentionConfig, GatedDeltaNetConfig
from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerBlockType, TransformerConfig
from olmo_core.testing import requires_gpu
from olmo_core.testing.utils import requires_fla

DOC_START, DOC_END, EOS, PLACEHOLDER = 900, 901, 999, 902
IGN = -100


def _row(n_docs=4, doc_len=6):
    ids = [11, 12]
    for d in range(n_docs):
        ids += [DOC_START, *(100 + doc_len * d + j for j in range(doc_len)), DOC_END]
    ids += [21, 22, 23, EOS]
    return ids


def _labels(ids, n_loss=3):
    lab = [IGN] * len(ids)
    first_eos = ids.index(EOS)
    for t in range(first_eos - n_loss, first_eos):
        lab[t] = ids[t + 1]
    return lab


def _hybrid(detach_gdn: bool, device="cuda"):
    """3 blocks: GDN, attention, GDN (the hybrid's shape in miniature)."""
    torch.manual_seed(0)
    ln = LayerNormConfig(name=LayerNormType.rms, bias=False)
    ff = FeedForwardConfig(hidden_size=512, bias=False)
    gdn = TransformerBlockConfig(name=TransformerBlockType.reordered_norm, sequence_mixer=GatedDeltaNetConfig(n_heads=8), layer_norm=ln, feed_forward=ff)
    attn = TransformerBlockConfig(name=TransformerBlockType.reordered_norm,
                                  attention=AttentionConfig(n_heads=8, rope=RoPEConfig(name=RoPEType.default)), layer_norm=ln, feed_forward=ff)
    cfg = TransformerConfig(d_model=256, vocab_size=1024, n_layers=3, block=gdn, block_overrides={1: attn},
                            lm_head=LMHeadConfig(layer_norm=ln, bias=False))
    model = cfg.build(init_device=device)
    model.init_weights()
    model.enable_pooled_soft_tokens(DOC_START, DOC_END, EOS, placeholder_id=PLACEHOLDER, keep_prob=0.0,
                                    detach_soft_kv=True, detach_soft_gdn=detach_gdn)
    model.pooled_projector.reset_parameters()
    model.train()
    model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=torch.tensor([[True, False, True, False]]))
    return model


def _slot_grads(model, device="cuda"):
    """Loss gradient w.r.t. every block's INPUT, split into slot columns and real columns."""
    ids = _row()
    x = torch.tensor([ids], device=device)
    lab = torch.tensor([_labels(ids)], device=device)
    cb = model._compact_pooled_soft_tokens(x, lab, IGN)[0]
    slot_cols = cb.soft_cols.tolist()
    real_cols = [c for c in range(cb.input_ids.shape[1]) if c not in slot_cols]
    captured = []

    def pre_hook(mod, args):
        h = args[0]
        h.retain_grad()
        captured.append(h)

    handles = [blk.register_forward_pre_hook(pre_hook) for blk in model.blocks.values()]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, labels=lab)
    out.loss.backward()
    for hd in handles:
        hd.remove()
    assert len(captured) == len(model.blocks) and len(slot_cols) == 2
    slot = [float(h.grad[0, slot_cols].abs().sum()) for h in captured]
    real = [float(h.grad[0, real_cols].abs().sum()) for h in captured]
    return slot, real


@requires_gpu
@requires_fla
def test_slots_get_zero_gradient_when_gdn_detached():
    slot, real = _slot_grads(_hybrid(detach_gdn=True))
    assert all(s == 0.0 for s in slot), slot  # no gradient reaches a slot at ANY block input
    assert all(r > 0.0 for r in real), real


@requires_gpu
@requires_fla
def test_slots_leak_gradient_through_gdn_without_the_detach():
    slot, _ = _slot_grads(_hybrid(detach_gdn=False))
    assert any(s > 0.0 for s in slot), slot  # the pre-2026-09-08 behaviour: GDN writes carry gradient


@requires_gpu
@requires_fla
def test_gdn_detach_leaves_forward_unchanged():
    torch.manual_seed(0)
    a, b = _hybrid(detach_gdn=True), _hybrid(detach_gdn=False)
    b.load_state_dict(a.state_dict())
    ids = _row()
    x = torch.tensor([ids], device="cuda")
    lab = torch.tensor([_labels(ids)], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        la, lb = a(x, labels=lab).loss, b(x, labels=lab).loss
    torch.testing.assert_close(la, lb)
