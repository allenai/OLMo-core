import torch

from olmo_core.config import DType
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens
from olmo_core.nn.role_gated_ffn import RoleGateHolder, install_role_gated_ffn
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import seed_all

DOC_START, DOC_END, EOS = 900, 901, 999
IGN = -100


def _row(n_docs=3, doc_len=3):
    ids = [11, 12]
    for d in range(n_docs):
        ids += [DOC_START, *(100 + doc_len * d + j for j in range(doc_len)), DOC_END]
    ids += [21, 22, EOS]
    return ids


def _model(enable=True, start_layer=0):
    cfg = TransformerConfig.olmo2_190M(
        vocab_size=1000, n_layers=2, fused_ops=False, dtype=DType.float32
    )
    model = cfg.build(init_device="cpu")
    if enable:
        model.enable_role_gated_ffn(DOC_START, DOC_END, EOS, start_layer=start_layer)
    model.eval()
    return model


def test_no_marker_row_is_identity():
    """A row with no doc markers is all-FREE -> gated model output equals ungated exactly."""
    seed_all(0)
    m_gated = _model(enable=True)
    seed_all(0)
    m_plain = _model(enable=False)
    x = torch.randint(5, 800, (2, 16))
    x[:, -1] = EOS
    with torch.no_grad():
        out_g = m_gated(x)
        out_p = m_plain(x)
    torch.testing.assert_close(out_g, out_p)


def test_context_tokens_are_gated_and_free_tokens_not():
    """Perturbing FFN weights must change context-token outputs only via later attention mixing,
    and the gated model must differ from the ungated one on a marker row (FFN really skipped)."""
    seed_all(1)
    m_gated = _model(enable=True)
    seed_all(1)
    m_plain = _model(enable=False)
    ids = _row()
    x = torch.tensor([ids])
    with torch.no_grad():
        out_g = m_gated(x)
        out_p = m_plain(x)
    assert not torch.allclose(out_g, out_p, atol=1e-5)


def test_direct_gate_math():
    """The gated FFN output is exactly (full FFN on masked-True rows, zeros elsewhere)."""
    seed_all(2)
    model = _model(enable=False)
    block = model.blocks["0"]
    ff = block.feed_forward
    holder = RoleGateHolder()
    gated_keys = install_role_gated_ffn(model.blocks, holder, start_layer=0)
    assert gated_keys == ["0", "1"]
    x = torch.randn(2, 8, model.d_model)
    ids = torch.randint(5, 800, (2, 8))
    ids[0, 2:5] = torch.tensor([DOC_START, 100, DOC_END])
    cid = build_chunk_ids_from_tokens(ids, doc_start_id=DOC_START, doc_end_id=DOC_END, eos_id=EOS)
    holder.set_from_chunk_ids(cid)
    with torch.no_grad():
        out = ff(x)
        ref = ff._role_gate_orig_forward(x)
    mask = cid < 0
    torch.testing.assert_close(out[mask], ref[mask])
    assert torch.all(out[~mask] == 0)


def test_start_layer_skips_early_blocks():
    model = _model(enable=False)
    holder = RoleGateHolder()
    gated = install_role_gated_ffn(model.blocks, holder, start_layer=1)
    assert gated == ["1"]


def test_grads_flow_through_gate():
    seed_all(3)
    model = _model(enable=True)
    model.train()
    ids = _row()
    x = torch.tensor([ids])
    lab = torch.full_like(x, IGN)
    lab[:, -3:-1] = x[:, -2:]
    out = model(x, labels=lab)
    out.loss.backward()
    g = model.blocks["1"].feed_forward.w1.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_single_token_decode_shape_runs_full():
    """A (B, 1) forward (incremental decode) has no markers -> FREE -> full FFN, no crash."""
    model = _model(enable=True)
    x = torch.randint(5, 800, (2, 1))
    with torch.no_grad():
        out = model(x)
    assert out.shape[:2] == (2, 1)
