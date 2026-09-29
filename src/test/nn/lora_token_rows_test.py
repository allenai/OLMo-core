"""Trainable special-token rows under LoRA: identity at init, gradient only on the delta,
head and input both adjusted, and a clean fold on merge."""

import torch
import torch.nn as nn

from olmo_core.nn.embedding import SplitVocabEmbedding
from olmo_core.nn.lora import (
    TOKEN_DELTA_IDS_NAME,
    TOKEN_DELTA_NAME,
    apply_trainable_token_rows,
    fold_token_rows,
)

V, X, D = 50, 4, 8
ROWS = [40, 41]


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_out = nn.Linear(D, V, bias=False)


class _LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = SplitVocabEmbedding(V, X, D)
        nn.init.normal_(self.embeddings.weight)
        nn.init.normal_(self.embeddings.extra_weight)
        self.lm_head = _Head()
        self.lm_head.w_out.weight = self.embeddings.weight  # tied, as Molmo2-4B

    def forward(self, ids):
        h = self.embeddings(ids)
        return self.lm_head.w_out(h)


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm = _LM()

    def forward(self, ids):
        return self.lm(ids)


def _freeze(m):
    for p in m.parameters():
        p.requires_grad_(False)


def test_identity_at_init_then_only_delta_trains():
    torch.manual_seed(0)
    m = _Wrapper()
    _freeze(m)
    ids = torch.tensor([[1, 40, 7, 41]])
    ref = m(ids).detach().clone()
    names = apply_trainable_token_rows(m, ROWS)
    assert names == [f"lm.embeddings.{TOKEN_DELTA_NAME}", f"lm.embeddings.{TOKEN_DELTA_IDS_NAME}"]
    out = m(ids)
    torch.testing.assert_close(out, ref)  # zero delta -> bit-identical
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable == [f"lm.embeddings.{TOKEN_DELTA_NAME}"]
    out[..., 40].sum().backward()
    d = m.lm.embeddings.token_delta
    assert d.grad is not None and d.grad.abs().sum() > 0
    assert m.lm.embeddings.weight.grad is None


def test_head_and_input_both_see_the_delta():
    torch.manual_seed(1)
    m = _Wrapper()
    _freeze(m)
    apply_trainable_token_rows(m, ROWS)
    emb = m.lm.embeddings
    with torch.no_grad():
        emb.token_delta.copy_(torch.randn_like(emb.token_delta))
    ids = torch.tensor([[40, 3]])
    h = emb(ids)
    # input side: row 40 shifted by delta[0], ordinary row untouched
    torch.testing.assert_close(h[0, 0], emb.weight[40] + emb.token_delta[0])
    torch.testing.assert_close(h[0, 1], emb.weight[3])
    # head side: logit columns 40/41 shifted by h @ delta, others unchanged
    x = torch.randn(1, 2, D)
    logits = m.lm.lm_head.w_out(x)
    base = x @ emb.weight.T
    torch.testing.assert_close(logits[..., :40], base[..., :40])
    torch.testing.assert_close(logits[..., 40], base[..., 40] + x @ emb.token_delta[0])
    torch.testing.assert_close(logits[..., 41], base[..., 41] + x @ emb.token_delta[1])


def test_fold_token_rows_into_state_dict():
    torch.manual_seed(2)
    m = _Wrapper()
    _freeze(m)
    apply_trainable_token_rows(m, ROWS)
    with torch.no_grad():
        m.lm.embeddings.token_delta.copy_(torch.randn_like(m.lm.embeddings.token_delta))
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    assert (
        f"lm.embeddings.{TOKEN_DELTA_NAME}" in sd and f"lm.embeddings.{TOKEN_DELTA_IDS_NAME}" in sd
    )
    expected = sd["lm.embeddings.weight"].clone()
    expected[ROWS] += sd[f"lm.embeddings.{TOKEN_DELTA_NAME}"]
    n = fold_token_rows(sd, weight_key="lm.embeddings.weight")
    assert n == 2
    assert not any(k.endswith((TOKEN_DELTA_NAME, TOKEN_DELTA_IDS_NAME)) for k in sd)
    torch.testing.assert_close(sd["lm.embeddings.weight"], expected)
    # folded table reproduces the adapted model's logits exactly
    x = torch.randn(1, 3, D)
    adapted = m.lm.lm_head.w_out(x)
    plain = x @ sd["lm.embeddings.weight"].T
    torch.testing.assert_close(adapted, plain)
