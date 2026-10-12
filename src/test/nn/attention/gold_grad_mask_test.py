"""
CPU unit tests for :mod:`olmo_core.nn.attention.gold_grad_mask` (pure-torch, no GPU).

A tiny multi-layer causal transformer applies the library's exact core op -- ``detach_kv`` -- inside
each attention block, on a marker-wrapped fixture, and we prove on the real autograd graph that:

  1. FORWARD IDENTITY -- logits are bit-identical with vs without the K/V detach.
  2. SEVERANCE (masked loss) -- with the detach and the loss placed only on the answer tokens (the SFT
     setup), the per-position input-embedding gradient at **non-selected** document positions is
     *exactly zero*, while selected-document, instruction and answer positions get nonzero gradient.
  3. CONTROL -- without the detach, the same positions DO receive gradient (the fixture isn't
     degenerate).
  4. Q-PATH NUANCE (full LM loss) -- under an every-position loss, non-selected positions get nonzero
     gradient via their own query path (``W_q`` is not detached), documenting why the O(1) guarantee
     needs the loss masked to the supervised tokens (which contradiction SFT does).

Plus policy tests for ``gold_chunks_from_gold_doc_indices`` / ``select_keep_docs`` and an end-to-end
check of ``make_fingerprint_gold_mask_fn``.
"""

import random

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.nn.attention.chunked_mask import (
    FREE_CHUNK_ID,
    build_chunk_ids_from_tokens,
)
from olmo_core.nn.attention.gold_grad_mask import (
    content_fingerprint,
    detach_kv,
    gold_chunks_from_gold_doc_indices,
    make_fingerprint_gold_mask_fn,
    select_keep_docs,
)

# ---- fixture: marker-wrapped ids with 3 docs; doc 1 is the ground-truth ----
EOS, DS, DE = 1000, 1001, 1002
# [pref pref] [DS d d d DE] [DS d d d DE] [DS d d d DE] [ans ans] EOS pad pad
IDS = [50, 51,
       DS, 60, 61, 62, DE,        # doc 0  (distractor)  positions 2..6
       DS, 70, 71, 72, DE,        # doc 1  (GOLD)        positions 7..11
       DS, 80, 81, 82, DE,        # doc 2  (distractor)  positions 12..16
       90, 91,                    # answer / free        positions 17..18
       EOS, EOS, EOS]             # eos + pad            positions 19..21
INPUT = torch.tensor([IDS], dtype=torch.long)
S = len(IDS)
GOLD_DOCS = {1}
DOC0 = list(range(2, 7))
DOC1 = list(range(7, 12))
DOC2 = list(range(12, 17))
PREFIX = [0, 1]
ANSWER = [17, 18]

D_MODEL, N_HEADS, N_LAYERS, VOCAB = 32, 4, 3, 1100
HEAD_DIM = D_MODEL // N_HEADS


def _keep_mask(gold_docs):
    """Build the (B, S) keep mask directly: FREE tokens + the given gold docs (mirrors the fn core)."""
    roles = build_chunk_ids_from_tokens(INPUT, doc_start_id=DS, doc_end_id=DE, eos_id=EOS)
    keep = roles == FREE_CHUNK_ID
    for gi in gold_docs:
        keep |= roles == gi
    return keep.to(torch.bool)


class ToyBlock(nn.Module):
    """Pre-norm causal MHA + MLP; K/V run through ``detach_kv`` when a keep mask is supplied."""

    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.wo = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.f1 = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.f2 = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x, keep_mask):
        B, T, _ = x.shape
        h = self.n1(x)
        # (B, T, n_heads, head_dim) -- the layout detach_kv / olmo sdpa expect.
        q = self.wq(h).view(B, T, N_HEADS, HEAD_DIM)
        k = self.wk(h).view(B, T, N_HEADS, HEAD_DIM)
        v = self.wv(h).view(B, T, N_HEADS, HEAD_DIM)
        if keep_mask is not None:
            k, v = detach_kv(k, v, keep_mask)
        att = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        att = att.transpose(1, 2).reshape(B, T, D_MODEL)
        x = x + self.wo(att)
        x = x + self.f2(F.gelu(self.f1(self.n2(x))))
        return x


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList(ToyBlock() for _ in range(N_LAYERS))
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, input_ids, keep_mask=None, return_h0=False):
        h0 = self.embed(input_ids)
        if return_h0:
            h0.retain_grad()
        x = h0
        for blk in self.blocks:
            x = blk(x, keep_mask)
        logits = self.head(x)
        return (logits, h0) if return_h0 else logits


def _model():
    torch.manual_seed(0)
    return ToyLM().double()  # float64: exact-zero / forward-identity checks are crisp


def _pos_grad_norms(model, keep_mask, label_positions):
    """CE loss on ``label_positions`` only; return (S,) per-position L2 norm of input-emb gradient."""
    model.zero_grad(set_to_none=True)
    logits, h0 = model(INPUT, keep_mask=keep_mask, return_h0=True)
    targets = (INPUT.roll(-1, dims=1)).clamp(0, VOCAB - 1)
    lp = torch.tensor(label_positions)
    loss = F.cross_entropy(logits[0, lp], targets[0, lp])
    loss.backward()
    return h0.grad[0].norm(dim=-1)


# --------------------------------------------------------------------------- #
# policy tests
# --------------------------------------------------------------------------- #


def test_gold_chunks_from_indices():
    assert gold_chunks_from_gold_doc_indices([[2, 8], [11, 17], [12, 13]]) == {1, 7, 10, 16, 11, 12}
    assert gold_chunks_from_gold_doc_indices([3, 5]) == {2, 4}


def test_select_keep_docs_gold_plus_random():
    present = list(range(20))
    gold = {1, 7, 10}
    rng = random.Random("x")
    keep = select_keep_docs(present, gold, n_random=2, mode="gold_plus_random", rng=rng)
    assert gold <= keep  # all gold retained
    assert len(keep) == len(gold) + 2  # + exactly 2 random
    assert (keep - gold).isdisjoint(gold)  # the 2 extras are non-gold


def test_select_keep_docs_random_only():
    present = list(range(20))
    gold = {1, 7, 10}
    rng = random.Random("y")
    keep = select_keep_docs(present, gold, n_random=2, mode="random_only", rng=rng)
    # same sparsity as gold_plus_random (|gold| + n_random), but gold NOT forced.
    assert len(keep) == len(gold) + 2
    assert keep <= set(present)


def test_select_keep_docs_gold_subsample():
    """O(1) AND base-rate preserving: a CONSTANT n_gold positives + n_random negatives."""
    present = list(range(100))                       # n=100: the setting O(1) actually pays off in
    gold = {1, 7, 10, 12, 15, 18}
    for seed in range(50):
        keep = select_keep_docs(
            present, gold, n_random=15, mode="gold_subsample", rng=random.Random(seed), n_gold=1
        )
        assert len(keep & gold) == 1                 # exactly n_gold positives, NOT all 6
        assert len(keep - gold) == 15                # ...and n_random negatives
        assert len(keep) == 16                       # constant in N -> genuinely O(1)


def test_select_keep_docs_random_nongold_is_gold_free():
    """The STRICT control: same sparsity, but ZERO gold docs kept (unlike ``random_only``)."""
    present = list(range(20))
    gold = {1, 7, 10, 12, 15, 18}  # ~6 gold of 20, as in the real n20 contradiction data
    for seed in range(50):
        keep = select_keep_docs(
            present, gold, n_random=2, mode="random_nongold", rng=random.Random(seed)
        )
        assert len(keep) == len(gold) + 2       # same sparsity as gold_plus_random / random_only
        assert not (keep & gold)                # ...but NEVER any gold gradient


def test_random_only_leaks_gold_by_chance():
    """Documents the confound that motivates ``random_nongold``.

    ``random_only`` draws from ALL docs, so it keeps gold docs BY CHANCE (~|gold|*n_keep/|present|).
    It is therefore a same-sparsity control, NOT a gold-free one, and ``gpr ~= random_only`` must not
    be read as "gold identity does not matter".
    """
    present = list(range(20))
    gold = {1, 7, 10, 12, 15, 18}
    kept_gold = [
        len(select_keep_docs(present, gold, n_random=2, mode="random_only", rng=random.Random(s)) & gold)
        for s in range(300)
    ]
    mean_gold = sum(kept_gold) / len(kept_gold)
    assert mean_gold > 1.5              # leaks real gold gradient (expectation 8*6/20 = 2.4)
    assert sum(g == 0 for g in kept_gold) / len(kept_gold) < 0.10   # rarely gold-free


def test_keep_mask_layout():
    roles = build_chunk_ids_from_tokens(INPUT, doc_start_id=DS, doc_end_id=DE, eos_id=EOS)[0].tolist()
    assert roles[DOC1[0]] == 1 and roles[DOC0[0]] == 0 and roles[DOC2[0]] == 2
    km = _keep_mask(GOLD_DOCS)[0]
    assert km[PREFIX].all() and km[ANSWER].all()          # free tokens learn
    assert km[torch.tensor(DOC1)].all()                   # gold doc learns
    assert not km[torch.tensor(DOC0)].any()               # distractor detached
    assert not km[torch.tensor(DOC2)].any()               # distractor detached
    assert km[19]                                         # real eos is FREE -> kept
    assert not km[20] and not km[21]                      # pad detached


# --------------------------------------------------------------------------- #
# mechanism tests
# --------------------------------------------------------------------------- #


def test_forward_identity():
    model = _model()
    with torch.no_grad():
        ref = model(INPUT, keep_mask=None)
        got = model(INPUT, keep_mask=_keep_mask(GOLD_DOCS))
    assert torch.equal(ref, got), f"forward changed (max|Δ|={(ref - got).abs().max().item():.2e})"


def test_severance_masked_loss():
    model = _model()
    g = _pos_grad_norms(model, _keep_mask(GOLD_DOCS), label_positions=ANSWER)
    distractor = torch.tensor(DOC0 + DOC2)
    assert g[distractor].max().item() == 0.0, f"distractor grad not zero: {g[distractor].max():.2e}"
    assert g[torch.tensor(DOC1)].min().item() > 0, "gold doc got no gradient"
    assert g[torch.tensor(PREFIX + ANSWER)].min().item() > 0, "free tokens got no gradient"


def test_control_no_mask_distractors_get_grad():
    model = _model()
    g = _pos_grad_norms(model, keep_mask=None, label_positions=ANSWER)
    distractor = torch.tensor(DOC0 + DOC2)
    assert g[distractor].min().item() > 0, "distractors had no grad even WITHOUT masking -- degenerate"


def test_qpath_nuance_full_lm_loss():
    model = _model()
    g = _pos_grad_norms(model, _keep_mask(GOLD_DOCS), label_positions=list(range(S - 1)))
    distractor = torch.tensor(DOC0 + DOC2)
    assert g[distractor].min().item() > 0, "expected nonzero distractor grad via query path"


def test_severance_with_one_random_kept():
    """Keeping gold doc 1 + random doc 0: doc0 now gets gradient, doc2 stays exactly zero."""
    model = _model()
    g = _pos_grad_norms(model, _keep_mask({1, 0}), label_positions=ANSWER)
    assert g[torch.tensor(DOC2)].max().item() == 0.0, "non-kept doc2 should be severed"
    assert g[torch.tensor(DOC0)].min().item() > 0, "kept random doc0 should get gradient"
    assert g[torch.tensor(DOC1)].min().item() > 0, "gold doc1 should get gradient"


# --------------------------------------------------------------------------- #
# fingerprint end-to-end
# --------------------------------------------------------------------------- #


def test_fingerprint_gold_mask_fn():
    # content = up to & incl. the single real eos (pos 0..19); pad excluded.
    fp = content_fingerprint(IDS[:20])
    # gold_doc_indices as 1-indexed "Claim N" pairs -> doc 1 gold (Claim 2). No random.
    table = {fp: sorted(gold_chunks_from_gold_doc_indices([[2, 2]]))}
    fn = make_fingerprint_gold_mask_fn(
        table, doc_start_id=DS, doc_end_id=DE, eos_id=EOS, n_random=0, mode="gold_plus_random"
    )
    km = fn(INPUT)[0]
    assert km[torch.tensor(DOC1)].all()          # gold kept
    assert not km[torch.tensor(DOC0)].any()      # distractor detached
    assert not km[torch.tensor(DOC2)].any()

    # An unknown fingerprint (e.g. warmup mock) -> all-True (no masking).
    other = torch.tensor([[50, 51, DS, 60, DE, 90, EOS]], dtype=torch.long)
    assert fn(other).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
