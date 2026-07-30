"""Unit test for scripts/lib/olmo_gold_grad_mask.py (pure-torch, no olmo / no CUDA).

We build a tiny multi-layer causal transformer that applies the library's exact core
op -- ``detach_kv`` -- inside each attention block, and the library's
``build_gold_key_mask`` policy on a marker-wrapped fixture. Then we prove, on the
real autograd graph, that the mechanism does *exactly* what we expect:

  1. FORWARD IDENTITY -- logits are bit-identical with vs without the K/V detach
     (``detach()`` preserves values; only the backward graph changes).
  2. SEVERANCE (masked loss) -- with the detach and the loss placed only on the
     answer tokens (the SFT setup), the per-position input-embedding gradient at
     **distractor-document** positions is *exactly zero*, while gold-document,
     instruction and answer positions get nonzero gradient.
  3. CONTROL -- without the detach, distractor positions DO receive gradient (so the
     test is actually exercising the mechanism, not a degenerate graph).
  4. Q-PATH NUANCE (full LM loss) -- if the loss is instead placed on *every*
     position (including distractors' own next-token prediction), distractor
     positions get nonzero gradient via their query path (``W_q`` is not detached).
     This documents *why* the guarantee requires loss masking to the supervised
     tokens, matching the contradiction SFT setup.

Run on a GPU node (login nodes have no python env), e.g. via the smoke sbatch, or:
    python -m pytest scripts/tests/test_gold_grad_mask.py
    python scripts/tests/test_gold_grad_mask.py      # standalone, prints PASS/FAIL
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from corpus_reasoning.lib.chunked_attention import FREE_CHUNK_ID
from corpus_reasoning.lib.olmo_flex_attention import build_roles
from corpus_reasoning.lib.olmo_gold_grad_mask import build_gold_key_mask, detach_kv

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


class ToyBlock(nn.Module):
    """Pre-norm causal MHA + MLP. K/V are run through ``detach_kv`` (the library op)
    when a gold mask is supplied, mirroring olmo-core's ``Attention.sdpa`` seam."""

    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.wo = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.f1 = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.f2 = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x, gold_mask):
        B, T, _ = x.shape
        h = self.n1(x)
        # (B, T, n_heads, head_dim) -- the layout detach_kv / olmo sdpa expect.
        q = self.wq(h).view(B, T, N_HEADS, HEAD_DIM)
        k = self.wk(h).view(B, T, N_HEADS, HEAD_DIM)
        v = self.wv(h).view(B, T, N_HEADS, HEAD_DIM)
        if gold_mask is not None:
            k, v = detach_kv(k, v, gold_mask)
        # -> (B, n_heads, T, head_dim) for F.sdpa
        att = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
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

    def forward(self, input_ids, gold_mask=None, return_h0=False):
        h0 = self.embed(input_ids)
        if return_h0:
            h0.retain_grad()  # so we can read per-position input gradient
        x = h0
        for blk in self.blocks:
            x = blk(x, gold_mask)
        logits = self.head(x)
        return (logits, h0) if return_h0 else logits


def _model():
    torch.manual_seed(0)
    return ToyLM().double()  # float64: exact-zero / forward-identity checks are crisp


def _gold_mask():
    return build_gold_key_mask(
        INPUT, doc_start_id=DS, doc_end_id=DE, eos_id=EOS, gold_doc_indices=GOLD_DOCS)


def _pos_grad_norms(model, gold_mask, label_positions):
    """Run fwd/bwd with CE loss on ``label_positions`` only; return (S,) per-position
    L2 norm of the input-embedding gradient."""
    model.zero_grad(set_to_none=True)
    logits, h0 = model(INPUT, gold_mask=gold_mask, return_h0=True)
    targets = (INPUT.roll(-1, dims=1)).clamp(0, VOCAB - 1)
    lp = torch.tensor(label_positions)
    loss = F.cross_entropy(logits[0, lp], targets[0, lp])
    loss.backward()
    return h0.grad[0].norm(dim=-1)  # (S,)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_gold_mask_layout():
    """build_gold_key_mask: FREE + gold doc => True; distractor docs + pad => False."""
    roles = build_roles(INPUT, DS, DE, EOS, mode="chunked")[0].tolist()
    assert roles[DOC1[0]] == 1 and roles[DOC0[0]] == 0 and roles[DOC2[0]] == 2
    gm = _gold_mask()[0]
    assert gm[PREFIX].all() and gm[ANSWER].all()          # free tokens learn
    assert gm[torch.tensor(DOC1)].all()                   # gold doc learns
    assert not gm[torch.tensor(DOC0)].any()               # distractor detached
    assert not gm[torch.tensor(DOC2)].any()               # distractor detached
    # build_roles treats the single real terminating eos (pos 19) as FREE (kept); only
    # the trailing pad region (>= first-eos + 1) is detached. Matches the flex test.
    assert gm[19]                                          # real eos is FREE -> kept
    assert not gm[20] and not gm[21]                       # pad detached


def test_forward_identity():
    """Detaching K/V must not change the forward output at all."""
    model = _model()
    with torch.no_grad():
        ref = model(INPUT, gold_mask=None)
        got = model(INPUT, gold_mask=_gold_mask())
    assert torch.equal(ref, got), \
        f"forward changed by detach (max|Δ|={(ref - got).abs().max().item():.2e})"


def test_severance_masked_loss():
    """Masked (answer-only) loss + detach => distractor input-grad is exactly zero,
    gold/instruction/answer grads are nonzero."""
    model = _model()
    g = _pos_grad_norms(model, _gold_mask(), label_positions=ANSWER)
    distractor = torch.tensor(DOC0 + DOC2)
    assert g[distractor].max().item() == 0.0, \
        f"distractor grad not zero: max={g[distractor].max().item():.2e}"
    assert g[torch.tensor(DOC1)].min().item() > 0, "gold doc got no gradient"
    assert g[torch.tensor(PREFIX + ANSWER)].min().item() > 0, "free tokens got no gradient"


def test_control_no_mask_distractors_get_grad():
    """Without the detach, the same distractor positions DO receive gradient -- proves
    the severance test isn't passing on a degenerate (already-zero) graph."""
    model = _model()
    g = _pos_grad_norms(model, gold_mask=None, label_positions=ANSWER)
    distractor = torch.tensor(DOC0 + DOC2)
    assert g[distractor].min().item() > 0, \
        "distractors had no grad even WITHOUT masking -- fixture/graph is degenerate"


def test_qpath_nuance_full_lm_loss():
    """Full (every-position) LM loss + detach => distractor positions get nonzero grad
    via their own query path (W_q is not detached). Documents why the guarantee needs
    the loss masked to supervised tokens."""
    model = _model()
    g = _pos_grad_norms(model, _gold_mask(), label_positions=list(range(S - 1)))
    distractor = torch.tensor(DOC0 + DOC2)
    assert g[distractor].min().item() > 0, \
        "expected nonzero distractor grad via query path under full LM loss"


def main():
    tests = [
        test_gold_mask_layout,
        test_forward_identity,
        test_severance_masked_loss,
        test_control_no_mask_distractors_get_grad,
        test_qpath_nuance_full_lm_loss,
    ]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    # A small quantitative readout for the headline property.
    model = _model()
    g = _pos_grad_norms(model, _gold_mask(), label_positions=ANSWER)
    print("\nper-position input-grad norm (masked answer loss, with detach):")
    print(f"  distractor doc0 max = {g[torch.tensor(DOC0)].max().item():.3e}")
    print(f"  distractor doc2 max = {g[torch.tensor(DOC2)].max().item():.3e}")
    print(f"  GOLD doc1      mean = {g[torch.tensor(DOC1)].mean().item():.3e}")
    print(f"  answer tokens  mean = {g[torch.tensor(ANSWER)].mean().item():.3e}")
    print("\nTEST PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nTEST FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
