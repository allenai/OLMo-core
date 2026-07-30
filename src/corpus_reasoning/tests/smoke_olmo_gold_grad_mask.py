"""GPU smoke for the gold-grad-mask monkeypatch on a real olmo-core model.

Builds a real (random-init) olmo-core Qwen3-0.6B on CUDA, synthesizes a tiny
marker-wrapped batch with 3 documents (doc 1 = ground truth), installs
``install_gold_grad_mask``, and runs forward + backward. Verifies on the *real*
attention modules:

  - install patches every Attention module (n_patched > 0) and the pre-hook sets the
    gold mask each forward;
  - FORWARD IDENTITY: logits with the K/V-detach installed are bit-identical to the
    stock forward (detach preserves values);
  - SEVERANCE: with the loss placed only on the answer tokens, the per-position
    input-embedding gradient at distractor-document positions is exactly zero, while
    gold-document and answer positions get nonzero gradient.

Needs no dataset / checkpoint / network. Run via jobs/smoke_olmo_gold_grad_mask.sh.
"""

import sys

import torch

from corpus_reasoning.lib.olmo_models import build_transformer_config, resolve_olmo_model
from corpus_reasoning.lib.olmo_gold_grad_mask import build_gold_key_mask, install_gold_grad_mask

BASE = "Qwen/Qwen3-0.6B-Base"
EOS, DS, DE = 151643, 1001, 1002  # real Qwen3 eos + synthetic marker ids
GOLD_DOCS = {1}

# Document spans in the synthesized row (see _synth_batch).
DOC0 = list(range(3, 9))      # distractor
DOC1 = list(range(9, 15))     # GOLD
DOC2 = list(range(15, 21))    # distractor
ANSWER = list(range(40, 46))


def _build_model():
    spec = resolve_olmo_model(BASE)
    cfg = build_transformer_config(spec)
    model = cfg.build(init_device="cuda")
    model.init_weights(device=torch.device("cuda"))
    return model


def _synth_batch():
    """[pref] <DS>doc0<DE> <DS>doc1<DE> <DS>doc2<DE> [filler] [answer] eos pad..."""
    torch.manual_seed(0)
    S = 64
    ids = torch.randint(10, 900, (1, S), dtype=torch.long)
    ids[0, 0:3] = torch.tensor([20, 21, 22])      # instruction prefix
    ids[0, 3] = DS;  ids[0, 8] = DE               # doc 0  -> positions 3..8
    ids[0, 9] = DS;  ids[0, 14] = DE              # doc 1  -> positions 9..14 (GOLD)
    ids[0, 15] = DS; ids[0, 20] = DE              # doc 2  -> positions 15..20
    ids[0, 46] = EOS                              # real terminator
    ids[0, 47:] = EOS                             # pad region
    return ids.cuda()


def _logits(model, ids):
    out = model(input_ids=ids)
    return out if isinstance(out, torch.Tensor) else out.logits


def main():
    model = _build_model()
    ids = _synth_batch()

    # Stock forward (no masking installed yet).
    with torch.no_grad():
        stock = _logits(model, ids).float()
    assert torch.isfinite(stock).all(), "stock forward non-finite"
    print(f"ok   stock forward: logits {tuple(stock.shape)} finite")

    gm_fn = lambda x: build_gold_key_mask(
        x, doc_start_id=DS, doc_end_id=DE, eos_id=EOS, gold_doc_indices=GOLD_DOCS)
    holder = install_gold_grad_mask(model, gm_fn)
    assert holder.n_patched > 0, "patched 0 attention modules"

    # Capture the per-position input-embedding gradient. olmo-core's Transformer holds
    # the token embedding under .embeddings; hook its output to retain grad.
    cap = {}

    def emb_hook(mod, inp, out):
        out.retain_grad()
        cap["h0"] = out

    h = model.embeddings.register_forward_hook(emb_hook)

    logits = _logits(model, ids)
    assert holder.gold_mask is not None, "pre-hook did not set gold_mask"

    # 1) forward identity vs stock
    delta = (logits.float() - stock).abs().max().item()
    assert delta == 0.0, f"detach changed the forward (max|Δ|={delta:.3e})"
    print(f"ok   forward identity: max|Δ vs stock| = {delta:.1e}")

    # 2) masked (answer-only) loss -> severance
    targets = ids.roll(-1, dims=1)
    lp = torch.tensor(ANSWER, device=ids.device)
    loss = torch.nn.functional.cross_entropy(logits[0, lp].float(), targets[0, lp])
    loss.backward()
    h.remove()

    g = cap["h0"].grad[0].float().norm(dim=-1)  # (S,)
    distractor = torch.tensor(DOC0 + DOC2, device=g.device)
    gmax = g[distractor].max().item()
    gold_min = g[torch.tensor(DOC1, device=g.device)].min().item()
    ans_min = g[lp].min().item()
    assert gmax == 0.0, f"distractor input-grad not zero: max={gmax:.3e}"
    assert gold_min > 0, "gold doc got no gradient"
    assert ans_min > 0, "answer tokens got no gradient"
    print(f"ok   severance: distractor max={gmax:.1e}  gold min={gold_min:.3e}  "
          f"answer min={ans_min:.3e}")
    print(f"ok   patched {holder.n_patched} attention modules")
    print("\nSMOKE PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nSMOKE FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
