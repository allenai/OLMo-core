"""Real-data smoke for gold-grad masking on the contradiction task (n=20, k=3).

Validates the whole chain on *real* examples and the real 20-document structure,
before any training:

  1. render + wrap + tokenize a few real rows from the n=20 contradiction train file
     EXACTLY as scripts/data/tokenize_unified_for_olmo does (build_prompt -> box-marker
     wrap_documents -> tokenize prompt/completion);
  2. derive the gold-document mask from the row's real ``gold_doc_indices`` (1-indexed
     "Claim N" display ids -> 0-based wrapped-doc/chunk index N-1), and CHECK the
     mapping is sound (build_roles recovers exactly 20 documents per row, and every
     gold chunk index is in range);
  3. build a real (random-init) olmo-core Qwen3-0.6B, install gold-grad masking, and
     on each row verify, with the loss on the completion tokens only (the SFT setup):
       - FORWARD IDENTITY: logits are bit-identical to the un-patched forward;
       - SEVERANCE: per-position input-embedding gradient at *distractor*-document
         tokens is exactly zero, while *gold*-document and completion tokens get
         nonzero gradient.

Needs the train JSONL + network for the tokenizer/config; no checkpoint. Run via
jobs/smoke_olmo_gold_grad_contradiction.sh.
"""

import json
import sys

import torch

from corpus_reasoning.lib import chunked_attention as _ca
from corpus_reasoning.lib.chunked_attention import FREE_CHUNK_ID, PAD_CHUNK_ID
from corpus_reasoning.lib.data_format import build_prompt
from corpus_reasoning.lib.olmo_flex_attention import build_roles
from corpus_reasoning.lib.olmo_gold_grad_mask import build_gold_key_mask, install_gold_grad_mask
from corpus_reasoning.lib.olmo_models import build_transformer_config, resolve_olmo_model
from corpus_reasoning.data.tokenize_unified_for_olmo import _single_token_id

BASE = "Qwen/Qwen3-0.6B-Base"
DATA = "data/contradiction_train_pubmed_both_n20_k3.jsonl"
MARK_S, MARK_E = "<|box_start|>", "<|box_end|>"
N_ROWS = 3            # examples to check
EXPECT_DOCS = 20      # n=20


def _build_model():
    spec = resolve_olmo_model(BASE)
    model = build_transformer_config(spec).build(init_device="cuda")
    model.init_weights(device=torch.device("cuda"))
    return model


def _tokenize_row(tok, ex, eos):
    """Render + wrap + tokenize one row exactly like tokenize_unified_for_olmo."""
    prompt, output = build_prompt(
        ex, task="contradiction", query_position="both", use_titles=True,
        use_alpaca=True, cot_mode="label")
    prompt = _ca.wrap_documents(prompt)          # box-marker wrap (markers set below)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    o_ids = tok(output, add_special_tokens=False).input_ids + [eos]
    return p_ids, o_ids


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    eos = tok.eos_token_id
    DS = _single_token_id(tok, MARK_S)
    DE = _single_token_id(tok, MARK_E)
    # Point chunked_attention's wrap markers at the box tokens (same as the tokenizer).
    _ca.DOC_START, _ca.DOC_END = MARK_S, MARK_E
    print(f"markers: box_start={DS} box_end={DE} eos={eos}")

    rows = []
    with open(DATA) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= N_ROWS:
                break

    # Per-example prepared tensors + gold chunk sets + region indices.
    prepared = []
    for r, ex in enumerate(rows):
        p_ids, o_ids = _tokenize_row(tok, ex, eos)
        ids = p_ids + o_ids
        input_ids = torch.tensor([ids], dtype=torch.long).cuda()
        # gold_doc_indices are 1-indexed Claim ids -> 0-based chunk index (claim - 1).
        gold = {x - 1 for pair in ex["gold_doc_indices"] for x in pair}

        roles = build_roles(input_ids.cpu(), DS, DE, eos, mode="chunked")[0]
        n_docs = int(roles.max().item()) + 1
        assert n_docs == EXPECT_DOCS, \
            f"row {r}: build_roles found {n_docs} docs, expected {EXPECT_DOCS}"
        assert max(gold) < n_docs, f"row {r}: gold chunk {max(gold)} >= n_docs {n_docs}"

        comp = list(range(len(p_ids), len(ids)))           # completion (loss) region
        gold_pos = torch.tensor(
            [i for i, c in enumerate(roles.tolist()) if c in gold])
        dist_pos = torch.tensor(
            [i for i, c in enumerate(roles.tolist())
             if c >= 0 and c not in gold])                 # distractor-doc tokens
        prepared.append(dict(input_ids=input_ids, gold=gold, comp=comp,
                             gold_pos=gold_pos, dist_pos=dist_pos, S=len(ids)))
        print(f"row {r}: len={len(ids)} gold_chunks={sorted(gold)} "
              f"#gold_tok={len(gold_pos)} #distractor_tok={len(dist_pos)}")

    model = _build_model()

    # Stock (un-patched) logits per row, kept on CPU for the identity check.
    stock = []
    with torch.no_grad():
        for p in prepared:
            out = model(input_ids=p["input_ids"])
            stock.append((out if isinstance(out, torch.Tensor) else out.logits).float().cpu())

    # Install gold-grad masking. The gold set differs per row; drive it via a closure.
    state = {"gold": set()}
    gm_fn = lambda x: build_gold_key_mask(
        x, doc_start_id=DS, doc_end_id=DE, eos_id=eos, gold_doc_indices=state["gold"])
    holder = install_gold_grad_mask(model, gm_fn)
    assert holder.n_patched > 0, "patched 0 attention modules"
    print(f"installed: patched {holder.n_patched} attention modules")

    cap = {}

    def emb_hook(mod, inp, out):  # MUST return None (a returned value replaces the output)
        out.retain_grad()
        cap["h0"] = out

    model.embeddings.register_forward_hook(emb_hook)

    for r, p in enumerate(prepared):
        state["gold"] = p["gold"]
        model.zero_grad(set_to_none=True)
        out = model(input_ids=p["input_ids"])
        logits = out if isinstance(out, torch.Tensor) else out.logits

        delta = (logits.float().cpu() - stock[r]).abs().max().item()
        assert delta == 0.0, f"row {r}: detach changed forward (max|Δ|={delta:.3e})"

        targets = p["input_ids"].roll(-1, dims=1)
        cp = torch.tensor(p["comp"], device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits[0, cp].float(), targets[0, cp])
        loss.backward()

        g = cap["h0"].grad[0].float().norm(dim=-1)         # (S,) per-position grad norm
        dmax = g[p["dist_pos"]].max().item()
        gmin = g[p["gold_pos"]].min().item()
        cmin = g[cp].min().item()
        assert dmax == 0.0, f"row {r}: distractor grad not zero (max={dmax:.3e})"
        assert gmin > 0, f"row {r}: a gold-doc token got zero gradient"
        assert cmin > 0, f"row {r}: a completion token got zero gradient"
        print(f"row {r}: forward Δ={delta:.0e}  distractor max={dmax:.0e}  "
              f"gold min={gmin:.3e}  completion min={cmin:.3e}  loss={loss.item():.3f}")

    print("\nSMOKE PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nSMOKE FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
