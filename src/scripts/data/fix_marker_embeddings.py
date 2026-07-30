"""
Repair the degenerate embeddings of the reserved marker tokens used by the document-chunk / landmark
data path.

Qwen3 never trains its unused special-token rows: ``<|box_start|>`` (151648) and ``<|box_end|>``
(151649) -- the document markers -- keep their shared initialization, so their embeddings are
*bit-identical* (cosine similarity 1.0000). The same holds for the landmark (151860) and pad (151863)
rows, which live past the real vocab in the padded region of the embedding matrix.

The consequence is that the model cannot tell an "open document" marker from a "close document" one:
every marker is the same out-of-distribution vector. The chunk_ids built from the token *ids* are
still correct, so the attention mask is right -- but the model's *perception* of the document
structure is destroyed. Empirically a Qwen3-0.6B trains to CE ~0.004 on a 100-claim contradiction
shard with no markers, and plateaus at CE ~0.79 (chance-level f1) on the byte-identical shard *with*
markers.

This script gives each marker a distinct, in-distribution embedding by **copying the row of a real,
trained delimiter token** (``«``, ``»``, ...) and adding a small perturbation so the markers can still
specialize. Marker ids are never loss targets (the label mask covers only the answer span), so only
the input embedding matters.

.. warning::
   **Fixing the cosine is not enough -- the norm matters just as much.** The original version of this
   script built each marker as ``mean(trained rows) + noise`` renormed to the norm of ``<|im_start|>``.
   That made the markers mutually distinguishable (cos ~0.01) but left them at norm **0.396** against a
   trained-token median of **1.415** -- roughly 1/3.6 of a real token, pointing in a meaningless
   "average token" direction. RMSNorm rescales every position to the same RMS, so such a row is
   amplified into a full-strength *noise* vector at each marker position. On the leak-free
   document-chunked shards (where a marker precedes every ``Claim N:`` label) training then flatlines
   at CE ~0.79 for **every** attention mask -- including plain causal -- i.e. an unrestricted model
   cannot even memorize the data. Deleting the markers, or giving them trained donor rows, restores
   learning (CE 0.058 vs 0.79 at 375 steps, single-variable). The "tied output head" rationale for the
   small norm does not apply: Qwen3-0.6B's ``lm_head`` is **not** tied to the embeddings here.
   See ``records/n100-chunked-marker-position-bug.md``.

Usage::

    python src/scripts/data/fix_marker_embeddings.py \\
        --base /path/to/base/model_and_optim --out /path/to/fixed --model-size 0.6B
"""

import argparse

import torch

from olmo_core.data import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_model_and_optim_state
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.data.document_chunk_landmark import (  # canonical ids -- never retype
    DOC_END_ID,
    DOC_START_ID,
    EOS_TOKEN_ID,
    LANDMARK_TOKEN_ID,
    PAD_TOKEN_ID,
    REAL_VOCAB_SIZE,
)

# Reserved ids inserted by the landmark instance sources (past the real vocab; see convert_*_to_sft.py).

# The real Qwen3 vocab ends here; rows at or beyond this index are untrained padding.


#: Trained delimiter tokens whose rows seed each marker. Delimiters are the right donors: the model
#: already reads them as "a boundary is here", which is exactly the marker's job.
DONOR_TOKENS = {
    DOC_START_ID: "«",
    DOC_END_ID: "»",
    LANDMARK_TOKEN_ID: "§",
    PAD_TOKEN_ID: "¶",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="source model_and_optim distcp dir")
    ap.add_argument("--out", required=True, help="destination dir (a model_and_optim/ is written under it)")
    ap.add_argument("--model-size", default="0.6B", choices=["0.6B", "4B"])
    ap.add_argument("--seed", type=int, default=34521)
    ap.add_argument(
        "--jitter",
        type=float,
        default=0.1,
        help="per-marker noise added to the donor row, as a fraction of the trained-row std, so the "
        "markers can specialize away from their donors during training",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    factory = {"0.6B": TransformerConfig.qwen3_0_6B, "4B": TransformerConfig.qwen3_4B}[args.model_size]
    model = factory(vocab_size=TokenizerConfig.qwen3().padded_vocab_size()).build(init_device="cpu")
    load_model_and_optim_state(args.base, model)

    emb = model.embeddings.weight.data
    trained = emb[:REAL_VOCAB_SIZE].float()
    median_norm = trained.norm(dim=-1).median()
    std = trained.std()

    g = torch.Generator().manual_seed(args.seed)
    names = {
        DOC_START_ID: "doc_start",
        DOC_END_ID: "doc_end",
        LANDMARK_TOKEN_ID: "landmark",
        PAD_TOKEN_ID: "pad",
    }
    print(f"trained-row median norm={median_norm:.4f}  (markers must land near this, NOT far below it)")
    for tid, name in names.items():
        donor_str = DONOR_TOKENS[tid]
        donor_ids = tok.encode(donor_str, add_special_tokens=False)
        if len(donor_ids) != 1 or donor_ids[0] >= REAL_VOCAB_SIZE or donor_ids[0] == EOS_TOKEN_ID:
            raise SystemExit(f"donor {donor_str!r} for {name} is not a single trained token: {donor_ids}")
        donor = donor_ids[0]
        before = emb[tid].float().norm()
        vec = emb[donor].float() + torch.randn(emb.shape[1], generator=g) * (std * args.jitter)
        emb[tid] = vec.to(emb.dtype)
        print(
            f"  {tid} {name:10s} <- {donor_str!r} (id {donor})   norm {before:.4f} -> "
            f"{emb[tid].float().norm():.4f}"
        )

    # 1) The markers must be mutually distinguishable (the original bug: all were bit-identical).
    ids = list(names)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            cos = torch.nn.functional.cosine_similarity(
                emb[ids[i]].float()[None], emb[ids[j]].float()[None]
            ).item()
            assert abs(cos) < 0.9, f"markers {ids[i]}/{ids[j]} too similar (cos={cos:.4f})"
    # 2) ...AND in-distribution in scale. A marker at a small fraction of a real token's norm is
    #    amplified by RMSNorm into full-strength noise, which silently destroys training on any
    #    marker-dense shard (see the module docstring).
    for tid, name in names.items():
        ratio = (emb[tid].float().norm() / median_norm).item()
        assert 0.5 < ratio < 2.0, (
            f"marker {name} norm is {ratio:.2f}x the trained-row median -- out of distribution. "
            "This is the exact failure the donor-row init exists to prevent."
        )
    print("markers are distinguishable AND in-distribution in norm")

    save_model_and_optim_state(f"{args.out}/model_and_optim", model)
    print(f"wrote fixed base -> {args.out}/model_and_optim")


if __name__ == "__main__":
    main()
