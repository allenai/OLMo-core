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

This script gives each marker a distinct, in-distribution embedding: the mean of the trained rows
plus small per-token noise (the standard initialization for newly-added tokens). Marker ids are never
loss targets (the label mask covers only the answer span), so only the input embedding matters; we
keep the norm in the range of the other special tokens so a tied output head cannot be nudged into
emitting them.

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="source model_and_optim distcp dir")
    ap.add_argument("--out", required=True, help="destination dir (a model_and_optim/ is written under it)")
    ap.add_argument("--model-size", default="0.6B", choices=["0.6B", "4B"])
    ap.add_argument("--seed", type=int, default=34521)
    args = ap.parse_args()

    factory = {"0.6B": TransformerConfig.qwen3_0_6B, "4B": TransformerConfig.qwen3_4B}[args.model_size]
    model = factory(vocab_size=TokenizerConfig.qwen3().padded_vocab_size()).build(init_device="cpu")
    load_model_and_optim_state(args.base, model)

    emb = model.embeddings.weight.data
    trained = emb[:REAL_VOCAB_SIZE].float()
    mean = trained.mean(dim=0)
    # Match the scale of the *trained* special tokens rather than the corpus mean, so a tied output
    # head does not gain a large logit for a token that should never be generated.
    target_norm = emb[151644].float().norm()  # <|im_start|>: a genuinely-trained special token

    g = torch.Generator().manual_seed(args.seed)
    markers = {
        DOC_START_ID: "doc_start",
        DOC_END_ID: "doc_end",
        LANDMARK_TOKEN_ID: "landmark",
        PAD_TOKEN_ID: "pad",
    }
    print(f"trained-row mean norm={trained.norm(dim=-1).mean():.4f}  target_norm={target_norm:.4f}")
    for tid, name in markers.items():
        before = emb[tid].float().clone()
        noise = torch.randn(emb.shape[1], generator=g) * trained.std()
        vec = mean + noise
        vec = vec / vec.norm() * target_norm
        emb[tid] = vec.to(emb.dtype)
        print(f"  {tid} {name:10s} norm {before.norm():.4f} -> {emb[tid].float().norm():.4f}")

    # The whole point is that the markers become mutually distinguishable -- assert it.
    ids = list(markers)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            cos = torch.nn.functional.cosine_similarity(
                emb[ids[i]].float()[None], emb[ids[j]].float()[None]
            ).item()
            assert abs(cos) < 0.5, f"markers {ids[i]}/{ids[j]} still near-identical (cos={cos:.4f})"
    print("all marker pairs are now distinguishable (|cos| < 0.5)")

    save_model_and_optim_state(f"{args.out}/model_and_optim", model)
    print(f"wrote fixed base -> {args.out}/model_and_optim")


if __name__ == "__main__":
    main()
