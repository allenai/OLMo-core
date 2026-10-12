"""
Bake zero-init soft-token projector keys into a copy of a Qwen3 base checkpoint.

The B1 soft-token arm's model has a ``pooled_projector`` submodule
(:class:`~olmo_core.nn.pooled_soft_token.PooledDocProjector`). Distributed-checkpoint loading is
driven by the destination model's keys, so training from a base that lacks the projector keys fails
at load. This writes a base copy WITH those keys (residual init: ``P(x) == x`` at step 0). Extra
keys in a checkpoint are ignored by loads into plain models, so the copy also works for the other
arms.

Usage (on the node holding the base, e.g. horton)::

    python research/token_routing/pooledkv_train/bake_projector_into_base.py \\
        --base /data/prasann/pooledkv_exp/q4b-dense-cpt-fixmark/model_and_optim \\
        --out  /data/prasann/pooledkv_exp/q4b-dense-cpt-fixmark-b1
"""

import argparse

from olmo_core.data import TokenizerConfig
from olmo_core.data.document_chunk_landmark import (
    DOC_END_ID,
    DOC_START_ID,
    EOS_TOKEN_ID,
    LANDMARK_TOKEN_ID,
)
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.nn.transformer import TransformerConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="source model_and_optim distcp dir")
    ap.add_argument("--out", required=True, help="destination dir (model_and_optim/ written under)")
    ap.add_argument("--model-size", default="4B", choices=["0.6B", "4B"])
    args = ap.parse_args()

    factory = {"0.6B": TransformerConfig.qwen3_0_6B, "4B": TransformerConfig.qwen3_4B}[
        args.model_size
    ]
    model = factory(vocab_size=TokenizerConfig.qwen3().padded_vocab_size()).build(init_device="cpu")
    load_model_and_optim_state(args.base, model)
    # Enable AFTER the load: creates + residual-inits the projector (P(x) == x at step 0).
    model.enable_pooled_soft_tokens(
        DOC_START_ID, DOC_END_ID, EOS_TOKEN_ID, placeholder_id=LANDMARK_TOKEN_ID
    )
    n_proj = sum(p.numel() for p in model.pooled_projector.parameters())
    print(f"projector: {n_proj / 1e6:.1f}M params, residual-init (P(x) == x)")
    save_model_and_optim_state(f"{args.out}/model_and_optim", model)
    print(f"wrote base with projector keys -> {args.out}/model_and_optim")


if __name__ == "__main__":
    main()
