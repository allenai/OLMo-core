"""
Bake nested-FFN-MoE router keys (and, optionally, an importance reordering of the FFN hidden
units) into a copy of a Qwen3 base checkpoint.

Two things happen here, and only the first is strictly required:

1. **Router + gain keys.** ``enable_nested_ffn_moe`` adds
   ``blocks.N.feed_forward.{_nffn_router.w.*,_nffn_gain}`` per routed block. Distributed-checkpoint
   loading is driven by the *destination* model's keys, so training from a base that lacks them
   fails at load -- the same reason ``bake_projector_into_base.py`` exists for the soft-token
   projector. The router is initialized to pick the full rung with probability ~1, so the baked
   copy is functionally identical to its source.

2. **Importance permutation** (``--permute``). The nested rungs use the FIRST ``width`` hidden
   units, so without reordering a 1/64 rung is an *arbitrary* 1/64 of the MLP. Permuting the
   hidden units (``w1``/``w3`` rows and ``w2`` columns together) is an exactly output-preserving
   reparameterization, so sorting them by importance costs nothing and makes every narrow rung a
   meaningfully better approximation of the full FFN from step 0.

   ``--permute weight`` uses the data-free proxy ``||w1[j]|| * ||w3[j]|| * ||w2[:,j]||``.
   ``--permute act --act-stats FILE`` uses measured mean ``|act(w1 x) * (w3 x)|`` per unit (a
   ``.pt`` file holding a ``{layer_index: tensor(hidden_size)}`` dict), which is the better signal
   if you have it.

Usage (on the node holding the base, e.g. sneetches)::

    python src/scripts/train/memexpress/ffnmoe/bake_ffn_moe_into_base.py \\
        --base /data/prasann/pooledkv_exp/q4b-dense-cpt-fixmark/model_and_optim \\
        --out  /data/prasann/ffnmoe_exp/q4b-dense-cpt-fixmark-ffnmoe \\
        --permute weight --start-layer 4 --divisors 1,4,16,64

``--with-projector`` additionally creates the soft-token projector keys. That is ONLY for a later
composition of this work with the pooled-doc-KV compression -- leave it off for the standalone
FFN study, where the source should be the plain ``q4b-dense-cpt-fixmark`` base.
"""

import argparse

import torch

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
from olmo_core.nn.nested_ffn_moe import (
    apply_ffn_permutation,
    ffn_importance_permutation,
    resolve_rung_widths,
)
from olmo_core.nn.transformer import TransformerConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="source model_and_optim distcp dir")
    ap.add_argument("--out", required=True, help="destination dir (model_and_optim/ written under)")
    ap.add_argument("--model-size", default="4B", choices=["0.6B", "4B"])
    ap.add_argument(
        "--with-projector",
        action="store_true",
        help="also create the soft-token projector keys -- ONLY for a later composition with the "
        "pooled-doc-KV arms; leave off for the standalone FFN study",
    )
    ap.add_argument("--start-layer", type=int, default=4)
    ap.add_argument(
        "--src-start-layer",
        type=int,
        default=-1,
        help="if >= 0, the SOURCE already carries trained router/gain keys from this layer on "
        "(e.g. a finished L12 run): those are loaded and kept, and fresh routers are added only "
        "on layers [--start-layer, src). Requires --permute none.",
    )
    ap.add_argument("--divisors", default="1,4,16,64", help="rung cost divisors")
    ap.add_argument("--width-multiple", type=int, default=8, help="rung width multiple (1 allows width-1 rungs)")
    ap.add_argument("--no-null", action="store_true", help="drop the zero-compute null rung")
    ap.add_argument(
        "--permute",
        default="weight",
        choices=["none", "weight", "act"],
        help="reorder FFN hidden units by importance (output-preserving)",
    )
    ap.add_argument(
        "--act-stats",
        default=None,
        help="--permute act: .pt file with {layer_index: tensor(hidden_size)} activation stats",
    )
    args = ap.parse_args()

    if args.permute == "act" and not args.act_stats:
        raise SystemExit("--permute act requires --act-stats")

    factory = {"0.6B": TransformerConfig.qwen3_0_6B, "4B": TransformerConfig.qwen3_4B}[
        args.model_size
    ]
    model = factory(vocab_size=TokenizerConfig.qwen3().padded_vocab_size()).build(init_device="cpu")
    divisors = [float(x) for x in args.divisors.split(",")]
    if args.src_start_layer >= 0:
        # Two-stage warm start: the source's routers (layers >= src) must EXIST on the model
        # before the load so their trained values are read instead of dropped as unexpected keys.
        if args.permute != "none":
            raise SystemExit("--src-start-layer requires --permute none (units already permuted)")
        if args.start_layer >= args.src_start_layer:
            raise SystemExit("--start-layer must be below --src-start-layer")
        model.enable_nested_ffn_moe(
            start_layer=args.src_start_layer, divisors=divisors, include_null=not args.no_null, width_multiple=args.width_multiple
        )
    load_model_and_optim_state(args.base, model)

    divisors = [float(x) for x in args.divisors.split(",")]
    hidden = next(iter(model.blocks.values())).feed_forward.w1.out_features
    widths, costs = resolve_rung_widths(hidden, divisors, include_null=not args.no_null, multiple_of=args.width_multiple)
    print(f"rungs (hidden={hidden}): widths={widths} costs={[round(c, 5) for c in costs]}")
    print(f"  cheapest non-null rung is {1 / costs[len(widths) - 2]:.0f}x smaller than full")

    if args.permute != "none":
        act_stats = torch.load(args.act_stats) if args.permute == "act" else {}
        n = 0
        for key, block in model.blocks.items():
            if int(key) < args.start_layer:
                continue
            ff = block.feed_forward
            stats = act_stats.get(int(key)) if args.permute == "act" else None
            if args.permute == "act" and stats is None:
                raise SystemExit(f"--act-stats has no entry for layer {key}")
            apply_ffn_permutation(ff, ffn_importance_permutation(ff, stats))
            n += 1
        print(f"permuted hidden units on {n} blocks ({args.permute} importance, output-preserving)")

    if args.with_projector:
        # Enable BEFORE the router so the projector's residual init (P(x) == x) is preserved.
        model.enable_pooled_soft_tokens(
            DOC_START_ID, DOC_END_ID, EOS_TOKEN_ID, placeholder_id=LANDMARK_TOKEN_ID
        )
        n_proj = sum(p.numel() for p in model.pooled_projector.parameters())
        print(f"projector: {n_proj / 1e6:.1f}M params, residual-init (P(x) == x)")

    if args.src_start_layer >= 0:
        from olmo_core.nn.nested_ffn_moe import install_nested_ffn_moe

        cfg = model._nested_ffn_moe
        added = install_nested_ffn_moe(
            model.blocks,
            cfg["holder"],
            start_layer=args.start_layer,
            widths=cfg["widths"],
            costs=cfg["costs"],
        )
        cfg["start_layer"] = args.start_layer
        cfg["holder"].start_layer = args.start_layer
        print(
            f"kept trained routers on layers >= {args.src_start_layer}; added fresh (full-rung) "
            f"routers on {len(added)} blocks: {added}"
        )
    else:
        model.enable_nested_ffn_moe(
            start_layer=args.start_layer, divisors=divisors, include_null=not args.no_null, width_multiple=args.width_multiple
        )
    n_router = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "_nffn_router" in name or "_nffn_gain" in name
    )
    print(f"router+gain: {n_router / 1e6:.3f}M params, init selects the full rung (p ~ 1)")

    save_model_and_optim_state(f"{args.out}/model_and_optim", model)
    print(f"wrote base with nested-FFN-MoE keys -> {args.out}/model_and_optim")


if __name__ == "__main__":
    main()
