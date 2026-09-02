"""
CPU smoke test: does the router actually give up compute, and does CE survive?

Trains a tiny transformer for a few hundred steps on a memorizable synthetic task with the nested
FFN mixture enabled, and prints the executed mean FFN cost alongside CE. This is the cheapest
check that the learning dynamics work at all -- the budget hinge must pull mean_cost down from
1.0 while CE stays finite. A run where mean_cost never moves means the budget weight is too low
(or the router LR too small); one where CE explodes means the target is too aggressive.

    python debug/ffnmoe/smoke_router_learns.py --steps 300 --target 0.1
"""

import argparse

import torch

from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--target", type=float, default=0.1)
    ap.add_argument("--budget-weight", type=float, default=1.0)
    ap.add_argument("--router-lr", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--explore", type=float, default=0.1)
    ap.add_argument("--recon-weight", type=float, default=0.0)
    ap.add_argument("--anneal-frac", type=float, default=0.3)
    args = ap.parse_args()

    seed_all(0)
    vocab, seq = 256, 64
    model = TransformerConfig.olmo2_190M(
        vocab_size=vocab, n_layers=4, fused_ops=False, dtype=DType.float32
    ).build(init_device="cpu")
    model.enable_nested_ffn_moe(
        start_layer=1,
        divisors=(1, 4, 16),
        target_cost=args.target,
        budget_weight=args.budget_weight,
        target_anneal_calls=int(args.steps * args.anneal_frac),
        explore_prob=args.explore,
        explore_anneal_calls=int(args.steps * args.anneal_frac),
        recon_frac=0.05 if args.recon_weight > 0 else 0.0,
        recon_weight=args.recon_weight,
    )
    holder = model._nested_ffn_moe["holder"]
    print(f"rungs: {model._nested_ffn_moe['widths']} costs={model._nested_ffn_moe['costs']}")

    router_params, backbone = [], []
    for name, p in model.named_parameters():
        (router_params if "_nffn" in name else backbone).append(p)
    opt = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr}, {"params": router_params, "lr": args.router_lr}]
    )

    # A fixed batch the model can memorize: any CE rise is attributable to lost FFN capacity.
    torch.manual_seed(1)
    batch = torch.randint(0, vocab, (4, seq))

    model.train()
    for step in range(1, args.steps + 1):
        out = model(batch, labels=batch.clone())
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            m = holder.metrics()
            fracs = " ".join(f"{m[f'ffn_moe/frac_rung{i}']:.2f}" for i in range(holder.n_rungs))
            print(
                f"step {step:4d} | ce {out.ce_loss.item():7.4f} | total {out.loss.item():7.4f} | "
                f"mean_cost {m['ffn_moe/mean_cost']:.4f} ({m['ffn_moe/speedup']:5.1f}x) | "
                f"target {m['ffn_moe/target']:.3f} | rungs {fracs}",
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        final = model(batch, labels=batch.clone())
    m = holder.metrics()
    print(
        f"\nFINAL (eval, no exploration): ce {final.ce_loss.item():.4f} | "
        f"mean_cost {m['ffn_moe/mean_cost']:.4f} -> {m['ffn_moe/speedup']:.1f}x FFN reduction "
        f"on routed layers"
    )


if __name__ == "__main__":
    main()
