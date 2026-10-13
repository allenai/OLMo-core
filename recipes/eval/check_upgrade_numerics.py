"""
Check that an existing checkpoint gives the same outputs on two code/env combinations.

Run once per (tree, env) with ``PYTHONPATH=<tree>/src``. Each run saves logits and a greedy
continuation for fixed inputs; ``--compare A.pt B.pt`` then reports the differences. Use it when
moving the branch to a new upstream / torch / fla, e.g.::

    PYTHONPATH=old/src old_env/bin/python check_upgrade_numerics.py --ckpt C --out old.pt
    PYTHONPATH=new/src new_env/bin/python check_upgrade_numerics.py --ckpt C --out new.pt
    python check_upgrade_numerics.py --compare old.pt new.pt
"""

import argparse
import json
import sys
from pathlib import Path


def run(ckpt: str, out: str, seq_len: int, gen_tokens: int) -> None:
    import torch

    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.transformer import TransformerConfig

    cfg = json.loads((Path(ckpt) / "config.json").read_text())["model"]
    model = TransformerConfig.from_dict(cfg).build(init_device="cpu")
    load_model_and_optim_state(str(Path(ckpt) / "model_and_optim"), model)
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()

    g = torch.Generator().manual_seed(0)
    # Real-vocab ids only (stay below the reserved/marker rows).
    ids = torch.randint(100, 150_000, (2, seq_len), generator=g).cuda()
    with torch.no_grad():
        logits = model(ids).float().cpu()
        gen = ids[:1, :64]
        for _ in range(gen_tokens):
            nxt = model(gen)[:, -1].argmax(-1, keepdim=True)
            gen = torch.cat([gen, nxt], dim=1)
    torch.save(
        {
            "logits": logits,
            "greedy": gen[0, 64:].cpu(),
            "torch": torch.__version__,
            "state_keys": sorted(model.state_dict().keys()),
        },
        out,
    )
    print(f"saved {out}: logits {tuple(logits.shape)} torch {torch.__version__}")


def compare(a_path: str, b_path: str) -> int:
    import torch

    a, b = torch.load(a_path), torch.load(b_path)
    ok = True
    if a["state_keys"] != b["state_keys"]:
        ok = False
        print("STATE KEYS DIFFER:", set(a["state_keys"]) ^ set(b["state_keys"]))
    la, lb = a["logits"], b["logits"]
    diff = (la - lb).abs()
    top1 = (la.argmax(-1) == lb.argmax(-1)).float().mean().item()
    pa, pb = la.log_softmax(-1), lb.log_softmax(-1)
    kl = (pa.exp() * (pa - pb)).sum(-1).mean().item()
    same_gen = torch.equal(a["greedy"], b["greedy"])
    print(f"torch {a['torch']} vs {b['torch']}")
    print(f"max|dlogit| {diff.max().item():.4f}  mean {diff.mean().item():.5f}  top1 agree {top1:.4f}")
    print(f"KL(a||b) per token {kl:.2e}  greedy continuation identical: {same_gen}")
    # bf16 across torch/kernel versions: expect tiny drift, not a different model.
    if top1 < 0.99 or kl > 1e-3:
        ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt")
    p.add_argument("--out")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--gen-tokens", type=int, default=32)
    p.add_argument("--compare", nargs=2)
    args = p.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    run(args.ckpt, args.out, args.seq_len, args.gen_tokens)
