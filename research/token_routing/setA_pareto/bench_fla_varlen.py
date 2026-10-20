"""Does a new sequence length cost a Triton re-autotune in a Qwen3.5 (GDN) TRAINING step?

The soft-token arm compacts every micro-batch to a data-dependent length, so each step can present
lengths the FLA kernels have never seen. This times forward+backward of a 4-layer Qwen3.5-4B-shaped
model (3 GDN + 1 attention block, real widths) on 1 GPU at a series of distinct lengths, first with
FLA's stock length-keyed autotuning, then after ``freeze_fla_length_autotune()`` at fresh lengths.

    python research/token_routing/setA_pareto/bench_fla_varlen.py
"""

import time

import torch

from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune
from olmo_core.nn.transformer import TransformerConfig


def step(model, T):
    ids = torch.randint(0, 200_000, (1, T), device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.shape[-1]), ids.view(-1))
    loss.backward()
    torch.cuda.synchronize()
    return time.time() - t0


def main():
    cfg = TransformerConfig.qwen3_5_4B(vocab_size=248320, n_layers=4)
    model = cfg.build(init_device="cuda").to(torch.bfloat16)
    print("warm (same length twice):", [f"{step(model, 3000):.2f}" for _ in range(2)], flush=True)
    for label, lengths in (("stock", [3100, 3500, 4400, 5600, 2200]),
                           ("frozen", [3300, 3900, 4900, 6100, 2600])):
        if label == "frozen":
            print("froze", freeze_fla_length_autotune(), "autotuners", flush=True)
        for T in lengths:
            a = step(model, T)
            b = step(model, T)
            print(f"{label:6s} T={T:5d} first={a:6.2f}s  repeat={b:6.2f}s", flush=True)


if __name__ == "__main__":
    main()
