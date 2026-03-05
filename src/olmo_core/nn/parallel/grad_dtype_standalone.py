#!/usr/bin/env python3
"""
Minimal single-process grad dtype test (no DDP).

Checks:
1) Forward/backward run in bf16.
2) Final grads are fp32.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.fc3(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--d-in", type=int, default=128)
    p.add_argument("--d-hidden", type=int, default=256)
    p.add_argument("--d-out", type=int, default=64)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", choices=["cuda", "cpu"], default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = TinyMLP(args.d_in, args.d_hidden, args.d_out).to(device).to(torch.bfloat16)

    for p in model.parameters():
        p.grad_dtype = torch.float32  # type: ignore[attr-defined]

    x = torch.randn(args.batch_size, args.d_in, device=device, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn(args.batch_size, args.d_out, device=device, dtype=torch.bfloat16)

    output = model(x)
    if output.dtype != torch.bfloat16:
        raise RuntimeError(f"Expected bf16 forward output, got {output.dtype}")

    loss = F.mse_loss(output, target)
    loss.backward()

    if x.grad is None or x.grad.dtype != torch.bfloat16:
        raise RuntimeError(f"Expected bf16 backward grad for input, got {None if x.grad is None else x.grad.dtype}")

    bad_final = []
    total_abs = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            bad_final.append(f"{name}: missing final grad")
            continue
        if p.grad.dtype != torch.float32:
            bad_final.append(f"{name}: final grad dtype {p.grad.dtype}, expected fp32")
        total_abs += p.grad.abs().sum().item()

    if bad_final:
        raise RuntimeError("Final grad checks failed: " + "; ".join(bad_final))

    print("PASS")
    print(f"device={device.type} loss_dtype={loss.dtype} final_grad_abs_sum={total_abs:.6f}")


if __name__ == "__main__":
    main()
