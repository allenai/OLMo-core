"""Bounded output-head screening, preserving BF16 projection and FP32 CE/z-loss.

Benchmark only: no training imports this file. Native compiled loss is the
reference. Existing Liger CE-only and fused-linear paths are candidates, not
assumed equivalent; report all loss/gradient differences before promotion.
"""

import importlib.metadata
import json
import os
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from olmo_core.nn.functional import cross_entropy_loss, fused_linear_cross_entropy_loss


def native(x, weight, labels):
    """Match the existing head's BF16 logits -> FP32 CE and z-loss boundary."""
    logits = F.linear(x, weight)
    ce, z = cross_entropy_loss(
        logits, labels, reduction="sum", compute_z_loss=True, z_loss_multiplier=1e-5
    )
    return (ce + z) / 262144, ce / 262144, z / 262144


def liger_ce(x, weight, labels):
    """Keep the native projection and weight/input-gradient GEMMs unchanged."""
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction

    logits = F.linear(x, weight).float()
    outputs = LigerCrossEntropyFunction.apply(
        logits, labels, None, -100, 1e-5, 0.0, "sum", None, True
    )
    total, z = outputs[:2]
    return total / 262144, (total - z) / 262144, z / 262144


def liger_linear(x, weight, labels):
    """Use the repo's existing low-memory head, including FP32 gradient accumulation."""
    total, z = fused_linear_cross_entropy_loss(
        x,
        weight,
        labels,
        reduction="sum",
        compute_z_loss=True,
        z_loss_multiplier=1e-5,
        accum_dtype=torch.float32,
    )
    return total / 262144, (total - z) / 262144, z / 262144


def error(ref, value):
    """Report numerical differences without silently converting a screening pass to sign-off."""
    ref, value = ref.float(), value.float()
    assert bool(torch.isfinite(value).all())
    difference = value - ref
    return {
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(difference.double().norm() / ref.double().norm().clamp_min(1e-30)),
        "mismatches": int((ref != value).sum()),
        "elements": ref.numel(),
    }


def main():
    """Compare full forward/backward on the production MB4 head, including ignored targets."""
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.get_device_capability(0)[0] == 10
    output = Path("/results/lm-loss")
    output.mkdir(parents=True, exist_ok=True)
    compiled = torch.compile(native)
    report = {
        "source": os.environ.get("GIT_REF"),
        "torch": torch.__version__,
        "liger": importlib.metadata.version("liger-kernel"),
        "gpu": torch.cuda.get_device_name(0),
        "rows": 32768,
        "hidden": 1024,
        "vocab": 100352,
        "loss_div_factor": 262144,
        "z_loss_multiplier": 1e-5,
        "cases": [],
        "caveat": "Isolated synthetic output-head screening; not trained-model or optimizer parity",
    }
    for scale in (0.02, 0.2):
        torch.manual_seed(20260905)
        x = torch.randn(32768, 1024, device="cuda", dtype=torch.bfloat16).requires_grad_()
        weight = (torch.randn(100352, 1024, device="cuda") * scale).requires_grad_()
        labels = torch.randint(100352, (32768,), device="cuda")
        labels[::17] = -100

        def step(fn, x=x, weight=weight, labels=labels):
            x.grad = None
            weight.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = fn(x, weight, labels)
            losses[0].backward()
            return losses

        ref_losses = tuple(t.detach().clone() for t in step(compiled))
        ref_dx, ref_dw = x.grad.clone(), weight.grad.clone()

        def measure(fn):
            for _ in range(4):
                step(fn)
            torch.cuda.synchronize()
            times = []
            for _ in range(20):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                step(fn)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            return {"median_ms": statistics.median(times), "mean_ms": statistics.mean(times)}

        row = {"weight_scale": scale, "baseline_before": measure(compiled), "candidates": []}
        for name, fn in (("liger-ce", liger_ce), ("liger-linear", liger_linear)):
            try:
                values = step(fn)
                candidate = {
                    "name": name,
                    "loss_errors": [error(a, b) for a, b in zip(ref_losses, values)],
                    "input_gradient": error(ref_dx, x.grad),
                    "weight_gradient": error(ref_dw, weight.grad),
                    **measure(fn),
                }
            except Exception as exc:
                # A package/API incompatibility is a failed screening case, not a speed result.
                candidate = {"name": name, "error": f"{type(exc).__name__}: {exc}"}
            row["candidates"].append(candidate)
            print("LM_LOSS_RESULT", json.dumps(candidate), flush=True)
        row["baseline_after"] = measure(compiled)
        report["cases"].append(row)
        (output / "summary.json").write_text(json.dumps(report, indent=2))
        del x, weight, labels, ref_dx, ref_dw, ref_losses
    if any("error" in candidate for row in report["cases"] for candidate in row["candidates"]):
        raise RuntimeError("At least one loss path failed screening; inspect summary.json")


if __name__ == "__main__":
    main()
