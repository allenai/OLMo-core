"""Benchmark strict-FP32 router GEMM operand layouts, including output copy cost.

Isolated diagnostic only. Algebraically transpose the complete product to expose
different SIMT tiling; never enable TF32 or change training's router precision.
"""

import json
from pathlib import Path

import torch
from olmoe3_router_blas_probe import measure


def main():
    """Compare complete contiguous-output regions with bracketed native timings."""
    torch.manual_seed(20260905)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x = torch.randn(32768, 1024, device="cuda", dtype=torch.bfloat16).float()
    weight = torch.randn(512, 1024, device="cuda", dtype=torch.float32) * 0.03
    dy = torch.randn(32768, 512, device="cuda", dtype=torch.float32) * 1e-5
    operands = {
        "forward": (x, weight.T),
        "input_gradient": (dy, weight),
        "weight_gradient": (dy.T, x),
    }
    rows = []
    for operation, (left, right) in operands.items():
        native = lambda left=left, right=right: torch.mm(left, right)
        transposed = lambda left=left, right=right: torch.mm(right.T, left.T).T.contiguous()
        expected, before = measure(native)
        actual, candidate = measure(transposed)
        _, after = measure(native)
        assert actual.is_contiguous() and actual.dtype == torch.float32
        assert bool(torch.isfinite(actual).all())
        delta = actual - expected
        # Sparse exact FP64 samples distinguish arithmetic drift from a bad result.
        ri = torch.arange(0, expected.shape[0], max(1, expected.shape[0] // 16), device="cuda")
        ci = torch.arange(0, expected.shape[1], max(1, expected.shape[1] // 16), device="cuda")
        fp64 = left[ri].double() @ right[:, ci].double()
        row = {
            "operation": operation,
            "native_before": before,
            "transposed": candidate,
            "native_after": after,
            "max_abs": delta.abs().max().item(),
            "relative_l2": (delta.norm() / expected.norm()).item(),
            "bf16_mismatch_fraction": (actual.bfloat16() != expected.bfloat16())
            .float()
            .mean()
            .item(),
            "native_fp64_sample_max_abs": (expected[ri][:, ci].double() - fp64).abs().max().item(),
            "candidate_fp64_sample_max_abs": (actual[ri][:, ci].double() - fp64).abs().max().item(),
        }
        if operation == "forward":
            ref_indices = expected.topk(16, dim=-1).indices
            got_indices = actual.topk(16, dim=-1).indices
            row["top16_order_mismatches"] = (ref_indices != got_indices).sum().item()
            row["top16_set_mismatches"] = (
                (ref_indices.sort().values != got_indices.sort().values).sum().item()
            )
        rows.append(row)
        print("ROUTER_LAYOUT_RESULT", json.dumps(row), flush=True)
    assert not torch.backends.cuda.matmul.allow_tf32
    Path("/results/router-layout.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
