"""Isolate ragged gated-GEMM TMA failures before attempting a training integration.

Each layout has an independent CUDA process. Interleaved scratch is internal only:
checkpoint weights retain concatenated up/gate layout. No training imports this file.
"""

import argparse
import importlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


def run_case(layout, tile, production, output):
    """Validate saved BF16 projections and activation, then time complete forward region."""
    import cutlass
    import torch
    import torch.nn.functional as F
    from cutlass import cute
    from olmoe3_gemm_swiglu_probe import up_gate_swiglu
    from quack.gemm_act import GemmGatedMixin, GemmGatedSm100

    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(20260905)
    module = importlib.import_module("quack.gemm_act")

    class RoundedGatedGemm(GemmGatedSm100):
        @cute.jit
        def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
            tRS_rD.store(tRS_rD.load().to(cutlass.BFloat16).to(cutlass.Float32))
            return GemmGatedMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)

    module.GemmGatedSm100 = RoundedGatedGemm
    module.gate_fn_map["rounded-up-gate-isolate"] = up_gate_swiglu
    experts = 512 if production else 4
    rows = experts * 1024
    cu = torch.arange(experts + 1, device="cuda", dtype=torch.int32) * 1024
    x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16).mul_(0.1)
    weights = torch.randn(experts, 2048, 512, device="cuda", dtype=torch.bfloat16).mul_(0.1)
    # The interleaved-control arm includes the packing cost in timing, not just GEMM.
    saved = torch.empty(rows, 2048, device="cuda", dtype=torch.bfloat16)
    activated = torch.empty(rows, 1024, device="cuda", dtype=torch.bfloat16)
    activation = torch.compile(lambda z: z[:, :1024] * F.silu(z[:, 1024:]), fullgraph=True)

    def native():
        pre = F.grouped_mm(x, weights.mT, offs=cu[1:])
        return pre, activation(pre)

    def candidate():
        w = weights
        if layout == "packed-control":
            w = weights.view(experts, 2, 1024, 512).transpose(1, 2).contiguous().view_as(weights)
        module.gemm_act(
            x,
            w,
            None if layout == "no-saved" else saved,
            None,
            activated,
            None,
            "rounded-up-gate-isolate",
            *tile,
            persistent=True,
            is_dynamic_persistent=True,
            cu_seqlens_m=cu,
            concat_layout=None if layout == "packed-control" else ("B",),
        )
        return saved, activated

    ref = native()
    got = candidate()
    torch.cuda.synchronize()
    saved_concat = saved.view(rows, 1024, 2).transpose(1, 2).reshape(rows, 2048)
    errors = []
    pairs = (
        [(ref[1], got[1])] if layout == "no-saved" else [(ref[0], saved_concat), (ref[1], got[1])]
    )
    for a, b in pairs:
        mismatches, max_abs, d2, r2 = 0, 0.0, 0.0, 0.0
        for start in range(0, rows, 4096):
            left, right = a[start : start + 4096].float(), b[start : start + 4096].float()
            assert bool(torch.isfinite(right).all())
            delta = right - left
            mismatches += int((left != right).sum())
            max_abs = max(max_abs, float(delta.abs().max()))
            d2 += float(delta.double().square().sum())
            r2 += float(left.double().square().sum())
        errors.append(
            {"mismatches": mismatches, "max_abs": max_abs, "relative_l2": (d2 / r2) ** 0.5}
        )

    def measure(fn):
        for _ in range(4):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b))
        return {"median_ms": statistics.median(times), "mean_ms": statistics.mean(times)}

    report = {
        "layout": layout,
        "tile": tile,
        "production": production,
        "errors": errors,
        "native_before": measure(native),
        "candidate": measure(candidate),
        "native_after": measure(native),
        "caveat": "Forward-only diagnostic; saved scratch reinterpretation is not training-qualified",
    }
    output.write_text(json.dumps(report, indent=2))
    print("GEMM_ISOLATE_RESULT", json.dumps(report), flush=True)


def main():
    """Use isolated processes so an illegal instruction cannot poison the next case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("no-saved", "interleaved-saved", "packed-control"))
    parser.add_argument("--tile", choices=("small", "wide", "large"), default="small")
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()
    output = Path("/results/gemm-isolate")
    output.mkdir(parents=True, exist_ok=True)
    if args.layout:
        run_case(
            args.layout,
            {"small": (128, 128, 1, 1), "wide": (128, 256, 1, 1), "large": (256, 256, 2, 1)}[
                args.tile
            ],
            args.production,
            output / f"{args.layout}-{args.tile}-{args.production}.json",
        )
        return
    cases = []
    for production in (False, True):
        for layout in ("no-saved", "interleaved-saved", "packed-control"):
            command = [sys.executable, "-u", __file__, "--layout", layout, "--tile", args.tile]
            if production:
                command += ["--production"]
            try:
                code = subprocess.run(command, timeout=480, check=False).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
            cases.append({"layout": layout, "production": production, "exit_code": code})
            (output / "cases.json").write_text(json.dumps(cases, indent=2))
    if any(case["exit_code"] != 0 for case in cases):
        raise RuntimeError("One or more isolated cases failed; inspect cases.json")


if __name__ == "__main__":
    main()
