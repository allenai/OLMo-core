"""Standalone GEMM/SwiGLU fusion with BF16 rounding and saved preactivations.

Private process only. Never imported by training. Retain the original concatenated
up/gate layout, both saved BF16 projections, and the BF16 boundary before activation.
"""

import importlib
import json
import statistics
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F


@cute.jit
def up_gate_swiglu(up, gate):
    if cutlass.const_expr(isinstance(up, tuple)):
        return (
            up[0] * (gate[0] / (1.0 + cute.math.exp(-gate[0], fastmath=False))),
            up[1] * (gate[1] / (1.0 + cute.math.exp(-gate[1], fastmath=False))),
        )
    return up * (gate / (1.0 + cute.math.exp(-gate, fastmath=False)))


def main():
    """Check every saved/activated value and bracket the fused operation's timing."""
    from quack.gemm_act import GemmGatedMixin, GemmGatedSm100

    from olmo_core.ops.swiglu_pairwise import pairwise_swiglu

    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(892)

    class RoundedGatedGemm(GemmGatedSm100):
        @cute.jit
        def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
            tRS_rD.store(tRS_rD.load().to(cutlass.BFloat16).to(cutlass.Float32))
            return GemmGatedMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)

    module = importlib.import_module("quack.gemm_act")
    module.GemmGatedSm100 = RoundedGatedGemm
    module.gate_fn_map["profile-up-gate-rounded"] = up_gate_swiglu
    output = Path("/results/gemm-swiglu")
    output.mkdir(parents=True, exist_ok=True)
    report = []
    activation = torch.compile(pairwise_swiglu, fullgraph=True)

    def measure(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        events = []
        for _ in range(30):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        values = [a.elapsed_time(b) for a, b in events]
        return {"median_ms": statistics.median(values), "mean_ms": statistics.mean(values)}

    for routing in ("uniform", "skew"):
        counts = [1024] * 512 if routing == "uniform" else [0, 80, 160, 3856] * 128
        counts = torch.tensor(counts, device="cuda", dtype=torch.int32)
        cu = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
        x = torch.randn(524288, 512, device="cuda", dtype=torch.bfloat16).mul_(0.1)
        weights = torch.randn(512, 2048, 512, device="cuda", dtype=torch.bfloat16).mul_(0.1)
        saved = torch.empty(524288, 2048, device="cuda", dtype=torch.bfloat16)
        activated = torch.empty(524288, 1024, device="cuda", dtype=torch.bfloat16)

        def native():
            preact = F.grouped_mm(x, weights.transpose(1, 2), offs=cu[1:])
            return preact, activation(preact)

        def fused():
            module.gemm_act(
                x,
                weights,
                saved,
                None,
                activated,
                None,
                "profile-up-gate-rounded",
                256,
                256,
                2,
                1,
                persistent=True,
                is_dynamic_persistent=True,
                cu_seqlens_m=cu,
                concat_layout=("B", "out"),
            )
            return saved, activated

        expected = native()
        actual = fused()
        errors = []
        for left, right in zip(expected, actual):
            mismatches = 0
            max_abs = 0.0
            for start in range(0, left.shape[0], 4096):
                a, b = left[start : start + 4096], right[start : start + 4096]
                mismatches += int((a != b).sum())
                max_abs = max(max_abs, float((a.float() - b.float()).abs().max()))
                assert bool(torch.isfinite(b).all())
            errors.append({"mismatches": mismatches, "max_abs": max_abs})
        row = {
            "routing": routing,
            "saved_preactivation_and_output_errors": errors,
            "native_before": measure(native),
            "fused": measure(fused),
            "native_after": measure(native),
        }
        report.append(row)
        (output / "summary.json").write_text(json.dumps(report, indent=2))
        print("GEMM_SWIGLU_PROBE", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
