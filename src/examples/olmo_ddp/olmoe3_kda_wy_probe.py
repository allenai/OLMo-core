"""Bounded KDA WY launch-configuration probe at the exact small-production shape.

No source changes to kernel-fun, no chunk-size/precision changes. Compare complete
forward/backward against the current CTA128 chain with fixed inputs, including
beta>1 and strong decay. A scheduling win must improve the whole chain to matter.
"""

import json
import os
import statistics
from pathlib import Path

import torch


def main():
    """Qualify outputs/all input gradients before timing alternate launch schedules."""
    from kernel_fun._common import support
    from kernel_fun.kda import chunk_kda, is_supported
    from kernel_fun.kda._kernels import bwd_wy_t

    support.MIN_CTAS = 128
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(507)
    output = Path(os.environ.get("RESULTS_DIR", "/results")) / "kda-wy"
    output.mkdir(parents=True, exist_ok=True)
    shape = (4, 8192, 8, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn(4, 8192, 8, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    beta = (torch.sigmoid(torch.randn(4, 8192, 8, device="cuda")) * 2).requires_grad_()
    dy = torch.randn_like(v)
    ok, reason = is_supported(q, v, use_qk_l2norm_in_kernel=True)
    assert ok, reason
    baseline = (3, 8, 80)
    schedules = [baseline, (2, 8, 80), (4, 8, 80), (3, 4, 128)]
    summary = {
        "source_commit": os.environ.get("GIT_REF"),
        "shape_q": shape,
        "shape_v": list(v.shape),
        "cases": [],
    }

    def schedule(config):
        bwd_wy_t.MAIN_STAGES, bwd_wy_t.SIDE_WARPS, bwd_wy_t.SIDE_MAXNREG = config

    for strength in (0.1, 16.0):
        g = (-torch.rand(shape, device="cuda") * strength).requires_grad_()
        inputs = (q, k, v, g, beta)

        def execute():
            y, _ = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
            gradients = torch.autograd.grad(y, inputs, dy)
            return (y, *gradients)

        schedule(baseline)
        expected = tuple(t.detach().clone() for t in execute())
        assert all(bool(torch.isfinite(t).all()) for t in expected)

        def measure():
            for _ in range(5):
                _result = execute()
            torch.cuda.synchronize()
            pairs = []
            for _ in range(20):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                _result = execute()
                b.record()
                pairs.append((a, b))
            torch.cuda.synchronize()
            values = [a.elapsed_time(b) for a, b in pairs]
            return {"median_ms": statistics.median(values), "mean_ms": statistics.mean(values)}

        for config in [*schedules, baseline]:
            schedule(config)
            actual = execute()
            deltas = []
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
                deltas.append(float((left.float() - right.float()).abs().max()))
            row = {
                "decay_strength": strength,
                "schedule": config,
                "max_abs_errors": deltas,
                **measure(),
            }
            summary["cases"].append(row)
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            print("KDA_WY_PROBE", json.dumps(row), flush=True)
        del expected, actual
    schedule(baseline)


if __name__ == "__main__":
    main()
