"""Isolated strict-FP32 cuBLAS/cuBLASLt probe for the small model's router GEMMs.

This module is never imported by training. TF32 remains disabled throughout.
"""

import json
import statistics
from pathlib import Path

import torch


def measure(fn):
    """Use warmed CUDA graph replay to avoid measuring Python dispatch jitter."""
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(30):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 30)
    return output.clone(), {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    """Measure all three router GEMMs and report actual CUDA kernel names and error."""
    torch.manual_seed(1824)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_capability()[0] == 10
    x = torch.randn(32768, 1024, device="cuda", dtype=torch.bfloat16).float()
    weight = (torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16) * 0.03).float()
    dy = torch.randn(32768, 512, device="cuda", dtype=torch.float32) * 1e-5
    operations = {
        "forward": lambda: torch.mm(x, weight.T),
        "input_gradient": lambda: torch.mm(dy, weight),
        "weight_gradient": lambda: torch.mm(dy.T, x),
    }
    references, results = {}, []
    for backend in ("cublas", "cublaslt", "cublas"):
        torch.backends.cuda.preferred_blas_library(backend)
        assert not torch.backends.cuda.matmul.allow_tf32
        arm = {"backend": backend, "operations": {}}
        for name, fn in operations.items():
            value, timing = measure(fn)
            reference = references.setdefault(name, value.clone())
            delta = value - reference
            timing.update(
                exact=torch.equal(value, reference),
                max_abs_error=delta.abs().max().item(),
                relative_l2_error=(delta.norm() / reference.norm()).item(),
                bf16_mismatch_fraction=(value.bfloat16() != reference.bfloat16())
                .float()
                .mean()
                .item(),
            )
            assert torch.isfinite(value).all(), (backend, name)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            timing["cuda_kernels"] = sorted(
                {
                    event.name
                    for event in prof.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA
                }
            )
            arm["operations"][name] = timing
        results.append(arm)
        print("ROUTER_BLAS_PROBE", json.dumps(arm), flush=True)
    Path("/results/router-blas.json").write_text(json.dumps(results, indent=2))
    print("ROUTER_BLAS_COMPLETE tf32=false precision=highest", flush=True)


if __name__ == "__main__":
    main()
