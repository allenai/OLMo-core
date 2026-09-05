"""One-GPU counter-access qualification on isolated hot-op shapes; no NCCL replay.

The grouped GEMM uses synthetic uniform routing, not a replay of trained routing.
Raw reports remain on Weka; compact hardware metrics go to Beaker results.
"""

import argparse
import csv
import io
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path


def worker(operation):
    """Capture one isolated compute kernel after warming its exact shape."""
    import torch
    import torch.nn.functional as F

    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if operation.startswith("grad-add"):
        from olmoe3_grad_add_bench import gradient_add

        shape = (512, 2048, 512)
        source = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        destination = torch.zeros(shape, dtype=torch.float32, device="cuda")

        def execute():
            return (
                destination.add_(source)
                if operation == "grad-add"
                else gradient_add(destination, source)
            )

        metadata = {"source_dtype": "bfloat16", "destination_dtype": "float32", "shape": shape}
    else:
        # Actual small-model expert widths and routed row capacity, uniform 1024/expert.
        x = torch.randn(4 * 8192 * 16, 512, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(512, 2048, 512, device="cuda", dtype=torch.bfloat16)
        offsets = torch.arange(1, 513, device="cuda", dtype=torch.int32) * 1024

        def execute():
            return F.grouped_mm(x, weight.transpose(1, 2), offs=offsets)

        metadata = {
            "input_shape": list(x.shape),
            "weight_shape": list(weight.shape),
            "routing": "synthetic uniform, 1024 rows/expert",
        }
    print(
        "NCU_WORKLOAD",
        json.dumps(
            {
                "operation": operation,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                **metadata,
            }
        ),
        flush=True,
    )
    for _ in range(3):
        output = execute()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    output = execute()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    assert bool(torch.isfinite(output).all().item())
    print("NCU_WORKER_COMPLETED", operation, flush=True)


def parse_metrics(text):
    """Read raw CLI metrics and reject reports without actual timing/counter values."""
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if '"Metric Name"' in line and '"Metric Value"' in line),
        None,
    )
    if start is None:
        return []
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    return [
        {
            "name": row.get("Metric Name"),
            "unit": row.get("Metric Unit"),
            "value": row.get("Metric Value"),
            "kernel": row.get("Kernel Name"),
        }
        for row in rows
        if row.get("Metric Name")
    ]


def driver(run_name):
    """Do not alter host counter permissions/clocks; report any access restriction."""
    if Path(run_name).name != run_name:
        raise ValueError("Expected a unique run name")
    root = Path("/weka/olmo-3p5-checkpoints/production-profiling") / run_name
    root.mkdir(exist_ok=False)
    # Numerical/memory-bandwidth experiment succeeds or fails independently of
    # hardware-counter permissions; it needs no NCCL or full-model allocation.
    subprocess.run(
        [sys.executable, "-u", "src/examples/olmo_ddp/olmoe3_grad_add_bench.py"],
        check=True,
        timeout=300,
    )
    candidates = (
        [Path(shutil.which("ncu"))]
        if shutil.which("ncu")
        else sorted(Path("/opt/nvidia").rglob("ncu"))
    )
    if not candidates:
        raise RuntimeError("No Nsight Compute CLI installed in this image")
    binary = candidates[-1]
    version = subprocess.check_output([str(binary), "--version"], text=True)
    print("NCU_VERSION", version, flush=True)
    summary = {
        "version": version,
        "binary": str(binary),
        "source_commit": os.environ.get("GIT_REF"),
        "cases": [],
    }
    for operation, kernel in (
        ("grad-add", "CUDAFunctor_add"),
        ("grad-add-triton", "_gradient_add"),
        ("grouped-up", "GemmUniversal"),
    ):
        report = root / operation
        command = [
            str(binary),
            "--set",
            "basic",
            "--profile-from-start",
            "off",
            "--clock-control",
            "none",
            "--cache-control",
            "all",
            "--replay-mode",
            "kernel",
            "--launch-count",
            "1",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            f"regex:{kernel}",
            "--export",
            str(report),
            sys.executable,
            "-u",
            __file__,
            "worker",
            operation,
        ]
        print("NCU_CASE_START", operation, flush=True)
        with (root / f"{operation}.log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = process.wait(timeout=450)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
                code = -999
        log_text = (root / f"{operation}.log").read_text()
        print(log_text[-8000:], flush=True)
        metrics = []
        if code == 0 and report.with_suffix(".ncu-rep").is_file():
            export = subprocess.run(
                [
                    str(binary),
                    "--import",
                    str(report.with_suffix(".ncu-rep")),
                    "--page",
                    "raw",
                    "--csv",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
            (root / f"{operation}-metrics.csv").write_text(export.stdout)
            if export.returncode == 0:
                metrics = parse_metrics(export.stdout)
        numeric_metrics = {row["name"] for row in metrics if row["value"] not in (None, "", "n/a")}
        valid = (
            code == 0
            and "NCU_WORKER_COMPLETED" in log_text
            and "gpu__time_duration.sum" in numeric_metrics
            and any("throughput" in name for name in numeric_metrics)
        )
        case = {
            "operation": operation,
            "exit_code": code,
            "valid_hardware_counters": valid,
            "metrics": metrics,
        }
        summary["cases"].append(case)
        print("NCU_CASE_RESULT", json.dumps(case), flush=True)
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        Path("/results/ncu-probe-summary.json").write_text(json.dumps(summary, indent=2))
    if not all(case["valid_hardware_counters"] for case in summary["cases"]):
        raise RuntimeError("Hardware-counter qualification incomplete; see per-case logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("driver", "worker"))
    parser.add_argument("argument")
    args = parser.parse_args()
    (driver if args.mode == "driver" else worker)(args.argument)
