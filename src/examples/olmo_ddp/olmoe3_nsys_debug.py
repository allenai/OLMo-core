"""Two-GPU Nsight reproducer: capture activation, pinned H2D, autograd NVTX, NCCL.

No model/checkpoint/data mutation. All raw captures stay on Weka. Only compact
qualification summaries go to Beaker results. A failed case is recorded, then
the next independent case runs; a success must contain actual CUDA trace data.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from olmoe3_nsys_tools import OLD_NSYS, install_nsys, validate_report


def worker():
    """Exercise the first captured H2D call and complete real backward/collective work."""
    import datetime

    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
    trace_ranks = json.loads(os.environ["NSYS_DEBUG_RANKS"])
    traced = rank in trace_ranks
    emit_nvtx = os.environ["NSYS_DEBUG_EMIT_NVTX"] == "1"
    torch.manual_seed(42)
    device = torch.device("cuda", rank)
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048, bias=False, device=device, dtype=torch.bfloat16),
        torch.nn.SiLU(),
        torch.nn.Linear(2048, 1024, bias=False, device=device, dtype=torch.bfloat16),
    )
    # Same microbatch token/d_model shape as the real small model, without the whole model.
    source = torch.randn(4, 8192, 1024, dtype=torch.bfloat16).pin_memory()
    nvtx_context = None
    metrics = []
    try:
        for step in range(1, 7):
            dist.barrier()
            torch.cuda.synchronize()
            if traced and step == 4:
                print(f"DEBUG_CAPTURE_START rank={rank}", flush=True)
                torch.cuda.cudart().cudaProfilerStart()
                if emit_nvtx:
                    nvtx_context = torch.autograd.profiler.emit_nvtx(record_shapes=True)
                    nvtx_context.__enter__()
            with torch.cuda.nvtx.range(f"debug-step-{step}") if traced else nullcontext():
                x = source.to(device, non_blocking=True)
                model.zero_grad(set_to_none=True)
                y = model(x)
                loss = y.float().square().mean()
                loss.backward()
                # Asynchronous gradient communication and a parameter gather exercise NCCL.
                handles = [dist.all_reduce(p.grad, async_op=True) for p in model.parameters()]
                for handle in handles:
                    handle.wait()
                local = next(model.parameters()).detach().flatten()[: 1024 * 1024].contiguous()
                gathered = torch.empty(
                    local.numel() * dist.get_world_size(), device=device, dtype=local.dtype
                )
                dist.all_gather_into_tensor(gathered, local)
                torch.cuda.synchronize()
                assert bool(torch.isfinite(loss).item())
                metrics.append({"step": step, "loss": loss.item()})
            if traced and step == 5:
                if nvtx_context is not None:
                    nvtx_context.__exit__(None, None, None)
                    nvtx_context = None
                torch.cuda.cudart().cudaProfilerStop()
                print(f"DEBUG_CAPTURE_STOP rank={rank}", flush=True)
        Path(os.environ["NSYS_DEBUG_CASE_DIR"], f"completed-rank-{rank}.json").write_text(
            json.dumps(metrics)
        )
    finally:
        if nvtx_context is not None:
            nvtx_context.__exit__(None, None, None)
            torch.cuda.cudart().cudaProfilerStop()
        dist.destroy_process_group()


def wrapper():
    """Only selected rank processes get Nsight injection; all ranks still participate."""
    rank = int(os.environ["LOCAL_RANK"])
    command = [sys.executable, "-u", __file__, "worker"]
    if rank in json.loads(os.environ["NSYS_DEBUG_RANKS"]):
        command = [
            os.environ["NSYS_DEBUG_BINARY"],
            "profile",
            "--trace=" + os.environ["NSYS_DEBUG_TRACE"],
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--kill=none",
            "--output",
            str(Path(os.environ["NSYS_DEBUG_CASE_DIR"], f"rank-{rank}")),
            *command,
        ]
    # Replace the wrapper so torchrun can kill the complete worker tree on failures.
    os.execvpe(command[0], command, os.environ)


def run_case(root, label, binary, trace, ranks, emit_nvtx):
    """Run one independent case with a hard deadline and exact target process cleanup."""
    case = root / label
    case.mkdir(exist_ok=False)
    env = dict(
        os.environ,
        NSYS_DEBUG_BINARY=str(binary),
        NSYS_DEBUG_TRACE=trace,
        NSYS_DEBUG_RANKS=json.dumps(ranks),
        NSYS_DEBUG_CASE_DIR=str(case),
        NSYS_DEBUG_EMIT_NVTX="1" if emit_nvtx else "0",
        OMP_NUM_THREADS="1",
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "--max-restarts=0",
        __file__,
        "wrapper",
    ]
    print("NSYS_DEBUG_CASE_START", label, flush=True)
    started = time.monotonic()
    with (case / "worker.log").open("w") as output:
        process = subprocess.Popen(
            command, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            returncode = process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
            returncode = -999
    entry = {
        "label": label,
        "binary": str(binary),
        "trace": trace,
        "ranks": ranks,
        "emit_autograd_nvtx": emit_nvtx,
        "exit_code": returncode,
        "wall_seconds": time.monotonic() - started,
        "reports": [],
        "success": False,
    }
    if returncode == 0 and len(list(case.glob("completed-rank-*.json"))) == 2:
        for rank in ranks:
            report = case / f"rank-{rank}.nsys-rep"
            if report.is_file():
                try:
                    entry["reports"].append({"rank": rank, **validate_report(binary, report)})
                except Exception as exc:
                    entry["reports"].append(
                        {"rank": rank, "error": str(exc), "valid_cuda_trace": False}
                    )
        entry["success"] = len(entry["reports"]) == len(ranks) and all(
            report["valid_cuda_trace"] and report.get("nccl_kernel_count", 0) > 0
            for report in entry["reports"]
        )
    if not entry["success"]:
        print(
            "NSYS_DEBUG_FAILURE_TAIL", label, (case / "worker.log").read_text()[-8000:], flush=True
        )
    print("NSYS_DEBUG_CASE_RESULT", json.dumps(entry), flush=True)
    (case / "summary.json").write_text(json.dumps(entry, indent=2))
    return entry


def driver(run_name):
    """Compare installed and standalone profilers without changing PyTorch or drivers."""
    if Path(run_name).name != run_name:
        raise ValueError("Expected a unique run name")
    root = Path("/weka/olmo-3p5-checkpoints/production-profiling") / run_name
    root.mkdir(exist_ok=False)
    subprocess.run(
        [
            sys.executable,
            "src/examples/olmo_ddp/olmoe3_profile_topology.py",
            str(root / "topology"),
            "--gpus",
            "2",
        ],
        check=True,
    )
    new_binary = install_nsys()
    versions = {
        label: subprocess.check_output([str(binary), "--version"], text=True).strip()
        for label, binary in (("installed", OLD_NSYS), ("standalone", new_binary))
    }
    summary = {
        "source_commit": os.environ.get("GIT_REF"),
        "versions": versions,
        "cases": [],
        "artifact_root": str(root),
    }
    for label, binary, trace, ranks, nvtx in (
        ("old-matched-two-ranks", OLD_NSYS, "cuda,nvtx,osrt", [0, 1], True),
        ("old-minimal-one-rank", OLD_NSYS, "cuda,nvtx", [0], False),
        ("new-matched-two-ranks", new_binary, "cuda,nvtx,osrt", [0, 1], True),
        ("new-minimal-one-rank", new_binary, "cuda,nvtx", [0], False),
    ):
        summary["cases"].append(run_case(root, label, binary, trace, ranks, nvtx))
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        Path("/results/nsys-debug-summary.json").write_text(json.dumps(summary, indent=2))
    if not any(case["success"] for case in summary["cases"]):
        raise RuntimeError("No configuration produced a complete real CUDA/NCCL trace")


def main():
    """Dispatch node driver, per-rank wrapper, or actual reproduction workload."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["driver", "wrapper", "worker"])
    parser.add_argument("run_name", nargs="?")
    args = parser.parse_args()
    if args.mode == "worker":
        worker()
    elif args.mode == "wrapper":
        wrapper()
    else:
        driver(args.run_name)


if __name__ == "__main__":
    main()
