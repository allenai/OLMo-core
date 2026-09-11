import hashlib
import importlib.util
import json
import time
from pathlib import Path
from unittest import mock

import torch
from olmo_core.kernels import swiglu as candidate
from triton.runtime import cache, jit

spec = importlib.util.spec_from_file_location(
    "swiglu_before_runtime_rows", "/validation/swiglu_before_runtime_rows.py"
)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
capacities = (1, 16, 17, 32, 37, 41, 1024, 16383, 16384, 16385, 21023, 31007)
original_compile = jit.JITFunction._do_compile
original_put = cache.FileCacheManager.put
results = []
for mode in (False, True):
    inputs = []
    torch.manual_seed(17)
    for rows in capacities:
        x = torch.randn(rows, 1904, device="cuda", dtype=torch.bfloat16)
        count = torch.tensor(max(0, rows - 3), device="cuda", dtype=torch.long)
        inputs.append((x, count))
    expected = []
    for name, module in [("baseline", baseline), ("candidate", candidate)]:
        observations = []
        for phase in ("first", "repeat"):
            misses = []
            writes = []
            times = []
            kernel_ms = []

            def observe(fn, *args, **kwargs):
                start = time.perf_counter()
                try:
                    return original_compile(fn, *args, **kwargs)
                finally:
                    misses.append(
                        {
                            "kernel": fn.fn.__module__ + "." + fn.fn.__name__,
                            "seconds": time.perf_counter() - start,
                        }
                    )

            def put(manager, data, filename, binary=True):
                result = original_put(manager, data, filename, binary=binary)
                writes.append(filename)
                return result

            with (
                mock.patch.object(jit.JITFunction, "_do_compile", observe),
                mock.patch.object(cache.FileCacheManager, "put", put),
            ):
                for i, (x, count) in enumerate(inputs):
                    out = torch.full((x.shape[0], 952), 77.0, device="cuda", dtype=torch.bfloat16)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    module.swiglu_valid_prefix(x, count, out=out, match_eager_rounding=mode)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)
                    if name == "baseline" and phase == "first":
                        expected.append(out.clone())
                    else:
                        assert torch.equal(expected[i], out), (mode, name, phase, capacities[i])
                    a, b = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    a.record()
                    for _ in range(30):
                        module.swiglu_valid_prefix(x, count, out=out, match_eager_rounding=mode)
                    b.record()
                    torch.cuda.synchronize()
                    kernel_ms.append(a.elapsed_time(b) / 30)
            observations.append(
                {
                    "phase": phase,
                    "call_wall_seconds": times,
                    "total_call_wall_seconds": sum(times),
                    "jit_misses": misses,
                    "cubin_writes": sum(f.endswith(".cubin") for f in writes),
                    "warm_launch_stream_ms": kernel_ms,
                    "warm_launch_scope": "CUDA stream elapsed per repeated Python launch; includes launch gaps, not isolated kernel active time",
                }
            )
        results.append(
            {"mode_eager_rounding": mode, "implementation": name, "passes": observations}
        )
report = {
    "passed": True,
    "device": torch.cuda.get_device_name(),
    "capacities": capacities,
    "hidden": 952,
    "all_baseline_candidate_outputs_bitwise_equal": True,
    "scope": "Local4090 forward-only standalone kernel; no model or optimizer update; changing capacities and fixed width. Both arithmetic modes preserve baseline bytes including untouched tails.",
    "baseline_source_sha256": hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(),
    "candidate_source_sha256": hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
    "results": results,
}
Path("/validation/swiglu-runtime-rows-probe.json").write_text(json.dumps(report, indent=2) + "\n")
for r in results:
    print(
        r["mode_eager_rounding"],
        r["implementation"],
        [
            (q["phase"], len(q["jit_misses"]), q["cubin_writes"], q["total_call_wall_seconds"])
            for q in r["passes"]
        ],
    )
