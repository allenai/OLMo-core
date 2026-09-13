"""Bounded two-GPU correctness and paired ownership-overhead qualification.

No training/checkpoint writes; results are emitted directly to Beaker logs.
"""

import inspect
import json
import statistics
import subprocess
import sys
import time
import types

import torch

from olmo_core.ops import rounded_wgrad


def benchmark():
    # Reconstruct only the previous ownership bookkeeping in a private module.
    # The actual QuACK kernel and its compiler/cache are shared, never patched.
    source = inspect.getsource(rounded_wgrad._RoundedWeightGemm)
    start = source.index("        # Checkpoint/saved-tensor hooks")
    end = source.index("        ctx.transpose", start)
    source = source[:start] + source[end:]
    start = source.index("        owner = ctx.weight_owner()")
    end = source.index("        if ctx.transpose:", start)
    source = (
        source[:start]
        + "        destination, done = weight._olmo_profile_begin_external_grad(weight)\n"
        + source[end:]
    )
    old = types.ModuleType("legacy_rounded_wgrad_ownership")
    old.__dict__.update(
        torch=torch, F=torch.nn.functional, rounded_wgrad_add=rounded_wgrad.rounded_wgrad_add
    )
    exec(compile(source, "<legacy-rounded-wgrad-ownership>", "exec"), old.__dict__)
    torch.cuda.set_device(0)
    torch.manual_seed(832)
    rows, experts, width, hidden = 65536, 512, 512, 1024
    counts = torch.full((experts,), rows // experts, device="cuda", dtype=torch.int32)
    cumulative = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
    for transpose, inputs, outputs in ((True, width, hidden * 2), (False, hidden, width)):
        shape = (experts, outputs, inputs) if transpose else (experts, inputs, outputs)
        weight = torch.nn.Parameter(torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.02)
        destination = torch.zeros_like(weight, dtype=torch.float32)
        done_calls = [0]

        def done():
            done_calls[0] += 1

        def begin(owner):
            assert owner is weight and owner.grad is None
            return destination, done

        weight._olmo_profile_begin_external_grad = begin
        x = torch.randn(rows, inputs, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn(rows, outputs, device="cuda", dtype=torch.bfloat16)
        functions = dict(old=old._RoundedWeightGemm, fixed=rounded_wgrad._RoundedWeightGemm)

        def execute(label):
            x.grad = None
            y = functions[label].apply(x, weight, cumulative, transpose, None, None)
            y.backward(dy)
            return y

        snapshots = []
        for label in functions:
            destination.zero_()
            y = execute(label)
            snapshots.append((y.detach().clone(), x.grad.clone(), destination.clone()))
        torch.testing.assert_close(snapshots[0], snapshots[1], rtol=0, atol=0)
        del snapshots, y
        for label in functions:
            for _ in range(5):
                execute(label)
        torch.cuda.synchronize()
        timings = {label: [] for label in functions}
        for _ in range(5):
            for label in ("old", "fixed", "fixed", "old"):
                start = time.perf_counter()
                for _ in range(10):
                    execute(label)
                torch.cuda.synchronize()
                timings[label].append((time.perf_counter() - start) * 100)
        medians = {k: statistics.median(v) for k, v in timings.items()}
        print(
            "OWNERSHIP_BENCH "
            + json.dumps(
                dict(
                    transpose=transpose,
                    rows=rows,
                    experts=experts,
                    exact_output_dgrad_wgrad=True,
                    median_ms=medians,
                    samples_ms=timings,
                    fixed_over_old=medians["fixed"] / medians["old"],
                )
            ),
            flush=True,
        )
        assert done_calls[0] == 212
        # A generous noise bound catches a substantial unexpected regression.
        assert medians["fixed"] < medians["old"] * 1.05, medians
        del weight, destination, x, dy


def main():
    assert torch.cuda.device_count() >= 2
    assert torch.cuda.get_device_capability()[0] == 10
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "src/test/ops/rounded_wgrad_checkpoint_test.py",
            "src/test/nn/parallel/swiglu_pairwise_test.py",
            "src/test/nn/moe/emo_document_pool_test.py::test_mixed_document_masks",
            "src/test/nn/moe/emo_document_pool_test.py::test_document_masks_at_cuda_grid_boundary",
        ],
        check=True,
    )
    benchmark()
    print("LC_RECOMPUTE_QUALIFICATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
