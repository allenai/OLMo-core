"""One-GPU runtime qualification and KDA microbenchmark at the exact small-model shape."""

import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from kernel_fun._common import support
from kernel_fun.kda import is_supported

from olmo_core.distributed.checkpoint import RemoteFileSystemReader
from olmo_core.nn.attention import KimiDeltaAttentionConfig
from olmo_core.nn.transformer import InitMethod


def main():
    """Check Nsight, source metadata, and full-layer forward/backward numerical agreement."""
    out = Path("/results")
    out.mkdir(exist_ok=True)
    root = Path(
        "/weka/olmo-3p5-checkpoints/production-cbs/olmoe3-small-cbs-16mi-from-step4000-lr1p85em3-uploader-r1/step7500"
    )
    metadata = RemoteFileSystemReader(str(root / "model_and_optim")).read_metadata()
    gains = {
        k: list(v.size)
        for k, v in metadata.state_dict_metadata.items()
        if ".q_norm.weight." in k or ".k_norm.weight." in k
    }
    print("Checkpoint QK gain metadata:", json.dumps(gains), flush=True)
    assert len([key for key in gains if key.endswith(".main")]) == 4, gains
    assert all(shape == [128] for key, shape in gains.items() if not key.endswith(".step")), gains

    nsys = "/opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/nsys"
    smoke = "import torch; x=torch.ones(32,device='cuda'); torch.cuda.cudart().cudaProfilerStart(); x=x+1; torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()"
    subprocess.run(
        [
            nsys,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--kill=none",
            "--output",
            str(out / "nsys-smoke"),
            sys.executable,
            "-c",
            smoke,
        ],
        check=True,
    )
    assert list(out.glob("nsys-smoke*.nsys-rep")), "Nsight produced no report"

    torch.manual_seed(42)
    model = KimiDeltaAttentionConfig(
        n_heads=8,
        n_v_heads=8,
        head_dim=128,
        expand_v=2.0,
        allow_neg_eigval=True,
        use_cute_kernel=True,
    ).build(1024, layer_idx=0, n_layers=16, init_device="cuda")
    model.init_weights(init_method=InitMethod.normal, d_model=1024, block_idx=0, num_blocks=16)
    x = torch.randn(4, 8192, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    q = torch.empty(4, 8192, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.empty(4, 8192, 8, 256, device="cuda", dtype=torch.bfloat16)
    print("Default KDA dispatch:", is_supported(q, v), flush=True)
    del q, v
    report = {
        "gpu": torch.cuda.get_device_name(),
        "shape": [4, 8192, 1024],
        "gains": gains,
        "arms": {},
    }
    reference = None
    default_floor = support.MIN_CTAS
    for arm in ("fla", "new-default", "new-cutoff-128"):
        model.use_cute_kernel = arm != "fla"
        for conv in (model.q_conv1d, model.k_conv1d, model.v_conv1d):
            conv.use_cute_kernel = arm != "fla"
        # Performance-only diagnostic; never changes the package or the default training path.
        support.MIN_CTAS = 128 if arm == "new-cutoff-128" else default_floor

        def iteration():
            model.zero_grad(set_to_none=True)
            x.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model(x)
            y.backward(dy)
            return y

        y = iteration()
        values = {"output": y.detach().float().cpu(), "input_grad": x.grad.float().cpu()}
        values.update(
            {
                name: p.grad.detach().float().cpu()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
        )
        errors = {}
        if reference is None:
            reference = values
        else:
            for name, actual in values.items():
                expected = reference[name]
                assert torch.isfinite(actual).all(), (arm, name)
                relative_rms = float((actual - expected).norm() / expected.norm().clamp_min(1e-12))
                errors[name] = relative_rms
            assert max(errors.values()) < 0.03, (arm, errors)
        del values, y
        for _ in range(4):
            iteration()
        times = []
        for _ in range(20):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            iteration()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        report["arms"][arm] = {
            "median_ms": statistics.median(times),
            "mean_ms": statistics.mean(times),
            "relative_l2_errors": errors,
        }
        (out / "kda-microbenchmark.json").write_text(json.dumps(report, indent=2))
        print("KDA_ARM", arm, json.dumps(report["arms"][arm]), flush=True)
    support.MIN_CTAS = default_floor


if __name__ == "__main__":
    main()
