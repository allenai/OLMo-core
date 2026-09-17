"""Single-GPU validation of document-end tiled attention; writes artifacts to /results.

Runs CUDA numerical tests, 20 synthetic optimizer steps using the OLMo3-190M
architecture with a small vocabulary, and GPU timing/memory benchmarks. No data
mounts, checkpoint downloads, external tracking, or model checkpoint writes.
"""

import argparse
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn.functional as F

from olmo_core.nn.attention import AttentionBackendName, AttentionType
from olmo_core.nn.transformer import TransformerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/results"))
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU validation requires CUDA; refusing a skipped-test success")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(101)
    torch.set_num_threads(4)
    environment = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    (root / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps(environment), flush=True)

    junit = root / "cuda-tests.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "src/test/nn/attention/landmark_document_end_tiled_test.py",
        "-k",
        "cuda",
        f"--junitxml={junit}",
    ]
    subprocess.run(command, check=True)
    cases = ET.parse(junit).findall(".//testcase")
    if len(cases) < 3 or any(c.find("skipped") is not None for c in cases):
        raise RuntimeError("Expected at least three executed CUDA tests with zero skips")

    # Same width/depth as the 190M base smoke-test recipe. Restrict vocabulary
    # and sequence length so this validates the backend without a data dependency.
    cfg = TransformerConfig.olmo3_190M(
        vocab_size=256,
        attn_backend=AttentionBackendName.torch,
        sliding_window=None,
        n_kv_heads=3,
    )
    cfg.block.sequence_mixer.name = AttentionType.document_end_compressive_landmark
    cfg.document_end_landmark_attention = dict(
        doc_start_id=10,
        doc_end_id=11,
        landmark_token_id=12,
        eos_id=13,
        pad_id=14,
    )
    model = cfg.build(init_device="cpu")
    model.init_weights(device=torch.device("cuda"))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    tokens = [1, 10] + [20] * 37 + [11, 12, 10] + [21] * 19 + [11, 12]
    tokens += [22] * (129 - len(tokens))
    ids = torch.tensor([tokens], device="cuda")
    target = ids[:, 1:].clone()
    target[(target == 10) | (target == 11) | (target == 12)] = -100
    losses = []
    start = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids[:, :-1])
            loss = F.cross_entropy(logits.flatten(0, 1).float(), target.flatten())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 0 or (step + 1) % 5 == 0:
            print(
                json.dumps({"step": step + 1, "loss": losses[-1], "grad_norm": float(norm)}),
                flush=True,
            )
    if args.steps > 1 and losses[-1] >= losses[0]:
        raise RuntimeError("Synthetic training loss did not decrease")
    training = {
        "steps": args.steps,
        "losses": losses,
        "num_params": model.num_params,
        "elapsed_seconds": time.perf_counter() - start,
    }
    (root / "training.json").write_text(json.dumps(training, indent=2) + "\n")
    del model, optimizer, logits, loss
    torch.cuda.empty_cache()

    for length, backend in [(256, "both"), (4096, "tiled")]:
        path = root / f"benchmark-{length}.json"
        with path.open("w") as output:
            subprocess.run(
                [
                    sys.executable,
                    "src/scripts/benchmarks/document_end_attention.py",
                    "--device",
                    "cuda",
                    "--dtype",
                    "bfloat16",
                    "--length",
                    str(length),
                    "--backend",
                    backend,
                    "--repeats",
                    "3",
                ],
                check=True,
                stdout=output,
            )
        print(path.read_text(), flush=True)
    (root / "SUCCESS").write_text("CUDA numerical tests, 20-step training, and benchmarks passed\n")
    print("GPU VALIDATION PASSED", flush=True)


if __name__ == "__main__":
    main()
