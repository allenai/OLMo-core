#!/usr/bin/env python3
"""Launch one-GPU Beaker jobs to convert ladder checkpoints to HF format."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


REPO = Path("/weka/oe-adapt-default/jacobm/olmoe3/OLMo-core")
DEFAULT_MANIFEST = REPO / "src/scripts/train/jacobm_olmoe_ladder/eval_1p2b_integration_cx1_cx2_targets.jsonl"
WORKSPACE = "ai2/OLMo-3-moe-experiments"
BUDGET = "ai2/oe-other"
IMAGE = "tianhuat/olmo-core-torch211-2404-cu128"
CLUSTER = "ai2/jupiter"


def load_targets(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def spec_for_target(target: dict, manifest: Path) -> str:
    name_suffix = os.environ.get("NAME_SUFFIX", "")
    name = f"olmoe3-hf-convert-{target['train_name']}-step{target['step']}{name_suffix}"
    only = target["train_name"]
    return f"""version: v2
tasks:
  - name: main
    image:
      beaker: {IMAGE}
    command: [/bin/bash, -lc]
    arguments:
      - |
        set -euxo pipefail
        REPO={REPO}
        cd "${{REPO}}"
        mkdir -p /results /tmp/olmoe3-hf-convert-work /tmp/olmoe3-hf-convert-logs
        export PYTHONPATH="${{REPO}}/src:${{PYTHONPATH:-}}"
        if command -v python3 >/dev/null 2>&1; then
          PYTHON_BIN=$(command -v python3)
        else
          PYTHON_BIN=$(command -v python)
        fi
        "${{PYTHON_BIN}}" --version | tee /results/python_version.txt
        "${{PYTHON_BIN}}" - <<'PY' | tee /results/dependency_check.txt
        import sys
        print("python", sys.executable)
        import torch, transformers, safetensors
        print("deps-ok", torch.__version__, transformers.__version__, safetensors.__version__)
        PY
        "${{PYTHON_BIN}}" src/scripts/train/jacobm_olmoe_ladder/convert_275m_eval_targets.py \\
          --manifest {manifest} \\
          --only {only} \\
          --work-dir /tmp/olmoe3-hf-convert-work \\
          --log-dir /results/convert-logs
        find /results -maxdepth 3 -type f -print | sort | tee /results/result_files.txt
    datasets:
      - mountPath: /weka/oe-adapt-default
        source:
          weka: oe-adapt-default
      - mountPath: /weka/oe-training-default
        source:
          weka: oe-training-default
    result:
      path: /results
    resources:
      gpuCount: 1
      sharedMemory: 10GiB
    context:
      priority: urgent
      minRuntime: 0s
    constraints:
      cluster:
        - {CLUSTER}
    timeout: 8h
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = load_targets(args.manifest)
    if args.only:
        targets = [t for t in targets if any(s in t["train_name"] for s in args.only)]

    print(f"targets {len(targets)}")
    for target in targets:
        name_suffix = os.environ.get("NAME_SUFFIX", "")
        name = f"olmoe3-hf-convert-{target['train_name']}-step{target['step']}{name_suffix}"
        spec = spec_for_target(target, args.manifest)
        if args.dry_run:
            print(f"--- {name} ---")
            print(spec)
            continue

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(spec)
            spec_path = Path(f.name)

        cmd = [
            "beaker",
            "experiment",
            "create",
            str(spec_path),
            "--workspace",
            WORKSPACE,
            "--name",
            name,
        ]
        result = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout.strip())
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout)


if __name__ == "__main__":
    main()
