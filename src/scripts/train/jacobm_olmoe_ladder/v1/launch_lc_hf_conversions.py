#!/usr/bin/env python3
"""Launch resumable HF conversions for completed V1 long-context models."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO = Path("/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/OLMo-core")
DDP_ROOT = Path("/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp")

WORKSPACE = "ai2/OLMo-3-moe-experiments"
BUDGET = "ai2/oe-other"
CLUSTER = "ai2/jupiter"
IMAGE = "tianhuat/olmo-core-torch212-2404-cu130"
NAME = "olmoe3-v1-completed-lc-hf-conversions"


@dataclass(frozen=True)
class Model:
    key: str
    run_name: str
    final_step: int

    @property
    def source(self) -> Path:
        return DDP_ROOT / "long-context" / self.run_name / f"step{self.final_step}"

    @property
    def output(self) -> Path:
        return DDP_ROOT / "hf" / "long-context" / self.run_name / f"step{self.final_step}"


MODELS = {
    model.key: model
    for model in (
        Model("275m-baseline", "lc-275m-baseline-cx8-mt2e-4-lc1e-4-64k-r2", 47_684),
        Model(
            "275m-integration-deep",
            "lc-275m-integration-deep-cx8-mt1p6e-4-lc8e-5-64k-r1",
            47_684,
        ),
        Model(
            "275m-integration-wide",
            "lc-275m-integration-wide-cx8-mt1p6e-4-lc8e-5-64k-r1",
            47_684,
        ),
        Model("480m-baseline", "lc-480m-baseline-cx8-mt8e-5-lc4e-5-64k-r1", 31_790),
        Model(
            "480m-integration-deep",
            "lc-480m-integration-deep-cx8-mt8e-5-lc4e-5-64k-r1",
            31_790,
        ),
        Model(
            "480m-integration-wide",
            "lc-480m-integration-wide-cx8-mt8e-5-lc4e-5-64k-r1",
            31_790,
        ),
        Model(
            "810m-integration-wide",
            "lc-810m-integration-wide-cx8-mt4e-5-lc2e-5-64k-r1",
            23_842,
        ),
        Model(
            "810m-baseline",
            "lc-810m-baseline-cx8-mt4e-5-lc2e-5-64k-r1",
            23_842,
        ),
        Model(
            "1p2b-integration-wide",
            "lc-1p2b-integration-wide-cx8-mt4e-5-lc2e-5-64k-r1",
            23_842,
        ),
        Model(
            "1p2b-baseline",
            "lc-1p2b-baseline-cx8-mt4e-5-lc2e-5-64k-r1",
            23_842,
        ),
    )
}


def task(model: Model) -> str:
    return f"""  - name: {model.key}
    image:
      beaker: {IMAGE}
    command: [/bin/bash, -lc]
    arguments:
      - |
        set -euo pipefail
        SOURCE={model.source}
        OUTPUT={model.output}
        REPO={REPO}
        MARKER="${{OUTPUT}}/conversion_complete.json"

        mkdir -p /results "${{OUTPUT}}" /tmp/olmoe3-lc-hf-convert
        cd "${{REPO}}"
        export PYTHONPATH="${{REPO}}/src:${{PYTHONPATH:-}}"

        if [[ -f "${{MARKER}}" ]] || \
           [[ -f "${{OUTPUT}}/model.safetensors" ]] || \
           [[ -f "${{OUTPUT}}/model.safetensors.index.json" ]]; then
          echo "HF weights already exist; validating the resumable output."
        else
          python -m olmo_core.nn.moe.v2.hf.convert_checkpoint \
            --ckpt-path "${{SOURCE}}" \
            --output-path "${{OUTPUT}}" \
            --work-dir /tmp/olmoe3-lc-hf-convert
        fi

        python - "${{SOURCE}}" "${{OUTPUT}}" "${{REPO}}" <<'PY'
        import datetime
        import hashlib
        import json
        import subprocess
        import sys
        from pathlib import Path

        source, output, repo = map(Path, sys.argv[1:])
        required = [
            output / "config.json",
            output / "tokenizer.json",
            output / "modeling_olmo3moe.py",
            output / "configuration_olmo3moe.py",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"HF conversion is incomplete; missing: {{missing}}")

        index_path = output / "model.safetensors.index.json"
        single_weight_path = output / "model.safetensors"
        if index_path.is_file():
            with index_path.open() as f:
                index = json.load(f)
            indexed_files = sorted(set(index["weight_map"].values()))
            weight_tensor_count = len(index["weight_map"])
        elif single_weight_path.is_file():
            from safetensors import safe_open

            indexed_files = [single_weight_path.name]
            with safe_open(single_weight_path, framework="pt", device="cpu") as weights:
                weight_tensor_count = len(list(weights.keys()))
        else:
            raise RuntimeError("HF conversion has neither single-file nor sharded safetensors")

        absent_indexed = [name for name in indexed_files if not (output / name).is_file()]
        if absent_indexed:
            raise RuntimeError(f"Missing indexed weight files: {{absent_indexed}}")

        sha = hashlib.sha256()
        for path in sorted(output.glob("*.py")) + [output / "config.json"]:
            sha.update(path.name.encode())
            sha.update(path.read_bytes())

        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        manifest = dict(
            status="COMPLETE",
            source_checkpoint=str(source),
            output_checkpoint=str(output),
            converter="olmo_core.nn.moe.v2.hf.convert_checkpoint",
            olmo_core_commit=commit,
            completed_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            weight_files=indexed_files,
            weight_bytes=sum((output / name).stat().st_size for name in indexed_files),
            weight_tensor_count=weight_tensor_count,
            code_and_config_sha256=sha.hexdigest(),
            mapping_validation="converter requires every HF parameter to be mapped",
        )
        marker = output / "conversion_complete.json"
        marker.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n")
        (Path("/results") / "conversion_complete.json").write_text(marker.read_text())
        print(json.dumps(manifest, indent=2, sort_keys=True))
        PY

        find "${{OUTPUT}}" -maxdepth 1 -type f -printf '%f %s bytes\\n' \
          | sort | tee /results/output_files.txt
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
      sharedMemory: 32GiB
    context:
      priority: urgent
      preemptible: true
    constraints:
      cluster:
        - {CLUSTER}
    timeout: 4h
"""


def build_spec(models: list[Model]) -> str:
    return f"version: v2\nbudget: {BUDGET}\ntasks:\n" + "".join(task(model) for model in models)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(MODELS),
        help="Model to convert; repeat as needed (default: every completed model missing HF).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = [MODELS[key] for key in args.model] if args.model else list(MODELS.values())
    if not args.model:
        selected = [model for model in selected if model.source.is_dir()]
    missing = [str(model.source) for model in selected if not model.source.is_dir()]
    if missing:
        raise RuntimeError(f"Source checkpoints do not exist: {missing}")

    spec = build_spec(selected)
    if args.dry_run:
        print(spec)
        return

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as file:
        file.write(spec)
        spec_path = Path(file.name)
    try:
        subprocess.run(
            [
                "beaker",
                "experiment",
                "create",
                str(spec_path),
                "--workspace",
                WORKSPACE,
                "--name",
                NAME,
            ],
            check=True,
        )
    finally:
        spec_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
