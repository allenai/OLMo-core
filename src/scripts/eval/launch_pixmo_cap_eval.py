"""
Launch the Molmo2 PixMo-Cap caption eval on Beaker (one experiment per model).

Submits directly via beaker-py (no gantry library required).  The container
uses the same gantry entrypoint as training jobs — it clones the repo, installs
it, then runs the eval script with ``python`` (single-GPU, no torchrun).

Example::

    python src/scripts/eval/launch_pixmo_cap_eval.py allenai/Molmo2-4B allenai/Molmo2-8B allenai/Molmo2-O-7B
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import List

from olmo_core.eval.cap_f1_judge import DEFAULT_JUDGE_MODEL

WEKA_MOUNT = "/weka/oe-training-default"
OUTPUT_DIR = f"{WEKA_MOUNT}/jasonr/molmo2-pixmo-cap-eval"

# Beaker image and gantry runtime dataset — copied from existing DLC eval experiments.
BEAKER_IMAGE = "01KAY1JXJFCPQ1J5W61E2D2Q8V"  # OLMoCoreBeakerImage.stable
GANTRY_DATASET = "01KP021DZFY1CD8EZ8CBTPGZXY"  # gantry runtime, mounted at /gantry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", nargs="+", help="HF model ids (e.g. allenai/Molmo2-4B).")
    p.add_argument("--cluster", default="ai2/jupiter")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--workspace", default="ai2/oe-encoder")
    p.add_argument("--budget", default="ai2/oe-other")
    p.add_argument("--priority", default="normal")
    p.add_argument(
        "--preemptible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run as preemptible/unallocated (default: True).",
    )
    p.add_argument(
        "--openai-secret",
        default="JASONR_OPENAI_API",
        help="Beaker secret holding the OpenAI API key.",
    )
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--split", default="validation", choices=("train", "validation"))
    p.add_argument("--limit", type=int, default=0, help="0 = full split (2048 examples).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(".", "-").lower()


def _git_ref() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def _git_branch() -> str:
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()


def _unique_suffix() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


def build_spec(args: argparse.Namespace, model_id: str) -> dict:
    slug = _slug(model_id)
    run_id = _unique_suffix()
    name = f"molmo2-pixmo-cap-{slug}-{run_id}"
    out_json = f"{OUTPUT_DIR}/{slug}-{run_id}/eval.json"

    judge_cache = f"{OUTPUT_DIR}/judge-cache-{args.judge_model.replace('/', '_')}"
    cmd_args: List[str] = [
        "python",
        "src/scripts/eval/molmo2_caption_eval.py",
        "--model", model_id,
        "--split", args.split,
        "--max-new-tokens", str(args.max_new_tokens),
        "--judge-model", args.judge_model,
        "--judge-cache-dir", judge_cache,
        "--out", out_json,
    ]
    # Always pass --limit explicitly so the eval script's default of 64 never silently kicks in.
    cmd_args += ["--limit", str(args.limit)]

    env_vars = [
        {"name": "GANTRY_VERSION", "value": "3.7.0"},
        {"name": "GITHUB_REPO", "value": "jason718/OLMo-core"},
        {"name": "GIT_REF", "value": _git_ref()},
        {"name": "GIT_BRANCH", "value": _git_branch()},
        {"name": "GANTRY_TASK_NAME", "value": "pixmo-cap-eval"},
        {"name": "RESULTS_DIR", "value": "/results"},
        {"name": "GANTRY_RUNTIME_DIR", "value": "/gantry-runtime"},
        {"name": "GANTRY_EXEC_METHOD", "value": "exec"},
        {"name": "GANTRY_DEFAULT_PYTHON_VERSION", "value": "3.12"},
        {"name": "GANTRY_USE_SYSTEM_PYTHON", "value": "1"},
        {"name": "GANTRY_PYTHON_MANAGER", "value": "uv"},
        {"name": "GANTRY_UV_ALL_EXTRAS", "value": "1"},
        {"name": "MOLMO_DATA_DIR", "value": f"{WEKA_MOUNT}/mm-olmo"},
        {"name": "HF_HOME", "value": f"{WEKA_MOUNT}/jasonr/hf-home"},
        {"name": "OPENAI_API_KEY", "secret": args.openai_secret},
        {"name": "LOG_FILTER_TYPE", "value": "local_rank0_only"},
        {"name": "OMP_NUM_THREADS", "value": "8"},
        {"name": "R2_PROFILE", "value": "R2"},
        {"name": "S3_PROFILE", "value": "S3"},
        {"name": "WEKA_PROFILE", "value": "WEKA"},
        {"name": "NUM_NODES", "value": "1"},
        {"name": "OLMO_CORE_VERSION", "value": "2.5.0"},
        {"name": "FORCE_COLOR", "value": "1"},
        {"name": "PYTORCH_KERNEL_CACHE_PATH", "value": "/root/.cache/torch/kernels"},
        {"name": "OLMO_SHARED_FS", "value": "1"},
    ]

    task_spec = {
        "name": "pixmo-cap-eval",
        "image": {"beaker": BEAKER_IMAGE},
        "command": ["bash", "/gantry/entrypoint.sh"],
        "arguments": cmd_args,
        "envVars": env_vars,
        "datasets": [
            {"mountPath": "/gantry", "source": {"beaker": GANTRY_DATASET}},
            {"mountPath": WEKA_MOUNT, "source": {"weka": "oe-training-default"}},
        ],
        "result": {"path": "/results"},
        "resources": {"gpuCount": args.num_gpus, "sharedMemory": "10 GiB"},
        # minRuntime and autoResume are incompatible with preemptible=True.
        "context": {
            "priority": args.priority,
            "preemptible": args.preemptible,
        },
        "constraints": {"cluster": [args.cluster]},
        "hostNetworking": False,
        "propagateFailure": False,
        "propagatePreemption": False,
    }

    return {
        "name": name,
        "workspace": args.workspace,
        "budget": args.budget,
        "description": (
            f"Molmo2 PixMo-Cap caption eval: model={model_id}, split={args.split}, "
            f"n={'all' if args.limit == 0 else args.limit}. Output → {out_json}."
        ),
        "tasks": [task_spec],
        "out_json": out_json,  # stash for printing, stripped before submission
    }


def main() -> int:
    args = parse_args()
    for model_id in args.models:
        spec = build_spec(args, model_id)
        out_json = spec.pop("out_json")

        if args.dry_run:
            print(f"\n=== {model_id} ===")
            print(json.dumps(spec, indent=2))
            continue

        # Submit via beaker CLI — avoids needing gantry installed locally.
        # The CLI spec file contains only tasks/budget/description; name and
        # workspace are passed as flags.
        import os
        import tempfile

        exp_name = spec["name"]
        file_spec = {
            "tasks": spec["tasks"],
            "budget": spec["budget"],
            "description": spec["description"],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(file_spec, tf)
            spec_path = tf.name
        try:
            result = subprocess.run(
                ["beaker", "experiment", "create", spec_path,
                 "--workspace", args.workspace, "--name", exp_name],
                capture_output=True, text=True,
            )
        finally:
            os.unlink(spec_path)
        if result.returncode != 0:
            print(f"[error] {model_id}:\n{result.stderr}", file=sys.stderr)
            return 1

        # Extract experiment ID from output.
        output = result.stdout.strip()
        exp_id = ""
        for line in output.splitlines():
            # beaker prints something like: "Experiment 01XYZ created …"
            if "01" in line:
                for token in line.split():
                    if token.startswith("01") and len(token) > 10:
                        exp_id = token.rstrip(".")
                        break
        url = f"https://beaker.org/orgs/ai2/workspaces/oe-encoder/work/{exp_id}" if exp_id else "?"
        print(f"\n[launch] {model_id} → {spec['name']}")
        print(f"  experiment id : {exp_id}")
        print(f"  url           : {url}")
        print(f"  output        : {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
