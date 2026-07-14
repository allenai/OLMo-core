#!/usr/bin/env python3
"""Convert clean 275M ladder eval targets to Hugging Face checkpoints.

This script intentionally uses the curated manifest in
``eval_275m_clean_targets.jsonl`` rather than scanning every completed run. The
manifest contains only observed-best, final 275M checkpoints for canonical
batch-size runs, excluding diagnostic curiosities and incorrect-batch artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
MANIFEST = Path(__file__).with_name("eval_275m_clean_targets.jsonl")
HF_ROOT = Path(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/hf-checkpoints"
)
EVAL_RESULTS_ROOT = Path(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/eval-results"
)
CHECKPOINT_INDEX = EVAL_RESULTS_ROOT / "CHECKPOINTS.jsonl"


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_targets(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def lr_from_target(target: dict[str, Any]) -> str:
    return str(target["lr"])


def cx_from_target(target: dict[str, Any]) -> str:
    return str(target["cx"]).removeprefix("Cx")


def batch_for_cx(cx: str) -> tuple[str, str, str]:
    if cx == "1":
        return "32", "2", "8"
    if cx == "2":
        return "48", "2", "8"
    if cx == "4":
        return "64", "4", "8"
    if cx == "8":
        return "96", "4", "8"
    raise ValueError(f"Unsupported Cx: {cx}")


def training_settings_for_target(target: dict[str, Any]) -> tuple[str, str, str, str]:
    cx = cx_from_target(target)
    default_global_batch, default_gpus, default_micro = batch_for_cx(cx)
    return (
        str(target.get("global_batch_size_seq", default_global_batch)),
        str(target.get("num_nodes", 1)),
        str(target.get("gpus_per_node", default_gpus)),
        str(target.get("micro_batch_size", default_micro)),
    )


def infer_script_and_args(target: dict[str, Any]) -> tuple[Path, list[str]]:
    train_name = target["train_name"]
    variant = target["variant"]
    cx = cx_from_target(target)
    lr = lr_from_target(target)
    global_batch, num_nodes, gpus, micro = training_settings_for_target(target)
    model_size = str(target.get("model_size", "275m"))
    common = [
        f"--model-size={model_size}",
        "--save-folder=/tmp/reconstructed-config-only",
        f"--name={train_name}",
        "--data-root=s3://ai2-llm",
        f"--lr={lr}",
        f"--chinchilla-multiple={cx}",
        f"--global-batch-size-seq={global_batch}",
        f"--num-nodes={num_nodes}",
        f"--gpus-per-node={gpus}",
        f"--micro-batch-size={micro}",
        "--ep-dim=1",
        "--ladder-evals",
        "--eval-task-set=fast",
        "--eval-interval=2000",
        "--save-interval=999999999",
        "--ephemeral-save-interval=500",
        "--no-pre-train-checkpoint",
        f"--tag={train_name}",
    ]

    if variant.lower().startswith("qwen-like active matched 4.5d") or variant.lower().startswith("qwen-like true 3.0d") or variant.startswith("active matched 4.5d") or variant.startswith("true 3.0d"):
        script = ROOT / "src/scripts/train/jacobm_olmoe_ladder/experiments/qwen3_like/qwen3_like_ladder.py"
        qwen = (
            "active_matched"
            if "active matched 4.5d" in variant.lower()
            else "true_3d_depth_matched"
        )
        return script, [*common, f"--qwen3-like={qwen}"]

    if variant.startswith("integration wide") or variant.startswith("integration deep"):
        script = ROOT / "src/scripts/train/jacobm_olmoe_ladder/experiments/integration/integration_ladder.py"
        if variant.startswith("integration deep"):
            integration = "deep_256e8k"
        elif "top16" in variant.lower() or "256e/top16" in variant.lower():
            integration = "wide_256e16k"
        else:
            integration = "wide_256e8k"
        return script, [*common, f"--integration-config={integration}"]

    script = ROOT / "src/scripts/train/jacobm_olmoe_ladder/moe_a0_ladder.py"
    if variant.startswith("coarse 24E/top2"):
        return script, [*common, "--expert-geometry=coarse_24e_top2"]
    if variant.startswith("fine 96E/top8"):
        return script, [*common, "--expert-geometry=fine_96e_top8"]
    if variant.startswith("high total 96E/top4"):
        return script, [*common, "--total-sparsity=high_total_96e_top4"]
    if variant.startswith("huge total 192E/top4"):
        return script, [*common, "--total-sparsity=huge_total_192e_top4"]
    if variant.startswith("no shared"):
        return script, [*common, "--shared-expert-config=no_shared_matched_active"]
    if variant.startswith("dense0"):
        return script, [*common, "--dense-schedule=dense0_shared"]
    if variant.startswith("dense2"):
        return script, [*common, "--dense-schedule=dense2_shared"]
    if variant.startswith("dense4"):
        return script, [*common, "--dense-schedule=dense4_shared"]
    if variant.startswith("baseline 48E/top4"):
        return script, common

    raise ValueError(f"Cannot infer config flags for variant: {variant!r}")


def reconstruct_config(target: dict[str, Any], config_path: Path) -> None:
    script, argv = infer_script_and_args(target)
    module_name = f"_olmoe_reconstruct_{re.sub('[^0-9A-Za-z_]', '_', target['train_name'])}"
    module = load_module(script, module_name)
    parser = module.get_parser()
    opts, overrides = parser.parse_known_args(argv)
    config = module.build_config(opts, overrides)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        json.dump(config.as_config_dict(), f)


def run_convert(target: dict[str, Any], config_path: Path, *, force: bool, work_dir: Path, log_dir: Path) -> None:
    output = Path(target["hf_checkpoint"])
    if (output / "config.json").exists() and (output / "model.safetensors").exists() and not force:
        print(f"skip existing {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "olmo_core.nn.moe.v2.hf.convert_checkpoint",
        "--ckpt-path",
        target["checkpoint"],
        "--output-path",
        str(output),
        "--config-path",
        str(config_path),
        "--work-dir",
        str(work_dir),
    ]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{target['train_name']}-step{target['step']}.log"
    print("convert", target["train_name"], target["cx"], target["variant"], "log", log_path)
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)


def append_checkpoint_index(target: dict[str, Any], config_path: Path, status: str, error: str | None) -> None:
    EVAL_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    row = {
        "train_run_name": target["train_name"],
        "step": target["step"],
        "olmo_checkpoint_path": target["checkpoint"],
        "olmo_config_path": str(config_path),
        "hf_checkpoint_path": target["hf_checkpoint"],
        "model_family": target.get("model_family", f"olmoe3-{target.get('model_size', '275m')}-ladder"),
        "variant": target["variant"],
        "cx": target["cx"],
        "lr": target["lr"],
        "wandb_url": target.get("wandb_url"),
        "convert_status": status,
        "convert_error": error,
        "converted_at": datetime.now(timezone.utc).isoformat(),
    }
    with CHECKPOINT_INDEX.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", action="append", default=[], help="Substring filter for train_name")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/olmoe3-hf-convert-work"))
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp/olmoe3-hf-convert-logs"))
    args = parser.parse_args()

    targets = load_targets(args.manifest)
    if args.only:
        targets = [t for t in targets if any(s in t["train_name"] for s in args.only)]
    if args.limit is not None:
        targets = targets[: args.limit]

    print(f"targets {len(targets)}")
    for target in targets:
        output = Path(target["hf_checkpoint"])
        config_path = output / "olmo_config.json"
        if Path(target["checkpoint"], "config.json").exists():
            config_path = Path(target["checkpoint"], "config.json")
        elif not config_path.exists() or args.force:
            reconstruct_config(target, config_path)

        if args.dry_run:
            print(target["train_name"], "->", output, "config", config_path)
            continue

        try:
            run_convert(target, config_path, force=args.force, work_dir=args.work_dir, log_dir=args.log_dir)
            append_checkpoint_index(target, config_path, "converted", None)
        except Exception as exc:
            append_checkpoint_index(target, config_path, "failed", repr(exc))
            raise


if __name__ == "__main__":
    main()
