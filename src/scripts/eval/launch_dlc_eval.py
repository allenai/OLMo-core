"""
Launch the Molmo2 DLC-Bench eval on Beaker (one Beaker experiment per
model variant).

Each experiment runs the eval driver in
``src/scripts/eval/molmo2_dlc_eval.py`` under ``torchrun`` on a single
8-GPU node. The 2,730 examples are data-parallel-split across the 8 GPUs,
rank 0 gathers predictions, drives the OpenAI judge, and writes a JSON to
the persistent Weka mount.

Example::

    python src/scripts/eval/launch_dlc_eval.py allenai/Molmo2-4B allenai/Molmo2-8B allenai/Molmo2-O-7B
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerEnvVar,
    BeakerLaunchConfig,
    BeakerWekaBucket,
    OLMoCoreBeakerImage,
)
from olmo_core.utils import generate_uuid

WEKA_MOUNT = "/weka/oe-training-default"
"""Where the Molmo eval data + we'll write outputs."""

OUTPUT_DIR_ON_WEKA = f"{WEKA_MOUNT}/jasonr/molmo2-dlc-eval"
"""Output JSONs and judge cache land here."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "models",
        nargs="+",
        help="HF model ids to evaluate (e.g. allenai/Molmo2-4B). One Beaker experiment per id.",
    )
    p.add_argument("--cluster", default="ai2/jupiter", help="Beaker cluster (default ai2/jupiter)")
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument(
        "--workspace",
        default="ai2/oe-encoder",
        help="Beaker workspace to launch into (default ai2/oe-encoder).",
    )
    p.add_argument("--budget", default="ai2/oe-other", help="Budget group (default ai2/oe-other).")
    p.add_argument("--priority", default="urgent")
    p.add_argument(
        "--openai-secret",
        default="JASONR_OPENAI_API",
        help="Beaker secret holding the OpenAI API key (mounted as OPENAI_API_KEY).",
    )
    p.add_argument(
        "--judge-model",
        default="gpt-4.1-2025-04-14",
        help="OpenAI judge model id passed to the eval driver.",
    )
    p.add_argument("--limit", type=int, default=0, help="0 = full split (2730).")
    p.add_argument("--max-new-tokens", type=int, default=448)
    p.add_argument(
        "--predictions-only",
        action="store_true",
        help="Pass through to the eval driver — skip the LLM judge.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the BeakerLaunchConfig and exit without launching.",
    )
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit launching with a dirty git tree.",
    )
    return p.parse_args()


def _model_slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(".", "-").lower()


def build_launch(args: argparse.Namespace, model_id: str) -> BeakerLaunchConfig:
    slug = _model_slug(model_id)
    run_id = generate_uuid()[:8]
    out_subdir = f"{OUTPUT_DIR_ON_WEKA}/{slug}-{run_id}"
    out_json = f"{out_subdir}/eval.json"
    judge_cache = f"{OUTPUT_DIR_ON_WEKA}/judge-cache-{args.judge_model.replace('/', '_')}"

    # Command run inside the Beaker container — note: torchrun is auto-prepended
    # by BeakerLaunchConfig when num_gpus > 1.
    cmd: List[str] = [
        "src/scripts/eval/molmo2_dlc_eval.py",
        "--model",
        model_id,
        "--out",
        out_json,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--judge-model",
        args.judge_model,
        "--judge-cache-dir",
        judge_cache,
        "--data-root",
        WEKA_MOUNT + "/mm-olmo",
    ]
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    if args.predictions_only:
        cmd.append("--predictions-only")

    return BeakerLaunchConfig(
        name=f"molmo2-dlc-{slug}-{run_id}",
        task_name="dlc-eval",
        cmd=cmd,
        budget=args.budget,
        workspace=args.workspace,
        clusters=[args.cluster],
        beaker_image=OLMoCoreBeakerImage.stable,
        num_nodes=1,
        num_gpus=args.num_gpus,
        priority=args.priority,
        preemptible=False,
        shared_filesystem=True,
        allow_dirty=args.allow_dirty,
        weka_buckets=[
            BeakerWekaBucket(bucket="oe-training-default", mount=WEKA_MOUNT),
        ],
        env_vars=[
            BeakerEnvVar(name="MOLMO_DATA_DIR", value=f"{WEKA_MOUNT}/mm-olmo"),
            BeakerEnvVar(name="HF_HOME", value=f"{WEKA_MOUNT}/jasonr/hf-home"),
            BeakerEnvVar(name="HF_HUB_OFFLINE", value="1"),
        ],
        env_secrets=[
            BeakerEnvSecret(
                name="OPENAI_API_KEY", secret=args.openai_secret, required=not args.predictions_only
            ),
        ],
        description=(
            f"Molmo2 DLC-Bench cap F1 eval: model={model_id}, "
            f"judge={args.judge_model}, n={'all' if args.limit == 0 else args.limit}. "
            f"Output → {out_json}."
        ),
    )


def main() -> int:
    args = parse_args()
    for model_id in args.models:
        cfg = build_launch(args, model_id)
        if args.dry_run:
            print(f"\n=== {model_id} ===")
            print(cfg)
            continue
        print(f"\n[launch] {model_id} → {cfg.name}", flush=True)
        workload = cfg.launch(follow=False)
        # The launch() return shape carries an id + URL we can surface.
        try:
            wid = workload.id  # type: ignore[attr-defined]
            print(f"  workload id: {wid}")
        except Exception:
            print(f"  workload: {workload}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
