"""Prepare a continuous 600B PT -> 100B MT decay; this module never submits jobs.

Start from the protected, non-decayed 14T native checkpoint. Keep EMO enabled,
the 8K context, 16Mi-token batch, and optimizer state. Reshard 128 -> 64 GPUs;
the initial restore uses the trainer's native RNG reseed on world-size change.
After that, preserve all 64 ranks' RNG states across retries and the data switch.
The LR decreases
linearly from 0.0011 to zero over BOTH data segments. At the boundary, only the
dataset, its seed/cursor, and its epoch change. There is no optimizer reset or
second warmup. One Beaker allocation runs two torchrun processes, separated by
an audited native checkpoint; they share the same absolute steps and LR schedule.

Preparation uses stdlib Python and reads the source checkpoint:
    python src/examples/olmo_ddp/olmoe3_combined_decay_plan.py --output PLAN.json

Do not submit the prepared Beaker specification until explicitly authorized.
Before submission, register plan['registration'] with the existing uploader,
check uploader health/free space, and verify the production-image startup.
The node entrypoint refuses to train without the matching registration.
No controller is armed by this planner, and no LC/SFT jobs are auto-launched.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path

import olmoe3_hero_fixed_decay_example as fixed

BATCH = fixed.BATCH
LR = fixed.LR
PT_TOKENS = 600_000_000_000
MT_TOKENS = 100_000_000_000
MT_SEED = 1387822106
MT_MIX_SHA256 = "1ed52c81c3f33fb864157f8e01ed0d0aff24548149c2ca4376182578181c25ba"
ROOT = Path("/weka/olmo-3p5-checkpoints/production-combined-decays")
PROTECTED = Path("/weka/olmo-3p5-checkpoints/protected/combined-decays")
SOURCE = Path("/weka/olmo-3p5-checkpoints/protected/olmo35-small-hero-20260907/emo/14T")
RUN_NAME = "olmo35-small-14t-pt600b-mt100b-continuous-20261014-emo"
BRANCH = "codex/hero14t-combined-decay-20261014"
WORKSPACE = "ai2/olmo3p5-training"


def digest(value):
    """Hash the complete immutable plan or other JSON value."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    """Atomically write a JSON receipt on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def make_plan(source=SOURCE, name=RUN_NAME):
    """Validate the existing native source and describe the unlaunched experiment."""
    plan = fixed.make_plan(Path(source), PT_TOKENS + MT_TOKENS, "emo", name)
    config = fixed.read_json(Path(plan["source"]) / "config.json")
    if plan["start_step"] != 834466 or plan["gpus"] != 128:
        raise ValueError("This prepared recipe requires the original 14T/128-GPU source")
    pt_steps = (PT_TOKENS + BATCH - 1) // BATCH
    mt_steps = (MT_TOKENS + BATCH - 1) // BATCH
    root = ROOT / name
    if root.exists():
        raise ValueError(f"Choose an unused output directory: {root}")
    plan.update(
        schema=2,
        source_gpus=128,
        gpus=64,
        initial_rng_policy="native-reseed-on-world-size-change",
        recipe="continuous-pt600b-mt100b-v1",
        save_folder=str(root),
        protected_folder=str(PROTECTED / name),
        pt_steps=pt_steps,
        mt_steps=mt_steps,
        switch_step=plan["start_step"] + pt_steps,
        decay_steps=pt_steps + mt_steps,
        end_step=plan["start_step"] + pt_steps + mt_steps,
        end_tokens=(plan["start_step"] + pt_steps + mt_steps) * BATCH,
        actual_decay_tokens=(pt_steps + mt_steps) * BATCH,
        requested_pt_tokens=PT_TOKENS,
        requested_mt_tokens=MT_TOKENS,
        pt_data_seed=config["data_loader"]["seed"],
        mt_data_seed=MT_SEED,
        mt_mixture_sha256=MT_MIX_SHA256,
        source_dataset_sha256=digest(config["dataset"]),
        workspace=WORKSPACE,
        cluster="ai2/holmes",
        optimizer_reset_at_switch=False,
        warmup_at_switch=0,
        emo_during_pt=True,
        emo_during_mt=True,
    )
    plan["registration"].update(checkpoint_root=str(root), remote_prefix=f"combined-decays/{name}")
    return plan


def validate_plan(plan):
    """Reject edits to schedule, source, output namespace, or state-transfer policy."""
    root = Path(plan["save_folder"])
    source = Path(plan["source"])
    info = fixed.checkpoint_info(source)
    if any(plan["source_gpus" if k == "gpus" else k] != v for k, v in info.items()):
        raise ValueError("Source metadata differs from the prepared plan")
    cfg_bytes = (source / "config.json").read_bytes()
    cfg = json.loads(cfg_bytes)
    checks = [
        plan["schema"] == 2,
        plan["recipe"] == "continuous-pt600b-mt100b-v1",
        plan["base_commit"] == fixed.BASE_COMMIT,
        re.fullmatch(r"[a-z0-9][a-z0-9-]{0,99}", plan["run_name"]) is not None,
        plan["start_step"] == 834466,
        plan["gpus"] == 64 and plan["source_gpus"] == 128,
        plan["initial_rng_policy"] == "native-reseed-on-world-size-change",
        plan["arm"] == "emo",
        root == ROOT / plan["run_name"],
        Path(plan["protected_folder"]) == PROTECTED / plan["run_name"],
        plan["source_config_sha256"] == hashlib.sha256(cfg_bytes).hexdigest(),
        plan["source_dataset_sha256"] == digest(cfg["dataset"]),
        cfg["train_module"]["scheduler"]["_CLASS_"].endswith(".ConstantWithWarmup"),
        cfg["train_module"]["optim"]["lr"] == LR,
        cfg["data_loader"]["global_batch_size"] == BATCH,
        plan["pt_steps"] == (PT_TOKENS + BATCH - 1) // BATCH,
        plan["mt_steps"] == (MT_TOKENS + BATCH - 1) // BATCH,
        plan["decay_steps"] == plan["pt_steps"] + plan["mt_steps"],
        plan["switch_step"] == plan["start_step"] + plan["pt_steps"],
        plan["end_step"] == plan["switch_step"] + plan["mt_steps"],
        plan["end_tokens"] == plan["end_step"] * BATCH,
        plan["actual_decay_tokens"] == plan["decay_steps"] * BATCH,
        plan["global_batch_tokens"] == BATCH,
        plan["requested_pt_tokens"] == PT_TOKENS,
        plan["requested_mt_tokens"] == MT_TOKENS,
        plan["requested_decay_tokens"] == PT_TOKENS + MT_TOKENS,
        plan["pt_data_seed"] == cfg["data_loader"]["seed"],
        plan["mt_data_seed"] == MT_SEED,
        plan["mt_mixture_sha256"] == MT_MIX_SHA256,
        plan["optimizer_reset_at_switch"] is False,
        plan["warmup_at_switch"] == 0,
        plan["emo_during_pt"] is True and plan["emo_during_mt"] is True,
        plan["workspace"] == WORKSPACE and plan["cluster"] == "ai2/holmes",
        plan["registration"]
        == dict(
            run_id=plan["run_name"],
            lineage_id=plan["run_name"],
            checkpoint_root=str(root),
            bucket_id="allenai/olmo-3p5-small",
            remote_prefix=f"combined-decays/{plan['run_name']}",
            deletion_mode="apply",
            min_local_checkpoints=2,
            delete_grace_seconds=3600,
        ),
    ]
    if not all(checks):
        raise ValueError("Plan validation failed; regenerate rather than editing derived fields")


def learning_rate(plan, step):
    """The single absolute-step linear schedule spans both data segments."""
    if not plan["start_step"] <= step <= plan["end_step"]:
        raise ValueError("Step outside the declared continuous decay")
    return LR * (plan["end_step"] - step) / plan["decay_steps"]


def checkpoint_phase(plan, step):
    """The boundary checkpoint is saved with PT data; the next update uses MT."""
    return "pt" if step <= plan["switch_step"] else "mt"


def select_checkpoint(plan):
    """Resume the latest complete checkpoint; never rewind past a corrupt completed save."""
    choices = [(plan["start_step"], plan["source"])]
    for path in Path(plan["save_folder"]).glob("step*"):
        if not re.fullmatch(r"step[0-9]+", path.name) or not (path / ".metadata.json").is_file():
            continue
        info = fixed.checkpoint_info(path)
        step = info["start_step"]
        marker = fixed.read_json(path / "combined-decay.json")
        cfg = fixed.read_json(path / "config.json")
        scheduler = cfg["train_module"]["scheduler"]
        if (
            path.is_symlink()
            or step != int(path.name[4:])
            or info["gpus"] != plan["gpus"]
            or not plan["start_step"] < step <= plan["end_step"]
            or marker != dict(plan_sha256=digest(plan), phase=checkpoint_phase(plan, step))
            or cfg["run_name"] != plan["run_name"]
            or not scheduler["_CLASS_"].endswith(".WSD")
            or scheduler.get("decay") != plan["decay_steps"]
        ):
            raise ValueError(f"Checkpoint does not match this continuous decay: {path}")
        choices.append((step, str(path)))
    return max(choices)


def transition_state(plan, saved, fresh_data):
    """Replace only PT loader/epoch state at the exact switch; retain counters and RNG."""
    if (
        saved["global_step"] != plan["switch_step"]
        or saved["global_train_tokens_seen"] != plan["switch_step"] * BATCH
        or saved["world_size"] != plan["gpus"]
        or saved["data_loader"]["seed"] != plan["pt_data_seed"]
        or fresh_data["seed"] != MT_SEED
        or fresh_data["tokens_processed"] != 0
        or fresh_data["batches_processed"] != 0
        or fresh_data["dataset_fingerprint"] == saved["data_loader"]["dataset_fingerprint"]
    ):
        raise ValueError("Invalid PT -> MT data transition")
    result = copy.deepcopy(saved)
    result["data_loader"] = copy.deepcopy(fresh_data)
    result["data_loader"]["epoch"] = 1
    result["epoch"] = 1
    return result


def protect_checkpoint(plan, step):
    """Pin the boundary and final immutable native checkpoints with hardlinks."""
    if step not in (plan["switch_step"], plan["end_step"]):
        raise ValueError("Only the declared endpoints may be protected here")
    source = Path(plan["save_folder"]) / f"step{step}"
    target = Path(plan["protected_folder"]) / f"step{step}" / "olmo-core"
    receipt = target.parent / "protected.json"
    if receipt.exists():
        row = fixed.read_json(receipt)
        if row["plan_sha256"] != digest(plan) or not (target / ".metadata.json").is_file():
            raise ValueError("Protected checkpoint identity mismatch")
        return
    fixed.checkpoint_info(source)
    for item in sorted(source.rglob("*")):
        dest = target / item.relative_to(source)
        if item.is_symlink():
            raise ValueError(f"Symlink in native checkpoint: {item}")
        if item.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                os.link(item, dest)
            if not os.path.samefile(item, dest):
                raise ValueError(f"Protected path conflicts: {dest}")
    write_json(
        receipt,
        dict(
            passed=True,
            source=str(source),
            destination=str(target),
            plan_sha256=digest(plan),
            step=step,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = make_plan(args.source, args.run_name)
    validate_plan(result)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
