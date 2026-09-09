"""Bounded, resumable batch orchestration for the explicitly requested hero checkpoints.

Starts only after both pilots passed conversion, native-vLLM parity and real eval smoke.
At most two GPU tasks are live/queued at once. Failed tasks stop orchestration for inspection;
they are never retried in an uncontrolled loop. All experiment names and revisions are recorded.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import re
import shutil
import time
from pathlib import Path

import yaml

from hero_hf_cleanup import cleanup
from hero_hf_download import BUCKET, SCRATCH, TARGETS, prepare_scratch, write_json
from olmoe3_small_hero_plan import Run

WORKSPACE = "ai2/olmo3p5-training"
TEMPLATES = Path(__file__).parent / "hero_hf_job_templates"


def source_available(arm, step, remote):
    """Availability hint only; the staging task verifies every actual source before use."""
    root = SCRATCH / arm / f"step{step}"
    return (
        (root / "download-success.json").is_file()
        or (Run(arm == "emo").root / f"step{step}" / ".metadata.json").is_file()
        or f"{arm}/receipts/step{step}.verified.json" in remote
    )


def remote_receipts(api):
    from huggingface_hub import BucketFile

    return {
        x.path
        for arm in ("emo", "non-emo")
        for x in api.list_bucket_tree(BUCKET, prefix=f"{arm}/receipts/", recursive=True)
        if isinstance(x, BucketFile)
    }


def build_spec(stage, arm, step, core_ref, plugins_ref):
    """Instantiate an allowlisted template with immutable commits and scoped model paths."""
    if (
        stage not in ("convert", "eval")
        or arm not in ("emo", "non-emo")
        or step not in TARGETS.values()
    ):
        raise ValueError("Job is outside the explicit campaign allowlist")
    for ref in (core_ref, plugins_ref):
        if not re.fullmatch(r"[0-9a-f]{40}", ref):
            raise ValueError("Use a full immutable Git commit for shallow fetch, not a branch")
    contents = (TEMPLATES / f"{stage}.yaml").read_text()
    for old, new in (
        ("__ARM__", arm),
        ("__STEP__", str(step)),
        ("__CORE_REF__", core_ref),
        ("__PLUGINS_REF__", plugins_ref),
    ):
        contents = contents.replace(old, new)
    if re.search(r"__[A-Z_]+__", contents):
        raise RuntimeError("Unresolved job placeholder")
    spec = yaml.safe_load(contents)
    for task in spec["tasks"]:
        if task["context"]["minRuntime"] != 0 or task["context"]["priority"] != "urgent":
            raise RuntimeError("Conversion/evaluation tasks must be urgent and unallocated")
        if task.get("result", {}).get("path"):
            raise RuntimeError("This pipeline may not put checkpoints in Beaker results")
    return spec


def main():
    from beaker import Beaker, BeakerExperimentSpec
    from beaker.exceptions import BeakerNotFoundError
    from google.protobuf.json_format import MessageToDict
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser()
    parser.add_argument("--core-ref", required=True)
    parser.add_argument("--plugins-ref", required=True)
    parser.add_argument("--max-active", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare_scratch()
    for arm, step in (("emo", 6000), ("non-emo", 11900)):
        pilot = SCRATCH / arm / f"step{step}"
        for marker in (
            "conversion-success.json",
            "vllm-parity-success.json",
            "eval-smoke-success.json",
        ):
            record = json.loads((pilot / marker).read_text()) if (pilot / marker).is_file() else {}
            if record.get("passed") is not True or record.get("diagnostic_only"):
                raise RuntimeError(f"{arm} pilot has not passed {marker}; no batch jobs launched")
    api = HfApi()
    with (
        (SCRATCH / "batch-controller.lock").open("a") as lock,
        Beaker.from_env(check_for_upgrades=False) as beaker,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        workspace = beaker.workspace.get(WORKSPACE)
        path = SCRATCH / "batch-state.json"
        if path.exists():
            state = json.loads(path.read_text())
            if (state["core_ref"], state["plugins_ref"]) != (args.core_ref, args.plugins_ref):
                raise RuntimeError("Resumption must use the original qualified commits")
        else:
            remote = remote_receipts(api)
            pairs = [
                ("emo", 6000),
                ("non-emo", 11900),
                ("non-emo", 6000),
                ("emo", 11900),
                ("emo", 17900),
                ("non-emo", 17900),
                ("emo", 23750),
            ]
            include_four = source_available("non-emo", 23750, remote)
            if include_four:
                pairs.append(("non-emo", 23750))
            state = {
                "core_ref": args.core_ref,
                "plugins_ref": args.plugins_ref,
                "created_at": time.time(),
                "include_non_emo_400b": include_four,
                "checkpoints": [
                    {"arm": arm, "step": step, "jobs": {}, "status": "pending"}
                    for arm, step in pairs
                ],
            }
            write_json(path, state)
        deadline = time.monotonic() + 24 * 3600
        while time.monotonic() < deadline:
            active = 0
            remote = remote_receipts(api)
            for row in state["checkpoints"]:
                row["active"] = False
                for stage, record in row["jobs"].items():
                    workload = beaker.workload.get(record["id"])
                    job = beaker.workload.get_latest_job(workload)
                    status = MessageToDict(job.status).get("status") if job else "STATUS_QUEUED"
                    record["status"] = status
                    if status in ("STATUS_FAILED", "STATUS_CANCELED", "STATUS_CANCELLED"):
                        state["needs_attention"] = {
                            "arm": row["arm"],
                            "step": row["step"],
                            "stage": stage,
                            "id": record["id"],
                            "status": status,
                        }
                        write_json(path, state)
                        raise RuntimeError(
                            f"Batch stopped for failed job: {state['needs_attention']}"
                        )
                    if status != "STATUS_SUCCEEDED":
                        row["active"] = True
                        active += 1
            for row in state["checkpoints"]:
                arm, step = row["arm"], row["step"]
                root = SCRATCH / arm / f"step{step}"
                if (root / "cleanup-success.json").is_file():
                    row["status"] = "complete"
                    continue
                if row["active"]:
                    continue
                if (root / "eval-smoke-success.json").is_file():
                    cleanup(arm, step)
                    row["status"] = "complete"
                    write_json(path, state)
                    continue
                stage = "eval" if (root / "conversion-success.json").is_file() else "convert"
                if stage in row["jobs"]:
                    raise RuntimeError(
                        f"Successful {stage} job did not publish required marker: {row}"
                    )
                if active >= args.max_active:
                    row["status"] = f"pending_{stage}"
                    continue
                if stage == "convert" and not source_available(arm, step, remote):
                    row["status"] = "waiting_for_source"
                    continue
                if stage == "convert" and shutil.disk_usage(SCRATCH).free < 11_200_000_000_000:
                    row["status"] = "waiting_for_storage"
                    continue
                name = f"olmo35-small-hf-20260909-{arm}-pt-step{step}-{stage}-r1"
                try:
                    existing = beaker.workload.get(f"jacobm/{name}")
                except BeakerNotFoundError:
                    spec = build_spec(stage, arm, step, args.core_ref, args.plugins_ref)
                    # Record the intended deterministic name before submitting; a restart can
                    # recover a successful create whose response was lost without duplicating it.
                    row["launch_intent"] = {"name": name, "stage": stage, "time": time.time()}
                    write_json(path, state)
                    existing = beaker.experiment.create(
                        spec=BeakerExperimentSpec.from_json(spec), name=name, workspace=workspace
                    )
                row["jobs"][stage] = {
                    "id": existing.experiment.id,
                    "name": name,
                    "status": "submitted",
                }
                row["status"] = stage
                active += 1
                write_json(path, state)
                print("HERO_BATCH_LAUNCHED", json.dumps(row), flush=True)
            state["updated_at"] = time.time()
            state["free_bytes"] = shutil.disk_usage(SCRATCH).free
            write_json(path, state)
            print("HERO_BATCH_STATUS", json.dumps(state), flush=True)
            if all(row["status"] == "complete" for row in state["checkpoints"]):
                print("HERO_BATCH_COMPLETE", flush=True)
                return
            time.sleep(30)
        raise TimeoutError("Batch did not complete within 24 hours; state preserved")


if __name__ == "__main__":
    main()
