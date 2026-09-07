"""Durable, single-owner controller: fresh bucket -> preflight -> smoke -> hero pair.

Never relaunch failed training or automatically extend a hero beyond its initial stop.
Reuse the qualified controller's fsync'd submission intents and exact-name reconciliation.
"""

import copy
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

from olmoe3_lr_sweep_plan import checkpoint_complete
from olmoe3_lr_sweep_watch import Controller, atomic_json, log, replace_env
from olmoe3_lr_sweep_watch import validation_spec as base_validation_spec
from olmoe3_small_hero_plan import (
    AUTOMATION,
    BRANCH,
    BUCKET,
    CAMPAIGN,
    COMPLETE,
    CONTROL,
    DATA_ROOT,
    DOLMA_MOUNT,
    MIX_SHA256,
    MOUNT,
    START_BYTES,
    STATE,
    UPLOADER,
    WORKSPACE,
    runs,
    validate_plan,
)


def with_mounts(task):
    """Mount dedicated volumes only; no payload ever enters Beaker results."""
    task["datasets"] = [d for d in task["datasets"] if d["mountPath"] == "/gantry"] + [
        {"mountPath": str(MOUNT), "source": {"weka": "olmo-3p5-checkpoints"}},
        {"mountPath": str(DOLMA_MOUNT), "source": {"weka": "dolma-3p5"}},
    ]
    task["result"] = {"path": "/noop-results"}


def training_spec(template, run, commit, smoke=False):
    """Retain the measured image/topology/host allowlist, changing orchestration only."""
    spec = copy.deepcopy(template)
    spec["tasks"] = [spec["tasks"][0]]
    t = spec["tasks"][0]
    t.update(name="train", replicas=8, leaderSelection=True, timeout="720h")
    t["arguments"] = [
        "python",
        "src/examples/olmo_ddp/olmoe3_small_hero_node.py",
        run.run_id,
        "ai2/holmes",
    ]
    with_mounts(t)
    t["context"].update(priority="urgent", minRuntime="1h", autoResume=True)
    # Infrastructure preemptions may resume this experiment, but no controller copy/retry loop.
    spec["retry"] = {"allowedTaskRetries": 0}
    spec["description"] = json.dumps({"campaign": CAMPAIGN, **run.as_dict(), "smoke": smoke})
    replace_env(
        t,
        {
            "GIT_REF": commit,
            "GIT_BRANCH": BRANCH,
            "OLMO35_HERO_SMOKE": "1" if smoke else "0",
            "OLMOE3_INTEGRATION_BASELINE": "optimized100b",
            "OLMOE3_INTEGRATION_COMMUNICATION": "none",
            "OLMOE3_BEAKER_WORKSPACE": WORKSPACE,
            "RESULTS_DIR": "/noop-results",
            "GANTRY_CHECK_FOR_UPGRADES": "0",
        },
    )
    assert t["replicas"] * t["resources"]["gpuCount"] == 64
    return spec


def validation_spec(template, commit):
    """Run full configuration checks in the actual production image without GPUs."""
    spec = base_validation_spec(template, commit)
    t = spec["tasks"][0]
    t["arguments"] = ["python", "src/examples/olmo_ddp/olmoe3_small_hero.py", "--validate-only"]
    with_mounts(t)
    replace_env(t, {"GIT_BRANCH": BRANCH, "GANTRY_CHECK_FOR_UPGRADES": "0"})
    return spec


def preflight(beaker):
    """Validate the completed mirror, live daemon, and newly created private hero bucket."""
    from olmo_checkpoint_uploader.backend import HuggingFaceBucketBackend
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore
    from olmoe3_lr_sweep_watch import status

    validate_plan()
    assert MOUNT.is_mount() and DOLMA_MOUNT.is_mount(), "Refuse overlay storage"
    fs = os.statvfs(MOUNT)
    free = fs.f_bavail * fs.f_frsize
    assert free >= START_BYTES, f"Only {free} bytes free before hero launch"
    assert status(beaker.workload.get(UPLOADER)) == "STATUS_RUNNING"
    completion = json.loads(COMPLETE.read_text())
    manifest = Path("src/olmo_core/data/mixes/Dolma3p5-14t.txt")
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == MIX_SHA256
    assert completion["mix_manifest_sha256"] == MIX_SHA256
    assert completion["data_root"] == str(DATA_ROOT)
    assert all(completion["verification"].values())
    inventory = json.loads((COMPLETE.parent / "source-inventory.json").read_text())
    assert inventory["inventory_digest_sha256"] == completion["inventory_digest_sha256"]
    assert len(inventory["objects"]) == completion["object_count"]
    for item in inventory["objects"]:
        path = DATA_ROOT / item["Key"]
        assert path.resolve().is_relative_to(DATA_ROOT.resolve()), path
        assert path.is_file() and path.stat().st_size == item["Size"], path
        with path.open("rb") as handle:
            assert handle.read(8), path
    backend = HuggingFaceBucketBackend()
    receipt = AUTOMATION / "fresh-bucket.json"
    intent = AUTOMATION / "fresh-bucket-intent.json"
    if receipt.exists():
        assert json.loads(receipt.read_text())["bucket"] == BUCKET
        backend.assert_private(BUCKET)
    else:
        if intent.exists():
            raise RuntimeError(
                "Ambiguous bucket creation: reconcile before adoption; do not reuse blindly"
            )
        atomic_json(intent, {"bucket": BUCKET, "source_commit": os.environ["GIT_REF"]})
        backend.create_private_empty(BUCKET, "allenai")
        atomic_json(
            receipt,
            {
                "bucket": BUCKET,
                "created_private_empty": True,
                "source_commit": os.environ["GIT_REF"],
            },
        )
        log("FRESH_PRIVATE_HERO_BUCKET_CREATED", bucket=BUCKET)
    store = StateStore(CONTROL, STATE)
    for r in runs() + runs(True):
        backend.assert_private(r.bucket)
        r.root.mkdir(parents=True, exist_ok=True)
        created = store.register(
            Registration(
                run_id=r.run_id,
                lineage_id=r.run_id,
                checkpoint_root=str(r.root),
                bucket_id=r.bucket,
                remote_prefix=r.prefix,
                deletion_mode="apply",
                min_local_checkpoints=3 if r.smoke else 2,
                delete_grace_seconds=3600,
            )
        )
        log(
            "HERO_REGISTRATION_READY",
            run=r.run_id,
            bucket=r.bucket,
            prefix=r.prefix,
            created=created,
        )
    atomic_json(
        AUTOMATION / "preflight.json",
        {
            "free_bytes": free,
            "dolma_completion": completion,
            "uploader": UPLOADER,
            "bucket": BUCKET,
            "plan": validate_plan(),
        },
    )
    log("HERO_PREFLIGHT_PASSED", free_bytes=free, dolma_files=completion["object_count"])


def verify_smoke():
    """Require exact saved/restored samples, identical initialization and first data batches."""
    left, right = runs(True)
    for r in (left, right):
        assert checkpoint_complete(r.root / "step4")
        complete = json.loads((r.root / "audit/complete-step4.json").read_text())
        assert complete["step"] == 4 and not complete["storage_paused"]
        for rank in range(64):
            restore = json.loads((r.root / "audit" / f"restore-step2-rank{rank}.json").read_text())
            assert restore["sampled_state_exact"] and restore["step"] == 2
    for filename in ["initial-weights-sha256.json"] + [
        f"input-step{s}-rank{rank}.sha256" for s in (1, 3) for rank in range(64)
    ]:
        assert (left.root / "audit" / filename).read_bytes() == (
            right.root / "audit" / filename
        ).read_bytes(), filename
    atomic_json(
        AUTOMATION / "smoke-passed.json",
        {
            "source_commit": os.environ["GIT_REF"],
            "sampled_state_exact": True,
            "init_and_data_match": True,
        },
    )
    log("HERO_SAVE_RESTORE_AND_PAIR_GATE_PASSED")


def main():
    """Submit both heroes only after the real-image and 64-GPU restore gates pass."""
    from beaker import Beaker

    AUTOMATION.mkdir(parents=True, exist_ok=True)
    with (AUTOMATION / "LOCK").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with Beaker.from_env(check_for_upgrades=False) as beaker:
            preflight(beaker)
            control = Controller(beaker, os.environ["GIT_REF"], automation=AUTOMATION)
            gates = [
                (
                    f"{CAMPAIGN}-config-validation",
                    validation_spec(control.template, control.commit),
                ),
                (
                    f"{CAMPAIGN}-save-restore-smoke",
                    training_spec(control.template, runs(True)[0], control.commit, smoke=True),
                ),
            ]
            for name, spec in gates:
                while True:
                    w = control.ensure(name, spec)
                    state = control.report(w)
                    if state == "STATUS_SUCCEEDED":
                        break
                    if state in ("STATUS_FAILED", "STATUS_CANCELED", "STATUS_STOPPED", "AMBIGUOUS"):
                        raise RuntimeError(f"Hero gate blocked: {name}: {state}")
                    time.sleep(30)
            verify_smoke()
            # Re-check current space and daemon after a potentially long allocation wait.
            preflight(beaker)
            submitted = []
            for r in runs():
                w = control.ensure(
                    f"{r.run_id}-train", training_spec(control.template, r, control.commit)
                )
                assert w is not None, "Ambiguous hero submission; stop and reconcile"
                submitted.append(
                    {"run": r.run_id, "experiment_id": w.experiment.id, "status": control.report(w)}
                )
            atomic_json(AUTOMATION / "heroes-submitted.json", submitted)
            log("BOTH_HERO_RUNS_SUBMITTED", runs=submitted)


if __name__ == "__main__":
    main()
