"""One approved 4Mi LC ablation with in-allocation resume gate and base/RULER evals."""

import copy
import fcntl
import json
import os
import time

from olmoe3_hero_lc_plan import (
    AUTOMATION,
    BRANCH,
    CONTROL,
    END,
    METADATA_CACHE,
    MOUNT,
    STATE,
    WORKSPACE,
    runs,
)
from olmoe3_lr_sweep_watch import Controller, atomic_json, log, replace_env, status


def main():
    """Submit only the EMO-ancestry LC arm and its evals; never launch SFT or sweeps."""
    from beaker import Beaker
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore
    from olmoe3_hero_decay_plan import validate_checkpoint

    commit = os.environ["GIT_REF"]
    assert MOUNT.is_mount()
    AUTOMATION.mkdir(parents=True, exist_ok=True)
    with (AUTOMATION / "LOCK").open("a") as lock, Beaker.from_env(
        check_for_upgrades=False
    ) as b:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (run,) = runs()
        validate_checkpoint(run.source, 5961, 16_777_216)
        proof = json.loads(run.source.with_name("step5961-copy.json").read_text())
        assert proof["all_file_hashes_verified"] and proof["destination"] == str(
            run.source
        )
        free = os.statvfs(MOUNT)
        assert free.f_bavail * free.f_frsize >= 10_000_000_000_000
        StateStore(CONTROL, STATE).register(
            Registration(
                run_id=run.run_id,
                lineage_id=run.run_id,
                checkpoint_root=str(run.root),
                bucket_id=run.bucket,
                remote_prefix=run.prefix,
                deletion_mode="apply",
                min_local_checkpoints=2,
                delete_grace_seconds=3600,
            )
        )
        run.root.mkdir(parents=True, exist_ok=True)
        c = object.__new__(Controller)
        c.beaker, c.workspace, c.commit = b, b.workspace.get(WORKSPACE), commit
        c.automation, c.last_status = AUTOMATION, {}
        spec = b.experiment.get_spec(
            b.workload.get("01M2RF2SK6K8KS2YJKT9RD6NY8")
        ).to_json()
        task = copy.deepcopy(spec["tasks"][0])
        refs = {v["name"]: v.get("value") for v in task["envVars"]}
        assert refs["GIT_REF"] == "d031ab975c0dd11e986b3f017e0aac2e3608b521"
        assert task["resources"]["gpuCount"] == 8
        task.update(name="lc-full", replicas=8, leaderSelection=True, timeout="48h")
        task["context"].update(priority="urgent", minRuntime="1h", autoResume=True)
        task["arguments"] = [
            "python",
            "src/examples/olmo_ddp/olmoe3_hero_lc_node.py",
            run.run_id,
        ]
        name = run.run_id + "-train"
        saved = AUTOMATION / "specs" / (name + ".json")
        if saved.exists():
            # Adopt exactly the persisted submission across inventory changes.
            spec = json.loads(saved.read_text())
        else:
            registered = {
                n.hostname for n in b.node.list(cluster=b.cluster.get("ai2/holmes"))
            }
            task["constraints"]["hostname"] = [
                h
                for h in task["constraints"]["hostname"]
                if h in registered
                and h
                not in {
                    "holmes-cs-aus-520.reviz.ai2.in",
                    "holmes-cs-aus-527.reviz.ai2.in",
                }
            ]
            assert len(task["constraints"]["hostname"]) >= 8
            replace_env(
                task,
                dict(
                    GIT_REF=commit,
                    GIT_BRANCH=BRANCH,
                    GANTRY_TASK_NAME="lc-full",
                    OLMO35_LC_PHASE="full",
                    OLMO35_HERO_EXPECTED_START="0",
                    OLMO35_HERO_STOP=str(END),
                    OLMO35_LC_LOAD=None,
                    WANDB_RUN_ID=None,
                    WANDB_RESUME=None,
                    CACHED_PATH_CACHE_ROOT=str(METADATA_CACHE),
                ),
            )
            spec["tasks"] = [task]
            spec["description"] = json.dumps(run.as_dict())
            spec["retry"] = {"allowedTaskRetries": 0}
        job = c.ensure(name, spec)
        assert job is not None, "Ambiguous submission requires review"
        atomic_json(
            AUTOMATION / "launch.json",
            dict(experiment=job.experiment.id, **run.as_dict()),
        )
        import olmoe3_hero_4t_evals as evaluations

        evaluations.AUTOMATION = AUTOMATION / "eval-pipeline"
        evaluations.tick("lc", b, commit, validate_only=True)
        previous = None
        while True:
            current = c.report(b.workload.get(job.experiment.id))
            if current in {"STATUS_FAILED", "STATUS_CANCELED", "STATUS_STOPPED"}:
                raise RuntimeError(f"LC needs operator review: {current}")
            free = os.statvfs(MOUNT)
            if free.f_bavail * free.f_frsize >= 10_000_000_000_000:
                result = evaluations.tick("lc", b, commit)
                if result != previous:
                    log("LC4MI_STATUS", training=current, evals=result)
                    previous = result
                rows = result.get(run.run_id, {})
                if all(
                    rows.get(k, {}).get("status") == "STATUS_SUCCEEDED"
                    for k in ("convert", "gen_mc", "math", "code", "ruler")
                ):
                    log("LC4MI_COMPLETE")
                    return
            time.sleep(60)


if __name__ == "__main__":
    main()
