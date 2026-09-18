"""CPU-only durable controller: config gate -> both-arm smoke -> all six points.

Decay is internal to every training job. Never launch hero runs or separate decay
jobs. Failed/ambiguous submissions require review; no automatic retry copies.
"""

import copy
import fcntl
import json
import os
import time

from olmoe3_hybrid_sweep_plan import (
    AUTOMATION,
    BATCH,
    BRANCH,
    BUCKET,
    CAMPAIGN,
    CONTROL,
    DOLMA_MOUNT,
    MOUNT,
    STATE,
    TRAIN_TEMPLATE,
    UPLOADER,
    WORKSPACE,
    runs,
    smoke_runs,
)
from olmoe3_lr_sweep_plan import checkpoint_complete
from olmoe3_lr_sweep_watch import Controller, atomic_json, log, replace_env, status
from olmoe3_small_hero_plan import COMPLETE, EXCLUDED_HOSTNAMES

BAD_HOSTS = EXCLUDED_HOSTNAMES | {"holmes-cs-aus-527.reviz.ai2.in"}
FAILED = {"STATUS_FAILED", "STATUS_CANCELED", "STATUS_STOPPED"}


def training_spec(template, run, commit):
    """Clone qualified hero runtime; isolate all run, checkpoint and W&B identity."""
    spec = copy.deepcopy(template)
    t = spec["tasks"][0]
    spec["tasks"] = [t]
    t.update(name="train", replicas=8, leaderSelection=True, timeout="48h")
    t["arguments"] = [
        "python",
        "src/examples/olmo_ddp/olmoe3_hybrid_sweep_node.py",
        run.run_id,
        "ai2/holmes",
    ]
    t["constraints"]["hostname"] = [h for h in t["constraints"]["hostname"] if h not in BAD_HOSTS]
    assert len(t["constraints"]["hostname"]) >= 8
    t["context"].update(priority="urgent", minRuntime="1h", autoResume=True)
    t["result"] = {"path": "/noop-results"}
    replace_env(
        t,
        {
            k: None
            for k in [
                "WANDB_RUN_ID",
                "WANDB_RESUME",
                "OLMO35_HERO_EXPECTED_START",
                "OLMO35_HERO_STOP",
                "OLMO35_HERO_ALLOW_CONTINUATION",
                "HYBRID_SMOKE_STOP",
                "HYBRID_SMOKE_EXPECTED_START",
            ]
        },
    )
    replace_env(
        t,
        dict(
            GIT_REF=commit,
            GIT_BRANCH=BRANCH,
            GANTRY_TASK_NAME="train",
            OLMOE3_BEAKER_WORKSPACE=WORKSPACE,
            RESULTS_DIR="/noop-results",
        ),
    )
    assert t["replicas"] * t["resources"]["gpuCount"] == 64
    assert {"/weka/dolma-3p5", "/weka/olmo-3p5-checkpoints"}.issubset(
        {d["mountPath"] for d in t["datasets"]}
    )
    spec["description"] = json.dumps({**run.as_dict(), "source_commit": commit})
    spec["retry"] = {"allowedTaskRetries": 0}
    return spec


def validation_spec(template, commit):
    """Same image and pinned packages, CPU-only Phobos with no requested resources."""
    spec = training_spec(template, runs()[0], commit)
    t = spec["tasks"][0]
    t.update(
        name="validate",
        replicas=1,
        leaderSelection=False,
        timeout="2h",
        constraints={"cluster": ["ai2/phobos"]},
        hostNetworking=False,
        propagateFailure=False,
        propagatePreemption=False,
    )
    t.pop("resources", None)
    t.pop("synchronizedStartTimeout", None)
    t["context"] = dict(priority="urgent", minRuntime="0s", autoResume=False)
    t["arguments"] = ["python", "src/examples/olmo_ddp/olmoe3_hybrid_sweep.py", "--validate-only"]
    setup = (
        'gh auth setup-git && uv pip install --python "$(command -v python)" --no-deps '
        ". 'flash-linear-attention==0.5.2' 'fla-core==0.5.2' 'nvidia-nccl-cu13==2.28.9' "
        "'kernel-fun @ git+https://github.com/allenai/kernel-fun-dev.git@"
        "7a6983baf2beb4ec4d7fe914ec9f6670438af99b#subdirectory=packages/kernel-fun' && "
        "python src/examples/olmo_ddp/olmoe3_small_hero_runtime.py"
    )
    replace_env(
        t,
        dict(
            GANTRY_TASK_NAME="validate",
            NUM_NODES="1",
            GANTRY_POST_SETUP_CMD=setup,
            OLMO_SYMM_VDEV2D_AUTO_BUILD="0",
        ),
    )
    spec["description"] = "CPU-only full-config validation for six3:1 LR runs and two smoke arms"
    return spec


def register(beaker):
    """Register only eight isolated namespaces; preserve all existing policies."""
    from olmo_checkpoint_uploader.backend import HuggingFaceBucketBackend
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore

    assert MOUNT.is_mount() and DOLMA_MOUNT.is_mount() and COMPLETE.is_file()
    fs = os.statvfs(MOUNT)
    assert fs.f_bavail * fs.f_frsize >= 12_000_000_000_000, "Need12TB free before launching"
    assert status(beaker.workload.get(UPLOADER)) == "STATUS_RUNNING"
    HuggingFaceBucketBackend().assert_private(BUCKET)
    store = StateStore(CONTROL, STATE)
    for run in runs() + smoke_runs():
        run.root.mkdir(parents=True, exist_ok=True)
        created = store.register(
            Registration(
                run_id=run.run_id,
                lineage_id=run.run_id,
                checkpoint_root=str(run.root),
                bucket_id=BUCKET,
                remote_prefix=run.prefix,
                deletion_mode="apply",
                min_local_checkpoints=4,
                delete_grace_seconds=3600,
            )
        )
        log("HYBRID_REGISTERED", run=run.run_id, created=created)
    atomic_json(AUTOMATION / "plan.json", [r.as_dict() for r in runs() + smoke_runs()])


def check_finished(run):
    """Never treat a graceful early stop as successful completion of this budget."""
    final = run.root / f"step{run.end}"
    assert checkpoint_complete(final), f"Incomplete final checkpoint: {final}"
    audit = json.loads((run.root / "audit" / f"completed-step{run.end}.json").read_text())
    assert audit["step"] == run.end and audit["tokens"] == run.end * BATCH
    assert not (run.root / "STORAGE_PAUSED.json").exists()
    for rank in range(64):
        item = json.loads((final / "resume_audit" / f"rank{rank}.json").read_text())
        assert (item["step"], item["tokens"], item["rank"], item["gpus"]) == (
            run.end,
            run.end * BATCH,
            rank,
            64,
        )
        assert (final / "train" / f"rank{rank}.pt").is_file()
        if run.smoke:
            proof = json.loads((run.root / "audit" / f"restore-step4-rank{rank}.json").read_text())
            assert proof["sampled_state_exact"]


def ensure_training(controller, template, run, commit):
    """Resolve existing allowed hosts for new jobs; preserve submitted specs on restart."""
    name = run.run_id + "-train"
    spec = training_spec(template, run, commit)
    intent = controller.automation / "submissions" / f"{name}.json"
    if intent.exists():
        # Host inventory may change while jobs run. Reconcile the exact persisted
        # submission, never turn a changed inventory into a second experiment.
        saved = json.loads((controller.automation / "specs" / f"{name}.json").read_text())
        hosts = saved["tasks"][0]["constraints"]["hostname"]
        assert len(hosts) >= 8 and set(hosts).issubset(spec["tasks"][0]["constraints"]["hostname"])
        spec["tasks"][0]["constraints"]["hostname"] = hosts
        assert spec == saved, f"Unexpected training-spec drift for {name}"
    else:
        cluster = controller.beaker.cluster.get("ai2/holmes")
        registered = {n.hostname for n in controller.beaker.node.list(cluster=cluster)}
        previous = spec["tasks"][0]["constraints"]["hostname"]
        hosts = [host for host in previous if host in registered]
        assert len(hosts) >= 8, "Fewer than eight qualified hosts remain registered"
        spec["tasks"][0]["constraints"]["hostname"] = hosts
        log("HYBRID_HOSTS_RECONCILED", name=name, removed=sorted(set(previous) - set(hosts)))
    return controller.ensure(name, spec)


def submit_points(controller, template, commit):
    """Submit both arms before waiting; adopt existing experiments without duplicates."""
    jobs = {}
    for emo in (True, False):
        arm_jobs = {}
        for run in (r for r in runs() if r.emo == emo):
            workload = ensure_training(controller, template, run, commit)
            assert workload is not None, "Ambiguous submission requires review"
            arm_jobs[run.run_id] = workload.experiment.id
        atomic_json(
            controller.automation / f"{'emo' if emo else 'non-emo'}-submitted.json", arm_jobs
        )
        jobs.update(arm_jobs)
    return jobs


def main():
    """Enforce smoke qualification with a lock and persistent submission intents."""
    from beaker import Beaker

    # A controller-only repair need not change the smoke-qualified training code.
    controller_commit = os.environ["GIT_REF"]
    commit = os.environ.get("HYBRID_TRAIN_COMMIT", controller_commit)
    gate_id = os.environ["HYBRID_CONFIG_GATE"]
    AUTOMATION.mkdir(parents=True, exist_ok=True)
    with (AUTOMATION / "LOCK").open("a") as lock, Beaker.from_env(check_for_upgrades=False) as b:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = object.__new__(Controller)
        c.beaker, c.workspace, c.commit = b, b.workspace.get(WORKSPACE), commit
        c.automation, c.last_status = AUTOMATION, {}
        while True:
            gate = b.workload.get(gate_id)
            state = c.report(gate)
            if state in FAILED:
                raise RuntimeError(f"Config gate failed: {gate_id}, {state}")
            if state == "STATUS_SUCCEEDED":
                break
            time.sleep(30)
        register(b)
        template = b.experiment.get_spec(b.workload.get(TRAIN_TEMPLATE)).to_json()
        atomic_json(AUTOMATION / "qualified-template.json", template)
        smoke = c.ensure(
            f"olmoe3-{CAMPAIGN}-smoke", training_spec(template, smoke_runs()[0], commit)
        )
        assert smoke is not None, "Ambiguous smoke submission requires review"
        while True:
            state = c.report(b.workload.get(smoke.experiment.id))
            if state in FAILED:
                raise RuntimeError(f"Smoke failed: {smoke.experiment.id}, {state}")
            if state == "STATUS_SUCCEEDED":
                for run in smoke_runs():
                    check_finished(run)
                atomic_json(
                    AUTOMATION / "smoke-passed.json",
                    dict(experiment=smoke.experiment.id, commit=commit),
                )
                break
            time.sleep(30)
        jobs = submit_points(c, template, commit)
        while True:
            states = {r.run_id: c.report(b.workload.get(jobs[r.run_id])) for r in runs()}
            atomic_json(
                AUTOMATION / "status.json",
                dict(
                    phase="all",
                    states=states,
                    jobs=jobs,
                    training_commit=commit,
                    controller_commit=controller_commit,
                ),
            )
            if any(s in FAILED for s in states.values()):
                raise RuntimeError(f"Sweep failure requires review: {states}")
            if all(s == "STATUS_SUCCEEDED" for s in states.values()):
                for run in runs():
                    check_finished(run)
                break
            time.sleep(30)
        atomic_json(AUTOMATION / "COMPLETE.json", dict(commit=commit, points=6))
        log("HYBRID_SWEEP_COMPLETE", points=6)


if __name__ == "__main__":
    main()
