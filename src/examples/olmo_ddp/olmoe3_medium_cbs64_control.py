"""Durable, fail-closed ordering: config ->64G restore/speed ->128G speed ->64G CBS.

No automatic retries, no source mutation, and no training allocation held while
waiting for an earlier workload. Separate exact-name submissions are reconciled
against saved intent, including after a controller preemption.
"""

import copy
import fcntl
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

from olmoe3_medium_cbs64_plan import (
    AUTOMATION,
    BRANCH_NAME,
    BUCKET,
    CAMPAIGN,
    CONTROL,
    CPU_CLUSTER,
    FORK_TOKENS,
    MOUNT,
    PARENT,
    PARENT_ROOT,
    PARENT_RUN,
    PHASES,
    ROOT,
    SUBMISSION_SUFFIX,
    UPLOADER,
    VARIANT,
    WAVES,
    WORKSPACE,
    phase_environment,
    validate,
)
from olmoe3_medium_cbs_control import atomic_json, log, replace_env, status

PARENT_EXPERIMENT = "01M1YJ10RHVFBH9BHFGKZVGJ8K"
VALIDATION_TEMPLATE = "01M1YDHB4K7930RQFR7W781HFC"
EXCLUDED = {f"holmes-cs-aus-{n}" for n in (485, 503, 516)}


def training_spec(template, wave, commit):
    phase = PHASES[WAVES[wave][0]]
    spec = copy.deepcopy(template)
    task = spec["tasks"][0]
    spec["tasks"] = [task]
    task.update(name="train", replicas=phase.gpus // 8, leaderSelection=True)
    assert task["resources"]["gpuCount"] == 8
    hosts = task["constraints"]["hostname"]
    task["constraints"]["hostname"] = [
        host for host in hosts if not any(host == n or host.startswith(n + ".") for n in EXCLUDED)
    ]
    assert len(task["constraints"]["hostname"]) >= phase.gpus // 8
    # A parent profiling template can contain ranks beyond this wave's world size.
    # These are timing/CBS jobs, not Nsight captures. Do not inherit any Nsight knobs.
    task["envVars"] = [v for v in task["envVars"] if not v["name"].startswith("OLMOE3_NSYS_")]
    task["arguments"] = [
        "python",
        "src/examples/olmo_ddp/olmoe3_medium_cbs64_node.py",
        wave,
        "ai2/holmes",
    ]
    replace_env(
        task,
        {
            "GIT_REF": commit,
            "GIT_BRANCH": BRANCH_NAME,
            "NUM_NODES": phase.gpus // 8,
            "RESULTS_DIR": "/noop-results",
            "OLMOE3_MEDIUM_GPUS": phase.gpus,
            "OLMOE3_MEDIUM_MB": 2,
            "OLMOE3_MEDIUM_BATCH": phase.batch,
            "OLMOE3_MEDIUM_DIAGNOSTIC": 0,
            "OLMOE3_DEEP_PROFILE_TEST": VARIANT,
            "OLMOE3_DEEP_PROFILE_PASS": "timing",
            "OLMOE3_MEDIUM_CAPTURE": 0,
            "OLMOE3_MEDIUM_CBS_RUN": None,
            "OLMOE3_MEDIUM_CBS_SMOKE": None,
            "OLMOE3_MEDIUM_CBS_STOP": None,
            "OLMOE3_MEDIUM_CBS_EXPECTED_START": None,
            "OLMOE3_DEEP_PROFILE_PLAN": None,
            "OLMOE3_MEDIUM_FOLLOWUP": None,
            "OLMOE3_BEAKER_WORKSPACE": WORKSPACE,
            "OLMOE3_WANDB_PROJECT": "olmoe3-production-cbs",
            "NCCL_PROTO": None,
        },
    )
    task["context"] = {"priority": "urgent", "minRuntime": "1h", "autoResume": False}
    task["timeout"] = "36h" if phase.cbs else "12h"
    task["result"] = {"path": "/noop-results"}
    spec["retry"] = {"allowedTaskRetries": 0}
    spec["description"] = json.dumps(
        {"campaign": CAMPAIGN, "wave": wave, "phases": WAVES[wave], "source_commit": commit}
    )
    return spec


def validation_environments(training, commit):
    """Derive config-validation settings from the actual GPU specs, not the CPU template."""
    environments = {}
    for wave, keys in WAVES.items():
        task = training_spec(training, wave, commit)["tasks"][0]
        # Secret references never enter the validation program or its provenance.
        env = {
            v["name"]: v["value"]
            for v in task["envVars"]
            if "value" in v and v["name"].startswith(("OLMO", "NCCL"))
        }
        for key in keys:
            environments[key] = phase_environment(env, PHASES[key])
    return environments


def config_spec(template, commit, training):
    spec = copy.deepcopy(template)
    assert len(spec["tasks"]) == 1
    task = spec["tasks"][0]
    assert not task.get("resources", {}).get("gpuCount"), "Config gate must not request GPUs"
    # Includes inherited sharedMemory, CPU and RAM: Phobos has no GPU-backed slots.
    task.pop("resources", None)
    task["constraints"] = {"cluster": [CPU_CLUSTER]}
    # The upstream repository was renamed; preserve the exact qualified source hash.
    for item in task["envVars"]:
        if item["name"] == "GANTRY_POST_SETUP_CMD":
            item["value"] = item["value"].replace(
                "github.com/allenai/kernel-fun.git@", "github.com/allenai/kernel-fun-dev.git@"
            )
    code = (
        "import os,subprocess,sys\n"
        "sys.path.insert(0,'src/examples/olmo_ddp')\n"
        "from olmoe3_medium_cbs64_plan import PHASES,validate\nvalidate()\n"
        f"environments={validation_environments(training, commit)!r}\n"
        "base={k:v for k,v in os.environ.items() if not k.startswith(('OLMO','NCCL'))}\n"
        "for phase,settings in environments.items():\n"
        " print('VALIDATING_TRAINING_ENV',phase,settings,flush=True)\n"
        " subprocess.run([sys.executable,'src/examples/olmo_ddp/olmoe3_medium_cbs64.py',"
        "'--validate-only'],env=dict(base,**settings),check=True)\n"
        "print('ALL_MEDIUM_CBS64_CONFIGS_VALIDATED',flush=True)\n"
    )
    task["arguments"] = ["python", "-u", "-c", code]
    replace_env(
        task,
        {
            "GIT_REF": commit,
            "GIT_BRANCH": BRANCH_NAME,
            "NUM_NODES": 1,
            "OLMOE3_MEDIUM_DIAGNOSTIC": 0,
            "RESULTS_DIR": "/noop-results",
        },
    )
    task["context"] = {"priority": "urgent", "minRuntime": "0s", "autoResume": False}
    task["timeout"] = "1h"
    task["result"] = {"path": "/noop-results"}
    spec["retry"] = {"allowedTaskRetries": 0}
    spec["description"] = f"{CAMPAIGN}: CPU-only exact-source config gate {commit}"
    return spec


def ensure(beaker, name, spec):
    from beaker import BeakerExperimentSpec

    path = AUTOMATION / "submissions" / f"{name}.json"
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    entry = json.loads(path.read_text()) if path.exists() else None
    workspace = beaker.workspace.get(WORKSPACE)
    if entry:
        assert entry["spec_sha256"] == digest, f"Spec changed for {name}"
        if entry.get("experiment_id"):
            return entry["experiment_id"]
    matches = [
        w
        for w in beaker.workload.list(workspace=workspace, name_or_description=name, limit=100)
        if w.HasField("experiment") and w.experiment.name == name
    ]
    if matches:
        assert len(matches) == 1 and entry, f"Untracked/ambiguous exact-name workload {name}"
        workload = matches[0]
    elif entry:
        raise RuntimeError(f"Ambiguous create; reconcile manually rather than duplicate: {name}")
    else:
        parsed = BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        atomic_json(AUTOMATION / "specs" / f"{name}.json", spec)
        entry = {"name": name, "phase": "submitting", "spec_sha256": digest}
        atomic_json(path, entry)
        workload = beaker.experiment.create(spec=parsed, name=name, workspace=workspace)
    eid = workload.experiment.id
    atomic_json(path, {**entry, "phase": "submitted", "experiment_id": eid})
    log("submitted_or_reconciled", name=name, experiment_id=eid)
    return eid


def wait_success(beaker, experiment):
    deadline = time.monotonic() + 72 * 3600
    previous = None
    while time.monotonic() < deadline:
        state = status(beaker.workload.get(experiment))
        if state != previous:
            log("gate_status", experiment_id=experiment, state=state)
            previous = state
        if state == "STATUS_SUCCEEDED":
            return
        if state in ("STATUS_FAILED", "STATUS_CANCELED", "STATUS_STOPPED"):
            raise RuntimeError(f"Gate failed; no downstream launches: {experiment} {state}")
        time.sleep(30)
    raise TimeoutError(f"Gate did not succeed within72h: {experiment}")


def source_preflight(beaker):
    from olmo_checkpoint_uploader.backend import HuggingFaceBucketBackend

    validate()
    assert MOUNT.is_mount(), "Actual Weka mount missing"
    assert not ROOT.is_symlink() and not PARENT.is_symlink()
    fs = os.statvfs(MOUNT)
    free = fs.f_bavail * fs.f_frsize
    assert free > 5_000_000_000_000, f"Insufficient free capacity {free}"
    assert status(beaker.workload.get(UPLOADER)) == "STATUS_RUNNING"
    HuggingFaceBucketBackend().assert_private(BUCKET)
    registration = json.loads((CONTROL / "registrations" / f"{PARENT_RUN}.json").read_text())
    assert registration["checkpoint_root"] == str(PARENT_ROOT)
    steps = sorted(
        int(p.name[4:])
        for p in PARENT_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("step") and p.name[4:].isdigit()
    )
    protected = steps[-registration["min_local_checkpoints"] :]
    assert 4000 in protected, "Original fork is not locally protected"
    # The parent is finished and cannot emit new checkpoints that displace the fork.
    assert status(beaker.workload.get(PARENT_EXPERIMENT)) == "STATUS_SUCCEEDED"
    for item in (".metadata.json", "model_and_optim/.metadata", "train/rank0.pt"):
        assert (PARENT / item).is_file(), f"Missing parent state: {item}"
    assert not json.loads((PARENT / ".metadata.json").read_text()).get("ephemeral")
    marker = CONTROL / "inbox" / PARENT_RUN / "step-000000004000.ready.json"
    assert marker.is_file(), "Parent ready marker missing"
    for rank in range(128):
        record = json.loads((PARENT_ROOT / "audit" / f"state-step4000-rank{rank}.json").read_text())
        assert (record["gpus"], record["rank"], record["step"], record["tokens"]) == (
            128,
            rank,
            4000,
            FORK_TOKENS,
        )
        assert len(record["tensors"]) == 3115
    log(
        "source_preflight_passed",
        free_bytes=free,
        source=str(PARENT),
        ready_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
    )


def registration_matches(existing, requested):
    """Preserve creation time on restart, while comparing every identity/policy field."""
    return existing == replace(requested, created_at=existing.created_at)


def register_outputs(wave):
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore

    store = StateStore(CONTROL, MOUNT / "uploader/state")
    for key in WAVES[wave]:
        phase = PHASES[key]
        if not phase.save:
            continue
        assert not phase.root.is_symlink()
        phase.root.mkdir(parents=True, exist_ok=True)
        registration = Registration(
            run_id=phase.run_id,
            lineage_id=phase.run_id,
            checkpoint_root=str(phase.root),
            bucket_id=BUCKET,
            remote_prefix=f"runs/{phase.run_id}",
            deletion_mode="apply",
            min_local_checkpoints=2,
            delete_grace_seconds=3600,
        )
        path = store.registration_path(phase.run_id)
        if path.exists():
            existing = Registration.from_dict(json.loads(path.read_text()))
            assert registration_matches(
                existing, registration
            ), f"Registration changed: {phase.run_id}"
        else:
            store.register(registration)
        log("uploader_registered", run_id=phase.run_id, keep=2)


def check_wave(wave, commit):
    for key in WAVES[wave]:
        phase = PHASES[key]
        summary = json.loads((phase.root / "audit/summary.json").read_text())
        assert summary["completed"] and summary["source_commit"] == commit
        assert summary["end"] == phase.end and summary["tokens"] == phase.tokens_at(phase.end)
        if phase.end - phase.start == 60:
            assert summary["routing"]["telemetry_complete"]
            assert summary["measured_skipped_updates"] == 0
        for rank in range(phase.gpus):
            audit = phase.root / "audit"
            complete = json.loads((audit / f"complete-rank{rank}.json").read_text())
            assert complete["source_commit"] == commit and complete["step"] == phase.end
            assert (audit / f"restore-rank{rank}.json").is_file()
            if phase.save:
                assert (audit / f"state-step{phase.end}-rank{rank}.json").is_file()
        log("phase_gate_passed", **summary)


def main():
    from beaker import Beaker

    commit = os.environ["GIT_REF"]
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    assert MOUNT.is_mount()
    AUTOMATION.mkdir(parents=True, exist_ok=True)
    with (
        (AUTOMATION / "controller.lock").open("a") as lock,
        Beaker.from_env(check_for_upgrades=False) as b,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_preflight(b)
        training = b.experiment.get_spec(b.workload.get(PARENT_EXPERIMENT)).to_json()
        config = b.experiment.get_spec(b.workload.get(VALIDATION_TEMPLATE)).to_json()
        gate = ensure(
            b, f"{CAMPAIGN}-config{SUBMISSION_SUFFIX}", config_spec(config, commit, training)
        )
        wait_success(b, gate)
        for wave in ("resume-speed64", "speed128"):
            source_preflight(b)
            register_outputs(wave)
            eid = ensure(
                b, f"{CAMPAIGN}-{wave}{SUBMISSION_SUFFIX}", training_spec(training, wave, commit)
            )
            wait_success(b, eid)
            check_wave(wave, commit)
        # These are not placed in the GPU queue until the128GPU gate succeeds.
        # They start from the original fork, never from a timing-pass endpoint.
        branches = {}
        for wave in ("cbs32", "cbs64"):
            source_preflight(b)
            register_outputs(wave)
            branches[wave] = ensure(
                b, f"{CAMPAIGN}-{wave}{SUBMISSION_SUFFIX}", training_spec(training, wave, commit)
            )
        atomic_json(AUTOMATION / "branches-submitted.json", branches)
        log("cbs_branches_submitted", **branches)
        for wave, eid in branches.items():
            wait_success(b, eid)
            check_wave(wave, commit)
        log("campaign_complete", branches=branches)


if __name__ == "__main__":
    main()
