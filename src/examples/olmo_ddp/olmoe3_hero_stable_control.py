"""CPU-only PT120000 watcher: independent arms, durable submissions, no failure retries."""

import argparse
import copy
import fcntl
import json
import os
import time

from olmoe3_hero_stable_eval import (
    ARMS,
    AUTOMATION,
    CAMPAIGN,
    END,
    MOUNT,
    ROOT,
    STAGES,
    WORKSPACE,
    build_specs,
    completed_bundle,
    converted,
    parent,
    prepare_scratch,
    qualified,
    ready,
    stage_source,
)
from olmoe3_lr_sweep_watch import Controller, atomic_json, log


def identity(spec):
    """Compare operational fields and require urgent, allocated, payload-free workers."""
    assert len(spec["tasks"]) == 1
    t = spec["tasks"][0]
    assert t["context"]["priority"] == "urgent"
    assert t["context"]["minRuntime"] in ("1h", "1h0m0s", "3600s", 3600000000000)
    assert not t.get("result", {}).get("path")
    return {
        k: t.get(k)
        for k in (
            "command",
            "arguments",
            "image",
            "datasets",
            "envVars",
            "resources",
            "constraints",
        )
    }


class EvalController(Controller):
    """Reuse exact-name/intent reconciliation without unrelated training initialization."""

    def __init__(self, beaker, commit):
        self.beaker = beaker
        self.commit = commit
        self.workspace = beaker.workspace.get(WORKSPACE)
        self.automation = AUTOMATION
        self.last_status = {}

    def ensure(self, name, spec):
        work = super().ensure(name, spec)
        if work is not None:
            assert identity(self.beaker.experiment.get_spec(work).to_json()) == identity(spec)
        return work


def advance(control, arm, specs):
    """Stage exactly one source, then release only verified successful dependencies."""
    if not stage_source(arm):
        return dict(state="waiting_for_checkpoint", step=END, parent=parent(arm).run_id)
    fs = os.statvfs(MOUNT)
    if fs.f_bavail * fs.f_frsize < 12_000_000_000_000:
        return dict(state="waiting_for_storage", source_staged=True)
    result = dict(state="in_progress", source_staged=True, jobs={})
    for stage in STAGES:
        if stage == "qualify":
            converted(arm)
        elif stage not in ("convert", "qualify"):
            qualified(arm)
        work = control.ensure(f"{CAMPAIGN}-{arm}-{stage}", specs[stage])
        state = control.report(work)
        result["jobs"][stage] = dict(
            status=state, experiment=work.experiment.id if work is not None else None
        )
        if stage in ("convert", "qualify") and state != "STATUS_SUCCEEDED":
            if state in ("STATUS_FAILED", "STATUS_CANCELED", "STATUS_CANCELLED", "AMBIGUOUS"):
                result["state"] = "needs_attention"
            return result
        if stage not in ("convert", "qualify") and state == "STATUS_SUCCEEDED":
            completed_bundle(arm, stage)
    states = [v["status"] for v in result["jobs"].values()]
    if all(s == "STATUS_SUCCEEDED" for s in states):
        result["state"] = "complete"
    elif any(
        s in ("STATUS_FAILED", "STATUS_CANCELED", "STATUS_CANCELLED", "AMBIGUOUS") for s in states
    ):
        result["state"] = "needs_attention"
    return result


def main():
    from beaker import Beaker, BeakerExperimentSpec

    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    commit = os.environ["GIT_REF"]
    assert MOUNT.is_mount(), "Refuse container-overlay staging/state"
    with Beaker.from_env(check_for_upgrades=False) as b:
        specs = {arm: build_specs(b, arm, commit) for arm in ARMS}
        for stages in specs.values():
            for spec in stages.values():
                parsed = BeakerExperimentSpec.from_json(copy.deepcopy(spec)).to_json()
                assert identity(parsed) == identity(spec)
        readiness = {arm: ready(parent(arm), END) for arm in ARMS}
        log(
            "STABLE_EVAL_PREFLIGHT_PASSED",
            source_commit=commit,
            ready=readiness,
            worker_specs=10,
            gpu_resources=0,
            step=END,
            tokens=END * 16777216,
        )
        if args.check_only:
            return
        AUTOMATION.mkdir(parents=True, exist_ok=True)
        with (AUTOMATION / "LOCK").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            plan = dict(
                campaign=CAMPAIGN,
                source_commit=commit,
                step=END,
                arms=list(ARMS),
                output_root=str(ROOT),
                workspace=WORKSPACE,
            )
            path = AUTOMATION / "plan.json"
            if path.exists():
                assert json.loads(path.read_text()) == plan, "Controller pin/target drift"
            else:
                atomic_json(path, plan)
            prepare_scratch()
            control = EvalController(b, commit)
            previous = None
            log("STABLE_EVAL_WATCHER_ARMED", **plan, poll_seconds=30, gpu_resources=0)
            while True:
                snapshot = {}
                for arm in ARMS:
                    try:
                        snapshot[arm] = advance(control, arm, specs[arm])
                    except Exception as error:
                        # A failed arm never blocks the other. Durable intents make
                        # reconciliation safe; no failed experiment is resubmitted.
                        snapshot[arm] = dict(
                            state="needs_attention", error=f"{type(error).__name__}: {error}"
                        )
                atomic_json(
                    AUTOMATION / "status.json",
                    dict(updated_at=time.time(), source_commit=commit, arms=snapshot),
                )
                if snapshot != previous:
                    log("STABLE_EVAL_STATUS", arms=snapshot)
                    previous = snapshot
                if all(row["state"] == "complete" for row in snapshot.values()):
                    log("BOTH_STABLE_2T_EVALS_COMPLETE")
                    return
                time.sleep(30)


if __name__ == "__main__":
    main()
