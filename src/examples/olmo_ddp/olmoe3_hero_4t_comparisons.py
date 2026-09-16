"""Defer token-matched stable/base evals until both 4T decays pass startup."""

import copy
import fcntl
import json
import os
import time

from olmoe3_hero_decay_plan import AUTOMATION as DECAY_AUTOMATION, DecayRun, ready
from olmoe3_hero_4t_stable import ROOT, STEPS, specs, setup
from olmoe3_hero_4t_baselines import MODELS, spec as baseline_spec
from olmoe3_small_hero_plan import Run, MOUNT, STATE, BUCKET
from olmoe3_lr_sweep_watch import Controller, atomic_json, log, status

AUTOMATION = DECAY_AUTOMATION / "comparisons"
REUSED = {
    "hero-3t-emo-gen_mc": "01M2F8ZCE4TJN47VTYRRP0S5N6",
    "hero-3t-emo-math": "01M2F8ZCTX2W3K1375EBZMNM2N",
    "hero-3t-emo-code": "01M2F8ZD863AWGTPKKKKDHF560",
    "hero-3t-non-emo-gen_mc": "01M2FZ4RB4WJ1PSS9505NQGS1Z",
    "hero-3t-non-emo-math": "01M2FZ4RRGN03T5HCXKEZ6DB52",
    "hero-3t-non-emo-code": "01M2FZ4S5MJRBVJ18KBSXVR1H5",
    "olmo3-3t-gen_mc": "01M2F7CKENMPA1SXTT5B9YWE3Q",
    "olmo3-3t-math": "01M2F7CKW9234YK91TFVFFD3J0",
    "olmo3-3t-code": "01M2F7CM9DAFEJSE7CASHGJG1B",
    "hybrid-3t-gen_mc": "01M2F7CMP9J94V7480V12W3AZQ",
    "hybrid-3t-math": "01M2F7CN49TKCSE6DBGJXZG176",
}


def startup_ready(b):
    for arm in ("emo", "non-emo"):
        r = DecayRun(arm)
        path = r.root / "audit/resume-smoke-saved.json"
        if not path.is_file():
            return False
        proof = json.loads(path.read_text())
        assert proof["all_64_ranks_verified"] and proof["step"] == 216002
        if status(b.workload.get(proof["experiment"])) not in (
            "STATUS_RUNNING",
            "STATUS_SUCCEEDED",
        ):
            return False
        for rank in range(64):
            p = r.root / "audit" / f"full-restore-step216002-rank{rank}.json"
            if not p.is_file():
                return False
            row = json.loads(p.read_text())
            assert all(row[k] for k in ("sampled_state_exact", "rng_exact", "data_state_exact"))
    return True


def available(arm, step):
    run = Run(arm == "emo")
    if ready(run, step):
        return True
    path = STATE / "checkpoints" / run.run_id / f"step-{step:012d}.json"
    if not path.is_file():
        return False
    row = json.loads(path.read_text())
    assert row["step"] == step and row["run_id"] == row["lineage_id"] == run.run_id
    assert row["bucket_id"] == BUCKET
    return row.get("remote_verified") is True and row.get("source_complete") is True


def main():
    from beaker import Beaker, BeakerExperimentSpec
    from olmoe3_hero_4t_eval_policy import validate_export

    assert MOUNT.is_mount()
    AUTOMATION.mkdir(parents=True, exist_ok=True)
    commit = os.environ["GIT_REF"]
    with (AUTOMATION / "LOCK").open("a") as lock, Beaker.from_env(check_for_upgrades=False) as b:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = object.__new__(Controller)
        c.beaker, c.workspace, c.commit = b, b.workspace.get("ai2/OLMo-3-moe-experiments"), commit
        c.last_status, c.automation = {}, AUTOMATION
        for eid in REUSED.values():
            assert status(b.workload.get(eid)) == "STATUS_SUCCEEDED"
        # The only missing 3T baseline bundle was canceled and has no replacement.
        assert status(b.workload.get("01M2F7CNHH5FKF2HFMSRDMBD5Q")) == "STATUS_CANCELED"
        base_targets = [(key, bundle) for key in MODELS for bundle in ("gen_mc", "math", "code")]
        base_targets.append(("hybrid-step716000", "code"))
        base_specs = {(k, s): baseline_spec(b, k, s, commit) for k, s in base_targets}
        hero_specs = {
            (arm, step): specs(b, arm, step, commit) for arm in ("emo", "non-emo") for step in STEPS
        }
        for s in list(base_specs.values()) + [
            s for group in hero_specs.values() for s in group.values()
        ]:
            BeakerExperimentSpec.from_json(copy.deepcopy(s))
        atomic_json(
            AUTOMATION / "plan.json",
            dict(
                commit=commit,
                hero_steps=STEPS,
                baselines=MODELS,
                reused=REUSED,
                missing_3t="hybrid-step716000/code",
                wait_for="both_decay_2step_and_full_restore",
            ),
        )
        log("FOUR_T_COMPARISONS_ARMED", gpus=0, reused_bundles=len(REUSED), new_baseline_bundles=13)
        previous = None
        while True:
            if not (AUTOMATION / "startup-confirmed.json").is_file():
                if not startup_ready(b):
                    log("COMPARISONS_WAIT_FOR_BOTH_DECAY_STARTUPS")
                    time.sleep(60)
                    continue
                atomic_json(
                    AUTOMATION / "startup-confirmed.json", dict(passed=True, time=time.time())
                )
                log("BOTH_DECAYS_STARTUP_CONFIRMED")
            free = os.statvfs(MOUNT)
            if free.f_bavail * free.f_frsize < 12_000_000_000_000:
                log("COMPARISONS_WAIT_FOR_STORAGE")
                time.sleep(60)
                continue
            rows = {}
            for (key, bundle), s in base_specs.items():
                name = f"olmo35-4t-compare-{key}-{bundle}-20260916"
                w = c.ensure(name, s)
                rows[key + "/" + bundle] = dict(
                    status=c.report(w), experiment=w.experiment.id if w else None
                )
            for (arm, step), group in hero_specs.items():
                key = f"hero-{arm}-step{step}"
                if not available(arm, step):
                    rows[key] = {"waiting": "verified_checkpoint"}
                    continue
                part = {}
                for stage in ("convert", "gen_mc", "math", "code"):
                    if stage != "convert":
                        validate_export(ROOT / arm / f"step{step}/hf")
                    w = c.ensure(
                        "olmo35-4t-compare-" + key + "-" + stage + "-20260916", group[stage]
                    )
                    state = c.report(w)
                    part[stage] = dict(status=state, experiment=w.experiment.id if w else None)
                    if stage == "convert" and state != "STATUS_SUCCEEDED":
                        break
                rows[key] = part
            atomic_json(
                AUTOMATION / "status.json", dict(updated_at=time.time(), runs=rows, reused=REUSED)
            )
            if rows != previous:
                log("FOUR_T_COMPARISONS_STATUS", runs=rows)
                previous = rows
            time.sleep(60)


if __name__ == "__main__":
    main()
