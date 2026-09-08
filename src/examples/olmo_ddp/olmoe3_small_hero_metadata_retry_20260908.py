"""Retry only the two cadence resumes blocked by W&B execution-metadata immutability."""

import copy
import fcntl
import os

from olmoe3_lr_sweep_watch import Controller, atomic_json, log, replace_env, status
from olmoe3_small_hero_cadence_20260908 import latest_complete, set_cancel_tag
from olmoe3_small_hero_control import preflight
from olmoe3_small_hero_plan import AUTOMATION, runs

PREVIOUS = {
    "emo": ("01M20XWNHVVVRBAX974TNAP7Q2", 16501, "lasc1m2x"),
    "non-emo": ("01M20XWV16MWGJC1A3X0DR5GWH", 12101, "aqb1droj"),
}
PREVIOUS_COMMIT = "930f215d44460b624cf91f485c03f18e8454689e"


def main():
    """Recheck unchanged source checkpoints and submit one metadata-fixed retry per arm."""
    from beaker import Beaker

    commit = os.environ["GIT_REF"]
    with (AUTOMATION / "LOCK").open("a") as lock, Beaker.from_env(
        check_for_upgrades=False
    ) as beaker:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        specs = {}
        for run in runs():
            old, expected, wandb_id = PREVIOUS[run.arm]
            workload = beaker.workload.get(old)
            assert status(workload) in ("STATUS_FAILED", "STATUS_CANCELED"), (old, status(workload))
            step, path = latest_complete(run)
            assert step == expected, "Unexpected checkpoint progress; require operator review"
            spec = beaker.experiment.get_spec(workload).to_json()
            assert len(spec["tasks"]) == 8
            for t in spec["tasks"]:
                env = {e["name"]: e.get("value") for e in t["envVars"]}
                assert env["GIT_REF"] == PREVIOUS_COMMIT
                assert env["OLMO35_HERO_EXPECTED_START"] == str(step)
                assert env["WANDB_RUN_ID"] == wandb_id and env["WANDB_RESUME"] == "must"
            t = copy.deepcopy(spec["tasks"][0])
            t.update(name="train", replicas=8, leaderSelection=True)
            replace_env(t, {"GIT_REF": commit})
            spec["tasks"] = [t]
            spec["retry"] = {"allowedTaskRetries": 0}
            specs[run.arm] = spec
            log(
                "HERO_METADATA_RETRY_CHECKPOINT_VERIFIED", run=run.run_id, step=step, path=str(path)
            )
        preflight(beaker)
        for _, _, wandb_id in PREVIOUS.values():
            set_cancel_tag(wandb_id, False)
        control = Controller(beaker, commit, automation=AUTOMATION)
        receipt = []
        for run in runs():
            workload = control.ensure(f"{run.run_id}-train-r4-cadence250-wandb", specs[run.arm])
            assert workload is not None, "Ambiguous submission; reconcile manually"
            old, step, wandb_id = PREVIOUS[run.arm]
            receipt.append(
                {
                    "run": run.run_id,
                    "experiment_id": workload.experiment.id,
                    "replaces": old,
                    "resume_step": step,
                    "wandb_id": wandb_id,
                    "source_commit": commit,
                }
            )
            atomic_json(AUTOMATION / "heroes-cadence-metadata-retry-20260908.json", receipt)
        log("BOTH_HERO_METADATA_FIXED_RESUMES_SUBMITTED", runs=receipt)


if __name__ == "__main__":
    main()
