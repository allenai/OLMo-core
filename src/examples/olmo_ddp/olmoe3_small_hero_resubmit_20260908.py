"""One user-authorized placement-only retry of the already-qualified hero pair.

Keep the exact smoke-tested training pin, roots, and uploader registrations.
Use separate durable submission intents; never overwrite the failed attempts.
"""

import fcntl
import json

from olmoe3_lr_sweep_watch import Controller, atomic_json, log, status
from olmoe3_small_hero_control import preflight, training_spec
from olmoe3_small_hero_plan import AUTOMATION, EXCLUDED_HOSTNAMES, runs

TRAINING_COMMIT = "171be9bef9517c12efbb517c0c4504e99d56e4e9"
SMOKE = "01M1Z1AQSAGRZY6F3ZMC3S076Q"
OLD_CONTROLLER = "01M1Z16E1AHQ3QGR0YBXHR4S3F"
PREVIOUS = {
    "emo": "01M1Z7KDFTB5HMGEF86EQJ1VSM",
    "non-emo": "01M1Z7KHC4WYCSMVSDKYFKKW3B",
}


def main():
    """Recheck launch gates, then submit each replacement exactly once."""
    from beaker import Beaker

    assert "holmes-cs-aus-503.reviz.ai2.in" in EXCLUDED_HOSTNAMES
    assert AUTOMATION.is_dir(), "Original campaign state must already exist"
    with (
        (AUTOMATION / "LOCK").open("a") as lock,
        Beaker.from_env(check_for_upgrades=False) as beaker,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert status(beaker.workload.get(OLD_CONTROLLER)) == "STATUS_SUCCEEDED"
        assert status(beaker.workload.get(SMOKE)) == "STATUS_SUCCEEDED"
        receipt = json.loads((AUTOMATION / "smoke-passed.json").read_text())
        assert receipt["source_commit"] == TRAINING_COMMIT
        assert receipt["sampled_state_exact"] and receipt["init_and_data_match"]
        for experiment in PREVIOUS.values():
            assert status(beaker.workload.get(experiment)) in (
                "STATUS_FAILED",
                "STATUS_CANCELED",
            ), "Never create a duplicate of a live or completed hero"
        # Includes real mount/capacity/data/bucket/daemon checks. No payload deletion.
        preflight(beaker)
        control = Controller(beaker, TRAINING_COMMIT, automation=AUTOMATION)
        submitted = []
        for run in runs():
            workload = control.ensure(
                f"{run.run_id}-train-r2-exclude503",
                training_spec(control.template, run, TRAINING_COMMIT),
            )
            assert workload is not None, "Ambiguous submission: stop and reconcile"
            submitted.append(
                {
                    "run": run.run_id,
                    "experiment_id": workload.experiment.id,
                    "replaces": PREVIOUS[run.arm],
                    "status": control.report(workload),
                    "training_commit": TRAINING_COMMIT,
                    "excluded_hostnames": sorted(EXCLUDED_HOSTNAMES),
                }
            )
        atomic_json(AUTOMATION / "heroes-resubmitted-20260908-exclude503.json", submitted)
        log("BOTH_HERO_REPLACEMENTS_SUBMITTED", runs=submitted)


if __name__ == "__main__":
    main()
