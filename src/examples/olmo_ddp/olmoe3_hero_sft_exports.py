"""Export-only recovery controller; cannot create, resume, or change any training job."""

import fcntl
import json
import os
import time

from olmoe3_hero_sft_convert import export_root, spec_for
from olmoe3_hero_sft_plan import AUTOMATION, MOUNT, runs
from olmoe3_lr_sweep_watch import Controller, atomic_json, log, status

WORKSPACE = "ai2/OLMo-3-moe-experiments"
DEPLOYMENT = AUTOMATION / "deployments/exports-r1"
TRAINING_PIN = "2610a90ced51542c10848a7d82e9534f3ef65923"


class ExportController(Controller):
    """Use the allocated evaluation workspace and a fresh, isolated intent namespace."""

    def __init__(self, beaker, commit):
        self.beaker = beaker
        self.workspace = beaker.workspace.get(WORKSPACE)
        self.commit = commit
        self.last_status = {}
        self.automation = DEPLOYMENT


def main():
    from beaker import Beaker

    assert MOUNT.is_mount()
    DEPLOYMENT.mkdir(parents=True, exist_ok=True)
    commit = os.environ["GIT_REF"]
    with (DEPLOYMENT / "LOCK").open("a") as lock, Beaker.from_env(check_for_upgrades=False) as b:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control = ExportController(b, commit)
        for run in runs():
            proof = json.loads((run.root / "audit/sft-success.json").read_text())
            assert proof["source_commit"] == TRAINING_PIN and proof["step"] == 1810
            assert proof["all_8_ranks_verified"]
            assert status(b.workload.get(proof["experiment"])) == "STATUS_SUCCEEDED"
        # Qualify one final export before fan-out. Then prioritize all final checkpoints.
        todo = [(r, epoch, step) for epoch, step in ((2, 1810), (1, 905)) for r in runs()]
        previous = None
        while True:
            snapshot = {}
            active = 0
            pilot_ok = False
            for index, (run, epoch, step) in enumerate(todo):
                key = run.run_id + f"-epoch{epoch}"
                name = key + "-convert-r1"
                intent = DEPLOYMENT / "submissions" / (name + ".json")
                exists = intent.exists()
                free = os.statvfs(MOUNT)
                storage_ok = free.f_bavail * free.f_frsize > 12_000_000_000_000
                if not exists and (not storage_ok or active >= 4 or (index and not pilot_ok)):
                    snapshot[key] = {"status": "waiting_for_pilot_capacity_or_storage"}
                    continue
                work = control.ensure(name, spec_for(b, run, step, commit))
                state = control.report(work) if work else "ambiguous_submission"
                snapshot[key] = {
                    "status": state,
                    "experiment": work.experiment.id if work else None,
                }
                if state == "STATUS_SUCCEEDED":
                    root = export_root(run) / run.arm / f"step{step}"
                    receipt = json.loads((root / "conversion-success.json").read_text())
                    metadata = json.loads((root / "hf/sft-metadata-audit.json").read_text())
                    assert receipt["passed"] and receipt["step"] == step and metadata["passed"]
                    if index == 0:
                        pilot_ok = True
                elif state not in ("STATUS_FAILED", "STATUS_CANCELED", "STATUS_STOPPED"):
                    active += 1
            atomic_json(
                DEPLOYMENT / "status.json", {"updated_at": time.time(), "exports": snapshot}
            )
            if snapshot != previous:
                log("SFT_EXPORT_STATUS", exports=snapshot)
                previous = snapshot
            if all(v["status"] == "STATUS_SUCCEEDED" for v in snapshot.values()):
                log("SFT_EXPORTS_COMPLETE", count=len(snapshot))
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
