"""Restore exactly one approved 4T fork and submit its audited 64-GPU decay once."""

import argparse
import copy
import fcntl
import json
import logging
import os
from pathlib import Path

from olmoe3_hero_decay_control import training_spec
from olmoe3_hero_decay_plan import AUTOMATION, BRANCH, PARENTS, START, DecayRun, validate_checkpoint
from olmoe3_lr_sweep_watch import Controller, atomic_json, log, status
from olmoe3_small_hero_plan import CONTROL, MOUNT, STATE, UPLOADER, WORKSPACE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("emo", "non-emo"))
    args = parser.parse_args()
    from beaker import Beaker, BeakerExperimentSpec
    from huggingface_hub import HfApi
    from huggingface_hub.utils import disable_progress_bars
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore
    import olmoe3_hero_bucket_download as download

    logging.basicConfig(level=logging.INFO)
    disable_progress_bars()
    assert MOUNT.is_mount()
    r = DecayRun(args.arm)
    commit = os.environ["GIT_REF"]
    AUTOMATION.mkdir(parents=True, exist_ok=True)
    with (AUTOMATION / f"restore-{r.arm}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        download.SCRATCH = AUTOMATION / "sources"
        download.SCRATCH.mkdir(parents=True, exist_ok=True)
        assert download.SCRATCH.resolve() == download.SCRATCH
        api = HfApi()
        assert api.bucket_info(r.bucket).private
        source = download.download(api, r.arm, START)
        assert source == r.source
        validate_checkpoint(source, START)
        receipt = json.loads((source.parent / "source-receipt.json").read_text())
        assert receipt["lineage_id"] == r.parent.run_id
        assert receipt["step"] == START
        with Beaker.from_env(check_for_upgrades=False) as b:
            assert status(b.workload.get(UPLOADER)) == "STATUS_RUNNING"
            assert os.statvfs(MOUNT).f_bavail * os.statvfs(MOUNT).f_frsize >= 12_000_000_000_000
            spec = training_spec(b.experiment.get_spec(b.workload.get(PARENTS[r.arm])).to_json(), r, commit)
            task = spec["tasks"][0]
            task["constraints"]["hostname"] = [h for h in task["constraints"]["hostname"] if h != "holmes-cs-aus-520.reviz.ai2.in"]
            BeakerExperimentSpec.from_json(copy.deepcopy(spec))
            store = StateStore(CONTROL, STATE)
            r.root.mkdir(parents=True, exist_ok=True)
            store.register(Registration(run_id=r.run_id, lineage_id=r.run_id,
                checkpoint_root=str(r.root), bucket_id=r.bucket, remote_prefix=r.prefix,
                deletion_mode="apply", min_local_checkpoints=2, delete_grace_seconds=3600))
            controller = object.__new__(Controller)
            controller.beaker = b
            controller.workspace = b.workspace.get(WORKSPACE)
            controller.commit = commit
            controller.last_status = {}
            controller.automation = AUTOMATION
            job = controller.ensure(r.run_id + "-train", spec)
            assert job is not None
            record = dict(**r.as_dict(), source_commit=commit, branch=BRANCH,
                experiment=job.experiment.id, state=status(job), restored_from_hf=True,
                inference_numerical_gate="disabled_by_user_20260916")
            atomic_json(AUTOMATION / f"launched-{r.arm}.json", record)
            log("FOUR_T_DECAY_SUBMITTED", **record)


if __name__ == "__main__":
    main()
