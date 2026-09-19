"""Wait for the one integration SFT export, then run the matched four-suite evals."""

import fcntl
import json
import os
import time

from olmoe3_hero_4t_eval_policy import validate_export
from olmoe3_hero_decay_eval import TEMPLATES
from olmoe3_hero_sft_convert import export_root
from olmoe3_hero_sft_eval_control import BUNDLES, spec_for
from olmoe3_hero_sft_exports import ExportController
from olmoe3_hero_sft_plan import AUTOMATION, MOUNT, data_plan, runs
from olmoe3_lr_sweep_watch import atomic_json, log


def main():
    """Use durable intents; do not retry failed evals or launch any other models."""
    from beaker import Beaker

    directory = AUTOMATION / "eval-controller"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "LOCK").open("a") as lock, Beaker.from_env(check_for_upgrades=False) as b:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = ExportController(b, os.environ["GIT_REF"])
        c.automation = directory
        template = b.experiment.get_spec(b.workload.get(TEMPLATES["gen_mc"])).to_json()
        (run,) = runs()
        while True:
            if not (AUTOMATION / "data-plan.json").is_file():
                time.sleep(30)
                continue
            data_plan()
            model = export_root(run) / run.arm / f"step{run.total_steps}/hf"
            if not (model / "_HERO_CONVERSION_SUCCESS.json").is_file():
                time.sleep(60)
                continue
            validate_export(model)
            assert json.loads((model / "sft-metadata-audit.json").read_text())["passed"]
            states = {}
            for bundle in BUNDLES:
                spec = spec_for(template, bundle, run, c.commit)
                w = c.ensure(run.run_id + "-epoch2-" + bundle, spec)
                assert w is not None
                states[bundle] = {"experiment": w.experiment.id, "status": c.report(w)}
            atomic_json(directory / "status.json", states)
            if all(s["status"] == "STATUS_SUCCEEDED" for s in states.values()):
                log("INTEGRATION_SFT_EVALS_COMPLETE", states=states)
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
