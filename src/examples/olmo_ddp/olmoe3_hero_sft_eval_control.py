"""Qualify final SFT exports and submit four chat evals per model exactly once."""

import copy
import fcntl
import json
import os
import re
import time

from olmoe3_hero_decay_eval import HELPER_REF, OLD_HELPER, OLD_ROOT, TEMPLATES
from olmoe3_hero_sft_convert import export_root
from olmoe3_hero_sft_exports import ExportController
from olmoe3_hero_sft_plan import AUTOMATION, MOUNT, runs
from olmoe3_lr_sweep_watch import atomic_json, log, replace_env

DEPLOYMENT = AUTOMATION / "deployments/posteval-r1"
ALPACA_PIN = "cd543a149df89434d8a54582c0151c0b945c3d20"
BUNDLES = ("math500", "ifbench", "humaneval", "alpaca")
QUALIFICATION_PIN = "6f1ef188ded568f7535e081429287a7308dba695"


def model_path(run):
    """Return the sole authorized evaluation epoch for this run."""
    return export_root(run) / run.arm / "step1810/hf"


def spec_for(template, stage, run, commit):
    """Reuse proven inference environment, without changing model weights or kernels."""
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert stage in (*BUNDLES, "qualify", "smoke")
    spec = copy.deepcopy(template)
    task = spec["tasks"][0]
    command = task["arguments"][0]
    model = str(model_path(run))
    if stage == "qualify":
        old_model = OLD_ROOT + "/emo/step75500/hf"
        cleanup = (
            "python /tmp/hero-core/src/examples/olmo_ddp/hero_hf_cleanup.py --arm emo --step 75500"
        )
        old_pin = "c25f6df75faefd513c0b5214d4cd7de91810cb4d"
        assert command.count(old_model) == 3 and command.count(cleanup) == 1
        assert command.count(old_pin) == 2
        command = command.replace(old_model, model).replace(
            cleanup, "# No raw checkpoint deletion."
        )
        command = command.replace(old_pin, HELPER_REF)
    else:
        old = f"python ladders/olmoe3/workloads/hero_full_eval.py gen_mc {OLD_ROOT}/emo/step6000/hf --instances 4"
        assert command.count(old) == 1 and command.count(OLD_HELPER) == 2
        command = command.replace(OLD_HELPER, HELPER_REF)
        wrapper = "/tmp/hero-sft-evals"
        extra = (
            f"git init --quiet {wrapper}\n"
            f"git -C {wrapper} remote add origin https://github.com/allenai/OLMo-core.git\n"
            f"git -C {wrapper} fetch --quiet --depth=1 origin {commit}\n"
            f"git -C {wrapper} checkout --quiet {commit}\n"
            "uv pip install --python /tmp/hero-eval-env/bin/python 'ifbench==0.2.0' "
            f"'git+https://github.com/tatsu-lab/alpaca_eval.git@{ALPACA_PIN}' "
            "'modal==1.5.1' 'swe-rex[modal] @ git+https://github.com/jdahm/SWE-ReX.git@127191d83184b3b626f6e149b75ca437e51a814b'\n"
            f"export PYTHONPATH={wrapper}/src/examples/olmo_ddp\n"
            f"python {wrapper}/src/examples/olmo_ddp/olmoe3_hero_sft_eval.py "
            f"--run {run.run_id} --bundle {stage}"
        )
        command = command.replace(old, extra)
        task["resources"] = {
            "cpuCount": 12,
            "gpuCount": 1,
            "memory": "96 GiB",
            "sharedMemory": "16 GiB",
        }
    task["arguments"] = [command]
    task["context"].update(priority="urgent", minRuntime=3600000000000, autoResume=False)
    task["timeout"] = "12h"
    if not any(d["mountPath"] == "/weka/oe-adapt-default" for d in task["datasets"]):
        task["datasets"].append(
            {"mountPath": "/weka/oe-adapt-default", "source": {"weka": "oe-adapt-default"}}
        )
    replace_env(task, {"GIT_REF": commit, "HF_TOKEN": None})
    task["envVars"].append({"name": "HF_TOKEN", "secret": "jacobm_HF_TOKEN"})
    if stage in ("smoke", "humaneval"):
        task["envVars"] += [
            {"name": "MODAL_TOKEN_ID", "secret": "jacobm_MODAL_TOKEN_ID"},
            {"name": "MODAL_TOKEN_SECRET", "secret": "jacobm_MODAL_TOKEN_SECRET"},
        ]
    if stage in ("smoke", "alpaca"):
        task["envVars"].append(
            {"name": "OPENAI_API_KEY", "secret": "jacobm_HERO_SFT_OPENAI_API_KEY"}
        )
    assert not task.get("result", {}).get("path")
    assert "hero_hf_cleanup.py" not in command
    assert task["constraints"]["cluster"] == ["ai2/jupiter", "ai2/ceres"]
    spec["retry"] = {"allowedTaskRetries": 0}
    spec["description"] = json.dumps(
        {
            "run": run.run_id,
            "epoch": 2,
            "step": 1810,
            "stage": stage,
            "model": model,
            "commit": commit,
            "sampling": "Think final answers; T=.6 P=.95 max_new_tokens=32768 seed=1234 n=1",
            "inference_profile": "bf16-grouped-fla-pilot-v1"
            if stage != "qualify"
            else "strict-fp32-reference",
        }
    )
    return spec


def main():
    """Wait for conversions; one real four-task smoke gates the 24 full suites."""
    from beaker import Beaker

    assert MOUNT.is_mount()
    DEPLOYMENT.mkdir(parents=True, exist_ok=True)
    commit = os.environ["GIT_REF"]
    with (DEPLOYMENT / "LOCK").open("a") as lock, Beaker.from_env(check_for_upgrades=False) as b:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control = ExportController(b, commit)
        control.automation = DEPLOYMENT
        templates = {
            stage: b.experiment.get_spec(b.workload.get(TEMPLATES[stage])).to_json()
            for stage in ("qualify", "gen_mc")
        }
        all_runs = list(runs())
        previous = None
        while True:
            snapshot = {}
            pilot = model_path(all_runs[0]).parent / "posttrain-evals-r1/smoke/success.json"
            pilot_ok = pilot.is_file() and json.loads(pilot.read_text()).get("passed") is True
            for index, run in enumerate(all_runs):
                root = model_path(run).parent
                conversion = root / "conversion-success.json"
                if not conversion.is_file():
                    snapshot[run.run_id] = {"waiting": "conversion"}
                    continue
                assert json.loads(conversion.read_text())["passed"]
                states = {}
                stages = ["qualify"] + (["smoke"] if index == 0 else [])
                if pilot_ok:
                    stages += list(BUNDLES)
                for stage in stages:
                    # Preserve already-submitted qualification/smoke intents byte-for-byte.
                    # Full-suite workers have not been released yet and use this revision.
                    stage_commit = QUALIFICATION_PIN if stage in ("qualify", "smoke") else commit
                    control.commit = stage_commit
                    template = templates["qualify" if stage == "qualify" else "gen_mc"]
                    name = run.run_id + "-epoch2-" + stage + "-r1"
                    work = control.ensure(name, spec_for(template, stage, run, stage_commit))
                    state = control.report(work) if work else "ambiguous_submission"
                    states[stage] = {
                        "status": state,
                        "experiment": work.experiment.id if work else None,
                    }
                    if stage == "qualify" and state != "STATUS_SUCCEEDED":
                        break
                    if stage not in ("qualify", "smoke") and state == "STATUS_SUCCEEDED":
                        proof = json.loads(
                            (root / f"posttrain-evals-r1/{stage}/success.json").read_text()
                        )
                        assert proof["passed"] and proof["bundle"] == stage
                snapshot[run.run_id] = states
            atomic_json(
                DEPLOYMENT / "status.json",
                {"updated_at": time.time(), "pilot_passed": pilot_ok, "evals": snapshot},
            )
            if snapshot != previous:
                log("SFT_POSTTRAIN_STATUS", pilot_passed=pilot_ok, evals=snapshot)
                previous = snapshot
            if all(
                all(row.get(bundle, {}).get("status") == "STATUS_SUCCEEDED" for bundle in BUNDLES)
                for row in snapshot.values()
            ):
                log("SFT_POSTTRAIN_COMPLETE", models=6, suites=24)
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
