"""Independent conversion/eval controller; no inference numerical-parity dependency."""

import argparse
import copy
import fcntl
import importlib
import json
import os
import subprocess
import sys
import time

from olmoe3_hero_decay_plan import AUTOMATION as CAMPAIGN_ROOT, MOUNT
from olmoe3_lr_sweep_watch import Controller, atomic_json, log, status

AUTOMATION = CAMPAIGN_ROOT / "eval-pipeline"
STAGES = ("decay", "mt", "lc", "sft")
WORKSPACE = "ai2/OLMo-3-moe-experiments"
PT_MT_LC_EVAL_COMMIT = "ca80837014c92771ca19b4bedafcd8c0dbd082c7"
TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def temperature_spec(spec, temperature):
    """Change only sampling temperature and scheduling for a bounded SFT sweep."""
    from olmoe3_lr_sweep_watch import replace_env

    assert temperature in TEMPERATURES
    spec = copy.deepcopy(spec)
    task = spec["tasks"][0]
    replace_env(task, {"HERO_SFT_TEMPERATURE": temperature})
    task["context"].update(priority="urgent", minRuntime="8h", autoResume=True)
    description = json.loads(spec["description"])
    description.update(
        temperature=temperature,
        sampling=f"Think final answers; T={temperature} P=.95 max_new_tokens=32768 seed=1234 n=1",
        sweep="4t-sft-temperature-20260918",
    )
    spec["description"] = json.dumps(description)
    return spec


def stage_call(stage):
    """Contain a slow/failed API tick without killing other stages or the watcher."""
    try:
        return subprocess.run(
            [sys.executable, __file__, "--tick", stage],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        # Durable per-submission intents are reconciled on the next tick. Never
        # create new experiment names merely because the API response timed out.
        log("FOUR_T_EVAL_TICK_RETRY", stage=stage, error_type=type(exc).__name__)
        return None


def controller(b, commit, stage):
    c = object.__new__(Controller)
    c.beaker, c.workspace, c.commit = b, b.workspace.get(WORKSPACE), commit
    c.last_status, c.automation = {}, AUTOMATION / stage
    return c


def record(c, name, spec):
    from beaker import BeakerExperimentSpec

    BeakerExperimentSpec.from_json(copy.deepcopy(spec))
    w = c.ensure(name, spec)
    return dict(status=c.report(w), experiment=w.experiment.id if w else None)


def tick(stage, b, commit, validate_only=False):
    # Each stage gets a fresh interpreter: older MT/LC adapters have module-level
    # path globals that must never be mixed in a single eval controller process.
    if stage != "sft":
        # Existing evals retain their exact specs/receipts; no duplicate attempts.
        commit = PT_MT_LC_EVAL_COMMIT
    c = controller(b, commit, stage)
    result = {}
    if stage == "sft":
        from olmoe3_hero_sft_plan import DATA_PLAN, runs
        from olmoe3_hero_sft_convert import export_root, spec_for as convert_spec
        from olmoe3_hero_sft_eval_control import spec_for, BUNDLES
        from olmoe3_hero_decay_eval import TEMPLATES
        from olmoe3_hero_4t_eval_policy import validate_export

        template = b.experiment.get_spec(b.workload.get(TEMPLATES["gen_mc"])).to_json()
        for r in runs():
            if not DATA_PLAN.is_file():
                result[r.run_id] = {"waiting": "sft_data_config_gate"}
                continue
            final_step = r.total_steps
            specs = {str(final_step): convert_spec(b, r, final_step, commit)}
            specs.update({bundle: spec_for(template, bundle, r, commit) for bundle in BUNDLES})
            if validate_only:
                from beaker import BeakerExperimentSpec

                for s in specs.values():
                    BeakerExperimentSpec.from_json(copy.deepcopy(s))
                result[r.run_id] = "validated"
                continue
            proof_path = r.root / "audit/sft-success.json"
            if not proof_path.is_file():
                result[r.run_id] = {"waiting": "sft_final_checkpoint"}
                continue
            proof = json.loads(proof_path.read_text())
            assert proof["step"] == final_step and proof["all_8_ranks_verified"]
            if status(b.workload.get(proof["experiment"])) != "STATUS_SUCCEEDED":
                result[r.run_id] = {"waiting": "successful_sft_exit"}
                continue
            rows = {}
            rows[str(final_step)] = record(
                c, r.run_id + f"-step{final_step}-convert", specs[str(final_step)]
            )
            if rows[str(final_step)]["status"] == "STATUS_SUCCEEDED":
                model = export_root(r) / r.arm / f"step{final_step}/hf"
                validate_export(model)
                assert json.loads((model / "sft-metadata-audit.json").read_text())["passed"]
                for bundle in BUNDLES:
                    rows[bundle] = record(c, r.run_id + "-epoch2-" + bundle, specs[bundle])
                    if rows[bundle]["status"] == "STATUS_SUCCEEDED":
                        receipt = json.loads(
                            (model.parent / f"posttrain-evals-r1/{bundle}/success.json").read_text()
                        )
                        assert receipt["passed"] and receipt["bundle"] == bundle
                if os.environ.get("HERO_SFT_TEMPERATURE_SWEEP") == "1":
                    rows["temperature_sweep"] = {}
                    for temperature in TEMPERATURES:
                        label = f"t{round(temperature * 10):02d}"
                        swept = rows["temperature_sweep"][label] = {}
                        for bundle in BUNDLES:
                            if temperature == 0.6:
                                # The canonical evaluations are the .6 cell;
                                # never run a duplicate copy of this baseline.
                                swept[bundle] = {**rows[bundle], "reused": True}
                                continue
                            name = r.run_id + "-epoch2-" + bundle + "-" + label
                            swept[bundle] = record(
                                c, name, temperature_spec(specs[bundle], temperature)
                            )
            result[r.run_id] = rows
    else:
        plan = importlib.import_module("olmoe3_hero_" + stage + "_plan")
        worker = importlib.import_module("olmoe3_hero_" + stage + "_eval")
        base = worker if stage == "decay" else worker.worker
        end = plan.END
        for r in plan.runs():
            specs = worker.eval_specs(b, r, commit)
            if stage == "lc":
                specs["ruler"] = worker.ruler_spec(b, r, commit)
            elif stage == "decay":
                import olmoe3_hero_ruler_control as rc

                specs["ruler"] = rc.worker_spec(
                    b.experiment.get_spec(b.workload.get(rc.WORKER_TEMPLATE)).to_json(),
                    "decay4t",
                    r.arm,
                    commit,
                )
            for s in specs.values():
                t = s["tasks"][0]
                t["context"].update(priority="urgent", minRuntime="6h", autoResume=True)
                t["timeout"] = "24h"
                from beaker import BeakerExperimentSpec

                BeakerExperimentSpec.from_json(copy.deepcopy(s))
            if validate_only:
                result[r.run_id] = "validated"
                continue
            proof_path = r.root / "audit" / f"{stage}-success.json"
            if not proof_path.is_file():
                result[r.run_id] = {"waiting": stage + "_final_checkpoint"}
                continue
            proof = json.loads(proof_path.read_text())
            assert proof["step"] == end and proof["all_64_ranks_verified"]
            if status(b.workload.get(proof["experiment"])) != "STATUS_SUCCEEDED":
                result[r.run_id] = {"waiting": "successful_training_exit"}
                continue
            rows = base.advance_evals(b, c, r, {k: v for k, v in specs.items() if k != "ruler"})
            if "ruler" in specs and rows["convert"]["status"] == "STATUS_SUCCEEDED":
                base.qualify_source(r)
                rows["ruler"] = record(c, r.run_id + "-ruler", specs["ruler"])
            result[r.run_id] = rows
    if not validate_only:
        atomic_json(
            AUTOMATION / (stage + "-status.json"), dict(updated_at=time.time(), runs=result)
        )
    print(json.dumps(result), flush=True)
    return result


def main():
    from beaker import Beaker

    p = argparse.ArgumentParser()
    p.add_argument("--tick", choices=STAGES)
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()
    commit = os.environ["GIT_REF"]
    if args.tick:
        with Beaker.from_env(check_for_upgrades=False) as b:
            return tick(args.tick, b, commit, args.validate_only)
    assert MOUNT.is_mount()
    AUTOMATION.mkdir(parents=True, exist_ok=True)
    with (AUTOMATION / "LOCK").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log(
            "FOUR_T_EVAL_PIPELINE_ARMED",
            commit=commit,
            gpus=0,
            numerical_parity="disabled_by_user_20260916",
            stages=STAGES,
        )
        previous = {}
        while True:
            free = os.statvfs(MOUNT)
            if free.f_bavail * free.f_frsize < 12_000_000_000_000:
                log("FOUR_T_EVAL_WAIT_STORAGE")
                time.sleep(60)
                continue
            for stage in STAGES:
                call = stage_call(stage)
                if call is None:
                    continue
                output = call.stdout.strip().splitlines()
                summary = output[-1] if output else ""
                if call.returncode:
                    log("FOUR_T_EVAL_NEEDS_ATTENTION", stage=stage, error=call.stderr[-5000:])
                elif previous.get(stage) != summary:
                    log("FOUR_T_EVAL_STATUS", stage=stage, status=summary)
                    previous[stage] = summary
            time.sleep(60)


if __name__ == "__main__":
    main()
