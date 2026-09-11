"""Reuse frozen hero conversion/parity/OLMoBase workers, changing only scoped identities."""

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from olmoe3_hero_decay_plan import (
    AUTOMATION,
    CORE_REF,
    END,
    EVAL_ROOT,
    HELPER_REF,
    MOUNT,
    DecayRun,
    ready,
    verified_copy,
)
from olmoe3_lr_sweep_watch import atomic_json, replace_env

WORKSPACE = "ai2/OLMo-3-moe-experiments"
TEMPLATES = dict(
    convert="01M2663ZFTZ5AARQZBG95Q3H5D",
    qualify="01M26G35SKW26HPG7VJDJYCMXQ",
    gen_mc="01M22C3WYS48MXH1V46X7EHFYA",
    math="01M22C3X69YMKA6YSR5M0ACSK6",
    code="01M22D1V352S52XWTM8GPHTVXM",
)
OLD_HELPER = "974cf8bc48b6bf73a9bab92195a6615e2b55ad5d"
OLD_ROOT = "/weka/olmo-3p5-checkpoints/scratch/hero-hf-20260909"
WRAPPER = "/tmp/hero-decay-wrapper/src/examples/olmo_ddp/olmoe3_hero_decay_eval.py"


def build_spec(template, stage, run, commit):
    """Leave package/kernel/recipe pins intact; allocate all workers and forbid raw cleanup."""
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    spec = copy.deepcopy(template)
    assert len(spec["tasks"]) == 1 and stage in TEMPLATES
    task = spec["tasks"][0]
    command = task["arguments"][0]
    model = str(EVAL_ROOT / run.arm / f"step{END}/hf")
    fetch = (
        "git init --quiet /tmp/hero-decay-wrapper\n"
        "git -C /tmp/hero-decay-wrapper remote add origin https://github.com/allenai/OLMo-core.git\n"
        f"git -C /tmp/hero-decay-wrapper fetch --quiet --depth=1 origin {commit}\n"
        f"git -C /tmp/hero-decay-wrapper checkout --quiet {commit}\n"
    )
    if stage == "convert":
        old = (
            "python src/examples/olmo_ddp/hero_hf_stage.py --arm emo --step 75500\n"
            "python src/examples/olmo_ddp/hero_hf_convert.py --arm emo --step 75500 --full --portable-reference --precise"
        )
        assert command.count(old) == 1 and command.count(CORE_REF) == 2
        command = command.replace(
            old,
            fetch
            + f"python {WRAPPER} convert --arm {run.arm} --source /tmp/hero-conversion-source",
        )
    elif stage == "qualify":
        old_model = OLD_ROOT + "/emo/step75500/hf"
        old_cleanup = (
            "python /tmp/hero-core/src/examples/olmo_ddp/hero_hf_cleanup.py --arm emo --step 75500"
        )
        assert command.count(old_model) == 3 and command.count(old_cleanup) == 1
        assert command.count("c25f6df75faefd513c0b5214d4cd7de91810cb4d") == 2
        command = command.replace(old_model, model).replace(
            old_cleanup, "# No checkpoint deletion in this campaign."
        )
        command = command.replace("c25f6df75faefd513c0b5214d4cd7de91810cb4d", HELPER_REF)
    else:
        gpu_count = 4 if stage == "gen_mc" else 8
        old = f"python ladders/olmoe3/workloads/hero_full_eval.py {stage} {OLD_ROOT}/emo/step6000/hf --instances {gpu_count}"
        assert command.count(old) == 1 and command.count(OLD_HELPER) == 2
        command = command.replace(OLD_HELPER, HELPER_REF).replace(
            old,
            fetch + "python ladders/olmoe3/workloads/hero_full_eval_checks.py\n"
            f"python {WRAPPER} {stage} --arm {run.arm} --source /tmp/hero-ladder",
        )
        assert task["resources"]["gpuCount"] == gpu_count
        if stage == "code":
            assert (
                "hero_code_sandbox_preflight.py" in command
                and "google-cloud-cli-583.0.0" in command
            )
    task["arguments"] = [command]
    task["context"].update(priority="urgent", minRuntime="1h")
    task["timeout"] = "6h"
    replace_env(task, {"GIT_REF": commit})
    assert not task.get("result", {}).get("path")
    assert "hero_hf_cleanup.py" not in command
    spec["retry"] = {"allowedTaskRetries": 0}
    spec["description"] = json.dumps(
        dict(
            run=run.run_id,
            stage=stage,
            step=END,
            model=model,
            training="pretraining decay",
            inference_profile="bf16-grouped-fla-pilot-v1",
            note="Same provisional fast inference and frozen dense PT recipes as earlier hero evals; no new numerical qualification claim.",
        )
    )
    return spec


def eval_specs(beaker, run, commit):
    return {
        stage: build_spec(
            beaker.experiment.get_spec(beaker.workload.get(eid)).to_json(), stage, run, commit
        )
        for stage, eid in TEMPLATES.items()
    }


def qualify_source(run):
    root = EVAL_ROOT / run.arm / f"step{END}"
    conversion = root / "hf/_HERO_CONVERSION_SUCCESS.json"
    for name in ("conversion-success.json", "vllm-parity-success.json", "eval-smoke-success.json"):
        row = json.loads((root / name).read_text())
        assert row.get("passed") is True and not row.get("diagnostic_only")
        if name == "vllm-parity-success.json":
            assert (
                row["precise"]
                and row["conversion_sha256"] == hashlib.sha256(conversion.read_bytes()).hexdigest()
            )


def advance_evals(beaker, control, run, specs):
    """Advance only successful dependencies, retaining durable one-submit intents per stage."""
    old_workspace = control.workspace
    result = {}
    try:
        control.workspace = beaker.workspace.get(WORKSPACE)
        for stage in TEMPLATES:
            if stage == "qualify":
                row = json.loads(
                    (EVAL_ROOT / run.arm / f"step{END}/conversion-success.json").read_text()
                )
                assert row.get("passed") is True
            if stage not in ("convert", "qualify"):
                qualify_source(run)
            w = control.ensure(f"{run.run_id}-{stage}", specs[stage])
            state = control.report(w)
            result[stage] = dict(status=state, experiment=w.experiment.id if w else None)
            if stage in ("convert", "qualify") and state != "STATUS_SUCCEEDED":
                return result
        result["complete"] = all(row["status"] == "STATUS_SUCCEEDED" for row in result.values())
        if result["complete"]:
            root = EVAL_ROOT / run.arm / f"step{END}"
            for bundle in ("gen_mc", "math", "code"):
                markers = list(root.glob(f"pilot-olmobase-{bundle}-*/full-eval-pilot-success.json"))
                assert len(markers) == 1
                row = json.loads(markers[0].read_text())
                assert row["passed"] and row["bundle"] == bundle
                assert (
                    hashlib.sha256(Path(row["metrics"]).read_bytes()).hexdigest()
                    == row["metrics_sha256"]
                )
    finally:
        control.workspace = old_workspace
    return result


def prepare_scratch():
    assert MOUNT.is_mount()
    assert EVAL_ROOT.resolve() == EVAL_ROOT
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    path = EVAL_ROOT / "_OWNER.json"
    owner = dict(campaign="olmo35-small-decay2t-20260911", scratch=str(EVAL_ROOT))
    if path.exists():
        assert json.loads(path.read_text()) == owner
    else:
        atomic_json(path, owner)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("convert", "gen_mc", "math", "code"))
    parser.add_argument("--arm", choices=("emo", "non-emo"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    run = DecayRun(args.arm)
    expected_ref = CORE_REF if args.stage == "convert" else HELPER_REF
    assert (
        subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip()
        == expected_ref
    )
    assert not subprocess.check_output(
        ["git", "-C", str(args.source), "status", "--porcelain"], text=True
    ).strip()
    prepare_scratch()
    root = EVAL_ROOT / run.arm / f"step{END}"
    if args.stage == "convert":
        assert ready(run, END)
        copy_receipt = verified_copy(run.root / f"step{END}", root / "olmo-core", END)
        atomic_json(
            root / "download-success.json",
            dict(
                passed=True,
                source_kind="local_copy",
                source=str(run.root / f"step{END}"),
                arm=run.arm,
                step=END,
                tokens=END * 16777216,
                raw_path=str(root / "olmo-core"),
                all_file_hashes_verified=True,
                bytes=copy_receipt["total_bytes"],
            ),
        )
        atomic_json(root / "source-receipt.json", copy_receipt)
        sys.path.insert(0, str(args.source / "src/examples/olmo_ddp"))
        import hero_hf_convert as convert

        # Only the allowlisted path/step adapter changes. Conversion and every tolerance are frozen.
        convert.SCRATCH = EVAL_ROOT
        convert.TARGETS = {2013: END}
        convert.prepare_scratch = prepare_scratch
        sys.argv = [
            str(args.source / "src/examples/olmo_ddp/hero_hf_convert.py"),
            "--arm",
            run.arm,
            "--step",
            str(END),
            "--full",
            "--portable-reference",
            "--precise",
        ]
        convert.main()
    else:
        qualify_source(run)
        sys.path.insert(0, str(args.source / "ladders/olmoe3/workloads"))
        import hero_full_eval as runtime

        runtime.FAST_MODELS = {EVAL_ROOT / arm / f"step{END}/hf" for arm in ("emo", "non-emo")}
        sys.argv = [
            runtime.__file__,
            args.stage,
            str(root / "hf"),
            "--instances",
            "4" if args.stage == "gen_mc" else "8",
            "--fast-pilot",
        ]
        runtime.main()


if __name__ == "__main__":
    main()
