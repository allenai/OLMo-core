"""Scoped epoch exports using the already-qualified LC HF conversion, no raw deletion."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from olmoe3_hero_sft_plan import BATCH, CAMPAIGN, DATA, MOUNT, data_plan, find_run
from olmoe3_lr_sweep_watch import atomic_json, status


def export_root(run):
    return MOUNT / "scratch" / CAMPAIGN / run.run_id


def spec_for(beaker, run, step, commit):
    from olmoe3_hero_decay_eval import TEMPLATES, build_spec

    spec = build_spec(
        beaker.experiment.get_spec(beaker.workload.get(TEMPLATES["convert"])).to_json(),
        "convert",
        run,
        commit,
    )
    task = spec["tasks"][0]
    old = (
        "python /tmp/hero-decay-wrapper/src/examples/olmo_ddp/olmoe3_hero_decay_eval.py convert --arm "
        + run.arm
    )
    new = (
        "python /tmp/hero-decay-wrapper/src/examples/olmo_ddp/olmoe3_hero_sft_convert.py --run "
        + run.run_id
        + f" --step {step}"
    )
    assert task["arguments"][0].count(old) == 1
    task["arguments"][0] = task["arguments"][0].replace(old, new)
    if not any(d["mountPath"] == "/weka/oe-adapt-default" for d in task["datasets"]):
        task["datasets"].append(
            {"mountPath": "/weka/oe-adapt-default", "source": {"weka": "oe-adapt-default"}}
        )
    spec["description"] = json.dumps(
        dict(
            stage="sft-epoch-conversion",
            step=step,
            output=str(export_root(run) / run.arm / f"step{step}/hf"),
            **run.as_dict(),
        )
    )
    return spec


def advance_conversions(beaker, control, run, commit):
    states = {}
    for epoch, step in enumerate((data_plan()["steps_per_epoch"], data_plan()["total_steps"]), 1):
        workload = control.ensure(
            run.run_id + f"-epoch{epoch}-convert", spec_for(beaker, run, step, commit)
        )
        state = status(workload) if workload else "waiting"
        if state == "STATUS_SUCCEEDED":
            receipt = json.loads(
                (export_root(run) / run.arm / f"step{step}/conversion-success.json").read_text()
            )
            assert receipt["passed"] and receipt["step"] == step
        states[str(epoch)] = {
            "status": state,
            "experiment": workload.experiment.id if workload else None,
        }
    states["complete"] = all(v["status"] == "STATUS_SUCCEEDED" for v in states.values())
    return states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    run = find_run(args.run)
    assert not run.smoke and args.step in (
        data_plan()["steps_per_epoch"],
        data_plan()["total_steps"],
    )
    import olmoe3_hero_decay_plan as copy_worker
    from olmoe3_hero_sft_node import validate_checkpoint

    assert (
        subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip()
        == copy_worker.CORE_REF
    )
    assert not subprocess.check_output(
        ["git", "-C", str(args.source), "status", "--porcelain"], text=True
    ).strip()
    scratch = export_root(run)
    assert MOUNT.is_mount() and scratch.resolve() == scratch
    atomic_json(scratch / "_OWNER.json", {"campaign": CAMPAIGN, "run": run.run_id, "stage": "sft"})
    root = scratch / run.arm / f"step{args.step}"
    # The copy/hash algorithm is unchanged; SFT has eight rank audits and a new batch.
    copy_worker.validate_checkpoint = validate_checkpoint
    receipt = copy_worker.verified_copy(
        run.root / f"step{args.step}", root / "olmo-core", args.step
    )
    atomic_json(
        root / "download-success.json",
        {
            "passed": True,
            "source_kind": "local_copy",
            "source": receipt["source"],
            "arm": run.arm,
            "step": args.step,
            "tokens": args.step * BATCH,
            "raw_path": str(root / "olmo-core"),
            "all_file_hashes_verified": True,
            "bytes": receipt["total_bytes"],
        },
    )
    atomic_json(root / "sft-provenance.json", run.as_dict())
    sys.path.insert(0, str(args.source / "src/examples/olmo_ddp"))
    import hero_hf_convert as convert

    import olmo_core.nn.hf.convert_checkpoint as hf_converter

    original = hf_converter.convert_checkpoint_to_hf

    def convert_sft(*positional, **keywords):
        assert keywords["max_sequence_length"] == 8192
        keywords["max_sequence_length"] = 65536
        # Use the exact saved tokenizer/template, before qualification and checksums.
        keywords["tokenizer_id"] = str(DATA / "train/tokenizer")
        return original(*positional, **keywords)

    hf_converter.convert_checkpoint_to_hf = convert_sft
    convert.SCRATCH, convert.TARGETS, convert.BATCH = scratch, {100: args.step}, BATCH
    convert.prepare_scratch = lambda: None  # The scoped ownership/mount check above replaces PT's.
    sys.argv = [
        convert.__file__,
        "--arm",
        run.arm,
        "--step",
        str(args.step),
        "--full",
        "--portable-reference",
        "--precise",
    ]
    convert.main()
    config = json.loads((root / "hf/config.json").read_text())
    assert config["max_position_embeddings"] == 65536
    assert (root / "hf/chat_template.jinja").read_text() == (
        DATA / "train/tokenizer/chat_template.jinja"
    ).read_text()


if __name__ == "__main__":
    main()
