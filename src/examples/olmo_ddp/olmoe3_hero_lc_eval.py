"""Frozen hero conversion/qualification/OLMoBase recipes for independent LC endpoints."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import olmoe3_hero_decay_eval as worker
from olmoe3_hero_decay_plan import CORE_REF, HELPER_REF, ready, verified_copy
from olmoe3_hero_lc_plan import BATCH, CAMPAIGN, END, EVAL_ROOT, MOUNT, LCRun
from olmoe3_lr_sweep_watch import atomic_json, status

worker.EVAL_ROOT = EVAL_ROOT
worker.END = END
worker.WRAPPER = "/tmp/hero-decay-wrapper/src/examples/olmo_ddp/olmoe3_hero_lc_eval.py"


def eval_specs(beaker, run, commit):
    specs = worker.eval_specs(beaker, run, commit)
    for stage, spec in specs.items():
        spec["description"] = json.dumps(
            dict(
                campaign=CAMPAIGN,
                stage=stage,
                **run.as_dict(),
                inference_profile="bf16-grouped-fla-pilot-v1",
            )
        )
    return specs


def ruler_spec(beaker, run, commit):
    """Reuse the pinned ordinary RULER recipe and its 4K-through-128K generation smoke."""
    import olmoe3_hero_ruler_control as control
    from olmoe3_hero_lc_ruler import model_path

    original = control.model_path
    try:
        control.model_path = model_path
        spec = control.worker_spec(
            beaker.experiment.get_spec(beaker.workload.get(control.WORKER_TEMPLATE)).to_json(),
            "lc100b",
            run.arm,
            commit,
        )
    finally:
        control.model_path = original
    task = spec["tasks"][0]
    old = "src/examples/olmo_ddp/olmoe3_hero_ruler.py"
    assert task["arguments"][0].count(old) == 1
    task["arguments"][0] = task["arguments"][0].replace(
        old, "src/examples/olmo_ddp/olmoe3_hero_lc_ruler.py"
    )
    spec["description"] = json.dumps(dict(stage="ruler", campaign=CAMPAIGN, **run.as_dict()))
    return spec


def advance_ruler(beaker, control, run, spec):
    """One RULER job per endpoint, after conversion and inference qualification succeed."""
    from olmoe3_hero_lc_ruler import model_path, verify_success

    worker.qualify_source(run)
    old_workspace = control.workspace
    try:
        control.workspace = beaker.workspace.get(worker.WORKSPACE)
        w = control.ensure(run.run_id + "-ruler", spec)
        state = control.report(w)
        if state == "STATUS_SUCCEEDED":
            verify_success(model_path("lc100b", run.arm))
        return dict(
            status=state,
            experiment=w.experiment.id if w else None,
            complete=state == "STATUS_SUCCEEDED",
        )
    finally:
        control.workspace = old_workspace


def prepare_scratch():
    assert MOUNT.is_mount() and EVAL_ROOT.resolve() == EVAL_ROOT
    owner = dict(campaign=CAMPAIGN, stage="long-context", scratch=str(EVAL_ROOT))
    path = EVAL_ROOT / "_OWNER.json"
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
    run = LCRun(args.arm)
    expected = CORE_REF if args.stage == "convert" else HELPER_REF
    assert (
        subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip()
        == expected
    )
    assert not subprocess.check_output(
        ["git", "-C", str(args.source), "status", "--porcelain"], text=True
    ).strip()
    prepare_scratch()
    root = EVAL_ROOT / run.arm / f"step{END}"
    if args.stage == "convert":
        assert ready(run, END)
        receipt = verified_copy(run.root / f"step{END}", root / "olmo-core", END)
        atomic_json(
            root / "download-success.json",
            dict(
                passed=True,
                source_kind="local_copy",
                source=receipt["source"],
                arm=run.arm,
                step=END,
                tokens=END * BATCH,
                raw_path=str(root / "olmo-core"),
                all_file_hashes_verified=True,
                bytes=receipt["total_bytes"],
            ),
        )
        atomic_json(root / "source-receipt.json", receipt)
        atomic_json(root / "lc-provenance.json", run.as_dict())
        sys.path.insert(0, str(args.source / "src/examples/olmo_ddp"))
        import hero_hf_convert as convert

        import olmo_core.nn.hf.convert_checkpoint as hf_converter

        # The frozen PT wrapper hardcodes 8192. Adapt only export metadata before
        # serialization/hashing/qualification, not weights, kernels or tolerances.
        original_convert = hf_converter.convert_checkpoint_to_hf

        def convert_lc(*positional, **keywords):
            assert keywords["max_sequence_length"] == 8192
            keywords["max_sequence_length"] = 65536
            return original_convert(*positional, **keywords)

        hf_converter.convert_checkpoint_to_hf = convert_lc

        convert.SCRATCH, convert.TARGETS, convert.prepare_scratch = (
            EVAL_ROOT,
            {100: END},
            prepare_scratch,
        )
        sys.argv = [
            convert.__file__,
            "--arm",
            run.arm,
            "--step",
            str(END),
            "--full",
            "--portable-reference",
            "--precise",
        ]
        convert.main()
        config = json.loads((root / "hf/config.json").read_text())
        assert config["max_position_embeddings"] == 65536
    else:
        worker.qualify_source(run)
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
