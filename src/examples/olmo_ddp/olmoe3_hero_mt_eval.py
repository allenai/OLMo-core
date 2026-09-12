"""Frozen hero conversion/qualification/OLMoBase recipes for independent MT endpoints."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import olmoe3_hero_decay_eval as worker
from olmoe3_hero_decay_plan import CORE_REF, HELPER_REF, ready, verified_copy
from olmoe3_hero_mt_plan import BATCH, CAMPAIGN, END, EVAL_ROOT, MOUNT, MTRun
from olmoe3_lr_sweep_watch import atomic_json

worker.EVAL_ROOT = EVAL_ROOT
worker.END = END
worker.WRAPPER = "/tmp/hero-decay-wrapper/src/examples/olmo_ddp/olmoe3_hero_mt_eval.py"


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


def prepare_scratch():
    assert MOUNT.is_mount() and EVAL_ROOT.resolve() == EVAL_ROOT
    owner = dict(campaign=CAMPAIGN, stage="midtraining", scratch=str(EVAL_ROOT))
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
    run = MTRun(args.arm)
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
        atomic_json(root / "mt-provenance.json", run.as_dict())
        sys.path.insert(0, str(args.source / "src/examples/olmo_ddp"))
        import hero_hf_convert as convert

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
