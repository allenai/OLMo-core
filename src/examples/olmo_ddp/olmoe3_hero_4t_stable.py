"""Two exact pretraining checkpoints per arm, independent of decay/posttraining."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import olmoe3_hero_decay_eval as worker
from olmoe3_hero_decay_plan import (
    AUTOMATION as DECAY_AUTOMATION,
    CORE_REF,
    HELPER_REF,
    MOUNT,
    validate_checkpoint,
    verified_copy,
)
from olmoe3_lr_sweep_watch import atomic_json
from olmoe3_small_hero_plan import Run, BATCH

ROOT = MOUNT / "scratch/olmo35-small-4t-stable-20260916"
STEPS = (216000, 240000)
FILE = "/tmp/hero-decay-wrapper/src/examples/olmo_ddp/olmoe3_hero_4t_stable.py"


def setup(step):
    assert step in STEPS
    worker.END, worker.EVAL_ROOT, worker.WRAPPER = step, ROOT, FILE


def specs(b, arm, step, commit):
    setup(step)
    result = worker.eval_specs(b, Run(arm == "emo"), commit)
    for stage, s in result.items():
        t = s["tasks"][0]
        old = f"python {FILE} {stage} --arm {arm}"
        assert t["arguments"][0].count(old) == 1
        t["arguments"][0] = t["arguments"][0].replace(old, old + f" --step {step}")
        s["description"] = json.dumps(
            dict(
                arm=arm,
                step=step,
                tokens=step * BATCH,
                stage=stage,
                schedule="WSD stable, no decay",
                model=str(ROOT / arm / f"step{step}/hf"),
            )
        )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("convert", "gen_mc", "math", "code"))
    p.add_argument("--arm", choices=("emo", "non-emo"), required=True)
    p.add_argument("--step", type=int, choices=STEPS, required=True)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    setup(args.step)
    assert MOUNT.is_mount()
    assert subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
    ).strip() == (CORE_REF if args.stage == "convert" else HELPER_REF)
    assert not subprocess.check_output(
        ["git", "-C", str(args.source), "status", "--porcelain"], text=True
    ).strip()
    ROOT.mkdir(parents=True, exist_ok=True)
    root = ROOT / args.arm / f"step{args.step}"
    if args.stage == "convert":
        # The fork downloads are immutable independent copies. Reuse them when
        # present; no second network transfer or dependence on live PT retention.
        fork = DECAY_AUTOMATION / "sources" / args.arm / f"step{args.step}/olmo-core"
        if fork.is_dir():
            receipt = verified_copy(fork, root / "olmo-core", args.step)
            atomic_json(
                root / "download-success.json",
                dict(
                    passed=True,
                    arm=args.arm,
                    step=args.step,
                    tokens=args.step * BATCH,
                    raw_path=str(root / "olmo-core"),
                    source=str(fork),
                    source_kind="verified_local_fork",
                    all_file_hashes_verified=True,
                    bytes=receipt["total_bytes"],
                ),
            )
        else:
            from huggingface_hub import HfApi
            import olmoe3_hero_bucket_download as download

            download.SCRATCH = ROOT
            download.download(HfApi(), args.arm, args.step)
        validate_checkpoint(root / "olmo-core", args.step)
        sys.path.insert(0, str(args.source / "src/examples/olmo_ddp"))
        import hero_hf_convert as convert
        from olmoe3_hero_4t_eval_policy import install_conversion

        convert.SCRATCH, convert.TARGETS, convert.prepare_scratch = (
            ROOT,
            {4: args.step},
            lambda: None,
        )
        install_conversion(convert)
        sys.argv = [
            convert.__file__,
            "--arm",
            args.arm,
            "--step",
            str(args.step),
            "--full",
            "--portable-reference",
        ]
        convert.main()
    else:
        sys.path.insert(0, str(args.source / "ladders/olmoe3/workloads"))
        import hero_full_eval as runtime
        from olmoe3_hero_4t_eval_policy import install_runtime

        runtime.FAST_MODELS = {ROOT / a / f"step{s}/hf" for a in ("emo", "non-emo") for s in STEPS}
        install_runtime(runtime)
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
