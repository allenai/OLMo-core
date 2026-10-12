"""Frozen hero conversion/qualification/OLMoBase recipes for the 14T + 200B mt20 endpoint."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from olmoe3_hero_mt_plan import (
    BATCH,
    CAMPAIGN,
    CORE_REF,
    END,
    EVAL_ROOT,
    HELPER_REF,
    MOUNT,
    MTRun,
    ready,
    verified_copy,
)
from olmoe3_lr_sweep_watch import atomic_json


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
    parser.add_argument("--arm", choices=("emo",), required=True)
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
        qualify_source(run)
        sys.path.insert(0, str(args.source / "ladders/olmoe3/workloads"))
        import hero_full_eval as runtime

        runtime.FAST_MODELS = {EVAL_ROOT / run.arm / f"step{END}/hf"}
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
