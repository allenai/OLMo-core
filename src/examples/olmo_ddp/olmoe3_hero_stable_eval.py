"""Scoped adapters for stable PT120000; numerical workers/recipes remain frozen."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import olmoe3_hero_decay_eval as decay_eval
from olmoe3_hero_decay_plan import (
    CORE_REF,
    END,
    HELPER_REF,
    inventory,
    ready,
    validate_checkpoint,
    verified_copy,
)
from olmoe3_lr_sweep_watch import atomic_json
from olmoe3_small_hero_plan import BATCH, CONTROL, MOUNT, Run

CAMPAIGN = "olmo35-small-stable2t-20260911"
BRANCH = "codex/small-hero-stable2t-evals-20260911"
ROOT = MOUNT / "scratch" / CAMPAIGN
AUTOMATION = MOUNT / "uploader/automation" / CAMPAIGN
ARMS = ("emo", "non-emo")
STAGES = tuple(decay_eval.TEMPLATES)
WORKSPACE = decay_eval.WORKSPACE
WRAPPER = "/tmp/hero-decay-wrapper/src/examples/olmo_ddp/olmoe3_hero_stable_eval.py"


def parent(arm):
    """Resolve only the two approved stable hero lineages."""
    if arm not in ARMS:
        raise ValueError(arm)
    return Run(arm == "emo")


def output_root(arm):
    """Return an isolated stable-eval destination, never a decay directory."""
    parent(arm)
    return ROOT / arm / f"step{END}"


def prepare_scratch():
    """Require the actual mount and an exact ownership marker."""
    assert MOUNT.is_mount() and ROOT.resolve() == ROOT
    ROOT.mkdir(parents=True, exist_ok=True)
    owner = dict(campaign=CAMPAIGN, scratch=str(ROOT), stage="stable PT", step=END)
    path = ROOT / "_OWNER.json"
    if path.exists():
        assert json.loads(path.read_text()) == owner
    else:
        atomic_json(path, owner)


def staged(arm):
    """Verify the durable copy receipt even after the original has been cleaned up."""
    run, root = parent(arm), output_root(arm)
    path = root / "olmo-core-copy.json"
    if not path.is_file():
        return False
    receipt = json.loads(path.read_text())
    assert receipt["source"] == str(run.root / f"step{END}")
    assert receipt["destination"] == str(root / "olmo-core")
    assert receipt["step"] == END and receipt["all_file_hashes_verified"] is True
    validate_checkpoint(root / "olmo-core", END)
    assert {k: v[0] for k, v in inventory(root / "olmo-core").items()} == receipt["sizes"]
    event = json.loads((root / "parent-ready.json").read_text())
    assert event["run_id"] == event["lineage_id"] == run.run_id
    assert event["step"] == END and event["checkpoint_path"] == receipt["source"]
    assert (
        event["checkpoint_metadata_sha256"]
        == hashlib.sha256((root / "olmo-core/.metadata.json").read_bytes()).hexdigest()
    )
    return True


def stage_source(arm):
    """Capture the completed checkpoint immediately; never alter parent retention/data."""
    if staged(arm):
        return True
    run, root = parent(arm), output_root(arm)
    if not ready(run, END):
        return False
    prepare_scratch()
    event = CONTROL / "inbox" / run.run_id / f"step-{END:012d}.ready.json"
    atomic_json(root / "parent-ready.json", json.loads(event.read_text()))
    verified_copy(run.root / f"step{END}", root / "olmo-core", END)
    return staged(arm)


def build_specs(beaker, arm, commit):
    """Reuse exact decay workers, changing only stable model paths and entrypoint."""
    run = parent(arm)
    specs = decay_eval.eval_specs(beaker, run, commit)
    for stage, spec in specs.items():
        task = spec["tasks"][0]
        command = task["arguments"][0]
        assert str(decay_eval.EVAL_ROOT) in command or decay_eval.WRAPPER in command
        command = command.replace(str(decay_eval.EVAL_ROOT), str(ROOT))
        command = command.replace(decay_eval.WRAPPER, WRAPPER)
        assert str(decay_eval.EVAL_ROOT) not in command
        assert "hero_hf_cleanup.py" not in command
        task["arguments"] = [command]
        spec["description"] = json.dumps(
            dict(
                campaign=CAMPAIGN,
                arm=arm,
                parent_lineage=run.run_id,
                stage=stage,
                step=END,
                tokens=END * BATCH,
                training="stable pretraining, no decay",
                model=str(output_root(arm) / "hf"),
                core_ref=CORE_REF,
                helper_ref=HELPER_REF,
                inference_profile="bf16-grouped-fla-pilot-v1",
                note="Same frozen conversion, qualification and provisional fast eval profile; no new numerical certification.",
            )
        )
    return specs


def converted(arm):
    """Require the completed conversion for this exact arm, step and output."""
    root = output_root(arm)
    result = json.loads((root / "conversion-success.json").read_text())
    assert result["passed"] is True
    assert (result["arm"], result["step"], result["tokens"]) == (arm, END, END * BATCH)
    assert result["output"] == str(root / "hf")
    assert result == json.loads((root / "hf/_HERO_CONVERSION_SUCCESS.json").read_text())
    return result


def qualified(arm):
    """Require the existing precise source qualification before provisional fast evals."""
    converted(arm)
    root = output_root(arm)
    sha = hashlib.sha256((root / "hf/_HERO_CONVERSION_SUCCESS.json").read_bytes()).hexdigest()
    for name in ("vllm-parity-success.json", "eval-smoke-success.json"):
        row = json.loads((root / name).read_text())
        assert row.get("passed") is True and not row.get("diagnostic_only")
        if name == "vllm-parity-success.json":
            assert row["precise"] and row["conversion_sha256"] == sha
    return sha


def completed_bundle(arm, bundle):
    """Accept only a complete full-suite receipt tied to the qualified export."""
    assert bundle in ("gen_mc", "math", "code")
    root = output_root(arm)
    markers = list(root.glob(f"pilot-olmobase-{bundle}-*/full-eval-pilot-success.json"))
    assert len(markers) == 1
    row = json.loads(markers[0].read_text())
    assert row["passed"] is True and row["bundle"] == bundle
    path = Path(row["metrics"])
    assert path.resolve().is_relative_to(root)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["metrics_sha256"]
    assert row["source_conversion_sha256"] == qualified(arm)
    assert row["inference_profile"] == "bf16-grouped-fla-pilot-v1"
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("convert", "gen_mc", "math", "code"))
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
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
    root = output_root(args.arm)
    if args.stage == "convert":
        assert staged(args.arm), "Watcher must publish a verified independent copy first"
        receipt = json.loads((root / "olmo-core-copy.json").read_text())
        atomic_json(
            root / "download-success.json",
            dict(
                passed=True,
                source_kind="local_copy",
                source=receipt["source"],
                arm=args.arm,
                step=END,
                tokens=END * BATCH,
                raw_path=str(root / "olmo-core"),
                all_file_hashes_verified=True,
                bytes=receipt["total_bytes"],
            ),
        )
        atomic_json(root / "source-receipt.json", receipt)
        sys.path.insert(0, str(args.source / "src/examples/olmo_ddp"))
        import hero_hf_convert as convert

        convert.SCRATCH, convert.TARGETS, convert.prepare_scratch = (
            ROOT,
            {2013: END},
            prepare_scratch,
        )
        sys.argv = [
            convert.__file__,
            "--arm",
            args.arm,
            "--step",
            str(END),
            "--full",
            "--portable-reference",
            "--precise",
        ]
        convert.main()
        converted(args.arm)
    else:
        qualified(args.arm)
        sys.path.insert(0, str(args.source / "ladders/olmoe3/workloads"))
        import hero_full_eval as runtime

        runtime.FAST_MODELS = {output_root(arm) / "hf" for arm in ARMS}
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
