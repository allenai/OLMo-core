"""Scoped epoch exports using the already-qualified LC HF conversion, no raw deletion."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from olmoe3_hero_sft_plan import BATCH, CAMPAIGN, DATA, MOUNT, find_run
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
    for epoch, step in [(run.epochs, run.total_steps)]:
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
    parser.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args()
    run = find_run(args.run)
    assert not run.smoke and args.step == run.total_steps
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
        result = original(*positional, **keywords)
        from olmoe3_hero_sft_metadata import install_metadata

        # Do this before strict qualification, serialization hashes and atomic publication.
        install_metadata(positional[1], DATA / "train/tokenizer")
        return result

    hf_converter.convert_checkpoint_to_hf = convert_sft
    if args.qualify_only:
        # Recovery is limited to the inspected, fully serialized failed final export.
        # Repeat the original stock validation, then all frozen full reference/cache gates.
        assert run.arm == "emo" and run.lr_label == "1em4" and args.step == 1810
        assert not (root / "hf").exists()
        diagnostic = json.loads((root / "conversion-diagnostic-r1.json").read_text())
        assert len(diagnostic) == 5
        assert all(r["logits"]["max_abs"] == 0 for r in diagnostic)
        import gc

        import torch
        from hero_hf_reference_ops import install

        from olmo_core.config import DType
        from olmo_core.nn.hf.config import _register_olmo3moe_auto_classes
        from olmo_core.nn.transformer.config import TransformerConfig

        install()
        _register_olmo3moe_auto_classes()
        import os

        os.environ["OLMO_HF_MOE_CORE_REFERENCE"] = "1"
        saved = hf_converter.load_config(root / "olmo-core")
        model = TransformerConfig.from_dict(convert.reference_config(saved["model"])).build(
            init_device="meta"
        )
        model.to_empty(device="cpu")
        hf_converter._load_ddp_optimizer_model_state(
            root / "olmo-core/model_and_optim",
            model,
            work_dir=str(root / "recovery-load"),
            return_state_dict=False,
        )
        from unittest.mock import patch

        original_randint = torch.randint

        def recorded_randint(*positional, **keywords):
            result = original_randint(*positional, **keywords)
            atomic_json(
                root / "stock-revalidation-inputs.json", {"input_ids": result.cpu().tolist()}
            )
            return result

        with patch.object(torch, "randint", recorded_randint):
            hf_converter.validate_conversion(
                root / "hf.partial",
                model,
                saved["dataset"]["tokenizer"]["vocab_size"],
                dtype=DType.bfloat16,
                device=torch.device("cuda"),
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print("SFT_STOCK_CONVERSION_REVALIDATION_PASSED", flush=True)
        from olmoe3_hero_sft_metadata import install_metadata

        install_metadata(root / "hf.partial", DATA / "train/tokenizer")
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
    if args.qualify_only:
        sys.argv.append("--qualify-only")
    convert.main()
    config = json.loads((root / "hf/config.json").read_text())
    assert config["max_position_embeddings"] == 65536
    from olmoe3_hero_sft_metadata import check_tokenizer
    from transformers import AutoTokenizer

    check_tokenizer(
        AutoTokenizer.from_pretrained(root / "hf", local_files_only=True),
        AutoTokenizer.from_pretrained(DATA / "train/tokenizer", local_files_only=True),
    )


if __name__ == "__main__":
    main()
