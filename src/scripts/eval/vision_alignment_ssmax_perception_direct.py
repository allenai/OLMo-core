"""Produce one manifest-bound direct SSMax perception evaluation receipt.

Run this command with the exact distributed topology in the finalized direct manifest.  The
manifest selects the sole run, while ``--step`` selects one of its fixed checkpoints.  No arm or
checkpoint override is accepted.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olmo_core.data.multimodal import MultimodalCollator
from olmo_core.eval import vision_alignment_ssmax_perception as paired
from olmo_core.eval.vision_alignment_ssmax_perception_direct import (
    EVALUATION_PRODUCER,
    EVALUATION_RECEIPT_FORMAT,
    REQUIRED_STEPS,
    SOURCES,
    TRAINING_ROLE,
    SSMaxPerceptionDirectEvidenceError,
    canonical_sha256,
    load_json,
    load_manifest,
    manifest_reference,
    sha256_file,
    validate_manifest_producer_source,
)
from olmo_core.train import prepare_training_environment, teardown_training_environment
from scripts.eval import vision_alignment_ssmax_bridge as bridge_runner
from scripts.eval import vision_alignment_ssmax_perception as paired_runner


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--step", type=int, choices=REQUIRED_STEPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-load-threads", type=int, default=8)
    parser.add_argument("--checkpoint-hash-workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.checkpoint_load_threads <= 0 or args.checkpoint_hash_workers <= 0:
        raise ValueError("Checkpoint thread/worker counts must be positive")
    manifest_path = args.manifest.expanduser().resolve()
    if sha256_file(manifest_path) != args.expected_manifest_sha256:
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct manifest differs from its explicit CLI pin"
        )
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable receipt {output_path}")
    manifest = load_manifest(manifest_path, verify_live=False)
    evaluator_source = validate_manifest_producer_source(
        manifest,
        producer=EVALUATION_PRODUCER,
        source_path=Path(__file__),
    )
    expected_world = int(manifest["topology"]["world_size"])
    if int(os.environ.get("WORLD_SIZE", "1")) != expected_world:
        raise ValueError(f"Manifest requires WORLD_SIZE={expected_world}")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepare_training_environment()
    try:
        run = manifest["run"]
        reference = bridge_runner._validate_checkpoint_distributed(
            run["checkpoints"]["0"],
            expected_step=0,
            workers=args.checkpoint_hash_workers,
        )
        candidate = (
            reference
            if args.step == 0
            else bridge_runner._validate_checkpoint_distributed(
                run["checkpoints"][str(args.step)],
                expected_step=args.step,
                workers=args.checkpoint_hash_workers,
            )
        )
        raw_config = load_json(Path(str(candidate["path"])) / "config.json")
        if not isinstance(raw_config, Mapping) or (
            raw_config.get("model_variant") != manifest["model_variant"]
            or raw_config.get("phase") != "perception"
            or raw_config.get("perception_trainability_arm") != TRAINING_ROLE
            or raw_config.get("required_run_name") != run["run_name"]
            or raw_config.get("data", {}).get("pack_sequences") is not False
        ):
            raise SSMaxPerceptionDirectEvidenceError(
                "Selected checkpoint violates the direct run identity"
            )
        rank_batch = int(manifest["evaluation"]["rank_batch_instances"])
        model, train_module = bridge_runner._build_model_and_module(
            raw_config, rank_batch_instances=rank_batch
        )
        reference_load = bridge_runner._strict_load(
            train_module,
            Path(str(reference["path"])),
            threads=args.checkpoint_load_threads,
        )
        reference_state = paired_runner._state_descriptors(model)
        if args.step == 0:
            candidate_load = reference_load
            candidate_state = reference_state
        else:
            candidate_load = bridge_runner._strict_load(
                train_module,
                Path(str(candidate["path"])),
                threads=args.checkpoint_load_threads,
            )
            candidate_state = paired_runner._state_descriptors(model)
        state = paired_runner._state_receipt(reference_state, candidate_state)

        tokenizer, datasets, content_ids = paired_runner._load_perception_datasets(
            raw_config, manifest
        )
        pairings = {
            source: load_json(
                paired.validate_artifact_reference(
                    manifest["pairings"][source], name=f"{source} pairing"
                )
            )
            for source in SOURCES
        }
        collator = MultimodalCollator(
            pad_token_id=int(tokenizer.pad_token_id),
            label_ignore_index=-100,
            pad_sequence_length=int(raw_config["data"]["sequence_length"]),
        )
        results = {
            source: paired_runner._evaluate_source(
                train_module,
                datasets[source],
                pairings[source],
                source=source,
                pairing_sha256=manifest["pairings"][source]["sha256"],
                collator=collator,
                work_dir=args.work_dir.expanduser().resolve(),
                rank_batch_instances=rank_batch,
            )
            for source in SOURCES
        }
        for result in results.values():
            result.pop("elapsed_seconds")
        attention_probe_path = paired.validate_artifact_reference(
            manifest["attention_probe"], name="SSMax attention probe"
        )
        attention_diagnostics = bridge_runner._run_attention_probe(
            train_module,
            datasets["pixmo_caption"],
            content_ids=content_ids["pixmo_caption"],
            probe_path=attention_probe_path,
            probe_sha256=manifest["attention_probe"]["sha256"],
            collator=collator,
            checkpoint_identity=candidate,
        )
        sentinel_path = paired.validate_artifact_reference(
            manifest["text_sentinel"], name="native text sentinel"
        )
        text_result = paired_runner._text_sentinel(
            train_module, sentinel_path, manifest["text_sentinel"]["sha256"]
        )
        status_ok = (
            state["frozen_lm"]["mismatch_count"] == 0
            and state["non_image_embedding_rows"]["mismatch_count"] == 0
        )
        payload: dict[str, Any] = {
            "format": EVALUATION_RECEIPT_FORMAT,
            "version": manifest["version"],
            "status": "passed" if status_ok else "failed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest": manifest_reference(manifest_path, manifest),
            "run_id": manifest["run_id"],
            "model_variant": manifest["model_variant"],
            "step": args.step,
            "checkpoint": dict(candidate),
            "strict_generic_dcp_load": candidate_load,
            "state": state,
            "text_sentinel": text_result,
            "attention_diagnostics": attention_diagnostics,
            "pairings": {source: dict(manifest["pairings"][source]) for source in SOURCES},
            "results": results,
            "evaluator": evaluator_source,
        }
        payload["content_sha256"] = canonical_sha256(payload)
        bridge_runner._write_distributed(output_path, payload)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
