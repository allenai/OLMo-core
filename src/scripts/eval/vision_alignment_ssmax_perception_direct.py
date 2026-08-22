"""Produce one manifest-bound direct SSMax perception evaluation receipt.

Run this command with the exact distributed topology in the finalized direct manifest.  The
manifest selects the sole run, while ``--step`` selects one of its fixed checkpoints.  No arm or
checkpoint override is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from olmo_core.data.multimodal import MultimodalCollator
from olmo_core.eval import vision_alignment_ssmax_perception as paired
from olmo_core.eval.vision_alignment_ssmax_perception_direct import (
    DIRECT_TEXT_SENTINEL_PROTOCOL,
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
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.train import prepare_training_environment, teardown_training_environment
from scripts.eval import vision_alignment_ssmax_bridge as bridge_runner
from scripts.eval import vision_alignment_ssmax_perception as paired_runner


def _local_cpu_snapshot(tensor: torch.Tensor) -> torch.Tensor:
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return local.detach().cpu().contiguous().clone()


def _tensor_descriptor(tensor: torch.Tensor, *, include_finite: bool) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "sha256": hashlib.sha256(bridge_runner._tensor_bytes(tensor)).hexdigest(),
    }
    if include_finite:
        descriptor["finite"] = bool(torch.isfinite(tensor).all().item())
    return descriptor


def _load_text_sentinel_inputs(
    train_module: Any, path: Path, expected_sha256: str
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct native text sentinel differs from its manifest pin"
        )
    try:
        payload = paired._validate_text_sentinel(path)
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    input_ids = torch.tensor([payload["input_ids"]], dtype=torch.long, device=train_module.device)
    labels = torch.tensor([payload["labels"]], dtype=torch.long, device=train_module.device)
    if input_ids.shape != (1, 256) or labels.shape != (1, 256):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct native text sentinel tensor geometry differs"
        )
    token_count = int((labels != -100).sum().item())
    if token_count != 256:
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct native text sentinel supervised-token count differs"
        )
    invariants = {
        "artifact_sha256": expected_sha256,
        "input": _tensor_descriptor(input_ids, include_finite=False),
        "labels": _tensor_descriptor(labels, include_finite=False),
        "token_count": token_count,
    }
    return input_ids, labels, invariants


def _snapshot_text_sentinel(
    train_module: Any, input_ids: torch.Tensor, labels: torch.Tensor
) -> dict[str, torch.Tensor]:
    output = train_module.eval_batch({"input_ids": input_ids, "labels": labels})
    if not isinstance(output, LMOutputWithLoss) or output.logits is None:
        raise RuntimeError("Direct native text sentinel did not return logits and CE")
    return {
        "logits": _local_cpu_snapshot(output.logits),
        "ce": _local_cpu_snapshot(output.ce_loss),
    }


def _text_sentinel_result(
    *,
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    invariants: Mapping[str, Any],
    reference_checkpoint: Mapping[str, Any],
    candidate_checkpoint: Mapping[str, Any],
    candidate_step: int,
    topology: Mapping[str, Any],
) -> dict[str, Any]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size != topology["world_size"]:
        raise RuntimeError("Direct native text sentinel world size differs from topology")
    if set(invariants) != {"artifact_sha256", "input", "labels", "token_count"}:
        raise RuntimeError("Direct native text sentinel invariants are malformed")
    reference_descriptors = {
        name: _tensor_descriptor(reference[name], include_finite=True) for name in ("logits", "ce")
    }
    candidate_descriptors = {
        name: _tensor_descriptor(candidate[name], include_finite=True) for name in ("logits", "ce")
    }
    descriptor_equality = {
        name: reference_descriptors[name] == candidate_descriptors[name]
        for name in ("logits", "ce")
    }
    logits_exact = bool(
        torch.equal(reference["logits"], candidate["logits"]) and descriptor_equality["logits"]
    )
    ce_exact = bool(torch.equal(reference["ce"], candidate["ce"]) and descriptor_equality["ce"])
    finite = all(
        descriptor["finite"]
        for descriptors in (reference_descriptors, candidate_descriptors)
        for descriptor in descriptors.values()
    )
    row = {
        "rank": rank,
        "reference": reference_descriptors,
        "candidate": candidate_descriptors,
        "logits_exact": logits_exact,
        "ce_exact": ce_exact,
        "passed": bool(logits_exact and ce_exact and finite),
    }
    fixed_invariants = {
        "protocol": DIRECT_TEXT_SENTINEL_PROTOCOL,
        "version": 1,
        **dict(invariants),
        "reference_step": 0,
        "reference_checkpoint_identity_sha256": reference_checkpoint["identity_sha256"],
        "candidate_step": candidate_step,
        "candidate_checkpoint_identity_sha256": candidate_checkpoint["identity_sha256"],
        "topology": dict(topology),
        "world_size": world_size,
        "rank_count": world_size,
    }
    packet = {"invariants": fixed_invariants, "row": row}
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, packet)
    rank_rows: list[dict[str, Any]] = []
    for expected_rank, gathered_packet in enumerate(gathered):
        if not isinstance(gathered_packet, Mapping):
            raise TypeError(f"Direct native text sentinel rank {expected_rank} packet is malformed")
        if gathered_packet.get("invariants") != fixed_invariants:
            raise RuntimeError(
                f"Direct native text sentinel rank {expected_rank} invariants differ"
            )
        gathered_row = gathered_packet.get("row")
        if not isinstance(gathered_row, Mapping) or gathered_row.get("rank") != expected_rank:
            raise RuntimeError("Direct native text sentinel rank rows are out of order")
        rank_rows.append(dict(gathered_row))
    mismatch_count = sum(not rank_row["passed"] for rank_row in rank_rows)
    result: dict[str, Any] = {
        **fixed_invariants,
        "rank_rows": rank_rows,
        "mismatch_count": mismatch_count,
        "all_ranks_passed": mismatch_count == 0,
        "rank_inventory_sha256": canonical_sha256(rank_rows),
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


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
        sentinel_path = paired.validate_artifact_reference(
            manifest["text_sentinel"], name="native text sentinel"
        )
        text_input_ids, text_labels, text_invariants = _load_text_sentinel_inputs(
            train_module, sentinel_path, manifest["text_sentinel"]["sha256"]
        )
        reference_load = bridge_runner._strict_load(
            train_module,
            Path(str(reference["path"])),
            threads=args.checkpoint_load_threads,
        )
        reference_state = paired_runner._state_descriptors(model)
        reference_text = _snapshot_text_sentinel(train_module, text_input_ids, text_labels)
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
        candidate_text = _snapshot_text_sentinel(train_module, text_input_ids, text_labels)
        text_result = _text_sentinel_result(
            reference=reference_text,
            candidate=candidate_text,
            invariants=text_invariants,
            reference_checkpoint=reference,
            candidate_checkpoint=candidate,
            candidate_step=args.step,
            topology=manifest["topology"],
        )
        del reference_text, candidate_text
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
        status_ok = (
            state["frozen_lm"]["mismatch_count"] == 0
            and state["non_image_embedding_rows"]["mismatch_count"] == 0
            and text_result["all_ranks_passed"]
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
