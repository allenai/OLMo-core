"""Produce one distributed, manifest-bound SSMax joint evaluation receipt.

The producer strictly loads a generic dense checkpoint, compares the two intentionally frozen
lexical surfaces with joint step 0, evaluates fixed correct/exact-geometry-wrong rows for every
visual source, computes CE/PPL on the fixed native-text holdout prefix, and runs the shared SSMax
collector on a joint-projection-specific fixed attention probe.  It does not duplicate the
separately pinned BLINK/MathVista evaluator.
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from olmo_core.data.multimodal import MultimodalCollator
from olmo_core.data.multimodal.native_text_replay import (
    NativeTextReplayDatasetConfig,
    NativeTextReplayVerificationReceipt,
)
from olmo_core.data.multimodal.ssmax_single_response import SSMaxSingleResponseDataset
from olmo_core.data.multimodal.vision_alignment_joint_provenance import (
    build_selected_joint_dataset,
    load_joint_visual_projection_manifest,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    load_pinned_vision_alignment_tokenizer,
    serialized_example_sha256,
)
from olmo_core.eval.vision_alignment_ssmax_joint import (
    EVALUATION_RECEIPT_FORMAT,
    IMAGE_TOKEN_ROWS,
    REQUIRED_STEPS,
    SCHEMA_VERSION,
    VISUAL_SOURCES,
    SSMaxJointEvidenceError,
    canonical_sha256,
    evaluator_source_reference,
    load_json,
    load_manifest,
    manifest_reference,
    sha256_file,
    validate_artifact_reference,
)
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.vision import MultimodalLM
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.utils import gc_cuda, move_to_device
from scripts.eval import vision_alignment_ssmax_bridge as bridge_runner
from scripts.eval import vision_alignment_ssmax_perception as perception_runner


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


def _single_response_branch(example: Mapping[str, Any], *, name: str) -> None:
    labels = torch.as_tensor(example["labels"])
    weights = torch.as_tensor(example["loss_masks"])
    if labels.ndim != 1 or weights.shape != labels.shape:
        raise ValueError(f"{name} labels/loss masks must be aligned one-dimensional rows")
    if not bool((weights > 0).any()):
        raise ValueError(f"{name} has no supervised response branch")


class _UnpackedModelInputDataset:
    """Project a joint row onto the model contract and reject packed metadata."""

    required = frozenset(
        {
            "input_ids",
            "labels",
            "loss_masks",
            "position_ids",
            "token_type_ids",
            "images",
            "pooled_patches_idx",
        }
    )

    def __init__(self, dataset: Any, *, source: str):
        self.dataset = dataset
        self.source = source

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0) -> dict[str, Any]:
        getter = getattr(self.dataset, "get", None)
        row = getter(index, epoch) if callable(getter) else self.dataset[index]
        if not isinstance(row, Mapping):
            raise TypeError(f"{self.source} row{index} is not an object")
        missing = self.required - set(row)
        unknown = set(row) - self.required - {"metadata"}
        if missing or unknown:
            raise ValueError(
                f"{self.source} row{index} fields differ: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if "subsegment_ids" in row or "example_ids" in row:
            raise ValueError(f"{self.source} row{index} contains packed sequence metadata")
        _single_response_branch(row, name=f"{self.source} row{index}")
        return {field: row[field] for field in sorted(self.required)}


def _inventory(descriptors: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(descriptors[name]) for name in sorted(descriptors)]
    return {
        "protocol": "logical-model-tensor-inventory-sha256-v1",
        "tensor_count": len(rows),
        "inventory_sha256": canonical_sha256(rows),
    }


def _state_descriptors(model: MultimodalLM) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    all_tensors = {}
    for name, tensor in sorted(parameters.items()):
        all_tensors[name] = bridge_runner._logical_tensor_descriptor(name, "parameter", tensor)
    for name, tensor in sorted(buffers.items()):
        key = f"buffer:{name}"
        all_tensors[key] = bridge_runner._logical_tensor_descriptor(key, "buffer", tensor)
    embedding = parameters.get("lm.embeddings.weight")
    output = parameters.get("lm.lm_head.w_out.weight")
    if embedding is None or output is None:
        raise RuntimeError("Could not locate frozen SSMax lexical surfaces")
    lexical = bridge_runner._replicated_row_descriptor(
        "lm.embeddings.weight[lexical_rows]",
        "frozen_lexical_input_rows",
        embedding,
        IMAGE_TOKEN_ROWS,
        invert=True,
    )
    output_descriptor = bridge_runner._logical_tensor_descriptor(
        "lm.lm_head.w_out.weight", "parameter", output
    )
    return {
        "full_model": _inventory(all_tensors),
        "frozen_lexical_input_rows": {lexical["name"]: lexical},
        "frozen_output_projection": {output_descriptor["name"]: output_descriptor},
    }


def _state_receipt(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "full_model": dict(candidate["full_model"]),
        "frozen_lexical_input_rows": perception_runner._compare_surface(
            reference["frozen_lexical_input_rows"], candidate["frozen_lexical_input_rows"]
        ),
        "frozen_output_projection": perception_runner._compare_surface(
            reference["frozen_output_projection"], candidate["frozen_output_projection"]
        ),
    }


def _native_config(raw: Mapping[str, Any], *, verify_source_hashes: bool) -> Any:
    values = {key: value for key, value in raw.items() if key != "_CLASS_"}
    values["verify_source_hashes"] = verify_source_hashes
    return NativeTextReplayDatasetConfig(**values)


def _native_identity(dataset: Any, *, examples: int) -> dict[str, Any]:
    row_hashes = [serialized_example_sha256(dataset.get(index, 0)) for index in range(examples)]
    provenance = [dict(dataset.provenance_for(index)) for index in range(examples)]
    return {
        "examples": examples,
        "manifest_path": str(dataset.manifest.path),
        "manifest_sha256": sha256_file(dataset.manifest.path),
        "fingerprint": dataset.manifest.content_fingerprint,
        "dataset_order_sha256": canonical_sha256(list(range(examples))),
        "row_provenance_sha256": canonical_sha256(provenance),
        "serialized_rows_sha256": canonical_sha256(row_hashes),
    }


def _load_native_holdout(
    raw_config: Mapping[str, Any], tokenizer: Any, *, examples: int
) -> tuple[Any, dict[str, Any]]:
    packet: list[Any] = [None]
    rank_zero_dataset = None
    if dist.get_rank() == 0:
        try:
            train_raw = raw_config["data"]["native_text_replay"]
            holdout_raw = raw_config["evaluation"]["native_text_holdout"]
            train = _native_config(train_raw, verify_source_hashes=True).build(tokenizer)
            rank_zero_dataset = _native_config(holdout_raw, verify_source_hashes=True).build(
                tokenizer
            )
            receipt = NativeTextReplayVerificationReceipt.load(
                Path(str(holdout_raw["verification_receipt_path"])),
                expected_sha256=holdout_raw["expected_verification_receipt_sha256"],
            )
            receipt.validate_pair(train.manifest, rank_zero_dataset.manifest)
            if examples > len(rank_zero_dataset):
                raise ValueError("manifest requests more native rows than the holdout contains")
            identity = _native_identity(rank_zero_dataset, examples=examples)
            packet[0] = {"ok": True, "identity": identity}
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"native holdout validation failed: {detail}")
    if rank_zero_dataset is None:
        rank_zero_dataset = _native_config(
            raw_config["evaluation"]["native_text_holdout"], verify_source_hashes=False
        ).build(tokenizer)
    local_identity = _native_identity(rank_zero_dataset, examples=examples)
    if local_identity != result["identity"]:
        raise RuntimeError("rank-local native holdout identity differs")
    return rank_zero_dataset, dict(local_identity)


def _loss_row(train_module: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    labels = batch["labels"]
    weights = batch["loss_masks"].float()
    if labels.shape[0] != 1:
        raise ValueError("native evaluation requires one row per rank")
    mask = weights > 0
    labeled = mask & (labels != -100)
    mask_tokens = int(mask.sum())
    tokens = int(labeled.sum())
    mask_weight = float(weights.masked_select(mask).sum())
    labeled_weight = float(weights.masked_select(labeled).sum())
    if mask_tokens <= 0 or tokens not in (0, mask_tokens):
        raise ValueError("native row has partial or empty supervision")
    device_batch = move_to_device(dict(batch), train_module.device)
    output = train_module.eval_batch(device_batch, return_response_logits=False)
    if not isinstance(output, LMOutputWithLoss) or output.logits is not None:
        raise RuntimeError("native evaluation did not return scalar response CE")
    summed_ce = float(output.ce_loss.detach().float().cpu())
    if not math.isfinite(summed_ce) or summed_ce < 0:
        raise RuntimeError("native evaluation returned invalid CE")
    if tokens == 0 and summed_ce != 0:
        raise RuntimeError("filtered native row has non-zero CE")
    del device_batch, output
    return {
        "tokens": tokens,
        "mask_weight": mask_weight,
        "loss_weight": labeled_weight,
        "summed_ce": summed_ce,
        "filtered": tokens == 0,
    }


def _evaluate_native(
    train_module: Any,
    dataset: Any,
    *,
    identity: Mapping[str, Any],
    examples: int,
    collator: MultimodalCollator,
) -> dict[str, Any]:
    group = train_module.dp_process_group
    if group is None:
        raise RuntimeError("joint evaluation requires a DP process group")
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if examples % world:
        raise RuntimeError("native holdout prefix is not divisible by the DP world")
    local = []
    for position in range(rank, examples, world):
        example = dataset.get(position, 0)
        if not isinstance(example, Mapping):
            raise TypeError(f"native row{position} is not an object")
        if "subsegment_ids" in example or "example_ids" in example:
            raise ValueError("native holdout contains packed metadata")
        _single_response_branch(example, name=f"native row{position}")
        batch = collator([dict(example)])
        row = _loss_row(train_module, batch)
        local.append({"position": position, **row})
        gc_cuda()
    gathered: list[Any] = [None for _ in range(world)]
    dist.all_gather_object(gathered, local, group=group)
    rows = [row for rank_rows in gathered for row in rank_rows]
    rows.sort(key=lambda row: row["position"])
    if [row["position"] for row in rows] != list(range(examples)):
        raise RuntimeError("distributed native rows are incomplete or duplicated")
    summed_ce = sum(float(row["summed_ce"]) for row in rows)
    loss_weight = sum(float(row["loss_weight"]) for row in rows)
    if loss_weight <= 0:
        raise RuntimeError("native holdout has no labeled loss mass")
    ce = summed_ce / loss_weight
    return {
        "examples": examples,
        "tokens": sum(int(row["tokens"]) for row in rows),
        "loss_weight": loss_weight,
        "summed_ce": summed_ce,
        "ce": ce,
        "ppl": math.exp(ce),
        "filtered_examples": sum(bool(row["filtered"]) for row in rows),
        "dataset_order_sha256": identity["dataset_order_sha256"],
        "row_provenance_sha256": identity["row_provenance_sha256"],
        "native_identity_sha256": canonical_sha256(identity),
        "per_example": rows,
    }


def _load_visual_datasets(
    raw_config: Mapping[str, Any], manifest: Mapping[str, Any], tokenizer: Any, token_ids: Any
) -> dict[str, Any]:
    projection_ref = manifest["joint_visual_projection"]
    projection = load_joint_visual_projection_manifest(
        projection_ref["path"],
        expected_token_ids=token_ids,
        expected_sha256=projection_ref["sha256"],
        verify_finevision_materialization=False,
        load_image_path_signatures=False,
    )
    if (
        raw_config["data"].get("joint_visual_projection_path") != projection_ref["path"]
        or raw_config["data"].get("joint_visual_projection_sha256") != projection_ref["sha256"]
    ):
        raise SSMaxJointEvidenceError("checkpoint names a different joint projection")
    single_response = raw_config["data"].get("ssmax_single_response_projection")
    if (
        not isinstance(single_response, Mapping)
        or single_response.get("seed") != manifest["evaluation"]["single_response_projection_seed"]
    ):
        raise SSMaxJointEvidenceError(
            "checkpoint and manifest single-response projection seeds differ"
        )
    return {
        source: _UnpackedModelInputDataset(
            SSMaxSingleResponseDataset(
                build_selected_joint_dataset(
                    projection,
                    tokenizer,
                    token_ids,
                    source,
                    logical_split="validation",
                    validate_required_annotations=True,
                ),
                source_name=source,
                logical_split="validation",
                seed=int(single_response["seed"]),
                loss_token_weighting=str(raw_config["data"]["loss_token_weighting"]),
            ),
            source=source,
        )
        for source in VISUAL_SOURCES
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.checkpoint_load_threads <= 0 or args.checkpoint_hash_workers <= 0:
        raise ValueError("checkpoint thread/worker counts must be positive")
    manifest_path = args.manifest.expanduser().resolve()
    if sha256_file(manifest_path) != args.expected_manifest_sha256:
        raise SSMaxJointEvidenceError("joint manifest differs from its CLI pin")
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt {output_path}")
    manifest = load_manifest(manifest_path, verify_live=False)
    if int(os.environ.get("WORLD_SIZE", "1")) != manifest["topology"]["world_size"]:
        raise ValueError("runtime WORLD_SIZE differs from the joint manifest")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepare_training_environment()
    try:
        reference = bridge_runner._validate_checkpoint_distributed(
            manifest["checkpoints"]["0"],
            expected_step=0,
            workers=args.checkpoint_hash_workers,
        )
        candidate = (
            reference
            if args.step == 0
            else bridge_runner._validate_checkpoint_distributed(
                manifest["checkpoints"][str(args.step)],
                expected_step=args.step,
                workers=args.checkpoint_hash_workers,
            )
        )
        raw_config = load_json(Path(str(candidate["path"])) / "config.json")
        if not isinstance(raw_config, Mapping) or (
            raw_config.get("model_variant") != manifest["model_variant"]
            or raw_config.get("phase") != "joint"
            or raw_config.get("required_run_name") != manifest["run_name"]
            or raw_config.get("data", {}).get("pack_sequences") is not False
        ):
            raise SSMaxJointEvidenceError("selected checkpoint violates joint run identity")
        rank_batch = int(manifest["evaluation"]["rank_batch_instances"])
        model, train_module = bridge_runner._build_model_and_module(
            raw_config, rank_batch_instances=rank_batch
        )
        reference_load = bridge_runner._strict_load(
            train_module,
            Path(str(reference["path"])),
            threads=args.checkpoint_load_threads,
        )
        reference_state = _state_descriptors(model)
        if args.step == 0:
            candidate_load = reference_load
            candidate_state = reference_state
        else:
            candidate_load = bridge_runner._strict_load(
                train_module,
                Path(str(candidate["path"])),
                threads=args.checkpoint_load_threads,
            )
            candidate_state = _state_descriptors(model)
        state = _state_receipt(reference_state, candidate_state)
        artifacts = raw_config["artifacts"]
        tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
            identifier=artifacts["tokenizer_id"],
            revision=artifacts["tokenizer_revision"],
            expected_fingerprint=artifacts["tokenizer_fingerprint"],
            cache_dir=artifacts["hf_cache_dir"],
        )
        visual = _load_visual_datasets(raw_config, manifest, tokenizer, token_ids)
        pairings = {
            source: load_json(
                validate_artifact_reference(manifest["pairings"][source], name=f"{source} pairing")
            )
            for source in VISUAL_SOURCES
        }
        collator = MultimodalCollator(
            pad_token_id=int(tokenizer.pad_token_id),
            label_ignore_index=-100,
            pad_sequence_length=8192,
        )
        results = {
            source: perception_runner._evaluate_source(
                train_module,
                visual[source],
                pairings[source],
                source=source,
                pairing_sha256=manifest["pairings"][source]["sha256"],
                collator=collator,
                work_dir=args.work_dir.expanduser().resolve(),
                rank_batch_instances=rank_batch,
            )
            for source in VISUAL_SOURCES
        }
        for result in results.values():
            result.pop("elapsed_seconds")
        attention_probe_path = validate_artifact_reference(
            manifest["attention_probe"], name="SSMax attention probe"
        )
        attention_diagnostics = bridge_runner._run_attention_probe(
            train_module,
            visual["pixmo_caption"],
            probe_path=attention_probe_path,
            probe_sha256=manifest["attention_probe"]["sha256"],
            collator=collator,
            checkpoint_identity=candidate,
        )
        native_dataset, native_identity = _load_native_holdout(
            raw_config,
            tokenizer,
            examples=int(manifest["evaluation"]["native_holdout_examples"]),
        )
        native = _evaluate_native(
            train_module,
            native_dataset,
            identity=native_identity,
            examples=int(manifest["evaluation"]["native_holdout_examples"]),
            collator=collator,
        )
        git = bridge_runner._git_identity()
        if git.get("revision") != manifest["git"]["ref"] or git.get("dirty") is not False:
            raise RuntimeError("evaluator checkout differs from the clean training git ref")
        status_ok = (
            state["frozen_lexical_input_rows"]["mismatch_count"] == 0
            and state["frozen_output_projection"]["mismatch_count"] == 0
        )
        payload: dict[str, Any] = {
            "format": EVALUATION_RECEIPT_FORMAT,
            "version": SCHEMA_VERSION,
            "status": "passed" if status_ok else "failed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest": manifest_reference(manifest_path, manifest),
            "run_id": manifest["run_id"],
            "model_variant": manifest["model_variant"],
            "step": args.step,
            "checkpoint": dict(candidate),
            "strict_generic_dcp_load": candidate_load,
            "state": state,
            "native_holdout": native,
            "pairings": {source: dict(manifest["pairings"][source]) for source in VISUAL_SOURCES},
            "results": results,
            "attention_diagnostics": attention_diagnostics,
            "evaluator": evaluator_source_reference(Path(__file__), git_ref=str(git["revision"])),
        }
        payload["content_sha256"] = canonical_sha256(payload)
        bridge_runner._write_distributed(output_path, payload)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
