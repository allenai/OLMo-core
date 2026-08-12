"""Focused contracts for the Vision Alignment promotion evidence boundary."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from olmo_core.eval import vision_alignment_promotion as promotion


def _candidate(tmp_path: Path) -> dict:
    checkpoint = tmp_path / "run" / "step500"
    checkpoint.mkdir(parents=True)
    return {
        "checkpoint": str(checkpoint),
        "global_step": 500,
        "phase": "bridge",
        "lineage_id": "bridge-real-v1",
        "checkpoint_config_sha256": "a" * 64,
        "checkpoint_identity_sha256": "b" * 64,
        "checkpoint_marker_sha256": "c" * 64,
        "dcp_metadata_sha256": "d" * 64,
        "state_file_inventory_sha256": "e" * 64,
        "data_contract_sha256": "f" * 64,
        "trainable_contract_sha256": "1" * 64,
        "vocab_size": 100_352,
        "image_embedding_rows": list(promotion.IMAGE_TOKEN_ROWS),
    }


def _receipt_candidate(candidate: dict) -> dict:
    return {
        "checkpoint": candidate["checkpoint"],
        "global_step": candidate["global_step"],
        "checkpoint_config_sha256": candidate["checkpoint_config_sha256"],
        "checkpoint_identity_sha256": candidate["checkpoint_identity_sha256"],
    }


def _step0(candidate: dict) -> dict:
    checkpoint = Path(candidate["checkpoint"]).parent / "step0"
    checkpoint.mkdir(exist_ok=True)
    return {
        "checkpoint": str(checkpoint),
        "global_step": 0,
        "checkpoint_config_sha256": candidate["checkpoint_config_sha256"],
        "checkpoint_identity_sha256": "3" * 64,
    }


def _state_text_evaluator(tmp_path: Path) -> dict[str, str]:
    live_path = (
        Path(promotion.__file__).resolve().parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_state_text.py"
    )
    return {
        "path": str(tmp_path / "gantry-runtime" / live_path.name),
        "sha256": promotion.sha256_file(live_path),
    }


def test_frozen_state_receipt_requires_complete_bitwise_equality(tmp_path: Path):
    candidate = _candidate(tmp_path)
    comparisons = [
        {
            "name": "vision.blocks.0.weight",
            "kind": "frozen_tensor",
            "dtype": "torch.bfloat16",
            "shape": [2, 2],
            "numel": 4,
            "reference_sha256": "4" * 64,
            "candidate_sha256": "4" * 64,
        },
        {
            "name": "lm.embeddings.weight[non_image_rows]",
            "kind": "non_image_embedding_rows",
            "dtype": "torch.bfloat16",
            "shape": [100_278, 8],
            "numel": 802_224,
            "reference_sha256": "5" * 64,
            "candidate_sha256": "5" * 64,
        },
    ]
    receipt = {
        "format": promotion.FROZEN_STATE_RECEIPT_FORMAT,
        "version": 1,
        "status": "passed",
        "created_at": "2026-08-12T00:00:00+00:00",
        "evaluator": _state_text_evaluator(tmp_path),
        "candidate": _receipt_candidate(candidate),
        "reference_checkpoint": _step0(candidate),
        "protocol": {
            "name": "logical-tensor-sha256-v1",
            "hash_algorithm": "sha256",
            "tensor_encoding": "dtype-shape-contiguous-little-endian-v1",
            "image_embedding_rows": list(promotion.IMAGE_TOKEN_ROWS),
        },
        "comparisons": comparisons,
        "summary": {
            "complete": True,
            "expected_frozen_tensor_count": 1,
            "compared_frozen_tensor_count": 1,
            "non_image_embedding_row_count": 100_346,
            "mismatch_count": 0,
            "comparison_inventory_sha256": promotion.canonical_sha256(
                sorted(comparisons, key=lambda item: (item["kind"], item["name"]))
            ),
        },
    }

    summary = promotion.validate_frozen_state_receipt(receipt, candidate=candidate)
    assert summary["frozen_tensor_count"] == 1

    changed = deepcopy(receipt)
    changed["comparisons"][0]["candidate_sha256"] = "6" * 64
    with pytest.raises(promotion.PromotionValidationError, match="differs"):
        promotion.validate_frozen_state_receipt(changed, candidate=candidate)


def test_text_retention_receipt_requires_image_free_exact_sentinel(tmp_path: Path):
    candidate = _candidate(tmp_path)
    parent_paths = tmp_path / "data_paths.txt"
    parent_paths.write_text("".join(f"s3://bucket/{index}.npy\n" for index in range(128)))
    sentinel_path = tmp_path / "sentinel.json"
    sentinel = {
        "format": promotion.TEXT_SENTINEL_FORMAT,
        "version": 1,
        "parent_checkpoint": str(tmp_path / "bare-parent"),
        "parent_checkpoint_config_sha256": "6" * 64,
        "parent_data_paths": {
            "path": str(parent_paths),
            "sha256": promotion.sha256_file(parent_paths),
            "count": 128,
        },
        "selection": {
            "algorithm": "evenly-spaced-parent-path-first-window-v1",
            "examples": 128,
            "sequence_length": 256,
            "dtype": "uint32-little-endian",
            "source_indices": list(range(128)),
        },
        "rows": [
            {
                "source_index": index,
                "source_path": f"s3://bucket/{index}.npy",
                "start": 0,
                "tokens": list(range(257)),
            }
            for index in range(128)
        ],
        "content_sha256": "",
    }
    sentinel["content_sha256"] = promotion.canonical_sha256(
        {key: value for key, value in sentinel.items() if key != "content_sha256"}
    )
    sentinel_path.write_text(json.dumps(sentinel, sort_keys=True) + "\n")
    sentinel_summary = promotion.validate_text_sentinel(sentinel)
    receipt = {
        "format": promotion.TEXT_RETENTION_RECEIPT_FORMAT,
        "version": 1,
        "status": "passed",
        "created_at": "2026-08-12T00:00:00+00:00",
        "evaluator": _state_text_evaluator(tmp_path),
        "candidate": _receipt_candidate(candidate),
        "reference_checkpoint": _step0(candidate),
        "dataset": {
            "path": str(sentinel_path),
            "sha256": promotion.sha256_file(sentinel_path),
            "fingerprint": sentinel_summary["fingerprint"],
            "examples": 128,
            "supervised_tokens": 32_768,
            "input_ids_sha256": sentinel_summary["input_ids_sha256"],
            "labels_sha256": sentinel_summary["labels_sha256"],
            "image_token_count": 0,
            "image_tensor_count": 0,
        },
        "protocol": {
            "name": "per-token-nll-and-argmax-v1",
            "atol": 1e-6,
            "rtol": 1e-6,
            "same_topology": True,
            "same_backend": True,
            "image_free": True,
        },
        "metrics": {
            "all_finite": True,
            "reference_mean_ce": 2.0,
            "candidate_mean_ce": 2.0,
            "max_abs_token_ce_delta": 0.0,
            "max_rel_token_ce_delta": 0.0,
            "argmax_matches": 32_768,
            "argmax_total": 32_768,
        },
    }
    assert (
        promotion.validate_text_retention_receipt(receipt, candidate=candidate)["argmax_match_rate"]
        == 1.0
    )

    receipt["dataset"]["image_token_count"] = 1
    with pytest.raises(promotion.PromotionValidationError, match="image-free"):
        promotion.validate_text_retention_receipt(receipt, candidate=candidate)


def test_cumulative_loss_mass_requires_all_batches_and_two_point_tolerance(tmp_path: Path):
    candidate = _candidate(tmp_path)
    source_root = Path(promotion.__file__).resolve().parents[2]
    recipe = source_root / "scripts" / "train" / "Vision-Alignment.py"
    producer = source_root / "scripts" / "eval" / "vision_alignment_loss_mass.py"
    dataset_fingerprints = [{"type": "test", "version": "v1", "value": "stable"}]
    loader_state = {
        "batches_processed": 500,
        "epoch": 1,
        "seed": 95818,
        "consecutive_data_errors": 0,
        "total_data_errors": 0,
        "packing_state": {
            "version": 5,
            "dp_rank": 0,
            "dp_world_size": 1,
            "dataset_fingerprints": dataset_fingerprints,
        },
    }
    rank_state = tmp_path / "rank0.pt"
    torch.save({"global_step": 500, "world_size": 1, "data_loader": loader_state}, rank_state)
    rank_inventory = [
        {
            "rank": 0,
            "path": str(rank_state),
            "sha256": promotion.sha256_file(rank_state),
        }
    ]
    checkpoint_state_sha = promotion.canonical_sha256([{"rank": 0, "state": loader_state}])

    def source(share, target):
        return {
            "examples": 100,
            "tokens": 1000,
            "positive_tokens": 500,
            "loss_weight": share * 1000,
            "active_loss_weight": share * 900,
            "loss_mass_share": share,
            "active_loss_mass_share": share,
            "target_loss_mass": target,
            "absolute_error": abs(share - target),
            "active_absolute_error": abs(share - target),
        }

    receipt = {
        "format": promotion.LOSS_MASS_RECEIPT_FORMAT,
        "version": 1,
        "status": "passed",
        "created_at": "2026-08-12T00:00:00+00:00",
        "candidate": _receipt_candidate(candidate),
        "protocol": {
            "name": "exact-packed-loader-cumulative-loss-mass-v1",
            "start_step": 0,
            "end_step": 500,
            "share_tolerance": 0.02,
            "exact_packing_cursor": True,
        },
        "loader": {
            "data_contract_sha256": candidate["data_contract_sha256"],
            "dataset_fingerprints_sha256": promotion.canonical_sha256(dataset_fingerprints),
            "initial_state_sha256": "b" * 64,
            "checkpoint_final_state_sha256": checkpoint_state_sha,
            "replayed_final_state_sha256": checkpoint_state_sha,
            "rank_state_inventory_sha256": promotion.canonical_sha256(rank_inventory),
            "rank_state_count": 1,
            "rank_states_global_step": 500,
            "rank_states_batches_processed": 500,
            "dp_world_size": 1,
            "batches_replayed": 500,
            "total_data_errors": 0,
        },
        "evidence": {
            "recipe": {
                "path": str(tmp_path / "gantry-runtime" / recipe.name),
                "sha256": promotion.sha256_file(recipe),
            },
            "producer": {
                "path": str(tmp_path / "gantry-runtime" / producer.name),
                "sha256": promotion.sha256_file(producer),
            },
            "rank_state_inventory": rank_inventory,
        },
        "sources": {
            "pixmo_caption": source(0.69, 0.7),
            "pixmo_transcript": source(0.31, 0.3),
        },
        "summary": {
            "total_loss_weight": 1000.0,
            "total_active_loss_weight": 900.0,
            "share_sum": 1.0,
            "active_share_sum": 1.0,
            "within_tolerance": True,
        },
    }
    assert (
        promotion.validate_loss_mass_receipt(receipt, candidate=candidate)["batches_replayed"]
        == 500
    )

    receipt["loader"]["batches_replayed"] = 499
    with pytest.raises(promotion.PromotionValidationError, match="incomplete"):
        promotion.validate_loss_mass_receipt(receipt, candidate=candidate)


def test_optimizer_guard_receipt_locks_the_single_step356_deviation(tmp_path: Path):
    candidate = _candidate(tmp_path)
    evidence = tmp_path / "output.log"
    evidence.write_text(
        "\n".join(
            f"[step={step}/500,epoch=1]\n    optim/step skipped=" f"{1.0 if step == 356 else 0.0}"
            for step in range(1, 501)
        )
        + "\nFinalizing successful W&B run\n"
    )
    rank_state = tmp_path / "rank0.pt"
    torch.save(
        {
            "global_step": 500,
            "data_loader": {"batches_processed": 500, "total_data_errors": 0},
            "callbacks": {"wandb": {"run_id": "f2rdhz4y"}},
        },
        rank_state,
    )
    permanent = []
    for step in [0, 100, 200, 250, 300, 400, 500]:
        checkpoint = Path(candidate["checkpoint"]).parent / f"step{step}"
        checkpoint.mkdir(exist_ok=True)
        marker = checkpoint / ".metadata.json"
        marker.write_text(json.dumps({"ephemeral": False}) + "\n")
        permanent.append(
            {
                "step": step,
                "path": str(checkpoint),
                "marker_sha256": promotion.sha256_file(marker),
            }
        )
    receipt = {
        "format": promotion.OPTIMIZER_GUARD_RECEIPT_FORMAT,
        "version": 1,
        "status": "passed",
        "created_at": "2026-08-12T00:00:00+00:00",
        "candidate": _receipt_candidate(candidate),
        "run": {
            "run_id": "f2rdhz4y",
            "global_steps": 500,
            "exit_code": 0,
            "rank_state_count": 1,
            "permanent_checkpoint_steps": [0, 100, 200, 250, 300, 400, 500],
            "nonfinite_metric_count": 0,
            "unexpected_anomaly_count": 0,
        },
        "rank_state_inventory": [
            {
                "rank": 0,
                "path": str(rank_state),
                "sha256": promotion.sha256_file(rank_state),
                "global_step": 500,
                "batches_processed": 500,
                "total_data_errors": 0,
                "run_id": "f2rdhz4y",
            }
        ],
        "permanent_checkpoints": permanent,
        "guarded_skips": [
            {
                "step": 356,
                "count": 1,
                "reason_code": "optimizer_safety_guard",
                "waiver_required": True,
            }
        ],
        "unexpected_guarded_skip_count": 0,
        "evidence_artifact": promotion.artifact_reference(evidence),
    }
    assert (
        promotion.validate_optimizer_guard_receipt(receipt, candidate=candidate)[
            "guarded_skip_step"
        ]
        == 356
    )

    receipt["guarded_skips"][0]["step"] = 355
    with pytest.raises(promotion.PromotionValidationError, match="step356"):
        promotion.validate_optimizer_guard_receipt(receipt, candidate=candidate)


def _metrics(gap: float, *, null: bool = False) -> dict:
    ci = {"low": -0.01 if null else max(gap / 2, 0.001), "high": gap + 0.01}
    return {
        window: {
            "correct_ce_mean": 1.0,
            "wrong_ce_mean": 1.0 + gap,
            "gap_wrong_minus_correct_mean": gap,
            "win_rate": 0.9 if gap > 0 else 0.5,
            "mean_gap_bootstrap_ci": dict(ci),
        }
        for window in promotion.PRIMARY_WINDOWS
    }


def _pairing(indices: tuple[int, int]) -> dict:
    recipient, donor = indices
    return {
        "pairs": [{"recipient": recipient, "donor": donor}],
        "rows": [
            {"index": recipient, "content_id": f"{recipient + 1:064x}"},
            {"index": donor, "content_id": f"{donor + 1:064x}"},
        ],
    }


def _matched(
    root: Path,
    *,
    gap: float,
    pairing_sha: str,
    pair_indices: tuple[int, int],
    identity: str,
    null: bool = False,
    pairing_seed: int = 6198,
) -> dict:
    return {
        "checkpoint": {"root": str(root), "identity_sha256": identity},
        "pairings": {source: _pairing(pair_indices) for source in promotion.SOURCES},
        "receipt": {
            "protocol": {
                "pairing_sha256": {source: pairing_sha for source in promotion.SOURCES},
                "pairing_seed": pairing_seed,
                "bootstrap": {"seed": pairing_seed + promotion.INDEPENDENT_PAIRING_SEED_OFFSET},
            },
            "validation": {"manifest_sha256": "a" * 64, "row_content_sha256": "b" * 64},
            "pairings": {
                source: {
                    "path": f"/pairings/{pairing_sha}/{source}.json",
                    "sha256": pairing_sha,
                    "provenance": "loaded",
                    "excluded_primary_pairing": None,
                }
                for source in promotion.SOURCES
            },
            "artifact_policy": {"expected_excluded_pairing_sha256": {}},
            "results": {
                source: {"metrics": _metrics(gap, null=null)} for source in promotion.SOURCES
            },
        },
    }


def test_matched_set_requires_one_narrow_waiver_and_disjoint_replication(tmp_path: Path):
    lineage = tmp_path / "bridge"
    matched = {
        "canary_step250": _matched(
            tmp_path / "canary" / "step250",
            gap=0.2,
            pairing_sha="a" * 64,
            pair_indices=(0, 1),
            identity="1" * 64,
        ),
        "bridge_step250": _matched(
            lineage / "step250",
            gap=0.19,
            pairing_sha="a" * 64,
            pair_indices=(0, 1),
            identity="2" * 64,
        ),
        "bridge_step500": _matched(
            lineage / "step500",
            gap=0.3,
            pairing_sha="a" * 64,
            pair_indices=(0, 1),
            identity="3" * 64,
        ),
        "independent_step0": _matched(
            lineage / "step0",
            gap=0.0,
            pairing_sha="b" * 64,
            pair_indices=(2, 3),
            identity="4" * 64,
            null=True,
            pairing_seed=6198 + promotion.INDEPENDENT_PAIRING_SEED_OFFSET,
        ),
        "independent_step500": _matched(
            lineage / "step500",
            gap=0.25,
            pairing_sha="b" * 64,
            pair_indices=(2, 3),
            identity="3" * 64,
            pairing_seed=6198 + promotion.INDEPENDENT_PAIRING_SEED_OFFSET,
        ),
    }
    for source in promotion.SOURCES:
        primary_metadata = matched["bridge_step500"]["receipt"]["pairings"][source]
        primary_indices, _ = promotion._pairing_population(
            matched["bridge_step500"]["pairings"][source]
        )
        exclusion = {
            "path": primary_metadata["path"],
            "sha256": primary_metadata["sha256"],
            "excluded_recipient_and_donor_count": len(primary_indices),
            "excluded_indices_sha256": "f" * 64,
        }
        matched["independent_step0"]["receipt"]["pairings"][source]["provenance"] = "built"
        for role in ("independent_step0", "independent_step500"):
            matched[role]["receipt"]["pairings"][source]["excluded_primary_pairing"] = dict(
                exclusion
            )
            matched[role]["receipt"]["artifact_policy"]["expected_excluded_pairing_sha256"][
                source
            ] = primary_metadata["sha256"]
    # The only preregistered miss is caption first32: 0.179 < 0.18.
    matched["bridge_step250"]["receipt"]["results"]["pixmo_caption"]["metrics"]["first_32"][
        "gap_wrong_minus_correct_mean"
    ] = 0.179
    summary, deviations = promotion._validate_matched_set(matched, checkpoint=lineage / "step500")
    assert summary["independent_step0_null_reproduced"] is True
    assert [deviation["id"] for deviation in deviations] == [promotion.STEP250_WAIVER_ID]

    matched["independent_step500"]["receipt"]["protocol"]["bootstrap"]["seed"] += 1
    with pytest.raises(promotion.PromotionValidationError, match="bootstrap seeds"):
        promotion._validate_matched_set(matched, checkpoint=lineage / "step500")
    matched["independent_step500"]["receipt"]["protocol"]["bootstrap"]["seed"] -= 1

    matched["independent_step500"]["pairings"] = matched["bridge_step500"]["pairings"]
    with pytest.raises(promotion.PromotionValidationError, match="overlaps"):
        promotion._validate_matched_set(matched, checkpoint=lineage / "step500")


def test_bundle_content_hash_rejects_semantic_mutation():
    bundle = {
        "format": promotion.PROMOTION_BUNDLE_FORMAT,
        "version": 1,
        "status": "ready_for_human_approval",
        "created_at": "2026-08-12T00:00:00+00:00",
        "policy": {"name": promotion.PROMOTION_POLICY, "required_waiver_ids": []},
        "candidate": {},
        "receipts": {},
        "deviations": [],
        "content_sha256": "a" * 64,
    }
    with pytest.raises(promotion.PromotionValidationError, match="content SHA-256"):
        promotion.validate_promotion_bundle(bundle)


def test_artifact_reference_hashes_raw_bytes(tmp_path: Path):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"status": "passed"}) + "\n")
    reference = promotion.artifact_reference(path)
    path.write_text(json.dumps({"status": "changed"}) + "\n")
    with pytest.raises(promotion.PromotionValidationError, match="SHA-256 differs"):
        promotion._load_reference(reference, name="test")


def test_implementation_reference_uses_recorded_path_when_canonical_is_unavailable(
    tmp_path: Path,
):
    recorded_path = tmp_path / "gantry-runtime" / "vision_alignment_state_text.py"
    recorded_path.parent.mkdir()
    recorded_path.write_text("# evaluator from the checked-out repository\n")
    reference = promotion.artifact_reference(recorded_path)
    unavailable_canonical = (
        tmp_path / "site-packages" / "scripts" / "eval" / "vision_alignment_state_text.py"
    )

    assert (
        promotion._validate_implementation_reference(
            reference,
            name="frozen-state evaluator",
            expected_basename="vision_alignment_state_text.py",
            canonical_path=unavailable_canonical,
        )
        == recorded_path.resolve()
    )

    recorded_path.write_text("# mutated evaluator\n")
    with pytest.raises(promotion.PromotionValidationError, match="differs from its pin"):
        promotion._validate_implementation_reference(
            reference,
            name="frozen-state evaluator",
            expected_basename="vision_alignment_state_text.py",
            canonical_path=unavailable_canonical,
        )


def test_implementation_reference_does_not_bypass_mismatched_canonical(tmp_path: Path):
    canonical_path = tmp_path / "installed" / "vision_alignment_state_text.py"
    canonical_path.parent.mkdir()
    canonical_path.write_text("# different installed evaluator\n")
    recorded_path = tmp_path / "gantry-runtime" / "vision_alignment_state_text.py"
    recorded_path.parent.mkdir()
    recorded_path.write_text("# evaluator matching the receipt\n")
    reference = promotion.artifact_reference(recorded_path)

    with pytest.raises(promotion.PromotionValidationError, match="differs from its pin"):
        promotion._validate_implementation_reference(
            reference,
            name="frozen-state evaluator",
            expected_basename="vision_alignment_state_text.py",
            canonical_path=canonical_path,
        )


def test_live_checkpoint_identity_hashes_every_dcp_file(tmp_path: Path):
    root = tmp_path / "step500"
    state = root / "model_and_optim"
    state.mkdir(parents=True)
    config = root / "config.json"
    marker = root / ".metadata.json"
    dcp_metadata = state / ".metadata"
    shard = state / "__0_0.distcp"
    config.write_text("{}\n")
    marker.write_text('{"ephemeral": false}\n')
    dcp_metadata.write_bytes(b"metadata")
    shard.write_bytes(b"checkpoint shard")
    files = sorted(state.iterdir())
    inventory = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": promotion.sha256_file(path),
        }
        for path in files
    ]
    identity = {
        "root": str(root),
        "state_dir": str(state),
        "config_sha256": promotion.sha256_file(config),
        "checkpoint_marker_sha256": promotion.sha256_file(marker),
        "dcp_metadata_sha256": promotion.sha256_file(dcp_metadata),
        "state_file_hash_algorithm": "sha256",
        "state_file_inventory_sha256": promotion.canonical_sha256(inventory),
        "state_file_inventory": inventory,
        "identity_sha256": "",
    }
    identity["identity_sha256"] = promotion.canonical_sha256(
        {key: value for key, value in identity.items() if key != "identity_sha256"}
    )

    validated = promotion._validate_checkpoint_identity(identity, name="candidate")
    promotion._validate_live_checkpoint_identity(validated, name="candidate", hash_workers=2)

    shard.write_bytes(b"tampered shard!")
    with pytest.raises(promotion.PromotionValidationError, match="DCP shard inventory differs"):
        promotion._validate_live_checkpoint_identity(validated, name="candidate", hash_workers=2)


def test_independent_matched_evaluator_requires_live_pinned_implementations(tmp_path: Path):
    evaluator = (
        Path(promotion.__file__).resolve().parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_matched_wrong.py"
    )
    pairing = Path(promotion.__file__).resolve().with_name("matched_wrong_image.py")
    reference = {
        "path": str(tmp_path / "missing-container" / evaluator.name),
        "sha256": promotion.sha256_file(evaluator),
        "pairing_implementation_path": str(tmp_path / "missing-container" / pairing.name),
        "pairing_implementation_sha256": promotion.sha256_file(pairing),
    }
    assert promotion._validate_matched_evaluator(
        reference, name="independent step0", verify_live=True
    )["sha256"] == promotion.sha256_file(evaluator)

    for field in ("sha256", "pairing_implementation_sha256"):
        wrong_sha = deepcopy(reference)
        wrong_sha[field] = "0" * 64
        with pytest.raises(promotion.PromotionValidationError, match="differs from its pin"):
            promotion._validate_matched_evaluator(
                wrong_sha, name="independent step0", verify_live=True
            )

    for field in ("path", "pairing_implementation_path"):
        wrong_name = deepcopy(reference)
        wrong_name[field] = str(tmp_path / "missing-container" / "wrong.py")
        with pytest.raises(promotion.PromotionValidationError, match="incompatible implementation"):
            promotion._validate_matched_evaluator(
                wrong_name, name="independent step0", verify_live=True
            )
