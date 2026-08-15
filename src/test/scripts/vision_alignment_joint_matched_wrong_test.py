"""Focused fail-closed contracts for the joint matched/wrong evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def module():
    path = (
        Path(__file__).parents[2] / "scripts" / "eval" / "vision_alignment_joint_matched_wrong.py"
    )
    spec = importlib.util.spec_from_file_location("joint_matched_wrong_test_module", path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def _args(module, *, pairing_only: bool):
    values = [
        "--checkpoint=/tmp/step4000",
        f"--expected-config-sha256={module.EXPECTED_CONFIG_SHA256}",
        "--pairing-dir=/tmp/pairings",
    ]
    if pairing_only:
        values.extend(["--pairing-only", "--pairing-manifest-output=/tmp/pairing-manifest.json"])
    else:
        values.extend(
            [
                "--pairing-manifest=/tmp/pairing-manifest.json",
                f"--expected-pairing-manifest-sha256={'a' * 64}",
                "--output=/tmp/result.json",
                "--work-dir=/tmp/joint-eval-work",
            ]
        )
    return module._parser().parse_args(values)


def test_pairing_is_world1_but_evaluation_is_exact_one_node_ep8(module, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    module._validate_args(_args(module, pairing_only=True))
    with pytest.raises(ValueError, match="WORLD_SIZE=8"):
        module._validate_args(_args(module, pairing_only=False))

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    module._validate_args(_args(module, pairing_only=False))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    with pytest.raises(ValueError, match="LOCAL_WORLD_SIZE=8"):
        module._validate_args(_args(module, pairing_only=False))


def test_pairing_seed_maximum_and_config_are_frozen(module, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    args = _args(module, pairing_only=True)
    args.examples = 504
    with pytest.raises(ValueError, match="examples=512"):
        module._validate_args(args)
    args = _args(module, pairing_only=True)
    args.pairing_seed = 7
    with pytest.raises(ValueError, match="seed.*6198"):
        module._validate_args(args)
    args = _args(module, pairing_only=True)
    args.expected_config_sha256 = "0" * 64
    with pytest.raises(ValueError, match="reviewed joint"):
        module._validate_args(args)


def test_evaluation_requires_explicit_work_dir(module, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    args = _args(module, pairing_only=False)
    args.work_dir = None
    with pytest.raises(ValueError, match="explicit --work-dir"):
        module._validate_args(args)


def test_strict_json_uses_exact_snapshot_and_rejects_duplicate_nonfinite(module, tmp_path):
    path = tmp_path / "artifact.json"
    path.write_bytes(b'{"value":1}\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload, actual = module._load_json_bytes(path, expected_sha256=digest, name="artifact")
    assert payload == {"value": 1}
    assert actual == digest

    path.write_bytes(b'{"value":1,"value":2}\n')
    with pytest.raises(ValueError, match="repeats key"):
        module._load_json_bytes(path, name="artifact")
    path.write_bytes(b'{"value":NaN}\n')
    with pytest.raises(ValueError, match="non-finite"):
        module._load_json_bytes(path, name="artifact")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_json_and_checkpoint_hashers_reject_fifo_without_blocking(module, tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="not a regular file"):
        module._load_json_bytes(fifo, name="fifo")
    with pytest.raises(ValueError, match="not a regular file"):
        module._stable_checkpoint_record(fifo, root=tmp_path)


def _example(*, metadata=None):
    return {
        "input_ids": np.asarray([1, 2], dtype=np.int64),
        "labels": np.asarray([2, -100], dtype=np.int64),
        "loss_masks": np.asarray([1.0, 0.0], dtype=np.float32),
        "position_ids": np.asarray([0, 1], dtype=np.int64),
        "token_type_ids": np.asarray([0, 0], dtype=np.int64),
        "images": np.zeros((1, 2, 3), dtype=np.float32),
        "pooled_patches_idx": np.asarray([[0, 1]], dtype=np.int64),
        "metadata": metadata,
    }


class _Dataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def get(self, index, epoch=0):
        assert epoch == 0
        return self.rows[index]

    def validate_image_content(self, indices=None):
        del indices
        return "a" * 64


def test_materialized_wrapper_validates_each_row_and_drops_only_metadata(module, monkeypatch):
    calls = []
    monkeypatch.setattr(
        module,
        "validate_joint_live_example",
        lambda row, **kwargs: calls.append((row, kwargs)),
    )
    raw = _example(metadata={"row": 0})
    validated = module._ValidatedJointDataset(
        _Dataset([raw]),
        source_name="pixmo_caption",
        source_kind="visual",
        token_ids=object(),
    )
    projected = module._PairingModelInputDataset(validated).get(0)
    assert set(projected) == set(raw) - {"metadata"}
    assert calls[0][1]["source_name"] == "pixmo_caption"
    raw["unknown"] = np.asarray([1])
    with pytest.raises(ValueError, match="extra=.*unknown"):
        module._PairingModelInputDataset(validated).get(0)


def test_loss_identity_preserves_filtered_native_training_divisor(module):
    labels = torch.tensor([[2, 3, -100]])
    masks = torch.tensor([[1.0, 1.0, 0.0]])
    ordinary = module._loss_identity({"labels": labels, "loss_masks": masks})
    assert ordinary == {
        "mask_tokens": 2,
        "labeled_tokens": 2,
        "mask_loss_weight": 2.0,
        "labeled_loss_weight": 2.0,
        "filtered": False,
    }
    filtered = module._loss_identity({"labels": torch.full_like(labels, -100), "loss_masks": masks})
    assert filtered["mask_loss_weight"] == 2.0
    assert filtered["labeled_loss_weight"] == 0.0
    assert filtered["filtered"] is True
    with pytest.raises(ValueError, match="partially"):
        module._loss_identity({"labels": torch.tensor([[2, -100, -100]]), "loss_masks": masks})


class _FakeTrainModule:
    device = torch.device("cpu")

    def __init__(self, summed_ce):
        self.summed_ce = summed_ce
        self.flags = []

    def eval_batch(self, batch, *, return_response_logits):
        del batch
        self.flags.append(return_response_logits)
        value = torch.tensor(float(self.summed_ce))
        assert module_placeholder is not None
        return module_placeholder.LMOutputWithLoss(
            logits=None, loss=value, ce_loss=value, z_loss=None
        )


# Assigned inside the test so the fake class can remain import-light at collection time.
module_placeholder = None


def test_scalar_forward_never_materializes_logits_and_allows_zero_filtered_ce(module):
    global module_placeholder
    module_placeholder = module
    masks = torch.tensor([[1.0, 1.0, 0.0]])
    labels = torch.full((1, 3), -100)
    train_module = _FakeTrainModule(0.0)
    result = module._forward_scalar_ce(train_module, {"labels": labels, "loss_masks": masks})
    assert train_module.flags == [False]
    assert result["ce"] is None
    assert result["summed_ce"] == 0.0
    with pytest.raises(ValueError, match="invalid CE"):
        module._forward_scalar_ce(_FakeTrainModule(1.0), {"labels": labels, "loss_masks": masks})


def _trainer_state(module, *, rank: int, step: int):
    total_errors = 1 if step == 8000 and rank in (0, 8) else 0
    return {
        "global_step": step,
        "global_train_tokens_seen": step * 1_048_576,
        "global_train_petaflops": 1.0,
        "max_steps": 16000,
        "data_loader": {
            "batches_processed": step,
            "epoch": 1,
            "seed": 95818,
            "consecutive_data_errors": 0,
            "total_data_errors": total_errors,
            "packing_state": {
                "dp_world_size": 16,
                "dp_rank": rank,
                "rank_instances": 8,
                "seq_len": 8192,
                "dataset_names": [
                    "audited_alignment",
                    "cosyn_point",
                    "count_numeric",
                    "native_text_replay",
                    "ocr_document",
                    "pixmo_caption",
                    "pixmo_points_basic",
                    "pixmo_points_high_frequency",
                    "pixmo_transcript",
                ],
            },
        },
        "epoch": 1,
        "world_size": 16,
        "rng": {},
        "callbacks": {
            "wandb": {
                "run_id": "4gxnu6we" if rank == 0 else None,
                "step": step,
                "name": module.EXPECTED_LINEAGE,
                "project": "vision-alignment",
            }
        },
    }


def test_trainer_state_freezes_real_progress_error_and_leader_run_id(module):
    module._validate_trainer_state(_trainer_state(module, rank=0, step=4000), rank=0, step=4000)
    module._validate_trainer_state(_trainer_state(module, rank=0, step=8000), rank=0, step=8000)
    module._validate_trainer_state(_trainer_state(module, rank=8, step=8000), rank=8, step=8000)
    bad = _trainer_state(module, rank=1, step=8000)
    bad["data_loader"]["total_data_errors"] = 1
    with pytest.raises(ValueError, match="differs"):
        module._validate_trainer_state(bad, rank=1, step=8000)
    bad = _trainer_state(module, rank=1, step=8000)
    bad["callbacks"]["wandb"]["run_id"] = "replica-run"
    with pytest.raises(ValueError, match="differs"):
        module._validate_trainer_state(bad, rank=1, step=8000)


def test_weights_only_same_byte_trainer_state_reader(module, tmp_path):
    root = tmp_path / "step8000"
    path = root / "train" / "rank8.pt"
    path.parent.mkdir(parents=True)
    state = _trainer_state(module, rank=8, step=8000)
    state["rng"] = {"numpy": np.random.RandomState(1).get_state()}
    torch.save(state, path)
    record, loaded = module._read_trainer_state(path, root=root)
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert module._validate_trainer_state(loaded, rank=8, step=8000)["total_data_errors"] == 1


def test_private_snapshot_copies_only_exact_inventory_bytes(module, tmp_path):
    root = tmp_path / "step4000"
    state = root / "model_and_optim"
    state.mkdir(parents=True)
    (state / ".metadata").write_bytes(b"metadata")
    (state / "__0_0.distcp").write_bytes(b"payload")
    inventory = [
        module._stable_checkpoint_record(path, root=root) for path in sorted(state.iterdir())
    ]
    identity = {"root": str(root), "state_file_inventory": inventory}
    copied_state = module._materialize_checkpoint_snapshot(
        identity, base_dir=tmp_path / "snapshots"
    )
    for record in inventory:
        copied = copied_state / Path(record["path"]).name
        assert copied.stat().st_size == record["size"]
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == record["sha256"]


def test_pairing_builder_selects_largest_common_multiple_of_eight(module, monkeypatch):
    limiting_source = module.JOINT_VISUAL_SOURCE_NAMES[3]

    def build(dataset, *, recipient_count, seed, content_ids, epoch):
        assert seed == 6198
        assert epoch == 0
        assert len(content_ids) == 512
        if dataset == limiting_source and recipient_count > 300:
            raise ValueError(
                "Could not select enough validation rows with a distinct exact-geometry image "
                f"donor: requested {recipient_count}, found 300 across 512 rows"
            )
        return {
            "version": 2,
            "recipient_count": recipient_count,
            "coverage": {"source": dataset},
            "pairs": [
                {"recipient": index, "donor": recipient_count + index}
                for index in range(recipient_count)
            ],
        }

    monkeypatch.setattr(module, "build_matched_wrong_image_pairing", build)
    datasets = {source: source for source in module.JOINT_VISUAL_SOURCE_NAMES}
    content_ids = {source: ("a" * 64,) * 512 for source in module.JOINT_VISUAL_SOURCE_NAMES}
    count, pairings = module._build_largest_common_pairings(datasets, content_ids=content_ids)
    assert count == 296
    assert all(pairing["recipient_count"] == 296 for pairing in pairings.values())
    assert module._build_largest_common_pairings(datasets, content_ids=content_ids) == (
        count,
        pairings,
    )


def test_immutable_output_refuses_replacement(module, tmp_path):
    output = tmp_path / "receipt.json"
    digest = module._write_json_exclusive(output, {"status": "valid"})
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module._write_json_exclusive(output, {"status": "different"})


def test_immutable_writer_rejects_preexisting_temporary_symlink(module, tmp_path):
    output = tmp_path / "receipt.json"
    victim = tmp_path / "victim"
    victim.write_bytes(b"do-not-touch")
    temporary = tmp_path / f".{output.name}.{os.getpid()}.tmp"
    temporary.symlink_to(victim)
    with pytest.raises(FileExistsError):
        module._write_bytes_exclusive(output, b"new-bytes")
    assert victim.read_bytes() == b"do-not-touch"
    assert temporary.is_symlink()
    assert not output.exists()


def test_clean_git_is_required_before_expensive_work(module):
    clean = {
        "revision": "a" * 40,
        "dirty": False,
        "status_sha256": "b" * 64,
        "tracked_diff_sha256": "c" * 64,
    }
    assert module._validate_clean_git_identity(clean) == clean
    dirty = dict(clean, dirty=True)
    with pytest.raises(ValueError, match="clean git"):
        module._validate_clean_git_identity(dirty)


def test_receipt_cross_schema_validates_exact_published_bytes(module, monkeypatch, tmp_path):
    producer = {"path": "evaluator", "sha256": "a" * 64}
    git = {
        "revision": "b" * 40,
        "dirty": False,
        "status_sha256": "c" * 64,
        "tracked_diff_sha256": "d" * 64,
    }
    payload = {"producer": producer, "git": git, "value": 1}
    seen = {}

    class Comparator:
        @staticmethod
        def validate_evaluator_receipt(path, expected_sha256, step, verify_live_checkpoint=False):
            raw = Path(path).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == expected_sha256
            assert step == 4000
            assert verify_live_checkpoint is False
            seen["raw"] = raw

    monkeypatch.setattr(module, "_load_comparator_contract", lambda: Comparator)
    monkeypatch.setattr(module, "_producer_identity", lambda: producer)
    monkeypatch.setattr(module.bridge, "_git_identity", lambda: git)
    output = tmp_path / "result" / "receipt.json"
    work = tmp_path / "work"
    digest = module._write_validated_receipt(
        output,
        payload,
        work_dir=work,
        step=4000,
    )
    assert output.read_bytes() == seen["raw"]
    assert digest == hashlib.sha256(seen["raw"]).hexdigest()
    assert list(work.glob(".joint-receipt-validation-*")) == []


def test_work_dir_must_be_empty_and_disjoint_from_all_evidence(module, tmp_path):
    checkpoint = tmp_path / "checkpoints" / "step4000"
    pairing = tmp_path / "pairings"
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "results" / "step4000.json"
    manifest = tmp_path / "pairing-manifest.json"
    for path in (checkpoint, pairing, artifacts, output.parent):
        path.mkdir(parents=True, exist_ok=True)
    raw_config = {
        "data": {
            "joint_visual_projection_path": str(artifacts / "projection.json"),
            "source_audit_path": str(artifacts / "audit.json"),
            "native_text_replay": {"manifest_path": str(artifacts / "train.json")},
        },
        "evaluation": {"native_text_holdout": {"manifest_path": str(artifacts / "holdout.json")}},
    }
    with pytest.raises(ValueError, match="overlaps protected evidence"):
        module._validate_work_dir(
            checkpoint / "eval-work",
            checkpoint_root=checkpoint,
            pairing_dir=pairing,
            output=output,
            pairing_manifest=manifest,
            raw_config=raw_config,
        )
    work = tmp_path / "work-step4000"
    assert (
        module._validate_work_dir(
            work,
            checkpoint_root=checkpoint,
            pairing_dir=pairing,
            output=output,
            pairing_manifest=manifest,
            raw_config=raw_config,
        )
        == work
    )
    work.mkdir()
    (work / "orphan").write_text("snapshot")
    with pytest.raises(ValueError, match="new or empty"):
        module._validate_work_dir(
            work,
            checkpoint_root=checkpoint,
            pairing_dir=pairing,
            output=output,
            pairing_manifest=manifest,
            raw_config=raw_config,
        )


def test_native_metrics_bind_five_filtered_rows_and_both_denominators(module):
    filtered = {334, 478, 610, 780, 792}
    rows = []
    for index in range(1000):
        is_filtered = index in filtered
        rows.append(
            {
                "dataset_index": index,
                "mask_tokens": 2,
                "labeled_tokens": 0 if is_filtered else 2,
                "mask_loss_weight": 2.0,
                "labeled_loss_weight": 0.0 if is_filtered else 2.0,
                "summed_ce": 0.0 if is_filtered else 4.0,
                "filtered": is_filtered,
                "ce": None if is_filtered else 2.0,
            }
        )
    metrics = module._native_metrics(rows)
    assert metrics["filtered_indices"] == [334, 478, 610, 780, 792]
    assert metrics["ce_loss"] == 2.0
    assert metrics["training_divisor_ce"] == pytest.approx(1.99)
    broken = [dict(row) for row in rows]
    broken[334]["ce"] = 0.0
    with pytest.raises(ValueError, match="dual-denominator"):
        module._native_metrics(broken)


def test_receipt_is_valid_but_explicitly_descriptive_and_nonpromotional(module, monkeypatch):
    monkeypatch.setattr(module, "_producer_identity", lambda: {"path": "p", "sha256": "s"})
    monkeypatch.setattr(module.bridge, "_git_identity", lambda: {"revision": "r"})
    visual = {source: {"source": source} for source in module.JOINT_VISUAL_SOURCE_NAMES}
    blank = {source: {"source": source} for source in module.BLANK_SOURCE_NAMES}
    payload = module._receipt_payload(
        checkpoint={"identity_sha256": "c"},
        checkpoint_config={"sha256": "d"},
        load_coverage={"complete": True},
        projection={"raw_sha256": "e"},
        source_audit={"fingerprint": "f"},
        tokenizer={"fingerprint": "g"},
        pairing_manifest={"sha256": "h"},
        protocol={"descriptive_only": True, "promotion_eligible": False},
        visual_results=visual,
        blank_results=blank,
        native_result={"examples": 1000},
        producer={"path": "p", "sha256": "s"},
        git={"revision": "r"},
    )
    assert payload["format"] == module.RECEIPT_FORMAT
    assert payload["status"] == "valid"
    assert payload["artifact_policy"]["descriptive_only"] is True
    assert payload["artifact_policy"]["promotion_eligible"] is False
    unsigned = dict(payload)
    digest = unsigned.pop("content_sha256")
    assert digest == module._canonical_sha256(unsigned)
