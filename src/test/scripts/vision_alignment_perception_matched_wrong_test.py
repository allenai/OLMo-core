"""Focused contracts for the perception matched-wrong evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_module():
    path = (
        Path(__file__).parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_perception_matched_wrong.py"
    )
    spec = importlib.util.spec_from_file_location("perception_matched_wrong_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(module, *, pairing_only: bool):
    values = [
        "--checkpoint=/tmp/step4000",
        f"--expected-config-sha256={'a' * 64}",
        "--profile-pair-receipt=/tmp/pair.json",
        f"--expected-profile-pair-receipt-sha256={'b' * 64}",
    ]
    if pairing_only:
        values.extend(["--pairing-only", "--pairing-manifest-output=/tmp/manifest.json"])
    else:
        values.append("--output=/tmp/result.json")
    return module._parser().parse_args(values)


def test_pairing_only_is_cpu_world1_but_model_evaluation_is_ep8(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("WORLD_SIZE", "1")
    module._validate_args(_args(module, pairing_only=True))
    with pytest.raises(ValueError, match="EP8 evaluation requires WORLD_SIZE=8"):
        module._validate_args(_args(module, pairing_only=False))

    monkeypatch.setenv("WORLD_SIZE", "8")
    module._validate_args(_args(module, pairing_only=False))
    with pytest.raises(ValueError, match="CPU pairing preparation requires WORLD_SIZE=1"):
        module._validate_args(_args(module, pairing_only=True))


def test_pairing_runtime_uses_gloo_without_preparing_cuda(monkeypatch):
    module = _load_module()
    calls = []
    monkeypatch.setattr(module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        module.dist,
        "init_process_group",
        lambda **kwargs: calls.append(("init", kwargs)),
    )
    monkeypatch.setattr(
        module.dist, "destroy_process_group", lambda: calls.append(("destroy", None))
    )
    monkeypatch.setattr(
        module,
        "prepare_training_environment",
        lambda: pytest.fail("CPU pairing prep initialized the CUDA training environment"),
    )

    state = module._initialize_runtime(pairing_only=True)
    assert calls[0][0] == "init"
    assert calls[0][1]["backend"] == "gloo"
    assert calls[0][1]["world_size"] == 1
    module._teardown_runtime(state)
    assert calls[-1] == ("destroy", None)


def test_json_loader_hashes_the_same_strict_byte_snapshot(tmp_path):
    module = _load_module()
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b'{"value":1}\n')
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    payload, digest = module._load_json_bytes(artifact, expected_sha256=expected, name="artifact")
    assert payload == {"value": 1}
    assert digest == expected

    artifact.write_bytes(b'{"value":1,"value":2}\n')
    with pytest.raises(ValueError, match="repeats key"):
        module._load_json_bytes(artifact, name="artifact")


def _model_example(*, image_value: float, metadata):
    return {
        "input_ids": np.asarray([1, 2, 3], dtype=np.int64),
        "labels": np.asarray([2, 3, -100], dtype=np.int64),
        "loss_masks": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        "position_ids": np.asarray([0, 1, 2], dtype=np.int64),
        "token_type_ids": np.asarray([0, 0, 0], dtype=np.int64),
        "images": np.full((1, 2, 3), image_value, dtype=np.float32),
        "pooled_patches_idx": np.asarray([[0, 1]], dtype=np.int64),
        "metadata": metadata,
    }


class _MetadataDataset:
    def __init__(self, examples):
        self.examples = examples
        self.validated_indices = []

    def __len__(self):
        return len(self.examples)

    def get(self, index, epoch=0):
        assert epoch == 0
        return self.examples[index]

    def validate_image_content(self, indices=None):
        self.validated_indices.append(indices)
        return "a" * 64


def test_pairing_projection_drops_only_metadata_and_delegates_image_validation():
    module = _load_module()
    raw = _model_example(
        image_value=1.0,
        metadata={"original_length": 1268, "truncated": False},
    )
    base = _MetadataDataset([raw])
    dataset = module._PairingModelInputDataset(base)

    projected = dataset.get(0)
    assert set(projected) == set(raw) - {"metadata"}
    assert all(projected[field] is raw[field] for field in projected)
    assert dataset.validate_image_content() == "a" * 64
    assert base.validated_indices == [None]


def test_pairing_projection_rejects_unknown_fields():
    module = _load_module()
    raw = _model_example(image_value=1.0, metadata={})
    raw["unversioned_control"] = np.asarray([1], dtype=np.int64)

    with pytest.raises(ValueError, match="unsupported non-model fields.*unversioned_control"):
        module._PairingModelInputDataset(_MetadataDataset([raw])).get(0)


def test_metadata_mutation_cannot_change_projected_collated_batch():
    module = _load_module()
    raw = _model_example(image_value=1.0, metadata={"original_length": 3})
    dataset = module._PairingModelInputDataset(_MetadataDataset([raw]))
    collator = module.MultimodalCollator(pad_token_id=0, pad_sequence_length=3)
    before = collator([dataset.get(0)])

    raw["metadata"] = {"original_length": 999999, "nested": {"arbitrary": object()}}
    after = collator([dataset.get(0)])

    assert set(before) == set(after)
    assert all(torch.equal(before[field], after[field]) for field in before)


def test_projected_dataset_builds_and_replays_pairing_after_metadata_drift():
    module = _load_module()
    examples = [
        _model_example(image_value=1.0, metadata={"row": 0}),
        _model_example(image_value=2.0, metadata={"row": 1}),
    ]
    dataset = module._PairingModelInputDataset(_MetadataDataset(examples))
    payload = module.build_matched_wrong_image_pairing(
        dataset,
        recipient_count=2,
        seed=6198,
        content_ids=("1" * 64, "2" * 64),
        epoch=0,
    )
    digest = module.matched_wrong_image_pairing_sha256(payload)

    examples[0]["metadata"] = {"row": "mutated", "arbitrary": object()}
    examples[1]["metadata"] = None
    correct = module.bridge.MultimodalFixedValidationDataset(
        dataset, pairing=payload, pairing_sha256=digest
    )
    wrong = module.bridge.MultimodalMatchedWrongImageDataset(
        dataset, pairing=payload, pairing_sha256=digest
    )

    for index, pair in enumerate(payload["pairs"]):
        correct_example = correct[index]
        wrong_example = wrong[index]
        donor = dataset.get(pair["donor"])
        assert "metadata" not in correct_example
        assert "metadata" not in wrong_example
        assert np.array_equal(wrong_example["images"], donor["images"])
        assert all(
            np.array_equal(correct_example[field], wrong_example[field])
            for field in correct_example
            if field != "images"
        )


def test_pairing_preparation_selects_largest_common_batch_multiple(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda packet, src: None)
    monkeypatch.setattr(
        module, "validate_matched_wrong_image_pairing", lambda *args, **kwargs: None
    )

    def build(dataset, *, recipient_count, seed, content_ids, epoch):
        del seed, content_ids, epoch
        if dataset == "limiting" and recipient_count > 300:
            raise ValueError(
                "Could not select enough validation rows with a distinct exact-geometry image "
                f"donor: requested {recipient_count}, found 300 across 512 rows"
            )
        return {
            "version": 2,
            "recipient_count": recipient_count,
            "coverage": {"dataset": dataset, "selected": recipient_count},
            "pairs": [
                {"recipient": index, "donor": recipient_count + index}
                for index in range(recipient_count)
            ],
        }

    def serialize(payload):
        return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()

    monkeypatch.setattr(module, "build_matched_wrong_image_pairing", build)
    monkeypatch.setattr(module, "serialize_matched_wrong_image_pairing", serialize)
    monkeypatch.setattr(
        module,
        "matched_wrong_image_pairing_sha256",
        lambda payload: hashlib.sha256(serialize(payload)).hexdigest(),
    )
    sources = list(module.PERCEPTION_SOURCE_NAMES)
    datasets = {source: source for source in sources}
    datasets[sources[2]] = "limiting"
    paths = {source: tmp_path / f"{source}.json" for source in sources}
    content_ids = {source: ("a" * 64,) * 512 for source in sources}

    examples, payloads, metadata = module._prepare_pairings_distributed(
        datasets,
        paths=paths,
        pins={},
        maximum_examples=512,
        seed=6198,
        content_ids=content_ids,
    )

    assert examples == 288
    assert set(payloads) == set(sources)
    assert set(metadata) == set(sources)
    assert all(payload["recipient_count"] == 288 for payload in payloads.values())
    assert all(path.is_file() for path in paths.values())


def test_late_new_pairing_pin_mismatch_publishes_nothing(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda packet, src: None)
    monkeypatch.setattr(
        module, "validate_matched_wrong_image_pairing", lambda *args, **kwargs: None
    )

    def build(dataset, *, recipient_count, seed, content_ids, epoch):
        del dataset, seed, content_ids, epoch
        return {
            "version": 2,
            "recipient_count": recipient_count,
            "coverage": {},
            "pairs": [
                {"recipient": index, "donor": recipient_count + index}
                for index in range(recipient_count)
            ],
        }

    def serialize(payload):
        return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()

    monkeypatch.setattr(module, "build_matched_wrong_image_pairing", build)
    monkeypatch.setattr(module, "serialize_matched_wrong_image_pairing", serialize)
    monkeypatch.setattr(
        module,
        "matched_wrong_image_pairing_sha256",
        lambda payload: hashlib.sha256(serialize(payload)).hexdigest(),
    )
    paths = {source: tmp_path / f"{source}.json" for source in module.PERCEPTION_SOURCE_NAMES}
    datasets = {source: [None] * 512 for source in module.PERCEPTION_SOURCE_NAMES}
    content_ids = {source: ("a" * 64,) * 512 for source in module.PERCEPTION_SOURCE_NAMES}

    expected_payload = build(
        None,
        recipient_count=32,
        seed=6198,
        content_ids=(),
        epoch=0,
    )
    expected_digest = hashlib.sha256(serialize(expected_payload)).hexdigest()
    pins = {source: expected_digest for source in module.PERCEPTION_SOURCE_NAMES}
    pins[module.PERCEPTION_SOURCE_NAMES[-1]] = "0" * 64
    with pytest.raises(RuntimeError, match="supplied SHA-256 pin"):
        module._prepare_pairings_distributed(
            datasets,
            paths=paths,
            pins=pins,
            maximum_examples=32,
            seed=6198,
            content_ids=content_ids,
        )
    assert not any(path.exists() for path in paths.values())


def test_perception_evaluator_reuses_bridge_helpers_without_modifying_source():
    module = _load_module()
    assert Path(module.bridge.__file__).name == "vision_alignment_matched_wrong.py"
    assert module.bridge.SCHEMA_VERSION == 3
    assert module.SCHEMA_VERSION == 4


def _tiny_checkpoint(module, root: Path):
    state_dir = root / "model_and_optim"
    state_dir.mkdir(parents=True)
    (state_dir / ".metadata").write_bytes(b"metadata")
    (state_dir / "__0_0.distcp").write_bytes(b"checkpoint payload")
    (root / "config.json").write_text("{}\n")
    (root / ".metadata.json").write_text('{"ephemeral":false}\n')
    return module._checkpoint_identity(root, root / "config.json", hash_workers=1)


def test_private_checkpoint_snapshot_copies_exact_attested_bytes(tmp_path):
    module = _load_module()
    identity = _tiny_checkpoint(module, tmp_path / "step4000")
    state_dir = module._materialize_checkpoint_snapshot(identity, base_dir=tmp_path / "snapshots")

    for record in identity["state_file_inventory"]:
        relative = Path(record["path"]).relative_to("model_and_optim")
        copied = state_dir / relative
        assert copied.stat().st_size == record["size"]
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == record["sha256"]


def test_private_checkpoint_snapshot_rejects_changed_or_symlinked_source(tmp_path):
    module = _load_module()
    root = tmp_path / "step4000"
    identity = _tiny_checkpoint(module, root)
    shard = root / "model_and_optim" / "__0_0.distcp"
    shard.write_bytes(b"changed payload")
    with pytest.raises(ValueError, match="changed or differed"):
        module._materialize_checkpoint_snapshot(identity, base_dir=tmp_path / "changed-snapshots")

    identity = _tiny_checkpoint(module, tmp_path / "second-step4000")
    state_dir = Path(identity["state_dir"])
    metadata = state_dir / ".metadata"
    target = tmp_path / "metadata-target"
    target.write_bytes(metadata.read_bytes())
    metadata.unlink()
    metadata.symlink_to(target)
    with pytest.raises(OSError):
        module._materialize_checkpoint_snapshot(identity, base_dir=tmp_path / "symlink-snapshots")
