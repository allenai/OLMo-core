"""Focused contracts for the native Vision Alignment matched-wrong evaluator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_module():
    path = Path(__file__).parents[2] / "scripts" / "eval" / "vision_alignment_matched_wrong.py"
    spec = importlib.util.spec_from_file_location(
        "vision_alignment_matched_wrong_test_module", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _response_logits(labels: torch.Tensor, *, target_logit: float, distractor_logit: float):
    logits = torch.zeros((len(labels), 4), dtype=torch.float32)
    logits[:, 3] = distractor_logit
    logits[torch.arange(len(labels)), labels] = target_logit
    return logits


class _FakeScalar:
    def __init__(self, value: str):
        self.value = value

    def as_py(self) -> str:
        return self.value


class _FakeArrowTable:
    def __init__(self, paths):
        self.paths = paths

    def column(self, name: str):
        assert name == "image"
        return [_FakeScalar(path) for path in self.paths]


class _FakeLiveDataset:
    def __init__(self, paths, fingerprint: str):
        self.paths = paths
        self._fingerprint = fingerprint
        self.data = _FakeArrowTable(paths)

    def __len__(self):
        return len(self.paths)


def test_response_ce_windows_gaps_and_bootstrap_are_per_example_and_deterministic():
    module = _load_module()
    batch = {
        "labels": torch.tensor([[-100, 0, 1, -100], [2, 1, 0, -100]]),
        "loss_masks": torch.tensor([[0.0, 1.0, 1.0, 0.0], [2.0, 2.0, 2.0, 0.0]]),
    }
    selected_labels = torch.tensor([0, 1, 2, 1, 0])
    correct = module._response_ce_by_example(
        batch,
        _response_logits(selected_labels, target_logit=3.0, distractor_logit=0.0),
    )
    wrong = module._response_ce_by_example(
        batch,
        _response_logits(selected_labels, target_logit=0.0, distractor_logit=2.0),
    )

    assert [row["response_tokens"] for row in correct] == [2, 3]
    assert correct[0]["windows"]["first_1"] == pytest.approx(correct[0]["windows"]["first_8"])
    assert correct[0]["windows"]["all"] < wrong[0]["windows"]["all"]
    records = [
        {
            "correct_ce": correct[index]["windows"],
            "wrong_ce": wrong[index]["windows"],
        }
        for index in range(2)
    ]
    first = module._aggregate_records(records, bootstrap_seed=17, bootstrap_samples=200)
    second = module._aggregate_records(records, bootstrap_seed=17, bootstrap_samples=200)

    assert first == second
    assert first["all"]["gap_wrong_minus_correct_mean"] > 0
    assert first["all"]["win_rate"] == 1.0
    assert first["first_32"]["examples"] == 2
    assert first["all"]["mean_gap_bootstrap_ci"]["low"] > 0


def test_correct_and_wrong_batches_may_change_only_image_pixels():
    module = _load_module()
    correct = {
        "input_ids": torch.tensor([[1, 2]]),
        "labels": torch.tensor([[2, 3]]),
        "pooled_patches_idx": torch.tensor([[[0, 1]]]),
        "images": torch.zeros((1, 1, 2, 3)),
    }
    wrong = {name: value.clone() for name, value in correct.items()}
    wrong["images"].fill_(1)
    module._assert_batches_match(correct, wrong)

    wrong["pooled_patches_idx"][0, 0, 0] = 9
    with pytest.raises(ValueError, match="pooled_patches_idx"):
        module._assert_batches_match(correct, wrong)


def test_pairing_paths_are_explicit_per_source(tmp_path):
    module = _load_module()
    args = SimpleNamespace(
        pairing=[f"pixmo_caption={tmp_path / 'caption.json'}"],
        pairing_dir=str(tmp_path / "pairs"),
    )
    paths = module._resolve_pairing_paths(
        args,
        tmp_path / "result.json",
        ["pixmo_caption", "pixmo_transcript"],
    )
    assert paths["pixmo_caption"] == (tmp_path / "caption.json").resolve()
    assert paths["pixmo_transcript"] == (tmp_path / "pairs" / "pixmo_transcript.json")

    with pytest.raises(ValueError, match="duplicate"):
        module._parse_pairing_paths(
            [
                f"pixmo_caption={tmp_path / 'one.json'}",
                f"pixmo_caption={tmp_path / 'two.json'}",
            ]
        )


def test_validation_manifest_pins_dataset_and_row_content(tmp_path):
    module = _load_module()
    artifact = tmp_path / "artifact"
    dataset = artifact / "dataset"
    dataset.mkdir(parents=True)
    live_dataset = _FakeLiveDataset(["images/a.png", "images/b.png"], "validation-v3")
    path_inventory = module.pixmo_row_path_inventory(live_dataset)
    content_path = artifact / "validation-row-images.sha256"
    content_ids = ("1" * 64, "2" * 64)
    content_path.write_text("".join(f"{value}\n" for value in content_ids))
    manifest = {
        "format": "vision_alignment_validation_manifest",
        "version": 3,
        "builder": {
            "row_image_paths_algorithm": path_inventory["algorithm"],
        },
        "output": {
            "dataset_path": "dataset",
            "splits": {
                "validation": {
                    "examples": 2,
                    "dataset_fingerprint": "validation-v3",
                    "row_image_paths_sha256": path_inventory["sha256"],
                    "unique_image_paths": 2,
                    "row_image_content_path": content_path.name,
                    "row_image_content_sha256": module._sha256_file(content_path),
                }
            },
        },
    }
    manifest_path = artifact / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    raw_config = {
        "evaluation": {
            "validation_manifest_path": str(manifest_path),
            "validation_manifest_sha256": module._sha256_file(manifest_path),
        }
    }

    loaded, actual_ids, identity = module._load_validation_manifest(raw_config, str(dataset))
    assert loaded == manifest
    assert actual_ids == content_ids
    assert identity["manifest_sha256"] == module._sha256_file(manifest_path)
    assert module._validate_live_validation_dataset(live_dataset, loaded) == {
        "dataset_fingerprint": "validation-v3",
        "examples": 2,
        "row_image_paths_algorithm": path_inventory["algorithm"],
        "row_image_paths_sha256": path_inventory["sha256"],
        "unique_image_paths": 2,
    }

    live_dataset._fingerprint = "different"
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        module._validate_live_validation_dataset(live_dataset, loaded)
    live_dataset._fingerprint = "validation-v3"
    live_dataset.paths[1] = "images/replaced.png"
    with pytest.raises(ValueError, match="row_image_paths_sha256"):
        module._validate_live_validation_dataset(live_dataset, loaded)

    with pytest.raises(ValueError, match="differs from manifest output"):
        module._load_validation_manifest(raw_config, str(tmp_path / "other"))

    manifest["version"] = 2
    manifest_path.write_text(json.dumps(manifest))
    raw_config["evaluation"]["validation_manifest_sha256"] = module._sha256_file(manifest_path)
    with pytest.raises(ValueError, match="incompatible format"):
        module._load_validation_manifest(raw_config, str(dataset))


def test_checkpoint_identity_hashes_every_state_file_by_content(tmp_path):
    module = _load_module()
    checkpoint = tmp_path / "step7"
    state = checkpoint / "model_and_optim"
    state.mkdir(parents=True)
    config = checkpoint / "config.json"
    config.write_text("{}\n")
    (checkpoint / ".metadata.json").write_text('{"version":"test"}\n')
    (state / ".metadata").write_bytes(b"metadata")
    shard = state / "__0_0.distcp"
    shard.write_bytes(b"same-size-one")

    first = module._checkpoint_identity(checkpoint, config, hash_workers=2)
    shard.write_bytes(b"same-size-two")
    second = module._checkpoint_identity(checkpoint, config, hash_workers=2)
    assert first["identity_sha256"] != second["identity_sha256"]
    first_shard = next(
        row for row in first["state_file_inventory"] if row["path"].endswith(".distcp")
    )
    second_shard = next(
        row for row in second["state_file_inventory"] if row["path"].endswith(".distcp")
    )
    assert first_shard["size"] == second_shard["size"]
    assert first_shard["sha256"] != second_shard["sha256"]
    assert first["dcp_metadata_sha256"] == module._sha256_file(state / ".metadata")


def test_rank_zero_output_write_failure_is_broadcast_and_raised(tmp_path, monkeypatch):
    module = _load_module()
    packets = []
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        module.dist, "broadcast_object_list", lambda packet, src: packets.append(dict(packet[0]))
    )

    def fail_write(path, payload):
        raise OSError("disk full")

    monkeypatch.setattr(module, "_write_json_atomic", fail_write)
    with pytest.raises(RuntimeError, match="OSError: disk full"):
        module._write_result_distributed(tmp_path / "result.json", {"result": 1})
    assert packets == [{"ok": False, "error": "OSError: disk full"}]


def test_nonzero_rank_raises_broadcast_output_failure_without_writing(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module.dist, "get_rank", lambda: 1)

    def receive_failure(packet, src):
        assert src == 0
        packet[0] = {"ok": False, "error": "OSError: disk full"}

    monkeypatch.setattr(module.dist, "broadcast_object_list", receive_failure)
    monkeypatch.setattr(
        module,
        "_write_json_atomic",
        lambda *args, **kwargs: pytest.fail("nonzero rank attempted to write output"),
    )
    with pytest.raises(RuntimeError, match="OSError: disk full"):
        module._write_result_distributed(tmp_path / "result.json", {"result": 1})


def test_model_builder_preserves_saved_freeze_surface(monkeypatch):
    module = _load_module()

    class FakeLMConfig:
        pass

    class FakeModelConfig:
        def __init__(self):
            self.lm = FakeLMConfig()

        def build(self, init_device):
            assert init_device == "meta"
            return "model"

    class FakeMultimodalConfig:
        @classmethod
        def from_dict(cls, value):
            assert value == {"kind": "model"}
            return FakeModelConfig()

    saved_freeze = ["vision.*", "lm.blocks.*"]
    fake_module_config = SimpleNamespace(
        freeze_params=list(saved_freeze),
        ep_config=SimpleNamespace(degree=3),
        rank_microbatch_size=0,
        max_sequence_length=0,
        compile_model=True,
        vision_activation_checkpointing=True,
        connector_activation_checkpointing=True,
        response_logits_only=False,
        diagnostics_interval=100,
    )

    class FakeTrainModuleConfig:
        @classmethod
        def from_dict(cls, value):
            assert value["freeze_params"] == saved_freeze
            return fake_module_config

    monkeypatch.setattr(module, "OLMoDDPModelConfig", FakeLMConfig)
    monkeypatch.setattr(module, "MultimodalLMConfig", FakeMultimodalConfig)
    monkeypatch.setattr(module, "MultimodalOLMoDDPTrainModuleConfig", FakeTrainModuleConfig)
    monkeypatch.setattr(module, "_configure_lm_for_eval", lambda config: None)
    model, config = module._build_model_and_module(
        {
            "model": {"kind": "model"},
            "train_module": {"freeze_params": saved_freeze},
        },
        sequence_length=2560,
        rank_batch_instances=4,
    )

    assert model == "model"
    assert config.freeze_params == saved_freeze
    assert config.ep_config.degree == 8
    assert config.rank_microbatch_size == 4 * 2560
    assert config.compile_model is False
    assert config.response_logits_only is True
