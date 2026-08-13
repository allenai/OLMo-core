from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from olmo_core.eval import vision_alignment_promotion as promotion


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_perception_state_text.py"
    )
    spec = importlib.util.spec_from_file_location("vision_alignment_state_text_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sentinel(tmp_path: Path) -> dict:
    paths = tmp_path / "data_paths.txt"
    paths.write_text("".join(f"s3://bucket/{index}.npy\n" for index in range(128)))
    value = {
        "format": promotion.TEXT_SENTINEL_FORMAT,
        "version": 1,
        "parent_checkpoint": str(tmp_path / "parent"),
        "parent_checkpoint_config_sha256": "a" * 64,
        "parent_data_paths": {
            "path": str(paths),
            "sha256": promotion.sha256_file(paths),
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
                "tokens": [token % 1000 for token in range(index, index + 257)],
            }
            for index in range(128)
        ],
        "content_sha256": "",
    }
    value["content_sha256"] = promotion.canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )
    # Exercise the strict JSON representation used by the actual producer.
    return json.loads(json.dumps(value))


def test_native_helper_is_emitted_and_raw_sha_pinned_for_every_receipt() -> None:
    module = _load_module()
    native_helper = module._load_matched_evaluator()
    snapshot_helper = module._load_snapshot_evaluator()
    helper_path = Path(native_helper.__file__).resolve()
    snapshot_path = Path(snapshot_helper.__file__).resolve()
    references = []
    for receipt_format in module._NATIVE_HELPER_RECEIPT_FORMATS:
        input_receipt = {"format": receipt_format, "status": "passed"}
        receipt = module._with_native_helper(input_receipt, native_helper, snapshot_helper)
        assert "native_helper" not in input_receipt
        assert set(receipt["native_helper"]) == {"path", "sha256"}
        assert receipt["native_helper"]["path"] == str(helper_path)
        assert receipt["native_helper"]["sha256"] == promotion.sha256_file(helper_path)
        assert receipt["snapshot_helper"] == {
            "path": str(snapshot_path),
            "sha256": promotion.sha256_file(snapshot_path),
        }
        references.append(receipt["native_helper"])
    assert all(reference == references[0] for reference in references[1:])


def test_native_helper_binding_rejects_wrong_module_or_existing_evidence() -> None:
    module = _load_module()
    receipt_format = next(iter(module._NATIVE_HELPER_RECEIPT_FORMATS))
    native_helper = module._load_matched_evaluator()
    snapshot_helper = module._load_snapshot_evaluator()
    with pytest.raises(RuntimeError, match="differs from"):
        module._with_native_helper(
            {"format": receipt_format},
            SimpleNamespace(__file__=module.__file__),
            snapshot_helper,
        )
    with pytest.raises(ValueError, match="already contains"):
        module._with_native_helper(
            {"format": receipt_format, "native_helper": {}},
            SimpleNamespace(),
            snapshot_helper,
        )
    with pytest.raises(ValueError, match="cannot be attached"):
        module._with_native_helper({"format": "unrelated"}, SimpleNamespace(), snapshot_helper)
    with pytest.raises(RuntimeError, match="checkpoint-snapshot helper"):
        module._with_native_helper(
            {"format": receipt_format},
            native_helper,
            SimpleNamespace(__file__=module.__file__),
        )


def test_private_checkpoint_loader_uses_and_removes_snapshot(tmp_path, monkeypatch) -> None:
    module = _load_module()
    snapshot_state = tmp_path / ".perception-checkpoint-snapshot-test" / "model_and_optim"
    snapshot_state.mkdir(parents=True)
    calls = []

    class SnapshotModule:
        @staticmethod
        def _materialize_checkpoint_snapshot_distributed(identity, *, base_dir):
            calls.append(("materialize", identity, base_dir))
            return snapshot_state

        @staticmethod
        def _remove_checkpoint_snapshot_distributed(state_dir):
            calls.append(("remove", state_dir))

    class NativeModule:
        @staticmethod
        def _native_checkpoint_load_coverage_distributed(train_module, state_dir):
            calls.append(("coverage", train_module, state_dir))
            return {"complete": True}

    class TrainModule:
        def load_state_dict_direct(self, state_dir, **kwargs):
            calls.append(("load", state_dir, kwargs))

    group = object()
    monkeypatch.setattr(module.dist, "group", SimpleNamespace(WORLD=group))
    coverage = module._load_private_checkpoint(
        TrainModule(),
        NativeModule(),
        SnapshotModule(),
        {"identity_sha256": "a" * 64},
        snapshot_base=tmp_path / "snapshots",
        checkpoint_load_threads=3,
    )
    assert coverage == {"complete": True}
    assert [call[0] for call in calls] == ["materialize", "coverage", "load", "remove"]
    assert calls[2][1] == snapshot_state
    assert calls[2][2]["thread_count"] == 3
    assert calls[2][2]["process_group"] is group


def test_evaluate_text_uses_all_supervised_positions(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    group = object()

    class FakeTrainModule:
        device = torch.device("cpu")
        dp_process_group = group

        def eval_batch(self, batch, *, return_response_logits):
            assert return_response_logits is True
            assert batch["input_ids"].device == self.device
            labels = batch["labels"].reshape(-1)
            logits = torch.full((labels.numel(), 1001), -10.0)
            logits[torch.arange(labels.numel()), labels] = 10.0
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(module.dist, "get_world_size", lambda selected: 1)
    monkeypatch.setattr(module.dist, "get_rank", lambda selected: 0)
    monkeypatch.setattr(module, "WORLD_SIZE", 1)

    def gather(output, value, *, group):
        output[0] = value

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    result = module._evaluate_text(FakeTrainModule(), _sentinel(tmp_path), batch_size=8)
    assert result["token_ce"].shape == (32_768,)
    assert result["argmax"].shape == (32_768,)
    assert torch.all(result["argmax"] < 1000)


@pytest.mark.parametrize("dp_rank", [0, 1])
def test_evaluate_text_shards_and_reconstructs_in_global_order(
    tmp_path: Path, monkeypatch, dp_rank: int
) -> None:
    module = _load_module()
    sentinel = _sentinel(tmp_path)
    summary = promotion.validate_text_sentinel(sentinel)
    group = object()
    batch_size = 8
    sequence_length = 256
    observed_first_tokens: list[int] = []

    class FakeTrainModule:
        device = torch.device("cpu")
        dp_process_group = group

        def eval_batch(self, batch, *, return_response_logits):
            assert return_response_logits is True
            observed_first_tokens.extend(batch["input_ids"][:, 0].tolist())
            labels = batch["labels"].reshape(-1)
            logits = torch.full((labels.numel(), 1001), -10.0)
            logits[torch.arange(labels.numel()), labels] = 10.0
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(module.dist, "get_world_size", lambda selected: 2)
    monkeypatch.setattr(module.dist, "get_rank", lambda selected: dp_rank)
    monkeypatch.setattr(module, "WORLD_SIZE", 2)

    def gather(output, value, *, group):
        assert value["global_start"] % (batch_size * 2) == dp_rank * batch_size
        global_start = value["global_start"] - dp_rank * batch_size
        local_labels = torch.tensor(
            summary["labels"][value["global_start"] : value["global_start"] + batch_size],
            dtype=torch.int64,
        )
        assert torch.equal(value["argmax"], local_labels)
        for rank in range(2):
            rank_start = global_start + rank * batch_size
            output[rank] = {
                "global_start": rank_start,
                "token_ce": torch.arange(rank_start, rank_start + batch_size, dtype=torch.float32)
                .reshape(batch_size, 1)
                .expand(batch_size, sequence_length)
                .clone(),
                "argmax": torch.tensor(
                    summary["labels"][rank_start : rank_start + batch_size],
                    dtype=torch.int64,
                ),
            }

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    result = module._evaluate_text(FakeTrainModule(), sentinel, batch_size=batch_size)

    expected_rows = [
        row
        for global_start in range(0, 128, batch_size * 2)
        for row in range(
            global_start + dp_rank * batch_size,
            global_start + (dp_rank + 1) * batch_size,
        )
    ]
    assert observed_first_tokens == expected_rows
    assert torch.equal(
        result["argmax"], torch.tensor(summary["labels"], dtype=torch.int64).reshape(-1)
    )
    assert torch.equal(
        result["token_ce"].reshape(128, sequence_length)[:, 0],
        torch.arange(128, dtype=torch.float32),
    )


def test_evaluate_text_requires_exact_global_batch_divisibility(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    group = object()

    class FakeTrainModule:
        device = torch.device("cpu")
        dp_process_group = group

        def eval_batch(self, batch, *, return_response_logits):
            raise AssertionError("Divisibility must be checked before evaluation")

    monkeypatch.setattr(module.dist, "get_world_size", lambda selected: 3)
    monkeypatch.setattr(module.dist, "get_rank", lambda selected: 0)
    monkeypatch.setattr(module, "WORLD_SIZE", 3)
    with pytest.raises(ValueError, match="global batch 24"):
        module._evaluate_text(FakeTrainModule(), _sentinel(tmp_path), batch_size=8)


@pytest.mark.parametrize(
    ("replacement", "error"),
    [
        ({"global_start": 2, "token_ce": torch.zeros(2, 3)}, "malformed fields"),
        (
            {
                "global_start": 3,
                "token_ce": torch.zeros(2, 3),
                "argmax": torch.zeros(2, 3, dtype=torch.int64),
            },
            "expected 2",
        ),
        (
            {
                "global_start": 2,
                "token_ce": torch.zeros(2, 3, dtype=torch.float64),
                "argmax": torch.zeros(2, 3, dtype=torch.int64),
            },
            "malformed token CE",
        ),
        (
            {
                "global_start": 2,
                "token_ce": torch.zeros(2, 3),
                "argmax": torch.zeros(6, dtype=torch.int64),
            },
            "malformed argmax",
        ),
    ],
)
def test_reconstruct_text_batch_rejects_malformed_rank_packets(replacement, error) -> None:
    module = _load_module()
    packets = [
        {
            "global_start": 0,
            "token_ce": torch.zeros(2, 3),
            "argmax": torch.zeros(2, 3, dtype=torch.int64),
        },
        replacement,
    ]
    with pytest.raises(RuntimeError, match=error):
        module._reconstruct_text_batch(packets, global_start=0, batch_size=2, sequence_length=3)


def test_rank_summary_consensus_reports_differing_rank(monkeypatch) -> None:
    module = _load_module()
    local = {"frozen_inventory_sha256": "a" * 64, "text_metrics": {"all_finite": True}}

    def gather(output, value):
        output[0] = value
        output[1] = {
            "frozen_inventory_sha256": "b" * 64,
            "text_metrics": {"all_finite": True},
        }

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match=r"rank 1:.*bbbb"):
        module._assert_rank_summary_consensus(local, world_size=2)


def test_model_state_descriptors_include_non_image_rows(monkeypatch) -> None:
    module = _load_module()

    class Part(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = torch.nn.Linear(1, 1, bias=False)
            self.lm = torch.nn.Module()
            self.lm.embeddings = torch.nn.Embedding(100_352, 1)

    class FakeTrainModule:
        def __init__(self):
            self.model_parts = [Part()]

        @staticmethod
        def _persistent_model_buffer_state_dict():
            return {}

    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 1)

    def gather(output, value, *, group):
        output[0] = value

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    descriptors = module._model_state_descriptors(FakeTrainModule(), ["vision.*"])
    assert descriptors["vision.weight"]["kind"] == "frozen_tensor"
    non_image = descriptors["lm.embeddings.weight[non_image_rows]"]
    assert non_image["kind"] == "non_image_embedding_rows"
    assert non_image["shape"] == [100_346, 1]


def test_model_state_descriptors_can_cover_all_model_tensors(monkeypatch) -> None:
    module = _load_module()

    class Part(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = torch.nn.Linear(1, 1, bias=False)
            self.connector = torch.nn.Linear(1, 1, bias=False)
            self.lm = torch.nn.Module()
            self.lm.embeddings = torch.nn.Embedding(100_352, 1)
            self.register_buffer("persistent", torch.tensor([3.0]))

    class FakeTrainModule:
        def __init__(self):
            self.model_parts = [Part()]

        def _persistent_model_buffer_state_dict(self):
            return {"persistent": self.model_parts[0].persistent}

    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 1)

    def gather(output, value, *, group):
        output[0] = value

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    descriptors = module._model_state_descriptors(
        FakeTrainModule(), ("*",), include_non_image_embedding_rows=False
    )
    assert set(descriptors) == {
        "buffer:persistent",
        "connector.weight",
        "lm.embeddings.weight",
        "vision.weight",
    }
    assert all(value["kind"] == "frozen_tensor" for value in descriptors.values())


def test_state_comparisons_are_exact_and_detect_mismatch() -> None:
    module = _load_module()
    left = {
        "a": {
            "kind": "frozen_tensor",
            "dtype": "torch.float32",
            "shape": [2],
            "numel": 2,
            "sha256": "a" * 64,
        }
    }
    right = json.loads(json.dumps(left))
    comparisons, mismatch_count = module._state_comparisons(left, right)
    assert mismatch_count == 0
    assert comparisons == [
        {
            "name": "a",
            "kind": "frozen_tensor",
            "dtype": "torch.float32",
            "shape": [2],
            "numel": 2,
            "reference_sha256": "a" * 64,
            "candidate_sha256": "a" * 64,
        }
    ]

    right["a"]["sha256"] = "b" * 64
    comparisons, mismatch_count = module._state_comparisons(left, right)
    assert mismatch_count == 1
    assert comparisons[0]["candidate_sha256"] == "b" * 64


def test_state_comparisons_reject_surface_and_shape_drift() -> None:
    module = _load_module()
    descriptor = {
        "kind": "frozen_tensor",
        "dtype": "torch.float32",
        "shape": [2],
        "numel": 2,
        "sha256": "a" * 64,
    }
    with pytest.raises(RuntimeError, match="comparison surfaces differ"):
        module._state_comparisons({"a": descriptor}, {"b": descriptor})
    changed = dict(descriptor, shape=[1, 2])
    with pytest.raises(RuntimeError, match="changed shape or dtype"):
        module._state_comparisons({"a": descriptor}, {"a": changed})
