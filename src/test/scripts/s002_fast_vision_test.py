import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "s002_fast_vision.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_s002_fast_vision_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fast_vision():
    return _load_module()


def test_representative_indices_are_deterministic_unique_and_bounded(fast_vision):
    first = fast_vision._representative_indices(100, 32, seed=6198)
    second = fast_vision._representative_indices(100, 32, seed=6198)

    assert first == second
    assert len(first) == len(set(first)) == 32
    assert min(first) >= 0
    assert max(first) < 100
    assert fast_vision._indices_sha256(first) == fast_vision._indices_sha256(second)

    with pytest.raises(ValueError, match="Requested 101"):
        fast_vision._representative_indices(100, 101, seed=6198)


def test_indexed_dataset_forwards_fixed_source_index_and_epoch(fast_vision):
    class Dataset:
        config = object()

        def get(self, index, epoch):
            return index, epoch

    selected = fast_vision.IndexedDataset(Dataset(), [9, 3, 7])

    assert selected.config is selected.dataset.config
    assert len(selected) == 3
    assert selected[0] == (9, 0)
    assert selected.get(1, epoch=4) == (3, 4)


def test_task_datasets_use_matched_validation_protocol(monkeypatch, fast_vision):
    created = {}

    class DatasetConfig:
        def __init__(self, task, **kwargs):
            created[task] = kwargs
            self.task = task

        def build(self, tokenizer):
            assert tokenizer == "tokenizer"
            return self

    monkeypatch.setattr(
        fast_vision,
        "PixMoCapDatasetConfig",
        lambda **kwargs: DatasetConfig("caption", **kwargs),
    )
    monkeypatch.setattr(
        fast_vision,
        "PixMoCountDatasetConfig",
        lambda **kwargs: DatasetConfig("count", **kwargs),
    )
    monkeypatch.setattr(
        fast_vision,
        "PixMoPointsDatasetConfig",
        lambda **kwargs: DatasetConfig("points", **kwargs),
    )

    token_ids = object()
    datasets = fast_vision._build_task_datasets(
        "tokenizer",
        token_ids,
        message_format="olmo3_chat",
        max_sequence_length=16384,
        max_crops=8,
        sample_seed=6198,
    )

    assert set(datasets) == {"caption", "count", "points"}
    assert all(config["split"] == "validation" for config in created.values())
    assert all(config["loss_token_weighting"] == "none" for config in created.values())
    assert all(config["message_format"] == "olmo3_chat" for config in created.values())
    assert all(config["token_ids"] is token_ids for config in created.values())
    assert created["caption"]["mode"] == "caption"
    assert created["caption"]["max_sequence_length"] == 16384
    assert created["count"]["counting"] is True
    assert created["points"]["kind"] == "basic"
    assert created["points"]["counting"] is False


def test_task_specs_use_stable_task_specific_permutations(fast_vision):
    datasets = {name: list(range(100)) for name in fast_vision.TASK_NAMES}
    specs = fast_vision._build_task_specs(
        datasets,
        fast_vision.TASK_NAMES,
        examples=16,
        sample_seed=6198,
    )

    assert [spec.name for spec in specs] == list(fast_vision.TASK_NAMES)
    for spec in specs:
        assert list(spec.indices) == fast_vision._representative_indices(
            100,
            16,
            seed=6198 + fast_vision.TASK_SEED_OFFSETS[spec.name],
        )
    assert len({tuple(spec.indices) for spec in specs}) == 3


def test_runtime_validation_checks_ep_and_batch_inputs(fast_vision):
    args = argparse.Namespace(
        examples=512,
        ep_degree=8,
        max_sequence_length=16384,
        rank_batch_instances=2,
        max_crops=8,
        checkpoint_load_threads=8,
        tasks=["caption", "count", "points"],
    )
    fast_vision._validate_args(args, world_size=8)

    args.ep_degree = 3
    with pytest.raises(ValueError, match="must be divisible"):
        fast_vision._validate_args(args, world_size=8)


def test_atomic_json_writer_replaces_temporary_file(tmp_path, fast_vision):
    output = tmp_path / "nested" / "result.json"
    fast_vision._write_json_atomic(output, {"complete": True})

    assert json.loads(output.read_text()) == {"complete": True}
    assert not output.with_suffix(".json.tmp").exists()


def test_evaluate_task_reduces_pre_compute_response_weight_once(monkeypatch, tmp_path, fast_vision):
    class Dataset:
        def __len__(self):
            return 10

    class Loader:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __iter__(self):
            yield {"input_ids": torch.zeros((1, 4), dtype=torch.long)}

        def __len__(self):
            return 1

    class Evaluator:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.ce_loss = type("Metric", (), {"weight": torch.tensor(2.0)})()

        def __iter__(self):
            return iter(Loader())

        def reset_metrics(self):
            pass

        def update_metrics(self, *args):
            del args

        def compute_metrics(self):
            # Mirrors MeanMetric.compute() mutating its internal weight via all-reduce.
            self.ce_loss.weight.mul_(8)
            return {"CE loss": torch.tensor(1.0), "PPL": torch.tensor(2.0)}

    class Output:
        ce_loss = torch.tensor(1.0)
        logits = None

    class TrainModule:
        device = torch.device("cpu")
        dp_process_group = object()

        def eval_batch(self, batch):
            del batch
            return Output()

    reduced = []

    def all_reduce_once(value, device, group=None):
        del device, group
        reduced.append(float(value.item()))
        return value

    monkeypatch.setattr(fast_vision, "MaxSequenceLengthDataset", lambda dataset, *a, **k: dataset)
    monkeypatch.setattr(fast_vision, "MultimodalDataLoader", Loader)
    monkeypatch.setattr(fast_vision, "MultimodalLMEvaluator", Evaluator)
    monkeypatch.setattr(fast_vision, "LMOutputWithLoss", Output)
    monkeypatch.setattr(fast_vision, "move_to_device", lambda batch, device: batch)
    monkeypatch.setattr(fast_vision, "all_reduce_value", all_reduce_once)
    monkeypatch.setattr(fast_vision, "gc_cuda", lambda: None)

    result = fast_vision._evaluate_task(
        TrainModule(),
        fast_vision.TaskSpec("count", Dataset(), [3]),
        collator=None,
        work_dir=tmp_path,
        max_sequence_length=16,
        rank_batch_instances=1,
        sample_seed=6198,
        token_ids=None,
        dp_world_size=1,
        dp_rank=0,
    )

    assert reduced == [2.0]
    assert result["response_token_weight"] == 2.0
    assert result["batches_per_dp_rank"] == 1
    assert result["examples_per_dp_rank"] == 1


def test_checkpoint_comparison_beaker_spec_is_three_matched_ep8_jobs(fast_vision):
    path = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "vision_moe"
        / "eval"
        / "stage2_fast_vision_checkpoint_comparison.yaml"
    )
    with path.open() as spec_file:
        spec = yaml.safe_load(spec_file)

    assert spec["budget"] == "ai2/oe-other"
    assert len(spec["tasks"]) == 3
    checkpoints = []
    outputs = []
    for task in spec["tasks"]:
        arguments = task["arguments"]
        assert arguments[:2] == ["python", "src/scripts/eval/s002_fast_vision.py"]
        checkpoints.append(arguments[arguments.index("--checkpoint") + 1])
        outputs.append(arguments[arguments.index("--output") + 1])
        assert arguments[arguments.index("--examples") + 1] == "512"
        assert arguments[arguments.index("--sample-seed") + 1] == "6198"
        assert arguments[arguments.index("--message-format") + 1] == "olmo3_chat"
        assert arguments[arguments.index("--max-sequence-length") + 1] == "16384"
        assert task["resources"]["gpuCount"] == 8
        assert task["constraints"]["cluster"] == ["ai2/holmes"]
        assert task["context"]["priority"] == "urgent"
        assert task["context"]["minRuntime"] == "8h0m0s"

    assert any("stage1-corrected-clean-32k" in checkpoint for checkpoint in checkpoints)
    assert any(checkpoint.endswith("/step50") for checkpoint in checkpoints)
    assert any(checkpoint.endswith("/step200") for checkpoint in checkpoints)
    assert len(set(outputs)) == 3
