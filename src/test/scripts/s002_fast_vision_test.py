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
        lambda **kwargs: DatasetConfig("point_count" if kwargs["counting"] else "points", **kwargs),
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

    assert set(datasets) == {"caption", "count", "points", "point_count"}
    assert all(config["split"] == "validation" for config in created.values())
    assert all(config["loss_token_weighting"] == "none" for config in created.values())
    assert all(config["message_format"] == "olmo3_chat" for config in created.values())
    assert all(config["token_ids"] is token_ids for config in created.values())
    assert created["caption"]["mode"] == "caption"
    assert created["caption"]["max_sequence_length"] == 16384
    assert created["count"]["counting"] is True
    assert created["points"]["kind"] == "basic"
    assert created["points"]["counting"] is False
    assert created["point_count"]["kind"] == "basic"
    assert created["point_count"]["counting"] is True


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
    by_name = {spec.name: tuple(spec.indices) for spec in specs}
    assert by_name["point_count"] == by_name["points"]

    mismatched = dict(datasets, point_count=list(range(99)))
    with pytest.raises(ValueError, match="same source index space"):
        fast_vision._build_task_specs(
            mismatched,
            ["point_count"],
            examples=16,
            sample_seed=6198,
        )


def test_numeric_count_protocol_requires_distinct_single_token_candidates(fast_vision):
    class Tokenizer:
        eos_token_id = 99

        @staticmethod
        def encode(text, add_special_tokens=False):
            assert add_special_tokens is False
            if text == "Counting":
                return [40, 41]
            if text == "<points":
                return [50, 51]
            return [int(text) + 10]

    protocol = fast_vision._numeric_count_token_protocol(Tokenizer())

    assert tuple(protocol.values) == tuple(range(2, 11))
    assert tuple(protocol.candidate_token_ids) == tuple(range(12, 21))
    assert protocol.eos_token_id == 99
    assert protocol.counting_prefix_token_id == 40
    assert protocol.points_prefix_token_id == 50


def test_numeric_count_statistics_follow_response_mask_row_order(fast_vision):
    protocol = fast_vision.NumericCountTokenProtocol(
        values=(2, 3),
        candidate_token_ids=(2, 3),
        eos_token_id=1,
        counting_prefix_token_id=4,
        points_prefix_token_id=5,
    )
    labels = torch.tensor(
        [
            [-100, 2, -100, 1, -100, -100],
            [3, -100, -100, -100, 1, -100],
        ]
    )
    loss_masks = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    # Rows are exactly the MultimodalLM response-only order: row 0 digit/EOS, then row 1.
    response_logits = torch.tensor(
        [
            [-2.0, -2.0, 4.0, 1.0, 2.0, 0.0],
            [-2.0, 5.0, -2.0, -2.0, -2.0, -2.0],
            [-2.0, -2.0, 5.0, 2.0, 0.0, 1.0],
            [-2.0, 4.0, -2.0, -2.0, -2.0, -2.0],
        ]
    )

    statistics = fast_vision._numeric_count_batch_statistics(
        {"labels": labels, "loss_masks": loss_masks},
        response_logits,
        protocol,
    )
    result = fast_vision._numeric_count_metrics(statistics, protocol)

    expected_candidate_nll = torch.nn.functional.cross_entropy(
        response_logits[[0, 2]][:, [2, 3]],
        torch.tensor([0, 1]),
    ).item()
    expected_digit_nll = torch.nn.functional.cross_entropy(
        response_logits[[0, 2]],
        torch.tensor([2, 3]),
    ).item()
    expected_eos_nll = torch.nn.functional.cross_entropy(
        response_logits[[1, 3]],
        torch.tensor([1, 1]),
    ).item()
    metrics = result["metrics"]
    assert metrics["candidate-normalized first-token NLL"] == pytest.approx(
        expected_candidate_nll, abs=1e-6
    )
    assert metrics["candidate top-1 accuracy"] == 0.5
    assert metrics["raw digit NLL"] == pytest.approx(expected_digit_nll, abs=1e-6)
    assert metrics["raw teacher-forced EOS NLL"] == pytest.approx(expected_eos_nll, abs=1e-6)
    assert result["target_histogram"] == {"2": 1, "3": 1}
    assert result["candidate_top1_prediction_histogram"] == {"2": 2, "3": 0}

    ignored_at_response = labels.clone()
    ignored_at_response[0, 1] = -100
    with pytest.raises(ValueError, match="must not have ignored labels"):
        fast_vision._numeric_count_batch_statistics(
            {"labels": ignored_at_response, "loss_masks": loss_masks},
            response_logits,
            protocol,
        )


def test_runtime_validation_checks_ep_and_batch_inputs(fast_vision):
    assert fast_vision.DEFAULT_RANK_BATCH_INSTANCES == 1
    args = argparse.Namespace(
        examples=512,
        ep_degree=8,
        max_sequence_length=16384,
        rank_batch_instances=1,
        max_crops=8,
        checkpoint_load_threads=8,
        tasks=["caption", "count", "points", "point_count"],
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


def test_count_discriminator_beaker_spec_is_matched_safe_and_pinned(fast_vision):
    path = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "vision_moe"
        / "eval"
        / "stage2_count_discriminator_checkpoint_comparison.yaml"
    )
    with path.open() as spec_file:
        spec = yaml.safe_load(spec_file)

    expected_hosts = [
        "holmes-cs-aus-520.reviz.ai2.in",
        "holmes-cs-aus-516.reviz.ai2.in",
        "holmes-cs-aus-505.reviz.ai2.in",
    ]
    assert spec["version"] == "v2"
    assert spec["budget"] == "ai2/oe-other"
    assert len(spec["tasks"]) == 3
    checkpoints = []
    outputs = []
    git_refs = set()
    for task in spec["tasks"]:
        arguments = task["arguments"]
        checkpoints.append(arguments[arguments.index("--checkpoint") + 1])
        outputs.append(arguments[arguments.index("--output") + 1])
        task_start = arguments.index("--tasks") + 1
        task_end = arguments.index("--examples")
        assert arguments[task_start:task_end] == ["count", "point_count"]
        assert arguments[arguments.index("--examples") + 1] == "512"
        assert arguments[arguments.index("--sample-seed") + 1] == "6198"
        assert arguments[arguments.index("--rank-batch-instances") + 1] == "1"
        assert task["resources"]["gpuCount"] == 8
        assert task["constraints"]["hostname"] == expected_hosts
        assert task["context"]["priority"] == "urgent"
        assert task["context"]["minRuntime"] == "8h0m0s"
        env = {item["name"]: item.get("value") for item in task["envVars"]}
        git_refs.add(env["GIT_REF"])
        assert env["GIT_BRANCH"] == "vision-moe"
        assert env["TMPDIR"] == "/results"

    assert any(checkpoint.endswith("/step32000") for checkpoint in checkpoints)
    assert any(checkpoint.endswith("/step50") for checkpoint in checkpoints)
    assert any(checkpoint.endswith("/step200") for checkpoint in checkpoints)
    assert len(set(outputs)) == 3
    assert len(git_refs) == 1
    assert git_refs.pop() == "771c954772413c378e36fc01dc57a3409529eafe"


def test_stage1_parent_count_discriminator_retry_matches_protocol_and_is_isolated(fast_vision):
    root = Path(__file__).resolve().parents[3]
    config_dir = root / "configs" / "vision_moe" / "eval"
    with (config_dir / "stage1_parent_count_discriminator_retry.yaml").open() as spec_file:
        retry_spec = yaml.safe_load(spec_file)
    with (config_dir / "stage2_count_discriminator_checkpoint_comparison.yaml").open() as spec_file:
        comparison_spec = yaml.safe_load(spec_file)

    expected_hosts = [
        "holmes-cs-aus-520.reviz.ai2.in",
        "holmes-cs-aus-516.reviz.ai2.in",
        "holmes-cs-aus-505.reviz.ai2.in",
    ]
    assert retry_spec["version"] == "v2"
    assert retry_spec["budget"] == "ai2/oe-other"
    assert len(retry_spec["tasks"]) == 1

    retry = retry_spec["tasks"][0]
    parent = comparison_spec["tasks"][0]
    retry_arguments = retry["arguments"]
    parent_arguments = parent["arguments"]
    assert retry_arguments[retry_arguments.index("--checkpoint") + 1] == (
        "/weka/oe-training-default/rustin/experiments/vision-moe/checkpoints/"
        "s002-stage1-corrected-clean-32k-b300-20260807/step32000"
    )
    for option in (
        "--checkpoint",
        "--tasks",
        "--examples",
        "--sample-seed",
        "--message-format",
        "--max-sequence-length",
        "--rank-batch-instances",
        "--max-crops",
        "--tokenizer",
        "--hf-cache",
        "--work-dir",
    ):
        retry_start = retry_arguments.index(option) + 1
        parent_start = parent_arguments.index(option) + 1
        if option == "--tasks":
            retry_end = retry_arguments.index("--examples")
            parent_end = parent_arguments.index("--examples")
        else:
            retry_end = retry_start + 1
            parent_end = parent_start + 1
        assert retry_arguments[retry_start:retry_end] == parent_arguments[parent_start:parent_end]

    retry_output = retry_arguments[retry_arguments.index("--output") + 1]
    parent_output = parent_arguments[parent_arguments.index("--output") + 1]
    assert retry_output.endswith(
        "fast-vision-count-discriminator-512-olmo3-chat-20260810-retry.json"
    )
    assert retry_output != parent_output
    assert retry["resources"]["gpuCount"] == 8
    assert retry["constraints"]["hostname"] == expected_hosts
    assert retry["context"]["priority"] == "urgent"
    assert retry["context"]["minRuntime"] == "8h0m0s"
    env = {item["name"]: item.get("value") for item in retry["envVars"]}
    assert env["GIT_REF"] == "771c954772413c378e36fc01dc57a3409529eafe"
    assert env["GIT_BRANCH"] == "vision-moe"
    assert env["TMPDIR"] == "/results"
