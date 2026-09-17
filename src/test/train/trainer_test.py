import json
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from olmo_core.train.callbacks.evaluator_callback import EvaluatorCallback
from olmo_core.train.callbacks.metric_saver import MetricSaverCallback
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.trainer import Trainer, _metric_value_to_tensor


def test_metric_value_to_tensor_accepts_large_python_int():
    value = 10**20
    tensor = _metric_value_to_tensor(value)

    assert tensor.dtype == torch.float64
    assert tensor.item() == float(value)


@pytest.mark.parametrize("pattern", ["*", "**", "step*"])
def test_checkpoint_globs_ignore_files_and_preserve_order(tmp_path, pattern):
    checkpoints = [tmp_path / "step2", tmp_path / "step10"]
    for checkpoint in checkpoints:
        checkpoint.mkdir()
        (checkpoint / ".metadata").touch()
        (checkpoint / "weights.pt").touch()
    (tmp_path / "config.json").touch()
    (tmp_path / "step-not-a-checkpoint.txt").touch()

    trainer = object.__new__(Trainer)
    trainer.checkpointer = object.__new__(Checkpointer)
    trainer.checkpoints_to_eval = [str(checkpoints[1]), f"{tmp_path}/{pattern}"]

    assert trainer._get_checkpoints_to_eval() == [str(path) for path in checkpoints]


def test_recursive_checkpoint_globs_find_nested_checkpoints(tmp_path):
    checkpoint = tmp_path / "run" / "step1"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").touch()
    (checkpoint / "weights.pt").touch()
    (tmp_path / "run" / "config.json").touch()

    trainer = object.__new__(Trainer)
    trainer.checkpointer = object.__new__(Checkpointer)
    trainer.checkpoints_to_eval = [f"{tmp_path}/**"]

    assert trainer._get_checkpoints_to_eval() == [str(checkpoint)]


def test_recursive_full_checkpoints_exclude_nested_model_state(tmp_path):
    checkpoints = [tmp_path / "step2", tmp_path / "step10"]
    for checkpoint in checkpoints:
        (checkpoint / "train").mkdir(parents=True)
        (checkpoint / "train/rank0.pt").touch()
        (checkpoint / "model_and_optim").mkdir()
        (checkpoint / "model_and_optim/.metadata").touch()
        (checkpoint / Checkpointer.METADATA_FNAME).touch()

    trainer = object.__new__(Trainer)
    trainer.checkpointer = object.__new__(Checkpointer)
    trainer.checkpoints_to_eval = [f"{tmp_path}/**"]
    assert trainer._get_checkpoints_to_eval() == [str(path) for path in checkpoints]

    trainer.checkpoints_to_eval = [f"{tmp_path}/**/model_and_optim"]
    assert trainer._get_checkpoints_to_eval() == sorted(
        str(path / "model_and_optim") for path in checkpoints
    )

    explicit = [str(checkpoints[0]), str(checkpoints[0] / "model_and_optim")]
    trainer.checkpoints_to_eval = explicit
    assert trainer._get_checkpoints_to_eval() == explicit


@pytest.mark.parametrize("explicit", [False, True])
def test_checkpoint_selection_preserves_missing_path_errors(tmp_path, explicit):
    trainer = object.__new__(Trainer)
    trainer.checkpointer = object.__new__(Checkpointer)
    trainer.checkpoints_to_eval = [str(tmp_path / "missing") if explicit else f"{tmp_path}/step*"]

    with pytest.raises(FileNotFoundError):
        trainer._get_checkpoints_to_eval()


def test_explicit_checkpoint_file_is_still_an_error(tmp_path):
    file = tmp_path / "weights.pt"
    file.touch()
    trainer = object.__new__(Trainer)
    trainer.checkpointer = object.__new__(Checkpointer)
    trainer.checkpoints_to_eval = [str(file)]

    with pytest.raises(RuntimeError, match="Not a directory"):
        trainer._get_checkpoints_to_eval()


@pytest.fixture
def checkpoint_eval_trainer(tmp_path, monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.save_folder = str(tmp_path / "results")
    trainer.checkpointer = Checkpointer(work_dir=tmp_path / "work", save_overwrite=False)
    trainer.device = torch.device("cpu")
    trainer.async_bookkeeping = False
    trainer.data_loader = SimpleNamespace(load_state_dict=Mock(), global_batch_size=999)
    trainer.train_module = SimpleNamespace(pre_train=Mock())
    trainer._metrics = OrderedDict()
    trainer._metrics_reduce_type = {}
    trainer._bookkeeping_queue = defaultdict(OrderedDict)
    # Deliberately stale state: an unknown checkpoint must not inherit these counters.
    trainer.global_step = 999
    trainer.global_train_tokens_seen = 9999
    trainer.global_train_petaflops = 99.0
    trainer.epoch = 9

    states = {}

    def checkpoint(name, step):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        (path / ".metadata").touch()
        states[str(path)] = (
            None
            if step is None
            else {
                "global_step": step,
                "global_train_tokens_seen": step * 7 + 3,
                "global_train_petaflops": step / 10,
                "epoch": 4,
                "data_loader": {},
                "world_size": 2,  # Skip RNG restoration in this single-rank test.
            }
        )
        return str(path)

    def load(path, train_module, *, load_trainer_state, **kwargs):
        del train_module, kwargs
        return states[path] if load_trainer_state is not False else None

    load_mock = Mock(side_effect=load)
    monkeypatch.setattr(trainer.checkpointer, "load", load_mock)
    evaluator = EvaluatorCallback()
    saver = MetricSaverCallback(save_interval=1)
    trainer.callbacks = {"evaluator": evaluator, "saver": saver}
    for callback in trainer.callbacks.values():
        callback.trainer = trainer

    def post_checkpoint_loaded(path):
        EvaluatorCallback.post_checkpoint_loaded(evaluator, path)
        trainer.record_metric("test/load_sequence", load_mock.call_count)
        trainer.record_metric("test/load_tokens", trainer.global_train_tokens_seen)
        # Exercise separate load/evaluation fragments with the real MetricSaver.
        trainer._log_metrics()

    monkeypatch.setattr(evaluator, "post_checkpoint_loaded", post_checkpoint_loaded)
    monkeypatch.setattr(
        evaluator,
        "perform_eval",
        lambda: trainer.record_metric("eval/test/score", load_mock.call_count),
    )
    return trainer, checkpoint, load_mock


@pytest.mark.parametrize("compute_flops", [False, True])
@pytest.mark.parametrize(
    "names,train_steps,logging_steps,reported_steps",
    [
        (["model-a", "model-b"], [None, None], [1, 2], [-1, -1]),
        (
            ["step100/model_and_optim", "step200/model_and_optim"],
            [None, None],
            [100, 200],
            [100, 200],
        ),
        (["full", "unnamed", "later"], [100, None, 200], [100, 101, 200], [100, -1, 200]),
        (
            ["run-a/step100/model_and_optim", "run-b/step100"],
            [None, 100],
            [100, 101],
            [100, 100],
        ),
        (["newer", "older"], [200, 100], [200, 201], [200, 100]),
        (["step999"], [100], [100], [100]),  # Saved state is authoritative.
        (["initial", "unnamed"], [0, None], [0, 1], [0, -1]),
    ],
)
def test_checkpoint_evaluation_uses_distinct_monotonic_metric_steps(
    checkpoint_eval_trainer, compute_flops, names, train_steps, logging_steps, reported_steps
):
    trainer, checkpoint, load_mock = checkpoint_eval_trainer
    trainer.checkpoints_to_eval = [checkpoint(name, step) for name, step in zip(names, train_steps)]
    if compute_flops:
        trainer.train_module.max_sequence_length = 16
        trainer.train_module.num_flops_per_token = lambda _: 1e12

    trainer.eval_checkpoints()

    results_dir = trainer.checkpointer.work_dir.parent / "results"
    snapshots = {
        int(path.stem.removeprefix("metrics_step")): json.loads(path.read_text())
        for path in results_dir.glob("metrics_step*.json")
    }
    assert sorted(snapshots) == logging_steps
    for index, (step, train_step, reported_step) in enumerate(
        zip(logging_steps, train_steps, reported_steps), start=1
    ):
        snapshot = snapshots[step]
        tokens = train_step * 7 + 3 if train_step is not None else 0
        petaflops = tokens / 1000 if compute_flops else (train_step or 0) / 10
        assert snapshot["eval/test/score"] == index
        assert snapshot["test/load_sequence"] == index
        assert snapshot["checkpoint/load_duration_s"] >= 0
        assert snapshot["checkpoint/train_step"] == reported_step
        assert snapshot["checkpoint/trainer_state_loaded"] == int(train_step is not None)
        assert snapshot["test/load_tokens"] == snapshot["throughput/total tokens"] == tokens
        assert snapshot["throughput/total petaflops"] == pytest.approx(petaflops)
    assert json.loads((results_dir / "metrics.json").read_text()) == snapshots[logging_steps[-1]]
    assert all(call.kwargs["load_optim_state"] is False for call in load_mock.call_args_list)
    assert trainer.checkpointer.save_overwrite is False


@pytest.mark.parametrize(
    "name,logging_step,reported_step", [("step100", 100, 100), ("model", 1, -1)]
)
def test_checkpoint_evaluation_without_trainer_state_resets_counters(
    checkpoint_eval_trainer, name, logging_step, reported_step
):
    trainer, checkpoint, _ = checkpoint_eval_trainer
    trainer.checkpoints_to_eval = [checkpoint(name, 200)]
    trainer.load_trainer_state = False

    trainer.eval_checkpoints()

    snapshot = json.loads(
        (
            trainer.checkpointer.work_dir.parent / f"results/metrics_step{logging_step}.json"
        ).read_text()
    )
    assert snapshot["checkpoint/train_step"] == reported_step
    assert snapshot["checkpoint/trainer_state_loaded"] == 0
    assert snapshot["test/load_tokens"] == snapshot["throughput/total tokens"] == 0
    assert snapshot["throughput/total petaflops"] == 0
    assert trainer.global_train_tokens_seen == trainer.global_train_petaflops == 0
    assert trainer.epoch == 1
    trainer.data_loader.load_state_dict.assert_not_called()


@pytest.mark.parametrize("load_trainer_state", [None, False])
def test_regular_weights_only_load_preserves_existing_trainer_state(
    checkpoint_eval_trainer, load_trainer_state
):
    trainer, checkpoint, _ = checkpoint_eval_trainer
    trainer.load_checkpoint(checkpoint("step100", None), load_trainer_state=load_trainer_state)

    assert trainer.global_step == 999
    assert trainer.global_train_tokens_seen == 9999
    assert trainer.global_train_petaflops == 99.0
    assert trainer.epoch == 9
    snapshot = json.loads(
        (trainer.checkpointer.work_dir.parent / "results/metrics_step999.json").read_text()
    )
    assert "checkpoint/train_step" not in snapshot
    assert "checkpoint/trainer_state_loaded" not in snapshot


def test_canceling_bookkeeping_op_can_remove_itself_from_queue():
    trainer = object.__new__(Trainer)
    trainer.bookkeeping_soft_timeout = 30
    trainer._bookkeeping_queue = defaultdict(OrderedDict)
    trainer._multi_thread_pool = ThreadPoolExecutor(max_workers=1)
    trainer._error = None

    op_name = "test_op"
    op_id = "existing"
    future: Future[None] = Future()
    trainer._bookkeeping_queue[op_name][op_id] = future
    future.add_done_callback(lambda _: trainer._bookkeeping_queue[op_name].pop(op_id, None))

    try:
        trainer.run_bookkeeping_op(
            lambda: None,
            op_name=op_name,
            allow_multiple=False,
            distributed=False,
        )
        trainer._join_bookkeeping_ops(timeout=1)
        assert trainer._error is None
        assert not trainer._bookkeeping_queue[op_name]
    finally:
        trainer.multi_thread_pool.shutdown(wait=True)
