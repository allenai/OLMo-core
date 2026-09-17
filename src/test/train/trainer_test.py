from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor

import pytest
import torch

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
