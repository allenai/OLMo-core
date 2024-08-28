import os
import time

import torch
import torch.distributed as dist

from olmo_core.distributed.utils import barrier, get_rank
from olmo_core.io import dir_is_empty, file_exists, is_url, normalize_path
from olmo_core.train.checkpoint import Checkpointer

from ..distributed.utils import run_distributed_test


def run_checkpointer(base_dir, model_factory):
    dir = f"{normalize_path(base_dir)}/{Checkpointer.checkpoint_dirname(10)}"

    if not is_url(dir):
        os.environ["OLMO_SHARED_FS"] = "1"

    checkpointer = Checkpointer()
    model = model_factory()
    optim = torch.optim.AdamW(model.parameters())

    # Save checkpoint.
    checkpointer.save(dir, model, optim, {"rank": get_rank()})
    barrier()

    assert file_exists((f"{dir}/train/rank0.pt"))
    assert file_exists((f"{dir}/train/rank1.pt"))
    assert not dir_is_empty((f"{dir}/model_and_optim"))
    assert checkpointer.dir_is_checkpoint(dir)
    assert list(checkpointer.find_checkpoints(base_dir)) == [(10, dir)]
    assert checkpointer.latest_checkpoint(base_dir) == dir

    # Load checkpoint.
    train_state = checkpointer.load(dir, model, optim)
    assert train_state is not None
    assert train_state["rank"] == get_rank()


def test_checkpointer_with_local_dir(tmp_path, tiny_model_factory):
    run_distributed_test(run_checkpointer, func_args=(tmp_path, tiny_model_factory))


def test_checkpointer_with_remote_dir(s3_checkpoint_dir, tiny_model_factory):
    run_distributed_test(
        run_checkpointer, func_args=(s3_checkpoint_dir, tiny_model_factory), start_method="spawn"
    )


def run_async_checkpointer(dir, model_factory):
    dir = normalize_path(dir)

    if not is_url(dir):
        os.environ["OLMO_SHARED_FS"] = "1"

    checkpointer = Checkpointer(process_group=dist.new_group())
    model = model_factory()
    optim = torch.optim.AdamW(model.parameters())

    # Save checkpoint.
    future = checkpointer.save_async(dir, model, optim, {"rank": get_rank()})
    future.result()
    time.sleep(0.1)  # allow done callback to run.
    barrier()

    assert file_exists((f"{dir}/train/rank0.pt"))
    assert file_exists((f"{dir}/train/rank1.pt"))
    assert not dir_is_empty((f"{dir}/model_and_optim"))
    assert checkpointer.dir_is_checkpoint(dir)

    # Load checkpoint.
    train_state = checkpointer.load(dir, model, optim)
    assert train_state is not None
    assert train_state["rank"] == get_rank()


def test_async_checkpointer_with_local_dir(tmp_path, tiny_model_factory):
    run_distributed_test(run_async_checkpointer, func_args=(tmp_path, tiny_model_factory))


def test_async_checkpointer_with_remote_dir(s3_checkpoint_dir, tiny_model_factory):
    run_distributed_test(
        run_async_checkpointer,
        func_args=(s3_checkpoint_dir, tiny_model_factory),
        start_method="spawn",
    )
