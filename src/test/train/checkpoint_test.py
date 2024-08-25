import os

import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank
from olmo_core.train.checkpoint import Checkpointer

from ..distributed.utils import run_distributed_test


def run_checkpointer_with_local_dir(dir, model_factory):
    os.environ["OLMO_SHARED_FS"] = "1"

    checkpointer = Checkpointer()
    model = model_factory()
    optim = torch.optim.AdamW(model.parameters())

    # Save checkpoint.
    checkpointer.save(dir, model, optim, {"rank": get_rank()})
    assert (dir / "train").is_dir()
    assert (dir / "train" / "rank0.pt").is_file()
    assert (dir / "train" / "rank1.pt").is_file()
    assert (dir / "model_and_optim").is_dir()

    # Load checkpoint.
    train_state = checkpointer.load(dir, model, optim)
    assert train_state is not None
    assert train_state["rank"] == get_rank()


def test_checkpointer_with_local_dir(tmp_path, tiny_model_factory):
    run_distributed_test(run_checkpointer_with_local_dir, func_args=(tmp_path, tiny_model_factory))


def run_async_checkpointer_with_local_dir(dir, model_factory):
    os.environ["OLMO_SHARED_FS"] = "1"

    checkpointer = Checkpointer(process_group=dist.new_group())
    model = model_factory()
    optim = torch.optim.AdamW(model.parameters())

    # Save checkpoint.
    future = checkpointer.save_async(dir, model, optim, {"rank": get_rank()})
    future.result()

    assert (dir / "train").is_dir()
    assert (dir / "train" / "rank0.pt").is_file()
    assert (dir / "train" / "rank1.pt").is_file()
    assert (dir / "model_and_optim").is_dir()

    # Load checkpoint.
    train_state = checkpointer.load(dir, model, optim)
    assert train_state is not None
    assert train_state["rank"] == get_rank()


def test_async_checkpointer_with_local_dir(tmp_path, tiny_model_factory):
    run_distributed_test(
        run_async_checkpointer_with_local_dir, func_args=(tmp_path, tiny_model_factory)
    )
