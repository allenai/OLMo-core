import datetime
import logging
import os
import sys
from typing import Any, Callable, Dict, Optional, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from olmo_core.distributed.utils import (
    OLMO_LOCAL_WORLD_SIZE_ENV_VAR,
    OLMO_NUM_NODES_ENV_VAR,
    is_distributed,
)

from ..utils import (
    DEVICES,
    GPU_MARKS,
    INIT_DEVICES,
    LOW_PRECISION_DTYPES,
    has_cuda,
    requires_gpu,
)

__all__ = [
    "has_cuda",
    "has_multiple_gpus",
    "requires_gpu",
    "requires_multi_gpu",
    "get_default_device",
    "init_process",
    "run_distributed_test",
    "DEVICES",
    "INIT_DEVICES",
    "BACKENDS",
    "LOW_PRECISION_DTYPES",
    "GPU_MARKS",
    "MULTI_GPU_MARKS",
]

has_multiple_gpus = has_cuda and torch.cuda.device_count() > 1

MULTI_GPU_MARKS = (
    pytest.mark.gpu,
    pytest.mark.skipif(not has_multiple_gpus, reason="Requires multiple GPUs"),
)


def requires_multi_gpu(func):
    for mark in MULTI_GPU_MARKS:
        func = mark(func)
    return func


BACKENDS = [
    pytest.param("gloo", id="backend=GLOO"),
    pytest.param(
        "cuda:nccl,cpu:gloo",
        id="backend=NCCL",
        marks=MULTI_GPU_MARKS,
    ),
]


def get_default_device():
    if is_distributed():
        backend = dist.get_backend()
        if dist.Backend.NCCL in backend:
            return torch.device("cuda")
        elif backend == dist.Backend.GLOO:
            return torch.device("cpu")
        else:
            raise NotImplementedError(backend)
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def init_process(
    process_rank: int,
    world_size: int,
    backend: str,
    log_from_all_ranks: bool,
    func: Callable,
    func_args: Optional[Tuple[Any, ...]] = None,
    func_kwargs: Optional[Dict[str, Any]] = None,
    primary_addr: str = "127.0.0.1",
    primary_port: int = 29500,
):
    assert world_size > 1

    old_log_record_factory = logging.getLogRecordFactory()

    def log_record_factory(*args, **kwargs) -> logging.LogRecord:
        record = old_log_record_factory(*args, **kwargs)
        setattr(record, "local_rank", dist.get_rank())
        return record

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "[rank %(local_rank)s] %(asctime)s:%(name)s:%(lineno)s:%(levelname)s: %(message)s"
        )
    )
    logging.setLogRecordFactory(log_record_factory)

    if log_from_all_ranks or process_rank == 0:
        logging.basicConfig(level=logging.DEBUG, handlers=[handler])

    log = logging.getLogger()

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{primary_addr}:{primary_port}",
        world_size=world_size,
        rank=process_rank,
        timeout=datetime.timedelta(seconds=120),
    )

    os.environ.setdefault(OLMO_NUM_NODES_ENV_VAR, "1")
    os.environ.setdefault(OLMO_LOCAL_WORLD_SIZE_ENV_VAR, str(world_size))

    log.info("Starting test...")

    if "nccl" in backend:
        torch.cuda.set_device(int(process_rank))

    try:
        func(*(func_args or []), **(func_kwargs or {}))
    finally:
        dist.destroy_process_group()


def run_distributed_test(
    func: Callable,
    world_size: int = 2,
    log_from_all_ranks: bool = False,
    backend: str = "gloo",
    start_method: Optional[str] = None,
    func_args: Optional[Tuple[Any, ...]] = None,
    func_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    This runs the `func` in a simulated distributed environment.
    """
    if start_method is None:
        start_method = "fork" if backend == "gloo" else "spawn"

    mp.start_processes(
        init_process,
        args=(world_size, backend, log_from_all_ranks, func, func_args, func_kwargs),
        nprocs=world_size,
        start_method=start_method,
    )
