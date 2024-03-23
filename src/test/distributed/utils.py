import datetime
from typing import Any, Callable, Dict, Optional, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from olmo_core.distributed.utils import is_distributed

BACKENDS = [pytest.param("gloo", id="backend=GLOO")]
DEVICES = [pytest.param(torch.device("cpu"), id="device=CPU")]

if torch.cuda.is_available():
    BACKENDS = [
        pytest.param("gloo", id="backend=GLOO"),
        pytest.param("nccl", id="backend=NCCL", marks=pytest.mark.gpu),
    ]
    DEVICES = [
        pytest.param(torch.device("cpu"), id="device=CPU"),
        pytest.param(torch.device("cuda"), id="device=CUDA", marks=pytest.mark.gpu),
    ]


def get_default_device():
    if is_distributed():
        backend = dist.get_backend()
        if backend == dist.Backend.GLOO:
            return torch.device("cpu")
        elif backend == dist.Backend.NCCL:
            return torch.device("cuda")
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
    func: Callable,
    func_args: Optional[Tuple[Any, ...]] = None,
    func_kwargs: Optional[Dict[str, Any]] = None,
    primary_addr: str = "127.0.0.1",
    primary_port: int = 29500,
):
    assert world_size > 1

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{primary_addr}:{primary_port}",
        world_size=world_size,
        rank=process_rank,
        timeout=datetime.timedelta(seconds=120),
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(int(process_rank))

    try:
        func(*(func_args or []), **(func_kwargs or {}))
    finally:
        dist.destroy_process_group()


def run_distributed_test(
    func: Callable,
    world_size: int = 2,
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
        args=(world_size, backend, func, func_args, func_kwargs),
        nprocs=world_size,
        start_method=start_method,
    )
