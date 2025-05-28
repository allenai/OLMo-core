import logging

import pytest
import torch
import torch.distributed as dist

from olmo_core.distributed.utils import is_distributed

log = logging.getLogger(__name__)

has_cuda = torch.cuda.is_available()
has_multiple_gpus = has_cuda and torch.cuda.device_count() > 1
has_flash_attn = False
has_torchao = False
has_grouped_gemm = False

try:
    import flash_attn  # type: ignore

    has_flash_attn = True
    del flash_attn
except ModuleNotFoundError:
    pass

try:
    import torchao  # type: ignore

    has_torchao = True
    del torchao
except ModuleNotFoundError:
    pass

try:
    import grouped_gemm  # type: ignore

    has_grouped_gemm = True
    del grouped_gemm
except ModuleNotFoundError:
    pass


GPU_MARKS = (pytest.mark.gpu, pytest.mark.skipif(not has_cuda, reason="Requires a GPU"))


def requires_gpu(func):
    for mark in GPU_MARKS:
        func = mark(func)
    return func


MULTI_GPU_MARKS = (
    pytest.mark.gpu,
    pytest.mark.skipif(not has_multiple_gpus, reason="Requires multiple GPUs"),
)


def requires_multi_gpu(func):
    for mark in MULTI_GPU_MARKS:
        func = mark(func)
    return func


FLASH_MARKS = (
    pytest.mark.gpu,
    pytest.mark.skipif(not has_flash_attn, reason="Requires flash-attn"),
)


def requires_flash_attn(func):
    for mark in FLASH_MARKS:
        func = mark(func)
    return func


GROUPED_GEMM_MARKS = (
    pytest.mark.gpu,
    pytest.mark.skipif(not has_grouped_gemm, reason="Requires grouped_gemm"),
)


def requires_grouped_gemm(func):
    for mark in GROUPED_GEMM_MARKS:
        func = mark(func)
    return func


INIT_DEVICES = [
    pytest.param(torch.device("meta"), id="device=meta"),
    pytest.param(torch.device("cpu"), id="device=CPU"),
    pytest.param(
        torch.device("cuda"),
        id="device=CUDA",
        marks=GPU_MARKS,
    ),
]

DEVICES = [
    pytest.param(torch.device("cpu"), id="device=CPU"),
    pytest.param(
        torch.device("cuda"),
        id="device=CUDA",
        marks=GPU_MARKS,
    ),
]


BACKENDS = [
    pytest.param("gloo", id="backend=GLOO"),
    pytest.param(
        "cuda:nccl,cpu:gloo",
        id="backend=NCCL",
        marks=MULTI_GPU_MARKS,
    ),
]

LOW_PRECISION_DTYPES = [
    pytest.param(torch.float16, id="dtype=float16"),
    pytest.param(
        torch.bfloat16,
        id="dtype=bfloat16",
        marks=GPU_MARKS,
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
