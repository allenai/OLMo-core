import pytest
import torch

has_cuda = torch.cuda.is_available()

GPU_MARKS = (pytest.mark.gpu, pytest.mark.skipif(not has_cuda, reason="Requires a GPU"))


def requires_gpu(func):
    for mark in GPU_MARKS:
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

LOW_PRECISION_DTYPES = [
    pytest.param(torch.float16, id="dtype=float16"),
    pytest.param(
        torch.bfloat16,
        id="dtype=bfloat16",
        marks=GPU_MARKS,
    ),
]


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
