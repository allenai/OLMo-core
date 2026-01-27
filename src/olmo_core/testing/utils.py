import logging
import os

import pytest
import torch

import olmo_core.nn.attention.flash_attn_api as flash_attn_api
import olmo_core.nn.attention.flash_linear_attn_api as flash_linear_attn_api

log = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")

has_cuda = torch.cuda.is_available()
has_multiple_gpus = has_cuda and torch.cuda.device_count() > 1
has_mps = torch.mps.is_available()
compute_capability = torch.cuda.get_device_capability()[0] if has_cuda else None
has_flash_attn_2 = flash_attn_api.has_flash_attn_2()
has_flash_attn_3 = flash_attn_api.has_flash_attn_3()
has_flash_attn_4 = flash_attn_api.has_flash_attn_4()
has_fla = flash_linear_attn_api.has_fla()
has_torchao = False
has_grouped_gemm = False
has_te = False
has_dion = False
has_quack = False


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

try:
    import transformer_engine.pytorch  # type: ignore

    has_te = True
    del transformer_engine
except ImportError:
    pass

try:
    import dion  # type: ignore

    has_dion = True
    del dion
except ImportError:
    pass

try:
    import quack  # type: ignore

    has_quack = True
    del quack
except ImportError:
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


def requires_compute_capability(min_cc: int):
    def decorator(func):
        return pytest.mark.skipif(
            compute_capability is None or compute_capability < min_cc,
            reason=f"Requires compute capability >={min_cc}, device has {compute_capability=}",
        )(func)

    return decorator


FLASH_2_MARKS = (
    pytest.mark.gpu,
    pytest.mark.skipif(not has_flash_attn_2, reason="Requires flash-attn 2"),
)


def requires_flash_attn_2(func):
    for mark in FLASH_2_MARKS:
        func = mark(func)
    return func


FLASH_3_MARKS = (
    pytest.mark.gpu,
    pytest.mark.skipif(not has_flash_attn_3, reason="Requires flash-attn 3"),
)


def requires_flash_attn_3(func):
    for mark in FLASH_3_MARKS:
        func = mark(func)
    return func


FLA_MARKS = (
    pytest.mark.gpu,
    pytest.mark.skipif(not has_fla, reason="Requires flash-linear-attention (fla)"),
)


def requires_fla(func):
    for mark in FLA_MARKS:
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


TE_MARKS = (
    pytest.mark.gpu,
    pytest.mark.skipif(not has_te, reason="Requires Transformer Engine"),
)


def requires_te(func):
    for mark in TE_MARKS:
        func = mark(func)
    return func


DION_MARKS = (
    pytest.mark.gpu,
    pytest.mark.skipif(not has_dion, reason="Requires Dion"),
)


def requires_dion(func):
    for mark in DION_MARKS:
        func = mark(func)
    return func


QUACK_MARKS = (
    pytest.mark.gpu,
    pytest.mark.skipif(not has_quack, reason="Requires Quack"),
)


def requires_quack(func):
    for mark in QUACK_MARKS:
        func = mark(func)
    return func


MPS_MARKS = (pytest.mark.skipif(not has_mps, reason="Requires MPS"),)


def requires_mps(func):
    for mark in MPS_MARKS:
        func = mark(func)
    return func


requires_hf_token = pytest.mark.skipif(
    HF_TOKEN is None,
    reason="HF_TOKEN environment variable not set - required for accessing gated models",
)


INIT_DEVICES = [
    pytest.param(torch.device("meta"), id="device=meta"),
    pytest.param(torch.device("cpu"), id="device=CPU"),
    pytest.param(torch.device("mps"), id="device=MPS", marks=MPS_MARKS),
    pytest.param(
        torch.device("cuda"),
        id="device=CUDA",
        marks=GPU_MARKS,
    ),
]

DEVICES = [
    pytest.param(torch.device("cpu"), id="device=CPU"),
    pytest.param(torch.device("mps"), id="device=MPS", marks=MPS_MARKS),
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
