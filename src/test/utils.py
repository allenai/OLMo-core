from os import PathLike
from pathlib import Path
from typing import Any, List, Tuple, Type, Union

import numpy as np
import pytest
import torch

has_cuda = torch.cuda.is_available()
has_flash_attn = False
has_megablocks = False

try:
    import flash_attn  # type: ignore

    has_flash_attn = True
    del flash_attn
except ModuleNotFoundError:
    pass

try:
    import megablocks  # type: ignore

    has_megablocks = True
    del megablocks
except ModuleNotFoundError:
    pass


GPU_MARKS = (pytest.mark.gpu, pytest.mark.skipif(not has_cuda, reason="Requires a GPU"))


def requires_gpu(func):
    for mark in GPU_MARKS:
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


MEGABLOCKS_MARKS = (
    pytest.mark.gpu,
    pytest.mark.skipif(not has_megablocks, reason="Requires megablocks"),
)


def requires_megablocks(func):
    for mark in MEGABLOCKS_MARKS:
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


Mmaps = List[Tuple[Union[Path, PathLike[Any], str], Any]]


def mk_mmaps(
    tmp_path: Path,
    prefix: str,
    num_files: int,
    size: int,
    dtype: Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]] = np.uint32,
    eos: int = 0,
    seq_length: int = 4,
    seed: int = 42,
) -> Mmaps:
    mmaps: Mmaps = []
    for i in range(num_files):
        filepath = f"{tmp_path}/{prefix}_{i}.npy"
        np.random.seed(seed)
        data = np.random.randint(1, np.iinfo(dtype).max, size=size, dtype=dtype)
        data = np.append(
            np.insert(data, np.arange(seq_length + 1, len(data), seq_length), eos), eos
        )
        mm = np.memmap(filepath, mode="w+", dtype=dtype, shape=(len(data),))
        mm[:] = data
        mm.flush()
        mmaps.append((Path(filepath), data))

    return mmaps
