import dataclasses
import gc
import os
import time
from enum import Enum
from typing import Any, Callable, Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel

from .exceptions import OLMoEnvironmentError

OLMO_NUM_THREADS_ENV_VAR = "OLMO_NUM_THREADS"


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


ShapeType = Union[torch.Size, List[int], Tuple[int, ...]]

# torch.float8 formats require 2.1; we do not support these dtypes on earlier versions
_float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
_float8_e5m2 = getattr(torch, "float8_e5m2", None)

TORCH_TO_NP_DTYPES = {
    torch.int64: np.int64,
    torch.float32: np.float32,
    torch.int32: np.int32,
    # XXX: This is ok because both have the same width
    torch.bfloat16: np.float16,
    torch.float16: np.float16,
    torch.int16: np.int16,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.bool: bool,
    torch.float64: np.float64,
    # XXX: This is ok because both have the same width and byteswap is a no-op anyway
    _float8_e4m3fn: np.uint8,
    _float8_e5m2: np.uint8,
}

TORCH_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    # "U64": torch.uint64,
    "I32": torch.int32,
    # "U32": torch.uint32,
    "I16": torch.int16,
    # "U16": torch.uint16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": _float8_e4m3fn,
    "F8_E5M2": _float8_e5m2,
}


TORCH_DTYPE_TO_STR = {v: k for k, v in TORCH_DTYPES.items()}


def default_thread_count() -> int:
    env_val = os.environ.get(OLMO_NUM_THREADS_ENV_VAR)
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            raise OLMoEnvironmentError(
                f"Invalid value for {OLMO_NUM_THREADS_ENV_VAR} environment variable ('{env_val}')"
            )
    else:
        return min(16, (os.cpu_count() or 1) + 4)


def wait_for(condition: Callable[[], bool], description: str, timeout: float = 10.0):
    """Wait for the condition function to return True."""
    start_time = time.monotonic()
    while not condition():
        time.sleep(0.5)
        if time.monotonic() - start_time > timeout:
            raise TimeoutError(f"{description} timed out")


def apply_to_tensors(fn, container: Any) -> None:
    """
    Recursively apply ``fn`` to all tensors in a container.
    """
    if isinstance(container, torch.Tensor):
        fn(container)
    elif isinstance(container, (list, tuple, set)):
        for x in container:
            apply_to_tensors(fn, x)
    elif isinstance(container, dict):
        for k, v in container.items():
            apply_to_tensors(fn, k)
            apply_to_tensors(fn, v)
    elif hasattr(container, "__dataclass_fields__"):
        for f in dataclasses.fields(container):
            name = f.name
            apply_to_tensors(fn, getattr(container, name))
    elif isinstance(container, BaseModel):
        apply_to_tensors(fn, container.model_dump())
    elif hasattr(container, "__next__"):
        for x in container:
            apply_to_tensors(fn, x)


def get_default_device() -> torch.device:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def seed_all(seed: int):
    """Seed all rng objects."""
    import random

    import numpy as np

    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)


def get_grad_norm(params: Iterable[nn.Parameter], norm_type: float) -> torch.Tensor:
    """
    Return the gradient norm of parameters, where the gradients are viewed as a single vector.

    The returned norm is in FP32 even if parameters/gradients are in a low precision. This is because the downstream
    use of this return value is a reduction across ranks.
    """
    grads = [param.grad for param in params if param.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    grad_dtypes = {grad.dtype for grad in grads}
    if len(grad_dtypes) != 1:
        raise ValueError(f"Requires uniform dtype across all gradients but got {grad_dtypes}")
    # Compute the gradient norm in FP32, where we treat the gradients as a
    # single vector
    grad_norm = torch.linalg.vector_norm(
        torch.stack(
            [torch.linalg.vector_norm(grad.detach(), norm_type, dtype=torch.float32) for grad in grads],
        ),
        norm_type,
        dtype=torch.float32,
    )
    return grad_norm


def same_storage(x: torch.Tensor, y: torch.Tensor) -> bool:
    """
    Check if two tensors share the same storage.
    """
    x_ptrs = set(e.data_ptr() for e in x.view(-1))
    y_ptrs = set(e.data_ptr() for e in y.view(-1))
    return (x_ptrs <= y_ptrs) or (y_ptrs <= x_ptrs)


def gc_cuda():
    """
    Run CUDA garbage collection.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def alloc_storage(tensor: torch.Tensor, size: torch.Size) -> None:
    """
    Allocate storage for ``tensor`` with the given size.

    Returns ``True`` if this method allocated storage and ``False`` if the storage was already allocated.
    """
    already_allocated = tensor._typed_storage()._size() == size.numel()
    if not already_allocated:
        tensor_storage_size = tensor._typed_storage()._size()
        if tensor_storage_size != 0:
            raise RuntimeError(f"Tensor storage should have been resized to be 0 but got {tensor_storage_size}")
        tensor._typed_storage()._resize_(size.numel())


@torch.no_grad()
def free_storage(tensor: torch.Tensor) -> None:
    """
    Frees the underlying storage of ``tensor``.

    Returns ``True`` if the method freed the storage and ``False`` if the storage was already freed.
    """
    already_freed = tensor._typed_storage()._size() == 0
    if not already_freed:
        if tensor.storage_offset() != 0:
            raise RuntimeError(
                "Freeing a tensor's storage is unsafe when it is not the sole occupant\n"
                f"storage offset: {tensor.storage_offset()}\n"
                f"storage size: {tensor._typed_storage()._size()}\n"
                f"tensor shape: {tensor.shape}",
            )
        tensor._typed_storage()._resize_(0)
