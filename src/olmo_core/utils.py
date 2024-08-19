import dataclasses
import gc
import os
import time
import uuid
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn
from pydantic import BaseModel

from .exceptions import OLMoEnvironmentError

OLMO_NUM_THREADS_ENV_VAR = "OLMO_NUM_THREADS"


def generate_uuid() -> str:
    return str(uuid.uuid4())


def get_default_thread_count() -> int:
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
            [
                torch.linalg.vector_norm(grad.detach(), norm_type, dtype=torch.float32)
                for grad in grads
            ],
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
            raise RuntimeError(
                f"Tensor storage should have been resized to be 0 but got {tensor_storage_size}"
            )
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


def get_document_lengths(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    doc_boundaries = torch.cat(
        [
            torch.tensor([-1], dtype=torch.int32),
            (input_ids == eos_token_id).nonzero(as_tuple=True)[0].to(dtype=torch.int32),
            torch.tensor(
                [] if input_ids[-1] == eos_token_id else [input_ids.shape[0] - 1], dtype=torch.int32
            ),
        ]
    )
    return doc_boundaries[1:] - doc_boundaries[:-1]


def get_cumulative_document_lengths(doc_lens: torch.Tensor) -> torch.Tensor:
    """
    Transform a batched tensor of document lengths into a 1D tensor of cumulative document
    lengths for the whole batch.
    """
    return torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=doc_lens.device),
            torch.cumsum(doc_lens.masked_select(doc_lens != 0), 0, dtype=torch.int32),
        ]
    )


def has_flash_attn() -> bool:
    """
    Check if flash-attn is available.
    """
    try:
        import flash_attn  # type: ignore

        del flash_attn
        return True
    except ModuleNotFoundError:
        return False
