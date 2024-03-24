import os
import pickle
import time
from typing import Any, Callable

import numpy as np
import torch

from .exceptions import OLMoEnvironmentError

OLMO_NUM_THREADS_ENV_VAR = "OLMO_NUM_THREADS"

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


def serialize_to_tensor(x: Any) -> torch.Tensor:
    serialized_bytes = pickle.dumps(x)
    return torch.frombuffer(bytearray(serialized_bytes), dtype=torch.uint8)


def deserialize_from_tensor(data: torch.Tensor) -> Any:
    assert data.dtype == torch.uint8
    return pickle.loads(bytearray([int(x.item()) for x in data.flatten()]))
