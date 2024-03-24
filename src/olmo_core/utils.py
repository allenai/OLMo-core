import os
import time
from typing import Callable

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
