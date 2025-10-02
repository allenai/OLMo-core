import concurrent.futures
from typing import Callable, List, Literal, Optional, Sequence, Type, TypeVar, Union

import numpy as np
import torch

import olmo_core.io as io
from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoEnvironmentError


def _warmup_clients(paths: Sequence[PathOrStr]):
    # Maybe create client up front to work around a threading issue in boto.
    if any(str(p).startswith("s3://") for p in paths):
        io._get_s3_client("s3")

    if any(str(p).startswith("r2://") for p in paths):
        try:
            io._get_s3_client("r2")
        except OLMoEnvironmentError:
            # R2 might not be needed, so ignore this error. We will get an error
            # later if R2 is needed.
            pass

    if any(str(p).startswith("weka://") for p in paths):
        try:
            io._get_s3_client("weka")
        except OLMoEnvironmentError:
            # Weka might not be needed, so ignore this error. We will get an error
            # later if Weka is needed.
            pass


T = TypeVar("T")


def path_map(
    func: Callable[[PathOrStr], T],
    paths: Sequence[PathOrStr],
    *,
    max_workers: Optional[int] = None,
    method: Literal["threads", "processes"] = "threads",
) -> List[T]:
    """
    Call a function on each path in the dataset, returning a list of the results, in order.

    :param func: The function to map to the paths and their indices.
    :param max_workers: The number of workers threads/processes. Set to 0 to execute synchronously
        in the main thread/process.
    :param method: Whether to use multi-threading or multi-processing.

    :returns: The results, in the same order as :data:`paths`.
    """
    if max_workers == 0:
        return [func(path) for path in paths]

    executor_class: Union[
        Type[concurrent.futures.ThreadPoolExecutor],
        Type[concurrent.futures.ProcessPoolExecutor],
    ]
    if method == "threads":
        _warmup_clients(paths)
        executor_class = concurrent.futures.ThreadPoolExecutor
    elif method == "processes":
        executor_class = concurrent.futures.ProcessPoolExecutor
    else:
        raise ValueError(method)

    with executor_class(max_workers=max_workers) as executor:
        futures = [executor.submit(func, path) for path in paths]

    return [future.result() for future in futures]


def format_fname_from_fields(prefix: str, **fields) -> str:
    parts = [prefix]
    for key in sorted(fields):
        value = fields[key]
        if value is not None:
            parts.append(f"{key}{value}")
    return "_".join(parts)


def as_tensor(array: Union[Sequence[int], Sequence[bool]]) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array
    elif isinstance(array, np.ndarray):
        if array.dtype == np.bool_:
            return torch.tensor(array)
        else:
            return torch.tensor(array.astype(np.int_), dtype=torch.long)
    else:
        return torch.tensor(array)
