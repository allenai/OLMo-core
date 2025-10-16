import concurrent.futures
from typing import Callable, List, Literal, Optional, Sequence, Type, TypeVar, Union

import numpy as np
import torch

import olmo_core.io as io
from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoEnvironmentError

from ..types import NumpyUIntTypes
from ..utils import get_rng


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
    Call a function on each path, returning a list of the results, in order.

    :param func: The function to map to the paths and their indices.
    :param max_workers: The number of workers threads/processes. Set to 0 to execute synchronously
        in the main thread/process.
    :param method: Whether to use multi-threading or multi-processing.

    :returns: The results, in the same order as :data:`paths`.
    """
    if max_workers == 0 or len(paths) <= 1:
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


def format_token_count(n: int) -> str:
    if n >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.1f}T"
    elif n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    else:
        return str(n)


def as_ndarray(array: Union[Sequence[int], Sequence[bool]]) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    else:
        return np.array(array)


def as_tensor(array: Union[Sequence[int], Sequence[bool]]) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array
    elif isinstance(array, np.ndarray):
        if array.dtype == np.bool_:
            return torch.tensor(array, device="cpu")
        else:
            return torch.tensor(array.astype(np.int_), dtype=torch.long, device="cpu")
    else:
        return torch.tensor(array, device="cpu")


def calculate_sample_sizes(
    source_sizes: Sequence[int],
    target_ratios: Sequence[float],
    max_repetition_factors: Sequence[float],
    target_size: Optional[int] = None,
) -> np.ndarray:
    """
    Calculate the number of items needed to sample from each source in order to match the target ratios.
    """
    assert len(source_sizes) == len(target_ratios) == len(max_repetition_factors)

    ratios = np.array(target_ratios)
    sizes = np.array(source_sizes)
    max_repetitions = np.array(max_repetition_factors)

    assert (ratios > 0.0).all()
    assert (max_repetitions >= 1.0).all()

    if target_size is None:
        target_size = sizes.sum()

    # Normalize ratios.
    ratios = ratios / ratios.sum()

    # Determine the number of items to sample from each source.
    # This is tricky because the sources may have different sizes, yet we want to stay
    # true to the sampling ratios while minimizing the number of dropped or over-sampled items.
    # To that end, the optimal natural distribution of items over sources is the one that
    # matches the target sampling ratios. We'll call that the 'ideal_sample_sizes'.
    ideal_sample_sizes = target_size * ratios
    # But since the actual (natural) distribution probably differs from the ideal one, it's
    # not possible to match the target ratios without some dropping or oversampling.
    # So we first calculate how much oversampling/repetition is needed per source, and then cap that
    # according to the given `max_repetitions_per_source`.
    max_repetitions_needed = np.maximum(ideal_sample_sizes / sizes, 1.0)
    repetitions_to_use = np.minimum(max_repetitions, max_repetitions_needed)
    # Now we can adjust sizes based on the repetitions needed.
    sizes_to_use = sizes * repetitions_to_use
    # Lastly, we need to adjust the ideal sample sizes down until by the smallest common factor
    # that would result in all sample sizes being less than or equal to the number of items available
    # from the corresponding source. We can calculate that factor by finding the source with the
    # largest relative difference between its available size (number of items after oversampling) and
    # its ideal sample size, and taking that ratio.
    adjustment_factor = min(1.0, (sizes_to_use / ideal_sample_sizes).min())
    actual_sample_sizes = ideal_sample_sizes * adjustment_factor

    # Sanity check.
    # Sample sizes should stay true to target ratios.
    assert np.allclose(ratios, actual_sample_sizes / actual_sample_sizes.sum())
    # And sample sizes shouldn't be larger than the number of items available.
    actual_sample_sizes = actual_sample_sizes.astype(np.uint64)
    assert (actual_sample_sizes <= sizes_to_use).all()

    return actual_sample_sizes


def build_global_indices(
    total_instances: int,
    *,
    sequence_length: int,
    max_sequence_length: int,
    seed: Optional[int],
    dtype: NumpyUIntTypes = np.uint32,
) -> np.ndarray:
    """
    Build global (as opposed to rank-local) instance indices as a numpy array, in a way that
    preserves the order of data when ``max_sequence_length`` is fixed but ``sequence_length`` changes.
    """
    assert total_instances < np.iinfo(dtype).max
    assert max_sequence_length % sequence_length == 0
    chunk_size = max_sequence_length // sequence_length
    # Length of dataset would be calculated incorrectly if this didn't hold.
    assert total_instances % chunk_size == 0

    # NOTE: To guarantee the same data order with `self.max_sequence_length` fixed but `self.sequence_length`
    # changing, we need `self.total_instances // chunk_size` to remain constant.
    # This is ensured by requiring `self.max_sequence_length` is a multiple of `self.sequence_length`
    # and assuming that `self.total_instances` is proportional to `chunk_size`, i.e.
    # if `self.sequence_length` is half of `self.max_sequence_length`, then `self.total_instances`
    # should double. This takes some care when implementing an `InstanceSource` to ensure that
    # excess tokens are dropped in a way that respects `self.max_sequence_length`, not `self.sequence_length`.
    chunk_indices = np.arange(total_instances // chunk_size, dtype=dtype)

    # Deterministically shuffle based on epoch and seed
    if seed is not None:
        rng = get_rng(seed)
        rng.shuffle(chunk_indices)

    if chunk_size == 1:
        return chunk_indices

    indices = np.repeat(chunk_indices * chunk_size, chunk_size)
    indices = indices.reshape((-1, chunk_size)) + np.arange(0, chunk_size).reshape((1, -1))
    indices = indices.reshape(-1)
    return indices
