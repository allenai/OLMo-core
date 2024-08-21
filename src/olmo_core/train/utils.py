import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.version
from packaging.version import parse as parse_version

from ..config import Config, StrEnum
from ..distributed.utils import get_world_size, init_distributed, is_distributed
from ..io import add_cached_path_clients
from ..utils import prepare_cli_environment

log = logging.getLogger(__name__)


class ReduceType(StrEnum):
    """
    An enumeration of the allowed ways to reduce a metric across ranks.
    """

    mean = "mean"
    sum = "sum"
    max = "max"


@dataclass
class LibRngState:
    version: Tuple[int, int]
    state: Any


@dataclass
class EnvRngStates(Config):
    python: LibRngState
    numpy: LibRngState
    torch: LibRngState
    cuda: Optional[LibRngState] = None

    def restore(self) -> bool:
        all_restored = True
        if self.python.version == _get_python_version():
            random.setstate(self.python.state)
        else:
            all_restored = False

        if self.numpy.version == _get_numpy_version():
            np.random.set_state(self.numpy.state)
        else:
            all_restored = False

        if self.torch.version == _get_torch_version():
            torch.set_rng_state(self.torch.state)
        else:
            all_restored = False

        if self.cuda is not None:
            if (
                torch.cuda.is_available()
                and torch.cuda.is_initialized()
                and self.cuda.version == _get_cuda_version()
            ):
                torch.cuda.set_rng_state(self.cuda.state)
            else:
                all_restored = False

        return all_restored

    @classmethod
    def current_state(cls) -> "EnvRngStates":
        python_rng = LibRngState(version=_get_python_version(), state=random.getstate())
        numpy_rng = LibRngState(version=_get_numpy_version(), state=np.random.get_state())
        torch_rng = LibRngState(version=_get_torch_version(), state=torch.random.get_rng_state())
        cuda_rng: Optional[LibRngState] = None
        if (cuda_version := _get_cuda_version()) is not None:
            cuda_rng = LibRngState(version=cuda_version, state=torch.cuda.get_rng_state())
        return cls(python=python_rng, numpy=numpy_rng, torch=torch_rng, cuda=cuda_rng)


def _get_python_version() -> Tuple[int, int]:
    return sys.version_info[0:2]


def _get_numpy_version() -> Tuple[int, int]:
    version = parse_version(np.version.short_version)
    return (version.major, version.minor)


def _get_torch_version() -> Tuple[int, int]:
    version = parse_version(torch.__version__)
    return (version.major, version.minor)


def _get_cuda_version() -> Optional[Tuple[int, int]]:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        assert torch.version.cuda is not None
        version = parse_version(torch.version.cuda)
        return (version.major, version.minor)
    else:
        return None


def reduce_metrics(
    metrics: Dict[int, Dict[str, torch.Tensor]],
    metrics_reduce_type: Dict[str, Optional[ReduceType]],
    device: torch.device,
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = defaultdict(dict)

    if not is_distributed():
        for step, step_metrics in metrics.items():
            for name, value in step_metrics.items():
                out[step][name] = value.item()
        return out

    # Flattened metrics by step and reduce type.
    sum_metric_names: List[List[str]] = []
    sum_metric_values: List[torch.Tensor] = []
    max_metric_names: List[List[str]] = []
    max_metric_values: List[torch.Tensor] = []

    for step in sorted(metrics.keys()):
        step_sum_metric_names: List[str] = []
        step_sum_metric_values: List[torch.Tensor] = []
        step_max_metric_names: List[str] = []
        step_max_metric_values: List[torch.Tensor] = []

        step_metrics = metrics[step]
        for name in sorted(step_metrics.keys()):
            value = step_metrics[name]
            reduce_type = metrics_reduce_type[name]
            if reduce_type == ReduceType.mean:
                step_sum_metric_names.append(name)
                step_sum_metric_values.append((value / get_world_size()).to(device))
            elif reduce_type == ReduceType.sum:
                step_sum_metric_names.append(name)
                step_sum_metric_values.append(value.to(device))
            elif reduce_type == ReduceType.max:
                step_max_metric_names.append(name)
                step_max_metric_values.append(value.to(device))
            elif reduce_type is None:
                out[step][name] = value.item()
            else:
                raise NotImplementedError()

        sum_metric_names.append(step_sum_metric_names)
        sum_metric_values.append(
            torch.stack(step_sum_metric_values)
            if step_sum_metric_values
            else torch.tensor([], device=device)
        )
        max_metric_names.append(step_max_metric_names)
        max_metric_values.append(
            torch.stack(step_max_metric_values)
            if step_max_metric_values
            else torch.tensor([], device=device)
        )

    max_num_sum_metrics = max(t.numel() for t in sum_metric_values)
    max_num_max_metrics = max(t.numel() for t in max_metric_values)

    all_sum_metrics = torch.stack(
        [F.pad(t, (0, max_num_sum_metrics - t.numel()), value=0.0) for t in sum_metric_values]
    )
    all_max_metrics = torch.stack(
        [F.pad(t, (0, max_num_max_metrics - t.numel()), value=0.0) for t in max_metric_values]
    )

    dist.reduce(all_sum_metrics, 0, op=dist.ReduceOp.SUM)
    dist.reduce(all_max_metrics, 0, op=dist.ReduceOp.MAX)

    for i, step in enumerate(sorted(metrics.keys())):
        step_sum_metric_names = sum_metric_names[i]
        step_sum_metric_items = all_sum_metrics[i].tolist()
        step_max_metric_names = max_metric_names[i]
        step_max_metric_items = all_max_metrics[i].tolist()
        for name, item in zip(step_sum_metric_names, step_sum_metric_items):
            out[step][name] = item
        for name, item in zip(step_max_metric_names, step_max_metric_items):
            out[step][name] = item

    return out


def prepare_training_environment(
    backend: Optional[str] = "nccl", timeout: timedelta = timedelta(minutes=30)
):
    """
    Prepare the environment for training, including setting up the distributed process group
    for distributed training.

    .. important::
        This should be invoked at the very start of your training script, such as at the beginning
        of the ``if __name__ == "__main__": ...`` block.

    :param backend: The distributed backend to use, if any. Set to ``None`` for non-distributed training.
    :param timeout: The timeout for initializing the distributed process group.
    """
    # Setting the mp start method to "spawn" avoids some data loader segfaults on LUMI.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")

    # Initialize process group.
    if backend is not None:
        init_distributed(backend=backend, timeout=timeout)

    # Configure logging, warning filters, exception hooks, and other CLI settings.
    prepare_cli_environment()

    # Add custom cached-path clients.
    add_cached_path_clients()
