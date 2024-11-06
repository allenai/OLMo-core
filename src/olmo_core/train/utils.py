import logging
import math
import random
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.version
from packaging.version import parse as parse_version

from ..config import Config
from ..distributed.utils import (
    get_local_tensor,
    get_reduce_divide_factor,
    get_world_size,
    is_distributed,
)
from ..utils import cuda_sync_debug_mode
from .common import ReduceType

log = logging.getLogger(__name__)


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

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], overrides: Optional[List[str]] = None
    ) -> "EnvRngStates":
        # overriding this from the base class since omegaconf doesn't like whatever objects
        # we get for the states, like numpy ndarrays.
        assert overrides is None
        return cls(
            python=LibRngState(**data["python"]),
            numpy=LibRngState(**data["numpy"]),
            torch=LibRngState(**data["torch"]),
            cuda=None if data.get("cuda") is None else LibRngState(**data["cuda"]),
        )


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


@torch.no_grad()
def move_metrics(
    source: Dict[int, Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict[int, Dict[str, torch.Tensor]]:
    non_blocking = device.type != "cpu"

    # Collate all metrics together, then transfer to device all at once.
    metrics_to_move_list = [
        get_local_tensor(m)
        for step_metrics in source.values()
        for m in step_metrics.values()
        # NOTE: compare device type since 'torch.device("cuda") != torch.device("cuda:0")'
        # even when both point to the same physical device.
        if m.device.type != device.type
    ]
    metrics_to_move: Optional[torch.Tensor] = None
    if metrics_to_move_list:
        # NOTE: this is a known host-device sync (potentially) so we don't need the warning
        with cuda_sync_debug_mode(0):
            metrics_to_move = torch.stack(metrics_to_move_list).to(
                device, non_blocking=non_blocking
            )

    # Collect output with moved tensors.
    target: Dict[int, Dict[str, torch.Tensor]] = OrderedDict()
    idx = 0
    for step, step_metrics in source.items():
        for name, m in step_metrics.items():
            if step not in target:
                target[step] = OrderedDict()
            if metrics_to_move is not None and m.device.type != device.type:
                target[step][name] = metrics_to_move[idx]
                idx += 1
            else:
                target[step][name] = m

    return target


@torch.no_grad()
def reduce_metrics(
    metrics: Dict[int, Dict[str, torch.Tensor]],
    metrics_reduce_type: Dict[str, Optional[ReduceType]],
    device: torch.device,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Dict[int, Dict[str, float]]:
    metrics = move_metrics(metrics, device)
    out: Dict[int, Dict[str, float]] = defaultdict(dict)

    if not is_distributed():
        for step, step_metrics in metrics.items():
            for name, value in step_metrics.items():
                out[step][name] = value.item()
        return out

    world_size = get_world_size(process_group)
    divide_factor = get_reduce_divide_factor(world_size)

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
            if reduce_type in (ReduceType.mean, ReduceType.sum):
                step_sum_metric_names.append(name)
                step_sum_metric_values.append(value)
            elif reduce_type == ReduceType.l2_norm:
                step_sum_metric_names.append(name)
                step_sum_metric_values.append(value.pow(2))
            elif reduce_type == ReduceType.max:
                step_max_metric_names.append(name)
                step_max_metric_values.append(value)
            elif reduce_type is None:
                out[step][name] = value.item()
            else:
                raise NotImplementedError()

        sum_metric_names.append(step_sum_metric_names)
        sum_metric_values.append(
            torch.stack(step_sum_metric_values)
            if step_sum_metric_values
            else torch.tensor([]).to(device=device, non_blocking=True)
        )
        max_metric_names.append(step_max_metric_names)
        max_metric_values.append(
            torch.stack(step_max_metric_values)
            if step_max_metric_values
            else torch.tensor([]).to(device=device, non_blocking=True)
        )

    max_num_sum_metrics = max(t.numel() for t in sum_metric_values)
    max_num_max_metrics = max(t.numel() for t in max_metric_values)

    all_sum_metrics = torch.stack(
        [F.pad(t, (0, max_num_sum_metrics - t.numel()), value=0.0) for t in sum_metric_values]
    )
    all_max_metrics = torch.stack(
        [F.pad(t, (0, max_num_max_metrics - t.numel()), value=0.0) for t in max_metric_values]
    )
    del sum_metric_values
    del max_metric_values

    all_sum_metrics.div_(divide_factor)

    dist.reduce(all_sum_metrics, 0, op=dist.ReduceOp.SUM, group=process_group)
    dist.reduce(all_max_metrics, 0, op=dist.ReduceOp.MAX, group=process_group)

    all_sum_metrics.mul_(divide_factor)

    # Transfer to CPU all at once (if not already on CPU).
    all_sum_metrics = all_sum_metrics.cpu()
    all_max_metrics = all_max_metrics.cpu()

    for i, step in enumerate(sorted(metrics.keys())):
        step_sum_metric_names = sum_metric_names[i]
        step_sum_metric_items = all_sum_metrics[i].tolist()
        step_max_metric_names = max_metric_names[i]
        step_max_metric_items = all_max_metrics[i].tolist()
        for name, item in zip(step_sum_metric_names, step_sum_metric_items):
            reduce_type = metrics_reduce_type[name]
            if reduce_type == ReduceType.mean:
                item = item / world_size
            elif reduce_type == ReduceType.l2_norm:
                item = math.sqrt(item)
            out[step][name] = item
        for name, item in zip(step_max_metric_names, step_max_metric_items):
            out[step][name] = item

    return out
