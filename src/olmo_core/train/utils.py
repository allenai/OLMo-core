import logging
import math
import random
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.version
from packaging.version import parse as parse_version

import olmo_core.distributed.utils as dist_utils

from ..config import Config
from ..utils import cuda_sync_debug_mode, move_to_device
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
    # Collate all metrics together, then transfer to device all at once.
    metrics_to_move_list = [
        dist_utils.get_local_tensor(m)
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
            metrics_to_move = move_to_device(torch.stack(metrics_to_move_list), device)

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


def get_metrics_reduce_type_by_step(
    metrics: Dict[int, Dict[str, torch.Tensor]],
    metrics_reduce_type: Dict[str, Optional[ReduceType]],
    process_group: Optional[dist.ProcessGroup] = None,
) -> Dict[int, Dict[str, Optional[ReduceType]]]:
    all_ranks_metrics_reduce_type = dist_utils.all_gather_object(
        metrics_reduce_type, group=process_group
    )

    out: Dict[int, Dict[str, Optional[ReduceType]]] = defaultdict(dict)
    for step in metrics.keys():
        for rank_metrics_reduce_type in all_ranks_metrics_reduce_type:
            for metric_name, reduce_type in rank_metrics_reduce_type.items():
                out[step][metric_name] = reduce_type

    return out


def get_metric_world_sizes_by_step(
    metrics: Dict[int, Dict[str, torch.Tensor]],
    metrics_reduce_type: Dict[str, Optional[ReduceType]],
    process_group: Optional[dist.ProcessGroup] = None,
) -> Dict[int, Dict[str, int]]:
    all_ranks_metrics_reduce_type: List[
        Dict[str, Optional[ReduceType]]
    ] = dist_utils.all_gather_object(metrics_reduce_type, group=process_group)

    all_steps_world_sizes: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for step in metrics.keys():
        for rank_metrics_reduce_type in all_ranks_metrics_reduce_type:
            for metric_name in rank_metrics_reduce_type.keys():
                all_steps_world_sizes[step][metric_name] += 1

    return all_steps_world_sizes


def check_metrics_consistent(
    metrics_reduce_type: Dict[str, Optional[ReduceType]],
    process_group: Optional[dist.ProcessGroup] = None,
) -> bool:
    metrics_to_reduce: Set[str] = set(k for k, v in metrics_reduce_type.items() if v is not None)
    all_ranks_metrics_to_reduce = dist_utils.all_gather_object(
        metrics_to_reduce, group=process_group
    )
    for rank in range(dist_utils.get_world_size(process_group)):
        if metrics_to_reduce != all_ranks_metrics_to_reduce[rank]:
            return False
    return True


@torch.no_grad()
def reduce_metrics(
    metrics: Dict[int, Dict[str, torch.Tensor]],
    metrics_reduce_type: Dict[str, Optional[ReduceType]],
    device: torch.device,
    process_group: Optional[dist.ProcessGroup] = None,
    metrics_consistent: bool = True,
) -> Dict[int, Dict[str, float]]:
    metrics = move_metrics(metrics, device)
    out: Dict[int, Dict[str, float]] = defaultdict(dict)

    if not dist_utils.is_distributed():
        for step, step_metrics in metrics.items():
            for name, value in step_metrics.items():
                out[step][name] = value.item()
        return out

    world_size = dist_utils.get_world_size(process_group)
    divide_factor = dist_utils.get_reduce_divide_factor(world_size)
    all_steps_metric_world_sizes: Dict[int, Dict[str, int]] = {}
    all_steps_metrics_reduce_type: Dict[int, Dict[str, Optional[ReduceType]]] = {}
    if not metrics_consistent:
        all_steps_metric_world_sizes = get_metric_world_sizes_by_step(
            metrics,
            metrics_reduce_type,
            process_group=process_group,
        )
        all_steps_metrics_reduce_type = get_metrics_reduce_type_by_step(
            metrics,
            metrics_reduce_type,
            process_group=process_group,
        )

    # Flattened metrics by step and reduce type.
    sum_metric_names: List[List[str]] = []
    sum_metric_values: List[torch.Tensor] = []
    max_metric_names: List[List[str]] = []
    max_metric_values: List[torch.Tensor] = []

    for step in sorted(metrics.keys()):
        step_metrics_reduce_type: Dict[
            str, Optional[ReduceType]
        ] = all_steps_metrics_reduce_type.get(step, metrics_reduce_type)
        step_sum_metric_names: List[str] = []
        step_sum_metric_values: List[torch.Tensor] = []
        step_max_metric_names: List[str] = []
        step_max_metric_values: List[torch.Tensor] = []

        step_metrics = metrics[step]

        sorted_metric_names: List[str]
        if not metrics_consistent:
            sorted_metric_names = sorted(all_steps_metric_world_sizes[step].keys())
        else:
            sorted_metric_names = sorted(step_metrics.keys())

        for name in sorted_metric_names:
            value = step_metrics.get(name)
            reduce_type = step_metrics_reduce_type[name]
            if reduce_type in (ReduceType.mean, ReduceType.sum):
                step_sum_metric_names.append(name)
                if value is None:
                    step_sum_metric_values.append(move_to_device(torch.tensor(0.0), device))
                else:
                    step_sum_metric_values.append(value)
            elif reduce_type == ReduceType.l2_norm:
                step_sum_metric_names.append(name)
                if value is None:
                    step_sum_metric_values.append(move_to_device(torch.tensor(0.0), device))
                else:
                    step_sum_metric_values.append(value.pow(2))
            elif reduce_type == ReduceType.max:
                step_max_metric_names.append(name)
                if value is None:
                    step_max_metric_values.append(
                        move_to_device(torch.tensor(float("-inf")), device)
                    )
                else:
                    step_max_metric_values.append(value)
            elif reduce_type is None:
                if value is not None:
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

    dist_utils.all_reduce(
        all_sum_metrics,
        op=dist.ReduceOp.SUM,
        group=process_group,
    )
    dist_utils.all_reduce(
        all_max_metrics,
        op=dist.ReduceOp.MAX,
        group=process_group,
    )

    # Transfer to CPU all at once (if not already on CPU).
    all_sum_metrics = all_sum_metrics.cpu()
    all_max_metrics = all_max_metrics.cpu()

    for i, step in enumerate(sorted(metrics.keys())):
        step_metrics_reduce_type = all_steps_metrics_reduce_type.get(step, metrics_reduce_type)
        step_sum_metric_names = sum_metric_names[i]
        step_sum_metric_items = all_sum_metrics[i].tolist()
        step_max_metric_names = max_metric_names[i]
        step_max_metric_items = all_max_metrics[i].tolist()
        for name, item in zip(step_sum_metric_names, step_sum_metric_items):
            item = item * divide_factor
            reduce_type = step_metrics_reduce_type[name]
            if reduce_type == ReduceType.mean:
                item = item / all_steps_metric_world_sizes.get(step, {}).get(name, world_size)
            elif reduce_type == ReduceType.l2_norm:
                item = math.sqrt(item)
            out[step][name] = item
        for name, item in zip(step_max_metric_names, step_max_metric_items):
            out[step][name] = item

    return out
