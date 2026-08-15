import math
import random

import pytest
import torch
import torch.distributed as dist

from olmo_core.testing import BACKENDS, run_distributed_test
from olmo_core.train import ReduceType
from olmo_core.train.utils import reduce_metrics
from olmo_core.utils import get_default_device


def run_reduce_metrics():
    device = get_default_device()
    raw_metrics = {
        0: {
            "train/CrossEntropyLoss": torch.tensor(2.0, device=device),
            "train/masked_instances": torch.tensor(1.0, device=device),
            "optim/total_grad_norm": torch.tensor(1.0, device=device),
        },
        1: {
            "train/CrossEntropyLoss": torch.tensor(
                1.5 if dist.get_rank() == 0 else 2.5, device=device
            ),
            "train/masked_instances": torch.tensor(
                0.0 if dist.get_rank() == 0 else 1.0, device=device
            ),
            "train/rank": torch.tensor(float(dist.get_rank()), device=device),
            "optim/weight_norm": torch.tensor(2.0 if dist.get_rank() == 0 else 3.0, device=device),
        },
    }
    metrics_reduce_type = {
        "train/CrossEntropyLoss": ReduceType.mean,
        "train/rank": ReduceType.max,
        "train/masked_instances": ReduceType.sum,
        "optim/total_grad_norm": None,
        "optim/weight_norm": ReduceType.l2_norm,
    }

    metrics = reduce_metrics(raw_metrics, metrics_reduce_type, device)
    if dist.get_rank() == 0:
        assert metrics == {
            0: {
                "train/CrossEntropyLoss": 2.0,
                "optim/total_grad_norm": 1.0,
                "train/masked_instances": 2.0,
            },
            1: {
                "train/CrossEntropyLoss": 2.0,
                "train/rank": 1.0,
                "train/masked_instances": 1.0,
                "optim/weight_norm": math.sqrt(13),
            },
        }


@pytest.mark.parametrize("backend", BACKENDS)
def test_reduce_metrics(backend):
    run_distributed_test(run_reduce_metrics, backend=backend)


def run_reduce_metrics_with_rank_local_shapes():
    device = get_default_device()
    rank = dist.get_rank()
    raw_metrics = {
        0: {
            "train/loss": torch.tensor(2.0, device=device),
            "train/negative_max": torch.tensor(-3.0 if rank == 0 else -2.0, device=device),
        }
    }
    if rank == 1:
        raw_metrics[0]["train/z_extra"] = torch.tensor(4.0, device=device)
        raw_metrics[1] = {"train/loss": torch.tensor(6.0, device=device)}

    metrics = reduce_metrics(
        raw_metrics,
        {
            "train/loss": ReduceType.mean,
            "train/negative_max": ReduceType.max,
            "train/z_extra": ReduceType.sum,
        },
        device,
        # Exercise the defensive shape negotiation even when a stale schema cache claims that
        # metrics are consistent.
        metrics_consistent=True,
    )

    assert metrics[0]["train/loss"] == 2.0
    assert metrics[0]["train/negative_max"] == -2.0
    if rank == 1:
        assert metrics[0]["train/z_extra"] == 4.0
        assert metrics[1]["train/loss"] == 3.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_reduce_metrics_with_rank_local_shapes(backend):
    run_distributed_test(run_reduce_metrics_with_rank_local_shapes, backend=backend)


def run_reduce_metrics_dynamic_schema_stress():
    device = get_default_device()
    rank = dist.get_rank()
    rng = random.Random(1729 + rank)

    for iteration in range(25):
        raw_metrics = {
            iteration: {
                "shared/loss": torch.tensor(float(rank + 1), device=device),
            }
        }
        metrics_reduce_type = {"shared/loss": ReduceType.mean}
        for metric_idx in range(rng.randint(0, 4)):
            name = f"rank_local/z{rank}_{metric_idx}"
            raw_metrics[iteration][name] = torch.tensor(rng.uniform(-5.0, 5.0), device=device)
            metrics_reduce_type[name] = rng.choice(
                [ReduceType.sum, ReduceType.max, ReduceType.mean]
            )
        if rng.random() < 0.5:
            raw_metrics[iteration + 1] = {
                "shared/loss": torch.tensor(float(rank + 1), device=device)
            }

        metrics = reduce_metrics(raw_metrics, metrics_reduce_type, device)
        assert metrics[iteration]["shared/loss"] == 1.5


@pytest.mark.parametrize("backend", BACKENDS)
def test_reduce_metrics_dynamic_schema_stress(backend):
    run_distributed_test(run_reduce_metrics_dynamic_schema_stress, backend=backend)
