import math

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


def run_reduce_metrics_inconsistent_per_step():
    # Regression test: ranks record DIFFERENT reducible metric names on the *same* step (here rank 0
    # logs an extra metric, as happens when e.g. a throughput metric is skipped on a rank whose
    # micro-batch had no tokens). The caller still passes ``metrics_consistent=True`` -- exactly what
    # the trainer does from its registry-level check, which doesn't catch per-step divergence. This
    # used to build mismatched all-reduce tensors across ranks and DEADLOCK. ``reduce_metrics`` must
    # detect the divergence, fall back to the consistent-handling path, and still return correct
    # values (no hang).
    device = get_default_device()
    rank = dist.get_rank()
    raw_metrics = {
        0: {
            "train/CrossEntropyLoss": torch.tensor(2.0 if rank == 0 else 4.0, device=device),
        },
    }
    metrics_reduce_type = {"train/CrossEntropyLoss": ReduceType.mean}
    if rank == 0:
        raw_metrics[0]["train/throughput"] = torch.tensor(5.0, device=device)
        metrics_reduce_type["train/throughput"] = ReduceType.mean

    metrics = reduce_metrics(raw_metrics, metrics_reduce_type, device, metrics_consistent=True)

    assert metrics[0]["train/CrossEntropyLoss"] == 3.0  # mean over both ranks: (2 + 4) / 2
    assert metrics[0]["train/throughput"] == 5.0  # recorded on one rank only


@pytest.mark.parametrize("backend", BACKENDS)
def test_reduce_metrics_inconsistent_per_step(backend):
    run_distributed_test(run_reduce_metrics_inconsistent_per_step, backend=backend)
