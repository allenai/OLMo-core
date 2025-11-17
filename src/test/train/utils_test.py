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
