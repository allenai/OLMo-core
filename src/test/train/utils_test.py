import pytest
import torch
import torch.distributed as dist

from olmo_core.train.utils import ReduceType, reduce_metrics

from ..distributed.utils import BACKENDS, get_default_device, run_distributed_test


def run_reduce_metrics():
    device = get_default_device()
    raw_metrics = {
        0: {
            "train/CrossEntropyLoss": torch.tensor(2.0, device=device),
            "optim/total_grad_norm": torch.tensor(1.0, device=device),
        },
        1: {
            "train/CrossEntropyLoss": torch.tensor(
                1.5 if dist.get_rank() == 0 else 2.5, device=device
            ),
            "train/rank": torch.tensor(float(dist.get_rank()), device=device),
        },
    }
    metrics_reduce_type = {
        "train/CrossEntropyLoss": ReduceType.mean,
        "train/rank": ReduceType.max,
        "optim/total_grad_norm": None,
    }

    metrics = reduce_metrics(raw_metrics, metrics_reduce_type, device)
    if dist.get_rank() == 0:
        assert metrics == {
            0: {"train/CrossEntropyLoss": 2.0, "optim/total_grad_norm": 1.0},
            1: {"train/CrossEntropyLoss": 2.0, "train/rank": 1.0},
        }


@pytest.mark.parametrize("backend", BACKENDS)
def test_reduce_metrics(backend):
    run_distributed_test(run_reduce_metrics, backend=backend)
