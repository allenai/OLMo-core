import math
from typing import Dict, Optional, cast

import pytest
import torch
import torch.distributed as dist

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.testing import BACKENDS, run_distributed_test
from olmo_core.train import ReduceType, Trainer
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


def run_reduce_metrics_with_an_empty_rank():
    device = get_default_device()
    rank = dist.get_rank()

    # Rank 0 recorded this step and rank 1 recorded nothing, which is what happens whenever a
    # callback reports from some ranks and not others. Rank 1 still has to enter the reduce.
    raw_metrics: Dict[int, Dict[str, torch.Tensor]] = (
        {
            3: {
                "train/CrossEntropyLoss": torch.tensor(2.0, device=device),
                "train/masked_instances": torch.tensor(1.0, device=device),
                "train/rank": torch.tensor(0.0, device=device),
            }
        }
        if rank == 0
        else {}
    )
    metrics_reduce_type: Dict[str, Optional[ReduceType]] = (
        {
            "train/CrossEntropyLoss": ReduceType.mean,
            "train/masked_instances": ReduceType.sum,
            "train/rank": ReduceType.max,
        }
        if rank == 0
        else {}
    )

    metrics = reduce_metrics(raw_metrics, metrics_reduce_type, device)

    # The mean is over the one rank that reported, not over the world, and both ranks come back
    # with the same answer.
    assert metrics == {
        3: {
            "train/CrossEntropyLoss": 2.0,
            "train/masked_instances": 1.0,
            "train/rank": 0.0,
        }
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_reduce_metrics_with_an_empty_rank(backend):
    run_distributed_test(run_reduce_metrics_with_an_empty_rank, backend=backend)


def run_reduce_metrics_with_nothing_anywhere():
    device = get_default_device()
    assert reduce_metrics({}, {}, device) == {}


@pytest.mark.parametrize("backend", BACKENDS)
def test_reduce_metrics_with_nothing_anywhere(backend):
    run_distributed_test(run_reduce_metrics_with_nothing_anywhere, backend=backend)


def run_reduce_metrics_with_a_metric_one_rank_never_saw():
    device = get_default_device()
    rank = dist.get_rank()

    raw_metrics: Dict[int, Dict[str, torch.Tensor]] = {
        4: {"train/CrossEntropyLoss": torch.tensor(2.0 if rank == 0 else 4.0, device=device)}
    }
    metrics_reduce_type: Dict[str, Optional[ReduceType]] = {
        "train/CrossEntropyLoss": ReduceType.mean
    }
    if rank == 0:
        raw_metrics[4]["train/skipped_batches"] = torch.tensor(1.0, device=device)
        metrics_reduce_type["train/skipped_batches"] = ReduceType.sum

    metrics = reduce_metrics(raw_metrics, metrics_reduce_type, device)

    # Rank 1 has never heard of 'train/skipped_batches' and so cannot know how to reduce it, but
    # it has to contribute a slot for it or the two ranks all-reduce differently shaped tensors.
    assert metrics == {
        4: {
            "train/CrossEntropyLoss": 3.0,
            "train/skipped_batches": 1.0,
        }
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_reduce_metrics_with_a_metric_one_rank_never_saw(backend):
    run_distributed_test(run_reduce_metrics_with_a_metric_one_rank_never_saw, backend=backend)


def test_a_distributed_bookkeeping_op_cannot_be_dropped_by_one_rank():
    # Dropping a repeat invocation is a decision about this rank's timing, so it must not be
    # available to an op that communicates. All three callbacks in the tree that ask for it pass
    # 'distributed=False'.
    trainer = cast(Trainer, object.__new__(Trainer))
    with pytest.raises(OLMoConfigurationError, match="allow_multiple=False"):
        trainer.run_bookkeeping_op(lambda: None, allow_multiple=False, distributed=True)
