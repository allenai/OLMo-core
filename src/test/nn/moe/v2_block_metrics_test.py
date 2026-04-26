from types import SimpleNamespace

import torch

from olmo_core.nn.moe.v2 import metrics
from olmo_core.train.common import ReduceType


def _new_metric_state():
    return SimpleNamespace(
        _ep_no_sync_rowwise_drop_tokens_sum=None,
        _ep_no_sync_rowwise_total_tokens_sum=None,
        _ep_no_sync_rowwise_symm_util_max=None,
    )


def test_ep_no_sync_rowwise_metrics_accumulate_add_and_reset():
    block = _new_metric_state()

    metrics.accumulate_ep_no_sync_rowwise_metrics(
        block,
        drop_token_cnt=torch.tensor(2),
        num_out_tokens=8,
        recv_splits_by_src_local=torch.tensor([2, 1, 1]),
        rank_capacity=4,
    )
    metrics.accumulate_ep_no_sync_rowwise_metrics(
        block,
        drop_token_cnt=torch.tensor(1),
        num_out_tokens=2,
        recv_splits_by_src_local=torch.tensor([1, 0, 0]),
        rank_capacity=8,
    )

    out = {}
    metrics.add_ep_no_sync_rowwise_metrics(block, out, ReduceType)

    assert set(out) == {"token drop rate", "symm buffer util"}
    torch.testing.assert_close(out["token drop rate"][0], torch.tensor(0.3))
    assert out["token drop rate"][1] == ReduceType.mean
    torch.testing.assert_close(out["symm buffer util"][0], torch.tensor(1.0))
    assert out["symm buffer util"][1] == ReduceType.max

    metrics.reset_ep_no_sync_rowwise_metrics(block)
    assert block._ep_no_sync_rowwise_drop_tokens_sum is None
    assert block._ep_no_sync_rowwise_total_tokens_sum is None
    assert block._ep_no_sync_rowwise_symm_util_max is None


def test_ep_no_sync_rowwise_metrics_ignore_zero_capacity():
    block = _new_metric_state()

    metrics.accumulate_ep_no_sync_rowwise_metrics(
        block,
        drop_token_cnt=torch.tensor(3),
        num_out_tokens=4,
        recv_splits_by_src_local=torch.tensor([4]),
        rank_capacity=0,
    )

    out = {}
    metrics.add_ep_no_sync_rowwise_metrics(block, out, ReduceType)
    assert out == {}
