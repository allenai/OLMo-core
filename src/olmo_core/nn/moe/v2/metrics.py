from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType
    from .block import MoEFusedV2TransformerBlock


def reset_ep_no_sync_rowwise_metrics(block: MoEFusedV2TransformerBlock) -> None:
    block._ep_no_sync_rowwise_drop_tokens_sum = None
    block._ep_no_sync_rowwise_total_tokens_sum = None
    block._ep_no_sync_rowwise_symm_util_max = None


def add_ep_no_sync_rowwise_metrics(
    block: MoEFusedV2TransformerBlock,
    out: Dict[str, Tuple[torch.Tensor, Optional[ReduceType]]],
    reduce_type_cls: type[ReduceType],
) -> None:
    if (
        block._ep_no_sync_rowwise_drop_tokens_sum is not None
        and block._ep_no_sync_rowwise_total_tokens_sum is not None
    ):
        drop_ratio = (
            block._ep_no_sync_rowwise_drop_tokens_sum.to(dtype=torch.float32)
            / block._ep_no_sync_rowwise_total_tokens_sum.to(dtype=torch.float32).clamp_min(1.0)
        ).clamp(0.0, 1.0)
        out["token drop rate"] = (drop_ratio, reduce_type_cls.mean)

    if block._ep_no_sync_rowwise_symm_util_max is not None:
        out["symm buffer util"] = (
            block._ep_no_sync_rowwise_symm_util_max.to(dtype=torch.float32),
            reduce_type_cls.max,
        )


def accumulate_ep_no_sync_rowwise_metrics(
    block: MoEFusedV2TransformerBlock,
    *,
    drop_token_cnt: torch.Tensor,
    num_out_tokens: int,
    recv_splits_by_src_local: torch.Tensor,
    rank_capacity: int,
) -> None:
    if rank_capacity <= 0:
        return

    drop_sum = drop_token_cnt.to(dtype=torch.float32)
    total_sum = torch.tensor(float(num_out_tokens), device=drop_sum.device)
    util = (
        recv_splits_by_src_local.sum(dtype=torch.float32)
        / torch.tensor(float(rank_capacity), device=drop_sum.device)
    )

    if block._ep_no_sync_rowwise_drop_tokens_sum is None:
        block._ep_no_sync_rowwise_drop_tokens_sum = drop_sum
    else:
        block._ep_no_sync_rowwise_drop_tokens_sum = (
            block._ep_no_sync_rowwise_drop_tokens_sum + drop_sum
        )

    if block._ep_no_sync_rowwise_total_tokens_sum is None:
        block._ep_no_sync_rowwise_total_tokens_sum = total_sum
    else:
        block._ep_no_sync_rowwise_total_tokens_sum = (
            block._ep_no_sync_rowwise_total_tokens_sum + total_sum
        )

    if block._ep_no_sync_rowwise_symm_util_max is None:
        block._ep_no_sync_rowwise_symm_util_max = util
    else:
        block._ep_no_sync_rowwise_symm_util_max = torch.maximum(
            block._ep_no_sync_rowwise_symm_util_max,
            util,
        )
