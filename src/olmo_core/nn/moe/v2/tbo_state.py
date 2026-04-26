from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from .block import MoEFusedV2TransformerBlock


@dataclass
class SyncedTboPendingContext:
    global_x: torch.Tensor
    send_counts: List[int]
    recv_counts: List[int]
    reversed_local_x_permutation_mapping: torch.Tensor
    local_x_global_routed_expert_weights: torch.Tensor
    hidden_shape_before_permute: torch.Size
    in_shape: torch.Size
    mixed_shared_out: Optional[torch.Tensor]
    attn_res_out: torch.Tensor
    last_block: MoEFusedV2TransformerBlock
