from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, cast

import nvtx
import torch

from ...moe.utils import async_copy_to_cpu, wait_stream_no_compile
from ..utils import moe_permute_no_compile, moe_unpermute_no_compile
from .routed_experts import requires_host_side_split_sizes

if TYPE_CHECKING:
    from .block import MoEFusedV2TransformerBlock


def combined_forward_no_ep(
    block: MoEFusedV2TransformerBlock,
    x: torch.Tensor,
    *,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> torch.Tensor:
    """Forward function without expert parallelism."""
    self = block
    assert self.routed_experts is not None
    assert self.routed_experts_router is not None

    B, S, D = x.shape

    block_inp = x
    del x

    attn_res_out = self._res_norm_attn(block_inp, **kwargs)
    kwargs.pop("max_doc_len", None)
    kwargs.pop("cu_doc_lens", None)
    moe_inp = self._prepare_moe_input(attn_res_out)

    (
        local_x_global_routed_expert_weights,
        local_x_global_routed_expert_indices,
        local_batch_size_per_global_routed_expert,
        routed_expert_router_aux_loss_info,
    ) = self.routed_experts_router(
        moe_inp,
        False,
        loss_div_factor=loss_div_factor,
    )

    if requires_host_side_split_sizes():
        local_batch_size_per_global_routed_expert_cpu, copy_stream, dtoh_event = async_copy_to_cpu(
            local_batch_size_per_global_routed_expert,
            event=self._dtoh_event,
        )
    else:
        dtoh_event = None
        local_batch_size_per_global_routed_expert_cpu = None

    if self.shared_experts_router:
        (
            local_x_global_shared_expert_weights,
            _,
            _,
            _,
        ) = self.shared_experts_router(
            moe_inp,
            True,
            loss_div_factor=loss_div_factor,
        )
    else:
        local_x_global_shared_expert_weights = None

    in_shape = moe_inp.size()

    mixed_shared_out = None
    if self.shared_experts is not None:
        wait_stream_no_compile(
            this_stream=self.get_dense_stream(),
            other_stream=torch.cuda.current_stream(),
        )

        with torch.cuda.stream(self.get_dense_stream()):
            shared_out = self.shared_experts(moe_inp)
            if self.shared_experts.num_experts == 1:
                mixed_shared_out = shared_out.squeeze(0)
            else:
                assert local_x_global_shared_expert_weights is not None
                _, _, E_s = local_x_global_shared_expert_weights.shape
                mixed_shared_out = (
                    torch.bmm(
                        local_x_global_shared_expert_weights.to(shared_out.dtype).reshape(
                            B * S, 1, E_s
                        ),
                        shared_out.permute(1, 2, 0, 3).contiguous().view(B * S, E_s, D),
                    )
                    .squeeze(1)
                    .view(B, S, D)
                )

    moe_inp = moe_inp.view(-1, in_shape[-1])

    routing_map = local_x_global_routed_expert_indices.view(
        -1, self.routed_experts_router.top_k
    ).int()
    num_out_tokens = routing_map.size(0) * self.routed_experts_router.top_k
    hidden_shape_before_permute = moe_inp.shape

    with nvtx.annotate("Permute", color="green"):
        permutated_input_tokens, reversed_input_permutation_mapping = moe_permute_no_compile(
            inp=moe_inp,
            routing_map=routing_map,
            num_out_tokens=num_out_tokens,
            map_type="index",
        )

    torch._dynamo.mark_dynamic(permutated_input_tokens, 0)

    if requires_host_side_split_sizes():
        assert dtoh_event is not None
        dtoh_event = cast(torch.cuda.Event, dtoh_event)
        dtoh_event.synchronize()
        mlp_x = self.routed_experts(
            permutated_input_tokens, local_batch_size_per_global_routed_expert_cpu
        )
    else:
        mlp_x = self.routed_experts(
            permutated_input_tokens, local_batch_size_per_global_routed_expert
        )

    with nvtx.annotate("Unpermute", color="green"):
        unpermutated_x: torch.Tensor = moe_unpermute_no_compile(
            inp=mlp_x,
            row_id_map=reversed_input_permutation_mapping,
            restore_shape=hidden_shape_before_permute,
            map_type="index",
            merging_probs=local_x_global_routed_expert_weights.view(
                -1, self.routed_experts_router.top_k
            ),
        )

    x_moe = unpermutated_x.view(in_shape)

    wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())

    mlp_out = self._merge_routed_and_shared(x_moe, mixed_shared_out)

    final_out = self._res_norm_mlp(attn_res_out, mlp_out)

    return self._attach_routed_aux_loss(
        final_out,
        routed_expert_router_aux_loss_info,
    )
