from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import nvtx
import torch

from ...moe.utils import wait_stream_no_compile
from .comm import (
    _DispatchRowwiseAutograd,
    _DispatchRowwiseFP8Autograd,
    _RowwiseCombineWeightedAutograd,
    _RowwiseCombineWeightedFP8Autograd,
)
from .ep_no_sync_buffers import (
    compute_ep_no_sync_rank_capacity,
    get_ep_no_sync_buffers,
    get_ep_no_sync_group_name,
    get_or_init_ep_no_sync_symm_tensor,
)
from .ep_no_sync_common import sync_tail_drop_allowed_splits_single_a2a
from .ep_no_sync_rowwise_helpers import (
    accumulate_ep_no_sync_rowwise_metrics,
    build_rowwise_route_maps,
)
from .fp8 import (
    shared_experts_forward1_rowwise_fp8,
    shared_experts_forward2_rowwise_fp8,
)
from .routed_experts import requires_host_side_split_sizes, use_torch_grouped_mm

if TYPE_CHECKING:
    from .block import MoEFusedV2TransformerBlock


def combined_forward_ep_no_sync_rowwise(
    block: MoEFusedV2TransformerBlock,
    x: torch.Tensor,
    *,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> torch.Tensor:
    """Forward with EP no-sync using row-wise NVSHMEM dispatch/combine."""
    self = block
    assert self.routed_experts is not None
    assert self.routed_experts_router is not None
    assert self.ep_enabled
    assert self.num_local_routed_experts is not None
    assert (
        use_torch_grouped_mm() == True
    ), "EP no-sync implementation requires torch.grouped_mm support"
    assert (
        not requires_host_side_split_sizes()
    ), "EP no-sync implementation does not support host-side split size communication"
    if self.ep_no_sync_use_2d_all_to_all:
        raise RuntimeError(
            "ep_no_sync_use_2d_all_to_all=True is no longer supported: "
            "the 2D all_to_all path was removed due to correctness/performance issues."
        )

    group_name = get_ep_no_sync_group_name(self)
    B, S, D = x.shape

    block_inp = x
    del x

    attn_res_out = self._checkpointed_res_norm_attn(block_inp, **kwargs)

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

    wait_stream_no_compile(
        this_stream=self.get_dense_stream(),
        other_stream=torch.cuda.current_stream(),
    )

    with torch.cuda.stream(self.get_dense_stream()):
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
    moe_inp = moe_inp.view(-1, in_shape[-1])
    rowwise_fp8_cfg = self.rowwise_fp8
    use_rowwise_fp8 = (
        rowwise_fp8_cfg is not None
        and rowwise_fp8_cfg.enabled
        and moe_inp.device.type == "cuda"
        and self.ep_no_sync_use_rowwise_all_to_all
    )
    if use_rowwise_fp8:
        assert rowwise_fp8_cfg is not None
        if not self._rowwise_fp8_checked:
            rowwise_fp8_cfg.assert_runtime_supported()
            self._rowwise_fp8_checked = True
    else:
        rowwise_fp8_cfg = None

    num_out_tokens = local_x_global_routed_expert_indices.numel()

    with torch.no_grad():
        with nvtx.annotate("ConfigCapacity", color="green"):
            requested_splits = local_batch_size_per_global_routed_expert.to(dtype=torch.long)
            rank_capacity = compute_ep_no_sync_rank_capacity(self, num_out_tokens)
            (
                allowed_splits,
                recv_splits_by_src_local,
                _drop_token_cnt,
                keep_from_src_dest_local,
            ) = cast(
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                sync_tail_drop_allowed_splits_single_a2a(
                    self,
                    requested_splits,
                    rank_capacity=rank_capacity,
                    return_keep_matrix=True,
                ),
            )
            dispatch_in_cap = num_out_tokens
            dispatch_out_cap = rank_capacity
            combine_in_cap = rank_capacity
            combine_out_cap = num_out_tokens
            accumulate_ep_no_sync_rowwise_metrics(
                self,
                drop_token_cnt=_drop_token_cnt,
                num_out_tokens=num_out_tokens,
                recv_splits_by_src_local=recv_splits_by_src_local,
                rank_capacity=rank_capacity,
            )

    buffers = get_ep_no_sync_buffers(
        self,
        dispatch_in_cap=dispatch_in_cap,
        dispatch_out_cap=dispatch_out_cap,
        combine_in_cap=combine_in_cap,
        combine_out_cap=combine_out_cap,
        d_model=moe_inp.shape[-1],
        dtype=moe_inp.dtype,
        device=moe_inp.device,
        need_dispatch_in=False,
        need_dispatch_meta=False,
        need_combine_meta=False,
        need_combine_out=False,
    )

    dispatch_out_q: Optional[torch.Tensor] = None
    dispatch_out_scales: Optional[torch.Tensor] = None
    combine_in_q: Optional[torch.Tensor] = None
    combine_in_scales: Optional[torch.Tensor] = None
    if use_rowwise_fp8:
        assert rowwise_fp8_cfg is not None
        if moe_inp.shape[1] % rowwise_fp8_cfg.block_size != 0:
            raise RuntimeError(
                "Rowwise FP8 requires hidden dim divisible by block_size: "
                f"hidden={moe_inp.shape[1]} block_size={rowwise_fp8_cfg.block_size}"
            )
        scale_cols = moe_inp.shape[1] // rowwise_fp8_cfg.block_size
        dispatch_out_q = get_or_init_ep_no_sync_symm_tensor(
            self,
            name="dispatch_out_rowwise_fp8_q",
            shape=(dispatch_out_cap, moe_inp.shape[1]),
            dtype=torch.float8_e4m3fn,
            device=moe_inp.device,
        )
        dispatch_out_scales = get_or_init_ep_no_sync_symm_tensor(
            self,
            name="dispatch_out_rowwise_fp8_scales",
            shape=(dispatch_out_cap, scale_cols),
            dtype=torch.float8_e8m0fnu,
            device=moe_inp.device,
        )
        combine_in_q = get_or_init_ep_no_sync_symm_tensor(
            self,
            name="combine_in_rowwise_fp8_q",
            shape=(combine_in_cap, moe_inp.shape[1]),
            dtype=torch.float8_e4m3fn,
            device=moe_inp.device,
        )
        combine_in_scales = get_or_init_ep_no_sync_symm_tensor(
            self,
            name="combine_in_rowwise_fp8_scales",
            shape=(combine_in_cap, scale_cols),
            dtype=torch.float8_e8m0fnu,
            device=moe_inp.device,
        )

    routing_map = local_x_global_routed_expert_indices.view(
        -1, self.routed_experts_router.top_k
    ).int()

    with torch.no_grad():
        padded_batch_size_per_local_expert = recv_splits_by_src_local.sum(
            dim=0,
            dtype=torch.long,
        )

    with torch.no_grad():
        dst_ranks, dst_rows = build_rowwise_route_maps(
            self,
            routing_map=routing_map,
            allowed_splits=allowed_splits,
            keep_from_src_dest_local=keep_from_src_dest_local,
        )
        rowwise_nblocks = self.ep_no_sync_rowwise_nblocks

    if self.shared_experts is not None:
        with torch.cuda.stream(self.get_dense_stream()):
            if use_rowwise_fp8:
                assert rowwise_fp8_cfg is not None
                shared_out_up, shared_out_gate = shared_experts_forward1_rowwise_fp8(
                    self,
                    moe_inp,
                    use_fast_accum=rowwise_fp8_cfg.use_fast_accum,
                )
            else:
                shared_out_up, shared_out_gate = self.shared_experts.forward1(moe_inp.view(B, S, D))
    else:
        shared_out_up, shared_out_gate = None, None

    with nvtx.annotate("Rowwise Dispatch", color="green"):
        if use_rowwise_fp8:
            assert rowwise_fp8_cfg is not None
            assert dispatch_out_q is not None
            assert dispatch_out_scales is not None
            dispatch_rank_major = _DispatchRowwiseFP8Autograd.apply(
                moe_inp,
                dst_ranks,
                dst_rows,
                buffers.dispatch_out,
                dispatch_out_q,
                dispatch_out_scales,
                rowwise_fp8_cfg.block_size,
                group_name,
                self.ep_pg,
                rowwise_nblocks,
            )
        else:
            dispatch_rank_major = _DispatchRowwiseAutograd.apply(
                moe_inp,
                dst_ranks,
                dst_rows,
                buffers.dispatch_out,
                group_name,
                self.ep_pg,
                rowwise_nblocks,
            )

    dispatch_rank_major = self.routed_experts(
        dispatch_rank_major,
        padded_batch_size_per_local_expert,
        down_proj_out=(None if use_rowwise_fp8 else buffers.combine_in.detach()),
        up_proj_input_grad_out=(None if use_rowwise_fp8 else buffers.dispatch_out.detach()),
        use_rowwise_fp8=use_rowwise_fp8,
        rowwise_fp8_input_q=(dispatch_out_q if use_rowwise_fp8 else None),
        rowwise_fp8_input_scales=(dispatch_out_scales if use_rowwise_fp8 else None),
    )

    wait_stream_no_compile(
        this_stream=self.get_dense_stream(),
        other_stream=torch.cuda.current_stream(),
    )

    with nvtx.annotate("Rowwise Combine Merge", color="green"):
        route_probs = local_x_global_routed_expert_weights.view(
            -1, self.routed_experts_router.top_k
        )

        if use_rowwise_fp8:
            assert rowwise_fp8_cfg is not None
            assert combine_in_q is not None
            assert combine_in_scales is not None
            local_x = _RowwiseCombineWeightedFP8Autograd.apply(
                dispatch_rank_major,
                buffers.combine_in,
                dst_ranks,
                dst_rows,
                route_probs,
                combine_in_q,
                combine_in_scales,
                rowwise_fp8_cfg.block_size,
                group_name,
                self.ep_pg,
                self.ep_no_sync_rowwise_nblocks,
            )
        else:
            local_x = _RowwiseCombineWeightedAutograd.apply(
                dispatch_rank_major,
                buffers.combine_in,
                dst_ranks,
                dst_rows,
                route_probs,
                group_name,
                self.ep_pg,
                self.ep_no_sync_rowwise_nblocks,
            )

    if self.shared_experts is not None:
        assert shared_out_up is not None
        assert shared_out_gate is not None

        with torch.cuda.stream(self.get_dense_stream()):
            if use_rowwise_fp8:
                assert rowwise_fp8_cfg is not None
                shared_out = shared_experts_forward2_rowwise_fp8(
                    self,
                    shared_out_up,
                    shared_out_gate,
                    attn_res_out.shape,
                    use_fast_accum=rowwise_fp8_cfg.use_fast_accum,
                )
            else:
                shared_out = self.shared_experts.forward2(
                    shared_out_up, shared_out_gate, attn_res_out.shape
                )
            if self.shared_experts_router:
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
            else:
                mixed_shared_out = shared_out.squeeze(0)
    else:
        mixed_shared_out = None

    local_x = local_x.view(in_shape)
    wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())

    mlp_out = self._merge_routed_and_shared(local_x, mixed_shared_out)

    final_out = self._res_norm_mlp(attn_res_out, mlp_out)
    return self._attach_routed_aux_loss(final_out, routed_expert_router_aux_loss_info)
