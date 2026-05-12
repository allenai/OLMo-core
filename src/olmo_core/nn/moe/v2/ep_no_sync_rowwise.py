from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import torch

from ...moe.utils import wait_stream_no_compile
from .comm import (
    _DispatchRowwiseAutograd,
    _DispatchRowwiseFP8Autograd,
    _RowwiseFP8DispatchExpertsCombineAutograd,
    _RowwiseCombineWeightedAutograd,
    _RowwiseCombineWeightedFP8Autograd,
)
from .checkpointing import get_rowwise_checkpoint_state
from .ep_no_sync_common import sync_tail_drop_allowed_splits_single_a2a
from .ep_no_sync_buffers import (
    acquire_ep_no_sync_fp8_dispatch_out_lease,
    acquire_ep_no_sync_rowwise_lifetime_leases,
    compute_ep_no_sync_rank_capacity,
    get_cached_ep_no_sync_buffers,
    get_cached_ep_no_sync_rowwise_fp8_buffers,
    get_ep_no_sync_buffers,
    get_ep_no_sync_group_name,
    get_ep_no_sync_rowwise_fp8_buffers,
    use_ep_no_sync_rowwise_symm_combine_gather,
    use_ep_no_sync_rowwise_symm_combine_out,
    use_ep_no_sync_rowwise_symm_dispatch_in,
)
from .fp8 import (
    shared_experts_forward1_rowwise_fp8,
    shared_experts_forward2_rowwise_fp8,
)
from .ep_no_sync_rowwise_helpers import (
    accumulate_ep_no_sync_rowwise_metrics,
    build_rowwise_route_maps,
)
from .routed_experts import requires_host_side_split_sizes, use_torch_grouped_mm

if TYPE_CHECKING:
    from .block import MoEFusedV2TransformerBlock


def combined_forward_ep_no_sync_rowwise(
    block: MoEFusedV2TransformerBlock,
    x: torch.Tensor,
    *,
    activation_checkpointing: Optional[bool] = None,
    accumulate_routed_aux_loss_metrics: Optional[bool] = None,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> torch.Tensor:
    """Forward with EP no-sync using row-wise NVSHMEM dispatch/combine."""
    self = block
    assert self.routed_experts is not None
    assert self.routed_experts_router is not None
    assert self.ep_enabled
    assert self.num_local_routed_experts is not None
    assert use_torch_grouped_mm() == True, "EP no-sync implementation requires torch.grouped_mm support"
    assert not requires_host_side_split_sizes(), "EP no-sync implementation does not support host-side split size communication"
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
    num_input_tokens = moe_inp.shape[0]
    top_k = self.routed_experts_router.top_k
    if activation_checkpointing is None or accumulate_routed_aux_loss_metrics is None:
        activation_checkpointing, accumulate_routed_aux_loss_metrics = get_rowwise_checkpoint_state()
    use_symm_dispatch_in = (not use_rowwise_fp8) and use_ep_no_sync_rowwise_symm_dispatch_in(self)
    use_symm_combine_out = (
        (not use_rowwise_fp8)
        and (not activation_checkpointing)
        and use_ep_no_sync_rowwise_symm_combine_out(self)
    )
    use_symm_combine_gather = (
        (not use_rowwise_fp8)
        and (not activation_checkpointing)
        and use_ep_no_sync_rowwise_symm_combine_gather(self)
    )
    lease_lifetime_buffers = torch.is_grad_enabled() and not activation_checkpointing
    lease_dispatch_out = (not use_rowwise_fp8) and lease_lifetime_buffers
    lease_combine_out = use_symm_combine_out and lease_lifetime_buffers
    lease_combine_gather = use_symm_combine_gather and lease_lifetime_buffers
    use_fused_rowwise_fp8 = use_rowwise_fp8 and rowwise_fp8_cfg is not None and rowwise_fp8_cfg.fused_autograd

    with torch.no_grad():
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
        combine_out_cap = num_input_tokens
        accumulate_ep_no_sync_rowwise_metrics(
            self,
            drop_token_cnt=_drop_token_cnt,
            num_out_tokens=num_out_tokens,
            recv_splits_by_src_local=recv_splits_by_src_local,
            rank_capacity=rank_capacity,
        )

    buffers = None
    dispatch_out_q: Optional[torch.Tensor] = None
    dispatch_out_scales: Optional[torch.Tensor] = None
    combine_in_q: Optional[torch.Tensor] = None
    combine_in_scales: Optional[torch.Tensor] = None
    if use_rowwise_fp8:
        assert rowwise_fp8_cfg is not None
        fp8_buffers = get_cached_ep_no_sync_rowwise_fp8_buffers(
            self,
            dispatch_out_cap=dispatch_out_cap,
            combine_in_cap=combine_in_cap,
            d_model=moe_inp.shape[1],
            block_size=rowwise_fp8_cfg.block_size,
            device=moe_inp.device,
            need_dispatch_out=not lease_lifetime_buffers,
        )
        if fp8_buffers is None:
            fp8_buffers = get_ep_no_sync_rowwise_fp8_buffers(
                self,
                dispatch_out_cap=dispatch_out_cap,
                combine_in_cap=combine_in_cap,
                d_model=moe_inp.shape[1],
                block_size=rowwise_fp8_cfg.block_size,
                device=moe_inp.device,
                lease_dispatch_out=False,
                need_dispatch_out=not lease_lifetime_buffers,
            )
        if lease_lifetime_buffers:
            dispatch_out_lease = acquire_ep_no_sync_fp8_dispatch_out_lease(
                self,
                dispatch_out_cap=dispatch_out_cap,
                d_model=moe_inp.shape[1],
                block_size=rowwise_fp8_cfg.block_size,
                device=moe_inp.device,
            )
            fp8_buffers = replace(
                fp8_buffers,
                dispatch_out_q=dispatch_out_lease.tensor("dispatch_out_q"),
                dispatch_out_scales=dispatch_out_lease.tensor("dispatch_out_scales"),
                dispatch_out_lease=dispatch_out_lease,
            )
        dispatch_out_q = fp8_buffers.dispatch_out_q
        dispatch_out_scales = fp8_buffers.dispatch_out_scales
        combine_in_q = fp8_buffers.combine_in_q
        combine_in_scales = fp8_buffers.combine_in_scales
    else:
        buffers = get_cached_ep_no_sync_buffers(
            self,
            dispatch_in_cap=dispatch_in_cap,
            dispatch_out_cap=dispatch_out_cap,
            combine_in_cap=combine_in_cap,
            combine_out_cap=combine_out_cap,
            d_model=moe_inp.shape[-1],
            dtype=moe_inp.dtype,
            device=moe_inp.device,
            need_dispatch_in=use_symm_dispatch_in,
            need_dispatch_meta=False,
            need_dispatch_out=not lease_dispatch_out,
            need_combine_in=True,
            need_combine_meta=False,
            need_combine_out=use_symm_combine_out and not lease_combine_out,
            need_combine_gather=use_symm_combine_gather and not lease_combine_gather,
            combine_gather_cap=num_input_tokens,
            combine_gather_top_k=top_k,
        )
        if buffers is None:
            buffers = get_ep_no_sync_buffers(
                self,
                dispatch_in_cap=dispatch_in_cap,
                dispatch_out_cap=dispatch_out_cap,
                combine_in_cap=combine_in_cap,
                combine_out_cap=combine_out_cap,
                d_model=moe_inp.shape[-1],
                dtype=moe_inp.dtype,
                device=moe_inp.device,
                need_dispatch_in=use_symm_dispatch_in,
                need_dispatch_meta=False,
                need_dispatch_out=not lease_dispatch_out,
                need_combine_in=True,
                need_combine_meta=False,
                need_combine_out=use_symm_combine_out and not lease_combine_out,
                need_combine_gather=use_symm_combine_gather and not lease_combine_gather,
                combine_gather_cap=num_input_tokens,
                combine_gather_top_k=top_k,
            )
        if lease_dispatch_out or lease_combine_out or lease_combine_gather:
            leases = acquire_ep_no_sync_rowwise_lifetime_leases(
                self,
                dispatch_out_cap=dispatch_out_cap,
                combine_out_cap=combine_out_cap,
                combine_gather_cap=num_input_tokens,
                combine_gather_top_k=top_k,
                d_model=moe_inp.shape[-1],
                dtype=moe_inp.dtype,
                device=moe_inp.device,
                need_dispatch_out=lease_dispatch_out,
                need_combine_out=lease_combine_out,
                need_combine_gather=lease_combine_gather,
            )
            buffers = replace(
                buffers,
                dispatch_out=(
                    leases.dispatch_out_lease.tensor("dispatch_out")
                    if leases.dispatch_out_lease is not None
                    else buffers.dispatch_out
                ),
                dispatch_out_is_shared=(
                    True if leases.dispatch_out_lease is not None else buffers.dispatch_out_is_shared
                ),
                combine_out=(
                    leases.combine_out_lease.tensor("combine_out")
                    if leases.combine_out_lease is not None
                    else buffers.combine_out
                ),
                combine_out_is_shared=(
                    True if leases.combine_out_lease is not None else buffers.combine_out_is_shared
                ),
                combine_gather=(
                    leases.combine_gather_lease.tensor("combine_gather")
                    if leases.combine_gather_lease is not None
                    else buffers.combine_gather
                ),
                dispatch_out_lease=leases.dispatch_out_lease,
                combine_out_lease=leases.combine_out_lease,
                combine_gather_lease=leases.combine_gather_lease,
            )

    routing_map = local_x_global_routed_expert_indices.view(
        -1, top_k
    ).int()

    with torch.no_grad():
        padded_batch_size_per_local_expert = recv_splits_by_src_local.sum(
            dim=0,
            dtype=torch.long,
        )
        routed_expert_offsets = torch.cumsum(
            padded_batch_size_per_local_expert.to(dtype=torch.int32),
            dim=0,
            dtype=torch.int32,
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

    route_probs = local_x_global_routed_expert_weights.view(
        -1, self.routed_experts_router.top_k
    )

    if use_fused_rowwise_fp8:
        assert rowwise_fp8_cfg is not None
        assert dispatch_out_q is not None
        assert dispatch_out_scales is not None
        assert combine_in_q is not None
        assert combine_in_scales is not None
        routed_experts = self.routed_experts
        routed_fp8_cfg = routed_experts.rowwise_fp8
        if routed_fp8_cfg is None or not routed_fp8_cfg.enabled:
            raise RuntimeError("fused rowwise FP8 autograd requires routed expert rowwise_fp8 to be enabled")
        up_gate_weight = routed_experts._rowwise_fp8_up_gate_weight
        down_weight = routed_experts._rowwise_fp8_down_weight
        up_gate_prequant = up_gate_weight.require_prequantized_rhs()
        up_gate_prequant_t = up_gate_weight.require_prequantized_rhs_for_dgrad()
        down_prequant = down_weight.require_prequantized_rhs()
        down_prequant_t = down_weight.require_prequantized_rhs_for_dgrad()

        up_wgrad_sink = None
        down_wgrad_sink = None
        up_wgrad_sink_transpose_last2 = False
        down_wgrad_sink_transpose_last2 = False
        up_gate_anchor = routed_experts.w_up_gate.transpose(1, 2)
        down_anchor = routed_experts.w_down
        if routed_fp8_cfg.fp8_only_params:
            up_wgrad_sink = up_gate_weight
            down_wgrad_sink = down_weight
            up_wgrad_sink_transpose_last2 = True
            up_gate_anchor = up_gate_anchor.detach()
            down_anchor = down_anchor.detach()

        local_x = _RowwiseFP8DispatchExpertsCombineAutograd.apply(
            moe_inp,
            dst_ranks,
            dst_rows,
            routed_expert_offsets,
            route_probs,
            dispatch_out_q,
            dispatch_out_scales,
            combine_in_q,
            combine_in_scales,
            up_gate_anchor,
            down_anchor,
            up_gate_prequant,
            up_gate_prequant_t,
            down_prequant,
            down_prequant_t,
            fp8_buffers.dispatch_out_lease,
            rowwise_fp8_cfg.block_size,
            rowwise_fp8_cfg.use_fast_accum,
            rowwise_fp8_cfg.fused_autograd_recompute_swiglu,
            group_name,
            self.ep_pg,
            rowwise_nblocks,
            up_wgrad_sink,
            up_wgrad_sink_transpose_last2,
            False,
            down_wgrad_sink,
            down_wgrad_sink_transpose_last2,
            False,
        )
    elif use_rowwise_fp8:
        assert rowwise_fp8_cfg is not None
        assert dispatch_out_q is not None
        assert dispatch_out_scales is not None
        dispatch_rank_major = _DispatchRowwiseFP8Autograd.apply(
            moe_inp,
            dst_ranks,
            dst_rows,
            dispatch_out_q,
            dispatch_out_scales,
            fp8_buffers.dispatch_out_lease,
            rowwise_fp8_cfg.block_size,
            group_name,
            self.ep_pg,
            rowwise_nblocks,
        )
    else:
        assert buffers is not None
        source_input_aliases_symm_input = False
        grad_out_aliases_symm_out = True
        dispatch_rank_major = _DispatchRowwiseAutograd.apply(
            moe_inp,
            buffers.dispatch_in if use_symm_dispatch_in else None,
            dst_ranks,
            dst_rows,
            buffers.dispatch_out,
            buffers.dispatch_out_lease,
            group_name,
            self.ep_pg,
            rowwise_nblocks,
            source_input_aliases_symm_input,
            grad_out_aliases_symm_out,
            True,
            False,
        )

    if not use_fused_rowwise_fp8:
        dispatch_rank_major = self.routed_experts(
            dispatch_rank_major,
            padded_batch_size_per_local_expert,
            down_proj_out=(None if use_rowwise_fp8 else buffers.combine_in.detach()),  # type: ignore[union-attr]
            up_proj_input_grad_out=(None if use_rowwise_fp8 else buffers.dispatch_out.detach()),  # type: ignore[union-attr]
            use_rowwise_fp8=use_rowwise_fp8,
            rowwise_fp8_input_q=(dispatch_out_q if use_rowwise_fp8 else None),
            rowwise_fp8_input_scales=(dispatch_out_scales if use_rowwise_fp8 else None),
        )

    wait_stream_no_compile(
        this_stream=self.get_dense_stream(),
        other_stream=torch.cuda.current_stream(),
    )

    if use_fused_rowwise_fp8:
        pass
    elif use_rowwise_fp8:
        assert rowwise_fp8_cfg is not None
        assert combine_in_q is not None
        assert combine_in_scales is not None
        local_x = _RowwiseCombineWeightedFP8Autograd.apply(
            dispatch_rank_major,
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
        assert buffers is not None
        expert_out_aliases_symm_expert_out = True
        local_x = _RowwiseCombineWeightedAutograd.apply(
            dispatch_rank_major,
            buffers.combine_in,
            buffers.combine_out if use_symm_combine_out else None,
            buffers.combine_out_lease if use_symm_combine_out else None,
            buffers.combine_gather if use_symm_combine_gather else None,
            buffers.combine_gather_lease if use_symm_combine_gather else None,
            dst_ranks,
            dst_rows,
            route_probs,
            group_name,
            self.ep_pg,
            self.ep_no_sync_rowwise_nblocks,
            expert_out_aliases_symm_expert_out,
            True,
            False,
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
                shared_out = self.shared_experts.forward2(shared_out_up, shared_out_gate, attn_res_out.shape)
            if self.shared_experts_router:
                assert local_x_global_shared_expert_weights is not None
                _, _, E_s = local_x_global_shared_expert_weights.shape
                mixed_shared_out = torch.bmm(
                    local_x_global_shared_expert_weights.to(shared_out.dtype).reshape(B * S, 1, E_s),
                    shared_out.permute(1, 2, 0, 3).contiguous().view(B * S, E_s, D),
                ).squeeze(1).view(B, S, D)
            else:
                mixed_shared_out = shared_out.squeeze(0)
    else:
        mixed_shared_out = None

    local_x = local_x.view(in_shape)
    wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())

    mlp_out = self._merge_routed_and_shared(local_x, mixed_shared_out)

    final_out = self._res_norm_mlp(attn_res_out, mlp_out)
    return self._attach_routed_aux_loss(
        final_out,
        routed_expert_router_aux_loss_info,
        accumulate_metrics=accumulate_routed_aux_loss_metrics,
    )
