from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import torch

from ...moe.utils import wait_stream_no_compile
from .checkpointing import get_rowwise_checkpoint_state
from .ep_no_sync_buffers import (
    compute_ep_no_sync_rank_capacity,
    get_ep_no_sync_group_name,
)
from .ep_no_sync_common import (
    padded_local_expert_splits_for_capacity,
    sync_tail_drop_allowed_splits_single_a2a,
)
from .ep_no_sync_rowwise_helpers import (
    accumulate_ep_no_sync_rowwise_metrics,
    build_rowwise_route_maps,
)
from .routed_experts import requires_host_side_split_sizes, use_torch_grouped_mm
from .tma_ibgda import (
    TmaIbgdaBackendConfig,
    tma_ibgda_empty_symmetric_expert_out,
    tma_ibgda_rowwise_combine_bf16,
    tma_ibgda_rowwise_dispatch_bf16,
)

if TYPE_CHECKING:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock


def _check_tma_ibgda_supported(block: "OLMoDDPTransformerBlock", x: torch.Tensor) -> None:
    if not block.ep_no_sync:
        raise RuntimeError("TMA/IBGDA rowwise backend requires ep_no_sync=True")
    if block.ep_no_sync_use_wave:
        raise RuntimeError(
            "TMA/IBGDA rowwise backend is separate from wave EP and does not use "
            "combined_forward_ep_wave"
        )
    if not block.ep_no_sync_use_rowwise_all_to_all:
        raise RuntimeError(
            "TMA/IBGDA backend replaces rowwise EP transport and requires "
            "ep_no_sync_use_rowwise_all_to_all=True"
        )
    if block.ep_no_sync_use_2d_all_to_all:
        raise RuntimeError("TMA/IBGDA rowwise backend does not support 2D all-to-all")
    if block.checkpoint_combined_ep_tbo:
        raise RuntimeError("TMA/IBGDA rowwise backend does not support TBO yet")
    if block.ep_pg is None:
        raise RuntimeError("TMA/IBGDA rowwise backend requires block.apply_ep(...) first")
    if x.ndim != 3:
        raise RuntimeError(f"TMA/IBGDA rowwise backend expects input [B, S, D], got {tuple(x.shape)}")
    if x.device.type != "cuda":
        raise RuntimeError("TMA/IBGDA rowwise backend requires CUDA inputs")
    if x.dtype != torch.bfloat16:
        raise RuntimeError(f"TMA/IBGDA rowwise backend requires bf16 inputs, got {x.dtype}")
    if block.routed_experts is None or block.routed_experts_router is None:
        raise RuntimeError("TMA/IBGDA rowwise backend requires routed experts and router")
    if block.num_local_routed_experts is None:
        raise RuntimeError("TMA/IBGDA rowwise backend requires local routed expert metadata")
    if requires_host_side_split_sizes():
        raise RuntimeError(
            "TMA/IBGDA rowwise backend does not support host-side split size communication"
        )
    if use_torch_grouped_mm() != True:
        raise RuntimeError("TMA/IBGDA rowwise backend requires torch.grouped_mm support")
    rowwise_fp8_cfg = block.rowwise_fp8
    if rowwise_fp8_cfg is not None and rowwise_fp8_cfg.enabled:
        raise RuntimeError("TMA/IBGDA rowwise backend starts with BF16 and rejects rowwise FP8")


def combined_forward_ep_no_sync_tma_ibgda(
    block: "OLMoDDPTransformerBlock",
    x: torch.Tensor,
    *,
    activation_checkpointing: Optional[bool] = None,
    accumulate_routed_aux_loss_metrics: Optional[bool] = None,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> torch.Tensor:
    """BF16 rowwise EP no-sync path using the OLMo-owned TMA/IBGDA backend."""

    del activation_checkpointing
    _check_tma_ibgda_supported(block, x)
    self = block
    assert self.routed_experts is not None
    assert self.routed_experts_router is not None
    assert self.ep_enabled
    assert self.num_local_routed_experts is not None
    assert self.ep_pg is not None

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
    num_out_tokens = local_x_global_routed_expert_indices.numel()
    num_input_tokens = moe_inp.shape[0]
    top_k = self.routed_experts_router.top_k
    if accumulate_routed_aux_loss_metrics is None:
        _, accumulate_routed_aux_loss_metrics = get_rowwise_checkpoint_state()

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
        accumulate_ep_no_sync_rowwise_metrics(
            self,
            drop_token_cnt=_drop_token_cnt,
            num_out_tokens=num_out_tokens,
            recv_splits_by_src_local=recv_splits_by_src_local,
            rank_capacity=rank_capacity,
        )

    routing_map = local_x_global_routed_expert_indices.view(-1, top_k).int()

    with torch.no_grad():
        batch_size_per_local_expert = recv_splits_by_src_local.sum(dim=0, dtype=torch.long)
        padded_local_expert_splits_for_capacity(
            recv_splits_by_src_local,
            rank_capacity=rank_capacity,
        )
        expert_batch_size_per_local_expert = batch_size_per_local_expert

    with torch.no_grad():
        dst_ranks, dst_rows = build_rowwise_route_maps(
            self,
            routing_map=routing_map,
            allowed_splits=allowed_splits,
            keep_from_src_dest_local=keep_from_src_dest_local,
        )

    if self.shared_experts is not None:
        with torch.cuda.stream(self.get_dense_stream()):
            shared_out_up, shared_out_gate = self.shared_experts.forward1(moe_inp.view(B, S, D))
    else:
        shared_out_up, shared_out_gate = None, None

    route_probs = local_x_global_routed_expert_weights.view(
        -1,
        self.routed_experts_router.top_k,
    ).to(dtype=torch.float32)
    if not route_probs.is_contiguous():
        route_probs = route_probs.contiguous()

    tma_num_sms = self.ep_no_sync_tma_ibgda_num_sms
    if tma_num_sms is None:
        tma_num_sms = self.ep_no_sync_rowwise_nblocks

    tma_config = TmaIbgdaBackendConfig(
        num_sms_dispatch=tma_num_sms,
        num_sms_combine=tma_num_sms,
        static_route_budget=rank_capacity,
        validate_gpu_route_preprocess=False,
        write_expert_out_to_symmetric=self.ep_no_sync_tma_ibgda_symmetric_expert_out,
    )

    dispatch_rank_major, dispatch_handle = tma_ibgda_rowwise_dispatch_bf16(
        moe_inp,
        dst_ranks,
        dst_rows,
        group_name,
        ep_world_size=self.ep_world_size,
        probs=None,
        rank_capacity=rank_capacity,
        process_group=self.ep_pg,
        config=tma_config,
    )

    write_symmetric_expert_out = tma_config.write_expert_out_to_symmetric
    if write_symmetric_expert_out and torch.is_grad_enabled():
        raise RuntimeError(
            "ep_no_sync_tma_ibgda_symmetric_expert_out is currently forward-only. "
            "Torch grouped_mm backward is not safe with NVSHMEM symmetric out= buffers yet."
        )
    symmetric_expert_out = None
    if write_symmetric_expert_out:
        symmetric_expert_out = tma_ibgda_empty_symmetric_expert_out(
            (dispatch_rank_major.shape[0], dispatch_rank_major.shape[1]),
            dtype=dispatch_rank_major.dtype,
            device=dispatch_rank_major.device,
            ep_world_size=self.ep_world_size,
            process_group=self.ep_pg,
            config=tma_config,
        )

    dispatch_rank_major = self.routed_experts(
        dispatch_rank_major,
        expert_batch_size_per_local_expert,
        down_proj_out=symmetric_expert_out,
        up_proj_input_grad_out=dispatch_rank_major.detach(),
        use_rowwise_fp8=False,
    )

    wait_stream_no_compile(
        this_stream=self.get_dense_stream(),
        other_stream=torch.cuda.current_stream(),
    )

    local_x = tma_ibgda_rowwise_combine_bf16(
        dispatch_rank_major,
        dst_ranks,
        dst_rows,
        group_name,
        ep_world_size=self.ep_world_size,
        probs=route_probs,
        handle=dispatch_handle,
        rank_capacity=rank_capacity,
        process_group=self.ep_pg,
        config=tma_config,
        expert_out_is_symmetric=False,
    )

    if self.shared_experts is not None:
        assert shared_out_up is not None
        assert shared_out_gate is not None
        with torch.cuda.stream(self.get_dense_stream()):
            shared_out = self.shared_experts.forward2(shared_out_up, shared_out_gate, attn_res_out.shape)
            mixed_shared_out = self._mix_shared_out(
                shared_out,
                local_x_global_shared_expert_weights,
                attn_res_out.shape,
            )
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
