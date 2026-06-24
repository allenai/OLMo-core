from __future__ import annotations

import warnings
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
from .ep_config import ExpertParallelPath
from .routed_experts import requires_host_side_split_sizes, use_torch_grouped_mm
from .tma_ibgda import (
    TmaIbgdaBackendConfig,
    tma_ibgda_empty_symmetric_expert_out,
    tma_ibgda_rowwise_combine_bf16,
    tma_ibgda_rowwise_dispatch_bf16,
)

if TYPE_CHECKING:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock


_TMA_IBGDA_EXPERIMENTAL_WARNING_EMITTED = False


def _warn_tma_ibgda_experimental() -> None:
    global _TMA_IBGDA_EXPERIMENTAL_WARNING_EMITTED
    if _TMA_IBGDA_EXPERIMENTAL_WARNING_EMITTED:
        return
    warnings.warn(
        "combined_forward_ep_no_sync_tma_ibgda is experimental: it is an "
        "opt-in BF16 rowwise EP transport backend, not the production default. "
        "Use rowwise_nvshmem for production runs unless this path has been "
        "validated on the target shape/node.",
        RuntimeWarning,
        stacklevel=2,
    )
    _TMA_IBGDA_EXPERIMENTAL_WARNING_EMITTED = True


def _resolve_symmetric_expert_out_flag(
    *,
    requested: bool,
    returned_expert_out: torch.Tensor,
    symmetric_expert_out: Optional[torch.Tensor],
) -> bool:
    if not requested:
        return False
    if symmetric_expert_out is None:
        raise RuntimeError(
            "TMA/IBGDA symmetric expert output was requested but no symmetric buffer "
            "was allocated"
        )

    same_tensor = (
        returned_expert_out.data_ptr() == symmetric_expert_out.data_ptr()
        and returned_expert_out.shape == symmetric_expert_out.shape
        and returned_expert_out.stride() == symmetric_expert_out.stride()
        and returned_expert_out.dtype == symmetric_expert_out.dtype
        and returned_expert_out.device == symmetric_expert_out.device
    )
    if not same_tensor:
        raise RuntimeError(
            "TMA/IBGDA symmetric expert output was requested, but routed experts "
            "returned a different tensor. This path requires the routed expert down "
            "projection to write directly into the symmetric output buffer."
        )
    return True


def _check_tma_ibgda_supported(block: "OLMoDDPTransformerBlock", x: torch.Tensor) -> None:
    if block.ep.path != ExpertParallelPath.rowwise_tma_ibgda:
        raise RuntimeError(
            "TMA/IBGDA rowwise backend requires "
            f"path={ExpertParallelPath.rowwise_tma_ibgda!r}"
        )
    if block.ep.schedule.value == "tbo" or block.ep.checkpoint_tbo:
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
    """Experimental BF16 rowwise EP no-sync path using TMA/IBGDA transport.

    This path is an opt-in rowwise transport replacement for experiments and
    benchmarking. It is guarded to BF16 CUDA execution without TBO/FP8 and
    should not replace the default rowwise NVSHMEM backend until validated for
    the specific training shape and cluster.
    """

    del activation_checkpointing
    _check_tma_ibgda_supported(block, x)
    _warn_tma_ibgda_experimental()

    block_inp = x
    del x

    attn_res_out = block._checkpointed_res_norm_attn(block_inp, **kwargs)

    kwargs.pop("max_doc_len", None)
    kwargs.pop("cu_doc_lens", None)
    return _combined_forward_ep_no_sync_tma_ibgda_moe_eager(
        block,
        attn_res_out,
        accumulate_routed_aux_loss_metrics=accumulate_routed_aux_loss_metrics,
        loss_div_factor=loss_div_factor,
    )


@torch.compiler.disable
def _combined_forward_ep_no_sync_tma_ibgda_moe_eager(
    block: "OLMoDDPTransformerBlock",
    attn_res_out: torch.Tensor,
    *,
    accumulate_routed_aux_loss_metrics: Optional[bool] = None,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
) -> torch.Tensor:
    """Run the TMA/IBGDA rowwise MoE section outside torch.compile.

    PyTorch AOTAutograd currently fails to partition the router sort/control
    dependency in this path. Keep the rowwise TMA/IBGDA transport and route-map
    setup eager while allowing the attention prefix of the block to remain
    compile eligible.
    """

    self = block
    assert self.routed_experts is not None
    assert self.routed_experts_router is not None
    assert self.ep_enabled
    assert self.num_local_routed_experts is not None
    assert self.ep_pg is not None

    group_name = get_ep_no_sync_group_name(self)
    B, S, D = attn_res_out.shape
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

    tma_config = TmaIbgdaBackendConfig(
        num_sms_dispatch=self.ep.resolved_tma_ibgda_dispatch_num_sms,
        num_sms_combine=self.ep.resolved_tma_ibgda_combine_num_sms,
        num_sms_preprocess=self.ep.resolved_tma_ibgda_preprocess_num_sms,
        static_route_budget=rank_capacity,
        validate_gpu_route_preprocess=False,
        write_expert_out_to_symmetric=self.ep.tma_ibgda_symmetric_expert_out,
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
    expert_out_is_symmetric = _resolve_symmetric_expert_out_flag(
        requested=write_symmetric_expert_out,
        returned_expert_out=dispatch_rank_major,
        symmetric_expert_out=symmetric_expert_out,
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
        expert_out_is_symmetric=expert_out_is_symmetric,
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
