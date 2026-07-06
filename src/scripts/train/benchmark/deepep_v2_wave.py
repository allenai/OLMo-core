from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .deepep_v2_core import (
    DeepEpV2ForwardResult,
    DeepEpV2State,
    DeepEpV2WaveComputeResult,
    DeepEpV2WaveDispatchResult,
    DeepEpV2WaveForwardResult,
    DeepEpV2WaveInput,
    _deep_ep_wait,
    _align_int,
    _deepep_v2_combine,
    _deepep_v2_compute_experts,
    _deepep_v2_compute_experts_nonexpanded,
    _deepep_v2_compute_experts_static_expanded,
    _deepep_v2_dispatch,
    _deepep_v2_dispatch_nonexpanded,
    _deepep_v2_dispatch_static_expanded,
    _deepep_v2_forward,
    _deepep_v2_forward_from_topk,
    _deepep_v2_forward_from_topk_nonexpanded,
    _deepep_v2_prepare_expert_grad,
    _deepep_v2_weight_expert_output,
    _run_deepep_v2_backward_from_result,
)


@dataclass
class DeepEpV2StaticExpandedWorkspace:
    recv_x: torch.Tensor
    expanded_topk_weights: torch.Tensor
    weighted_expert_out: torch.Tensor
    combined_parts: list[torch.Tensor] | None = None
    merged_x: torch.Tensor | None = None


def _build_deepep_v2_wave_inputs(
    state: DeepEpV2State,
    *,
    num_waves: int,
) -> list[DeepEpV2WaveInput]:
    if num_waves < 1:
        raise RuntimeError("--deepep-wave-num-waves must be >= 1")
    if num_waves > state.num_local_experts:
        raise RuntimeError(
            "--deepep-wave-num-waves cannot exceed local experts "
            f"({num_waves} > {state.num_local_experts})"
        )

    tokens = state.source_input.shape[0]
    if num_waves > tokens:
        raise RuntimeError(
            "--deepep-wave-num-waves cannot exceed source tokens "
            f"({num_waves} > {tokens})"
        )

    wave_inputs: list[DeepEpV2WaveInput] = []
    wave_base = 0
    for wave_idx in range(num_waves):
        token_start = (wave_idx * tokens) // num_waves
        token_end = ((wave_idx + 1) * tokens) // num_waves
        if token_start == token_end:
            raise RuntimeError(
                f"empty DeepEP source-token wave {wave_idx}: tokens={tokens} "
                f"num_waves={num_waves}"
            )
        topk_idx = state.topk_idx.narrow(
            0,
            token_start,
            token_end - token_start,
        ).contiguous()
        topk_weights = state.topk_weights.narrow(
            0,
            token_start,
            token_end - token_start,
        ).contiguous()
        valid_idx = topk_idx.to(dtype=torch.long)
        valid_idx = valid_idx[valid_idx >= 0]
        global_counts = torch.bincount(valid_idx, minlength=state.num_experts).to(
            dtype=torch.long,
        )
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
        local_start = state.rank * state.num_local_experts
        local_end = local_start + state.num_local_experts
        local_counts = [
            int(v)
            for v in global_counts[local_start:local_end].detach().cpu().tolist()
        ]
        wave_rows = sum(
            _align_int(count, state.expert_alignment)
            for count in local_counts
        )
        batch_size_per_expert = torch.zeros(
            (state.num_local_experts,),
            device=state.source_input.device,
            dtype=torch.int32,
        )
        batch_size_per_expert.copy_(
            torch.tensor(
                local_counts,
                device=state.source_input.device,
                dtype=torch.int32,
            )
        )
        wave_inputs.append(
            DeepEpV2WaveInput(
                wave_idx=wave_idx,
                expert_start=0,
                expert_end=state.num_local_experts,
                wave_base=wave_base,
                wave_end=wave_base + wave_rows,
                batch_size_per_expert=batch_size_per_expert,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                source_start=token_start,
                source_end=token_end,
            )
        )
        wave_base += wave_rows
    return wave_inputs


def _deepep_v2_wave_label(label: str, wave: DeepEpV2WaveInput) -> str:
    return (
        f"{label}/wave_{wave.wave_idx}_tokens_{wave.source_start}_{wave.source_end}"
        f"_experts_{wave.expert_start}_{wave.expert_end}"
    )


def _sum_deepep_v2_wave_outputs(
    wave_outputs: list[torch.Tensor],
    *,
    label: str,
) -> torch.Tensor:
    if not wave_outputs:
        raise RuntimeError("DeepEP V2 wave forward produced no wave outputs")
    torch.cuda.nvtx.range_push(label)
    try:
        combined = wave_outputs[0]
        for partial in wave_outputs[1:]:
            combined = combined + partial
    finally:
        torch.cuda.nvtx.range_pop()
    return combined


def _merge_deepep_v2_wave_outputs(
    wave_outputs: list[torch.Tensor],
    *,
    wave_inputs: list[DeepEpV2WaveInput],
    label: str,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if not wave_outputs:
        raise RuntimeError("DeepEP V2 wave produced no wave outputs")

    expected_start = 0
    source_token_major = len(wave_outputs) == len(wave_inputs)
    for wave, output in zip(wave_inputs, wave_outputs):
        if (
            wave.source_start != expected_start
            or wave.source_rows <= 0
            or output.shape[0] != wave.source_rows
        ):
            source_token_major = False
            break
        expected_start = wave.source_end

    if not source_token_major:
        return _sum_deepep_v2_wave_outputs(wave_outputs, label=label)

    if len(wave_outputs) == 1 and out is None:
        return wave_outputs[0]

    torch.cuda.nvtx.range_push(label)
    try:
        if out is not None:
            for wave, partial in zip(wave_inputs, wave_outputs):
                out.narrow(0, wave.source_start, wave.source_rows).copy_(partial)
            return out
        return torch.cat(wave_outputs, dim=0)
    finally:
        torch.cuda.nvtx.range_pop()


def _deepep_v2_wave_grad_combined_x(
    result: DeepEpV2WaveForwardResult,
    wave_result: DeepEpV2ForwardResult,
) -> torch.Tensor:
    if result.grad_combined_x is None:
        result.grad_combined_x = torch.ones_like(result.combined_x)
    grad_combined_x = result.grad_combined_x
    wave = wave_result.wave_input
    if (
        wave is not None
        and grad_combined_x.shape[0] != wave_result.combined_x.shape[0]
        and wave_result.combined_x.shape[0] == wave.source_rows
    ):
        return grad_combined_x.narrow(0, wave.source_start, wave.source_rows)
    return grad_combined_x


def _deepep_v2_static_expanded_buffers(
    state: DeepEpV2State,
    wave_inputs: list[DeepEpV2WaveInput],
    *,
    preallocate_combine: bool = False,
) -> DeepEpV2StaticExpandedWorkspace:
    expanded_rows = max((wave.wave_end for wave in wave_inputs), default=0)
    recv_x = torch.empty(
        (expanded_rows, state.source_input.shape[1]),
        device=state.source_input.device,
        dtype=state.source_input.dtype,
    )
    expanded_topk_weights = torch.empty(
        (expanded_rows,),
        device=state.topk_weights.device,
        dtype=state.topk_weights.dtype,
    )
    weighted_expert_out = torch.empty_like(recv_x)
    combined_parts = None
    merged_x = None
    if preallocate_combine:
        combined_parts = [
            torch.empty(
                (wave.source_rows, state.source_input.shape[1]),
                device=state.source_input.device,
                dtype=state.source_input.dtype,
            )
            for wave in wave_inputs
        ]
        merged_x = torch.empty_like(state.source_input)
    return DeepEpV2StaticExpandedWorkspace(
        recv_x=recv_x,
        expanded_topk_weights=expanded_topk_weights,
        weighted_expert_out=weighted_expert_out,
        combined_parts=combined_parts,
        merged_x=merged_x,
    )


def _deepep_v2_wave_combine_out(
    workspace: DeepEpV2StaticExpandedWorkspace | None,
    wave: DeepEpV2WaveInput,
) -> torch.Tensor | None:
    if workspace is None or workspace.combined_parts is None:
        return None
    return workspace.combined_parts[wave.wave_idx]


def _deepep_v2_wave_forward_sequential(
    state: DeepEpV2State,
    *,
    wave_inputs: list[DeepEpV2WaveInput],
    wave_layout: str,
    label: str,
    track_expert_grad: bool,
    do_cpu_sync: bool,
    workspace: DeepEpV2StaticExpandedWorkspace | None = None,
) -> DeepEpV2WaveForwardResult:
    wave_results: list[DeepEpV2ForwardResult] = []
    wave_outputs: list[torch.Tensor] = []
    static_buffers = None
    if wave_layout == "expand_static":
        static_buffers = (
            workspace
            if workspace is not None
            else _deepep_v2_static_expanded_buffers(state, wave_inputs)
        )
    for wave in wave_inputs:
        wave_label = _deepep_v2_wave_label(label, wave)
        source_input = state.source_input.narrow(
            0,
            wave.source_start,
            wave.source_rows,
        )
        if wave_layout == "expand":
            result = _deepep_v2_forward_from_topk(
                state,
                source_input=source_input,
                topk_idx=wave.topk_idx,
                topk_weights=wave.topk_weights,
                label=wave_label,
                track_expert_grad=track_expert_grad,
                async_with_compute_stream=state.async_with_compute_stream,
                do_cpu_sync=do_cpu_sync,
            )
        elif wave_layout == "expand_static":
            assert static_buffers is not None
            recv_x, expanded_topk_weights, handle, _dispatch_event = (
                _deepep_v2_dispatch_static_expanded(
                    state,
                    wave=wave,
                    recv_x_out=static_buffers.recv_x,
                    recv_topk_weights_out=static_buffers.expanded_topk_weights,
                    label=f"{wave_label}/dispatch_static",
                    async_with_compute_stream=state.async_with_compute_stream,
                    do_cpu_sync=do_cpu_sync,
                )
            )
            recv_x_for_experts, expert_out, expanded_weights, expert_out_is_weighted = (
                _deepep_v2_compute_experts_static_expanded(
                    state,
                    recv_x=recv_x,
                    expanded_topk_weights=expanded_topk_weights,
                    weighted_expert_out=static_buffers.weighted_expert_out,
                    wave=wave,
                    label=f"{wave_label}/experts_static",
                    track_expert_grad=track_expert_grad,
                )
            )
            combined_x, _combine_event = _deepep_v2_combine(
                state,
                weighted_expert_out=static_buffers.weighted_expert_out,
                handle=handle,
                label=f"{wave_label}/combine_static",
                async_with_compute_stream=state.async_with_compute_stream,
                combined_x_out=_deepep_v2_wave_combine_out(static_buffers, wave),
            )
            result = DeepEpV2ForwardResult(
                recv_x=recv_x_for_experts,
                expanded_topk_weights=expanded_weights,
                expert_out=expert_out,
                combined_x=combined_x,
                handle=handle,
                expert_out_is_weighted=expert_out_is_weighted,
                wave_input=wave,
                static_wave=wave,
                static_recv_x_global=recv_x,
            )
        elif wave_layout == "nonexpand_pack":
            result = _deepep_v2_forward_from_topk_nonexpanded(
                state,
                source_input=source_input,
                topk_idx=wave.topk_idx,
                topk_weights=wave.topk_weights,
                label=wave_label,
                track_expert_grad=track_expert_grad,
                async_with_compute_stream=state.async_with_compute_stream,
                do_cpu_sync=do_cpu_sync,
            )
        else:
            raise ValueError(wave_layout)
        result.wave_input = wave
        wave_results.append(result)
        wave_outputs.append(result.combined_x)

    return DeepEpV2WaveForwardResult(
        wave_results=wave_results,
        combined_x=_merge_deepep_v2_wave_outputs(
            wave_outputs,
            wave_inputs=wave_inputs,
            label=f"{label}/merge_waves",
            out=static_buffers.merged_x if static_buffers is not None else None,
        ),
    )


def _deepep_v2_wave_forward_overlapped(
    state: DeepEpV2State,
    *,
    wave_inputs: list[DeepEpV2WaveInput],
    wave_layout: str,
    label: str,
    track_expert_grad: bool,
    do_cpu_sync: bool,
    workspace: DeepEpV2StaticExpandedWorkspace | None = None,
) -> DeepEpV2WaveForwardResult:
    if (
        state.wave_dispatch_stream is None
        or state.wave_compute_stream is None
        or state.wave_combine_stream is None
    ):
        raise RuntimeError("DeepEP V2 wave streams were not initialized")

    async_mode = True
    static_buffers = None
    if wave_layout == "expand_static":
        static_buffers = (
            workspace
            if workspace is not None
            else _deepep_v2_static_expanded_buffers(state, wave_inputs)
        )

    dispatch_results: list[DeepEpV2WaveDispatchResult] = []
    for wave in wave_inputs:
        wave_label = _deepep_v2_wave_label(label, wave)
        source_input = state.source_input.narrow(
            0,
            wave.source_start,
            wave.source_rows,
        )
        with torch.cuda.stream(state.wave_dispatch_stream):
            if wave_layout == "expand":
                recv_x, expanded_topk_weights, handle, dispatch_event = _deepep_v2_dispatch(
                    state,
                    source_input=source_input,
                    topk_idx=wave.topk_idx,
                    topk_weights=wave.topk_weights,
                    label=f"{wave_label}/dispatch",
                    async_with_compute_stream=async_mode,
                    do_cpu_sync=do_cpu_sync,
                    wait_for_completion=False,
                )
                recv_topk_idx = None
                recv_topk_weights = None
            elif wave_layout == "expand_static":
                assert static_buffers is not None
                recv_x, expanded_topk_weights, handle, dispatch_event = (
                    _deepep_v2_dispatch_static_expanded(
                        state,
                        wave=wave,
                        recv_x_out=static_buffers.recv_x,
                        recv_topk_weights_out=static_buffers.expanded_topk_weights,
                        label=f"{wave_label}/dispatch_static",
                        async_with_compute_stream=async_mode,
                        do_cpu_sync=do_cpu_sync,
                        wait_for_completion=False,
                    )
                )
                recv_topk_idx = None
                recv_topk_weights = None
            elif wave_layout == "nonexpand_pack":
                recv_x, recv_topk_idx, recv_topk_weights, handle, dispatch_event = (
                    _deepep_v2_dispatch_nonexpanded(
                        state,
                        source_input=source_input,
                        topk_idx=wave.topk_idx,
                        topk_weights=wave.topk_weights,
                        label=f"{wave_label}/dispatch_nonexpanded",
                        async_with_compute_stream=async_mode,
                        do_cpu_sync=do_cpu_sync,
                        wait_for_completion=False,
                    )
                )
                expanded_topk_weights = None
            else:
                raise ValueError(wave_layout)

        dispatch_results.append(
            DeepEpV2WaveDispatchResult(
                wave=wave,
                recv_x=recv_x,
                expanded_topk_weights=expanded_topk_weights,
                recv_topk_idx=recv_topk_idx,
                recv_topk_weights=recv_topk_weights,
                handle=handle,
                event=dispatch_event,
            )
        )

    compute_results: list[DeepEpV2WaveComputeResult] = []
    for dispatched in dispatch_results:
        wave = dispatched.wave
        wave_label = _deepep_v2_wave_label(label, wave)
        with torch.cuda.stream(state.wave_compute_stream):
            _deep_ep_wait(dispatched.event, async_with_compute_stream=async_mode)
            if wave_layout == "expand":
                assert dispatched.expanded_topk_weights is not None
                recv_x_for_experts, expert_out, expanded_weights, expert_out_is_weighted = _deepep_v2_compute_experts(
                    state,
                    recv_x=dispatched.recv_x,
                    expanded_topk_weights=dispatched.expanded_topk_weights,
                    handle=dispatched.handle,
                    label=f"{wave_label}/experts",
                    track_expert_grad=track_expert_grad,
                    weight_in_swiglu=state.weighting_mode == "swiglu",
                )
                weighted_expert_out = (
                    expert_out
                    if expert_out_is_weighted
                    else _deepep_v2_weight_expert_output(
                        expert_out,
                        expanded_weights,
                        label=wave_label,
                        mode=state.weighting_mode,
                    )
                )
            elif wave_layout == "expand_static":
                assert static_buffers is not None
                assert dispatched.expanded_topk_weights is not None
                weighted_expert_out = static_buffers.weighted_expert_out
                recv_x_for_experts, expert_out, expanded_weights, expert_out_is_weighted = (
                    _deepep_v2_compute_experts_static_expanded(
                        state,
                        recv_x=dispatched.recv_x,
                        expanded_topk_weights=dispatched.expanded_topk_weights,
                        weighted_expert_out=weighted_expert_out,
                        wave=wave,
                        label=f"{wave_label}/experts_static",
                        track_expert_grad=track_expert_grad,
                    )
                )
            else:
                assert dispatched.recv_topk_idx is not None
                assert dispatched.recv_topk_weights is not None
                recv_x_for_experts, weighted_expert_out, expanded_weights = (
                    _deepep_v2_compute_experts_nonexpanded(
                        state,
                        recv_x=dispatched.recv_x,
                        recv_topk_idx=dispatched.recv_topk_idx,
                        recv_topk_weights=dispatched.recv_topk_weights,
                        handle=dispatched.handle,
                        label=f"{wave_label}/experts_nonexpanded",
                        track_expert_grad=track_expert_grad,
                    )
                )
                expert_out = weighted_expert_out
                expert_out_is_weighted = True
            compute_done = torch.cuda.Event(enable_timing=False)
            compute_done.record()

        compute_results.append(
            DeepEpV2WaveComputeResult(
                wave=wave,
                recv_x_for_experts=recv_x_for_experts,
                expert_out=expert_out,
                expanded_weights=expanded_weights,
                expert_out_is_weighted=expert_out_is_weighted,
                weighted_expert_out=weighted_expert_out,
                handle=dispatched.handle,
                recv_x_global=dispatched.recv_x if wave_layout == "expand_static" else None,
                compute_done=compute_done,
            )
        )

    wave_results: list[DeepEpV2ForwardResult] = []
    wave_outputs: list[torch.Tensor] = []
    combine_events: list[object] = []
    for computed in compute_results:
        wave = computed.wave
        wave_label = _deepep_v2_wave_label(label, wave)
        with torch.cuda.stream(state.wave_combine_stream):
            state.wave_combine_stream.wait_event(computed.compute_done)
            combined_x, combine_event = _deepep_v2_combine(
                state,
                weighted_expert_out=computed.weighted_expert_out,
                handle=computed.handle,
                label=f"{wave_label}/combine",
                async_with_compute_stream=async_mode,
                wait_for_completion=False,
                combined_x_out=_deepep_v2_wave_combine_out(static_buffers, wave),
            )

        wave_results.append(
            DeepEpV2ForwardResult(
                recv_x=computed.recv_x_for_experts,
                expanded_topk_weights=computed.expanded_weights,
                expert_out=computed.expert_out,
                combined_x=combined_x,
                handle=computed.handle,
                expert_out_is_weighted=computed.expert_out_is_weighted,
                wave_input=wave,
                static_wave=wave if wave_layout == "expand_static" else None,
                static_recv_x_global=computed.recv_x_global,
            )
        )
        wave_outputs.append(combined_x)
        combine_events.append(combine_event)

    for event in combine_events:
        _deep_ep_wait(event, async_with_compute_stream=async_mode)

    return DeepEpV2WaveForwardResult(
        wave_results=wave_results,
        combined_x=_merge_deepep_v2_wave_outputs(
            wave_outputs,
            wave_inputs=wave_inputs,
            label=f"{label}/merge_waves",
            out=static_buffers.merged_x if static_buffers is not None else None,
        ),
    )


def _deepep_v2_wave_forward(
    state: DeepEpV2State,
    *,
    wave_inputs: list[DeepEpV2WaveInput],
    wave_layout: str,
    label: str,
    track_expert_grad: bool,
    overlap: bool,
    do_cpu_sync: bool,
    workspace: DeepEpV2StaticExpandedWorkspace | None = None,
) -> DeepEpV2WaveForwardResult:
    if overlap:
        return _deepep_v2_wave_forward_overlapped(
            state,
            wave_inputs=wave_inputs,
            wave_layout=wave_layout,
            label=label,
            track_expert_grad=track_expert_grad,
            do_cpu_sync=do_cpu_sync,
            workspace=workspace,
        )
    return _deepep_v2_wave_forward_sequential(
        state,
        wave_inputs=wave_inputs,
        wave_layout=wave_layout,
        label=label,
        track_expert_grad=track_expert_grad,
        do_cpu_sync=do_cpu_sync,
        workspace=workspace,
    )


def _validate_deepep_v2_wave_forward(
    state: DeepEpV2State,
    *,
    wave_inputs: list[DeepEpV2WaveInput],
    wave_layout: str,
    overlap: bool,
    wave_do_cpu_sync: bool,
    atol: float,
    rank: int,
) -> None:
    with torch.no_grad():
        reference = _deepep_v2_forward(
            state,
            label="BENCH/deepep_v2_wave/validate/reference_no_wave",
            track_expert_grad=False,
        )
        candidate = _deepep_v2_wave_forward(
            state,
            wave_inputs=wave_inputs,
            wave_layout=wave_layout,
            label="BENCH/deepep_v2_wave/validate/wave",
            track_expert_grad=False,
            overlap=overlap,
            do_cpu_sync=wave_do_cpu_sync,
        )

    torch.cuda.nvtx.range_push("BENCH/deepep_v2_wave/validate/compare")
    try:
        local_max_abs = (reference.combined_x.float() - candidate.combined_x.float()).abs().max()
        global_max_abs = local_max_abs.detach().clone()
        dist.all_reduce(global_max_abs, op=dist.ReduceOp.MAX)
        max_abs = float(global_max_abs.item())
    finally:
        torch.cuda.nvtx.range_pop()

    if rank == 0:
        print(
            "[bench] deepep_v2_wave forward validation: "
            f"layout={wave_layout} overlap={overlap} "
            f"max_abs={max_abs:.6g} atol={atol:.6g}",
            flush=True,
        )
    if max_abs > atol:
        raise RuntimeError(
            "deepep_v2_wave forward validation failed: "
            f"layout={wave_layout} max_abs={max_abs:.6g} > atol={atol:.6g}"
        )


def _clone_routed_expert_grads(module: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if param.grad is None else param.grad.detach().clone()
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def _max_abs_between_grad_maps(
    reference: dict[str, torch.Tensor | None],
    candidate: dict[str, torch.Tensor | None],
) -> tuple[torch.Tensor, list[str]]:
    max_abs = torch.zeros((), device="cuda", dtype=torch.float32)
    mismatched: list[str] = []
    for name in sorted(set(reference) | set(candidate)):
        ref_grad = reference.get(name)
        cand_grad = candidate.get(name)
        if ref_grad is None and cand_grad is None:
            continue
        if (
            ref_grad is None
            or cand_grad is None
            or tuple(ref_grad.shape) != tuple(cand_grad.shape)
        ):
            max_abs = torch.full((), float("inf"), device="cuda", dtype=torch.float32)
            mismatched.append(name)
            continue
        if ref_grad.numel() > 0:
            diff = (ref_grad.float() - cand_grad.float()).abs().max()
            max_abs = torch.maximum(max_abs, diff)
    return max_abs, mismatched


def _validate_deepep_v2_wave_backward(
    state: DeepEpV2State,
    *,
    wave_inputs: list[DeepEpV2WaveInput],
    wave_layout: str,
    overlap: bool,
    wave_do_cpu_sync: bool,
    atol: float,
    rank: int,
) -> None:
    state.routed_experts.zero_grad(set_to_none=True)
    reference = _deepep_v2_forward(
        state,
        label="BENCH/deepep_v2_wave/validate_backward/reference_no_wave_forward",
        track_expert_grad=True,
    )
    reference.grad_combined_x = torch.ones_like(reference.combined_x)
    reference_grad_x = _run_deepep_v2_backward_from_result(
        state,
        reference,
        label="BENCH/deepep_v2_wave/validate_backward/reference_no_wave_backward",
        zero_expert_grads=False,
    )
    reference_param_grads = _clone_routed_expert_grads(state.routed_experts)

    state.routed_experts.zero_grad(set_to_none=True)
    candidate = _deepep_v2_wave_forward(
        state,
        wave_inputs=wave_inputs,
        wave_layout=wave_layout,
        label="BENCH/deepep_v2_wave/validate_backward/wave_forward",
        track_expert_grad=True,
        overlap=overlap,
        do_cpu_sync=wave_do_cpu_sync,
    )
    candidate.grad_combined_x = torch.ones_like(candidate.combined_x)
    candidate_grad_x = _run_deepep_v2_wave_backward_from_result(
        state,
        candidate,
        label="BENCH/deepep_v2_wave/validate_backward/wave_backward",
        zero_expert_grads=False,
        overlap=overlap,
    )
    candidate_param_grads = _clone_routed_expert_grads(state.routed_experts)

    torch.cuda.nvtx.range_push("BENCH/deepep_v2_wave/validate_backward/compare")
    try:
        if tuple(reference_grad_x.shape) != tuple(candidate_grad_x.shape):
            local_input_max_abs = torch.full((), float("inf"), device="cuda", dtype=torch.float32)
        else:
            local_input_max_abs = (reference_grad_x.float() - candidate_grad_x.float()).abs().max()
        input_max_abs = local_input_max_abs.detach().clone()
        dist.all_reduce(input_max_abs, op=dist.ReduceOp.MAX)

        local_param_max_abs, mismatched_grads = _max_abs_between_grad_maps(
            reference_param_grads,
            candidate_param_grads,
        )
        param_max_abs = local_param_max_abs.detach().clone()
        dist.all_reduce(param_max_abs, op=dist.ReduceOp.MAX)
        max_abs = max(float(input_max_abs.item()), float(param_max_abs.item()))
    finally:
        state.routed_experts.zero_grad(set_to_none=True)
        torch.cuda.nvtx.range_pop()

    if rank == 0:
        mismatch_suffix = (
            f" local_mismatched_grads={mismatched_grads}"
            if mismatched_grads
            else ""
        )
        print(
            "[bench] deepep_v2_wave backward validation: "
            f"layout={wave_layout} overlap={overlap} "
            f"input_max_abs={float(input_max_abs.item()):.6g} "
            f"param_max_abs={float(param_max_abs.item()):.6g} "
            f"atol={atol:.6g}{mismatch_suffix}",
            flush=True,
        )
    if max_abs > atol:
        raise RuntimeError(
            "deepep_v2_wave backward validation failed: "
            f"layout={wave_layout} max_abs={max_abs:.6g} > atol={atol:.6g}"
        )


def _run_deepep_v2_static_wave_backward_from_result(
    state: DeepEpV2State,
    result: DeepEpV2ForwardResult,
    *,
    label: str,
) -> torch.Tensor:
    if result.static_wave is None or result.static_recv_x_global is None:
        raise RuntimeError("expand_static backward requires static wave metadata")
    if result.grad_combined_x is None:
        result.grad_combined_x = torch.ones_like(result.combined_x)
    if not hasattr(state.buffer, "dispatch_cached_expanded_into"):
        raise RuntimeError(
            "deepep_v2_wave layout 'expand_static' backward requires the modified "
            "DeepEP working copy with ElasticBuffer.dispatch_cached_expanded_into. "
            "Use --deepep-path /workspace/DeepEP."
        )

    wave = result.static_wave
    grad_weighted_expert_out_global = torch.empty_like(result.static_recv_x_global)
    torch.cuda.nvtx.range_push(f"{label}/combine_backward_dispatch_static")
    try:
        _grad_weighted_expert_out, _grad_topk_idx, _grad_topk_weights, _handle, event = (
            state.buffer.dispatch_cached_expanded_into(
                result.grad_combined_x,
                handle=result.handle,
                recv_x_out=grad_weighted_expert_out_global,
                num_sms=state.num_sms,
                num_qps=state.num_qps,
                async_with_compute_stream=state.async_with_compute_stream,
            )
        )
        _deep_ep_wait(event, async_with_compute_stream=state.async_with_compute_stream)
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/slice_expert_grad_static")
    try:
        grad_weighted_expert_out = grad_weighted_expert_out_global.narrow(
            0,
            wave.wave_base,
            wave.wave_rows,
        )
    finally:
        torch.cuda.nvtx.range_pop()
    grad_expert_out = _deepep_v2_prepare_expert_grad(
        state,
        grad_weighted_expert_out=grad_weighted_expert_out,
        expanded_weights=result.expanded_topk_weights,
        expert_out_is_weighted=result.expert_out_is_weighted,
        label=f"{label}/prepare_expert_grad_static",
    )

    torch.cuda.nvtx.range_push(f"{label}/experts_backward_static")
    try:
        torch.autograd.backward(result.expert_out, grad_expert_out)
    finally:
        torch.cuda.nvtx.range_pop()

    if result.recv_x.grad is None:
        raise RuntimeError("deepep_v2 static expert backward did not produce grad for recv_x")

    grad_recv_x_global = torch.empty_like(result.static_recv_x_global)
    torch.cuda.nvtx.range_push(f"{label}/stage_recv_grad_static")
    try:
        grad_recv_x_global.narrow(0, wave.wave_base, wave.wave_rows).copy_(result.recv_x.grad)
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/dispatch_backward_combine_static")
    try:
        combined_grad_x, _combined_grad_topk_weights, event = state.buffer.combine(
            grad_recv_x_global,
            handle=result.handle,
            num_sms=state.num_sms,
            num_qps=state.num_qps,
            async_with_compute_stream=state.async_with_compute_stream,
        )
        _deep_ep_wait(event, async_with_compute_stream=state.async_with_compute_stream)
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/zero_grad_static")
    try:
        result.recv_x.grad = None
    finally:
        torch.cuda.nvtx.range_pop()
    return combined_grad_x


def _run_deepep_v2_static_wave_backward_overlapped(
    state: DeepEpV2State,
    result: DeepEpV2WaveForwardResult,
    *,
    label: str,
    zero_expert_grads: bool = True,
) -> torch.Tensor:
    if (
        state.wave_dispatch_stream is None
        or state.wave_compute_stream is None
        or state.wave_combine_stream is None
    ):
        raise RuntimeError("DeepEP V2 wave streams were not initialized")
    if not result.wave_results:
        raise RuntimeError("DeepEP V2 wave backward produced no wave results")
    if result.grad_combined_x is None:
        result.grad_combined_x = torch.ones_like(result.combined_x)
    if not hasattr(state.buffer, "dispatch_cached_expanded_into"):
        raise RuntimeError(
            "deepep_v2_wave layout 'expand_static' backward requires the modified "
            "DeepEP working copy with ElasticBuffer.dispatch_cached_expanded_into. "
            "Use --deepep-path /workspace/DeepEP."
        )

    recv_x_global = result.wave_results[0].static_recv_x_global
    if recv_x_global is None:
        raise RuntimeError("overlapped static wave backward requires static receive buffers")
    grad_weighted_expert_out_global = torch.empty_like(recv_x_global)
    grad_recv_x_global = torch.empty_like(recv_x_global)
    async_mode = True

    backward_dispatches: list[tuple[int, DeepEpV2ForwardResult, object]] = []
    for wave_idx, wave_result in enumerate(result.wave_results):
        if wave_result.static_wave is None or wave_result.static_recv_x_global is None:
            raise RuntimeError("overlapped wave backward currently requires expand_static results")
        wave_result.grad_combined_x = _deepep_v2_wave_grad_combined_x(result, wave_result)
        with torch.cuda.stream(state.wave_dispatch_stream):
            torch.cuda.nvtx.range_push(f"{label}/wave_{wave_idx}/combine_backward_dispatch_static")
            try:
                _grad_weighted_expert_out, _grad_topk_idx, _grad_topk_weights, _handle, event = (
                    state.buffer.dispatch_cached_expanded_into(
                        wave_result.grad_combined_x,
                        handle=wave_result.handle,
                        recv_x_out=grad_weighted_expert_out_global,
                        num_sms=state.num_sms,
                        num_qps=state.num_qps,
                        async_with_compute_stream=async_mode,
                    )
                )
            finally:
                torch.cuda.nvtx.range_pop()
        backward_dispatches.append((wave_idx, wave_result, event))

    compute_results: list[tuple[int, DeepEpV2ForwardResult, torch.cuda.Event]] = []
    staged_recv_grads: list[torch.Tensor] = []
    for wave_idx, wave_result, dispatch_event in backward_dispatches:
        wave = wave_result.static_wave
        assert wave is not None
        with torch.cuda.stream(state.wave_compute_stream):
            _deep_ep_wait(dispatch_event, async_with_compute_stream=async_mode)

            torch.cuda.nvtx.range_push(f"{label}/wave_{wave_idx}/slice_expert_grad_static")
            try:
                grad_weighted_expert_out = grad_weighted_expert_out_global.narrow(
                    0,
                    wave.wave_base,
                    wave.wave_rows,
                )
            finally:
                torch.cuda.nvtx.range_pop()
            grad_expert_out = _deepep_v2_prepare_expert_grad(
                state,
                grad_weighted_expert_out=grad_weighted_expert_out,
                expanded_weights=wave_result.expanded_topk_weights,
                expert_out_is_weighted=wave_result.expert_out_is_weighted,
                label=f"{label}/wave_{wave_idx}/prepare_expert_grad_static",
            )

            torch.cuda.nvtx.range_push(f"{label}/wave_{wave_idx}/experts_backward_static")
            try:
                torch.autograd.backward(wave_result.expert_out, grad_expert_out)
            finally:
                torch.cuda.nvtx.range_pop()

            if wave_result.recv_x.grad is None:
                raise RuntimeError(
                    "deepep_v2 static expert backward did not produce grad for recv_x"
                )

            torch.cuda.nvtx.range_push(f"{label}/wave_{wave_idx}/stage_recv_grad_static")
            try:
                recv_grad = wave_result.recv_x.grad
                staged_recv_grads.append(recv_grad)
                grad_recv_x_global.narrow(0, wave.wave_base, wave.wave_rows).copy_(recv_grad)
            finally:
                torch.cuda.nvtx.range_pop()

            compute_done = torch.cuda.Event(enable_timing=False)
            compute_done.record()
            wave_result.recv_x.grad = None
        compute_results.append((wave_idx, wave_result, compute_done))

    combined_grad_parts: list[torch.Tensor] = []
    combine_events: list[object] = []
    for wave_idx, wave_result, compute_done in compute_results:
        with torch.cuda.stream(state.wave_combine_stream):
            state.wave_combine_stream.wait_event(compute_done)
            torch.cuda.nvtx.range_push(f"{label}/wave_{wave_idx}/dispatch_backward_combine_static")
            try:
                combined_grad_x, _combined_grad_topk_weights, event = state.buffer.combine(
                    grad_recv_x_global,
                    handle=wave_result.handle,
                    num_sms=state.num_sms,
                    num_qps=state.num_qps,
                    async_with_compute_stream=async_mode,
                )
            finally:
                torch.cuda.nvtx.range_pop()
        combined_grad_parts.append(combined_grad_x)
        combine_events.append(event)

    for event in combine_events:
        _deep_ep_wait(event, async_with_compute_stream=async_mode)

    torch.cuda.nvtx.range_push(f"{label}/zero_grad")
    try:
        if zero_expert_grads:
            state.routed_experts.zero_grad(set_to_none=True)
    finally:
        torch.cuda.nvtx.range_pop()

    # Keep staged recv grad tensors alive until the final stream waits above.
    del staged_recv_grads
    wave_inputs = [
        wave_result.wave_input
        for wave_result in result.wave_results
        if wave_result.wave_input is not None
    ]
    if len(wave_inputs) == len(result.wave_results):
        return _merge_deepep_v2_wave_outputs(
            combined_grad_parts,
            wave_inputs=wave_inputs,
            label=f"{label}/merge_input_grads",
        )
    return _sum_deepep_v2_wave_outputs(combined_grad_parts, label=f"{label}/sum_input_grads")


def _prepare_deepep_v2_wave_backward(
    state: DeepEpV2State,
    *,
    wave_inputs: list[DeepEpV2WaveInput],
    wave_layout: str,
    label: str,
    overlap: bool,
    do_cpu_sync: bool,
    workspace: DeepEpV2StaticExpandedWorkspace | None = None,
) -> DeepEpV2WaveForwardResult:
    result = _deepep_v2_wave_forward(
        state,
        wave_inputs=wave_inputs,
        wave_layout=wave_layout,
        label=f"{label}/forward_prep",
        track_expert_grad=True,
        overlap=overlap,
        do_cpu_sync=do_cpu_sync,
        workspace=workspace,
    )
    torch.cuda.nvtx.range_push(f"{label}/grad_prep")
    try:
        result.grad_combined_x = torch.ones_like(result.combined_x)
    finally:
        torch.cuda.nvtx.range_pop()
    return result


def _run_deepep_v2_wave_backward_from_result(
    state: DeepEpV2State,
    result: DeepEpV2WaveForwardResult,
    *,
    label: str,
    zero_expert_grads: bool = True,
    overlap: bool = False,
) -> torch.Tensor:
    if result.grad_combined_x is None:
        result.grad_combined_x = torch.ones_like(result.combined_x)

    if overlap and all(wave_result.static_wave is not None for wave_result in result.wave_results):
        return _run_deepep_v2_static_wave_backward_overlapped(
            state,
            result,
            label=label,
            zero_expert_grads=zero_expert_grads,
        )

    # First version is waved but intentionally sequential. It preserves the
    # same reverse communication semantics as no-wave DeepEP: combine backward
    # is a dispatch with the forward handle, and dispatch backward is a combine
    # with the forward handle.
    combined_grad_parts: list[torch.Tensor] = []
    for wave_idx, wave_result in enumerate(result.wave_results):
        wave_result.grad_combined_x = _deepep_v2_wave_grad_combined_x(result, wave_result)
        if wave_result.static_wave is not None:
            combined_grad_parts.append(
                _run_deepep_v2_static_wave_backward_from_result(
                    state,
                    wave_result,
                    label=f"{label}/wave_{wave_idx}",
                )
            )
        else:
            combined_grad_parts.append(
                _run_deepep_v2_backward_from_result(
                    state,
                    wave_result,
                    label=f"{label}/wave_{wave_idx}",
                    zero_expert_grads=False,
                )
            )

    torch.cuda.nvtx.range_push(f"{label}/zero_grad")
    try:
        if zero_expert_grads:
            state.routed_experts.zero_grad(set_to_none=True)
    finally:
        torch.cuda.nvtx.range_pop()
    wave_inputs = [
        wave_result.wave_input
        for wave_result in result.wave_results
        if wave_result.wave_input is not None
    ]
    if len(wave_inputs) == len(result.wave_results):
        return _merge_deepep_v2_wave_outputs(
            combined_grad_parts,
            wave_inputs=wave_inputs,
            label=f"{label}/merge_input_grads",
        )
    return _sum_deepep_v2_wave_outputs(combined_grad_parts, label=f"{label}/sum_input_grads")


def _run_one_deepep_v2_wave_iter(
    state: DeepEpV2State,
    *,
    wave_inputs: list[DeepEpV2WaveInput],
    wave_layout: str,
    label: str,
    pass_type: str,
    overlap: bool,
    do_cpu_sync: bool,
    workspace: DeepEpV2StaticExpandedWorkspace | None = None,
) -> None:
    if pass_type == "forward":
        with torch.no_grad():
            _deepep_v2_wave_forward(
                state,
                wave_inputs=wave_inputs,
                wave_layout=wave_layout,
                label=label,
                track_expert_grad=False,
                overlap=overlap,
                do_cpu_sync=do_cpu_sync,
                workspace=workspace,
            )
        return

    result = _deepep_v2_wave_forward(
        state,
        wave_inputs=wave_inputs,
        wave_layout=wave_layout,
        label=f"{label}/forward",
        track_expert_grad=True,
        overlap=overlap,
        do_cpu_sync=do_cpu_sync,
        workspace=workspace,
    )
    _run_deepep_v2_wave_backward_from_result(
        state,
        result,
        label=f"{label}/backward",
        overlap=overlap,
    )
