from __future__ import annotations

import argparse

import torch
import torch.distributed as dist


def _init_probe_routed_expert_weights(
    routed_experts: torch.nn.Module,
    *,
    weight_init: str,
) -> None:
    if weight_init == "empty":
        return
    if not hasattr(routed_experts, "w_up_gate") or not hasattr(routed_experts, "w_down"):
        raise RuntimeError("probe weight init requires RoutedExperts-like module")
    with torch.no_grad():
        if weight_init == "normal":
            routed_experts.w_up_gate.normal_(mean=0.0, std=0.02)
            routed_experts.w_down.normal_(mean=0.0, std=0.02)
        elif weight_init == "normal1":
            routed_experts.w_up_gate.normal_(mean=0.0, std=1.0)
            routed_experts.w_down.normal_(mean=0.0, std=1.0)
        elif weight_init == "uniform":
            routed_experts.w_up_gate.uniform_(-0.02, 0.02)
            routed_experts.w_down.uniform_(-0.02, 0.02)
        elif weight_init == "rand_sign":
            routed_experts.w_up_gate.bernoulli_(0.5).mul_(2.0).sub_(1.0).mul_(0.02)
            routed_experts.w_down.bernoulli_(0.5).mul_(2.0).sub_(1.0).mul_(0.02)
        elif weight_init == "zero":
            routed_experts.w_up_gate.zero_()
            routed_experts.w_down.zero_()
        elif weight_init == "fill":
            routed_experts.w_up_gate.fill_(0.02)
            routed_experts.w_down.fill_(0.02)
        else:
            raise ValueError(weight_init)


def _resolve_weight_init_value(weight_init: str, *, source_default: str) -> str:
    return source_default if weight_init == "source_default" else weight_init


def _resolve_probe_weight_init(args: argparse.Namespace, *, source_default: str) -> str:
    return _resolve_weight_init_value(
        str(args.deepep_probe_weight_init),
        source_default=source_default,
    )


def _run_pre_dispatch_expert_probe(
    routed_experts: torch.nn.Module,
    *,
    mode_name: str,
    num_iters: int,
    tokens: int,
    top_k: int,
    d_model: int,
    input_dtype: torch.dtype,
    pass_type: str,
    rank: int,
    world_size: int,
) -> None:
    if num_iters <= 0:
        return

    if not hasattr(routed_experts, "w_up_gate"):
        raise RuntimeError(f"{mode_name} pre-dispatch probe requires RoutedExperts-like module")
    num_local_experts = int(routed_experts.w_up_gate.shape[0])
    valid_rows = tokens * top_k
    base = valid_rows // num_local_experts
    remainder = valid_rows % num_local_experts
    counts_cpu = torch.full(
        (num_local_experts,),
        base,
        dtype=torch.int32,
    )
    if remainder:
        counts_cpu[:remainder] += 1
    batch_size_per_expert = counts_cpu.to(device="cuda")

    # Match the rowwise BF16 call surface: expert-major input, explicit down
    # output buffer, and input dgrad buffer for the first grouped-mm.
    fake_x = (0.2 * torch.randn(valid_rows, d_model, device="cuda")).to(input_dtype)
    track_grad = pass_type != "forward"
    if track_grad:
        fake_x = fake_x.detach().requires_grad_(True)
    down_proj_out = torch.empty_like(fake_x)
    up_proj_input_grad_out = fake_x.detach()

    times: list[float] = []
    for idx in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        label = f"BENCH/{mode_name}/pre_dispatch_experts/iter_{idx}"
        start.record()
        torch.cuda.nvtx.range_push(label)
        try:
            out = routed_experts(
                fake_x,
                batch_size_per_expert,
                down_proj_out=down_proj_out,
                up_proj_input_grad_out=up_proj_input_grad_out,
            )
        finally:
            torch.cuda.nvtx.range_pop()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        del out
        if fake_x.grad is not None:
            fake_x.grad = None

    local = torch.tensor(times, device="cuda", dtype=torch.float32)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    if rank == 0:
        max_by_iter = torch.stack(gathered, dim=0).amax(dim=0).detach().cpu().tolist()
        print(
            f"[bench] {mode_name} pre_dispatch_experts "
            f"iters={num_iters} "
            f"valid_rows={valid_rows} "
            f"counts={counts_cpu.tolist()} "
            f"max_rank_ms={[round(float(v), 3) for v in max_by_iter]}",
            flush=True,
        )

    del fake_x, down_proj_out, up_proj_input_grad_out, batch_size_per_expert
    torch.cuda.synchronize()


def _deepep_pre_dispatch_expert_iters(args: argparse.Namespace) -> int:
    return max(int(args.pre_dispatch_expert_iters), int(args.deepep_pre_dispatch_expert_iters))
