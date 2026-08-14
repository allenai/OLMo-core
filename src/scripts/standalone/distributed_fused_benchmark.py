"""Distributed fused OLMoE3 forward/backward benchmark.

Place this file beside ``fused_model.py``. On a Linux CUDA system, follow the
installation instructions at the top of that file, then launch, for example::

    torchrun --standalone --nproc-per-node=8 distributed_fused_benchmark.py \
        --ep-degree -1 --sequence-length 8192 --microbatch-sequences 1

For a single-GPU installation test, use the 30M smoke rung::

    torchrun --standalone --nproc-per-node=1 distributed_fused_benchmark.py \
        --model-size 30m --ep-degree 1 --sequence-length 512 \
        --microbatch-sequences 1 --warmup 1 --iterations 2 --no-compile

``--ep-degree -1`` uses every launched rank for expert parallelism. The harness
uses OLMo-core's production rowwise-NVSHMEM EP path, BF16 model/forward/backward,
FP32 gradient reduction and accumulation, compilation, EMo, and global load
balancing by default. It reports warmup-free forward/backward throughput; pass
``--include-optimizer-step`` to include the explicitly configured AdamW update.
Routed experts are EP-sharded; dense parameters use the ladder's replicated DDP
policy (there is intentionally no FSDP in this benchmark).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist

from fused_model import FusedModelOptions, build_fused_config
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.optim import OLMoDDPOptimizerConfig, OptimGroupOverride
from olmo_core.train.train_module import (
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)


def enable_single_gpu_runtime_fallbacks() -> bool:
    """Install dependency-light, alignment-safe fallbacks for EP degree one.

    OLMo-core currently imports its no-EP ``moe_permute``/``moe_unpermute``
    functions from Transformer Engine, but TE is not an OLMo-core dependency.
    This fallback keeps the single-GPU installation smoke test usable on CUDA
    runtime images without NVCC. It is deliberately not used for EP benchmarks.
    """
    from olmo_core.nn.moe import utils as moe_utils

    if moe_utils.moe_permute is not None and moe_utils.moe_unpermute is not None:
        return False

    def permute(
        *,
        inp: torch.Tensor,
        routing_map: torch.Tensor,
        num_out_tokens: int,
        map_type: str,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        if map_type != "index":
            raise ValueError(f"single-GPU fallback requires map_type='index', got {map_type!r}")
        flat_experts = routing_map.reshape(-1)
        assignment_order = torch.argsort(flat_experts, stable=True)
        if assignment_order.numel() != num_out_tokens:
            raise ValueError(
                f"expected {num_out_tokens} routed rows, got {assignment_order.numel()}"
            )
        top_k = routing_map.shape[-1]
        token_indices = torch.div(assignment_order, top_k, rounding_mode="floor")
        return inp.index_select(0, token_indices), assignment_order

    def unpermute(
        *,
        inp: torch.Tensor,
        row_id_map: torch.Tensor,
        restore_shape: torch.Size | tuple[int, ...],
        map_type: str,
        merging_probs: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if map_type != "index":
            raise ValueError(f"single-GPU fallback requires map_type='index', got {map_type!r}")
        if merging_probs is None:
            raise ValueError("single-GPU fallback requires merging_probs")
        assignment_rows = torch.empty_like(inp).index_copy(0, row_id_map.long(), inp)
        token_count, top_k = merging_probs.shape
        merged = assignment_rows.view(token_count, top_k, -1)
        merged = (merged * merging_probs.unsqueeze(-1).to(merged.dtype)).sum(1)
        return merged.view(restore_shape)

    moe_utils.moe_permute = permute
    moe_utils.moe_unpermute = unpermute

    # OLMo-core's meta-device materialization may place an expert parameter at
    # a non-16-byte storage offset. PyTorch grouped_mm rejects such operands,
    # even in eager mode. Clone only offending views at the call boundary;
    # clone remains differentiable, so gradients still reach the parameter.
    from olmo_core.nn.moe.v2 import routed_experts

    original_gmm = routed_experts.gmm

    def aligned_gmm(
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        trans_b: bool = False,
        out: torch.Tensor | None = None,
        input_grad_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if a.data_ptr() % 16:
            a = a.clone()
        if b.data_ptr() % 16:
            b = b.clone()
        return original_gmm(
            a,
            b,
            batch_sizes,
            trans_b=trans_b,
            out=out,
            input_grad_out=input_grad_out,
        )

    routed_experts.gmm = aligned_gmm
    return True


@dataclass(frozen=True)
class BenchmarkResult:
    model_size: str
    world_size: int
    ep_degree: int
    sequence_length: int
    microbatch_sequences_per_rank: int
    global_tokens_per_iteration: int
    warmup_iterations: int
    measured_iterations: int
    compile_model: bool
    include_optimizer_step: bool
    mean_iteration_seconds: float
    tokens_per_second: float
    estimated_tflops_per_gpu: float
    peak_allocated_gib_per_gpu: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=("30m", "3p5b"), default="3p5b")
    parser.add_argument(
        "--ep-degree",
        type=int,
        default=-1,
        help="Expert-parallel degree; -1 uses all launched ranks (default: -1)",
    )
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--microbatch-sequences", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--emo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--global-load-balancing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-optimizer-step",
        action="store_true",
        help="Time AdamW as part of each iteration (allocates optimizer state)",
    )
    return parser.parse_args()


def initialize_distributed() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(f"Launch with torchrun; missing environment variables: {missing}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return torch.device("cuda", local_rank)


def build_train_module(args: argparse.Namespace, device: torch.device):
    world_size = dist.get_world_size()
    resolved_ep_degree = world_size if args.ep_degree < 0 else args.ep_degree
    if resolved_ep_degree < 1:
        raise ValueError("--ep-degree must be positive or -1")
    if world_size % resolved_ep_degree:
        raise ValueError(
            f"world size {world_size} must be divisible by EP degree {resolved_ep_degree}"
        )

    if resolved_ep_degree == 1:
        fallback_enabled = enable_single_gpu_runtime_fallbacks()
        if fallback_enabled and dist.get_rank() == 0:
            print(
                "Transformer Engine unavailable: using the PyTorch permutation fallback "
                "for the single-GPU smoke test.",
                flush=True,
            )
        if fallback_enabled and args.compile:
            # Inductor can lower the fallback's index_select output to a
            # temporary allocation that does not satisfy grouped_mm's 16-byte
            # data-pointer contract. Keep the production fused operators, but
            # do not wrap the surrounding model in torch.compile in this
            # dependency-light smoke-test path.
            args.compile = False
            if dist.get_rank() == 0:
                print(
                    "Disabling outer torch.compile: its grouped_mm lowering is not "
                    "alignment-safe with the PyTorch permutation fallback. FLA, "
                    "FlashAttention, and grouped-MM kernels remain enabled.",
                    flush=True,
                )
    else:
        from olmo_core.nn.moe import utils as moe_utils

        if moe_utils.moe_permute is None or moe_utils.moe_unpermute is None:
            raise RuntimeError(
                "Multi-GPU EP requires Transformer Engine's compiled permutation operators. "
                "Install a compatible prebuilt transformer-engine[pytorch] wheel or use a "
                "CUDA development image with nvcc."
            )

    model_config = build_fused_config(
        FusedModelOptions(
            model_size=args.model_size,
            emo_enabled=args.emo,
            global_load_balancing=args.global_load_balancing,
        )
    )
    # The train module applies EP to this meta model before materialization, so
    # routed expert parameters are allocated sharded rather than replicated.
    model = model_config.build(init_device="meta")
    optim = OLMoDDPOptimizerConfig(
        lr=8e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0}),
            OptimGroupOverride(
                params=["*routed_experts.w_up_gate", "*routed_experts.w_down"],
                opts={"lr": 8e-4},
            ),
        ],
        compile=args.compile and args.include_optimizer_step,
        dtype=DType.float32,
        sigma_factor=6,
        max_grad_norm=1.0,
        use_distributed=True,
    )
    train_config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=args.microbatch_sequences * args.sequence_length,
        max_sequence_length=args.sequence_length,
        optim=optim,
        compile_model=args.compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            only_allreduce_last_microbatch=True,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
        ),
        ep_config=(
            TransformerExpertParallelConfig(degree=args.ep_degree)
            if resolved_ep_degree > 1
            else None
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )
    module = train_config.build(model, device=device)
    return model_config, module, resolved_ep_degree


def make_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(args.seed + dist.get_rank())
    shape = (args.microbatch_sequences, args.sequence_length)
    input_ids = torch.randint(0, 100_352, shape, device=device, generator=generator)
    # Avoid random EOS tokens splitting documents, so each sequence is one EMo document.
    input_ids.masked_fill_(input_ids == 100_257, 0)
    labels = input_ids.roll(-1, dims=1)
    labels[:, -1] = -100
    return input_ids, labels


def benchmark_iteration(
    module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_div_factor: torch.Tensor,
    include_optimizer_step: bool,
) -> None:
    module.zero_grads()
    output = module.model_forward_no_pipeline(
        input_ids,
        labels=labels,
        ignore_index=-100,
        loss_reduction="sum",
        z_loss_multiplier=1e-5,
        loss_div_factor=loss_div_factor,
        return_logits=False,
    )
    assert isinstance(output, LMOutputWithLoss)
    output.loss.backward()
    for model in module.model_parts:
        model.finalize_grad_reduce()
        model.post_batch()
    if include_optimizer_step:
        optimizer = module._require_optimizer()
        optimizer.step()
        for model in module.model_parts:
            model.post_optim_step()


def reduced_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--warmup must be non-negative and --iterations must be positive")
    device = initialize_distributed()
    try:
        config, module, ep_degree = build_train_module(args, device)
        input_ids, labels = make_batch(args, device)
        loss_div_factor = (labels != -100).sum()
        dist.all_reduce(loss_div_factor, group=module.dp_process_group)
        loss_div_factor = loss_div_factor.clamp_min(1) / dist.get_world_size(
            module.dp_process_group
        )

        if dist.get_rank() == 0:
            message = f"warming up {args.warmup} iterations"
            if args.compile:
                message += "; first compiled iteration may take minutes"
            print(message, flush=True)
        for _ in range(args.warmup):
            benchmark_iteration(
                module,
                input_ids,
                labels,
                loss_div_factor=loss_div_factor,
                include_optimizer_step=args.include_optimizer_step,
            )
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        durations: list[float] = []
        for _ in range(args.iterations):
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            benchmark_iteration(
                module,
                input_ids,
                labels,
                loss_div_factor=loss_div_factor,
                include_optimizer_step=args.include_optimizer_step,
            )
            torch.cuda.synchronize()
            durations.append(reduced_max(time.perf_counter() - start, device))

        mean_seconds = sum(durations) / len(durations)
        global_tokens = dist.get_world_size() * args.microbatch_sequences * args.sequence_length
        flops_per_token = module.num_flops_per_token(args.sequence_length) or 0
        result = BenchmarkResult(
            model_size=args.model_size,
            world_size=dist.get_world_size(),
            ep_degree=ep_degree,
            sequence_length=args.sequence_length,
            microbatch_sequences_per_rank=args.microbatch_sequences,
            global_tokens_per_iteration=global_tokens,
            warmup_iterations=args.warmup,
            measured_iterations=args.iterations,
            compile_model=args.compile,
            include_optimizer_step=args.include_optimizer_step,
            mean_iteration_seconds=mean_seconds,
            tokens_per_second=global_tokens / mean_seconds,
            estimated_tflops_per_gpu=(
                flops_per_token
                * args.microbatch_sequences
                * args.sequence_length
                / mean_seconds
                / 1e12
            ),
            peak_allocated_gib_per_gpu=reduced_max(
                torch.cuda.max_memory_allocated(device) / 2**30, device
            ),
        )
        if dist.get_rank() == 0:
            print(json.dumps(asdict(result), indent=2))
            print(
                f"model params: total={config.num_params:,}, active={config.num_active_params:,}"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
