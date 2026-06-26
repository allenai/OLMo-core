"""
Benchmark OLMo rowwise EP against the OLMo-owned fused BF16 MegaMoE target.

Example:
    PYTHONPATH=src torchrun --standalone --nproc-per-node=4 \
        src/scripts/train/wave_ep_moe_bench.py

Nsight Systems example:
    PYTHONPATH=src nsys profile -o wave_ep_moe_ep4 \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        torchrun --standalone --nproc-per-node=4 \
        src/scripts/train/wave_ep_moe_bench.py --modes rowwise --profile

Modes:
    rowwise: baseline rowwise NVSHMEM EP.
    rowwise_wave: expert-major rowwise-wave EP.
    wave: model-facing forward-only OLMo wave bring-up path.
    standard_ep_mega: standalone standard-shape EP4/32-expert fused BF16
        MegaMoE megakernel path. This is a kernel benchmark, not model wiring.
    standard_ep_mega_peer_group: standalone EP4 rank-local BF16 MegaMoE path
        with real symmetric peer workspaces and normal per-rank kernel launch.
    standard_ep_mega_collective: standalone EP4/world4 rank-local BF16
        MegaMoE NVSHMEM world collective-launch diagnostic path.
    deepep_v2: standalone no-wave DeepEP V2 dispatch + OLMo grouped expert
        MLP + DeepEP V2 combine path. This is a benchmark bring-up path, not
        model wiring yet.
    deepep_v2_wave: standalone DeepEP V2 wave dispatch + OLMo grouped expert
        MLP + DeepEP V2 combine path with optional stream pipelining.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
import types
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig


@dataclass(frozen=True)
class BenchCase:
    name: str
    use_wave: bool
    use_bf16_persistent_mega: bool = False
    kernel_standard_ep_mega: bool = False
    kernel_standard_ep_mega_peer_group: bool = False
    kernel_standard_ep_mega_collective: bool = False
    kernel_standard_ep_mega_umma: bool = False
    rowwise_wave: bool = False
    deepep_v2: bool = False
    deepep_v2_wave: bool = False


def _parse_modes(raw: str) -> list[BenchCase]:
    cases: list[BenchCase] = []
    for part in raw.split(","):
        mode = part.strip().lower()
        if not mode:
            continue
        if mode == "rowwise":
            cases.append(BenchCase("rowwise", False))
        elif mode in {"deepep_v2", "deep_ep_v2", "deepep"}:
            cases.append(BenchCase("deepep_v2", False, deepep_v2=True))
        elif mode in {"deepep_v2_wave", "deep_ep_v2_wave", "deepep_wave"}:
            cases.append(BenchCase("deepep_v2_wave", False, deepep_v2_wave=True))
        elif mode in {
            "rowwise_wave",
            "rowwise_wave_expert",
            "rowwise_wave_expert_sequential",
        }:
            cases.append(BenchCase("rowwise_wave", False, rowwise_wave=True))
        elif mode in {
            "wave",
            "olmo_wave",
            "bf16_persistent_mega",
            "persistent_mega",
            "wave_bf16_persistent",
            "wave_bf16_persistent_mega",
        }:
            cases.append(BenchCase("bf16_persistent_mega", True, True))
        elif mode in {
            "standard_ep_mega",
            "standard_ep_full_mega",
            "kernel_standard_ep_mega",
            "kernel_mega",
        }:
            cases.append(BenchCase("standard_ep_mega", False, False, True))
        elif mode in {
            "standard_ep_mega_umma",
            "standard_ep_full_mega_umma",
            "kernel_standard_ep_mega_umma",
            "kernel_mega_umma",
        }:
            cases.append(BenchCase("standard_ep_mega_umma", False, False, True, False, False, True))
        elif mode in {
            "standard_ep_mega_peer_group",
            "standard_ep_peer_group",
            "kernel_standard_ep_peer_group",
            "kernel_mega_peer_group",
        }:
            cases.append(
                BenchCase(
                    "standard_ep_mega_peer_group",
                    False,
                    False,
                    False,
                    True,
                    False,
                )
            )
        elif mode in {
            "standard_ep_mega_peer_group_umma",
            "standard_ep_peer_group_umma",
            "kernel_standard_ep_peer_group_umma",
            "kernel_mega_peer_group_umma",
        }:
            cases.append(
                BenchCase(
                    "standard_ep_mega_peer_group_umma",
                    False,
                    False,
                    False,
                    True,
                    False,
                    True,
                )
            )
        elif mode in {
            "standard_ep_mega_collective",
            "standard_ep_collective",
            "kernel_standard_ep_collective",
            "kernel_mega_collective",
        }:
            cases.append(
                BenchCase(
                    "standard_ep_mega_collective",
                    False,
                    False,
                    False,
                    False,
                    True,
                )
            )
        elif mode in {
            "standard_ep_mega_collective_umma",
            "standard_ep_collective_umma",
            "kernel_standard_ep_collective_umma",
            "kernel_mega_collective_umma",
        }:
            cases.append(
                BenchCase(
                    "standard_ep_mega_collective_umma",
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                )
            )
        else:
            raise ValueError(
                f"Unknown mode {mode!r}. Expected rowwise,rowwise_wave,deepep_v2,deepep_v2_wave,wave,bf16_persistent_mega,"
                "standard_ep_mega,standard_ep_mega_umma,"
                "standard_ep_mega_peer_group,standard_ep_mega_peer_group_umma,"
                "standard_ep_mega_collective,standard_ep_mega_collective_umma"
            )
    if not cases:
        raise ValueError("At least one benchmark mode is required")
    return cases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MoE-only OLMo EP benchmark. By default it patches dense attention "
            "to identity so large token counts remain an EP/MoE measurement."
        )
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[16384])
    parser.add_argument(
        "--modes",
        type=str,
        default="rowwise",
        help=(
            "Comma-separated modes. 'rowwise' is the current baseline. "
            "'rowwise_wave' selects expert-major rowwise waves. "
            "'deepep_v2' runs a standalone DeepEP V2 no-wave "
            "dispatch/GEMM/combine benchmark. "
            "'deepep_v2_wave' runs the standalone DeepEP V2 wave "
            "dispatch/GEMM/combine benchmark. "
            "'wave'/'bf16_persistent_mega' select the model-facing forward-only "
            "wave path. 'standard_ep_mega' runs the standalone standard-shape "
            "fused BF16 megakernel. Suffix standard_ep_mega modes with '_umma' "
            "to use the 128x128x64 BF16 TMA/UMMA compute branch. "
            "'standard_ep_mega_peer_group' runs the rank-local peer-workspace "
            "kernel and requires 4 ranks. 'standard_ep_mega_collective' runs "
            "the rank-local NVSHMEM collective-launch diagnostic path and "
            "requires 4 ranks."
        ),
    )
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--rowwise-nblocks", type=int, default=32)
    parser.add_argument(
        "--deepep-path",
        type=str,
        default=os.getenv("OLMO_DEEPEP_PATH", "/workspace/DeepEP"),
        help=(
            "Optional path to a DeepEP checkout. Added to sys.path before "
            "importing deep_ep for --modes deepep_v2."
        ),
    )
    parser.add_argument(
        "--deepep-num-sms",
        type=int,
        default=0,
        help="DeepEP dispatch/combine SM count. 0 lets DeepEP choose.",
    )
    parser.add_argument(
        "--deepep-num-qps",
        type=int,
        default=0,
        help="DeepEP dispatch/combine QP count. 0 lets DeepEP choose.",
    )
    parser.add_argument(
        "--deepep-num-allocated-qps",
        type=int,
        default=0,
        help=(
            "DeepEP ElasticBuffer allocated QP count. 0 lets DeepEP choose; "
            "otherwise this should be >= --deepep-num-qps."
        ),
    )
    parser.add_argument(
        "--deepep-expert-alignment",
        type=int,
        default=1,
        help="DeepEP per-local-expert receive alignment for expanded dispatch.",
    )
    parser.add_argument(
        "--deepep-max-tokens-factor",
        type=float,
        default=1.0,
        help=(
            "Multiplier for DeepEP ElasticBuffer num_max_tokens_per_rank. "
            "This is the DeepEP-side capacity overprovisioning probe; "
            "1.0 means exactly --tokens."
        ),
    )
    parser.add_argument(
        "--deepep-expert-buffer-mode",
        choices=("none", "down", "all"),
        default="none",
        help=(
            "Ablation for DeepEP standalone expert MLP buffers. 'none' lets "
            "RoutedExperts allocate grouped-mm outputs internally. 'down' "
            "passes an explicit down_proj_out buffer for Linear2. 'all' also "
            "passes a rowwise-style input_grad_out buffer for Linear1 dgrad."
        ),
    )
    parser.add_argument(
        "--deepep-pre-dispatch-expert-iters",
        type=int,
        default=0,
        help=(
            "Diagnostic: run this many fake rowwise-style RoutedExperts.forward "
            "calls before the first DeepEP dispatch. This tests whether expert "
            "GEMMs are only slow after DeepEP communication has run."
        ),
    )
    parser.add_argument(
        "--pre-dispatch-expert-iters",
        type=int,
        default=0,
        help=(
            "Diagnostic: run this many fake rowwise-style RoutedExperts.forward "
            "calls before the first dispatch/communication in modes that support "
            "the probe."
        ),
    )
    parser.add_argument(
        "--deepep-skip-import-buffer-for-pre-dispatch-probe",
        action="store_true",
        help=(
            "Diagnostic for --modes deepep_v2 with fake pre-dispatch expert "
            "probes: skip importing deep_ep and skip ElasticBuffer construction, "
            "then return after the probe. This isolates whether DeepEP import/"
            "buffer setup changes subsequent expert GEMM timing."
        ),
    )
    parser.add_argument(
        "--deepep-probe-routed-experts-source",
        choices=("standalone", "rowwise_apply_ep"),
        default="standalone",
        help=(
            "Diagnostic source for fake deepep_v2 pre-dispatch expert probes. "
            "'standalone' builds the local-only RoutedExperts used by the "
            "standalone DeepEP path. 'rowwise_apply_ep' builds a normal rowwise "
            "MoE block, applies EP sharding, and probes block.routed_experts."
        ),
    )
    parser.add_argument(
        "--deepep-probe-weight-init",
        choices=("source_default", "normal", "normal1", "uniform", "rand_sign", "empty", "zero", "fill"),
        default="source_default",
        help=(
            "Diagnostic local expert weight initialization for fake deepep_v2 "
            "pre-dispatch probes. source_default preserves current behavior: "
            "standalone initializes local weights with normal_(0, 0.02), while "
            "rowwise_apply_ep leaves the apply_ep-created local weights as "
            "torch.empty(). 'fill' writes constant 0.02."
        ),
    )
    parser.add_argument(
        "--model-local-expert-weight-init",
        choices=("source_default", "normal", "normal1", "uniform", "rand_sign", "empty", "zero", "fill"),
        default="normal",
        help=(
            "Local expert weight initialization for model-backed benchmark "
            "modes after block.apply_ep(...). The default is normal because "
            "the benchmark otherwise times torch.empty() expert weights. "
            "Use 'empty' or 'source_default' only for diagnostics."
        ),
    )
    parser.add_argument(
        "--deepep-do-cpu-sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether DeepEP dispatch performs CPU sync for exact counts. "
            "Keep enabled for first milestone correctness/timing."
        ),
    )
    parser.add_argument(
        "--deepep-async",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use DeepEP async_with_compute_stream and wait explicitly before "
            "dependent work."
        ),
    )
    parser.add_argument(
        "--deepep-prefer-overlap-with-compute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to DeepEP ElasticBuffer prefer_overlap_with_compute.",
    )
    parser.add_argument(
        "--deepep-allow-hybrid-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to DeepEP ElasticBuffer allow_hybrid_mode.",
    )
    parser.add_argument(
        "--deepep-allow-multiple-reduction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to DeepEP ElasticBuffer allow_multiple_reduction.",
    )
    parser.add_argument(
        "--deepep-wave-num-waves",
        type=int,
        default=4,
        help="Number of contiguous local-expert waves for --modes deepep_v2_wave.",
    )
    parser.add_argument(
        "--deepep-wave-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For --modes deepep_v2_wave, enqueue dispatch, expert compute, "
            "and combine on separate streams using DeepEP async events."
        ),
    )
    parser.add_argument(
        "--deepep-wave-layout",
        choices=("expand", "expand_static", "nonexpand_pack"),
        default="expand",
        help=(
            "DeepEP V2 wave data layout. 'expand' uses DeepEP's one-row-per-route "
            "expanded dispatch. 'expand_static' uses the modified DeepEP "
            "dispatch_expanded_into API to write each wave into a preallocated "
            "global expanded buffer. 'nonexpand_pack' uses non-expanded DeepEP "
            "dispatch then locally packs valid routes into expert-major grouped-GEMM input."
        ),
    )
    parser.add_argument(
        "--deepep-validate-wave-forward",
        action="store_true",
        help=(
            "For --modes deepep_v2_wave, run one no-wave forward and one wave "
            "forward before warmup, compare combined outputs across ranks, and "
            "raise if the max absolute difference exceeds "
            "--deepep-validate-wave-forward-atol."
        ),
    )
    parser.add_argument(
        "--deepep-validate-wave-forward-atol",
        type=float,
        default=5e-2,
        help="Absolute tolerance for --deepep-validate-wave-forward.",
    )
    parser.add_argument(
        "--deepep-wave-do-cpu-sync",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override DeepEP dispatch CPU sync in --modes deepep_v2_wave. "
            "By default it follows --deepep-do-cpu-sync."
        ),
    )
    parser.add_argument(
        "--rowwise-wave-num-waves",
        type=int,
        default=4,
        help="Number of contiguous local-expert waves for --modes rowwise_wave.",
    )
    parser.add_argument(
        "--rowwise-wave-recompute-linear1",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether rowwise_wave backward recomputes expert linear1/up-gate. "
            "Default is false, which saves the forward linear1 output."
        ),
    )
    parser.add_argument(
        "--rowwise-wave-recompute-act",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether rowwise_wave backward recomputes the SwiGLU activation output. "
            "Default is false, which saves the forward activation output."
        ),
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument(
        "--pass-type",
        choices=("forward", "backward", "forward_backward"),
        default="forward_backward",
        help=(
            "Measure forward only, backward only, or training-style "
            "forward+backward. Backward-only prepares the graph before the "
            "timed region."
        ),
    )
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on expert modules.")
    parser.add_argument("--no-compile", action="store_true", help="Deprecated no-op; compile is disabled by default.")
    parser.add_argument(
        "--compile-block",
        action="store_true",
        help="With --compile, compile the entire block instead of only expert modules.",
    )
    parser.add_argument("--no-shared-expert", action="store_true")
    parser.add_argument("--shared-hidden-size", type=int, default=4096)
    parser.add_argument(
        "--full-block",
        action="store_true",
        help="Do not patch attention/residual helpers to identity.",
    )
    parser.add_argument(
        "--random-routing",
        action="store_true",
        help="Use router random_expert_assignment=True instead of uniform assignment.",
    )
    parser.add_argument(
        "--balanced-routing",
        choices=("default", "deepep"),
        default="default",
        help=(
            "Benchmark-only deterministic routing override. 'deepep' makes "
            "model-backed rowwise modes use the same global top-k index formula "
            "as standalone deepep_v2."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Wrap measured iterations in cudaProfilerStart/Stop for nsys capture.",
    )
    parser.add_argument(
        "--check-standard-ep-umma-parity",
        action="store_true",
        help=(
            "For standard_ep_mega*_umma modes, run one untimed WMMA baseline "
            "iteration and assert UMMA output parity before timing."
        ),
    )
    parser.add_argument(
        "--standard-ep-umma-parity-atol",
        type=float,
        default=2e-2,
        help="Absolute tolerance for --check-standard-ep-umma-parity.",
    )
    return parser.parse_args()


def _dtype_config(name: str) -> tuple[DType, torch.dtype]:
    if name == "bf16":
        return DType.bfloat16, torch.bfloat16
    if name == "fp32":
        return DType.float32, torch.float32
    raise ValueError(name)


def _init_dist() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return dist.get_rank(), local_rank, dist.get_world_size()


def _build_ep_mesh(world_size: int) -> DeviceMesh:
    mesh = torch.arange(world_size, dtype=torch.int).view(1, world_size)
    return DeviceMesh(
        device_type="cuda",
        mesh=mesh,
        mesh_dim_names=("ep_dp", "ep_mp"),
    )


def _build_block(
    *,
    d_model: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    capacity_factor: float,
    rowwise_nblocks: int,
    use_wave: bool,
    use_bf16_persistent_mega: bool,
    rowwise_wave: bool,
    rowwise_wave_num_waves: int,
    rowwise_wave_recompute_linear1: bool,
    rowwise_wave_recompute_act: bool,
    include_shared_expert: bool,
    shared_hidden_size: int,
    uniform_routing: bool,
    random_routing: bool,
    config_dtype: DType,
) -> OLMoDDPTransformerBlock:
    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=config_dtype,
    )
    block = OLMoDDPTransformerBlock(
        d_model=d_model,
        block_idx=0,
        n_layers=1,
        sequence_mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=2,
            n_kv_heads=2,
            bias=False,
            use_flash=False,
            dtype=config_dtype,
        ),
        attention_norm=layer_norm,
        routed_experts_router=MoERouterConfigV2(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            gating_function=MoERouterGatingFunction.softmax,
            uniform_expert_assignment=uniform_routing and not random_routing,
            random_expert_assignment=random_routing,
            lb_loss_weight=None,
            z_loss_weight=None,
            dtype=config_dtype,
        ),
        routed_experts=RoutedExpertsConfig(
            d_model=d_model,
            hidden_size=hidden_size,
            num_experts=num_experts,
            bias=False,
            dtype=config_dtype,
        ),
        shared_experts=(
            SharedExpertsConfig(
                d_model=d_model,
                hidden_size=shared_hidden_size,
                num_experts=1,
                bias=False,
                dtype=config_dtype,
            )
            if include_shared_expert
            else None
        ),
        shared_experts_router=None,
        feed_forward_norm=layer_norm,
        ep=ExpertParallelConfig(
            path=(
                ExpertParallelPath.rowwise_wave
                if rowwise_wave
                else ExpertParallelPath.wave_mega
                if use_wave
                else ExpertParallelPath.rowwise_nvshmem
            ),
            capacity_factor=capacity_factor,
            major_align=1,
            rowwise_nblocks=rowwise_nblocks,
            rowwise_wave_num_waves=(
                rowwise_wave_num_waves if rowwise_wave else 1
            ),
            rowwise_wave_recompute_linear1=(
                rowwise_wave_recompute_linear1 if rowwise_wave else False
            ),
            rowwise_wave_recompute_act=(
                rowwise_wave_recompute_act if rowwise_wave else False
            ),
            wave_use_bf16_persistent_mega_forward=(
                use_bf16_persistent_mega and use_wave
            ),
        ),
        init_device="cuda",
    )
    return block


def _patch_moe_only(block: OLMoDDPTransformerBlock) -> None:
    def ident_attn(self, block_inp, **kwargs):
        return block_inp

    def mlp_only(self, residual, mlp_out):
        return mlp_out

    block._checkpointed_res_norm_attn = types.MethodType(ident_attn, block)
    block._res_norm_attn = types.MethodType(ident_attn, block)
    block._res_norm_mlp = types.MethodType(mlp_only, block)


def _compile_hot_modules(block: OLMoDDPTransformerBlock) -> None:
    if block.routed_experts is not None:
        block.routed_experts.forward = torch.compile(  # type: ignore[method-assign]
            block.routed_experts.forward,
            fullgraph=False,
            dynamic=False,
        )
        block.routed_experts.forward_row_offset = torch.compile(  # type: ignore[method-assign]
            block.routed_experts.forward_row_offset,
            fullgraph=False,
            dynamic=False,
        )
    if block.shared_experts is not None:
        block.shared_experts.forward1 = torch.compile(  # type: ignore[method-assign]
            block.shared_experts.forward1,
            fullgraph=False,
            dynamic=False,
        )
        block.shared_experts.forward2 = torch.compile(  # type: ignore[method-assign]
            block.shared_experts.forward2,
            fullgraph=False,
            dynamic=False,
        )


def _install_ep_balanced_router(block: OLMoDDPTransformerBlock) -> None:
    """Install deterministic routing that exercises every EP destination rank.

    The default uniform benchmark router can concentrate routes on a subset of
    expert shards for tiny smoke shapes. The peer-group wave path is still an
    experimental standard-shape bring-up kernel, so benchmark it with a route
    pattern that intentionally sends one top-k slot to each EP rank.
    """

    def _make_forward(router):
        def _forward(local_x, scores_only, loss_div_factor=None):
            del loss_div_factor
            B, S, _ = local_x.shape
            if scores_only:
                return torch.ones(
                    B,
                    S,
                    router.num_experts,
                    device=local_x.device,
                    dtype=local_x.dtype,
                ), None, None, None

            tokens = B * S
            token_idx = torch.arange(tokens, device=local_x.device, dtype=torch.long).view(
                tokens,
                1,
            )
            topk_idx = torch.arange(
                router.top_k,
                device=local_x.device,
                dtype=torch.long,
            ).view(1, router.top_k)
            experts_per_rank = router.num_experts // router.top_k
            expert_indices = topk_idx * experts_per_rank + (
                token_idx + dist.get_rank() + topk_idx
            ) % experts_per_rank
            expert_indices = expert_indices.view(B, S, router.top_k).contiguous()

            expert_weights = torch.full(
                (B, S, router.top_k),
                1.0 / float(router.top_k),
                device=local_x.device,
                dtype=local_x.dtype,
            )
            batch_size_per_expert = torch.bincount(
                expert_indices.reshape(-1),
                minlength=router.num_experts,
            ).to(dtype=torch.long)
            return expert_weights, expert_indices, batch_size_per_expert, None

        return _forward

    assert block.routed_experts_router is not None
    block.routed_experts_router.forward = _make_forward(block.routed_experts_router)


def _install_deepep_balanced_router(
    block: OLMoDDPTransformerBlock,
    *,
    world_size: int,
) -> None:
    """Install the exact deterministic top-k formula used by deepep_v2."""

    def _make_forward(router):
        def _forward(local_x, scores_only, loss_div_factor=None):
            del loss_div_factor
            B, S, _ = local_x.shape
            if scores_only:
                return torch.ones(
                    B,
                    S,
                    router.num_experts,
                    device=local_x.device,
                    dtype=local_x.dtype,
                ), None, None, None

            if router.num_experts % world_size != 0:
                raise RuntimeError(
                    "deepep balanced routing requires num_experts divisible by "
                    f"world_size ({router.num_experts} vs {world_size})"
                )

            tokens = B * S
            local_experts = router.num_experts // world_size
            token_idx = torch.arange(
                tokens,
                device=local_x.device,
                dtype=torch.long,
            ).view(tokens, 1)
            slot_idx = torch.arange(
                router.top_k,
                device=local_x.device,
                dtype=torch.long,
            ).view(1, router.top_k)
            expert_rank = (token_idx + dist.get_rank() + slot_idx) % world_size
            local_expert = (token_idx * router.top_k + slot_idx) % local_experts
            expert_indices = (
                expert_rank * local_experts + local_expert
            ).view(B, S, router.top_k).contiguous()

            expert_weights = torch.full(
                (B, S, router.top_k),
                1.0 / float(router.top_k),
                device=local_x.device,
                dtype=local_x.dtype,
            )
            batch_size_per_expert = torch.bincount(
                expert_indices.reshape(-1),
                minlength=router.num_experts,
            ).to(dtype=torch.long)
            return expert_weights, expert_indices, batch_size_per_expert, None

        return _forward

    assert block.routed_experts_router is not None
    block.routed_experts_router.forward = _make_forward(block.routed_experts_router)


def _cuda_profiler_start() -> None:
    torch.cuda.cudart().cudaProfilerStart()


def _cuda_profiler_stop() -> None:
    torch.cuda.cudart().cudaProfilerStop()


def _run_one_iter(
    block: OLMoDDPTransformerBlock,
    *,
    tokens: int,
    d_model: int,
    input_dtype: torch.dtype,
    label: str,
    pass_type: str,
    static_input: torch.Tensor | None = None,
) -> None:
    if static_input is None:
        torch.cuda.nvtx.range_push(f"{label}/input")
        try:
            x = torch.randn(
                1,
                tokens,
                d_model,
                device="cuda",
                dtype=input_dtype,
                requires_grad=(pass_type in ("backward", "forward_backward")),
            )
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        if pass_type != "forward":
            raise RuntimeError("static_input is only supported for forward profiling")
        x = static_input

    torch.cuda.nvtx.range_push(f"{label}/forward")
    try:
        if pass_type == "forward":
            with torch.no_grad():
                y = block(x)
        else:
            y = block(x)
    finally:
        torch.cuda.nvtx.range_pop()

    if pass_type == "forward":
        return

    torch.cuda.nvtx.range_push(f"{label}/loss")
    try:
        loss = y.square().mean()
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/backward")
    try:
        loss.backward()
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/zero_grad")
    try:
        block.zero_grad(set_to_none=True)
    finally:
        torch.cuda.nvtx.range_pop()


def _prepare_backward_loss(
    block: OLMoDDPTransformerBlock,
    *,
    tokens: int,
    d_model: int,
    input_dtype: torch.dtype,
    label: str,
) -> torch.Tensor:
    torch.cuda.nvtx.range_push(f"{label}/input")
    try:
        x = torch.randn(
            1,
            tokens,
            d_model,
            device="cuda",
            dtype=input_dtype,
            requires_grad=True,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/forward_prep")
    try:
        y = block(x)
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/loss_prep")
    try:
        return y.square().mean()
    finally:
        torch.cuda.nvtx.range_pop()


def _run_backward_from_loss(
    block: OLMoDDPTransformerBlock,
    loss: torch.Tensor,
    *,
    label: str,
) -> None:
    torch.cuda.nvtx.range_push(f"{label}/backward")
    try:
        loss.backward()
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/zero_grad")
    try:
        block.zero_grad(set_to_none=True)
    finally:
        torch.cuda.nvtx.range_pop()


def _import_deepep(deepep_path: str):
    if deepep_path:
        deepep_path = os.path.abspath(deepep_path)
        if os.path.isdir(deepep_path) and deepep_path not in sys.path:
            sys.path.insert(0, deepep_path)
    try:
        import deep_ep  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(
            "Failed to import DeepEP for --modes deepep_v2. "
            "Build/install DeepEP first, or pass --deepep-path /path/to/DeepEP. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e
    return deep_ep


@dataclass
class DeepEpV2State:
    deep_ep: object
    buffer: object
    routed_experts: torch.nn.Module
    source_input: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    rank: int
    world_size: int
    num_experts: int
    num_local_experts: int
    num_max_tokens_per_rank: int
    expert_alignment: int
    num_sms: int
    num_qps: int
    expert_buffer_mode: str
    async_with_compute_stream: bool
    do_cpu_sync: bool
    wave_dispatch_stream: torch.cuda.Stream | None = None
    wave_compute_stream: torch.cuda.Stream | None = None
    wave_combine_stream: torch.cuda.Stream | None = None


@dataclass
class DeepEpV2ForwardResult:
    recv_x: torch.Tensor
    expanded_topk_weights: torch.Tensor
    expert_out: torch.Tensor
    combined_x: torch.Tensor
    handle: object
    grad_combined_x: torch.Tensor | None = None


@dataclass
class DeepEpV2WaveForwardResult:
    wave_results: list[DeepEpV2ForwardResult]
    combined_x: torch.Tensor
    grad_combined_x: torch.Tensor | None = None


@dataclass(frozen=True)
class DeepEpV2WaveInput:
    wave_idx: int
    expert_start: int
    expert_end: int
    wave_base: int
    wave_end: int
    batch_size_per_expert: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def wave_rows(self) -> int:
        return self.wave_end - self.wave_base


def _make_balanced_topk_idx(
    *,
    tokens: int,
    top_k: int,
    num_experts: int,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_experts % world_size != 0:
        raise RuntimeError(
            f"deepep_v2 requires --num-experts divisible by ranks "
            f"({num_experts} vs {world_size})"
        )
    local_experts = num_experts // world_size
    token_idx = torch.arange(tokens, device="cuda", dtype=torch.long).view(tokens, 1)
    slot_idx = torch.arange(top_k, device="cuda", dtype=torch.long).view(1, top_k)
    expert_rank = (token_idx + rank + slot_idx) % world_size
    local_expert = (token_idx * top_k + slot_idx) % local_experts
    return (expert_rank * local_experts + local_expert).to(dtype=dtype).contiguous()


def _align_int(value: int, alignment: int) -> int:
    if alignment <= 1:
        return int(value)
    return ((int(value) + alignment - 1) // alignment) * alignment


def _deepep_v2_local_expert_counts(state: DeepEpV2State) -> list[int]:
    topk_idx_long = state.topk_idx.to(dtype=torch.long)
    valid_idx = topk_idx_long[topk_idx_long >= 0]
    global_counts = torch.bincount(
        valid_idx,
        minlength=state.num_experts,
    ).to(dtype=torch.long)
    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)

    local_start = state.rank * state.num_local_experts
    local_end = local_start + state.num_local_experts
    return [
        int(v)
        for v in global_counts[local_start:local_end].detach().cpu().tolist()
    ]


def _deepep_v2_expanded_offsets(
    counts: Sequence[int],
    *,
    expert_alignment: int,
) -> list[int]:
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + _align_int(int(count), expert_alignment))
    return offsets


def _expanded_expert_counts(handle: object, expert_alignment: int) -> torch.Tensor:
    psum = handle.psum_num_recv_tokens_per_expert
    if psum.ndim != 1:
        raise RuntimeError(
            "DeepEP handle.psum_num_recv_tokens_per_expert must be 1D "
            f"(got shape={tuple(psum.shape)})"
        )
    starts = torch.empty_like(psum)
    starts.fill_(0)
    if psum.numel() > 1:
        previous = psum[:-1]
        if expert_alignment == 1:
            starts[1:] = previous
        else:
            starts[1:] = ((previous + expert_alignment - 1) // expert_alignment) * expert_alignment
    return (psum - starts).to(dtype=torch.int32)


def _num_recv_tokens(handle: object, *, device: torch.device) -> torch.Tensor:
    psum = handle.psum_num_recv_tokens_per_scaleup_rank
    if psum.ndim != 1:
        raise RuntimeError(
            "DeepEP handle.psum_num_recv_tokens_per_scaleup_rank must be 1D "
            f"(got shape={tuple(psum.shape)})"
        )
    return psum[-1].to(device=device, dtype=torch.long)


def _deep_ep_wait(event: object, *, async_with_compute_stream: bool) -> None:
    if async_with_compute_stream:
        event.current_stream_wait()


def _reshape_expanded_weights(
    expanded_topk_weights: torch.Tensor | None,
    *,
    num_rows: int,
    dtype: torch.dtype,
    default_weight: float = 1.0,
) -> torch.Tensor:
    if expanded_topk_weights is None:
        return torch.full((num_rows, 1), default_weight, device="cuda", dtype=dtype)
    if expanded_topk_weights.numel() != num_rows:
        raise RuntimeError(
            "DeepEP expanded top-k weights do not match expanded rows: "
            f"weights={tuple(expanded_topk_weights.shape)} rows={num_rows}"
        )
    return expanded_topk_weights.reshape(num_rows, 1).to(dtype=dtype)


def _validate_deepep_v2_args(args: argparse.Namespace, *, world_size: int) -> None:
    if args.dtype != "bf16":
        raise RuntimeError("deepep_v2 currently supports --dtype bf16 only")
    if os.getenv("EP_REUSE_NCCL_COMM", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        raise RuntimeError(
            "deepep_v2 currently requires EP_REUSE_NCCL_COMM=0 in this "
            "environment. Reusing Torch's NCCL communicator can load a mixed "
            "NCCL runtime and segfault during DeepEP ElasticBuffer setup."
        )
    if args.d_model % 256 != 0:
        raise RuntimeError(
            "deepep_v2 BF16 combine requires --d-model divisible by 256 "
            f"(got {args.d_model})."
        )
    if args.num_experts % world_size != 0:
        raise RuntimeError(
            f"deepep_v2 requires --num-experts divisible by ranks "
            f"({args.num_experts} vs {world_size})"
        )
    if args.deepep_expert_alignment < 1:
        raise RuntimeError("--deepep-expert-alignment must be >= 1")
    if args.deepep_max_tokens_factor < 1.0:
        raise RuntimeError("--deepep-max-tokens-factor must be >= 1.0")


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


def _build_deepep_v2_probe_routed_experts(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    config_dtype: DType,
    reset_seed: bool = True,
) -> torch.nn.Module:
    if reset_seed:
        torch.manual_seed(20260625 + rank)
    num_local_experts = args.num_experts // world_size
    routed_experts = RoutedExpertsConfig(
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        num_experts=num_local_experts,
        bias=False,
        dtype=config_dtype,
    ).build(init_device="cuda")
    _init_probe_routed_expert_weights(
        routed_experts,
        weight_init=_resolve_probe_weight_init(args, source_default="normal"),
    )
    routed_experts.train()
    if args.compile and not args.no_compile:
        routed_experts.forward = torch.compile(  # type: ignore[method-assign]
            routed_experts.forward,
            fullgraph=False,
            dynamic=False,
        )
    return routed_experts


def _build_rowwise_apply_ep_probe_routed_experts(
    args: argparse.Namespace,
    *,
    world_size: int,
    ep_mesh: DeviceMesh,
    config_dtype: DType,
) -> torch.nn.Module:
    block = _build_block(
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        capacity_factor=args.capacity_factor,
        rowwise_nblocks=args.rowwise_nblocks,
        use_wave=False,
        use_bf16_persistent_mega=False,
        rowwise_wave=False,
        rowwise_wave_num_waves=args.rowwise_wave_num_waves,
        rowwise_wave_recompute_linear1=args.rowwise_wave_recompute_linear1,
        rowwise_wave_recompute_act=args.rowwise_wave_recompute_act,
        include_shared_expert=not args.no_shared_expert,
        shared_hidden_size=args.shared_hidden_size,
        uniform_routing=not args.random_routing,
        random_routing=args.random_routing,
        config_dtype=config_dtype,
    )
    if not args.full_block:
        _patch_moe_only(block)
    block.apply_ep(ep_mesh)
    assert block.routed_experts is not None
    _init_probe_routed_expert_weights(
        block.routed_experts,
        weight_init=_resolve_probe_weight_init(args, source_default="empty"),
    )
    if args.balanced_routing == "deepep":
        if args.random_routing:
            raise RuntimeError("--balanced-routing deepep conflicts with --random-routing")
        _install_deepep_balanced_router(block, world_size=world_size)
    block.train()
    if args.compile and not args.no_compile:
        if args.compile_block:
            block = torch.compile(block, fullgraph=False, dynamic=False)
        else:
            _compile_hot_modules(block)
    if block.routed_experts is None:
        raise RuntimeError("rowwise_apply_ep probe failed to build routed experts")
    return block.routed_experts


def _build_deepep_v2_state(
    args: argparse.Namespace,
    *,
    tokens: int,
    rank: int,
    world_size: int,
    config_dtype: DType,
    input_dtype: torch.dtype,
) -> DeepEpV2State:
    _validate_deepep_v2_args(args, world_size=world_size)

    deep_ep = _import_deepep(args.deepep_path)
    num_max_tokens_per_rank = int(math.ceil(tokens * args.deepep_max_tokens_factor))
    num_allocated_qps = max(args.deepep_num_allocated_qps, args.deepep_num_qps)
    buffer = deep_ep.ElasticBuffer(
        dist.group.WORLD,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=args.d_model,
        num_topk=args.top_k,
        deterministic=False,
        allow_hybrid_mode=args.deepep_allow_hybrid_mode,
        allow_multiple_reduction=args.deepep_allow_multiple_reduction,
        prefer_overlap_with_compute=args.deepep_prefer_overlap_with_compute,
        num_allocated_qps=num_allocated_qps,
        explicitly_destroy=True,
    )
    num_sms = (
        int(args.deepep_num_sms)
        if args.deepep_num_sms != 0
        else int(buffer.get_theoretical_num_sms(args.num_experts, args.top_k))
    )
    num_qps = (
        int(args.deepep_num_qps)
        if args.deepep_num_qps != 0
        else int(buffer.get_theoretical_num_qps(num_sms))
    )

    torch.manual_seed(20260625 + rank)
    source_input = (0.2 * torch.randn(tokens, args.d_model, device="cuda")).to(input_dtype)
    topk_idx = _make_balanced_topk_idx(
        tokens=tokens,
        top_k=args.top_k,
        num_experts=args.num_experts,
        world_size=world_size,
        rank=rank,
        dtype=deep_ep.topk_idx_t,
    )
    topk_weights = torch.full(
        (tokens, args.top_k),
        1.0 / float(args.top_k),
        device="cuda",
        dtype=torch.float32,
    )

    routed_experts = _build_deepep_v2_probe_routed_experts(
        args,
        rank=rank,
        world_size=world_size,
        config_dtype=config_dtype,
        reset_seed=False,
    )

    return DeepEpV2State(
        deep_ep=deep_ep,
        buffer=buffer,
        routed_experts=routed_experts,
        source_input=source_input,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        rank=rank,
        world_size=world_size,
        num_experts=args.num_experts,
        num_local_experts=args.num_experts // world_size,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        expert_alignment=args.deepep_expert_alignment,
        num_sms=num_sms,
        num_qps=num_qps,
        expert_buffer_mode=str(args.deepep_expert_buffer_mode),
        async_with_compute_stream=bool(args.deepep_async),
        do_cpu_sync=bool(args.deepep_do_cpu_sync),
        wave_dispatch_stream=torch.cuda.Stream(),
        wave_compute_stream=torch.cuda.Stream(),
        wave_combine_stream=torch.cuda.Stream(),
    )


def _deepep_v2_dispatch(
    state: DeepEpV2State,
    *,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    label: str,
    async_with_compute_stream: bool,
    do_cpu_sync: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, object, object]:
    torch.cuda.nvtx.range_push(label)
    try:
        recv_x, _recv_topk_idx, expanded_topk_weights, handle, event = state.buffer.dispatch(
            state.source_input,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=state.num_experts,
            num_max_tokens_per_rank=state.num_max_tokens_per_rank,
            expert_alignment=state.expert_alignment,
            num_sms=state.num_sms,
            num_qps=state.num_qps,
            async_with_compute_stream=async_with_compute_stream,
            do_cpu_sync=do_cpu_sync,
            do_expand=True,
            use_tma_aligned_col_major_sf=True,
        )
        _deep_ep_wait(event, async_with_compute_stream=async_with_compute_stream)
    finally:
        torch.cuda.nvtx.range_pop()
    return recv_x, expanded_topk_weights, handle, event


def _deepep_v2_dispatch_static_expanded(
    state: DeepEpV2State,
    *,
    wave: DeepEpV2WaveInput,
    recv_x_out: torch.Tensor,
    recv_topk_weights_out: torch.Tensor,
    label: str,
    async_with_compute_stream: bool,
    do_cpu_sync: bool,
) -> tuple[torch.Tensor, torch.Tensor, object, object]:
    if not hasattr(state.buffer, "dispatch_expanded_into"):
        raise RuntimeError(
            "deepep_v2_wave layout 'expand_static' requires the modified "
            "DeepEP working copy with ElasticBuffer.dispatch_expanded_into. "
            "Use --deepep-path /workspace/DeepEP."
        )

    torch.cuda.nvtx.range_push(label)
    try:
        recv_x, _recv_topk_idx, expanded_topk_weights, handle, event = (
            state.buffer.dispatch_expanded_into(
                state.source_input,
                topk_idx=wave.topk_idx,
                topk_weights=wave.topk_weights,
                recv_x_out=recv_x_out,
                recv_topk_weights_out=recv_topk_weights_out,
                expanded_row_offset=wave.wave_base,
                num_experts=state.num_experts,
                num_max_tokens_per_rank=state.num_max_tokens_per_rank,
                expert_alignment=state.expert_alignment,
                num_sms=state.num_sms,
                num_qps=state.num_qps,
                async_with_compute_stream=async_with_compute_stream,
                do_cpu_sync=do_cpu_sync,
            )
        )
        _deep_ep_wait(event, async_with_compute_stream=async_with_compute_stream)
    finally:
        torch.cuda.nvtx.range_pop()
    return recv_x, expanded_topk_weights, handle, event


def _deepep_v2_dispatch_nonexpanded(
    state: DeepEpV2State,
    *,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    label: str,
    async_with_compute_stream: bool,
    do_cpu_sync: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, object, object]:
    torch.cuda.nvtx.range_push(label)
    try:
        recv_x, recv_topk_idx, recv_topk_weights, handle, event = state.buffer.dispatch(
            state.source_input,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=state.num_experts,
            num_max_tokens_per_rank=state.num_max_tokens_per_rank,
            expert_alignment=state.expert_alignment,
            num_sms=state.num_sms,
            num_qps=state.num_qps,
            async_with_compute_stream=async_with_compute_stream,
            do_cpu_sync=do_cpu_sync,
            do_expand=False,
            use_tma_aligned_col_major_sf=True,
        )
        _deep_ep_wait(event, async_with_compute_stream=async_with_compute_stream)
    finally:
        torch.cuda.nvtx.range_pop()
    return recv_x, recv_topk_idx, recv_topk_weights, handle, event


def _deepep_v2_compute_experts(
    state: DeepEpV2State,
    *,
    recv_x: torch.Tensor,
    expanded_topk_weights: torch.Tensor | None,
    handle: object,
    label: str,
    track_expert_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size_per_expert = _expanded_expert_counts(handle, state.expert_alignment)
    recv_x_for_experts = recv_x.detach().requires_grad_(True) if track_expert_grad else recv_x
    down_proj_out = None
    up_proj_input_grad_out = None
    if state.expert_buffer_mode in {"down", "all"}:
        down_proj_out = torch.empty_like(recv_x)
    if state.expert_buffer_mode == "all":
        up_proj_input_grad_out = recv_x_for_experts.detach()

    torch.cuda.nvtx.range_push(label)
    try:
        expert_out = state.routed_experts(
            recv_x_for_experts,
            batch_size_per_expert,
            down_proj_out=down_proj_out,
            up_proj_input_grad_out=up_proj_input_grad_out,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    expanded_weights = _reshape_expanded_weights(
        expanded_topk_weights,
        num_rows=expert_out.shape[0],
        dtype=expert_out.dtype,
    )
    return recv_x_for_experts, expert_out, expanded_weights


def _deepep_v2_compute_experts_static_expanded(
    state: DeepEpV2State,
    *,
    recv_x: torch.Tensor,
    expanded_topk_weights: torch.Tensor,
    weighted_expert_out: torch.Tensor,
    wave: DeepEpV2WaveInput,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if state.expert_alignment != 1:
        raise RuntimeError(
            "deepep_v2_wave layout 'expand_static' currently requires "
            "--deepep-expert-alignment 1. The current autograd RoutedExperts "
            "path consumes packed expert rows, while aligned static rows can "
            "contain padding between experts."
        )

    recv_x_for_experts = recv_x.narrow(0, wave.wave_base, wave.wave_rows)
    expanded_topk_weights_for_experts = expanded_topk_weights.narrow(
        0,
        wave.wave_base,
        wave.wave_rows,
    )
    down_proj_out = None
    if state.expert_buffer_mode in {"down", "all"}:
        down_proj_out = torch.empty_like(recv_x_for_experts)

    torch.cuda.nvtx.range_push(label)
    try:
        expert_out = state.routed_experts(
            recv_x_for_experts,
            wave.batch_size_per_expert,
            down_proj_out=down_proj_out,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    expanded_weights = _reshape_expanded_weights(
        expanded_topk_weights_for_experts,
        num_rows=expert_out.shape[0],
        dtype=expert_out.dtype,
    )
    weighted_slice = _deepep_v2_weight_expert_output(
        expert_out,
        expanded_weights,
        label=label,
    )
    torch.cuda.nvtx.range_push(f"{label}/store_global_weighted")
    try:
        weighted_expert_out.narrow(0, wave.wave_base, wave.wave_rows).copy_(weighted_slice)
    finally:
        torch.cuda.nvtx.range_pop()

    return recv_x_for_experts, expert_out, expanded_weights


def _deepep_v2_compute_experts_nonexpanded(
    state: DeepEpV2State,
    *,
    recv_x: torch.Tensor,
    recv_topk_idx: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    handle: object,
    label: str,
    track_expert_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recv_x_for_experts = recv_x.detach().requires_grad_(True) if track_expert_grad else recv_x

    torch.cuda.nvtx.range_push(f"{label}/pack_routes")
    try:
        recv_token_idx = torch.arange(
            recv_topk_idx.shape[0],
            device=recv_topk_idx.device,
            dtype=torch.long,
        ).view(-1, 1)
        valid_route_mask = (
            (recv_token_idx < _num_recv_tokens(handle, device=recv_topk_idx.device))
            & (recv_topk_idx >= 0)
            & (recv_topk_idx < state.num_local_experts)
        )
        route_token_idx, route_slot_idx = torch.nonzero(
            valid_route_mask,
            as_tuple=True,
        )
        route_expert_idx = recv_topk_idx[route_token_idx, route_slot_idx].to(torch.long)
        route_order = torch.argsort(route_expert_idx, stable=True)
        route_token_idx = route_token_idx[route_order]
        route_slot_idx = route_slot_idx[route_order]
        route_expert_idx = route_expert_idx[route_order]
        batch_size_per_expert = torch.bincount(
            route_expert_idx,
            minlength=state.num_local_experts,
        ).to(dtype=torch.int32)
        packed_x = recv_x_for_experts.index_select(0, route_token_idx)
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(label)
    try:
        expert_out = state.routed_experts(
            packed_x,
            batch_size_per_expert,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/local_reduce")
    try:
        route_weights = recv_topk_weights[route_token_idx, route_slot_idx].to(
            dtype=expert_out.dtype,
        ).view(-1, 1)
        local_accum = torch.zeros_like(recv_x_for_experts)
        local_accum.index_add_(0, route_token_idx, expert_out * route_weights)
        local_accum_weights = torch.ones(
            (local_accum.shape[0], 1),
            device=local_accum.device,
            dtype=local_accum.dtype,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    return recv_x_for_experts, local_accum, local_accum_weights


def _deepep_v2_weight_expert_output(
    expert_out: torch.Tensor,
    expanded_weights: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    torch.cuda.nvtx.range_push(f"{label}/weight")
    try:
        weighted_expert_out = expert_out * expanded_weights
    finally:
        torch.cuda.nvtx.range_pop()
    return weighted_expert_out


def _deepep_v2_combine(
    state: DeepEpV2State,
    *,
    weighted_expert_out: torch.Tensor,
    handle: object,
    label: str,
    async_with_compute_stream: bool,
) -> tuple[torch.Tensor, object]:
    torch.cuda.nvtx.range_push(label)
    try:
        combined_x, _combined_topk_weights, event = state.buffer.combine(
            weighted_expert_out,
            handle=handle,
            num_sms=state.num_sms,
            num_qps=state.num_qps,
            async_with_compute_stream=async_with_compute_stream,
        )
        _deep_ep_wait(event, async_with_compute_stream=async_with_compute_stream)
    finally:
        torch.cuda.nvtx.range_pop()
    return combined_x, event


def _deepep_v2_forward_from_topk(
    state: DeepEpV2State,
    *,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    label: str,
    track_expert_grad: bool,
    async_with_compute_stream: bool,
    do_cpu_sync: bool,
) -> DeepEpV2ForwardResult:
    recv_x, expanded_topk_weights, handle, _dispatch_event = _deepep_v2_dispatch(
        state,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        label=f"{label}/dispatch",
        async_with_compute_stream=async_with_compute_stream,
        do_cpu_sync=do_cpu_sync,
    )
    recv_x_for_experts, expert_out, expanded_weights = _deepep_v2_compute_experts(
        state,
        recv_x=recv_x,
        expanded_topk_weights=expanded_topk_weights,
        handle=handle,
        label=f"{label}/experts",
        track_expert_grad=track_expert_grad,
    )
    weighted_expert_out = _deepep_v2_weight_expert_output(
        expert_out,
        expanded_weights,
        label=label,
    )
    combined_x, _combine_event = _deepep_v2_combine(
        state,
        weighted_expert_out=weighted_expert_out,
        handle=handle,
        label=f"{label}/combine",
        async_with_compute_stream=async_with_compute_stream,
    )

    return DeepEpV2ForwardResult(
        recv_x=recv_x_for_experts,
        expanded_topk_weights=expanded_weights,
        expert_out=expert_out,
        combined_x=combined_x,
        handle=handle,
    )


def _deepep_v2_forward_from_topk_nonexpanded(
    state: DeepEpV2State,
    *,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    label: str,
    track_expert_grad: bool,
    async_with_compute_stream: bool,
    do_cpu_sync: bool,
) -> DeepEpV2ForwardResult:
    recv_x, recv_topk_idx, recv_topk_weights, handle, _dispatch_event = (
        _deepep_v2_dispatch_nonexpanded(
            state,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            label=f"{label}/dispatch_nonexpanded",
            async_with_compute_stream=async_with_compute_stream,
            do_cpu_sync=do_cpu_sync,
        )
    )
    recv_x_for_experts, local_accum, local_accum_weights = (
        _deepep_v2_compute_experts_nonexpanded(
            state,
            recv_x=recv_x,
            recv_topk_idx=recv_topk_idx,
            recv_topk_weights=recv_topk_weights,
            handle=handle,
            label=f"{label}/experts_nonexpanded",
            track_expert_grad=track_expert_grad,
        )
    )
    combined_x, _combine_event = _deepep_v2_combine(
        state,
        weighted_expert_out=local_accum,
        handle=handle,
        label=f"{label}/combine_nonexpanded",
        async_with_compute_stream=async_with_compute_stream,
    )

    return DeepEpV2ForwardResult(
        recv_x=recv_x_for_experts,
        expanded_topk_weights=local_accum_weights,
        expert_out=local_accum,
        combined_x=combined_x,
        handle=handle,
    )


def _deepep_v2_forward(
    state: DeepEpV2State,
    *,
    label: str,
    track_expert_grad: bool,
) -> DeepEpV2ForwardResult:
    return _deepep_v2_forward_from_topk(
        state,
        topk_idx=state.topk_idx,
        topk_weights=state.topk_weights,
        label=label,
        track_expert_grad=track_expert_grad,
        async_with_compute_stream=state.async_with_compute_stream,
        do_cpu_sync=state.do_cpu_sync,
    )


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

    topk_idx_long = state.topk_idx.to(dtype=torch.long)
    valid = topk_idx_long >= 0
    local_expert = torch.remainder(topk_idx_long, state.num_local_experts)
    invalid_idx = torch.full_like(state.topk_idx, -1)
    zero_weights = torch.zeros_like(state.topk_weights)
    local_counts = _deepep_v2_local_expert_counts(state)
    local_offsets = _deepep_v2_expanded_offsets(
        local_counts,
        expert_alignment=state.expert_alignment,
    )

    wave_inputs: list[DeepEpV2WaveInput] = []
    for wave_idx in range(num_waves):
        expert_start = (wave_idx * state.num_local_experts) // num_waves
        expert_end = ((wave_idx + 1) * state.num_local_experts) // num_waves
        if expert_start == expert_end:
            raise RuntimeError(
                f"empty DeepEP wave {wave_idx}: local_experts={state.num_local_experts} "
                f"num_waves={num_waves}"
            )
        wave_mask = valid & (local_expert >= expert_start) & (local_expert < expert_end)
        batch_size_per_expert = torch.zeros(
            (state.num_local_experts,),
            device=state.source_input.device,
            dtype=torch.int32,
        )
        if expert_end > expert_start:
            batch_size_per_expert[expert_start:expert_end] = torch.tensor(
                local_counts[expert_start:expert_end],
                device=state.source_input.device,
                dtype=torch.int32,
            )
        wave_inputs.append(
            DeepEpV2WaveInput(
                wave_idx=wave_idx,
                expert_start=expert_start,
                expert_end=expert_end,
                wave_base=local_offsets[expert_start],
                wave_end=local_offsets[expert_end],
                batch_size_per_expert=batch_size_per_expert,
                topk_idx=torch.where(wave_mask, state.topk_idx, invalid_idx).contiguous(),
                topk_weights=torch.where(wave_mask, state.topk_weights, zero_weights).contiguous(),
            )
        )
    return wave_inputs


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


def _deepep_v2_static_expanded_buffers(
    state: DeepEpV2State,
    wave_inputs: list[DeepEpV2WaveInput],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return recv_x, expanded_topk_weights, weighted_expert_out


def _deepep_v2_wave_forward_sequential(
    state: DeepEpV2State,
    *,
    wave_inputs: list[DeepEpV2WaveInput],
    wave_layout: str,
    label: str,
    track_expert_grad: bool,
    do_cpu_sync: bool,
) -> DeepEpV2WaveForwardResult:
    if wave_layout == "expand_static" and track_expert_grad:
        raise RuntimeError(
            "deepep_v2_wave layout 'expand_static' is forward-only for now. "
            "DeepEP cached expanded backward is not implemented in this path yet."
        )
    wave_results: list[DeepEpV2ForwardResult] = []
    wave_outputs: list[torch.Tensor] = []
    static_buffers = (
        _deepep_v2_static_expanded_buffers(state, wave_inputs)
        if wave_layout == "expand_static"
        else None
    )
    for wave in wave_inputs:
        wave_label = f"{label}/wave_{wave.wave_idx}_experts_{wave.expert_start}_{wave.expert_end}"
        if wave_layout == "expand":
            result = _deepep_v2_forward_from_topk(
                state,
                topk_idx=wave.topk_idx,
                topk_weights=wave.topk_weights,
                label=wave_label,
                track_expert_grad=track_expert_grad,
                async_with_compute_stream=state.async_with_compute_stream,
                do_cpu_sync=do_cpu_sync,
            )
        elif wave_layout == "expand_static":
            assert static_buffers is not None
            recv_x_out, expanded_topk_weights_out, weighted_expert_out = static_buffers
            recv_x, expanded_topk_weights, handle, _dispatch_event = (
                _deepep_v2_dispatch_static_expanded(
                    state,
                    wave=wave,
                    recv_x_out=recv_x_out,
                    recv_topk_weights_out=expanded_topk_weights_out,
                    label=f"{wave_label}/dispatch_static",
                    async_with_compute_stream=state.async_with_compute_stream,
                    do_cpu_sync=do_cpu_sync,
                )
            )
            recv_x_for_experts, expert_out, expanded_weights = (
                _deepep_v2_compute_experts_static_expanded(
                    state,
                    recv_x=recv_x,
                    expanded_topk_weights=expanded_topk_weights,
                    weighted_expert_out=weighted_expert_out,
                    wave=wave,
                    label=f"{wave_label}/experts_static",
                )
            )
            combined_x, _combine_event = _deepep_v2_combine(
                state,
                weighted_expert_out=weighted_expert_out,
                handle=handle,
                label=f"{wave_label}/combine_static",
                async_with_compute_stream=state.async_with_compute_stream,
            )
            result = DeepEpV2ForwardResult(
                recv_x=recv_x_for_experts,
                expanded_topk_weights=expanded_weights,
                expert_out=expert_out,
                combined_x=combined_x,
                handle=handle,
            )
        elif wave_layout == "nonexpand_pack":
            result = _deepep_v2_forward_from_topk_nonexpanded(
                state,
                topk_idx=wave.topk_idx,
                topk_weights=wave.topk_weights,
                label=wave_label,
                track_expert_grad=track_expert_grad,
                async_with_compute_stream=state.async_with_compute_stream,
                do_cpu_sync=do_cpu_sync,
            )
        else:
            raise ValueError(wave_layout)
        wave_results.append(result)
        wave_outputs.append(result.combined_x)

    return DeepEpV2WaveForwardResult(
        wave_results=wave_results,
        combined_x=_sum_deepep_v2_wave_outputs(wave_outputs, label=f"{label}/sum_waves"),
    )


def _deepep_v2_wave_forward_overlapped(
    state: DeepEpV2State,
    *,
    wave_inputs: list[DeepEpV2WaveInput],
    wave_layout: str,
    label: str,
    track_expert_grad: bool,
    do_cpu_sync: bool,
) -> DeepEpV2WaveForwardResult:
    if (
        state.wave_dispatch_stream is None
        or state.wave_compute_stream is None
        or state.wave_combine_stream is None
    ):
        raise RuntimeError("DeepEP V2 wave streams were not initialized")
    if wave_layout == "expand_static" and track_expert_grad:
        raise RuntimeError(
            "deepep_v2_wave layout 'expand_static' is forward-only for now. "
            "DeepEP cached expanded backward is not implemented in this path yet."
        )

    async_mode = True
    static_buffers = (
        _deepep_v2_static_expanded_buffers(state, wave_inputs)
        if wave_layout == "expand_static"
        else None
    )

    dispatch_records: list[
        tuple[
            DeepEpV2WaveInput,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            object,
            object,
        ]
    ] = []
    for wave in wave_inputs:
        wave_label = f"{label}/wave_{wave.wave_idx}_experts_{wave.expert_start}_{wave.expert_end}"
        with torch.cuda.stream(state.wave_dispatch_stream):
            if wave_layout == "expand":
                recv_x, expanded_topk_weights, handle, dispatch_event = _deepep_v2_dispatch(
                    state,
                    topk_idx=wave.topk_idx,
                    topk_weights=wave.topk_weights,
                    label=f"{wave_label}/dispatch",
                    async_with_compute_stream=async_mode,
                    do_cpu_sync=do_cpu_sync,
                )
                recv_topk_idx = None
                recv_topk_weights = None
            elif wave_layout == "expand_static":
                assert static_buffers is not None
                recv_x_out, expanded_topk_weights_out, weighted_expert_out = static_buffers
                recv_x, expanded_topk_weights, handle, dispatch_event = (
                    _deepep_v2_dispatch_static_expanded(
                        state,
                        wave=wave,
                        recv_x_out=recv_x_out,
                        recv_topk_weights_out=expanded_topk_weights_out,
                        label=f"{wave_label}/dispatch_static",
                        async_with_compute_stream=async_mode,
                        do_cpu_sync=do_cpu_sync,
                    )
                )
                recv_topk_idx = None
                recv_topk_weights = None
            elif wave_layout == "nonexpand_pack":
                recv_x, recv_topk_idx, recv_topk_weights, handle, dispatch_event = (
                    _deepep_v2_dispatch_nonexpanded(
                        state,
                        topk_idx=wave.topk_idx,
                        topk_weights=wave.topk_weights,
                        label=f"{wave_label}/dispatch_nonexpanded",
                        async_with_compute_stream=async_mode,
                        do_cpu_sync=do_cpu_sync,
                    )
                )
                expanded_topk_weights = None
            else:
                raise ValueError(wave_layout)

        dispatch_records.append(
            (
                wave,
                recv_x,
                expanded_topk_weights,
                recv_topk_idx,
                recv_topk_weights,
                handle,
                dispatch_event,
            )
        )

    compute_records: list[
        tuple[
            DeepEpV2WaveInput,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            object,
            torch.cuda.Event,
        ]
    ] = []
    for (
        wave,
        recv_x,
        expanded_topk_weights,
        recv_topk_idx,
        recv_topk_weights,
        handle,
        dispatch_event,
    ) in dispatch_records:
        wave_label = f"{label}/wave_{wave.wave_idx}_experts_{wave.expert_start}_{wave.expert_end}"
        with torch.cuda.stream(state.wave_compute_stream):
            _deep_ep_wait(dispatch_event, async_with_compute_stream=async_mode)
            if wave_layout == "expand":
                recv_x_for_experts, expert_out, expanded_weights = _deepep_v2_compute_experts(
                    state,
                    recv_x=recv_x,
                    expanded_topk_weights=expanded_topk_weights,
                    handle=handle,
                    label=f"{wave_label}/experts",
                    track_expert_grad=track_expert_grad,
                )
                weighted_expert_out = _deepep_v2_weight_expert_output(
                    expert_out,
                    expanded_weights,
                    label=wave_label,
                )
            elif wave_layout == "expand_static":
                assert static_buffers is not None
                _recv_x_out, _expanded_topk_weights_out, weighted_expert_out = static_buffers
                recv_x_for_experts, expert_out, expanded_weights = (
                    _deepep_v2_compute_experts_static_expanded(
                        state,
                        recv_x=recv_x,
                        expanded_topk_weights=expanded_topk_weights,
                        weighted_expert_out=weighted_expert_out,
                        wave=wave,
                        label=f"{wave_label}/experts_static",
                    )
                )
            else:
                assert recv_topk_idx is not None
                assert recv_topk_weights is not None
                recv_x_for_experts, weighted_expert_out, expanded_weights = (
                    _deepep_v2_compute_experts_nonexpanded(
                        state,
                        recv_x=recv_x,
                        recv_topk_idx=recv_topk_idx,
                        recv_topk_weights=recv_topk_weights,
                        handle=handle,
                        label=f"{wave_label}/experts_nonexpanded",
                        track_expert_grad=track_expert_grad,
                    )
                )
                expert_out = weighted_expert_out
            compute_done = torch.cuda.Event(enable_timing=False)
            compute_done.record()

        compute_records.append(
            (
                wave,
                recv_x_for_experts,
                expanded_weights,
                expert_out,
                weighted_expert_out,
                handle,
                compute_done,
            )
        )

    wave_results: list[DeepEpV2ForwardResult] = []
    wave_outputs: list[torch.Tensor] = []
    combine_events: list[object] = []
    for (
        wave,
        recv_x_for_experts,
        expanded_weights,
        expert_out,
        weighted_expert_out,
        handle,
        compute_done,
    ) in compute_records:
        wave_label = f"{label}/wave_{wave.wave_idx}_experts_{wave.expert_start}_{wave.expert_end}"
        with torch.cuda.stream(state.wave_combine_stream):
            state.wave_combine_stream.wait_event(compute_done)
            combined_x, combine_event = _deepep_v2_combine(
                state,
                weighted_expert_out=weighted_expert_out,
                handle=handle,
                label=f"{wave_label}/combine",
                async_with_compute_stream=async_mode,
            )

        wave_results.append(
            DeepEpV2ForwardResult(
                recv_x=recv_x_for_experts,
                expanded_topk_weights=expanded_weights,
                expert_out=expert_out,
                combined_x=combined_x,
                handle=handle,
            )
        )
        wave_outputs.append(combined_x)
        combine_events.append(combine_event)

    for event in combine_events:
        _deep_ep_wait(event, async_with_compute_stream=async_mode)

    return DeepEpV2WaveForwardResult(
        wave_results=wave_results,
        combined_x=_sum_deepep_v2_wave_outputs(wave_outputs, label=f"{label}/sum_waves"),
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
) -> DeepEpV2WaveForwardResult:
    if overlap:
        return _deepep_v2_wave_forward_overlapped(
            state,
            wave_inputs=wave_inputs,
            wave_layout=wave_layout,
            label=label,
            track_expert_grad=track_expert_grad,
            do_cpu_sync=do_cpu_sync,
        )
    return _deepep_v2_wave_forward_sequential(
        state,
        wave_inputs=wave_inputs,
        wave_layout=wave_layout,
        label=label,
        track_expert_grad=track_expert_grad,
        do_cpu_sync=do_cpu_sync,
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


def _prepare_deepep_v2_backward(
    state: DeepEpV2State,
    *,
    label: str,
) -> DeepEpV2ForwardResult:
    result = _deepep_v2_forward(
        state,
        label=f"{label}/forward_prep",
        track_expert_grad=True,
    )
    torch.cuda.nvtx.range_push(f"{label}/grad_prep")
    try:
        result.grad_combined_x = torch.ones_like(result.combined_x)
    finally:
        torch.cuda.nvtx.range_pop()
    return result


def _run_deepep_v2_backward_from_result(
    state: DeepEpV2State,
    result: DeepEpV2ForwardResult,
    *,
    label: str,
    zero_expert_grads: bool = True,
) -> None:
    if result.grad_combined_x is None:
        result.grad_combined_x = torch.ones_like(result.combined_x)

    torch.cuda.nvtx.range_push(f"{label}/combine_backward_dispatch")
    try:
        grad_weighted_expert_out, _grad_topk_idx, _grad_topk_weights, _handle, event = state.buffer.dispatch(
            result.grad_combined_x,
            handle=result.handle,
            num_sms=state.num_sms,
            num_qps=state.num_qps,
            async_with_compute_stream=state.async_with_compute_stream,
        )
        _deep_ep_wait(event, async_with_compute_stream=state.async_with_compute_stream)
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/unweight")
    try:
        grad_expert_out = grad_weighted_expert_out * result.expanded_topk_weights
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/experts_backward")
    try:
        torch.autograd.backward(result.expert_out, grad_expert_out)
    finally:
        torch.cuda.nvtx.range_pop()

    if result.recv_x.grad is None:
        raise RuntimeError("deepep_v2 expert backward did not produce grad for recv_x")

    torch.cuda.nvtx.range_push(f"{label}/dispatch_backward_combine")
    try:
        _combined_grad_x, _combined_grad_topk_weights, event = state.buffer.combine(
            result.recv_x.grad,
            handle=result.handle,
            num_sms=state.num_sms,
            num_qps=state.num_qps,
            async_with_compute_stream=state.async_with_compute_stream,
        )
        _deep_ep_wait(event, async_with_compute_stream=state.async_with_compute_stream)
    finally:
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}/zero_grad")
    try:
        result.recv_x.grad = None
        if zero_expert_grads:
            state.routed_experts.zero_grad(set_to_none=True)
    finally:
        torch.cuda.nvtx.range_pop()


def _prepare_deepep_v2_wave_backward(
    state: DeepEpV2State,
    *,
    wave_inputs: list[DeepEpV2WaveInput],
    wave_layout: str,
    label: str,
    overlap: bool,
    do_cpu_sync: bool,
) -> DeepEpV2WaveForwardResult:
    result = _deepep_v2_wave_forward(
        state,
        wave_inputs=wave_inputs,
        wave_layout=wave_layout,
        label=f"{label}/forward_prep",
        track_expert_grad=True,
        overlap=overlap,
        do_cpu_sync=do_cpu_sync,
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
) -> None:
    if result.grad_combined_x is None:
        result.grad_combined_x = torch.ones_like(result.combined_x)

    # First version is waved but intentionally sequential. It preserves the
    # same reverse communication semantics as no-wave DeepEP: combine backward
    # is a dispatch with the forward handle, and dispatch backward is a combine
    # with the forward handle.
    for wave_idx, wave_result in enumerate(result.wave_results):
        wave_result.grad_combined_x = result.grad_combined_x
        _run_deepep_v2_backward_from_result(
            state,
            wave_result,
            label=f"{label}/wave_{wave_idx}",
            zero_expert_grads=False,
        )

    torch.cuda.nvtx.range_push(f"{label}/zero_grad")
    try:
        state.routed_experts.zero_grad(set_to_none=True)
    finally:
        torch.cuda.nvtx.range_pop()


def _run_one_deepep_v2_iter(
    state: DeepEpV2State,
    *,
    label: str,
    pass_type: str,
) -> None:
    if pass_type == "forward":
        with torch.no_grad():
            _deepep_v2_forward(
                state,
                label=label,
                track_expert_grad=False,
            )
        return

    result = _deepep_v2_forward(
        state,
        label=f"{label}/forward",
        track_expert_grad=True,
    )
    _run_deepep_v2_backward_from_result(
        state,
        result,
        label=f"{label}/backward",
    )


def _run_one_deepep_v2_wave_iter(
    state: DeepEpV2State,
    *,
    wave_inputs: list[DeepEpV2WaveInput],
    wave_layout: str,
    label: str,
    pass_type: str,
    overlap: bool,
    do_cpu_sync: bool,
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
    )
    _run_deepep_v2_wave_backward_from_result(
        state,
        result,
        label=f"{label}/backward",
    )


def _bench_deepep_v2_case(
    args: argparse.Namespace,
    *,
    tokens: int,
    rank: int,
    world_size: int,
    ep_mesh: DeviceMesh,
    use_wave: bool = False,
) -> None:
    mode_name = "deepep_v2_wave" if use_wave else "deepep_v2"
    if rank == 0:
        if use_wave:
            print(
                "[bench] mode=deepep_v2_wave runs standalone DeepEP V2 wave "
                "expanded dispatch + OLMo grouped expert MLP + DeepEP V2 "
                "combine. It is benchmark bring-up, not model-path wiring yet.",
                flush=True,
            )
        else:
            print(
                "[bench] mode=deepep_v2 runs standalone DeepEP V2 expanded dispatch "
                "+ OLMo grouped expert MLP + DeepEP V2 combine. It is benchmark "
                "bring-up, not model-path wiring yet.",
                flush=True,
            )

    torch.manual_seed(20260625 + rank)
    torch.cuda.reset_peak_memory_stats()
    config_dtype, input_dtype = _dtype_config(args.dtype)
    deepep_probe_iters = _deepep_pre_dispatch_expert_iters(args)

    if args.deepep_skip_import_buffer_for_pre_dispatch_probe:
        if deepep_probe_iters <= 0:
            raise RuntimeError(
                "--deepep-skip-import-buffer-for-pre-dispatch-probe requires "
                "--pre-dispatch-expert-iters or --deepep-pre-dispatch-expert-iters"
            )
        _validate_deepep_v2_args(args, world_size=world_size)
        torch.cuda.nvtx.range_push(f"BENCH/{mode_name}/tokens_{tokens}/build_probe_no_import_buffer")
        try:
            torch.manual_seed(20260625 + rank)
            source_input = (0.2 * torch.randn(tokens, args.d_model, device="cuda")).to(input_dtype)
            topk_idx = _make_balanced_topk_idx(
                tokens=tokens,
                top_k=args.top_k,
                num_experts=args.num_experts,
                world_size=world_size,
                rank=rank,
                dtype=torch.int64,
            )
            topk_weights = torch.full(
                (tokens, args.top_k),
                1.0 / float(args.top_k),
                device="cuda",
                dtype=torch.float32,
            )
            if args.deepep_probe_routed_experts_source == "standalone":
                routed_experts = _build_deepep_v2_probe_routed_experts(
                    args,
                    rank=rank,
                    world_size=world_size,
                    config_dtype=config_dtype,
                    reset_seed=False,
                )
            elif args.deepep_probe_routed_experts_source == "rowwise_apply_ep":
                routed_experts = _build_rowwise_apply_ep_probe_routed_experts(
                    args,
                    world_size=world_size,
                    ep_mesh=ep_mesh,
                    config_dtype=config_dtype,
                )
            else:
                raise ValueError(args.deepep_probe_routed_experts_source)
        finally:
            torch.cuda.nvtx.range_pop()

        if rank == 0:
            print(
                "[bench] deepep_v2 diagnostic: skipped deep_ep import and "
                "ElasticBuffer construction; running fake pre-dispatch expert "
                "probe only. "
                f"routed_experts_source={args.deepep_probe_routed_experts_source} "
                f"weight_init={args.deepep_probe_weight_init}",
                flush=True,
            )

        if args.profile:
            dist.barrier()
            _cuda_profiler_start()
        _run_pre_dispatch_expert_probe(
            routed_experts,
            mode_name=f"{mode_name}_no_import_buffer",
            num_iters=deepep_probe_iters,
            tokens=tokens,
            top_k=args.top_k,
            d_model=args.d_model,
            input_dtype=input_dtype,
            pass_type=args.pass_type,
            rank=rank,
            world_size=world_size,
        )
        if args.profile:
            _cuda_profiler_stop()
            dist.barrier()

        del routed_experts, source_input, topk_idx, topk_weights
        torch.cuda.synchronize()
        return

    torch.cuda.nvtx.range_push(f"BENCH/{mode_name}/tokens_{tokens}/build")
    try:
        state = _build_deepep_v2_state(
            args,
            tokens=tokens,
            rank=rank,
            world_size=world_size,
            config_dtype=config_dtype,
            input_dtype=input_dtype,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    try:
        if rank == 0:
            wave_do_cpu_sync = (
                bool(args.deepep_wave_do_cpu_sync)
                if args.deepep_wave_do_cpu_sync is not None
                else state.do_cpu_sync
            )
            print(
                f"[bench] {mode_name} config: num_sms={state.num_sms} "
                f"num_qps={state.num_qps} allocated_qps={state.buffer.num_allocated_qps} "
                f"local_experts={state.num_local_experts} "
                f"num_max_tokens_per_rank={state.num_max_tokens_per_rank} "
                f"deepep_max_tokens_factor={args.deepep_max_tokens_factor} "
                f"expert_buffer_mode={state.expert_buffer_mode} "
                f"expert_alignment={state.expert_alignment} "
                f"async={state.async_with_compute_stream} "
                f"do_cpu_sync={state.do_cpu_sync} "
                f"wave_num_waves={args.deepep_wave_num_waves if use_wave else 0} "
                f"wave_overlap={bool(args.deepep_wave_overlap) if use_wave else False} "
                f"wave_layout={args.deepep_wave_layout if use_wave else 'none'} "
                f"wave_do_cpu_sync={wave_do_cpu_sync if use_wave else False}",
                flush=True,
            )
            if use_wave and args.deepep_wave_overlap and wave_do_cpu_sync:
                print(
                    "[bench] warning: deepep_v2_wave overlap is enabled while "
                    "wave dispatch CPU sync is enabled; launch-side CPU waits "
                    "can limit communication/compute pipelining. Use "
                    "--no-deepep-wave-do-cpu-sync to test the no-CPU-sync path.",
                    flush=True,
                )

        wave_inputs: list[DeepEpV2WaveInput] | None = None
        wave_overlap = bool(args.deepep_wave_overlap)
        wave_do_cpu_sync = (
            bool(args.deepep_wave_do_cpu_sync)
            if args.deepep_wave_do_cpu_sync is not None
            else state.do_cpu_sync
        )
        if use_wave:
            torch.cuda.nvtx.range_push(f"BENCH/{mode_name}/tokens_{tokens}/build_wave_inputs")
            try:
                wave_inputs = _build_deepep_v2_wave_inputs(
                    state,
                    num_waves=int(args.deepep_wave_num_waves),
                )
            finally:
                torch.cuda.nvtx.range_pop()
            if rank == 0:
                wave_ranges = [
                    f"{w.expert_start}:{w.expert_end}@rows{w.wave_base}:{w.wave_end}"
                    for w in wave_inputs
                ]
                print(
                    f"[bench] deepep_v2_wave local expert/row ranges={wave_ranges}",
                    flush=True,
                )
            if args.deepep_validate_wave_forward:
                _validate_deepep_v2_wave_forward(
                    state,
                    wave_inputs=wave_inputs,
                    wave_layout=str(args.deepep_wave_layout),
                    overlap=wave_overlap,
                    wave_do_cpu_sync=wave_do_cpu_sync,
                    atol=float(args.deepep_validate_wave_forward_atol),
                    rank=rank,
                )

        profile_started = False
        if args.profile and deepep_probe_iters > 0:
            dist.barrier()
            _cuda_profiler_start()
            profile_started = True

        _run_pre_dispatch_expert_probe(
            state.routed_experts,
            mode_name=mode_name,
            num_iters=deepep_probe_iters,
            tokens=tokens,
            top_k=args.top_k,
            d_model=args.d_model,
            input_dtype=input_dtype,
            pass_type=args.pass_type,
            rank=rank,
            world_size=world_size,
        )

        for idx in range(args.warmup):
            label = f"BENCH/{mode_name}/tokens_{tokens}/warmup_{idx}"
            torch.cuda.nvtx.range_push(f"{label}/total")
            try:
                warmup_pass_type = (
                    "forward_backward"
                    if args.pass_type == "backward"
                    else args.pass_type
                )
                if use_wave:
                    assert wave_inputs is not None
                    _run_one_deepep_v2_wave_iter(
                        state,
                        wave_inputs=wave_inputs,
                        wave_layout=str(args.deepep_wave_layout),
                        label=label,
                        pass_type=warmup_pass_type,
                        overlap=wave_overlap,
                        do_cpu_sync=wave_do_cpu_sync,
                    )
                else:
                    _run_one_deepep_v2_iter(
                        state,
                        label=label,
                        pass_type=warmup_pass_type,
                    )
            finally:
                torch.cuda.nvtx.range_pop()

        warmup_done = torch.cuda.Event(enable_timing=False)
        warmup_done.record()
        warmup_done.synchronize()

        if args.profile and not profile_started:
            dist.barrier()
            _cuda_profiler_start()

        host_sync_timing = os.getenv("OLMO_BENCH_HOST_SYNC_TIMING", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        host_times: list[float] = []
        for idx in range(args.iters):
            label = f"BENCH/{mode_name}/tokens_{tokens}/iter_{idx}"
            backward_result = None
            if args.pass_type == "backward":
                torch.cuda.nvtx.range_push(f"{label}/prep")
                try:
                    if use_wave:
                        assert wave_inputs is not None
                        backward_result = _prepare_deepep_v2_wave_backward(
                            state,
                            wave_inputs=wave_inputs,
                            wave_layout=str(args.deepep_wave_layout),
                            label=label,
                            overlap=wave_overlap,
                            do_cpu_sync=wave_do_cpu_sync,
                        )
                    else:
                        backward_result = _prepare_deepep_v2_backward(
                            state,
                            label=label,
                        )
                finally:
                    torch.cuda.nvtx.range_pop()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            if host_sync_timing:
                torch.cuda.synchronize()
            host_start = time.perf_counter() if host_sync_timing else 0.0
            start.record()
            torch.cuda.nvtx.range_push(f"{label}/total")
            try:
                if args.pass_type == "backward":
                    assert backward_result is not None
                    if use_wave:
                        assert isinstance(backward_result, DeepEpV2WaveForwardResult)
                        _run_deepep_v2_wave_backward_from_result(
                            state,
                            backward_result,
                            label=label,
                        )
                    else:
                        assert isinstance(backward_result, DeepEpV2ForwardResult)
                        _run_deepep_v2_backward_from_result(
                            state,
                            backward_result,
                            label=label,
                        )
                else:
                    if use_wave:
                        assert wave_inputs is not None
                        _run_one_deepep_v2_wave_iter(
                            state,
                            wave_inputs=wave_inputs,
                            wave_layout=str(args.deepep_wave_layout),
                            label=label,
                            pass_type=args.pass_type,
                            overlap=wave_overlap,
                            do_cpu_sync=wave_do_cpu_sync,
                        )
                    else:
                        _run_one_deepep_v2_iter(
                            state,
                            label=label,
                            pass_type=args.pass_type,
                        )
            finally:
                torch.cuda.nvtx.range_pop()
            end.record()
            events.append((start, end))
            if host_sync_timing:
                torch.cuda.synchronize()
                host_times.append((time.perf_counter() - host_start) * 1000.0)

        if args.profile:
            _cuda_profiler_stop()
            dist.barrier()

        if not events:
            return
        events[-1][1].synchronize()
        times = [start.elapsed_time(end) for start, end in events]
        local_ms = statistics.median(times)
        local_host_ms = statistics.median(host_times) if host_times else float("nan")
        local_mem_gib = torch.cuda.max_memory_allocated() / 1024**3
        local = torch.tensor([local_ms, local_host_ms, local_mem_gib], device="cuda")
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)

        if rank == 0:
            max_ms = _median_rank_ms(gathered)
            max_host_ms = max(float(v[1].item()) for v in gathered)
            max_mem_gib = max(float(v[2].item()) for v in gathered)
            host_timing_part = (
                f"host_ms/iter(max_rank)={max_host_ms:.3f} "
                if math.isfinite(max_host_ms)
                else ""
            )
            throughput_ms = max_host_ms if math.isfinite(max_host_ms) else max_ms
            print(
                "BENCH "
                f"{mode_name}: ranks={world_size} tokens/rank={tokens} "
                f"pass={args.pass_type} standalone=True moe_only=True shared=False "
                f"dtype={args.dtype} "
                f"compile={'experts' if args.compile and not args.no_compile else 'none'} "
                f"d={args.d_model} hidden={args.hidden_size} experts={args.num_experts} "
                f"local_experts={state.num_local_experts} top_k={args.top_k} "
                "balanced_routing=deepep "
                f"num_sms={state.num_sms} num_qps={state.num_qps} "
                f"num_max_tokens_per_rank={state.num_max_tokens_per_rank} "
                f"deepep_max_tokens_factor={args.deepep_max_tokens_factor} "
                f"expert_buffer_mode={state.expert_buffer_mode} "
                f"expert_alignment={state.expert_alignment} "
                f"async={state.async_with_compute_stream} "
                f"do_cpu_sync={state.do_cpu_sync} "
                f"wave_num_waves={args.deepep_wave_num_waves if use_wave else 0} "
                f"wave_overlap={wave_overlap if use_wave else False} "
                f"wave_layout={args.deepep_wave_layout if use_wave else 'none'} "
                f"wave_do_cpu_sync={wave_do_cpu_sync if use_wave else False} "
                f"ms/iter(max_rank)={max_ms:.3f} "
                f"{host_timing_part}"
                f"local_tokens/s={tokens / (throughput_ms / 1000.0):.1f} "
                f"global_tokens/s={tokens * world_size / (throughput_ms / 1000.0):.1f} "
                f"max_mem_GiB={max_mem_gib:.2f}",
                flush=True,
            )
    finally:
        if hasattr(state.buffer, "destroy"):
            torch.cuda.synchronize()
            dist.barrier()
            state.buffer.destroy()
            dist.barrier()


@dataclass
class StandardEpMegaKernelState:
    source_input: torch.Tensor
    route_expert_indices: torch.Tensor
    probs: torch.Tensor
    up_gate_weight: torch.Tensor
    down_weight: torch.Tensor
    gathered_out: torch.Tensor
    out: torch.Tensor
    workspace_config: dict[str, int]
    workspace: torch.Tensor
    rank_workspace_bases: torch.Tensor
    global_counts: torch.Tensor
    global_offsets: torch.Tensor
    expert_cursors: torch.Tensor
    packed_route: torch.Tensor
    route_to_slot: torch.Tensor
    packed_input: torch.Tensor
    h: torch.Tensor
    packed_expert_out: torch.Tensor
    barrier_state: torch.Tensor
    w1_up: torch.Tensor | None = None
    w1_gate: torch.Tensor | None = None


def _build_standard_ep_mega_kernel_state(
    args: argparse.Namespace,
    *,
    tokens: int,
    rank: int,
    world_size: int,
    peer_group: bool,
    collective: bool,
    umma: bool = False,
) -> StandardEpMegaKernelState:
    if args.dtype != "bf16":
        raise RuntimeError("standard_ep_mega currently supports --dtype bf16 only")
    if args.pass_type != "forward":
        raise RuntimeError("standard_ep_mega currently supports --pass-type forward only")
    if args.num_experts != 32 or args.top_k != 4:
        raise RuntimeError("standard_ep_mega requires --num-experts 32 --top-k 4")
    if tokens > 16384:
        raise RuntimeError("standard_ep_mega supports at most --tokens 16384")
    if (peer_group or collective) and world_size != 4:
        raise RuntimeError(
            "standard_ep_mega_peer_group/collective requires torchrun --nproc-per-node=4"
        )

    from olmo_core.kernels.wave_mega_ep import (
        rowwise_bf16_mega_moe_standard_ep_workspace_config,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(20260623 + rank)
    source_input = (0.2 * torch.randn(tokens, args.d_model, device=device)).to(torch.bfloat16)
    token_idx = torch.arange(tokens, device=device, dtype=torch.long).view(tokens, 1)
    topk_idx = torch.arange(args.top_k, device=device, dtype=torch.long).view(1, args.top_k)
    if peer_group or collective:
        route_expert_indices = (
            topk_idx * 8 + (token_idx + rank + topk_idx) % 8
        ).to(dtype=torch.long)
    else:
        route_expert_indices = (token_idx * args.top_k + topk_idx + rank) % args.num_experts
    probs = torch.full(
        (tokens, args.top_k),
        1.0 / float(args.top_k),
        device=device,
        dtype=torch.float32,
    )
    torch.manual_seed(20260623)
    up_gate_weight = (
        0.15
        * torch.randn(
            args.num_experts,
            2 * args.hidden_size,
            args.d_model,
            device=device,
        )
    ).to(torch.bfloat16)
    down_weight = (
        0.15
        * torch.randn(
            args.num_experts,
            args.hidden_size,
            args.d_model,
            device=device,
        )
    ).to(torch.bfloat16)
    gathered_out = torch.empty(
        (tokens, args.top_k, args.d_model),
        device=device,
        dtype=torch.bfloat16,
    )
    out = torch.empty((tokens, args.d_model), device=device, dtype=torch.bfloat16)
    workspace_config = rowwise_bf16_mega_moe_standard_ep_workspace_config(
        num_tokens=tokens,
        hidden=args.d_model,
        intermediate=args.hidden_size,
    )
    if workspace_config["top_k"] != args.top_k:
        raise RuntimeError(
            f"standard_ep_mega workspace top_k={workspace_config['top_k']} "
            f"does not match --top-k {args.top_k}"
        )
    if workspace_config["num_total_experts"] != args.num_experts:
        raise RuntimeError(
            "standard_ep_mega workspace num_total_experts="
            f"{workspace_config['num_total_experts']} does not match "
            f"--num-experts {args.num_experts}"
        )
    num_route_slots = workspace_config["num_route_slots"]
    if num_route_slots != tokens * args.top_k:
        raise RuntimeError(
            f"standard_ep_mega workspace num_route_slots={num_route_slots} "
            f"does not match tokens*top_k={tokens * args.top_k}"
        )
    if workspace_config["packed_values"] % num_route_slots != 0:
        raise RuntimeError("standard_ep_mega workspace packed_values is not route-aligned")
    if workspace_config["h_values"] % num_route_slots != 0:
        raise RuntimeError("standard_ep_mega workspace h_values is not route-aligned")
    packed_hidden = workspace_config["packed_values"] // num_route_slots
    h_hidden = workspace_config["h_values"] // num_route_slots
    if packed_hidden != args.d_model or h_hidden != args.hidden_size:
        raise RuntimeError(
            "standard_ep_mega workspace tensor widths do not match "
            f"d_model/hidden_size ({packed_hidden}, {h_hidden}) vs "
            f"({args.d_model}, {args.hidden_size})"
        )
    if peer_group or collective:
        from olmo_core.kernels import olmo_symm_mem

        workspace = olmo_symm_mem.empty(
            (workspace_config["workspace_stride_bytes"],),
            dtype=torch.uint8,
            device=device,
            group=dist.group.WORLD,
        )
        rank_workspace_bases = olmo_symm_mem.peer_base_ptrs(
            workspace,
            group=dist.group.WORLD,
        )
        packed_capacity = workspace_config["local_packed_capacity"]
    else:
        workspace = torch.empty(
            (workspace_config["workspace_bytes"],),
            device=device,
            dtype=torch.uint8,
        )
        rank_workspace_bases = torch.empty(
            (workspace_config["num_ranks"],),
            device=device,
            dtype=torch.long,
        )
        packed_capacity = num_route_slots
        if umma:
            packed_capacity += workspace_config["num_total_experts"] * (128 - 1)
    global_counts = torch.empty(
        (workspace_config["num_total_experts"],),
        device=device,
        dtype=torch.long,
    )
    global_offsets = torch.empty(
        (workspace_config["num_total_experts"] + 1,),
        device=device,
        dtype=torch.long,
    )
    expert_cursors = torch.empty_like(global_counts)
    packed_route = torch.empty((packed_capacity,), device=device, dtype=torch.long)
    route_to_slot = torch.empty((num_route_slots,), device=device, dtype=torch.long)
    packed_input = torch.empty(
        (packed_capacity, packed_hidden),
        device=device,
        dtype=torch.bfloat16,
    )
    h = torch.empty((packed_capacity, h_hidden), device=device, dtype=torch.bfloat16)
    packed_expert_out = torch.empty_like(packed_input)
    w1_up = torch.empty_like(h) if umma else None
    w1_gate = torch.empty_like(h) if umma else None
    barrier_state = torch.empty(
        (workspace_config["barrier_state_len"],),
        device=device,
        dtype=torch.int32,
    )
    return StandardEpMegaKernelState(
        source_input=source_input,
        route_expert_indices=route_expert_indices.contiguous(),
        probs=probs.contiguous(),
        up_gate_weight=up_gate_weight.contiguous(),
        down_weight=down_weight.contiguous(),
        gathered_out=gathered_out,
        out=out,
        workspace_config=workspace_config,
        workspace=workspace,
        rank_workspace_bases=rank_workspace_bases,
        global_counts=global_counts,
        global_offsets=global_offsets,
        expert_cursors=expert_cursors,
        packed_route=packed_route,
        route_to_slot=route_to_slot,
        packed_input=packed_input,
        h=h,
        packed_expert_out=packed_expert_out,
        barrier_state=barrier_state,
        w1_up=w1_up,
        w1_gate=w1_gate,
    )


def _run_one_standard_ep_mega_kernel_iter(
    state: StandardEpMegaKernelState,
    *,
    label: str,
    rank: int,
    peer_group: bool,
    collective: bool,
    umma: bool = False,
) -> None:
    from olmo_core.kernels.wave_mega_ep import (
        rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace,
        rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_collective_world_umma,
        rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_collective_world,
        rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group_umma,
        rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group,
        rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_umma,
    )

    if collective:
        nvtx_name = "standard_ep_forward_collective_world_umma" if umma else "standard_ep_forward_collective_world"
    elif peer_group:
        nvtx_name = "standard_ep_forward_peer_group_umma" if umma else "standard_ep_forward_peer_group"
    else:
        nvtx_name = "standard_ep_forward_persistent_workspace_umma" if umma else "standard_ep_forward_persistent_workspace"
    torch.cuda.nvtx.range_push(f"{label}/{nvtx_name}")
    try:
        if umma:
            if state.w1_up is None or state.w1_gate is None:
                raise RuntimeError("standard_ep UMMA state is missing W1 scratch tensors")
            if collective:
                rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_collective_world_umma(
                    state.source_input,
                    state.gathered_out,
                    state.out,
                    state.route_expert_indices,
                    state.probs,
                    state.up_gate_weight,
                    state.down_weight,
                    state.workspace,
                    state.rank_workspace_bases,
                    state.global_counts,
                    state.global_offsets,
                    state.expert_cursors,
                    state.packed_route,
                    state.route_to_slot,
                    state.packed_input,
                    state.h,
                    state.packed_expert_out,
                    state.barrier_state,
                    state.w1_up,
                    state.w1_gate,
                    caller_rank_idx=rank,
                )
            elif peer_group:
                rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group_umma(
                    state.source_input,
                    state.gathered_out,
                    state.out,
                    state.route_expert_indices,
                    state.probs,
                    state.up_gate_weight,
                    state.down_weight,
                    state.workspace,
                    state.rank_workspace_bases,
                    state.global_counts,
                    state.global_offsets,
                    state.expert_cursors,
                    state.packed_route,
                    state.route_to_slot,
                    state.packed_input,
                    state.h,
                    state.packed_expert_out,
                    state.barrier_state,
                    state.w1_up,
                    state.w1_gate,
                    caller_rank_idx=rank,
                )
            else:
                rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_umma(
                    state.source_input,
                    state.gathered_out,
                    state.out,
                    state.route_expert_indices,
                    state.probs,
                    state.up_gate_weight,
                    state.down_weight,
                    state.workspace,
                    state.rank_workspace_bases,
                    state.global_counts,
                    state.global_offsets,
                    state.expert_cursors,
                    state.packed_route,
                    state.route_to_slot,
                    state.packed_input,
                    state.h,
                    state.packed_expert_out,
                    state.barrier_state,
                    state.w1_up,
                    state.w1_gate,
                )
        elif collective:
            rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_collective_world(
                state.source_input,
                state.gathered_out,
                state.out,
                state.route_expert_indices,
                state.probs,
                state.up_gate_weight,
                state.down_weight,
                state.workspace,
                state.rank_workspace_bases,
                state.global_counts,
                state.global_offsets,
                state.expert_cursors,
                state.packed_route,
                state.route_to_slot,
                state.packed_input,
                state.h,
                state.packed_expert_out,
                state.barrier_state,
                caller_rank_idx=rank,
            )
        elif peer_group:
            rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group(
                state.source_input,
                state.gathered_out,
                state.out,
                state.route_expert_indices,
                state.probs,
                state.up_gate_weight,
                state.down_weight,
                state.workspace,
                state.rank_workspace_bases,
                state.global_counts,
                state.global_offsets,
                state.expert_cursors,
                state.packed_route,
                state.route_to_slot,
                state.packed_input,
                state.h,
                state.packed_expert_out,
                state.barrier_state,
                caller_rank_idx=rank,
            )
        else:
            rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace(
                state.source_input,
                state.gathered_out,
                state.out,
                state.route_expert_indices,
                state.probs,
                state.up_gate_weight,
                state.down_weight,
                state.workspace,
                state.rank_workspace_bases,
                state.global_counts,
                state.global_offsets,
                state.expert_cursors,
                state.packed_route,
                state.route_to_slot,
                state.packed_input,
                state.h,
                state.packed_expert_out,
                state.barrier_state,
            )
    finally:
        torch.cuda.nvtx.range_pop()

def _median_rank_ms(values: Iterable[torch.Tensor]) -> float:
    return max(float(v[0].item()) for v in values)


def _bench_standard_ep_mega_kernel_case(
    args: argparse.Namespace,
    *,
    tokens: int,
    rank: int,
    world_size: int,
    peer_group: bool = False,
    collective: bool = False,
    umma: bool = False,
) -> None:
    if collective:
        mode_name = "standard_ep_mega_collective_umma" if umma else "standard_ep_mega_collective"
    elif peer_group:
        mode_name = "standard_ep_mega_peer_group_umma" if umma else "standard_ep_mega_peer_group"
    else:
        mode_name = "standard_ep_mega_umma" if umma else "standard_ep_mega"
    if rank == 0:
        if collective:
            print(
                f"[bench] mode={mode_name} runs the standalone "
                "standard-shape EP4/world4 OLMo-owned BF16 rank-local "
                "NVSHMEM world collective-launch diagnostic path"
                f"{' with the 128x128x64 TMA/UMMA compute branch.' if umma else '.'}",
                flush=True,
            )
        elif peer_group:
            print(
                f"[bench] mode={mode_name} runs the standalone "
                "standard-shape EP4 OLMo-owned BF16 rank-local megakernel with "
                "real symmetric peer workspaces and normal per-rank launch"
                f"{' plus the 128x128x64 TMA/UMMA compute branch.' if umma else '.'}",
                flush=True,
            )
        else:
            print(
                f"[bench] mode={mode_name} runs the standalone standard-shape "
                "OLMo-owned BF16 fused MegaMoE megakernel. It is not model wiring "
                "or a distributed peer-window transport benchmark"
                f"{' and uses the 128x128x64 TMA/UMMA compute branch.' if umma else '.'}",
                flush=True,
            )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.nvtx.range_push(f"BENCH/{mode_name}/tokens_{tokens}/build")
    try:
        state = _build_standard_ep_mega_kernel_state(
            args,
            tokens=tokens,
            rank=rank,
            world_size=world_size,
            peer_group=peer_group,
            collective=collective,
            umma=umma,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    if umma and args.check_standard_ep_umma_parity:
        torch.cuda.nvtx.range_push(f"BENCH/{mode_name}/tokens_{tokens}/parity")
        try:
            baseline_state = _build_standard_ep_mega_kernel_state(
                args,
                tokens=tokens,
                rank=rank,
                world_size=world_size,
                peer_group=peer_group,
                collective=collective,
                umma=False,
            )
            _run_one_standard_ep_mega_kernel_iter(
                baseline_state,
                label=f"BENCH/{mode_name}/tokens_{tokens}/parity_wmma",
                rank=rank,
                peer_group=peer_group,
                collective=collective,
                umma=False,
            )
            _run_one_standard_ep_mega_kernel_iter(
                state,
                label=f"BENCH/{mode_name}/tokens_{tokens}/parity_umma",
                rank=rank,
                peer_group=peer_group,
                collective=collective,
                umma=True,
            )
            torch.cuda.synchronize()
            diff = (baseline_state.out.float() - state.out.float()).abs()
            local = torch.tensor(
                [float(diff.max().item()), float(diff.mean().item())],
                device="cuda",
                dtype=torch.float32,
            )
            global_diff = local.clone()
            dist.all_reduce(global_diff, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(
                    "CHECK "
                    f"{mode_name}: tokens/rank={tokens} "
                    f"max_abs={float(global_diff[0].item()):.6g} "
                    f"max_mean_abs={float(global_diff[1].item()):.6g} "
                    f"atol={args.standard_ep_umma_parity_atol:.6g}",
                    flush=True,
                )
            if float(global_diff[0].item()) > args.standard_ep_umma_parity_atol:
                raise RuntimeError(
                    f"{mode_name} parity check failed: "
                    f"max_abs={float(global_diff[0].item()):.6g} "
                    f"> atol={args.standard_ep_umma_parity_atol:.6g}"
                )
            dist.barrier()
        finally:
            torch.cuda.nvtx.range_pop()

    for idx in range(args.warmup):
        label = f"BENCH/{mode_name}/tokens_{tokens}/warmup_{idx}"
        torch.cuda.nvtx.range_push(f"{label}/total")
        try:
            _run_one_standard_ep_mega_kernel_iter(
                state,
                label=label,
                rank=rank,
                peer_group=peer_group,
                collective=collective,
                umma=umma,
            )
        finally:
            torch.cuda.nvtx.range_pop()

    warmup_done = torch.cuda.Event(enable_timing=False)
    warmup_done.record()
    warmup_done.synchronize()

    if args.profile:
        dist.barrier()
        _cuda_profiler_start()

    host_sync_timing = os.getenv("OLMO_BENCH_HOST_SYNC_TIMING", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    host_times: list[float] = []
    for idx in range(args.iters):
        label = f"BENCH/{mode_name}/tokens_{tokens}/iter_{idx}"
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if host_sync_timing:
            torch.cuda.synchronize()
        host_start = time.perf_counter() if host_sync_timing else 0.0
        start.record()
        torch.cuda.nvtx.range_push(f"{label}/total")
        try:
            _run_one_standard_ep_mega_kernel_iter(
                state,
                label=label,
                rank=rank,
                peer_group=peer_group,
                collective=collective,
                umma=umma,
            )
        finally:
            torch.cuda.nvtx.range_pop()
        end.record()
        events.append((start, end))
        if (peer_group or collective) and not args.profile:
            # These modes enqueue NVSHMEM stream barriers before each launch.
            # Keep measured iterations host-ordered so the benchmark does not
            # stack multiple distributed barrier epochs on the stream.
            end.synchronize()

    if args.profile:
        _cuda_profiler_stop()
        dist.barrier()

    if not events:
        return
    events[-1][1].synchronize()
    times = [start.elapsed_time(end) for start, end in events]
    local_ms = statistics.median(times)
    local_mem_gib = torch.cuda.max_memory_allocated() / 1024**3
    local = torch.tensor([local_ms, local_mem_gib], device="cuda")
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)

    if rank == 0:
        max_ms = _median_rank_ms(gathered)
        max_mem_gib = max(float(v[1].item()) for v in gathered)
        print(
            "BENCH "
            f"{mode_name}: ranks={world_size} tokens/rank={tokens} "
            f"pass=forward kernel_only=True peer_group={peer_group} "
            f"collective={collective} dtype=bf16 "
            f"d={args.d_model} hidden={args.hidden_size} experts={args.num_experts} "
            f"top_k={args.top_k} "
            f"ms/iter(max_rank)={max_ms:.3f} "
            f"local_tokens/s={tokens / (max_ms / 1000.0):.1f} "
            f"max_mem_GiB={max_mem_gib:.2f}",
            flush=True,
        )


def _bench_case(
    args: argparse.Namespace,
    case: BenchCase,
    *,
    tokens: int,
    rank: int,
    world_size: int,
    ep_mesh: DeviceMesh,
) -> None:
    if case.deepep_v2 or case.deepep_v2_wave:
        _bench_deepep_v2_case(
            args,
            tokens=tokens,
            rank=rank,
            world_size=world_size,
            ep_mesh=ep_mesh,
            use_wave=case.deepep_v2_wave,
        )
        return
    if case.kernel_standard_ep_mega:
        _bench_standard_ep_mega_kernel_case(
            args,
            tokens=tokens,
            rank=rank,
            world_size=world_size,
            umma=case.kernel_standard_ep_mega_umma,
        )
        return
    if case.kernel_standard_ep_mega_peer_group:
        _bench_standard_ep_mega_kernel_case(
            args,
            tokens=tokens,
            rank=rank,
            world_size=world_size,
            peer_group=True,
            umma=case.kernel_standard_ep_mega_umma,
        )
        return
    if case.kernel_standard_ep_mega_collective:
        _bench_standard_ep_mega_kernel_case(
            args,
            tokens=tokens,
            rank=rank,
            world_size=world_size,
            collective=True,
            umma=case.kernel_standard_ep_mega_umma,
        )
        return

    capacity_factor = args.capacity_factor
    if rank == 0 and case.use_wave:
        print(
            "[bench] mode=wave/bf16_persistent_mega selects the production "
            "OLMo-owned fused BF16 MegaMoE target through the model-facing "
            "forward-only bring-up path.",
            flush=True,
        )
    if rank == 0 and case.rowwise_wave:
        print(
            "[bench] mode=rowwise_wave selects the experimental expert-major "
            "rowwise backend, not the MegaMoE megakernel path.",
            flush=True,
        )
    torch.manual_seed(20260619 + rank)
    torch.cuda.reset_peak_memory_stats()
    config_dtype, input_dtype = _dtype_config(args.dtype)

    torch.cuda.nvtx.range_push(f"BENCH/{case.name}/tokens_{tokens}/build")
    try:
        block = _build_block(
            d_model=args.d_model,
            hidden_size=args.hidden_size,
            num_experts=args.num_experts,
            top_k=args.top_k,
            capacity_factor=capacity_factor,
            rowwise_nblocks=args.rowwise_nblocks,
            use_wave=case.use_wave,
            use_bf16_persistent_mega=case.use_bf16_persistent_mega,
            rowwise_wave=case.rowwise_wave,
            rowwise_wave_num_waves=args.rowwise_wave_num_waves,
            rowwise_wave_recompute_linear1=args.rowwise_wave_recompute_linear1,
            rowwise_wave_recompute_act=args.rowwise_wave_recompute_act,
            include_shared_expert=not args.no_shared_expert,
            shared_hidden_size=args.shared_hidden_size,
            uniform_routing=not args.random_routing,
            random_routing=args.random_routing,
            config_dtype=config_dtype,
        )
        if not args.full_block:
            _patch_moe_only(block)
        block.apply_ep(ep_mesh)
        if block.routed_experts is not None:
            _init_probe_routed_expert_weights(
                block.routed_experts,
                weight_init=_resolve_weight_init_value(
                    str(args.model_local_expert_weight_init),
                    source_default="empty",
                ),
            )
        if args.balanced_routing == "deepep":
            if args.random_routing:
                raise RuntimeError("--balanced-routing deepep conflicts with --random-routing")
            _install_deepep_balanced_router(block, world_size=world_size)
        elif case.use_wave and not args.random_routing:
            _install_ep_balanced_router(block)
        block.train()
        compile_enabled = bool(args.compile and not args.no_compile)
        if compile_enabled:
            if args.compile_block:
                block = torch.compile(block, fullgraph=False, dynamic=False)
            else:
                _compile_hot_modules(block)
    finally:
        torch.cuda.nvtx.range_pop()

    static_input = None
    if args.profile and args.pass_type == "forward":
        static_input = torch.randn(
            1,
            tokens,
            args.d_model,
            device="cuda",
            dtype=input_dtype,
        )

    profile_started = False
    if args.profile and args.pre_dispatch_expert_iters > 0:
        dist.barrier()
        _cuda_profiler_start()
        profile_started = True

    if args.pre_dispatch_expert_iters > 0:
        if block.routed_experts is None:
            raise RuntimeError(f"{case.name} pre-dispatch probe requires routed experts")
        _run_pre_dispatch_expert_probe(
            block.routed_experts,
            mode_name=case.name,
            num_iters=int(args.pre_dispatch_expert_iters),
            tokens=tokens,
            top_k=args.top_k,
            d_model=args.d_model,
            input_dtype=input_dtype,
            pass_type=args.pass_type,
            rank=rank,
            world_size=world_size,
        )

    for idx in range(args.warmup):
        label = f"BENCH/{case.name}/tokens_{tokens}/warmup_{idx}"
        torch.cuda.nvtx.range_push(f"{label}/total")
        try:
            _run_one_iter(
                block,
                tokens=tokens,
                d_model=args.d_model,
                input_dtype=input_dtype,
                label=label,
                pass_type=(
                    "forward_backward"
                    if args.pass_type == "backward"
                    else args.pass_type
                ),
                static_input=static_input,
            )
        finally:
            torch.cuda.nvtx.range_pop()
    warmup_done = torch.cuda.Event(enable_timing=False)
    warmup_done.record()
    warmup_done.synchronize()

    if args.profile and not profile_started:
        dist.barrier()
        _cuda_profiler_start()

    host_sync_timing = os.getenv("OLMO_BENCH_HOST_SYNC_TIMING", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    host_times: list[float] = []
    for idx in range(args.iters):
        label = f"BENCH/{case.name}/tokens_{tokens}/iter_{idx}"
        backward_loss = None
        if args.pass_type == "backward":
            torch.cuda.nvtx.range_push(f"{label}/prep")
            try:
                backward_loss = _prepare_backward_loss(
                    block,
                    tokens=tokens,
                    d_model=args.d_model,
                    input_dtype=input_dtype,
                    label=label,
                )
            finally:
                torch.cuda.nvtx.range_pop()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if host_sync_timing:
            torch.cuda.synchronize()
        host_start = time.perf_counter() if host_sync_timing else 0.0
        start.record()
        torch.cuda.nvtx.range_push(f"{label}/total")
        try:
            if args.pass_type == "backward":
                assert backward_loss is not None
                _run_backward_from_loss(
                    block,
                    backward_loss,
                    label=label,
                )
            else:
                _run_one_iter(
                    block,
                    tokens=tokens,
                    d_model=args.d_model,
                    input_dtype=input_dtype,
                    label=label,
                    pass_type=args.pass_type,
                    static_input=static_input,
                )
        finally:
            torch.cuda.nvtx.range_pop()
        end.record()
        events.append((start, end))
        if host_sync_timing:
            torch.cuda.synchronize()
            host_times.append((time.perf_counter() - host_start) * 1000.0)
        if (case.use_wave or case.rowwise_wave) and not args.profile:
            # These experimental modes use rowwise/NVSHMEM stream barriers.
            # Keep benchmark iterations host-ordered. During nsys profiling,
            # leave iterations queued and synchronize after cudaProfilerStop()
            # so the profile does not contain inter-iteration host syncs.
            end.synchronize()

    if args.profile:
        _cuda_profiler_stop()
        dist.barrier()

    if events:
        events[-1][1].synchronize()
    times = [start.elapsed_time(end) for start, end in events]
    local_ms = statistics.median(times)
    local_host_ms = statistics.median(host_times) if host_times else float("nan")
    local_mem_gib = torch.cuda.max_memory_allocated() / 1024**3
    local = torch.tensor([local_ms, local_host_ms, local_mem_gib], device="cuda")
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)

    if rank == 0:
        max_ms = _median_rank_ms(gathered)
        max_host_ms = max(float(v[1].item()) for v in gathered)
        max_mem_gib = max(float(v[2].item()) for v in gathered)
        host_timing_part = (
            f"host_ms/iter(max_rank)={max_host_ms:.3f} "
            if math.isfinite(max_host_ms)
            else ""
        )
        throughput_ms = max_host_ms if math.isfinite(max_host_ms) else max_ms
        print(
            "BENCH "
            f"{case.name}: ranks={world_size} tokens/rank={tokens} "
            f"pass={args.pass_type} moe_only={not args.full_block} "
            f"shared={not args.no_shared_expert} dtype={args.dtype} "
            f"compile={'none' if not args.compile or args.no_compile else ('block' if args.compile_block else 'experts')} "
            f"d={args.d_model} hidden={args.hidden_size} experts={args.num_experts} "
            f"top_k={args.top_k} cap={capacity_factor} "
            f"balanced_routing={args.balanced_routing} "
            f"rowwise_wave_num_waves={args.rowwise_wave_num_waves if case.rowwise_wave else 0} "
            f"rowwise_wave_recompute_linear1={args.rowwise_wave_recompute_linear1 if case.rowwise_wave else False} "
            f"rowwise_wave_recompute_act={args.rowwise_wave_recompute_act if case.rowwise_wave else False} "
            f"ms/iter(max_rank)={max_ms:.3f} "
            f"{host_timing_part}"
            f"local_tokens/s={tokens / (throughput_ms / 1000.0):.1f} "
            f"global_tokens/s={tokens * world_size / (throughput_ms / 1000.0):.1f} "
            f"max_mem_GiB={max_mem_gib:.2f}",
            flush=True,
        )

    if os.getenv("OLMO_BENCH_OS_EXIT_AFTER_BENCH", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        torch.cuda.synchronize()
        exit_sleep_s = float(os.getenv("OLMO_BENCH_EXIT_SLEEP_S", "0"))
        if exit_sleep_s > 0:
            time.sleep(exit_sleep_s)
        os._exit(0)

    if os.getenv("OLMO_BENCH_HARD_EXIT_AFTER_BENCH", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        torch.cuda.synchronize()
        exit_sleep_s = float(os.getenv("OLMO_BENCH_EXIT_SLEEP_S", "0"))
        if exit_sleep_s > 0:
            time.sleep(exit_sleep_s)
        return


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    args = _parse_args()
    rank, _, world_size = _init_dist()
    cases = _parse_modes(args.modes)
    ep_mesh = _build_ep_mesh(world_size)

    try:
        for tokens in args.tokens:
            for case in cases:
                _bench_case(
                    args,
                    case,
                    tokens=tokens,
                    rank=rank,
                    world_size=world_size,
                    ep_mesh=ep_mesh,
                )
    finally:
        if (
            dist.is_initialized()
            and os.getenv("OLMO_BENCH_HARD_EXIT_AFTER_BENCH", "0").strip().lower()
            not in {"1", "true", "yes", "on"}
        ):
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
