from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchCase:
    name: str
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
            cases.append(BenchCase("rowwise"))
        elif mode in {"deepep_v2", "deep_ep_v2", "deepep"}:
            cases.append(BenchCase("deepep_v2", deepep_v2=True))
        elif mode in {"deepep_v2_wave", "deep_ep_v2_wave", "deepep_wave"}:
            cases.append(BenchCase("deepep_v2_wave", deepep_v2_wave=True))
        elif mode in {
            "rowwise_wave",
            "rowwise_wave_expert",
            "rowwise_wave_expert_sequential",
        }:
            cases.append(BenchCase("rowwise_wave", rowwise_wave=True))
        else:
            raise ValueError(
                f"Unknown mode {mode!r}. Expected rowwise,rowwise_wave,deepep_v2,deepep_v2_wave"
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
            "dispatch/GEMM/combine benchmark."
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
        "--deepep-weighting",
        choices=("post", "post_triton", "swiglu"),
        default="post",
        help=(
            "How standalone DeepEP expanded expert rows are multiplied by router "
            "weights before combine. 'post' preserves the original PyTorch "
            "post-Linear2 multiply. 'post_triton' uses a tiled Triton rowwise "
            "scale kernel after Linear2. 'swiglu' pushes the row weight before "
            "Linear2 so torch.compile can fuse it into the SwiGLU pointwise "
            "region; backward then uses the weighted expert output directly."
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
        "--deepep-validate-wave-backward",
        action="store_true",
        help=(
            "For --modes deepep_v2_wave, run one no-wave backward and one wave "
            "backward before warmup, then compare source-side gradients and "
            "local routed expert parameter gradients."
        ),
    )
    parser.add_argument(
        "--deepep-validate-wave-backward-atol",
        type=float,
        default=5e-2,
        help="Absolute tolerance for --deepep-validate-wave-backward.",
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
        "--sync-between-iters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Synchronize the full CUDA device after each measured iteration. "
            "This is a profiling diagnostic for allocator lifetime effects; "
            "it intentionally changes the host/GPU schedule."
        ),
    )
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
    return parser.parse_args()
