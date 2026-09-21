"""Microbenchmark the LM's FlexAttention forward and backward at Stage-2 geometry.

Why this exists: profiling Stage-2 at ``pack_max_crops=80`` put the LM's flex-attention
**backward** at 19.56% of GPU time -- the largest single kernel -- against a presumed
forward of 5.26%, a ratio of ~3.7x. Attention's backward/forward FLOP ratio is ~2.5x
(5 matmuls against 2), so that implies a ~1.5x efficiency gap. This isolates the kernel so
the gap can be confirmed and tuned without spending an 8-GPU training job, and it settles a
second question the profile could not: whether that 5.26% row really is the flex forward.

Runs single-GPU; no distributed setup, because this is a kernel question.

    python src/scripts/perf/flex_attention_bench.py run          # on a GPU box
    python src/scripts/perf/flex_attention_bench.py launch <name> <cluster>
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from olmo_core.config import Config, StrEnum
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention.backend import FlexAttentionBackend
from olmo_core.utils import generate_uuid, prepare_cli_environment

log = logging.getLogger(__name__)

# Stage-2 LM attention geometry: qwen3-4B-ish (d_model 2560, 32 q heads / 8 kv heads,
# head_dim 128) at the packed sequence length, with the shipped rank microbatch of 2 packs.
B, S, N_HEADS, N_KV_HEADS, HEAD_DIM = 2, 16384, 32, 8, 128

# Pack composition at crops=80: ~13 examples per 16k pack, each ~6 crops -> ~864 pooled
# image tokens plus ~396 text tokens. Image-heavy, which matters because the mask's
# bidirectional image term is what makes the block pattern denser than plain causal.
N_EXAMPLES, EX_LEN, IMAGE_TOKENS_PER_EX = 13, 1260, 864

WARMUP, TRIALS = 5, 20


class SubCmd(StrEnum):
    launch = "launch"
    run = "run"
    dry_run = "dry_run"


@dataclass
class Arm:
    name: str
    note: str
    compile_kwargs: Dict[str, Any] = field(default_factory=dict)
    kernel_options: Optional[Dict[str, int]] = None
    enable_gqa: bool = False


def build_arms() -> List[Arm]:
    """Arm A is the shipped configuration; every other arm changes exactly one thing."""
    arms = [
        Arm("A-baseline", "shipped: torch.compile(flex), no kernel_options, kv expanded"),
        Arm("B-static", "+ dynamic=False", compile_kwargs={"dynamic": False}),
        Arm(
            "C-autotune",
            '+ mode="max-autotune-no-cudagraphs"',
            compile_kwargs={"mode": "max-autotune-no-cudagraphs"},
        ),
        # E is grouped with the tuning arms because it is the one structural change:
        # `_repeat_kv` does `.expand(...).reshape(...)`, and reshape on an expanded tensor
        # cannot be a view, so k/v are materialised at 4x size and flex reads 32 kv heads
        # where 8 would do. `enable_gqa=True` broadcasts in-register instead.
        Arm("E-gqa", "enable_gqa=True, kv NOT expanded", enable_gqa=True),
    ]
    # D: explicit backward block-size sweep. BLOCK_M1/N1 drive the dkv kernel and
    # BLOCK_M2/N2 the dq kernel; the defaults are not tuned for this mask density.
    for m1, n1, m2, n2 in ((32, 128, 128, 32), (64, 128, 128, 64), (128, 64, 64, 128)):
        arms.append(
            Arm(
                f"D-blk-{m1}x{n1}-{m2}x{n2}",
                "explicit backward block sizes",
                kernel_options={
                    "BLOCK_M1": m1,
                    "BLOCK_N1": n1,
                    "BLOCK_M2": m2,
                    "BLOCK_N2": n2,
                },
            )
        )
    return arms


def build_mask_vectors(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """``is_image`` / ``example_id`` of shape (B, S), matching what MultimodalLM passes."""
    is_image = torch.zeros((B, S), dtype=torch.bool, device=device)
    # -1 is the pad sentinel the mask_mod keys off; the tail past N_EXAMPLES*EX_LEN keeps it.
    example_id = torch.full((B, S), -1, dtype=torch.int32, device=device)
    for i in range(N_EXAMPLES):
        lo = i * EX_LEN
        hi = min(lo + EX_LEN, S)
        example_id[:, lo:hi] = i
        is_image[:, lo : min(lo + IMAGE_TOKENS_PER_EX, hi)] = True
    return is_image, example_id


def make_qkv(device: torch.device, *, expand_kv: bool):
    """Return (q, k, v) in flex layout (B, H, S, D), leaves marked for autograd.

    ``expand_kv`` reproduces `_repeat_kv`'s materialisation so arm E can be compared
    against it honestly: the expansion cost belongs to the arms that do it.
    """
    kv_heads = N_HEADS if expand_kv else N_KV_HEADS
    q = torch.randn(B, N_HEADS, S, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, kv_heads, S, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, kv_heads, S, HEAD_DIM, device=device, dtype=torch.bfloat16)
    for t in (q, k, v):
        t.requires_grad_(True)
    return q, k, v


def _time(fn, n: int) -> float:
    """Median ms over n runs. Median because a single stall distorts a mean badly here."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(n):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def run_arm(arm: Arm, block_mask, device: torch.device) -> Optional[Dict[str, float]]:
    from torch.nn.attention.flex_attention import flex_attention

    torch._dynamo.reset()  # so each arm compiles its own kernels rather than reusing A's
    compiled = torch.compile(flex_attention, **arm.compile_kwargs)
    q, k, v = make_qkv(device, expand_kv=not arm.enable_gqa)
    scale = HEAD_DIM**-0.5

    def fwd():
        kwargs: Dict[str, Any] = {"block_mask": block_mask, "scale": scale}
        if arm.kernel_options is not None:
            kwargs["kernel_options"] = arm.kernel_options
        if arm.enable_gqa:
            kwargs["enable_gqa"] = True
        return compiled(q, k, v, **kwargs)

    def fwd_bwd():
        out = fwd()
        out.sum().backward()

    try:
        for _ in range(WARMUP):
            fwd_bwd()
            q.grad = k.grad = v.grad = None
    except Exception as e:  # a kernel_options combination the template rejects
        log.warning("arm %s failed to run: %s", arm.name, str(e)[:300])
        return None

    fwd_ms = _time(lambda: fwd(), TRIALS)

    def step():
        fwd_bwd()
        q.grad = k.grad = v.grad = None

    both_ms = _time(step, TRIALS)
    bwd_ms = both_ms - fwd_ms
    return {"fwd": fwd_ms, "bwd": bwd_ms, "total": both_ms, "ratio": bwd_ms / fwd_ms}


def do_run() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("needs a GPU")
    device = torch.device("cuda")
    log.info("device: %s", torch.cuda.get_device_name(0))
    log.info(
        "geometry: B=%d S=%d heads=%d/%d head_dim=%d, %d examples x %d tokens (%d image)",
        B, S, N_HEADS, N_KV_HEADS, HEAD_DIM, N_EXAMPLES, EX_LEN, IMAGE_TOKENS_PER_EX,
    )

    is_image, example_id = build_mask_vectors(device)
    # Reuse the production mask builder so the block pattern is the real one, not an
    # approximation -- density is exactly what the backward's cost tracks.
    block_mask = FlexAttentionBackend.build_block_mask_from_vectors(
        B=B, S=S, device=device, is_image=is_image, example_id=example_id
    )
    log.info("block mask:\n%s", block_mask)

    results: Dict[str, Dict[str, float]] = {}
    for arm in build_arms():
        log.info("--- %s (%s)", arm.name, arm.note)
        r = run_arm(arm, block_mask, device)
        if r is not None:
            results[arm.name] = r
            log.info(
                "    fwd %.2f ms | bwd %.2f ms | total %.2f ms | bwd/fwd %.2fx",
                r["fwd"], r["bwd"], r["total"], r["ratio"],
            )

    base = results.get("A-baseline")
    print("\n%-26s %9s %9s %9s %9s %10s" % ("arm", "fwd ms", "bwd ms", "total", "bwd/fwd", "vs A"))
    for name, r in results.items():
        delta = "" if base is None else "%+.1f%%" % (100 * (r["total"] / base["total"] - 1))
        print(
            "%-26s %9.2f %9.2f %9.2f %8.2fx %10s"
            % (name, r["fwd"], r["bwd"], r["total"], r["ratio"], delta)
        )
    if base is not None:
        best = min(results.items(), key=lambda kv: kv[1]["total"])
        print(
            "\nbest: %s at %.2f ms total (%+.1f%% vs baseline), bwd/fwd %.2fx vs %.2fx"
            % (
                best[0], best[1]["total"],
                100 * (best[1]["total"] / base["total"] - 1),
                best[1]["ratio"], base["ratio"],
            )
        )
        # Pre-registered gate from the plan: <10% improvement means the gap is inherent to
        # FlexAttention at this mask density, and no 8-GPU job should follow.
        gain = 1 - best[1]["total"] / base["total"]
        print("GATE: %s (%.1f%% vs the 10%% bar)" % ("PASS" if gain > 0.10 else "FAIL", 100 * gain))


@dataclass
class BenchmarkConfig(Config):
    launch: BeakerLaunchConfig


def build_config(script: str, run_name: str, cluster: str, overrides: List[str]):
    launch_config = BeakerLaunchConfig(
        name=f"{run_name}-{generate_uuid()[:8]}",
        budget="ai2/oe-other",
        cmd=[script, SubCmd.run, run_name, cluster, *overrides],
        task_name="flex-bench",
        workspace="ai2/molmofication",
        clusters=[cluster],
        beaker_image=OLMoCoreBeakerImage.stable,
        num_nodes=1,
        num_gpus=1,
        allow_dirty=False,
    )
    return BenchmarkConfig(launch=launch_config).merge(overrides)


if __name__ == "__main__":
    prepare_cli_environment()
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = SubCmd(sys.argv[1])
    if cmd == SubCmd.run:
        do_run()
    else:
        _, _, run_name, cluster, *overrides = sys.argv
        config = build_config(sys.argv[0], run_name, cluster, overrides)
        log.info(config)
        if cmd == SubCmd.launch:
            config.launch.launch(follow=True)
