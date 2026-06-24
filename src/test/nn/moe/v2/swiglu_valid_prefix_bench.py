import argparse
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from olmo_core.kernels.swiglu import swiglu_valid_prefix


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark valid-prefix SwiGLU against torch.compile."
    )
    parser.add_argument("--rows", type=int, default=81920)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--valid-rows", type=int, default=16384)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--configs",
        type=str,
        default="8x256x1024x4,8x512x1024x4,16x256x1024x4,16x512x1024x4,32x128x1024x4,32x256x1024x4,32x512x1024x4",
        help="Comma-separated Triton configs as BLOCK_MxBLOCK_NxROW_PROGRAMSxWARPS.",
    )
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def _torch_valid_prefix_swiglu_impl(
    x: torch.Tensor,
    num_elements: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    hidden = x.shape[1] // 2
    valid_x = x[:num_elements]
    y = valid_x[:, :hidden] * F.silu(valid_x[:, hidden:])
    out[:num_elements].copy_(y)
    return out


_torch_valid_prefix_swiglu_compiled = torch.compile(
    _torch_valid_prefix_swiglu_impl,
    fullgraph=True,
    dynamic=False,
)


@dataclass(frozen=True)
class TritonConfig:
    block_m: int
    block_n: int
    row_programs: int
    num_warps: int

    @property
    def name(self) -> str:
        return (
            f"triton_bm{self.block_m}_bn{self.block_n}"
            f"_rp{self.row_programs}_w{self.num_warps}"
        )


def _parse_configs(raw: str) -> list[TritonConfig]:
    out = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        values = [int(x) for x in part.split("x")]
        if len(values) != 4:
            raise ValueError(
                f"Invalid config {part!r}; expected BLOCK_MxBLOCK_NxROW_PROGRAMSxWARPS"
            )
        out.append(TritonConfig(*values))
    if not out:
        raise ValueError("At least one Triton config is required")
    return out


def _time_ms(fn: Callable[[], None], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / float(iters)


def _assert_close_kwargs(dtype: torch.dtype) -> dict[str, float]:
    if dtype == torch.float32:
        return {"atol": 1e-6, "rtol": 1e-6}
    return {"atol": 2e-2, "rtol": 2e-2}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args = _parse_args()
    if args.valid_rows < 0 or args.valid_rows > args.rows:
        raise ValueError("--valid-rows must be within [0, rows]")

    device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)
    torch.manual_seed(20260624)
    x = torch.randn((args.rows, args.hidden * 2), device=device, dtype=dtype)
    num_elements = torch.tensor(args.valid_rows, device=device, dtype=torch.long)
    out = torch.empty((args.rows, args.hidden), device=device, dtype=dtype)
    ref_out = torch.empty_like(out)

    # Compile and validate the torch baseline before timing.
    _torch_valid_prefix_swiglu_compiled(x, num_elements, ref_out)
    torch.cuda.synchronize()
    ref_prefix = ref_out[: args.valid_rows].detach().clone()

    print(
        "SWIGLU_BENCH "
        f"rows={args.rows} valid_rows={args.valid_rows} hidden={args.hidden} "
        f"dtype={args.dtype} warmup={args.warmup} iters={args.iters}",
        flush=True,
    )

    torch_ms = _time_ms(
        lambda: _torch_valid_prefix_swiglu_compiled(x, num_elements, out),
        warmup=args.warmup,
        iters=args.iters,
    )
    torch.testing.assert_close(
        out[: args.valid_rows],
        ref_prefix,
        **_assert_close_kwargs(dtype),
    )
    valid_gib = args.valid_rows * args.hidden * 3 * torch.empty((), dtype=dtype).element_size() / 1024**3
    print(
        f"SWIGLU_RESULT torch_compile ms={torch_ms:.4f} "
        f"valid_GiB_per_s={valid_gib / (torch_ms / 1000.0):.1f}",
        flush=True,
    )

    best_name = "torch_compile"
    best_ms = torch_ms
    for cfg in _parse_configs(args.configs):
        def run_cfg() -> None:
            swiglu_valid_prefix(
                x,
                num_elements,
                out=out,
                block_m=cfg.block_m,
                block_n=cfg.block_n,
                row_programs=cfg.row_programs,
                num_warps=cfg.num_warps,
            )

        run_cfg()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out[: args.valid_rows],
            ref_prefix,
            **_assert_close_kwargs(dtype),
            msg=f"Mismatch for {cfg.name}",
        )
        ms = _time_ms(run_cfg, warmup=args.warmup, iters=args.iters)
        print(
            f"SWIGLU_RESULT {cfg.name} ms={ms:.4f} "
            f"valid_GiB_per_s={valid_gib / (ms / 1000.0):.1f} "
            f"speedup_vs_torch={torch_ms / ms:.3f}",
            flush=True,
        )
        if ms < best_ms:
            best_ms = ms
            best_name = cfg.name

    print(
        f"SWIGLU_BEST name={best_name} ms={best_ms:.4f} "
        f"speedup_vs_torch={torch_ms / best_ms:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
