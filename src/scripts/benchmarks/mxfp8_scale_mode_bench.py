from __future__ import annotations

import argparse
import contextlib
import inspect
import os
from collections.abc import Callable, Iterator
from typing import Any

import torch

from olmo_core.kernels import mxfp8_utils
from olmo_core.kernels.mxfp8_utils import (
    dequantize_rows_from_mxfp8,
    quantize_rows_to_mxfp8,
)

_TORCHAO_COMPILED_Q: dict[str, Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = {}


@contextlib.contextmanager
def _temporary_env(**updates: str) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _get_triton_mxfp_state() -> tuple[Any, Any, Any, str | None]:
    try:
        from triton_kernels.numerics_details.mxfp import (  # type: ignore[import-not-found]
            DequantScaleRoundingMode,
            downcast_to_mxfp,
            upcast_from_mxfp,
        )
    except Exception as e:
        return None, None, None, str(e)
    return downcast_to_mxfp, upcast_from_mxfp, DequantScaleRoundingMode, None


def _get_torchao_mxfp_state() -> tuple[Any, Any, Any, str | None]:
    try:
        from torchao.prototype.mx_formats.config import (  # type: ignore[import-not-found]
            ScaleCalculationMode,
        )
        from torchao.prototype.mx_formats.mx_tensor import (  # type: ignore[import-not-found]
            to_dtype,
            to_mx,
        )
    except Exception as e:
        return None, None, None, str(e)
    return to_mx, to_dtype, ScaleCalculationMode, None


def _make_input(rows: int, cols: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    if cols % 32 != 0:
        raise ValueError(f"cols must be divisible by 32, got {cols}")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, cols), device="cuda", dtype=torch.float32, generator=generator)
    x.mul_(0.5)

    blocks = x.view(rows, cols // 32, 32)
    exponents = torch.randint(
        low=-2,
        high=5,
        size=(rows, cols // 32),
        device="cuda",
        generator=generator,
    )
    signs = torch.where(
        torch.rand((rows, cols // 32), device="cuda", generator=generator) < 0.5,
        -1.0,
        1.0,
    )
    # Put one value just above an E4M3 saturation boundary in every block so
    # floor vs rceil differences are visible in precision metrics.
    boundary = 1.01 + 0.20 * torch.rand((rows, cols // 32), device="cuda", generator=generator)
    boundary = boundary * 448.0 * torch.pow(2.0, exponents.to(torch.float32))
    blocks[:, :, 0] = signs * boundary
    return x.to(dtype=dtype).contiguous()


def _bytes_for(qdata: torch.Tensor, scales: torch.Tensor) -> int:
    return qdata.numel() * qdata.element_size() + scales.numel() * scales.element_size()


def _precision_metrics(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scales: torch.Tensor,
) -> dict[str, float]:
    dq = dequantize_rows_from_mxfp8(
        qdata,
        scales,
        block_size=32,
        out_dtype=torch.float32,
    )
    x_fp32 = x.to(torch.float32)
    err = dq - x_fp32
    abs_err = err.abs()
    q_fp32 = qdata.to(torch.float32)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    return {
        "mse": float(err.square().mean().item()),
        "mae": float(abs_err.mean().item()),
        "max_abs": float(abs_err.max().item()),
        "sat_pct": float((q_fp32.abs() == fp8_max).float().mean().mul(100).item()),
        "scale_min": float(scales.to(torch.float32).min().item()),
        "scale_max": float(scales.to(torch.float32).max().item()),
        "storage_mib": _bytes_for(qdata, scales) / (1024 * 1024),
    }


def _timed_ms(fn: Callable[[], Any], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _olmo_quantize(
    x: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    with _temporary_env(OLMO_MXFP8_Q_BACKEND="olmo"):
        return quantize_rows_to_mxfp8(x, block_size=32, scale_mode=mode)  # type: ignore[arg-type]


def _te_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    return mxfp8_utils._quantize_to_mxfp8_te(  # type: ignore[attr-defined]
        x,
        block_size=32,
        scale_mode="rceil",
        out=None,
        scales_out=None,
    )


def _require_te_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    result = _te_quantize(x)
    if result is None:
        raise RuntimeError("TransformerEngine MXFP8 path became unavailable")
    return result


def _triton_mxfp_quantize(x: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    downcast_to_mxfp, _, DequantScaleRoundingMode, _ = _get_triton_mxfp_state()
    if downcast_to_mxfp is None:
        return None
    rounding_mode = (
        DequantScaleRoundingMode.ROUND_UP
        if mode == "rceil"
        else DequantScaleRoundingMode.ROUND_DOWN
    )
    kwargs = {"DEQUANT_SCALE_ROUNDING_MODE": rounding_mode}
    if "scale_dtype" in inspect.signature(downcast_to_mxfp).parameters:
        kwargs["scale_dtype"] = torch.uint8
    qdata, scales_u8 = downcast_to_mxfp(
        x,
        torch.float8_e4m3fn,
        axis=1,
        **kwargs,
    )
    return qdata, scales_u8.view(torch.float8_e8m0fnu)


def _require_triton_mxfp_quantize(x: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    result = _triton_mxfp_quantize(x, mode)
    if result is None:
        raise RuntimeError("Triton upstream MXFP path became unavailable")
    return result


def _torchao_quantize(x: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    to_mx, _, ScaleCalculationMode, _ = _get_torchao_mxfp_state()
    if to_mx is None:
        return None
    scale_mode = ScaleCalculationMode.RCEIL if mode == "rceil" else ScaleCalculationMode.FLOOR
    scales, qdata = to_mx(
        x,
        torch.float8_e4m3fn,
        32,
        scaling_mode=scale_mode,
    )
    return qdata, scales


def _require_torchao_quantize(x: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    result = _torchao_quantize(x, mode)
    if result is None:
        raise RuntimeError("torchao MXFP path became unavailable")
    return result


def _torchao_compiled_quantize(
    x: torch.Tensor, mode: str
) -> tuple[torch.Tensor, torch.Tensor] | None:
    to_mx, _, ScaleCalculationMode, _ = _get_torchao_mxfp_state()
    if to_mx is None:
        return None
    if mode not in _TORCHAO_COMPILED_Q:
        scale_mode = ScaleCalculationMode.RCEIL if mode == "rceil" else ScaleCalculationMode.FLOOR

        def quantize_fn(inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            scales, qdata = to_mx(
                inp,
                torch.float8_e4m3fn,
                32,
                scaling_mode=scale_mode,
            )
            return qdata, scales

        _TORCHAO_COMPILED_Q[mode] = torch.compile(quantize_fn, fullgraph=True)
    return _TORCHAO_COMPILED_Q[mode](x)


def _require_torchao_compiled_quantize(
    x: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = _torchao_compiled_quantize(x, mode)
    if result is None:
        raise RuntimeError("compiled torchao MXFP path became unavailable")
    return result


def _olmo_dequantize(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    with _temporary_env(OLMO_MXFP8_DQ_BACKEND="olmo"):
        return dequantize_rows_from_mxfp8(
            qdata,
            scales,
            block_size=32,
            out_dtype=out_dtype,
        )


def _te_dequantize(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor | None:
    with _temporary_env(OLMO_MXFP8_DQ_BACKEND="te"):
        out = dequantize_rows_from_mxfp8(
            qdata,
            scales,
            block_size=32,
            out_dtype=out_dtype,
        )
    return out


def _triton_mxfp_dequantize(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor | None:
    _, upcast_from_mxfp, _, _ = _get_triton_mxfp_state()
    if upcast_from_mxfp is None:
        return None
    return upcast_from_mxfp(qdata, scales.view(torch.uint8), out_dtype, axis=1)


def _torchao_dequantize(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor | None:
    _, to_dtype, _, _ = _get_torchao_mxfp_state()
    if to_dtype is None:
        return None
    return to_dtype(qdata, scales, torch.float8_e4m3fn, 32, out_dtype)


def _print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name:14s} "
        f"mse={metrics['mse']:.6e} "
        f"mae={metrics['mae']:.6e} "
        f"max_abs={metrics['max_abs']:.6e} "
        f"sat={metrics['sat_pct']:.4f}% "
        f"scale=[{metrics['scale_min']:.3e}, {metrics['scale_max']:.3e}] "
        f"storage={metrics['storage_mib']:.2f} MiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MXFP8 scale rounding modes. Runs precision/comparison by "
            "default; pass --speed to time OLMo floor, OLMo rceil, and TE."
        )
    )
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", "--repeat", dest="iters", type=int, default=50)
    parser.add_argument("--speed", action="store_true")
    parser.add_argument("--skip-te", action="store_true")
    parser.add_argument("--skip-triton-mxfp", action="store_true")
    parser.add_argument("--skip-torchao", action="store_true")
    parser.add_argument("--skip-torchao-compile", action="store_true")
    parser.add_argument("--dq-dtype", choices=("bf16", "fp16", "fp32"), default="fp32")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MXFP8 scale mode benchmark")

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    x = _make_input(args.rows, args.cols, dtype=dtype, seed=args.seed)

    cases: list[tuple[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]] = [
        ("olmo-floor", lambda: _olmo_quantize(x, "floor")),
        ("olmo-rceil", lambda: _olmo_quantize(x, "rceil")),
    ]
    if not args.skip_triton_mxfp:
        _, _, _, triton_mxfp_error = _get_triton_mxfp_state()
        if triton_mxfp_error is not None:
            print(f"triton-mxfp   skipped: {triton_mxfp_error}")
        else:
            cases.extend(
                [
                    ("triton-mxfp-floor", lambda: _require_triton_mxfp_quantize(x, "floor")),
                    ("triton-mxfp-rceil", lambda: _require_triton_mxfp_quantize(x, "rceil")),
                ]
            )
    if not args.skip_torchao:
        _, _, _, torchao_error = _get_torchao_mxfp_state()
        if torchao_error is not None:
            print(f"torchao       skipped: {torchao_error}")
        else:
            cases.extend(
                [
                    ("torchao-floor", lambda: _require_torchao_quantize(x, "floor")),
                    ("torchao-rceil", lambda: _require_torchao_quantize(x, "rceil")),
                ]
            )
            if not args.skip_torchao_compile:
                cases.append(
                    (
                        "torchao-rceil-compile",
                        lambda: _require_torchao_compiled_quantize(x, "rceil"),
                    )
                )
    if not args.skip_te:
        te_result = _te_quantize(x)
        if te_result is None:
            print(
                "te            skipped: TransformerEngine MXFP8 path unavailable for this shape/runtime"
            )
        else:
            cases.append(("te-rceil", lambda: _require_te_quantize(x)))

    print(f"rows={args.rows} cols={args.cols} dtype={dtype} speed={args.speed}")
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, fn in cases:
        qdata, scales = fn()
        torch.cuda.synchronize()
        outputs[name] = (qdata, scales)
        _print_metrics(name, _precision_metrics(x, qdata, scales))

    floor_q, floor_s = outputs["olmo-floor"]
    rceil_q, rceil_s = outputs["olmo-rceil"]
    print(
        "olmo rceil vs floor "
        f"q_byte_equal={torch.equal(rceil_q.view(torch.uint8), floor_q.view(torch.uint8))} "
        f"scale_byte_equal={torch.equal(rceil_s.view(torch.uint8), floor_s.view(torch.uint8))}"
    )

    if "te-rceil" in outputs:
        te_q, te_s = outputs["te-rceil"]
        print(
            "olmo rceil vs TE    "
            f"q_byte_equal={torch.equal(rceil_q.view(torch.uint8), te_q.view(torch.uint8))} "
            f"scale_byte_equal={torch.equal(rceil_s.view(torch.uint8), te_s.view(torch.uint8))}"
        )
    if "triton-mxfp-floor" in outputs:
        tri_q, tri_s = outputs["triton-mxfp-floor"]
        print(
            "olmo floor vs Triton "
            f"q_byte_equal={torch.equal(floor_q.view(torch.uint8), tri_q.view(torch.uint8))} "
            f"scale_byte_equal={torch.equal(floor_s.view(torch.uint8), tri_s.view(torch.uint8))}"
        )
    if "triton-mxfp-rceil" in outputs:
        tri_q, tri_s = outputs["triton-mxfp-rceil"]
        print(
            "olmo rceil vs Triton "
            f"q_byte_equal={torch.equal(rceil_q.view(torch.uint8), tri_q.view(torch.uint8))} "
            f"scale_byte_equal={torch.equal(rceil_s.view(torch.uint8), tri_s.view(torch.uint8))}"
        )
    if "torchao-floor" in outputs:
        ta_q, ta_s = outputs["torchao-floor"]
        print(
            "olmo floor vs torchao"
            f" q_byte_equal={torch.equal(floor_q.view(torch.uint8), ta_q.view(torch.uint8))} "
            f"scale_byte_equal={torch.equal(floor_s.view(torch.uint8), ta_s.view(torch.uint8))}"
        )
    if "torchao-rceil" in outputs:
        ta_q, ta_s = outputs["torchao-rceil"]
        print(
            "olmo rceil vs torchao"
            f" q_byte_equal={torch.equal(rceil_q.view(torch.uint8), ta_q.view(torch.uint8))} "
            f"scale_byte_equal={torch.equal(rceil_s.view(torch.uint8), ta_s.view(torch.uint8))}"
        )
    if "torchao-rceil-compile" in outputs:
        ta_q, ta_s = outputs["torchao-rceil-compile"]
        print(
            "olmo rceil vs torchao-compile"
            f" q_byte_equal={torch.equal(rceil_q.view(torch.uint8), ta_q.view(torch.uint8))} "
            f"scale_byte_equal={torch.equal(rceil_s.view(torch.uint8), ta_s.view(torch.uint8))}"
        )

    if not args.speed:
        return

    dq_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dq_dtype]

    print(f"quantize timing warmup={args.warmup} iters={args.iters}")
    for name, fn in cases:
        ms = _timed_ms(fn, warmup=args.warmup, iters=args.iters)
        print(f"{name:14s} {ms:8.4f} ms")

    dq_cases: list[tuple[str, Callable[[], torch.Tensor]]] = [
        (
            "olmo-dq",
            lambda: _olmo_dequantize(outputs["olmo-rceil"][0], outputs["olmo-rceil"][1], dq_dtype),
        )
    ]
    if "triton-mxfp-rceil" in outputs:
        dq_cases.append(
            (
                "triton-mxfp-dq",
                lambda: _triton_mxfp_dequantize(
                    outputs["triton-mxfp-rceil"][0],
                    outputs["triton-mxfp-rceil"][1],
                    dq_dtype,
                ),
            )
        )
    if "torchao-rceil" in outputs:
        dq_cases.append(
            (
                "torchao-dq",
                lambda: _torchao_dequantize(
                    outputs["torchao-rceil"][0],
                    outputs["torchao-rceil"][1],
                    dq_dtype,
                ),
            )
        )
    if "te-rceil" in outputs:
        dq_cases.append(
            (
                "te-dq",
                lambda: _te_dequantize(outputs["te-rceil"][0], outputs["te-rceil"][1], dq_dtype),
            )
        )

    print(f"dequantize timing dtype={dq_dtype} warmup={args.warmup} iters={args.iters}")
    for name, fn in dq_cases:
        ms = _timed_ms(fn, warmup=args.warmup, iters=args.iters)
        print(f"{name:14s} {ms:8.4f} ms")


if __name__ == "__main__":
    main()
