from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import torch


_CUDA_EXTENSION = None
_CUDA_EXTENSION_ATTEMPTED = False
_CUDA_EXTENSION_ERROR: Optional[Exception] = None


def _load_cuda_extension():
    global _CUDA_EXTENSION
    global _CUDA_EXTENSION_ATTEMPTED
    global _CUDA_EXTENSION_ERROR
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION
    if _CUDA_EXTENSION_ATTEMPTED and _CUDA_EXTENSION is None:
        raise RuntimeError("CUDA chunk reorder extension is unavailable") from _CUDA_EXTENSION_ERROR

    _CUDA_EXTENSION_ATTEMPTED = True
    try:
        from torch.utils.cpp_extension import load

        this_dir = Path(__file__).resolve().parent
        cpp_src = this_dir / "cuda" / "moe_chunk_reorder.cpp"
        cu_src = this_dir / "cuda" / "moe_chunk_reorder_kernel.cu"
        _CUDA_EXTENSION = load(
            name="olmo_moe_chunk_reorder_ext",
            sources=[str(cpp_src), str(cu_src)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as e:
        _CUDA_EXTENSION_ERROR = e
        raise RuntimeError(f"Failed to build/load CUDA chunk reorder extension: {e}") from e

    return _CUDA_EXTENSION


def _check_perm_inputs(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int,
    out: Optional[torch.Tensor],
) -> None:
    if inp.ndim != 2:
        raise ValueError(f"permute expects rank-2 input, got shape={tuple(inp.shape)}")
    if not inp.is_cuda:
        raise ValueError("permute expects CUDA input")
    if routing_map.ndim != 2:
        raise ValueError(
            f"routing_map must be rank-2 [num_tokens, topK], got shape={tuple(routing_map.shape)}"
        )
    if routing_map.shape[0] != inp.shape[0]:
        raise ValueError(
            f"routing_map/input rows mismatch: routing_map={routing_map.shape[0]} input={inp.shape[0]}"
        )
    if routing_map.shape[1] != 1:
        raise ValueError(f"only topK=1 is supported, got routing_map shape={tuple(routing_map.shape)}")
    if routing_map.device != inp.device:
        raise ValueError(
            f"routing_map/input device mismatch: routing_map={routing_map.device} input={inp.device}"
        )
    if num_out_tokens < 0:
        raise ValueError(f"num_out_tokens must be non-negative, got {num_out_tokens}")
    if out is not None:
        expected_shape = (num_out_tokens, inp.shape[1])
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out shape mismatch: expected={expected_shape} got={tuple(out.shape)}"
            )
        if out.dtype != inp.dtype:
            raise ValueError(f"out dtype mismatch: out={out.dtype} inp={inp.dtype}")
        if out.device != inp.device:
            raise ValueError(f"out device mismatch: out={out.device} inp={inp.device}")


def _check_unperm_inputs(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    num_tokens: int,
    out: Optional[torch.Tensor],
) -> None:
    if inp.ndim != 2:
        raise ValueError(f"unpermute expects rank-2 input, got shape={tuple(inp.shape)}")
    if not inp.is_cuda:
        raise ValueError("unpermute expects CUDA input")
    if row_id_map.ndim != 1:
        raise ValueError(f"row_id_map must be rank-1, got shape={tuple(row_id_map.shape)}")
    if row_id_map.device != inp.device:
        raise ValueError(
            f"row_id_map/input device mismatch: row_id_map={row_id_map.device} input={inp.device}"
        )
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if out is not None:
        expected_shape = (num_tokens, inp.shape[1])
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out shape mismatch: expected={expected_shape} got={tuple(out.shape)}"
            )
        if out.dtype != inp.dtype:
            raise ValueError(f"out dtype mismatch: out={out.dtype} inp={inp.dtype}")
        if out.device != inp.device:
            raise ValueError(f"out device mismatch: out={out.device} inp={inp.device}")


def _chunk_permute_torch(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    *,
    num_out_tokens: int,
    out: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    routing_flat = routing_map.reshape(-1).to(dtype=torch.int64)
    sorted_row_id = torch.argsort(routing_flat, stable=True)
    keep_row_id = sorted_row_id[:num_out_tokens]

    row_id_map = torch.full((inp.shape[0],), -1, dtype=torch.int32, device=inp.device)
    if keep_row_id.numel() > 0:
        row_id_map.index_copy_(
            0,
            keep_row_id,
            torch.arange(keep_row_id.numel(), device=inp.device, dtype=torch.int32),
        )

    output = inp.index_select(0, keep_row_id)
    if out is not None:
        out.copy_(output)
        output = out
    return output, row_id_map


def _chunk_unpermute_torch(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    *,
    num_tokens: int,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    if num_tokens == 0:
        output = inp.new_empty((0, inp.shape[1]))
        if out is not None:
            out.copy_(output)
            return out
        return output

    row_id_map_int64 = row_id_map.to(dtype=torch.int64)
    valid = row_id_map_int64 >= 0
    safe_row_id = torch.where(valid, row_id_map_int64, torch.zeros_like(row_id_map_int64))
    gathered = inp.index_select(0, safe_row_id)
    gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))

    if out is not None:
        out.copy_(gathered)
        return out
    return gathered


def _chunk_permute_by_row_id_map_torch(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    *,
    num_out_tokens: int,
) -> torch.Tensor:
    output = inp.new_zeros((num_out_tokens, inp.shape[1]))
    if row_id_map.numel() == 0 or num_out_tokens == 0:
        return output

    row_id_map_int64 = row_id_map.to(dtype=torch.int64)
    valid = (row_id_map_int64 >= 0) & (row_id_map_int64 < num_out_tokens)
    safe_dst_idx = torch.where(valid, row_id_map_int64, torch.zeros_like(row_id_map_int64))
    src_rows = inp * valid.unsqueeze(-1).to(dtype=inp.dtype)
    output.index_add_(0, safe_dst_idx, src_rows)
    return output


def _chunk_permute_cuda(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    *,
    num_out_tokens: int,
    out: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    ext = _load_cuda_extension()
    return ext.chunk_permute_fwd_cuda(inp, routing_map, num_out_tokens, out)


def _chunk_unpermute_cuda(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    *,
    num_tokens: int,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    ext = _load_cuda_extension()
    return ext.chunk_unpermute_fwd_cuda(inp, row_id_map, num_tokens, out)


def _chunk_permute_by_row_id_map_cuda(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    *,
    num_out_tokens: int,
) -> torch.Tensor:
    ext = _load_cuda_extension()
    return ext.chunk_permute_by_row_id_map_cuda(inp, row_id_map, num_out_tokens, None)


def _dispatch_permute(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    *,
    num_out_tokens: int,
    out: Optional[torch.Tensor],
    backend: Literal["cuda", "triton"],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if backend == "cuda":
        return _chunk_permute_cuda(inp, routing_map, num_out_tokens=num_out_tokens, out=out)
    if backend == "triton":
        return _chunk_permute_torch(inp, routing_map, num_out_tokens=num_out_tokens, out=out)
    raise ValueError(f"Invalid backend: {backend}")


def _dispatch_unpermute(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    *,
    num_tokens: int,
    out: Optional[torch.Tensor],
    backend: Literal["cuda", "triton"],
) -> torch.Tensor:
    if backend == "cuda":
        return _chunk_unpermute_cuda(inp, row_id_map, num_tokens=num_tokens, out=out)
    if backend == "triton":
        return _chunk_unpermute_torch(inp, row_id_map, num_tokens=num_tokens, out=out)
    raise ValueError(f"Invalid backend: {backend}")


def _dispatch_permute_by_row_id_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    *,
    num_out_tokens: int,
    backend: Literal["cuda", "triton"],
) -> torch.Tensor:
    if backend == "cuda":
        return _chunk_permute_by_row_id_map_cuda(inp, row_id_map, num_out_tokens=num_out_tokens)
    if backend == "triton":
        return _chunk_permute_by_row_id_map_torch(inp, row_id_map, num_out_tokens=num_out_tokens)
    raise ValueError(f"Invalid backend: {backend}")


class _ChunkPermuteFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        inp: torch.Tensor,
        routing_map: torch.Tensor,
        num_out_tokens: int,
        backend: str,
        out: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output, row_id_map = _dispatch_permute(
            inp,
            routing_map,
            num_out_tokens=num_out_tokens,
            out=out,
            backend=backend,  # type: ignore[arg-type]
        )
        if out is not None and output is out:
            ctx.mark_dirty(output)

        # Keep metadata as direct references to avoid saved_tensors lifetime
        # issues under compiled autograd.
        ctx.row_id_map = row_id_map
        ctx.num_tokens = inp.shape[0]
        ctx.backend = backend
        ctx.mark_non_differentiable(row_id_map)
        return output, row_id_map

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_row_id_map: torch.Tensor):  # type: ignore[override]
        del grad_row_id_map

        grad_inp = None
        if ctx.needs_input_grad[0]:
            grad_inp = _dispatch_unpermute(
                grad_output,
                ctx.row_id_map,
                num_tokens=ctx.num_tokens,
                out=None,
                backend=ctx.backend,  # type: ignore[arg-type]
            )

        return grad_inp, None, None, None, None


class _ChunkUnpermuteFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        inp: torch.Tensor,
        row_id_map: torch.Tensor,
        num_tokens: int,
        backend: str,
        out: Optional[torch.Tensor],
    ) -> torch.Tensor:
        output = _dispatch_unpermute(
            inp,
            row_id_map,
            num_tokens=num_tokens,
            out=out,
            backend=backend,  # type: ignore[arg-type]
        )
        if out is not None and output is out:
            ctx.mark_dirty(output)

        ctx.row_id_map = row_id_map
        ctx.input_rows = inp.shape[0]
        ctx.backend = backend
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        grad_inp = None
        if ctx.needs_input_grad[0]:
            grad_inp = _dispatch_permute_by_row_id_map(
                grad_output,
                ctx.row_id_map,
                num_out_tokens=ctx.input_rows,
                backend=ctx.backend,  # type: ignore[arg-type]
            )

        return grad_inp, None, None, None, None


def moe_chunk_permute(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    *,
    num_out_tokens: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    backend: Literal["auto", "cuda", "triton"] = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not inp.is_cuda:
        raise ValueError("moe_chunk_permute expects CUDA input")
    if backend not in ("auto", "cuda", "triton"):
        raise ValueError(f"Invalid backend: {backend}")

    if num_out_tokens is None:
        num_out_tokens = int(inp.shape[0])
    _check_perm_inputs(inp, routing_map, num_out_tokens, out)

    routing_map_i32 = routing_map
    if routing_map_i32.dtype != torch.int32:
        routing_map_i32 = routing_map_i32.to(dtype=torch.int32)

    if backend == "auto":
        errors = []
        for candidate in ("cuda", "triton"):
            try:
                return _ChunkPermuteFunction.apply(
                    inp,
                    routing_map_i32,
                    num_out_tokens,
                    candidate,
                    out,
                )
            except Exception as e:
                errors.append((candidate, str(e)))
        errors_str = "; ".join([f"{name}: {msg}" for name, msg in errors])
        raise RuntimeError(f"All chunk permute backends failed ({errors_str})")

    return _ChunkPermuteFunction.apply(
        inp,
        routing_map_i32,
        num_out_tokens,
        backend,
        out,
    )


def moe_chunk_unpermute(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    *,
    num_tokens: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    backend: Literal["auto", "cuda", "triton"] = "auto",
) -> torch.Tensor:
    if not inp.is_cuda:
        raise ValueError("moe_chunk_unpermute expects CUDA input")
    if backend not in ("auto", "cuda", "triton"):
        raise ValueError(f"Invalid backend: {backend}")

    if num_tokens is None:
        num_tokens = int(row_id_map.numel())
    _check_unperm_inputs(inp, row_id_map, num_tokens, out)

    row_id_map_i32 = row_id_map
    if row_id_map_i32.dtype != torch.int32:
        row_id_map_i32 = row_id_map_i32.to(dtype=torch.int32)

    if backend == "auto":
        errors = []
        for candidate in ("cuda", "triton"):
            try:
                return _ChunkUnpermuteFunction.apply(
                    inp,
                    row_id_map_i32,
                    num_tokens,
                    candidate,
                    out,
                )
            except Exception as e:
                errors.append((candidate, str(e)))
        errors_str = "; ".join([f"{name}: {msg}" for name, msg in errors])
        raise RuntimeError(f"All chunk unpermute backends failed ({errors_str})")

    return _ChunkUnpermuteFunction.apply(
        inp,
        row_id_map_i32,
        num_tokens,
        backend,
        out,
    )
