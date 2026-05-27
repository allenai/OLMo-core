from __future__ import annotations

import os
from typing import Optional, Tuple

import nvtx
import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton may be unavailable in some test envs.
    triton = None
    tl = None


_F8E4M3_LARGEST_POW2 = 8.0
_F8E8M0_EXP_BIAS = 127.0
_F8E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_FP32_TINY = float(torch.finfo(torch.float32).tiny)

# Default launch configs for Triton kernels. Q/DQ can autotune a small set of
# nearby configs, but these defaults are still used by non-autotuned kernels.
_MXFP8_Q_BLOCK_M = 32
_MXFP8_Q_BLOCK_N = 256
_MXFP8_Q_NUM_WARPS = 4
_MXFP8_Q_NUM_STAGES = 3

_MXFP8_DQ_BLOCK_M = 32
_MXFP8_DQ_BLOCK_N = 256
_MXFP8_DQ_NUM_WARPS = 4
_MXFP8_DQ_NUM_STAGES = 3

_MXFP8_SWIGLU_BLOCK_M = 8
_MXFP8_SWIGLU_BLOCK_N = 256
_MXFP8_SWIGLU_NUM_WARPS = 4
_MXFP8_SWIGLU_NUM_STAGES = 1

_MXFP8_SWIGLU_FROM_FP8_BLOCK_M = 64
_MXFP8_SWIGLU_FROM_FP8_BLOCK_N = 64
_MXFP8_SWIGLU_FROM_FP8_NUM_WARPS = 4
_MXFP8_SWIGLU_FROM_FP8_NUM_STAGES = 3

_MXFP8_REDUCE_BLOCK_M = 64
_MXFP8_REDUCE_BLOCK_N = 128
_MXFP8_REDUCE_NUM_WARPS = 8
_MXFP8_REDUCE_NUM_STAGES = 3

_MXFP8_DOT_BLOCK_M = 32
_MXFP8_DOT_BLOCK_N = 256
_MXFP8_DOT_NUM_WARPS = 4
_MXFP8_DOT_NUM_STAGES = 2

_TE_MXFP8_IMPORT_ATTEMPTED = False
_TE_MXFP8_STATE = None


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _mxfp8_backend(kind: str) -> str:
    specific_backend = os.environ.get(f"OLMO_MXFP8_{kind}_BACKEND")
    backend = specific_backend
    if backend is None:
        backend = os.environ.get("OLMO_MXFP8_QDQ_BACKEND", "triton")
    return backend.strip().lower()


def _mxfp8_backend_wants_te(kind: str) -> bool:
    return _mxfp8_backend(kind) in {"te", "transformer_engine"}


def _get_te_mxfp8_state():
    global _TE_MXFP8_IMPORT_ATTEMPTED, _TE_MXFP8_STATE
    if _TE_MXFP8_IMPORT_ATTEMPTED:
        return _TE_MXFP8_STATE

    _TE_MXFP8_IMPORT_ATTEMPTED = True
    try:
        import transformer_engine.pytorch  # noqa: F401
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.tensor.mxfp8_tensor import (
            MXFP8Quantizer,
            MXFP8Tensor,
        )
    except Exception:
        _TE_MXFP8_STATE = None
        return None

    quantizer = MXFP8Quantizer(
        tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    _TE_MXFP8_STATE = (tex, MXFP8Tensor, quantizer)
    return _TE_MXFP8_STATE


def _te_mxfp8_compact_scale_layout(rows: int, cols: int, block_size: int) -> bool:
    scale_cols = cols // block_size
    return rows % 128 == 0 and scale_cols % 4 == 0


def to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """
    Rearrange matrix scales to cuBLAS blocked/swizzled layout.

    Input shape is [H, W] where W is per-row scale groups.
    Output is flattened blocked representation expected by MX kernels.
    """
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols), device=input_matrix.device, dtype=input_matrix.dtype
        )
        padded[:rows, :cols] = input_matrix

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten().contiguous()


if triton is not None:
    _MXFP8_Q_AUTOTUNE_CONFIGS = [
        # Keep a broader set while we characterize B200 codegen. Two-chunk Q is
        # the default for the routed up/gate save path, so wide 2H activations do
        # not depend on one perfect full-width specialization.
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    ]

    _MXFP8_DQ_AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    ]

    @triton.jit
    def _dest_indices_for_block(
        row_offs,
        col_offs,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        # Rearrange (128,4) into the SWIZZLE_32_4_4 destination index pattern.
        r_div_32 = row_offs // 32
        r_mod_32 = row_offs % 32
        dest_indices = r_mod_32 * 16 + r_div_32 * 4 + col_offs
        return tl.reshape(dest_indices, (BLOCK_ROWS * BLOCK_COLS))

    @triton.jit
    def _triton_scale_swizzle_m_groups(
        scales_ptr,
        scales_stride_dim0,
        scales_stride_dim1,
        scale_cols,
        orig_offsets,
        output_scales_ptr,
        output_scales_stride_dim0,
        output_stride_per_block,
        output_stride_per_row_of_blocks,
        num_groups: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        group_pid = tl.program_id(0)
        block_col_pid = tl.program_id(1)

        input_group_start_row = tl.load(orig_offsets + group_pid - 1, mask=group_pid > 0, other=0)
        input_group_end_row = tl.load(
            orig_offsets + group_pid, mask=group_pid < num_groups, other=0
        )
        output_group_start_row = 0
        prev_group_end_for_start = 0
        for i in tl.static_range(0, num_groups):
            group_end_for_start = tl.load(orig_offsets + i)
            group_size = group_end_for_start - prev_group_end_for_start
            padded_group_size = ((group_size + BLOCK_ROWS - 1) // BLOCK_ROWS) * BLOCK_ROWS
            output_group_start_row += tl.where(i < group_pid, padded_group_size, 0)
            prev_group_end_for_start = group_end_for_start

        row_offs = tl.arange(0, BLOCK_ROWS)[:, None]
        col_offs = tl.arange(0, BLOCK_COLS)[None, :]
        dest_indices_flat = _dest_indices_for_block(
            row_offs,
            col_offs,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
        )

        block_row_id = 0
        current_start_row = input_group_start_row
        while current_start_row < input_group_end_row:
            block_row_offs = current_start_row + row_offs
            block_col_offs = block_col_pid * BLOCK_COLS + col_offs
            block_offs = block_row_offs * scales_stride_dim0 + block_col_offs * scales_stride_dim1
            mask = (block_row_offs < input_group_end_row) & (block_col_offs < scale_cols)
            input_scales = tl.load(scales_ptr + block_offs, mask=mask, other=0)
            scales_flat = tl.reshape(input_scales, (BLOCK_ROWS * BLOCK_COLS))

            output_block_offsets = (
                output_group_start_row * output_scales_stride_dim0
                + (block_row_id * output_stride_per_row_of_blocks)
                + (block_col_pid * output_stride_per_block)
            )
            tl.store(output_scales_ptr + output_block_offsets + dest_indices_flat, scales_flat)

            block_row_id += 1
            current_start_row += BLOCK_ROWS

    @triton.jit
    def _triton_scale_swizzle_per_group_3d(
        input_ptr,
        input_stride_dim0,
        input_stride_dim1,
        input_stride_dim2,
        output_ptr,
        output_stride_dim0,
        output_block_stride,
        scale_rows,
        scale_cols,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        pid_group = tl.program_id(0)
        pid_row = tl.program_id(1)
        pid_col = tl.program_id(2)

        input_ptr += pid_group * input_stride_dim0
        output_ptr += pid_group * output_stride_dim0

        row_offs = tl.arange(0, BLOCK_ROWS)[:, None]
        col_offs = tl.arange(0, BLOCK_COLS)[None, :]
        dest_indices_flat = _dest_indices_for_block(
            row_offs,
            col_offs,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
        )

        start_row = pid_row * BLOCK_ROWS
        start_col = pid_col * BLOCK_COLS
        global_rows = start_row + row_offs
        global_cols = start_col + col_offs
        mask = (global_rows < scale_rows) & (global_cols < scale_cols)

        input_scales = tl.load(
            input_ptr + global_rows * input_stride_dim1 + global_cols * input_stride_dim2,
            mask=mask,
            other=0.0,
        )
        scales_flat = tl.reshape(input_scales, (BLOCK_ROWS * BLOCK_COLS))

        local_numel = BLOCK_ROWS * BLOCK_COLS
        block_offset = pid_col * local_numel + (pid_row * output_block_stride)
        tl.store(output_ptr + block_offset + dest_indices_flat, scales_flat)

    # Broader Q autotune while we characterize B200 codegen. The routed up/gate
    # save path defaults to two chunks, so the widest activation no longer
    # depends on one fragile full-width specialization.
    @triton.autotune(
        configs=_MXFP8_Q_AUTOTUNE_CONFIGS,
        key=["n_rows", "n_cols", "BLOCK_SIZE", "USE_BF16_EXP"],
    )
    @triton.jit
    def _triton_mxfp8_quantize_dim0(
        x_ptr,
        x_stride_0,
        x_stride_1,
        q_ptr,
        q_stride_0,
        q_stride_1,
        scale_ptr,
        s_stride_0,
        s_stride_1,
        n_rows,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        USE_BF16_EXP: tl.constexpr,
    ):
        # Quantizes contiguous [M, K] to:
        # - q_ptr: e4m3 values [M, K]
        # - scale_ptr: e8m0 biased exponents [M, K//BLOCK_SIZE] as uint8
        FP32_TINY: tl.constexpr = 1.1754943508222875e-38
        F8E4M3_MAX: tl.constexpr = 448.0
        E8M0_EXP_BIAS: tl.constexpr = 127.0
        E8M0_EXP_BIAS_INT: tl.constexpr = 127
        F8E4M3_LARGEST_POW2: tl.constexpr = 8.0
        F8E4M3_LARGEST_POW2_INT: tl.constexpr = 8
        BF16_MBITS: tl.constexpr = 7
        SCALE_BLOCKS_PER_TILE: tl.constexpr = BLOCK_N // BLOCK_SIZE

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

        x_offsets = row_idx * x_stride_0 + col_idx * x_stride_1
        q_offsets = row_idx * q_stride_0 + col_idx * q_stride_1
        q_mask = (row_idx < n_rows) & (col_idx < n_cols)
        x = tl.load(x_ptr + x_offsets, mask=q_mask, other=0.0)

        x_r = x.reshape(BLOCK_M * SCALE_BLOCKS_PER_TILE, BLOCK_SIZE).to(tl.float32)
        max_abs = tl.max(tl.abs(x_r), axis=1)
        max_abs = tl.where(max_abs == max_abs, max_abs, 0.0)  # NaN -> 0
        max_abs = tl.maximum(max_abs, FP32_TINY)

        if USE_BF16_EXP:
            max_abs_bf16 = max_abs.to(tl.bfloat16)
            max_abs_i16 = max_abs_bf16.to(tl.int16, bitcast=True)
            largest_p2 = ((max_abs_i16 >> BF16_MBITS) & 0xFF) - E8M0_EXP_BIAS_INT
            scale_unbiased = largest_p2 - F8E4M3_LARGEST_POW2_INT
            scale_unbiased = tl.maximum(scale_unbiased, -E8M0_EXP_BIAS_INT)
            scale_unbiased = tl.minimum(scale_unbiased, E8M0_EXP_BIAS_INT)
            scale_biased_i32 = (scale_unbiased + E8M0_EXP_BIAS_INT).to(tl.int32)
            scale_biased_u8 = scale_biased_i32.to(tl.uint8)
            inv_exp = 254 - scale_biased_i32
            inv_bits = inv_exp.to(tl.uint32) << 23
            inv_bits = tl.where(inv_exp == 0, 0x00400000, inv_bits)
            inv_scale = inv_bits.to(tl.float32, bitcast=True)
            q_hp = x_r * inv_scale[:, None]
        else:
            largest_p2 = tl.floor(tl.log2(max_abs))
            scale_unbiased = largest_p2 - F8E4M3_LARGEST_POW2
            scale_unbiased = tl.maximum(scale_unbiased, -E8M0_EXP_BIAS)
            scale_unbiased = tl.minimum(scale_unbiased, E8M0_EXP_BIAS)

            scale_biased_u8 = (scale_unbiased + E8M0_EXP_BIAS).to(tl.uint8)
            dequant_scale = tl.exp2(scale_unbiased).to(tl.float32)
            q_hp = x_r / dequant_scale[:, None]

        q_hp = tl.where(q_hp == q_hp, q_hp, 0.0)
        q_hp = tl.maximum(q_hp, -F8E4M3_MAX)
        q_hp = tl.minimum(q_hp, F8E4M3_MAX)

        q_tile = tl.reshape(q_hp, (BLOCK_M, BLOCK_N)).to(tl.float8e4nv)
        tl.store(q_ptr + q_offsets, q_tile, mask=q_mask)

        n_scale_cols = tl.cdiv(n_cols, BLOCK_SIZE)
        scale_col_idx = pid_n * SCALE_BLOCKS_PER_TILE + tl.arange(0, SCALE_BLOCKS_PER_TILE)[None, :]
        scale_offsets = row_idx * s_stride_0 + scale_col_idx * s_stride_1
        scale_mask = (row_idx < n_rows) & (scale_col_idx < n_scale_cols)
        scale_tile = tl.reshape(scale_biased_u8, (BLOCK_M, SCALE_BLOCKS_PER_TILE))
        tl.store(scale_ptr + scale_offsets, scale_tile, mask=scale_mask)

    # NOTE: Deliberately no @triton.autotune here.
    @triton.jit
    def _triton_mxfp8_quantize_grouped_blocked_tiled_dim0(
        x_ptr,
        x_stride_0,
        x_stride_1,
        q_ptr,
        q_stride_0,
        q_stride_1,
        blocked_scale_ptr,
        blocked_scale_stride_0,
        offs,
        n_rows,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        USE_BF16_EXP: tl.constexpr,
    ):
        # Same quantization tile shape as _triton_mxfp8_quantize_dim0, but
        # writes scale bytes directly in grouped SWIZZLE_32_4_4 layout.
        FP32_TINY: tl.constexpr = 1.1754943508222875e-38
        F8E4M3_MAX: tl.constexpr = 448.0
        E8M0_EXP_BIAS: tl.constexpr = 127.0
        E8M0_EXP_BIAS_INT: tl.constexpr = 127
        F8E4M3_LARGEST_POW2: tl.constexpr = 8.0
        F8E4M3_LARGEST_POW2_INT: tl.constexpr = 8
        BF16_MBITS: tl.constexpr = 7
        SCALE_BLOCKS_PER_TILE: tl.constexpr = BLOCK_N // BLOCK_SIZE

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

        x_offsets = row_idx * x_stride_0 + col_idx * x_stride_1
        q_offsets = row_idx * q_stride_0 + col_idx * q_stride_1
        q_mask = (row_idx < n_rows) & (col_idx < n_cols)
        x = tl.load(x_ptr + x_offsets, mask=q_mask, other=0.0)

        x_r = x.reshape(BLOCK_M * SCALE_BLOCKS_PER_TILE, BLOCK_SIZE).to(tl.float32)
        max_abs = tl.max(tl.abs(x_r), axis=1)
        max_abs = tl.where(max_abs == max_abs, max_abs, 0.0)  # NaN -> 0
        max_abs = tl.maximum(max_abs, FP32_TINY)

        if USE_BF16_EXP:
            max_abs_bf16 = max_abs.to(tl.bfloat16)
            max_abs_i16 = max_abs_bf16.to(tl.int16, bitcast=True)
            largest_p2 = ((max_abs_i16 >> BF16_MBITS) & 0xFF) - E8M0_EXP_BIAS_INT
            scale_unbiased = largest_p2 - F8E4M3_LARGEST_POW2_INT
            scale_unbiased = tl.maximum(scale_unbiased, -E8M0_EXP_BIAS_INT)
            scale_unbiased = tl.minimum(scale_unbiased, E8M0_EXP_BIAS_INT)
            scale_biased_i32 = (scale_unbiased + E8M0_EXP_BIAS_INT).to(tl.int32)
            scale_biased_u8 = scale_biased_i32.to(tl.uint8)
            inv_exp = 254 - scale_biased_i32
            inv_bits = inv_exp.to(tl.uint32) << 23
            inv_bits = tl.where(inv_exp == 0, 0x00400000, inv_bits)
            inv_scale = inv_bits.to(tl.float32, bitcast=True)
            q_hp = x_r * inv_scale[:, None]
        else:
            largest_p2 = tl.floor(tl.log2(max_abs))
            scale_unbiased = largest_p2 - F8E4M3_LARGEST_POW2
            scale_unbiased = tl.maximum(scale_unbiased, -E8M0_EXP_BIAS)
            scale_unbiased = tl.minimum(scale_unbiased, E8M0_EXP_BIAS)

            scale_biased_u8 = (scale_unbiased + E8M0_EXP_BIAS).to(tl.uint8)
            dequant_scale = tl.exp2(scale_unbiased).to(tl.float32)
            q_hp = x_r / dequant_scale[:, None]

        q_hp = tl.where(q_hp == q_hp, q_hp, 0.0)
        q_hp = tl.maximum(q_hp, -F8E4M3_MAX)
        q_hp = tl.minimum(q_hp, F8E4M3_MAX)

        q_tile = tl.reshape(q_hp, (BLOCK_M, BLOCK_N)).to(tl.float8e4nv)
        tl.store(q_ptr + q_offsets, q_tile, mask=q_mask)

        row_vec = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        prev_group_end = 0
        output_group_start = tl.zeros((BLOCK_M,), dtype=tl.int32)
        running_output_group_start = 0
        local_row = row_vec
        row_is_active = tl.zeros((BLOCK_M,), dtype=tl.int1)
        for group_idx in tl.static_range(0, NUM_GROUPS):
            group_end = tl.load(offs + group_idx)
            in_group = (row_vec >= prev_group_end) & (row_vec < group_end)
            local_row = tl.where(in_group, row_vec - prev_group_end, local_row)
            output_group_start = tl.where(in_group, running_output_group_start, output_group_start)
            row_is_active = row_is_active | in_group
            group_size = group_end - prev_group_end
            running_output_group_start += ((group_size + 127) // 128) * 128
            prev_group_end = group_end

        scale_col_idx = pid_n * SCALE_BLOCKS_PER_TILE + tl.arange(0, SCALE_BLOCKS_PER_TILE)
        scale_tile = tl.reshape(scale_biased_u8, (BLOCK_M, SCALE_BLOCKS_PER_TILE))
        local_row_2d = local_row[:, None]
        output_group_start_2d = output_group_start[:, None]
        scale_col_2d = scale_col_idx[None, :]

        macro_row_block = local_row_2d // 128
        macro_col_block = scale_col_2d // 4
        local_row_in_macro = local_row_2d % 128
        local_col = scale_col_2d % 4
        group_in_macro = local_row_in_macro // 32
        sub_row = local_row_in_macro % 32

        swizzled_offsets = (
            (output_group_start_2d + macro_row_block * 128) * blocked_scale_stride_0
            + macro_col_block * 512
            + sub_row * 16
            + group_in_macro * 4
            + local_col
        )
        n_scale_cols = tl.cdiv(n_cols, BLOCK_SIZE)
        scale_mask = (
            (row_vec[:, None] < n_rows) & row_is_active[:, None] & (scale_col_2d < n_scale_cols)
        )
        tl.store(blocked_scale_ptr + swizzled_offsets, scale_tile, mask=scale_mask)

    # NOTE: Deliberately no @triton.autotune here.
    @triton.jit
    def _triton_swiglu_quantize_dim0(
        up_gate_ptr,
        ug_stride_0,
        ug_stride_1,
        h_ptr,
        h_stride_0,
        h_stride_1,
        q_ptr,
        q_stride_0,
        q_stride_1,
        scale_ptr,
        s_stride_0,
        s_stride_1,
        n_rows,
        hidden,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
    ):
        # Computes h = up * silu(gate) and quantizes h to MXFP8 in one pass.
        FP32_TINY: tl.constexpr = 1.1754943508222875e-38
        F8E4M3_MAX: tl.constexpr = 448.0
        E8M0_EXP_BIAS: tl.constexpr = 127.0
        F8E4M3_LARGEST_POW2: tl.constexpr = 8.0
        SCALE_BLOCKS_PER_TILE: tl.constexpr = BLOCK_N // BLOCK_SIZE

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        h_mask = (row_idx < n_rows) & (col_idx < hidden)

        up_offsets = row_idx * ug_stride_0 + col_idx * ug_stride_1
        gate_offsets = row_idx * ug_stride_0 + (col_idx + hidden) * ug_stride_1
        if EVEN_M and EVEN_N:
            up = tl.load(up_gate_ptr + up_offsets).to(tl.float32)
            gate = tl.load(up_gate_ptr + gate_offsets).to(tl.float32)
        else:
            up = tl.load(up_gate_ptr + up_offsets, mask=h_mask, other=0.0).to(tl.float32)
            gate = tl.load(up_gate_ptr + gate_offsets, mask=h_mask, other=0.0).to(tl.float32)
        silu_gate = gate * tl.sigmoid(gate)
        h = up * silu_gate

        h_offsets = row_idx * h_stride_0 + col_idx * h_stride_1
        if EVEN_M and EVEN_N:
            tl.store(h_ptr + h_offsets, h)
        else:
            tl.store(h_ptr + h_offsets, h, mask=h_mask)

        h_r = h.reshape(BLOCK_M * SCALE_BLOCKS_PER_TILE, BLOCK_SIZE)
        max_abs = tl.max(tl.abs(h_r), axis=1)
        max_abs = tl.where(max_abs == max_abs, max_abs, 0.0)  # NaN -> 0
        max_abs = tl.maximum(max_abs, FP32_TINY)

        largest_p2 = tl.floor(tl.log2(max_abs))
        scale_unbiased = largest_p2 - F8E4M3_LARGEST_POW2
        scale_unbiased = tl.maximum(scale_unbiased, -E8M0_EXP_BIAS)
        scale_unbiased = tl.minimum(scale_unbiased, E8M0_EXP_BIAS)
        scale_biased_u8 = (scale_unbiased + E8M0_EXP_BIAS).to(tl.uint8)

        inv_dequant_scale = tl.exp2(-scale_unbiased).to(tl.float32)
        q_hp = h_r * inv_dequant_scale[:, None]
        q_hp = tl.where(q_hp == q_hp, q_hp, 0.0)
        q_hp = tl.maximum(q_hp, -F8E4M3_MAX)
        q_hp = tl.minimum(q_hp, F8E4M3_MAX)
        q_tile = tl.reshape(q_hp, (BLOCK_M, BLOCK_N)).to(tl.float8e4nv)

        q_offsets = row_idx * q_stride_0 + col_idx * q_stride_1
        if EVEN_M and EVEN_N:
            tl.store(q_ptr + q_offsets, q_tile)
        else:
            tl.store(q_ptr + q_offsets, q_tile, mask=h_mask)

        n_scale_cols = hidden // BLOCK_SIZE
        scale_col_idx = pid_n * SCALE_BLOCKS_PER_TILE + tl.arange(0, SCALE_BLOCKS_PER_TILE)[None, :]
        scale_offsets = row_idx * s_stride_0 + scale_col_idx * s_stride_1
        scale_mask = (row_idx < n_rows) & (scale_col_idx < n_scale_cols)
        scale_tile = tl.reshape(scale_biased_u8, (BLOCK_M, SCALE_BLOCKS_PER_TILE))
        if EVEN_M and EVEN_N:
            tl.store(scale_ptr + scale_offsets, scale_tile)
        else:
            tl.store(scale_ptr + scale_offsets, scale_tile, mask=scale_mask)

    # NOTE: Deliberately no @triton.autotune here.
    @triton.jit
    def _triton_swiglu_quantize_from_mxfp8_dim0(
        up_gate_q_ptr,
        uq_stride_0,
        uq_stride_1,
        up_gate_scales_u8_ptr,
        us_stride_0,
        us_stride_1,
        h_ptr,
        h_stride_0,
        h_stride_1,
        q_ptr,
        q_stride_0,
        q_stride_1,
        scale_ptr,
        s_stride_0,
        s_stride_1,
        n_rows,
        hidden,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # Computes h = up * silu(gate) from MXFP8 input and requantizes h to MXFP8.
        FP32_TINY: tl.constexpr = 1.1754943508222875e-38
        F8E4M3_MAX: tl.constexpr = 448.0
        E8M0_EXP_BIAS: tl.constexpr = 127.0
        F8E4M3_LARGEST_POW2: tl.constexpr = 8.0
        SCALE_BLOCKS_PER_TILE: tl.constexpr = BLOCK_N // BLOCK_SIZE

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        h_mask = (row_idx < n_rows) & (col_idx < hidden)

        up_offsets = row_idx * uq_stride_0 + col_idx * uq_stride_1
        gate_offsets = row_idx * uq_stride_0 + (col_idx + hidden) * uq_stride_1
        up_q = tl.load(up_gate_q_ptr + up_offsets, mask=h_mask, other=0.0).to(tl.float32)
        gate_q = tl.load(up_gate_q_ptr + gate_offsets, mask=h_mask, other=0.0).to(tl.float32)

        up_scale_cols = col_idx // BLOCK_SIZE
        gate_scale_cols = (col_idx + hidden) // BLOCK_SIZE
        up_scale_offsets = row_idx * us_stride_0 + up_scale_cols * us_stride_1
        gate_scale_offsets = row_idx * us_stride_0 + gate_scale_cols * us_stride_1
        up_scale_u8 = tl.load(
            up_gate_scales_u8_ptr + up_scale_offsets,
            mask=h_mask,
            other=0,
        ).to(tl.int32)
        gate_scale_u8 = tl.load(
            up_gate_scales_u8_ptr + gate_scale_offsets,
            mask=h_mask,
            other=0,
        ).to(tl.int32)
        up_scale = tl.exp2(up_scale_u8.to(tl.float32) - E8M0_EXP_BIAS)
        gate_scale = tl.exp2(gate_scale_u8.to(tl.float32) - E8M0_EXP_BIAS)

        up = up_q * up_scale
        gate = gate_q * gate_scale
        silu_gate = gate * tl.sigmoid(gate)
        h = up * silu_gate

        h_offsets = row_idx * h_stride_0 + col_idx * h_stride_1
        tl.store(h_ptr + h_offsets, h, mask=h_mask)

        h_r = h.reshape(BLOCK_M * SCALE_BLOCKS_PER_TILE, BLOCK_SIZE)
        max_abs = tl.max(tl.abs(h_r), axis=1)
        max_abs = tl.where(max_abs == max_abs, max_abs, 0.0)  # NaN -> 0
        max_abs = tl.maximum(max_abs, FP32_TINY)

        largest_p2 = tl.floor(tl.log2(max_abs))
        scale_unbiased = largest_p2 - F8E4M3_LARGEST_POW2
        scale_unbiased = tl.maximum(scale_unbiased, -E8M0_EXP_BIAS)
        scale_unbiased = tl.minimum(scale_unbiased, E8M0_EXP_BIAS)
        scale_biased_u8 = (scale_unbiased + E8M0_EXP_BIAS).to(tl.uint8)

        inv_dequant_scale = tl.exp2(-scale_unbiased).to(tl.float32)
        q_hp = h_r * inv_dequant_scale[:, None]
        q_hp = tl.where(q_hp == q_hp, q_hp, 0.0)
        q_hp = tl.maximum(q_hp, -F8E4M3_MAX)
        q_hp = tl.minimum(q_hp, F8E4M3_MAX)
        q_tile = tl.reshape(q_hp, (BLOCK_M, BLOCK_N)).to(tl.float8e4nv)

        q_offsets = row_idx * q_stride_0 + col_idx * q_stride_1
        tl.store(q_ptr + q_offsets, q_tile, mask=h_mask)

        n_scale_cols = hidden // BLOCK_SIZE
        scale_col_idx = pid_n * SCALE_BLOCKS_PER_TILE + tl.arange(0, SCALE_BLOCKS_PER_TILE)[None, :]
        scale_offsets = row_idx * s_stride_0 + scale_col_idx * s_stride_1
        scale_mask = (row_idx < n_rows) & (scale_col_idx < n_scale_cols)
        scale_tile = tl.reshape(scale_biased_u8, (BLOCK_M, SCALE_BLOCKS_PER_TILE))
        tl.store(scale_ptr + scale_offsets, scale_tile, mask=scale_mask)

    # NOTE: Deliberately no @triton.autotune here.
    @triton.jit
    def _triton_mxfp8_reduce_gathered_dim1(
        gathered_q_ptr,
        gq_stride_0,
        gq_stride_1,
        gq_stride_2,
        gathered_scales_u8_ptr,
        gs_stride_0,
        gs_stride_1,
        gs_stride_2,
        probs_ptr,
        p_stride_0,
        p_stride_1,
        valid_mask_ptr,
        v_stride_0,
        v_stride_1,
        out_acc_ptr,
        o_stride_0,
        o_stride_1,
        n_rows,
        top_k,
        d_model,
        BLOCK_SIZE: tl.constexpr,
        HAS_PROBS: tl.constexpr,
        HAS_VALID_MASK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # Reduces gathered [N, K, D] MXFP8 routes into [N, D] in fp32.
        E8M0_EXP_BIAS: tl.constexpr = 127.0
        SCALE_BLOCKS_PER_TILE: tl.constexpr = BLOCK_N // BLOCK_SIZE

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        row_ids = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_idx = row_ids[:, None]
        col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        out_mask = (row_idx < n_rows) & (col_idx < d_model)

        n_scale_cols = d_model // BLOCK_SIZE
        scale_col_idx = pid_n * SCALE_BLOCKS_PER_TILE + tl.arange(0, SCALE_BLOCKS_PER_TILE)[None, :]
        scale_mask = (row_idx < n_rows) & (scale_col_idx < n_scale_cols)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        k = 0
        while k < top_k:
            route_mask = out_mask
            scale_route_mask = scale_mask
            route_valid = tl.full((BLOCK_M,), 1, dtype=tl.int1)
            if HAS_VALID_MASK:
                route_valid = tl.load(
                    valid_mask_ptr + row_ids * v_stride_0 + k * v_stride_1,
                    mask=(row_ids < n_rows),
                    other=0,
                ).to(tl.int1)
                route_mask = route_mask & route_valid[:, None]
                scale_route_mask = scale_route_mask & route_valid[:, None]

            q_offsets = row_idx * gq_stride_0 + k * gq_stride_1 + col_idx * gq_stride_2
            q = tl.load(gathered_q_ptr + q_offsets, mask=route_mask, other=0.0).to(tl.float32)

            s_offsets = row_idx * gs_stride_0 + k * gs_stride_1 + scale_col_idx * gs_stride_2
            s_u8 = tl.load(gathered_scales_u8_ptr + s_offsets, mask=scale_route_mask, other=0).to(
                tl.int32
            )
            scales = tl.exp2(s_u8.to(tl.float32) - E8M0_EXP_BIAS)

            q_blocks = q.reshape(BLOCK_M * SCALE_BLOCKS_PER_TILE, BLOCK_SIZE)
            scale_blocks = scales.reshape(BLOCK_M * SCALE_BLOCKS_PER_TILE)
            route = tl.reshape(
                q_blocks * scale_blocks[:, None],
                (BLOCK_M, BLOCK_N),
            )
            if HAS_PROBS:
                probs_row = tl.load(
                    probs_ptr + row_ids * p_stride_0 + k * p_stride_1,
                    mask=(row_ids < n_rows),
                    other=0.0,
                ).to(tl.float32)
                if HAS_VALID_MASK:
                    probs_row = tl.where(route_valid, probs_row, 0.0)
                route = route * probs_row[:, None]

            acc += route
            k += 1

        out_offsets = row_idx * o_stride_0 + col_idx * o_stride_1
        tl.store(out_acc_ptr + out_offsets, acc, mask=out_mask)

    @triton.autotune(
        configs=_MXFP8_DQ_AUTOTUNE_CONFIGS,
        key=["n_rows", "n_cols", "BLOCK_SIZE"],
    )
    @triton.jit
    def _triton_mxfp8_dequantize_dim0(
        q_ptr,
        q_stride_0,
        q_stride_1,
        scales_u8_ptr,
        s_stride_0,
        s_stride_1,
        out_ptr,
        o_stride_0,
        o_stride_1,
        n_rows,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        E8M0_EXP_BIAS: tl.constexpr = 127.0
        SCALE_BLOCKS_PER_TILE: tl.constexpr = BLOCK_N // BLOCK_SIZE
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        out_mask = (row_idx < n_rows) & (col_idx < n_cols)

        q_offsets = row_idx * q_stride_0 + col_idx * q_stride_1
        q = tl.load(q_ptr + q_offsets, mask=out_mask, other=0.0).to(tl.float32)

        n_scale_cols = tl.cdiv(n_cols, BLOCK_SIZE)
        scale_col_idx = pid_n * SCALE_BLOCKS_PER_TILE + tl.arange(0, SCALE_BLOCKS_PER_TILE)[None, :]
        scale_mask = (row_idx < n_rows) & (scale_col_idx < n_scale_cols)
        s_offsets = row_idx * s_stride_0 + scale_col_idx * s_stride_1
        s_u8 = tl.load(scales_u8_ptr + s_offsets, mask=scale_mask, other=0).to(tl.int32)
        scales = tl.exp2(s_u8.to(tl.float32) - E8M0_EXP_BIAS)

        q_blocks = q.reshape(BLOCK_M * SCALE_BLOCKS_PER_TILE, BLOCK_SIZE)
        scale_blocks = scales.reshape(BLOCK_M * SCALE_BLOCKS_PER_TILE)
        out_val = tl.reshape(
            q_blocks * scale_blocks[:, None],
            (BLOCK_M, BLOCK_N),
        )
        out_offsets = row_idx * o_stride_0 + col_idx * o_stride_1
        tl.store(out_ptr + out_offsets, out_val, mask=out_mask)

    # NOTE: Deliberately no @triton.autotune here.
    @triton.jit
    def _triton_mxfp8_dot_gathered_with_grad(
        gathered_q_ptr,
        gq_stride_0,
        gq_stride_1,
        gq_stride_2,
        gathered_scales_u8_ptr,
        gs_stride_0,
        gs_stride_1,
        gs_stride_2,
        grad_out_ptr,
        go_stride_0,
        go_stride_1,
        valid_mask_ptr,
        v_stride_0,
        v_stride_1,
        out_ptr,
        o_stride_0,
        o_stride_1,
        n_rows,
        top_k,
        d_model,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        E8M0_EXP_BIAS: tl.constexpr = 127.0
        SCALE_BLOCKS_PER_TILE: tl.constexpr = BLOCK_N // BLOCK_SIZE

        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        row_ids = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rows_mask = row_ids < n_rows
        route_valid = tl.load(
            valid_mask_ptr + row_ids * v_stride_0 + pid_k * v_stride_1,
            mask=rows_mask,
            other=0,
        ).to(tl.int1)
        active_rows = rows_mask & route_valid

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        col_start = 0
        while col_start < d_model:
            col_idx = col_start + tl.arange(0, BLOCK_N)
            q_mask = active_rows[:, None] & (col_idx[None, :] < d_model)

            q_offsets = (
                row_ids[:, None] * gq_stride_0
                + pid_k * gq_stride_1
                + col_idx[None, :] * gq_stride_2
            )
            q = tl.load(gathered_q_ptr + q_offsets, mask=q_mask, other=0.0).to(tl.float32)

            scale_col_idx = (col_start // BLOCK_SIZE) + tl.arange(0, SCALE_BLOCKS_PER_TILE)
            s_mask = active_rows[:, None] & (scale_col_idx[None, :] < (d_model // BLOCK_SIZE))
            s_offsets = (
                row_ids[:, None] * gs_stride_0
                + pid_k * gs_stride_1
                + scale_col_idx[None, :] * gs_stride_2
            )
            s_u8 = tl.load(gathered_scales_u8_ptr + s_offsets, mask=s_mask, other=0).to(tl.int32)
            scales = tl.exp2(s_u8.to(tl.float32) - E8M0_EXP_BIAS)
            q_blocks = q.reshape(BLOCK_M * SCALE_BLOCKS_PER_TILE, BLOCK_SIZE)
            scale_blocks = scales.reshape(BLOCK_M * SCALE_BLOCKS_PER_TILE)
            route = tl.reshape(
                q_blocks * scale_blocks[:, None],
                (BLOCK_M, BLOCK_N),
            )

            go_offsets = row_ids[:, None] * go_stride_0 + col_idx[None, :] * go_stride_1
            grad = tl.load(grad_out_ptr + go_offsets, mask=q_mask, other=0.0).to(tl.float32)

            acc += tl.sum(route * grad, axis=1)
            col_start += BLOCK_N

        out_offsets = row_ids * o_stride_0 + pid_k * o_stride_1
        tl.store(out_ptr + out_offsets, acc, mask=rows_mask)


def _to_blocked_m_groups_triton(
    scales_tensor: torch.Tensor,
    offs: torch.Tensor,
    *,
    zero_unwritten_tail: bool = True,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for CUDA no-sync grouped MXFP8 scale swizzle")

    rows, cols = scales_tensor.shape
    num_groups = int(offs.shape[0])
    num_col_blocks = ceil_div(cols, 4)
    padded_cols = num_col_blocks * 4

    # Upper-bound static padding per group to avoid host-side offset inspection.
    padded_rows = rows + num_groups * 128
    if zero_unwritten_tail:
        output = scales_tensor.new_zeros((padded_rows, padded_cols))
    else:
        # The Triton kernel writes every row/column that grouped-mm can read
        # from the per-group 128-row padded layout. The extra upper-bound tail
        # rows are unreachable through `offs`, so the hot path can skip memset.
        output = scales_tensor.new_empty((padded_rows, padded_cols))

    block_rows = 128
    block_cols = 4
    output_stride_per_block = block_rows * block_cols
    output_stride_per_row_of_blocks = block_rows * block_cols * (padded_cols // block_cols)

    grid = (num_groups, num_col_blocks)
    _triton_scale_swizzle_m_groups[grid](
        scales_tensor.view(torch.uint8),
        scales_tensor.stride(0),
        scales_tensor.stride(1),
        cols,
        offs,
        output.view(torch.uint8),
        output.stride(0),
        output_stride_per_block,
        output_stride_per_row_of_blocks,
        num_groups=num_groups,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
    )
    return output


def _to_blocked_m_groups_fallback(scales_tensor: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """
    Reference path (CPU/non-Triton). Not intended for CUDA no-sync training.
    """
    rows, cols = scales_tensor.shape
    num_groups = int(offs.shape[0])
    padded_rows = rows + num_groups * 128
    output = scales_tensor.new_zeros((padded_rows, cols))

    offs_cpu = offs.to(device="cpu", dtype=torch.int64)
    in_start = 0
    out_start = 0
    for end in offs_cpu.tolist():
        end_i = int(end)
        group_rows = end_i - in_start
        if group_rows <= 0:
            in_start = end_i
            continue

        blocked = to_blocked(scales_tensor[in_start:end_i]).view(-1, cols)
        out_end = out_start + blocked.shape[0]
        output[out_start:out_end].copy_(blocked)
        in_start = end_i
        out_start = out_end

    return output


def _to_blocked_per_group_triton(scales: torch.Tensor) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for CUDA per-group MXFP8 scale swizzle")

    num_groups, rows, cols = scales.shape
    num_row_blocks = ceil_div(rows, 128)
    num_col_blocks = ceil_div(cols, 4)
    padded_rows = num_row_blocks * 128
    padded_cols = num_col_blocks * 4
    output = scales.new_empty((num_groups, padded_rows * padded_cols))

    block_rows = 128
    block_cols = 4
    output_block_stride = block_rows * block_cols * (padded_cols // block_cols)
    grid = (num_groups, num_row_blocks, num_col_blocks)
    _triton_scale_swizzle_per_group_3d[grid](
        scales.view(torch.uint8),
        scales.stride(0),
        scales.stride(1),
        scales.stride(2),
        output.view(torch.uint8),
        output.stride(0),
        output_block_stride,
        rows,
        cols,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
    )
    return output


def _to_blocked_per_group(scales: torch.Tensor) -> torch.Tensor:
    """
    Vectorized per-group blocked swizzle for scales shaped [G, R, C].
    """
    if scales.ndim != 3:
        raise ValueError(f"Expected rank-3 scales [G, R, C], got {tuple(scales.shape)}")
    if scales.is_cuda and triton is not None:
        return _to_blocked_per_group_triton(scales)

    g, rows, cols = scales.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = scales
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (g, padded_rows, padded_cols),
            device=scales.device,
            dtype=scales.dtype,
        )
        padded[:, :rows, :cols] = scales

    blocks = padded.view(g, n_row_blocks, 128, n_col_blocks, 4).permute(0, 1, 3, 2, 4)
    rearranged = blocks.reshape(g, n_row_blocks * n_col_blocks, 4, 32, 4).transpose(2, 3)
    return rearranged.reshape(g, -1).contiguous()


def _quantize_to_mxfp8_torch(
    x: torch.Tensor,
    *,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_shape = x.shape
    x_blocks = x.reshape(*orig_shape[:-1], orig_shape[-1] // block_size, block_size)

    max_abs = torch.amax(torch.abs(x_blocks), dim=-1)
    max_abs = torch.clamp(max_abs, min=torch.finfo(torch.float32).tiny)

    largest_p2 = torch.floor(torch.log2(max_abs))
    scale_unbiased = largest_p2 - _F8E4M3_LARGEST_POW2
    scale_unbiased = torch.clamp(scale_unbiased, -_F8E8M0_EXP_BIAS, _F8E8M0_EXP_BIAS)
    scale_biased = (scale_unbiased + _F8E8M0_EXP_BIAS).to(torch.uint8)
    scales = scale_biased.view(torch.float8_e8m0fnu)

    dequant_scale = scales.to(torch.float32).unsqueeze(-1)
    qdata_hp = x_blocks.to(torch.float32) / dequant_scale
    qdata_hp = torch.nan_to_num(
        qdata_hp,
        nan=0.0,
        posinf=_F8E4M3_MAX,
        neginf=-_F8E4M3_MAX,
    )
    qdata_hp = torch.clamp(qdata_hp, min=-_F8E4M3_MAX, max=_F8E4M3_MAX)
    qdata = qdata_hp.to(torch.float8_e4m3fn)
    qdata = qdata.reshape(orig_shape).contiguous()
    scales = scales.reshape(orig_shape[0], -1).contiguous()
    return qdata, scales


def _quantize_to_mxfp8_triton(
    x: torch.Tensor,
    *,
    block_size: int = 32,
    out: Optional[torch.Tensor] = None,
    scales_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if triton is None or tl is None:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda:
        raise RuntimeError("Triton MXFP8 quantization requires CUDA")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported dtype for Triton MXFP8 quantization: {x.dtype}")

    rows, cols = x.shape
    if out is not None:
        qdata = out
    else:
        qdata = torch.empty((rows, cols), device=x.device, dtype=torch.float8_e4m3fn)
    if scales_out is not None:
        scales_u8 = scales_out.view(torch.uint8)
    else:
        scales_u8 = torch.empty((rows, cols // block_size), device=x.device, dtype=torch.uint8)

    def grid(meta):
        return (
            triton.cdiv(rows, meta["BLOCK_M"]),
            triton.cdiv(cols, meta["BLOCK_N"]),
        )

    _triton_mxfp8_quantize_dim0[grid](
        x,
        x.stride(0),
        x.stride(1),
        qdata,
        qdata.stride(0),
        qdata.stride(1),
        scales_u8,
        scales_u8.stride(0),
        scales_u8.stride(1),
        rows,
        cols,
        BLOCK_SIZE=block_size,
        USE_BF16_EXP=x.dtype == torch.bfloat16,
    )
    if scales_out is not None:
        return qdata, scales_out
    return qdata, scales_u8.view(torch.float8_e8m0fnu)


def _swiglu_quantize_rows_to_mxfp8_torch(
    up_gate: torch.Tensor,
    *,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = up_gate.shape[1] // 2
    up, gate = up_gate.split(hidden, dim=-1)
    h = up * torch.nn.functional.silu(gate)
    qdata, scales = quantize_to_mxfp8(h, block_size=block_size)
    return h, qdata, scales


def _swiglu_quantize_rows_to_mxfp8_triton(
    up_gate: torch.Tensor,
    *,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None or tl is None:
        raise RuntimeError("Triton is not available")
    if not up_gate.is_cuda:
        raise RuntimeError("Triton SwiGLU MXFP8 quantization requires CUDA")
    if up_gate.ndim != 2:
        raise ValueError(f"Expected rank-2 up_gate [M, 2H], got {tuple(up_gate.shape)}")
    if up_gate.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported dtype for Triton SwiGLU MXFP8 quantization: {up_gate.dtype}")
    if block_size != 32:
        raise ValueError(f"Only block_size=32 is supported (got {block_size})")
    if up_gate.shape[1] % 2 != 0:
        raise ValueError(
            f"up_gate last dim must be even for SwiGLU split, got {tuple(up_gate.shape)}"
        )

    up_gate_contig = up_gate if up_gate.is_contiguous() else up_gate.contiguous()
    rows = up_gate_contig.shape[0]
    hidden = up_gate_contig.shape[1] // 2
    if hidden % block_size != 0:
        raise ValueError(
            f"SwiGLU hidden dim must be divisible by {block_size}, got hidden={hidden}"
        )
    if _MXFP8_SWIGLU_BLOCK_N % block_size != 0:
        raise ValueError(
            f"SWIGLU BLOCK_N must be divisible by {block_size}, got {_MXFP8_SWIGLU_BLOCK_N}"
        )
    h = torch.empty((rows, hidden), device=up_gate_contig.device, dtype=up_gate_contig.dtype)
    qdata = torch.empty((rows, hidden), device=up_gate_contig.device, dtype=torch.float8_e4m3fn)
    scales_u8 = torch.empty(
        (rows, hidden // block_size), device=up_gate_contig.device, dtype=torch.uint8
    )

    grid = (
        triton.cdiv(rows, _MXFP8_SWIGLU_BLOCK_M),
        triton.cdiv(hidden, _MXFP8_SWIGLU_BLOCK_N),
    )
    _triton_swiglu_quantize_dim0[grid](
        up_gate_contig,
        up_gate_contig.stride(0),
        up_gate_contig.stride(1),
        h,
        h.stride(0),
        h.stride(1),
        qdata,
        qdata.stride(0),
        qdata.stride(1),
        scales_u8,
        scales_u8.stride(0),
        scales_u8.stride(1),
        rows,
        hidden,
        BLOCK_SIZE=block_size,
        BLOCK_M=_MXFP8_SWIGLU_BLOCK_M,
        BLOCK_N=_MXFP8_SWIGLU_BLOCK_N,
        EVEN_M=rows % _MXFP8_SWIGLU_BLOCK_M == 0,
        EVEN_N=hidden % _MXFP8_SWIGLU_BLOCK_N == 0,
        num_warps=_MXFP8_SWIGLU_NUM_WARPS,
        num_stages=_MXFP8_SWIGLU_NUM_STAGES,
    )
    return h, qdata, scales_u8.view(torch.float8_e8m0fnu)


def swiglu_quantize_rows_to_mxfp8(
    up_gate: torch.Tensor,
    *,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SwiGLU activation and quantize it to MXFP8 in one API.

    Returns:
      h: high-precision activation [M, H]
      qdata: float8_e4m3fn [M, H]
      scales: float8_e8m0fnu [M, H//block_size]
    """
    if up_gate.ndim != 2:
        raise ValueError(f"Expected rank-2 up_gate [M, 2H], got {tuple(up_gate.shape)}")
    if block_size != 32:
        raise ValueError(f"Only block_size=32 is supported (got {block_size})")
    if up_gate.shape[1] % 2 != 0:
        raise ValueError(f"up_gate last dim must be even, got {tuple(up_gate.shape)}")
    hidden = up_gate.shape[1] // 2
    if hidden % block_size != 0:
        raise ValueError(
            f"SwiGLU hidden dim must be divisible by {block_size}, got hidden={hidden}"
        )

    if up_gate.is_cuda:
        if triton is None:
            raise RuntimeError("Triton is required for CUDA SwiGLU MXFP8 quantization")
        return _swiglu_quantize_rows_to_mxfp8_triton(up_gate, block_size=block_size)

    return _swiglu_quantize_rows_to_mxfp8_torch(up_gate, block_size=block_size)


def _swiglu_quantize_rows_from_mxfp8_torch(
    up_gate_q: torch.Tensor,
    up_gate_scales: torch.Tensor,
    *,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    up_gate = dequantize_from_mxfp8(
        up_gate_q,
        up_gate_scales,
        block_size=block_size,
        out_dtype=torch.bfloat16,
    )
    return _swiglu_quantize_rows_to_mxfp8_torch(up_gate, block_size=block_size)


def _swiglu_quantize_rows_from_mxfp8_triton(
    up_gate_q: torch.Tensor,
    up_gate_scales: torch.Tensor,
    *,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None or tl is None:
        raise RuntimeError("Triton is not available")
    if not up_gate_q.is_cuda or not up_gate_scales.is_cuda:
        raise RuntimeError("Triton FP8 SwiGLU requires CUDA tensors")
    if up_gate_q.dtype != torch.float8_e4m3fn:
        raise ValueError(f"up_gate_q must be float8_e4m3fn, got {up_gate_q.dtype}")
    if up_gate_scales.dtype != torch.float8_e8m0fnu:
        raise ValueError(f"up_gate_scales must be float8_e8m0fnu, got {up_gate_scales.dtype}")
    if up_gate_q.ndim != 2 or up_gate_scales.ndim != 2:
        raise ValueError(
            "Expected rank-2 up_gate_q/up_gate_scales, "
            f"got {tuple(up_gate_q.shape)} and {tuple(up_gate_scales.shape)}"
        )
    if up_gate_q.shape[0] != up_gate_scales.shape[0]:
        raise ValueError(
            f"Row mismatch: q={tuple(up_gate_q.shape)} scales={tuple(up_gate_scales.shape)}"
        )
    if up_gate_q.shape[1] % 2 != 0:
        raise ValueError(
            f"up_gate_q last dim must be even for SwiGLU split, got {tuple(up_gate_q.shape)}"
        )
    if up_gate_q.shape[1] % block_size != 0:
        raise ValueError(
            f"up_gate_q last dim must be divisible by {block_size}, got {tuple(up_gate_q.shape)}"
        )
    if up_gate_scales.shape[1] != up_gate_q.shape[1] // block_size:
        raise ValueError(
            "up_gate_scales shape mismatch: "
            f"expected second dim {up_gate_q.shape[1] // block_size}, got {up_gate_scales.shape[1]}"
        )

    up_gate_q_contig = up_gate_q if up_gate_q.is_contiguous() else up_gate_q.contiguous()
    up_gate_scales_contig = (
        up_gate_scales if up_gate_scales.is_contiguous() else up_gate_scales.contiguous()
    )
    up_gate_scales_u8 = up_gate_scales_contig.view(torch.uint8)

    rows = up_gate_q_contig.shape[0]
    hidden = up_gate_q_contig.shape[1] // 2
    h = torch.empty((rows, hidden), device=up_gate_q_contig.device, dtype=torch.bfloat16)
    qdata = torch.empty((rows, hidden), device=up_gate_q_contig.device, dtype=torch.float8_e4m3fn)
    scales_u8 = torch.empty(
        (rows, hidden // block_size), device=up_gate_q_contig.device, dtype=torch.uint8
    )

    grid = (
        triton.cdiv(rows, _MXFP8_SWIGLU_FROM_FP8_BLOCK_M),
        triton.cdiv(hidden, _MXFP8_SWIGLU_FROM_FP8_BLOCK_N),
    )
    _triton_swiglu_quantize_from_mxfp8_dim0[grid](
        up_gate_q_contig,
        up_gate_q_contig.stride(0),
        up_gate_q_contig.stride(1),
        up_gate_scales_u8,
        up_gate_scales_u8.stride(0),
        up_gate_scales_u8.stride(1),
        h,
        h.stride(0),
        h.stride(1),
        qdata,
        qdata.stride(0),
        qdata.stride(1),
        scales_u8,
        scales_u8.stride(0),
        scales_u8.stride(1),
        rows,
        hidden,
        BLOCK_SIZE=block_size,
        BLOCK_M=_MXFP8_SWIGLU_FROM_FP8_BLOCK_M,
        BLOCK_N=_MXFP8_SWIGLU_FROM_FP8_BLOCK_N,
        num_warps=_MXFP8_SWIGLU_FROM_FP8_NUM_WARPS,
        num_stages=_MXFP8_SWIGLU_FROM_FP8_NUM_STAGES,
    )
    return h, qdata, scales_u8.view(torch.float8_e8m0fnu)


def swiglu_quantize_rows_from_mxfp8(
    up_gate_q: torch.Tensor,
    up_gate_scales: torch.Tensor,
    *,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SwiGLU from MXFP8 input and requantize output to MXFP8.

    Inputs:
      up_gate_q: float8_e4m3fn [M, 2H]
      up_gate_scales: float8_e8m0fnu [M, 2H//block_size]

    Returns:
      h: bfloat16 [M, H]
      qdata: float8_e4m3fn [M, H]
      scales: float8_e8m0fnu [M, H//block_size]
    """
    if block_size != 32:
        raise ValueError(f"Only block_size=32 is supported (got {block_size})")
    if up_gate_q.ndim != 2 or up_gate_scales.ndim != 2:
        raise ValueError(
            "Expected rank-2 up_gate_q/up_gate_scales, "
            f"got {tuple(up_gate_q.shape)} and {tuple(up_gate_scales.shape)}"
        )

    if up_gate_q.is_cuda and up_gate_scales.is_cuda:
        if triton is None:
            raise RuntimeError("Triton is required for CUDA SwiGLU MXFP8 dequant+quant")
        return _swiglu_quantize_rows_from_mxfp8_triton(
            up_gate_q,
            up_gate_scales,
            block_size=block_size,
        )

    return _swiglu_quantize_rows_from_mxfp8_torch(
        up_gate_q,
        up_gate_scales,
        block_size=block_size,
    )


def _quantize_to_mxfp8_te(
    x: torch.Tensor,
    *,
    block_size: int,
    out: Optional[torch.Tensor],
    scales_out: Optional[torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if (
        block_size != 32
        or not x.is_cuda
        or x.ndim != 2
        or not x.is_contiguous()
        or x.dtype not in {torch.bfloat16, torch.float16, torch.float32}
    ):
        return None

    rows, cols = x.shape
    if rows % 32 != 0 or cols % block_size != 0:
        return None
    if not _te_mxfp8_compact_scale_layout(rows, cols, block_size):
        return None

    state = _get_te_mxfp8_state()
    if state is None:
        return None
    tex, MXFP8Tensor, quantizer = state

    if out is None and scales_out is None:
        te_tensor = quantizer(x)
        qdata = te_tensor._rowwise_data.view(torch.float8_e4m3fn)
        scales = te_tensor._rowwise_scale_inv.view(torch.float8_e8m0fnu)
        return qdata, scales[:rows, : cols // block_size]

    if out is None or scales_out is None:
        return None
    if not out.is_contiguous() or not scales_out.is_contiguous():
        return None

    te_tensor = MXFP8Tensor(
        shape=tuple(x.shape),
        dtype=x.dtype,
        rowwise_data=out.view(torch.uint8),
        rowwise_scale_inv=scales_out.view(torch.uint8),
        columnwise_data=None,
        columnwise_scale_inv=None,
        fp8_dtype=tex.DType.kFloat8E4M3,
        quantizer=quantizer,
    )
    quantizer.update_quantized(x, te_tensor)
    return out, scales_out


@nvtx.annotate("quantize_to_mxfp8", color="red")
def quantize_to_mxfp8(
    x: torch.Tensor,
    *,
    block_size: int = 32,
    out: Optional[torch.Tensor] = None,
    scales_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D high-precision tensor to MXFP8.

    Returns:
      qdata: float8_e4m3fn [M, K]
      scales: float8_e8m0fnu [M, K//block_size]
    """
    if x.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor, got shape={tuple(x.shape)}")
    if block_size != 32:
        raise ValueError(f"Only block_size=32 is supported (got {block_size})")
    if x.shape[-1] % block_size != 0:
        raise ValueError(
            f"Last dim must be divisible by {block_size} for MXFP8 quantization, got {tuple(x.shape)}"
        )
    expected_scales_shape = (x.shape[0], x.shape[1] // block_size)
    if out is not None:
        if tuple(out.shape) != tuple(x.shape):
            raise ValueError(
                f"out shape mismatch: expected {tuple(x.shape)}, got {tuple(out.shape)}"
            )
        if out.dtype != torch.float8_e4m3fn:
            raise ValueError(f"out dtype mismatch: expected {torch.float8_e4m3fn}, got {out.dtype}")
        if out.device != x.device:
            raise ValueError(f"out device mismatch: expected {x.device}, got {out.device}")
    if scales_out is not None:
        if tuple(scales_out.shape) != expected_scales_shape:
            raise ValueError(
                f"scales_out shape mismatch: expected {expected_scales_shape}, got {tuple(scales_out.shape)}"
            )
        if scales_out.dtype != torch.float8_e8m0fnu:
            raise ValueError(
                f"scales_out dtype mismatch: expected {torch.float8_e8m0fnu}, got {scales_out.dtype}"
            )
        if scales_out.device != x.device:
            raise ValueError(
                f"scales_out device mismatch: expected {x.device}, got {scales_out.device}"
            )
        if not scales_out.is_contiguous():
            raise ValueError("scales_out must be contiguous")
    if out is not None and not out.is_contiguous():
        raise ValueError("out must be contiguous")

    # CUDA fast path: fused Triton quantization avoids eager pointwise op chains.
    if x.is_cuda:
        if _mxfp8_backend_wants_te("Q"):
            te_result = _quantize_to_mxfp8_te(
                x,
                block_size=block_size,
                out=out,
                scales_out=scales_out,
            )
            if te_result is not None:
                return te_result
        if triton is None:
            raise RuntimeError("Triton is required for CUDA MXFP8 quantization")
        return _quantize_to_mxfp8_triton(
            x,
            block_size=block_size,
            out=out,
            scales_out=scales_out,
        )

    qdata, scales = _quantize_to_mxfp8_torch(x, block_size=block_size)
    if out is not None:
        out.copy_(qdata)
        qdata = out
    if scales_out is not None:
        scales_out.copy_(scales)
        scales = scales_out
    return qdata, scales


def _dequantize_from_mxfp8_te(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if (
        out is not None
        or block_size != 32
        or not qdata.is_cuda
        or not scales.is_cuda
        or not qdata.is_contiguous()
        or not scales.is_contiguous()
        or qdata.dtype != torch.float8_e4m3fn
        or scales.dtype != torch.float8_e8m0fnu
        or out_dtype not in {torch.bfloat16, torch.float16, torch.float32}
    ):
        return None

    rows, cols = qdata.shape
    if rows % 32 != 0 or cols % block_size != 0:
        return None
    if not _te_mxfp8_compact_scale_layout(rows, cols, block_size):
        return None

    state = _get_te_mxfp8_state()
    if state is None:
        return None
    tex, MXFP8Tensor, quantizer = state

    te_tensor = MXFP8Tensor(
        shape=tuple(qdata.shape),
        dtype=out_dtype,
        rowwise_data=qdata.view(torch.uint8),
        rowwise_scale_inv=scales.view(torch.uint8),
        columnwise_data=None,
        columnwise_scale_inv=None,
        fp8_dtype=tex.DType.kFloat8E4M3,
        quantizer=quantizer,
    )
    return te_tensor.dequantize(dtype=out_dtype)


@nvtx.annotate("dequantize_from_mxfp8", color="red")
def dequantize_from_mxfp8(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Dequantize MXFP8 qdata/scales back to high precision.

    qdata shape: [M, K]
    scales shape: [M, K//block_size]
    """
    if qdata.ndim != 2 or scales.ndim != 2:
        raise ValueError(
            f"Expected rank-2 qdata/scales, got qdata={tuple(qdata.shape)} scales={tuple(scales.shape)}"
        )
    if block_size != 32:
        raise ValueError(f"Only block_size=32 is supported (got {block_size})")
    if qdata.shape[0] != scales.shape[0]:
        raise ValueError(
            f"qdata/scales row mismatch: qdata={tuple(qdata.shape)} scales={tuple(scales.shape)}"
        )
    if qdata.shape[1] % block_size != 0:
        raise ValueError(
            f"qdata last dim must be divisible by {block_size}, got {tuple(qdata.shape)}"
        )
    if scales.shape[1] != qdata.shape[1] // block_size:
        raise ValueError(
            f"scales second dim must be K//{block_size}, got qdata={tuple(qdata.shape)} scales={tuple(scales.shape)}"
        )

    m, k = qdata.shape

    if out is not None:
        if tuple(out.shape) != (m, k):
            raise ValueError(f"out shape mismatch: expected {(m, k)}, got {tuple(out.shape)}")
        if out.dtype != out_dtype:
            raise ValueError(f"out dtype mismatch: expected {out_dtype}, got {out.dtype}")
        if out.device != qdata.device:
            raise ValueError(f"out device mismatch: expected {qdata.device}, got {out.device}")
        out_tensor = out
    else:
        if _mxfp8_backend_wants_te("DQ"):
            te_out = _dequantize_from_mxfp8_te(
                qdata,
                scales,
                block_size=block_size,
                out_dtype=out_dtype,
                out=out,
            )
            if te_out is not None:
                return te_out
        out_tensor = torch.empty((m, k), device=qdata.device, dtype=out_dtype)

    use_triton = qdata.is_cuda and scales.is_cuda and triton is not None
    if qdata.is_cuda and scales.is_cuda and triton is None:
        raise RuntimeError("Triton is required for CUDA MXFP8 dequantization")
    if use_triton:
        q_contig = qdata if qdata.is_contiguous() else qdata.contiguous()
        scales_contig = scales if scales.is_contiguous() else scales.contiguous()
        scales_u8 = scales_contig.view(torch.uint8)
        out_contig = out_tensor if out_tensor.is_contiguous() else out_tensor.contiguous()

        def grid(meta):
            return (
                triton.cdiv(m, meta["BLOCK_M"]),
                triton.cdiv(k, meta["BLOCK_N"]),
            )

        _triton_mxfp8_dequantize_dim0[grid](
            q_contig,
            q_contig.stride(0),
            q_contig.stride(1),
            scales_u8,
            scales_u8.stride(0),
            scales_u8.stride(1),
            out_contig,
            out_contig.stride(0),
            out_contig.stride(1),
            m,
            k,
            BLOCK_SIZE=block_size,
        )
        if out_contig is not out_tensor:
            out_tensor.copy_(out_contig)
        return out_tensor

    q_blocks = qdata.reshape(m, k // block_size, block_size)
    x = q_blocks.to(torch.float32) * scales.to(torch.float32).unsqueeze(-1)
    x = x.reshape(m, k).to(out_dtype)
    out_tensor.copy_(x)
    return out_tensor


def _reduce_gathered_rows_from_mxfp8_triton(
    gathered_q: torch.Tensor,
    gathered_scales: torch.Tensor,
    *,
    probs: Optional[torch.Tensor],
    valid_mask: Optional[torch.Tensor],
    block_size: int = 32,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if triton is None or tl is None:
        raise RuntimeError("Triton is not available")
    if not gathered_q.is_cuda or not gathered_scales.is_cuda:
        raise RuntimeError("Triton reduce requires CUDA tensors")
    if gathered_q.ndim != 3 or gathered_scales.ndim != 3:
        raise ValueError(
            "Expected gathered tensors rank-3 [N,K,D] and [N,K,D//32], "
            f"got {tuple(gathered_q.shape)} and {tuple(gathered_scales.shape)}"
        )

    n, top_k, d_model = gathered_q.shape
    if gathered_scales.shape != (n, top_k, d_model // block_size):
        raise ValueError(
            "gathered_scales shape mismatch for gathered_q: "
            f"q={tuple(gathered_q.shape)} scales={tuple(gathered_scales.shape)}"
        )

    q_contig = gathered_q if gathered_q.is_contiguous() else gathered_q.contiguous()
    scales_contig = (
        gathered_scales if gathered_scales.is_contiguous() else gathered_scales.contiguous()
    )
    scales_u8 = scales_contig.view(torch.uint8)
    probs_tensor = probs
    if probs_tensor is None:
        probs_tensor = torch.empty((1, 1), device=q_contig.device, dtype=torch.float32)
    elif not probs_tensor.is_contiguous():
        probs_tensor = probs_tensor.contiguous()
    valid_mask_tensor = valid_mask
    if valid_mask_tensor is None:
        valid_mask_tensor = torch.empty((1, 1), device=q_contig.device, dtype=torch.bool)
    elif not valid_mask_tensor.is_contiguous():
        valid_mask_tensor = valid_mask_tensor.contiguous()

    if out is None:
        out_tensor = torch.empty((n, d_model), device=q_contig.device, dtype=torch.float32)
    else:
        if tuple(out.shape) != (n, d_model):
            raise ValueError(f"out shape mismatch: expected {(n, d_model)}, got {tuple(out.shape)}")
        if out.device != q_contig.device:
            raise ValueError(f"out device mismatch: expected {q_contig.device}, got {out.device}")
        if not out.dtype.is_floating_point:
            raise ValueError(f"out must have floating dtype, got {out.dtype}")
        out_tensor = out

    block_m = _MXFP8_REDUCE_BLOCK_M
    block_n = _MXFP8_REDUCE_BLOCK_N
    grid = (
        triton.cdiv(n, block_m),
        triton.cdiv(d_model, block_n),
    )
    _triton_mxfp8_reduce_gathered_dim1[grid](
        q_contig,
        q_contig.stride(0),
        q_contig.stride(1),
        q_contig.stride(2),
        scales_u8,
        scales_u8.stride(0),
        scales_u8.stride(1),
        scales_u8.stride(2),
        probs_tensor,
        probs_tensor.stride(0),
        probs_tensor.stride(1),
        valid_mask_tensor,
        valid_mask_tensor.stride(0),
        valid_mask_tensor.stride(1),
        out_tensor,
        out_tensor.stride(0),
        out_tensor.stride(1),
        n,
        top_k,
        d_model,
        BLOCK_SIZE=block_size,
        HAS_PROBS=probs is not None,
        HAS_VALID_MASK=valid_mask is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=_MXFP8_REDUCE_NUM_WARPS,
        num_stages=_MXFP8_REDUCE_NUM_STAGES,
    )
    return out_tensor


@nvtx.annotate("reduce_gathered_rows_from_mxfp8")
def reduce_gathered_rows_from_mxfp8(
    gathered_q: torch.Tensor,
    gathered_scales: torch.Tensor,
    out: torch.Tensor,
    *,
    probs: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    block_size: int = 32,
    gathered_out: Optional[torch.Tensor] = None,
) -> None:
    """
    Reduce gathered MXFP8 routes [N, K, D] into out [N, D].

    If `gathered_out` is provided, it receives dequantized [N, K, D] in `out` dtype.
    """
    if gathered_q.ndim != 3 or gathered_scales.ndim != 3:
        raise ValueError(
            "Expected gathered_q/gathered_scales rank-3 [N,K,D]/[N,K,D//32], "
            f"got {tuple(gathered_q.shape)} and {tuple(gathered_scales.shape)}"
        )
    if block_size != 32:
        raise ValueError(f"Only block_size=32 is supported (got {block_size})")

    n, top_k, d_model = gathered_q.shape
    if out.shape != (n, d_model):
        raise ValueError(f"out shape mismatch: expected {(n, d_model)}, got {tuple(out.shape)}")
    if gathered_scales.shape != (n, top_k, d_model // block_size):
        raise ValueError(
            "gathered_scales shape mismatch for gathered_q: "
            f"q={tuple(gathered_q.shape)} scales={tuple(gathered_scales.shape)}"
        )
    if probs is not None and probs.shape != (n, top_k):
        raise ValueError(f"probs shape mismatch: expected {(n, top_k)}, got {tuple(probs.shape)}")
    if valid_mask is not None and valid_mask.shape != (n, top_k):
        raise ValueError(
            f"valid_mask shape mismatch: expected {(n, top_k)}, got {tuple(valid_mask.shape)}"
        )
    if gathered_out is not None and gathered_out.shape != gathered_q.shape:
        raise ValueError(
            f"gathered_out shape mismatch: expected {tuple(gathered_q.shape)}, got {tuple(gathered_out.shape)}"
        )

    use_triton_reduce = (
        gathered_out is None
        and gathered_q.is_cuda
        and gathered_scales.is_cuda
        and out.is_cuda
        and triton is not None
    )
    if gathered_out is None and gathered_q.is_cuda and gathered_scales.is_cuda and triton is None:
        raise RuntimeError("Triton is required for CUDA MXFP8 gather-reduce")

    if use_triton_reduce:
        _reduce_gathered_rows_from_mxfp8_triton(
            gathered_q,
            gathered_scales,
            probs=probs,
            valid_mask=valid_mask,
            block_size=block_size,
            out=out,
        )
        return
    else:
        assert False, "Deprecated"

    gathered_hp = dequantize_from_mxfp8(
        gathered_q.reshape(n * top_k, d_model),
        gathered_scales.reshape(n * top_k, d_model // block_size),
        block_size=block_size,
        out_dtype=torch.float32,
    ).view(n, top_k, d_model)

    if gathered_out is not None:
        gathered_out.copy_(gathered_hp.to(gathered_out.dtype))

    if valid_mask is not None:
        gathered_hp = gathered_hp * valid_mask.to(torch.float32).unsqueeze(-1)

    if probs is not None:
        gathered_hp = gathered_hp * probs.to(torch.float32).unsqueeze(-1)

    out.copy_(gathered_hp.sum(dim=1).to(out.dtype))


def dot_gathered_rows_mxfp8_with_grad(
    gathered_q: torch.Tensor,
    gathered_scales: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    block_size: int = 32,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Compute per-route dot products for grad(probs):
      out[n, k] = dot(dequant(gathered_q[n, k, :]), grad_out[n, :])
    Invalid routes are masked out via `valid_mask`.
    """
    if gathered_q.ndim != 3 or gathered_scales.ndim != 3:
        raise ValueError(
            "Expected gathered_q/gathered_scales rank-3 [N,K,D]/[N,K,D//32], "
            f"got {tuple(gathered_q.shape)} and {tuple(gathered_scales.shape)}"
        )
    if grad_out.ndim != 2:
        raise ValueError(f"Expected grad_out rank-2 [N,D], got {tuple(grad_out.shape)}")
    if block_size != 32:
        raise ValueError(f"Only block_size=32 is supported (got {block_size})")

    n, top_k, d_model = gathered_q.shape
    if gathered_scales.shape != (n, top_k, d_model // block_size):
        raise ValueError(
            "gathered_scales shape mismatch for gathered_q: "
            f"q={tuple(gathered_q.shape)} scales={tuple(gathered_scales.shape)}"
        )
    if grad_out.shape != (n, d_model):
        raise ValueError(
            f"grad_out shape mismatch: expected {(n, d_model)}, got {tuple(grad_out.shape)}"
        )
    if valid_mask.shape != (n, top_k):
        raise ValueError(
            f"valid_mask shape mismatch: expected {(n, top_k)}, got {tuple(valid_mask.shape)}"
        )

    use_triton = (
        gathered_q.is_cuda
        and gathered_scales.is_cuda
        and grad_out.is_cuda
        and valid_mask.is_cuda
        and triton is not None
    )
    if (
        gathered_q.is_cuda
        and gathered_scales.is_cuda
        and grad_out.is_cuda
        and valid_mask.is_cuda
        and triton is None
    ):
        raise RuntimeError("Triton is required for CUDA MXFP8 gathered-dot")
    if use_triton:
        q_contig = gathered_q if gathered_q.is_contiguous() else gathered_q.contiguous()
        scales_contig = (
            gathered_scales if gathered_scales.is_contiguous() else gathered_scales.contiguous()
        )
        grad_contig = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
        valid_contig = valid_mask if valid_mask.is_contiguous() else valid_mask.contiguous()
        scales_u8 = scales_contig.view(torch.uint8)
        out = torch.empty((n, top_k), device=q_contig.device, dtype=torch.float32)
        block_m = _MXFP8_DOT_BLOCK_M
        block_n = _MXFP8_DOT_BLOCK_N
        grid = (triton.cdiv(n, block_m), top_k)
        _triton_mxfp8_dot_gathered_with_grad[grid](
            q_contig,
            q_contig.stride(0),
            q_contig.stride(1),
            q_contig.stride(2),
            scales_u8,
            scales_u8.stride(0),
            scales_u8.stride(1),
            scales_u8.stride(2),
            grad_contig,
            grad_contig.stride(0),
            grad_contig.stride(1),
            valid_contig,
            valid_contig.stride(0),
            valid_contig.stride(1),
            out,
            out.stride(0),
            out.stride(1),
            n,
            top_k,
            d_model,
            BLOCK_SIZE=block_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=_MXFP8_DOT_NUM_WARPS,
            num_stages=_MXFP8_DOT_NUM_STAGES,
        )
        if out_dtype != torch.float32:
            out = out.to(dtype=out_dtype)
        return out

    gathered_hp = dequantize_from_mxfp8(
        gathered_q.reshape(n * top_k, d_model),
        gathered_scales.reshape(n * top_k, d_model // block_size),
        block_size=block_size,
        out_dtype=torch.float32,
    ).view(n, top_k, d_model)
    gathered_hp = gathered_hp * valid_mask.to(torch.float32).unsqueeze(-1)
    dot = (gathered_hp * grad_out.to(torch.float32).unsqueeze(1)).sum(dim=-1)
    if out_dtype != torch.float32:
        dot = dot.to(dtype=out_dtype)
    return dot


def quantize_grouped_2d_to_mxfp8_blocked(
    x: torch.Tensor,
    offs: torch.Tensor,
    *,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize grouped 2D input for scaled_grouped_mm (2D-3D).

    Returns:
      qdata: [M, K] float8_e4m3fn
      scales_blocked: [>=M, K//block_size] float8_e8m0fnu in grouped blocked layout packing.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x to be rank-2, got {tuple(x.shape)}")
    if offs.ndim != 1:
        raise ValueError(f"Expected offs to be rank-1, got {tuple(offs.shape)}")
    if x.shape[1] % block_size != 0:
        raise ValueError(f"x last dim must be divisible by {block_size}, got {tuple(x.shape)}")

    k_blocks = x.shape[1] // block_size
    if k_blocks % 4 != 0:
        raise ValueError(
            f"MXFP8 blocked scales require K//{block_size} divisible by 4, got {k_blocks}"
        )

    if x.is_cuda and offs.device.type != "cuda":
        raise RuntimeError(
            "quantize_grouped_2d_to_mxfp8_blocked requires CUDA offs when x is CUDA "
            "(CPU offs would reintroduce host-sync behavior)."
        )

    if offs.dtype != torch.int32:
        offs = offs.to(dtype=torch.int32)
    if offs.device != x.device:
        offs = offs.to(device=x.device)

    # Keep capacity/static shape for mat_a and avoid active-row compaction.
    qdata, scales = quantize_to_mxfp8(x, block_size=block_size)

    if x.is_cuda:
        scales_blocked = _to_blocked_m_groups_triton(scales, offs)
    else:
        scales_blocked = _to_blocked_m_groups_fallback(scales, offs)

    return qdata, scales_blocked


def _grouped_blocked_scale_shape(
    rows: int,
    cols: int,
    num_groups: int,
    *,
    block_size: int,
) -> tuple[int, int]:
    scale_cols = cols // block_size
    padded_scale_cols = ceil_div(scale_cols, 4) * 4
    # Keep the same upper-bound shape as _to_blocked_m_groups_triton. The
    # exact per-group padded row count depends on device-side offs; this avoids
    # host inspection and leaves unused capacity rows zero.
    return rows + num_groups * 128, padded_scale_cols


def _quantize_grouped_2d_to_mxfp8_blocked_fused_triton(
    x: torch.Tensor,
    offs: torch.Tensor,
    *,
    block_size: int = 32,
    out: Optional[torch.Tensor] = None,
    blocked_scales_out: Optional[torch.Tensor] = None,
    zero_unwritten_tail: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for fused grouped MXFP8 quantize+swizzle")
    if not x.is_cuda:
        raise RuntimeError("fused grouped MXFP8 quantize+swizzle requires CUDA")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported dtype for fused grouped MXFP8 quantization: {x.dtype}")

    rows, cols = x.shape
    num_groups = int(offs.shape[0])
    blocked_shape = _grouped_blocked_scale_shape(
        rows,
        cols,
        num_groups,
        block_size=block_size,
    )

    qdata = out if out is not None else torch.empty_like(x, dtype=torch.float8_e4m3fn)
    if tuple(qdata.shape) != tuple(x.shape):
        raise ValueError(f"out shape mismatch: expected {tuple(x.shape)}, got {tuple(qdata.shape)}")
    if qdata.dtype != torch.float8_e4m3fn:
        raise ValueError(f"out dtype mismatch: expected {torch.float8_e4m3fn}, got {qdata.dtype}")
    if qdata.device != x.device:
        raise ValueError(f"out device mismatch: expected {x.device}, got {qdata.device}")
    if not qdata.is_contiguous():
        raise ValueError("out must be contiguous")

    if blocked_scales_out is None and zero_unwritten_tail:
        blocked_scales = torch.zeros(blocked_shape, device=x.device, dtype=torch.float8_e8m0fnu)
    elif blocked_scales_out is None:
        blocked_scales = torch.empty(blocked_shape, device=x.device, dtype=torch.float8_e8m0fnu)
    else:
        blocked_scales = blocked_scales_out
        if tuple(blocked_scales.shape) != blocked_shape:
            raise ValueError(
                f"blocked_scales_out shape mismatch: expected {blocked_shape}, got {tuple(blocked_scales.shape)}"
            )
        if blocked_scales.dtype != torch.float8_e8m0fnu:
            raise ValueError(
                f"blocked_scales_out dtype mismatch: expected {torch.float8_e8m0fnu}, got {blocked_scales.dtype}"
            )
        if blocked_scales.device != x.device:
            raise ValueError(
                f"blocked_scales_out device mismatch: expected {x.device}, got {blocked_scales.device}"
            )
        if not blocked_scales.is_contiguous():
            raise ValueError("blocked_scales_out must be contiguous")
        if zero_unwritten_tail:
            blocked_scales.zero_()

    if offs.dtype != torch.int32:
        offs = offs.to(dtype=torch.int32)
    if offs.device != x.device:
        offs = offs.to(device=x.device)

    grid = (
        triton.cdiv(rows, _MXFP8_Q_BLOCK_M),
        triton.cdiv(cols, _MXFP8_Q_BLOCK_N),
    )
    _triton_mxfp8_quantize_grouped_blocked_tiled_dim0[grid](
        x,
        x.stride(0),
        x.stride(1),
        qdata,
        qdata.stride(0),
        qdata.stride(1),
        blocked_scales.view(torch.uint8),
        blocked_scales.stride(0),
        offs,
        rows,
        cols,
        BLOCK_SIZE=block_size,
        BLOCK_M=_MXFP8_Q_BLOCK_M,
        BLOCK_N=_MXFP8_Q_BLOCK_N,
        NUM_GROUPS=num_groups,
        USE_BF16_EXP=x.dtype == torch.bfloat16,
        num_warps=_MXFP8_Q_NUM_WARPS,
        num_stages=_MXFP8_Q_NUM_STAGES,
    )
    return qdata, blocked_scales


def quantize_grouped_2d_to_mxfp8_blocked_fused(
    x: torch.Tensor,
    offs: torch.Tensor,
    *,
    block_size: int = 32,
    out: Optional[torch.Tensor] = None,
    blocked_scales_out: Optional[torch.Tensor] = None,
    zero_unwritten_tail: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize grouped 2D input and write scales directly in grouped blocked layout.

    This is equivalent to quantize_rows_to_mxfp8(x) followed by
    grouped_scales_to_mxfp8_blocked(scales, offs), but fuses the scale swizzle
    into the quantization kernel for CUDA inputs.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x to be rank-2, got {tuple(x.shape)}")
    if offs.ndim != 1:
        raise ValueError(f"Expected offs to be rank-1, got {tuple(offs.shape)}")
    if block_size != 32:
        raise ValueError(f"Only block_size=32 is supported (got {block_size})")
    if x.shape[1] % block_size != 0:
        raise ValueError(f"x last dim must be divisible by {block_size}, got {tuple(x.shape)}")
    k_blocks = x.shape[1] // block_size
    if k_blocks % 4 != 0:
        raise ValueError(
            f"MXFP8 blocked scales require K//{block_size} divisible by 4, got {k_blocks}"
        )

    if x.is_cuda:
        return _quantize_grouped_2d_to_mxfp8_blocked_fused_triton(
            x,
            offs,
            block_size=block_size,
            out=out,
            blocked_scales_out=blocked_scales_out,
            zero_unwritten_tail=zero_unwritten_tail,
        )

    qdata, scales_blocked = quantize_grouped_2d_to_mxfp8_blocked(
        x,
        offs,
        block_size=block_size,
    )
    if out is not None:
        out.copy_(qdata)
        qdata = out
    if blocked_scales_out is not None:
        blocked_scales_out.copy_(scales_blocked)
        scales_blocked = blocked_scales_out
    return qdata, scales_blocked


def grouped_scales_to_mxfp8_blocked(
    scales: torch.Tensor,
    offs: torch.Tensor,
    *,
    zero_unwritten_tail: bool = True,
) -> torch.Tensor:
    """
    Convert grouped row-wise MXFP8 scales [M, K//32] into grouped blocked layout.
    """
    if scales.ndim != 2:
        raise ValueError(f"Expected scales rank-2 [M, K//32], got {tuple(scales.shape)}")
    if offs.ndim != 1:
        raise ValueError(f"Expected offs rank-1, got {tuple(offs.shape)}")
    if offs.dtype != torch.int32:
        offs = offs.to(dtype=torch.int32)
    if offs.device != scales.device:
        offs = offs.to(device=scales.device)

    if scales.is_cuda:
        return _to_blocked_m_groups_triton(
            scales,
            offs,
            zero_unwritten_tail=zero_unwritten_tail,
        )
    return _to_blocked_m_groups_fallback(scales, offs)


def quantize_grouped_weight_3d_to_mxfp8_blocked(
    mat_b: torch.Tensor,
    *,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize grouped 3D RHS operand [G, K, N] for scaled_grouped_mm.

    Returns:
      qdata: [G, K, N] float8_e4m3fn
      scales_blocked: [G, blocked_scale_K * blocked_scale_N] float8_e8m0fnu
    """
    if mat_b.ndim != 3:
        raise ValueError(f"Expected mat_b rank-3 [G, K, N], got {tuple(mat_b.shape)}")

    g, k, n = mat_b.shape
    if k % block_size != 0:
        raise ValueError(
            f"mat_b K dim must be divisible by {block_size}, got shape={tuple(mat_b.shape)}"
        )

    # Quantize non-transposed expert weights [N, K], then expose [K, N] as
    # transposed views so trailing dims are already column-major (stride [1, K]).
    mat_b_nk = mat_b.transpose(-2, -1)  # [G, N, K]
    if mat_b_nk.is_contiguous():
        q_nk, s_nk = quantize_to_mxfp8(mat_b_nk.reshape(g * n, k), block_size=block_size)
        q_nk = q_nk.reshape(g, n, k)
        s_nk = s_nk.reshape(g, n, k // block_size)
    else:
        q_parts = []
        s_parts = []
        for i in range(g):
            q_i, s_i = quantize_to_mxfp8(mat_b_nk[i], block_size=block_size)
            q_parts.append(q_i)
            s_parts.append(s_i)
        q_nk = torch.stack(q_parts, dim=0)
        s_nk = torch.stack(s_parts, dim=0)

    qdata = q_nk.transpose(-2, -1)  # [G, K, N], column-major trailing dims
    scales_blocked = _to_blocked_per_group(s_nk)  # [G, ...]
    return qdata, scales_blocked


def quantize_grouped_weight_3d_to_mxfp8_unblocked(
    mat_b: torch.Tensor,
    *,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize grouped 3D RHS operand [G, K, N] to:
      qdata: [G, K, N] float8_e4m3fn (column-major trailing dims)
      scales_nk: [G, N, K//block_size] float8_e8m0fnu (unblocked, non-swizzled)
    """
    if mat_b.ndim != 3:
        raise ValueError(f"Expected mat_b rank-3 [G, K, N], got {tuple(mat_b.shape)}")

    g, k, n = mat_b.shape
    if k % block_size != 0:
        raise ValueError(
            f"mat_b K dim must be divisible by {block_size}, got shape={tuple(mat_b.shape)}"
        )

    # Quantize non-transposed expert weights [N, K], then expose [K, N] as
    # transposed views so trailing dims are column-major (stride [1, K]).
    mat_b_nk = mat_b.transpose(-2, -1)  # [G, N, K]
    if mat_b_nk.is_contiguous():
        q_nk, s_nk = quantize_to_mxfp8(mat_b_nk.reshape(g * n, k), block_size=block_size)
        q_nk = q_nk.reshape(g, n, k)
        s_nk = s_nk.reshape(g, n, k // block_size)
    else:
        q_parts = []
        s_parts = []
        for i in range(g):
            q_i, s_i = quantize_to_mxfp8(mat_b_nk[i], block_size=block_size)
            q_parts.append(q_i)
            s_parts.append(s_i)
        q_nk = torch.stack(q_parts, dim=0)
        s_nk = torch.stack(s_parts, dim=0)

    qdata = q_nk.transpose(-2, -1)  # [G, K, N]
    return qdata, s_nk.contiguous()


def quantize_rows_to_mxfp8(
    x: torch.Tensor,
    *,
    block_size: int = 32,
    out: Optional[torch.Tensor] = None,
    scales_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience alias used by rowwise comm paths."""
    return quantize_to_mxfp8(
        x,
        block_size=block_size,
        out=out,
        scales_out=scales_out,
    )


@torch.compiler.disable
def quantize_row_halves_to_mxfp8(
    x: torch.Tensor,
    *,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a row-major [M, 2H] tensor as two [M, H] halves.

    This is equivalent to quantize_rows_to_mxfp8(x) when H is block-aligned,
    since MXFP8 scales are independent 32-column blocks. Keep this helper out
    of Dynamo/Inductor: tracing the non-contiguous FP8 half views can make
    Inductor generate an fp8 masked-load fill constant that Triton cannot lower.
    The useful work is still done by the two explicit Triton quantize kernels.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor, got shape={tuple(x.shape)}")
    if block_size != 32:
        raise ValueError(f"Only block_size=32 is supported (got {block_size})")
    if x.shape[1] % 2 != 0:
        raise ValueError(f"Expected even last dim for split quantization, got {tuple(x.shape)}")

    hidden = x.shape[1] // 2
    if hidden % block_size != 0:
        raise ValueError(f"Split dim must be divisible by {block_size}, got hidden={hidden}")
    if not x.is_cuda or triton is None or _mxfp8_backend_wants_te("Q"):
        return quantize_rows_to_mxfp8(x, block_size=block_size)

    rows = x.shape[0]
    qdata = torch.empty((rows, 2 * hidden), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (rows, (2 * hidden) // block_size),
        device=x.device,
        dtype=torch.float8_e8m0fnu,
    )
    scale_hidden = hidden // block_size
    _quantize_to_mxfp8_triton(
        x[:, :hidden],
        block_size=block_size,
        out=qdata[:, :hidden],
        scales_out=scales[:, :scale_hidden],
    )
    _quantize_to_mxfp8_triton(
        x[:, hidden:],
        block_size=block_size,
        out=qdata[:, hidden:],
        scales_out=scales[:, scale_hidden:],
    )
    return qdata, scales


def dequantize_rows_from_mxfp8(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convenience alias used by rowwise comm paths."""
    return dequantize_from_mxfp8(
        qdata,
        scales,
        block_size=block_size,
        out_dtype=out_dtype,
        out=out,
    )
