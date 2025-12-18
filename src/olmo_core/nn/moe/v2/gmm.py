import torch
import triton
import triton.language as tl

# (same kernel as before; unchanged)
@triton.jit
def _moe_gmm_fwd_kernel(
    A_ptr, B_ptr, BS_ptr, C_ptr,
    M_total: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_be: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_block = tl.program_id(0)
    pid_n = tl.program_id(1)

    remaining = pid_block
    row_prefix = 0
    found = 0
    expert_id = 0
    block_in_expert = 0
    expert_row_start = 0
    expert_m = 0

    for e in tl.static_range(0, E):
        m_e = tl.load(BS_ptr + e).to(tl.int32)
        blocks_e = tl.cdiv(m_e, BLOCK_M)

        found_prev = found
        take = (found_prev == 0) & (remaining < blocks_e)

        expert_id = tl.where(take, e, expert_id)
        block_in_expert = tl.where(take, remaining, block_in_expert)
        expert_row_start = tl.where(take, row_prefix, expert_row_start)
        expert_m = tl.where(take, m_e, expert_m)

        found = tl.where(take, 1, found_prev)
        remaining = remaining - tl.where(found_prev == 0, blocks_e, 0)
        row_prefix += m_e

    if found == 0:
        return

    m_start = expert_row_start + block_in_expert * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    row_end = expert_row_start + expert_m
    m_mask = (offs_m < row_end) & (offs_m < M_total)
    n_mask = offs_n < N

    b_base = B_ptr + expert_id * stride_be

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.static_range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # logical A is (M_total, K); addressing uses stride_am/stride_ak passed in
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        b = tl.load(
            b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(OUT_DTYPE), mask=m_mask[:, None] & n_mask[None, :])


@torch.compiler.disable
def triton_gmm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
) -> torch.Tensor:
    """
    Drop-in replacement for grouped_gemm.ops.gmm(a, b, batch_sizes, trans_b)
    with added support for trans_a.

    Semantics:
      - Logical GEMM is always:  C = A @ B  where logical A is (M_total, K) and B is (K, N)
      - If trans_a=False:
            a is stored as (M_total, K)
      - If trans_a=True:
            a is stored as (K, M_total) and treated as A = a.T (no explicit transpose)

      - If trans_b=False:
            b is stored as (E, K, N)
      - If trans_b=True:
            b is stored as (E, N, K) and treated as B = b.T (per expert)

    Assumption:
      - Rows are grouped by expert in order 0..E-1 in the *logical* A rows (M dimension),
        i.e., the first batch_sizes[0] rows belong to expert 0, etc.
    """
    assert a.is_cuda and b.is_cuda, "CUDA-only"
    assert a.ndim == 2 and b.ndim == 3
    E = b.shape[0]

    # batch_sizes stays on GPU (no .cpu())
    batch_sizes = batch_sizes.to(dtype=torch.int32, device=a.device)

    # Interpret A (logical shape M_total x K) via trans_a
    if not trans_a:
        # a: (M_total, K)
        M_total = a.shape[0]
        K = a.shape[1]
        stride_am = a.stride(0)
        stride_ak = a.stride(1)
    else:
        # a: (K, M_total), logical A = a.T
        K = a.shape[0]
        M_total = a.shape[1]
        # logical address A[m,k] == a[k,m]
        # => m uses a.stride(1), k uses a.stride(0)
        stride_am = a.stride(1)
        stride_ak = a.stride(0)

    # Interpret B as logical (E, K, N) via trans_b
    if not trans_b:
        # b: (E, K, N)
        assert b.shape[1] == K, f"b K-dim mismatch: b.shape[1]={b.shape[1]} vs K={K}"
        N = b.shape[2]
        stride_be = b.stride(0)
        stride_bk = b.stride(1)
        stride_bn = b.stride(2)
    else:
        # b: (E, N, K), logical B = b.T
        assert b.shape[2] == K, f"b K-dim mismatch: b.shape[2]={b.shape[2]} vs K={K}"
        N = b.shape[1]
        stride_be = b.stride(0)
        stride_bk = b.stride(2)  # step along K in logical view
        stride_bn = b.stride(1)  # step along N in logical view

    out = torch.empty((M_total, N), device=a.device, dtype=a.dtype)

    # Heuristic tile sizes (tune/autotune for H100/B200 + your shapes)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    num_warps = 8
    num_stages = 4

    total_blocks_upper = triton.cdiv(M_total, BLOCK_M) + E
    grid = (total_blocks_upper, triton.cdiv(N, BLOCK_N))

    out_dtype = tl.bfloat16 if out.dtype == torch.bfloat16 else tl.float16

    _moe_gmm_fwd_kernel[grid](
        a, b, batch_sizes, out,
        M_total=M_total, N=N, K=K, E=E,
        stride_am=stride_am, stride_ak=stride_ak,
        stride_be=stride_be, stride_bk=stride_bk, stride_bn=stride_bn,
        stride_cm=out.stride(0), stride_cn=out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        OUT_DTYPE=out_dtype,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out



class GroupedGemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, batch_sizes, trans_b):
        # assert torch.count_nonzero(batch_sizes) != 0, "Input batch_sizes should not be all zeros!"
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.trans_b = trans_b
        return triton_gmm(a, b, batch_sizes, trans_a=False, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        a, b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b

        agrad = None
        if ctx.needs_input_grad[0]:
            agrad = triton_gmm(
                grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

        bgrad = None
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad, a) if trans_b else (a, grad)
            bgrad = triton_gmm(
                lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
        return agrad, bgrad, None, None


def gmm(a, b, batch_sizes, trans_b=False):
    return GroupedGemm.apply(a, b, batch_sizes, trans_b)

