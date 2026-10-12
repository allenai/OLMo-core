"""
GPU tests for the fast landmark kernel (``landmark_fast``) across head dims, including
``head_dim=256`` (Qwen3.5). The ``<= 128`` cases double as a regression check that extending the
kernel to larger head dims did not change its behavior: the fast kernel must stay bit-identical to
the original ``FusedLandmarkAttention``.
"""

import math

import pytest
import torch

from olmo_core.nn.attention.landmark import (
    build_block_doc_id,
    landmark_grouped_softmax,
)
from olmo_core.nn.attention.landmark_fast import fused_landmark_attention_fast
from olmo_core.nn.attention.landmark_kernel import (
    fused_landmark_attention,
    has_landmark_kernel,
)
from olmo_core.nn.attention.ring import UlyssesContextParallelStyle
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.testing import requires_gpu, requires_multi_gpu, run_distributed_test


def _eager_landmark_reference(q, k, v, block_size):
    """Dense eager landmark attention over ``(B, H, T, d)`` tensors (full-context, causal)."""
    B, H, T, d = q.shape
    att = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    att_mask = torch.tril(torch.ones((1, 1, T, T), device=q.device), diagonal=0) == 1.0
    sec = torch.arange(T, device=q.device) // block_size
    last_section_mask = (sec[None, :] == sec[:, None]).unsqueeze(0).unsqueeze(1)
    is_mem = ((torch.arange(T, device=q.device) % block_size) == (block_size - 1)).view(1, 1, 1, T)
    mask = att_mask & ~(last_section_mask & is_mem)
    last_section_mask = (last_section_mask & mask).expand(B, H, T, T)
    is_mem_ = (is_mem & mask).expand(B, H, T, T)
    att = att.masked_fill(~mask, float("-inf"))
    att = landmark_grouped_softmax(att, -1, is_mem_, last_section_mask).to(q.dtype)
    att = att.masked_fill(~mask, 0.0)
    return att @ v


def _eager_landmark_reference_packed(q, k, v, block_size, doc_id_tok):
    """Eager landmark attention with a block-diagonal document mask (sequence packing).

    ``doc_id_tok`` is a ``(B, T)`` per-token document id; a query never attends across a document
    boundary (nor to another document's landmark tokens).
    """
    B, H, T, d = q.shape
    att = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    att_mask = torch.tril(torch.ones((1, 1, T, T), device=q.device), diagonal=0) == 1.0
    # Block-diagonal document mask: (B, 1, T, T).
    same_doc = (doc_id_tok[:, :, None] == doc_id_tok[:, None, :]).unsqueeze(1)
    att_mask = att_mask & same_doc
    sec = torch.arange(T, device=q.device) // block_size
    last_section_mask = (sec[None, :] == sec[:, None]).unsqueeze(0).unsqueeze(1)
    is_mem = ((torch.arange(T, device=q.device) % block_size) == (block_size - 1)).view(1, 1, 1, T)
    mask = att_mask & ~(last_section_mask & is_mem)
    last_section_mask = (last_section_mask & mask).expand(B, H, T, T)
    is_mem_ = (is_mem & mask).expand(B, H, T, T)
    att = att.masked_fill(~mask, float("-inf"))
    att = landmark_grouped_softmax(att, -1, is_mem_, last_section_mask).to(q.dtype)
    att = att.masked_fill(~mask, 0.0)
    return att @ v


def _packing_layout(block_size: int):
    """Two batch rows with distinct block-aligned document layouts (T = 4 blocks each).

    Returns ``(T, cu_doc_lens, doc_id_tok)`` where ``cu_doc_lens`` is the flattened-over-batch
    cumulative and ``doc_id_tok`` is the per-token ``(B=2, T)`` document id.
    """
    T = block_size * 4
    cu_doc_lens = torch.tensor(
        [0, 2 * block_size, T, T + block_size, 2 * T], dtype=torch.int32, device="cuda"
    )
    # Row 0: docs [2 blocks, 2 blocks]; row 1: docs [1 block, 3 blocks].
    doc_id_tok = torch.zeros(2, T, dtype=torch.long, device="cuda")
    doc_id_tok[0, 2 * block_size :] = 1
    doc_id_tok[1, :block_size] = 2
    doc_id_tok[1, block_size:] = 3
    return T, cu_doc_lens, doc_id_tok


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize("head_dim, mem_freq", [(64, 15), (128, 15), (256, 63)])
def test_fast_kernel_packing_forward_matches_eager(head_dim: int, mem_freq: int):
    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, n_heads = 2, 4
    T, cu_doc_lens, doc_id_tok = _packing_layout(block_size)
    doc_id = build_block_doc_id(cu_doc_lens, B, T, block_size)
    q = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)

    out_kernel = fused_landmark_attention_fast(
        q, k, v, is_mem, block_size=block_size, doc_id=doc_id
    )
    out_eager = _eager_landmark_reference_packed(q, k, v, block_size, doc_id_tok)

    torch.testing.assert_close(out_kernel, out_eager, rtol=1e-2, atol=1e-2)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize(
    "head_dim, mem_freq, dtype",
    [(64, 15, torch.float32), (128, 15, torch.float32), (256, 63, torch.bfloat16)],
)
def test_fast_kernel_packing_backward_matches_eager(head_dim, mem_freq, dtype):
    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, n_heads = 2, 4
    T, cu_doc_lens, doc_id_tok = _packing_layout(block_size)
    doc_id = build_block_doc_id(cu_doc_lens, B, T, block_size)
    scale = head_dim**-0.5
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)
    base = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=dtype)
    grad_out = torch.rand_like(base)

    def grads(use_kernel):
        q, k, v = (base.clone().requires_grad_(True) for _ in range(3))
        if use_kernel:
            out = fused_landmark_attention_fast(
                q, k, v, is_mem, sm_scale=scale, block_size=block_size, doc_id=doc_id
            )
        else:
            out = _eager_landmark_reference_packed(q, k, v, block_size, doc_id_tok)
        out.backward(grad_out)
        return out, q.grad, k.grad, v.grad

    out_k, dq_k, dk_k, dv_k = grads(True)
    out_e, dq_e, dk_e, dv_e = grads(False)

    out_tol = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(rtol=1e-2, atol=1e-2)
    grad_tol = dict(rtol=1e-3, atol=1e-3) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out_k, out_e, **out_tol)
    torch.testing.assert_close(dq_k, dq_e, **grad_tol)
    torch.testing.assert_close(dk_k, dk_e, **grad_tol)
    torch.testing.assert_close(dv_k, dv_e, **grad_tol)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize(
    "head_dim, mem_freq",
    [(64, 15), (128, 15), (256, 15), (256, 63)],  # (256, 63) is the Qwen3.5 training config
)
def test_fast_kernel_forward_matches_eager(head_dim: int, mem_freq: int):
    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, n_heads = 2, 4
    T = block_size * 4
    q = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)

    out_kernel = fused_landmark_attention_fast(q, k, v, is_mem, block_size=block_size)
    out_eager = _eager_landmark_reference(q, k, v, block_size)

    torch.testing.assert_close(out_kernel, out_eager, rtol=1e-2, atol=1e-2)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize(
    "head_dim, mem_freq, dtype",
    [
        # fp32 makes the eager comparison exact up to accumulation noise (mirrors the original
        # kernel test).
        (64, 15, torch.float32),
        (128, 15, torch.float32),
        (256, 15, torch.float32),
        # The Qwen3.5 training config. fp32 is not runnable here -- at head_dim 256 and BLOCK=64
        # the backward's fp32 dot/trans operand tiles exceed H100 shared memory (~278KB vs 232KB)
        # at any num_stages -- so validate in bf16, the dtype training actually uses, with
        # accumulation-noise tolerances.
        (256, 63, torch.bfloat16),
    ],
)
def test_fast_kernel_backward_matches_eager(head_dim: int, mem_freq: int, dtype: torch.dtype):
    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, n_heads = 2, 4
    T = block_size * 4
    scale = head_dim**-0.5
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)
    base = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=dtype)
    grad_out = torch.rand_like(base)

    def grads(use_kernel):
        q, k, v = (base.clone().requires_grad_(True) for _ in range(3))
        if use_kernel:
            out = fused_landmark_attention_fast(
                q, k, v, is_mem, sm_scale=scale, block_size=block_size
            )
        else:
            out = _eager_landmark_reference(q, k, v, block_size)
        out.backward(grad_out)
        return out, q.grad, k.grad, v.grad

    out_k, dq_k, dk_k, dv_k = grads(True)
    out_e, dq_e, dk_e, dv_e = grads(False)

    out_tol = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(rtol=1e-2, atol=1e-2)
    grad_tol = dict(rtol=1e-3, atol=1e-3) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out_k, out_e, **out_tol)
    torch.testing.assert_close(dq_k, dq_e, **grad_tol)
    torch.testing.assert_close(dk_k, dk_e, **grad_tol)
    torch.testing.assert_close(dv_k, dv_e, **grad_tol)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fast_kernel_unchanged_vs_original_for_small_head_dims(head_dim: int, dtype: torch.dtype):
    # Regression check for the head_dim <= 128 path after adding head_dim 256 support: the fast
    # kernel reuses the original forward kernel, so the forward output must be exactly equal.
    # Gradients match only up to last-ulp reassociation (1-2 ulps on a handful of elements): the
    # two backwards run with different num_warps, which retiles the tl.dot reductions. torch's
    # dtype-aware default tolerances comfortably cover that ulp noise while staying far tighter
    # than the eager-reference comparisons above.
    torch.manual_seed(0)
    mem_freq = 15
    block_size = mem_freq + 1
    B, n_heads = 2, 4
    T = block_size * 4
    scale = head_dim**-0.5
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)
    base = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=dtype)
    grad_out = torch.rand_like(base)

    def grads(fn):
        q, k, v = (base.clone().requires_grad_(True) for _ in range(3))
        out = fn(q, k, v, is_mem, sm_scale=scale, block_size=block_size)
        out.backward(grad_out)
        return out, q.grad, k.grad, v.grad

    out_f, dq_f, dk_f, dv_f = grads(fused_landmark_attention_fast)
    out_o, dq_o, dk_o, dv_o = grads(fused_landmark_attention)

    torch.testing.assert_close(out_f, out_o, rtol=0, atol=0)
    torch.testing.assert_close(dq_f, dq_o)
    torch.testing.assert_close(dk_f, dk_o)
    torch.testing.assert_close(dv_f, dv_o)


def _fast_landmark_transformer_config(seq_len: int, mem_freq: int) -> TransformerConfig:
    # Small fast-landmark transformer (fused Triton kernel path) for the context-parallel test.
    # n_heads / n_kv_heads must be divisible by the CP degree (world_size=2); mem_freq >= 15.
    return TransformerConfig.llama_like(
        d_model=128,
        vocab_size=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=2,
        qk_norm=True,
        rope_theta=10_000,
        fast_landmark=True,
        mem_freq=mem_freq,
    )


def _run_fast_landmark_ulysses_cp_packed(
    checkpoint_dir: str, inputs_path: str, doc_lens, max_doc_len: int, seq_len: int, mem_freq: int
):
    from torch.distributed.tensor import DTensor, Shard, init_device_mesh

    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.distributed.utils import get_full_tensor, get_world_size
    from olmo_core.utils import get_default_device

    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("cp",))

    def _build():
        model = _fast_landmark_transformer_config(seq_len, mem_freq).build()
        model.init_weights(device=device, max_seq_len=seq_len)
        load_model_and_optim_state(checkpoint_dir, model)
        model.eval()
        return model

    model = _build()
    model.apply_cp(mesh["cp"], uly=UlyssesContextParallelStyle())

    input_ids = torch.load(inputs_path, map_location=device)
    doc_lens_t = torch.tensor([doc_lens], dtype=torch.int32)
    with torch.no_grad():
        local_logits = model(input_ids=input_ids, doc_lens=doc_lens_t, max_doc_lens=[max_doc_len])
    logits = get_full_tensor(DTensor.from_local(local_logits, mesh, (Shard(1),)))

    # Reference: the same packed forward on a single rank (no CP) -- the already-validated non-CP
    # packing path of the fused kernel. Matching it proves CP shard reconstruction + per-document
    # RoPE (incl. the boundary-straddling document) under the fused kernel.
    model_full = _build()
    with torch.no_grad():
        expected = model_full(input_ids=input_ids, doc_lens=doc_lens_t, max_doc_lens=[max_doc_len])
    torch.testing.assert_close(logits, expected, rtol=1e-3, atol=1e-3)


@requires_multi_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
def test_fast_landmark_ulysses_cp_packing_matches_full(tmp_path):
    # Ulysses CP + sequence packing on the fused fast-landmark kernel, with a document straddling the
    # CP rank boundary. CP must reproduce the single-rank packed forward. This is the GPU-only
    # counterpart to the eager CPU checks in landmark_test.py (the fused kernel is CUDA + triton only).
    from olmo_core.distributed.checkpoint import save_model_and_optim_state
    from olmo_core.utils import get_default_device

    torch.manual_seed(0)
    mem_freq = 15  # block_size = 16 (kernel requires mem_freq >= 15)
    seq_len = 64  # world_size=2 -> T_local=32; 4 landmark blocks of 16
    doc_lens = [16, 32, 16]  # the 32-token doc spans global [16, 48), straddling the boundary at 32
    assert sum(doc_lens) == seq_len

    device = get_default_device()
    model = _fast_landmark_transformer_config(seq_len, mem_freq).build()
    model.init_weights(device=device, max_seq_len=seq_len)
    model.eval()

    input_ids = torch.randint(0, 256, (1, seq_len))  # B must be 1 for CP + intra-document masking

    inputs_path = tmp_path / "x.pt"
    checkpoint_dir = tmp_path / "checkpoint"
    torch.save(input_ids, inputs_path)
    save_model_and_optim_state(checkpoint_dir, model)

    run_distributed_test(
        _run_fast_landmark_ulysses_cp_packed,
        backend="nccl",
        world_size=2,
        func_args=(
            str(checkpoint_dir),
            str(inputs_path),
            doc_lens,
            max(doc_lens),
            seq_len,
            mem_freq,
        ),
    )
