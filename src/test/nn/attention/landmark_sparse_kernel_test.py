"""
Tests for ``SparseLandmarkAttention`` sequence packing (intra-document masking):

  * CPU: the doc-aware eager forms (``sparse_landmark_attention`` / ``..._ref``) agree, and a packed
    forward/backward equals gradient accumulation over each (batch-element, document) sub-sequence.
  * GPU: the fused Triton kernel's document masking (forward + backward) matches the eager reference.
"""

import pytest
import torch

from olmo_core.nn.attention.landmark import build_block_doc_id
from olmo_core.nn.attention.landmark_sparse import (
    sparse_landmark_attention,
    sparse_landmark_attention_ref,
)
from olmo_core.nn.attention.landmark_sparse_kernel import (
    has_sparse_kernel,
    sparse_landmark_attention_triton_train,
)
from olmo_core.nn.attention.ring import UlyssesContextParallelStyle
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.testing import requires_gpu, run_distributed_test


def _sparse_landmark_transformer_config(seq_len: int) -> TransformerConfig:
    # Small sparse-landmark transformer (eager torch fallback on CPU) for the context-parallel test.
    # n_heads / n_kv_heads must be divisible by the CP degree (world_size=2).
    return TransformerConfig.llama_like(
        d_model=64,
        vocab_size=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=2,
        qk_norm=True,
        rope_theta=10_000,
        sparse_landmark=True,
        mem_freq=3,
        num_landmarks=1,  # block_size = mem_freq + num_landmarks = 4
    )


def _run_sparse_landmark_ulysses_cp_packed(
    checkpoint_dir: str, inputs_path: str, doc_lens, max_doc_len: int, seq_len: int
):
    from torch.distributed.tensor import DTensor, Shard, init_device_mesh

    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.distributed.utils import get_full_tensor, get_world_size

    mesh = init_device_mesh("cpu", (get_world_size(),), mesh_dim_names=("cp",))

    model = _sparse_landmark_transformer_config(seq_len).build()
    model.apply_cp(mesh["cp"], uly=UlyssesContextParallelStyle())
    model.init_weights(device=torch.device("cpu"), max_seq_len=seq_len)
    load_model_and_optim_state(checkpoint_dir, model)
    model.eval()

    input_ids = torch.load(inputs_path, map_location="cpu")
    with torch.no_grad():
        local_logits = model(
            input_ids=input_ids,
            doc_lens=torch.tensor([doc_lens], dtype=torch.int32),
            max_doc_lens=[max_doc_len],
        )
    logits = DTensor.from_local(local_logits, mesh, (Shard(1),))

    # Reference: the same packed forward on a single rank (no CP), the already-validated non-CP path.
    model_full = _sparse_landmark_transformer_config(seq_len).build()
    model_full.init_weights(device=torch.device("cpu"), max_seq_len=seq_len)
    load_model_and_optim_state(checkpoint_dir, model_full)
    model_full.eval()
    with torch.no_grad():
        expected = model_full(
            input_ids=input_ids,
            doc_lens=torch.tensor([doc_lens], dtype=torch.int32),
            max_doc_lens=[max_doc_len],
        )
    torch.testing.assert_close(get_full_tensor(logits), expected, rtol=1e-4, atol=1e-4)


def test_sparse_landmark_ulysses_cp_packing_matches_full(tmp_path):
    # Ulysses CP + sequence packing for SparseLandmarkAttention, with a document straddling the CP
    # rank boundary. CP must reproduce the single-rank packed forward exactly.
    from olmo_core.distributed.checkpoint import save_model_and_optim_state

    torch.manual_seed(0)
    seq_len = 16  # world_size=2 -> T_local=8; block_size=4
    doc_lens = [4, 8, 4]  # the 8-token doc spans global [4, 12), straddling the rank boundary at 8
    assert sum(doc_lens) == seq_len

    model = _sparse_landmark_transformer_config(seq_len).build()
    model.init_weights(device=torch.device("cpu"), max_seq_len=seq_len)
    model.eval()

    input_ids = torch.randint(0, 256, (1, seq_len))  # B must be 1 for CP + intra-document masking

    inputs_path = tmp_path / "x.pt"
    checkpoint_dir = tmp_path / "checkpoint"
    torch.save(input_ids, inputs_path)
    save_model_and_optim_state(checkpoint_dir, model)

    run_distributed_test(
        _run_sparse_landmark_ulysses_cp_packed,
        backend="gloo",
        world_size=2,
        func_args=(str(checkpoint_dir), str(inputs_path), doc_lens, max(doc_lens), seq_len),
    )


def _layout(block_size, device="cpu"):
    """Two batch rows with distinct chunk-aligned document layouts (T = 4 chunks each)."""
    L = block_size
    T = L * 4
    # Flattened-over-batch cu_doc_lens: row 0 docs [2 chunks, 2 chunks]; row 1 docs [1 chunk, 3].
    cu_doc_lens = torch.tensor([0, 2 * L, T, T + L, 2 * T], dtype=torch.int32, device=device)
    doc_id = build_block_doc_id(cu_doc_lens, 2, T, L)
    return T, doc_id


def test_sparse_packing_ref_matches_efficient():
    torch.manual_seed(0)
    B, H, D, L, G = 2, 4, 16, 4, 1
    T, doc_id = _layout(L)
    q, k, v = (torch.randn(B, H, T, D, dtype=torch.float64) for _ in range(3))
    o_eff = sparse_landmark_attention(q, k, v, L, num_landmarks=G, doc_id=doc_id)
    o_ref = sparse_landmark_attention_ref(q, k, v, L, num_landmarks=G, doc_id=doc_id)
    torch.testing.assert_close(o_eff, o_ref, rtol=1e-12, atol=1e-12)


def test_sparse_packing_matches_grad_accumulation():
    """Packed forward/backward == gradient accumulation over each (row, document) sub-sequence."""
    torch.manual_seed(0)
    B, H, D, L, G = 2, 4, 16, 4, 1
    T, doc_id = _layout(L)
    # Per-row document layouts (token spans), matching ``_layout``.
    rows = [[(0, 2 * L), (2 * L, 4 * L)], [(0, L), (L, 4 * L)]]
    base = torch.randn(B, H, T, D, dtype=torch.float64)
    grad_out = torch.randn_like(base)

    q = base.clone().requires_grad_(True)
    k = base.clone().requires_grad_(True)
    v = base.clone().requires_grad_(True)
    out = sparse_landmark_attention(q, k, v, L, num_landmarks=G, doc_id=doc_id)
    out.backward(grad_out)

    qr = base.clone().requires_grad_(True)
    kr = base.clone().requires_grad_(True)
    vr = base.clone().requires_grad_(True)
    out_ref = torch.zeros_like(base)
    for b, docs in enumerate(rows):
        for s, e in docs:
            out_ref[b : b + 1, :, s:e] = sparse_landmark_attention(
                qr[b : b + 1, :, s:e],
                kr[b : b + 1, :, s:e],
                vr[b : b + 1, :, s:e],
                L,
                num_landmarks=G,
            )
    out_ref.backward(grad_out)

    torch.testing.assert_close(out, out_ref, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(q.grad, qr.grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(k.grad, kr.grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(v.grad, vr.grad, rtol=1e-10, atol=1e-10)


@requires_gpu
@pytest.mark.skipif(not has_sparse_kernel(), reason="requires triton sparse landmark kernel")
@pytest.mark.parametrize("block_size, num_landmarks", [(16, 1), (16, 4), (64, 1)])
def test_sparse_kernel_packing_matches_eager(block_size: int, num_landmarks: int):
    # The fused kernel's document masking (fwd + bwd) must match the eager reference, in fp32 so the
    # comparison is exact up to accumulation noise.
    torch.manual_seed(0)
    B, H, D, L, G = 2, 4, 64, block_size, num_landmarks
    T, doc_id = _layout(L, device="cuda")
    scale = D**-0.5
    base = torch.rand(B, H, T, D, device="cuda", dtype=torch.float32)
    grad_out = torch.rand_like(base)

    def run(use_kernel):
        q, k, v = (base.clone().requires_grad_(True) for _ in range(3))
        if use_kernel:
            out = sparse_landmark_attention_triton_train(
                q, k, v, L, num_landmarks=G, scale=scale, doc_id=doc_id
            )
        else:
            out = sparse_landmark_attention(q, k, v, L, num_landmarks=G, scale=scale, doc_id=doc_id)
        out.backward(grad_out)
        return out, q.grad, k.grad, v.grad

    out_k, dq_k, dk_k, dv_k = run(True)
    out_e, dq_e, dk_e, dv_e = run(False)

    torch.testing.assert_close(out_k, out_e, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dq_k, dq_e, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dk_k, dk_e, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dv_k, dv_e, rtol=1e-3, atol=1e-3)
