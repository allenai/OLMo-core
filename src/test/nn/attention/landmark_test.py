import math

import pytest
import torch

from olmo_core.data.composable.concat_and_chunk_instance_source import (
    ConcatAndChunkInstanceSource,
)
from olmo_core.data.composable.landmark_instance_source import LandmarkInstanceSource
from olmo_core.data.composable.token_source import InMemoryTokenSource
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig, AttentionType, LandmarkAttention
from olmo_core.nn.attention.landmark import (
    build_block_doc_id,
    build_local_packed_position_ids,
    landmark_grouped_softmax,
)
from olmo_core.nn.attention.landmark_kernel import (
    fused_landmark_attention,
    has_landmark_kernel,
)
from olmo_core.nn.attention.ring import UlyssesContextParallelStyle
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.testing import requires_gpu, run_distributed_test


def _landmark_attention(
    *,
    d_model: int = 64,
    n_heads: int = 8,
    n_kv_heads: int = 2,
    head_dim: int = 8,
    mem_freq: int = 3,
    qk_norm: bool = True,
) -> LandmarkAttention:
    config = AttentionConfig(
        name=AttentionType.landmark,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        bias=False,
        mem_freq=mem_freq,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False) if qk_norm else None,
        use_head_qk_norm=qk_norm,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
    )
    attn = config.build(d_model, layer_idx=0, n_layers=2)
    assert isinstance(attn, LandmarkAttention)
    return attn


def test_landmark_config_builds():
    attn = _landmark_attention(mem_freq=3)
    assert attn.mem_freq == 3
    assert attn.block_size == 4


def test_landmark_mem_freq_rejected_on_non_landmark():
    with pytest.raises(OLMoConfigurationError):
        AttentionConfig(name=AttentionType.default, n_heads=8, mem_freq=3).build(
            64, layer_idx=0, n_layers=1
        )


def test_landmark_eager_forward_shape():
    attn = _landmark_attention()
    attn.eval()
    B, T, d_model = 2, 12, 64  # T multiple of block_size (4)
    x = torch.randn(B, T, d_model)
    with torch.no_grad():
        out = attn(x)
    assert out.shape == (B, T, d_model)
    assert torch.isfinite(out).all()


def test_landmark_eager_training_backward():
    # The default (eager) path must be fully differentiable so training works without the
    # fused kernel, on plain CPU.
    attn = _landmark_attention()
    attn.train()
    assert attn.use_kernel is False
    B, T, d_model = 2, 12, 64
    x = torch.randn(B, T, d_model, requires_grad=True)
    out = attn(x)
    (out**2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [p.grad for p in attn.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_landmark_use_kernel_rejected_on_non_landmark():
    with pytest.raises(OLMoConfigurationError):
        AttentionConfig(name=AttentionType.default, n_heads=8, landmark_use_kernel=True).build(
            64, layer_idx=0, n_layers=1
        )


def test_landmark_requires_seq_len_multiple_of_block_size():
    attn = _landmark_attention(mem_freq=3)  # block_size 4
    attn.eval()
    with pytest.raises(OLMoConfigurationError):
        attn(torch.randn(1, 10, 64))


def test_build_block_doc_id():
    # block_size 4; B=2, T=8 (2 blocks/row). Flattened-over-batch cu_doc_lens with a batch edge at 8.
    # Row 0: docs [4, 4] -> blocks [0, 1]; row 1: docs [8] -> blocks [2, 2] (one document).
    doc_id = build_block_doc_id(torch.tensor([0, 4, 8, 16]), batch_size=2, seq_len=8, block_size=4)
    assert doc_id.dtype == torch.int32
    assert doc_id.tolist() == [[0, 1], [2, 2]]


def test_build_block_doc_id_rejects_unaligned():
    # The guard against the "wrong kind of packing": non-block-aligned doc boundaries (e.g.
    # EOS-derived doc_lens) are rejected with a message pointing at LandmarkPackingInstanceSource.
    with pytest.raises(ValueError, match="LandmarkPackingInstanceSource"):
        build_block_doc_id(torch.tensor([0, 6, 16]), batch_size=2, seq_len=8, block_size=4)


def test_landmark_rejects_unaligned_document_boundary():
    # block_size 4; a boundary at 6 is not a multiple of block_size, which would mis-group landmarks.
    # The error tells the user to use LandmarkPackingInstanceSource (not EOS-based / generic packing).
    attn = _landmark_attention(mem_freq=3)
    attn.eval()
    with pytest.raises(OLMoConfigurationError, match="LandmarkPackingInstanceSource"):
        attn(torch.randn(1, 12, 64), cu_doc_lens=torch.tensor([0, 6, 12], dtype=torch.int32))


def _packing_equivalence_check(*, mem_freq: int, doc_lens, d_model: int = 64, n_heads: int = 8):
    """
    Run a packed forward/backward (one sequence with ``cu_doc_lens``) and compare against
    gradient accumulation over the same documents fed one-at-a-time. Outputs and *all* parameter
    gradients (plus the per-document input gradients) must match, which is exactly the invariant
    packed SFT relies on.
    """
    block_size = mem_freq + 1
    assert all(L % block_size == 0 for L in doc_lens)
    T = sum(doc_lens)
    torch.manual_seed(0)

    # Disable QK-norm here: its fused RMSNorm kernel reduces the weight gradient in float32, which
    # introduces a ~1e-7 packed-vs-unpacked discrepancy that is an artifact of that kernel, *not* of
    # the landmark masking. Without it, the packing equivalence holds to float64 machine precision,
    # which is what isolates and proves the masking is exactly correct.
    attn = _landmark_attention(d_model=d_model, n_heads=n_heads, mem_freq=mem_freq, qk_norm=False)
    attn.train()
    assert attn.use_kernel is False

    # A single packed sequence holding all documents back-to-back.
    x_packed = torch.randn(1, T, d_model, dtype=torch.float64, requires_grad=True)
    attn.double()
    cu_doc_lens = torch.tensor([0, *torch.tensor(doc_lens).cumsum(0).tolist()], dtype=torch.int32)

    out_packed = attn(x_packed, cu_doc_lens=cu_doc_lens)
    out_packed.pow(2).sum().backward()
    packed_param_grads = {n: p.grad.clone() for n, p in attn.named_parameters()}

    # Now feed each document separately, accumulating gradients (no grad reset between docs).
    for p in attn.parameters():
        p.grad = None
    x_unpacked = x_packed.detach().clone().requires_grad_(True)
    outs = []
    start = 0
    for L in doc_lens:
        sl = x_unpacked[:, start : start + L, :]
        outs.append(attn(sl))  # no cu_doc_lens: a standalone document
        start += L
    out_accum = torch.cat(outs, dim=1)
    out_accum.pow(2).sum().backward()
    accum_param_grads = {n: p.grad.clone() for n, p in attn.named_parameters()}

    # Forward outputs must match document-for-document.
    torch.testing.assert_close(out_packed, out_accum, rtol=1e-10, atol=1e-10)
    # Input gradients must match.
    torch.testing.assert_close(x_packed.grad, x_unpacked.grad, rtol=1e-10, atol=1e-10)
    # Accumulated parameter gradients must match the packed parameter gradients.
    for name in packed_param_grads:
        torch.testing.assert_close(
            packed_param_grads[name], accum_param_grads[name], rtol=1e-9, atol=1e-9, msg=name
        )


def test_build_local_packed_position_ids_straddling_doc():
    # Full sequence T=16 sharded across cp_world_size=2 (T_local=8). Documents [4, 8, 4]: the middle
    # 8-token document straddles the rank boundary at position 8, so rank 1's first four of its tokens
    # must continue that document's positions (4,5,6,7) rather than reset to 0.
    cu_doc_lens = torch.tensor([0, 4, 12, 16], dtype=torch.int32)
    pos0 = build_local_packed_position_ids(cu_doc_lens, 1, 8, cp_rank=0, cp_world_size=2)
    pos1 = build_local_packed_position_ids(cu_doc_lens, 1, 8, cp_rank=1, cp_world_size=2)
    torch.testing.assert_close(pos0, torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]))
    torch.testing.assert_close(pos1, torch.tensor([[4, 5, 6, 7, 0, 1, 2, 3]]))

    # Concatenating the per-rank shards must reproduce the full-sequence per-document positions (i.e.
    # the standard ``cu_doc_lens`` reset over the whole sequence), which is what the non-CP path uses.
    full = torch.cat([pos0, pos1], dim=1)
    flat = torch.arange(16)
    doc_id = torch.bucketize(flat, cu_doc_lens.to(torch.long)[1:], right=True)
    expected_full = (flat - cu_doc_lens.to(torch.long)[doc_id]).view(1, 16)
    torch.testing.assert_close(full, expected_full)


def test_build_local_packed_position_ids_batched():
    # Flattened-over-batch convention with B=2, T=8, cp_world_size=2 (T_local=4). Each row has its own
    # layout; row 0 docs [4, 4], row 1 docs [8] (one document straddling the boundary at local pos 4).
    cu_doc_lens = torch.tensor([0, 4, 8, 16], dtype=torch.int32)
    pos0 = build_local_packed_position_ids(cu_doc_lens, 2, 4, cp_rank=0, cp_world_size=2)
    pos1 = build_local_packed_position_ids(cu_doc_lens, 2, 4, cp_rank=1, cp_world_size=2)
    # Row 0: global [0,4)->doc0 pos 0-3 (rank0), [4,8)->doc1 pos 0-3 (rank1).
    # Row 1: global flat [8,16) is one doc; rank0 holds [8,12) pos 0-3, rank1 holds [12,16) pos 4-7.
    torch.testing.assert_close(pos0, torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]))
    torch.testing.assert_close(pos1, torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]))


def test_landmark_packing_matches_grad_accumulation():
    # Two documents of different lengths, both multiples of block_size (=4).
    _packing_equivalence_check(mem_freq=3, doc_lens=[8, 12])


def test_landmark_packing_matches_grad_accumulation_three_docs():
    _packing_equivalence_check(mem_freq=3, doc_lens=[4, 8, 8])


def test_landmark_packing_matches_grad_accumulation_batched():
    """
    Packing with batch_size > 1, where each batch element has its *own* document layout, must still
    equal gradient accumulation over every (batch-element, document) sub-sequence fed alone. This
    pins the flattened-over-batch ``cu_doc_lens`` convention (matching the flash backend).
    """
    mem_freq, block_size = 3, 4
    rows = [[8, 12], [4, 16]]  # two batch elements, each summing to T=20
    T = 20
    assert all(sum(r) == T and all(L % block_size == 0 for L in r) for r in rows)
    torch.manual_seed(0)

    attn = _landmark_attention(d_model=64, n_heads=8, mem_freq=mem_freq, qk_norm=False)
    attn.train().double()

    x = torch.randn(len(rows), T, 64, dtype=torch.float64, requires_grad=True)
    # Flattened-over-batch cumulative lengths: [0, 8, 20, 24, 40].
    flat = []
    running = 0
    for r in rows:
        for L in r:
            running += L
            flat.append(running)
    cu_doc_lens = torch.tensor([0, *flat], dtype=torch.int32)

    out_packed = attn(x, cu_doc_lens=cu_doc_lens)
    out_packed.pow(2).sum().backward()
    packed_grads = {n: p.grad.clone() for n, p in attn.named_parameters()}

    # Reference: every (row, document) sub-sequence on its own; assemble per row and accumulate grads.
    for p in attn.parameters():
        p.grad = None
    x_ref = x.detach().clone().requires_grad_(True)
    rows_out = []
    for b, r in enumerate(rows):
        start = 0
        doc_outs = []
        for L in r:
            doc_outs.append(attn(x_ref[b : b + 1, start : start + L, :]))
            start += L
        rows_out.append(torch.cat(doc_outs, dim=1))
    out_ref = torch.cat(rows_out, dim=0)
    out_ref.pow(2).sum().backward()
    ref_grads = {n: p.grad.clone() for n, p in attn.named_parameters()}

    torch.testing.assert_close(out_packed, out_ref, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-10, atol=1e-10)
    for name in packed_grads:
        torch.testing.assert_close(
            packed_grads[name], ref_grads[name], rtol=1e-9, atol=1e-9, msg=name
        )


def test_landmark_packing_no_cross_document_attention():
    # A query in the second document must place zero attention probability on the first document.
    attn = _landmark_attention(mem_freq=3)
    attn.eval()
    T = 12
    attn_mask, is_mem, last_section_mask = attn._landmark_masks(
        T, torch.device("cpu"), torch.float32, cu_doc_lens=torch.tensor([0, 4, 12])
    )
    logits = torch.randn(1, 1, T, T)
    logits = logits + attn_mask
    logits = torch.maximum(logits, torch.tensor(torch.finfo(logits.dtype).min))
    probs = landmark_grouped_softmax(
        logits, dim=-1, is_mem=is_mem.expand(1, 1, T, T), last_section_mask=last_section_mask
    )
    # Queries 4..11 are in document 2; keys 0..3 are document 1.
    assert torch.allclose(probs[0, 0, 4:, :4], torch.zeros(T - 4, 4), atol=1e-6)
    # And every query still normalizes to 1 over its own document.
    assert torch.allclose(probs.sum(-1), torch.ones(1, 1, T), atol=1e-5)


def test_landmark_model_packing_matches_grad_accumulation():
    """
    End-to-end check through the full transformer (the path SFT uses): a packed forward driven by
    ``doc_lens``/``max_doc_lens`` must equal gradient accumulation over the same documents fed one
    at a time. This exercises the whole wiring: ``doc_lens`` -> ``cu_doc_lens`` in the model ->
    per-document RoPE reset + block-diagonal masking in ``LandmarkAttention``.
    """
    torch.manual_seed(0)
    doc_lens = [8, 12]  # multiples of block_size (4)
    T = sum(doc_lens)
    # qk_norm=False so the comparison is exact to float64 (see _packing_equivalence_check).
    config = TransformerConfig.llama_like(
        d_model=64,
        vocab_size=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=2,
        qk_norm=False,
        rope_theta=10_000,
        landmark=True,
        mem_freq=3,
    )
    model = config.build()
    model.init_weights(device=torch.device("cpu"), max_seq_len=T)
    model.double()
    model.train()

    input_ids = torch.randint(0, 256, (1, T))

    # Packed: a single sequence with explicit document boundaries.
    logits_packed = model(
        input_ids=input_ids,
        doc_lens=torch.tensor([doc_lens], dtype=torch.int32),
        max_doc_lens=[max(doc_lens)],
    )
    logits_packed.pow(2).sum().backward()
    packed_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    # Gradient accumulation: each document as its own standalone sequence.
    for p in model.parameters():
        p.grad = None
    logits_list = []
    start = 0
    for L in doc_lens:
        logits_list.append(model(input_ids=input_ids[:, start : start + L]))
        start += L
    logits_accum = torch.cat(logits_list, dim=1)
    logits_accum.pow(2).sum().backward()
    accum_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    torch.testing.assert_close(logits_packed, logits_accum, rtol=1e-9, atol=1e-9)
    assert packed_grads.keys() == accum_grads.keys()
    for name in packed_grads:
        # The fused RMSNorm kernel reduces its *weight* gradient in float32, so the block/lm_head
        # norm weights carry a ~1e-7 packed-vs-unpacked artifact (every other parameter — linear
        # weights, embeddings, attention — matches to ~1e-16). The tolerance here clears that
        # artifact while still being orders of magnitude tighter than any real masking bug, which
        # would corrupt gradients by O(1).
        torch.testing.assert_close(
            packed_grads[name], accum_grads[name], rtol=2e-6, atol=2e-6, msg=name
        )


def test_landmark_grouped_softmax_rows_sum_to_one():
    attn = _landmark_attention(mem_freq=3)
    B, n_heads, T = 2, 4, 12
    attn_mask, is_mem, last_section_mask = attn._landmark_masks(
        T, torch.device("cpu"), torch.float32
    )
    logits = torch.randn(B, n_heads, T, T)
    logits = logits + attn_mask
    logits = torch.maximum(logits, torch.tensor(torch.finfo(logits.dtype).min))
    probs = landmark_grouped_softmax(
        logits,
        dim=-1,
        is_mem=is_mem.expand(B, n_heads, T, T),
        last_section_mask=last_section_mask.expand(B, 1, T, T),
    )
    # Every query attends to at least itself, so all rows should normalize to 1.
    assert torch.allclose(probs.sum(-1), torch.ones(B, n_heads, T), atol=1e-5)


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


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize("mem_freq", [15, 63])
def test_landmark_kernel_matches_eager(mem_freq: int):
    # The fused kernel's tl.dot requires tile dims >= 16, hence mem_freq >= 15.
    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, n_heads, d = 2, 4, 64
    T = block_size * 4
    attn = _landmark_attention(
        d_model=n_heads * d, n_heads=n_heads, n_kv_heads=n_heads, head_dim=d, mem_freq=mem_freq
    ).cuda()

    q = torch.rand(B, n_heads, T, d, device="cuda", dtype=torch.bfloat16)
    k = torch.rand(B, n_heads, T, d, device="cuda", dtype=torch.bfloat16)
    v = torch.rand(B, n_heads, T, d, device="cuda", dtype=torch.bfloat16)
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)

    out_kernel = fused_landmark_attention(
        q, k, v, is_mem, sm_scale=attn.softmax_scale, block_size=block_size
    )
    out_eager = attn._eager_forward(q, k, v)

    torch.testing.assert_close(out_kernel, out_eager, rtol=1e-2, atol=1e-2)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize("mem_freq", [15, 63])
def test_landmark_kernel_backward_matches_eager(mem_freq: int):
    # Validate the fused kernel's gradients against the eager autograd reference. Run in fp32 so the
    # comparison is exact (bf16 differs only by accumulation noise).
    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, n_heads, d = 2, 4, 64
    T = block_size * 4
    scale = d**-0.5
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)
    base = torch.rand(B, n_heads, T, d, device="cuda", dtype=torch.float32)
    grad_out = torch.rand_like(base)

    def grads(use_kernel):
        q, k, v = (base.clone().requires_grad_(True) for _ in range(3))
        if use_kernel:
            out = fused_landmark_attention(q, k, v, is_mem, sm_scale=scale, block_size=block_size)
        else:
            out = _eager_landmark_reference(q, k, v, block_size)
        out.backward(grad_out)
        return out, q.grad, k.grad, v.grad

    out_k, dq_k, dk_k, dv_k = grads(True)
    out_e, dq_e, dk_e, dv_e = grads(False)

    torch.testing.assert_close(out_k, out_e, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dq_k, dq_e, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dk_k, dk_e, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dv_k, dv_e, rtol=1e-3, atol=1e-3)


def _packing_doc_layout(block_size: int):
    """Two batch rows with distinct block-aligned document layouts (T = 4 blocks each)."""
    T = block_size * 4
    # Row 0: docs [2 blocks, 2 blocks]; row 1: docs [1 block, 3 blocks].
    cu_doc_lens = torch.tensor(
        [0, 2 * block_size, T, T + block_size, 2 * T], dtype=torch.int32, device="cuda"
    )
    return T, cu_doc_lens


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize("mem_freq", [15, 63])
def test_landmark_kernel_packing_matches_eager(mem_freq: int):
    # The fused kernel's document masking must match the (grad-accumulation-verified) eager path.
    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, n_heads, d = 2, 4, 64
    T, cu_doc_lens = _packing_doc_layout(block_size)
    attn = _landmark_attention(
        d_model=n_heads * d, n_heads=n_heads, n_kv_heads=n_heads, head_dim=d, mem_freq=mem_freq
    ).cuda()
    doc_id = build_block_doc_id(cu_doc_lens, B, T, block_size)

    q = torch.rand(B, n_heads, T, d, device="cuda", dtype=torch.bfloat16)
    k = torch.rand(B, n_heads, T, d, device="cuda", dtype=torch.bfloat16)
    v = torch.rand(B, n_heads, T, d, device="cuda", dtype=torch.bfloat16)
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)

    out_kernel = fused_landmark_attention(
        q, k, v, is_mem, sm_scale=attn.softmax_scale, block_size=block_size, doc_id=doc_id
    )
    out_eager = attn._eager_forward(q, k, v, cu_doc_lens=cu_doc_lens)

    torch.testing.assert_close(out_kernel, out_eager, rtol=1e-2, atol=1e-2)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize("mem_freq", [15, 63])
def test_landmark_kernel_packing_backward_matches_eager(mem_freq: int):
    # Validate the fused kernel's *document-masked* gradients against the eager autograd reference,
    # in fp32 so the comparison is exact (bf16 differs only by accumulation noise).
    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, n_heads, d = 2, 4, 64
    T, cu_doc_lens = _packing_doc_layout(block_size)
    scale = d**-0.5
    doc_id = build_block_doc_id(cu_doc_lens, B, T, block_size)
    attn = _landmark_attention(
        d_model=n_heads * d,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        head_dim=d,
        mem_freq=mem_freq,
        qk_norm=False,
    ).cuda()
    attn.softmax_scale = scale
    base = torch.rand(B, n_heads, T, d, device="cuda", dtype=torch.float32)
    grad_out = torch.rand_like(base)
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)

    def grads(use_kernel):
        q, k, v = (base.clone().requires_grad_(True) for _ in range(3))
        if use_kernel:
            out = fused_landmark_attention(
                q, k, v, is_mem, sm_scale=scale, block_size=block_size, doc_id=doc_id
            )
        else:
            out = attn._eager_forward(q, k, v, cu_doc_lens=cu_doc_lens)
        out.backward(grad_out)
        return out, q.grad, k.grad, v.grad

    out_k, dq_k, dk_k, dv_k = grads(True)
    out_e, dq_e, dk_e, dv_e = grads(False)

    torch.testing.assert_close(out_k, out_e, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dq_k, dq_e, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dk_k, dk_e, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dv_k, dv_e, rtol=1e-3, atol=1e-3)


def test_landmark_instance_source(tmp_path):
    mem_freq, mem_id = 3, 999
    # Content tokens 0..23 -> two content instances of length 12 (a multiple of mem_freq).
    tokens = InMemoryTokenSource(tokens=list(range(24)), work_dir=tmp_path)
    content = ConcatAndChunkInstanceSource(tokens, sequence_length=12, work_dir=tmp_path)
    source = LandmarkInstanceSource(content, mem_freq=mem_freq, mem_id=mem_id, work_dir=tmp_path)

    block_size = mem_freq + 1
    assert source.sequence_length == 12 // mem_freq * block_size  # 16
    assert len(source) == len(content) == 2

    inst = source[0]
    ids = list(inst["input_ids"])
    mask = list(inst["label_mask"])
    assert len(ids) == source.sequence_length == 16
    assert len(mask) == 16
    # landmark tokens at every block_size-th position (the last of each block)
    landmark_positions = [i for i in range(16) if (i % block_size) == (block_size - 1)]
    assert landmark_positions == [3, 7, 11, 15]
    for i in range(16):
        if i in landmark_positions:
            assert ids[i] == mem_id
            assert mask[i] is False
        else:
            assert ids[i] != mem_id
            assert mask[i] is True
    # content tokens preserved in order
    assert [t for t in ids if t != mem_id] == list(range(12))


def test_landmark_instance_source_requires_multiple_of_mem_freq(tmp_path):
    tokens = InMemoryTokenSource(tokens=list(range(20)), work_dir=tmp_path)
    content = ConcatAndChunkInstanceSource(tokens, sequence_length=10, work_dir=tmp_path)
    with pytest.raises(OLMoConfigurationError):
        LandmarkInstanceSource(content, mem_freq=3, mem_id=999, work_dir=tmp_path)


def test_landmark_instance_source_exclude_landmark_predictors(tmp_path):
    from olmo_core.data.utils import get_labels

    mem_freq, mem_id = 3, 999  # block_size 4
    block_size = mem_freq + 1
    tokens = InMemoryTokenSource(tokens=list(range(100, 124)), work_dir=tmp_path)
    content = ConcatAndChunkInstanceSource(tokens, sequence_length=12, work_dir=tmp_path)

    default = LandmarkInstanceSource(content, mem_freq=mem_freq, mem_id=mem_id, work_dir=tmp_path)
    excluded = LandmarkInstanceSource(
        content,
        mem_freq=mem_freq,
        mem_id=mem_id,
        exclude_landmark_predictors=True,
        work_dir=tmp_path,
    )

    # Changing the option must change the fingerprint (so cached artifacts aren't reused).
    assert default.fingerprint != excluded.fingerprint

    # input_ids are identical; only label_mask differs.
    d_inst, e_inst = default[0], excluded[0]
    assert list(d_inst["input_ids"]) == list(e_inst["input_ids"])

    landmark_positions = [i for i in range(16) if (i % block_size) == (block_size - 1)]  # 3,7,11,15

    def loss_positions(inst):
        batch = {
            "input_ids": torch.tensor([list(map(int, inst["input_ids"]))]),
            "label_mask": torch.tensor([list(map(bool, inst["label_mask"]))]),
        }
        labels = get_labels(batch, label_ignore_index=-100)[0]
        return {i for i in range(labels.numel()) if labels[i].item() != -100}

    default_loss = loss_positions(d_inst)
    excluded_loss = loss_positions(e_inst)

    # By default, interior landmark positions contribute as predictors.
    assert default_loss & set(landmark_positions) == {3, 7, 11}  # 15 is the final position
    # With the option on, no landmark position contributes, and that's the only difference.
    assert excluded_loss & set(landmark_positions) == set()
    assert default_loss - excluded_loss == {3, 7, 11}


def _landmark_transformer_config(seq_len: int) -> TransformerConfig:
    # Small landmark transformer (eager path) used by the context-parallel test. n_heads and
    # n_kv_heads must be divisible by the CP degree (world_size=2).
    return TransformerConfig.llama_like(
        d_model=64,
        vocab_size=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=2,
        qk_norm=True,
        rope_theta=10_000,
        landmark=True,
        mem_freq=3,  # block_size 4
    )


def _run_landmark_ulysses_cp(
    checkpoint_dir: str, inputs_path: str, outputs_path: str, seq_len: int
):
    from torch.distributed.tensor import DTensor, Shard, init_device_mesh

    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.distributed.utils import get_full_tensor, get_world_size

    mesh = init_device_mesh("cpu", (get_world_size(),), mesh_dim_names=("cp",))

    model = _landmark_transformer_config(seq_len).build()
    model.apply_cp(mesh["cp"], uly=UlyssesContextParallelStyle())
    model.init_weights(device=torch.device("cpu"), max_seq_len=seq_len)
    load_model_and_optim_state(checkpoint_dir, model)
    model.eval()

    # The model shards input_ids and the RoPE buffers internally via the Ulysses load balancer, so
    # we pass the full (replicated) input; each rank returns its sequence shard of the logits.
    input_ids = torch.load(inputs_path, map_location="cpu")
    with torch.no_grad():
        local_logits = model(input_ids=input_ids)
    logits = DTensor.from_local(local_logits, mesh, (Shard(1),))

    expected = torch.load(outputs_path, map_location="cpu")
    torch.testing.assert_close(get_full_tensor(logits), expected, rtol=1e-4, atol=1e-4)


def test_landmark_ulysses_cp_matches_full(tmp_path):
    # Ulysses CP must reproduce a single-rank full-sequence forward: LandmarkAttention.forward
    # gathers the complete sequence (with n_heads / cp_degree heads) via all-to-all, so the grouped
    # softmax still sees every preceding block's landmark. The model itself shards the input and the
    # RoPE buffers, so this also covers RoPE under CP.
    from olmo_core.distributed.checkpoint import save_model_and_optim_state

    torch.manual_seed(0)
    seq_len = 16  # 4 landmark blocks; divisible by block_size (4) and world_size (2)

    model = _landmark_transformer_config(seq_len).build()
    model.init_weights(device=torch.device("cpu"), max_seq_len=seq_len)
    model.eval()

    input_ids = torch.randint(0, 256, (2, seq_len))
    with torch.no_grad():
        expected = model(input_ids=input_ids)

    inputs_path = tmp_path / "x.pt"
    outputs_path = tmp_path / "y.pt"
    checkpoint_dir = tmp_path / "checkpoint"
    torch.save(input_ids, inputs_path)
    torch.save(expected, outputs_path)
    save_model_and_optim_state(checkpoint_dir, model)

    run_distributed_test(
        _run_landmark_ulysses_cp,
        backend="gloo",
        world_size=2,
        func_args=(str(checkpoint_dir), str(inputs_path), str(outputs_path), seq_len),
    )


def _run_landmark_ulysses_cp_packed(
    checkpoint_dir: str, inputs_path: str, doc_lens, max_doc_len: int, seq_len: int
):
    from torch.distributed.tensor import DTensor, Shard, init_device_mesh

    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.distributed.utils import get_full_tensor, get_world_size

    mesh = init_device_mesh("cpu", (get_world_size(),), mesh_dim_names=("cp",))

    model = _landmark_transformer_config(seq_len).build()
    model.apply_cp(mesh["cp"], uly=UlyssesContextParallelStyle())
    model.init_weights(device=torch.device("cpu"), max_seq_len=seq_len)
    load_model_and_optim_state(checkpoint_dir, model)
    model.eval()

    # The model shards input_ids and the RoPE buffers internally via the Ulysses load balancer (the
    # document boundaries are passed through unchanged). Each rank returns its sequence shard.
    input_ids = torch.load(inputs_path, map_location="cpu")
    with torch.no_grad():
        local_logits = model(
            input_ids=input_ids,
            doc_lens=torch.tensor([doc_lens], dtype=torch.int32),
            max_doc_lens=[max_doc_len],
        )
    logits = DTensor.from_local(local_logits, mesh, (Shard(1),))

    # Reference: the same packed forward on a single rank (no CP). This is the already-validated
    # non-CP packing path, so matching it proves the CP shard reconstruction + per-document RoPE
    # (including the boundary-straddling document) are correct.
    model_full = _landmark_transformer_config(seq_len).build()
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


def test_landmark_ulysses_cp_packing_matches_full(tmp_path):
    # Ulysses CP + sequence packing (intra-document masking). The packed sequence contains a document
    # that *straddles* the CP rank boundary, which exercises the per-document RoPE reset on the local
    # shard (positions must stay continuous across the boundary) and the block-diagonal masking built
    # on the gathered full sequence. CP must reproduce the single-rank packed forward exactly.
    from olmo_core.distributed.checkpoint import save_model_and_optim_state

    torch.manual_seed(0)
    seq_len = 16  # world_size=2 -> T_local=8; block_size=4
    doc_lens = [4, 8, 4]  # the 8-token doc spans global [4, 12), straddling the rank boundary at 8
    assert sum(doc_lens) == seq_len

    model = _landmark_transformer_config(seq_len).build()
    model.init_weights(device=torch.device("cpu"), max_seq_len=seq_len)
    model.eval()

    input_ids = torch.randint(0, 256, (1, seq_len))  # B must be 1 for CP + intra-document masking

    inputs_path = tmp_path / "x.pt"
    checkpoint_dir = tmp_path / "checkpoint"
    torch.save(input_ids, inputs_path)
    save_model_and_optim_state(checkpoint_dir, model)

    run_distributed_test(
        _run_landmark_ulysses_cp_packed,
        backend="gloo",
        world_size=2,
        func_args=(str(checkpoint_dir), str(inputs_path), doc_lens, max(doc_lens), seq_len),
    )


def test_landmark_rejects_ring_cp():
    from torch.distributed.device_mesh import DeviceMesh

    from olmo_core.nn.attention.ring import (
        RingAttentionZigZagLoadBalancer,
        RingContextParallelStyle,
    )

    attn = _landmark_attention(mem_freq=3)
    # An empty DeviceMesh shell is enough to exercise the guard, which fires before any collective
    # is created.
    fake_mesh = DeviceMesh.__new__(DeviceMesh)
    with pytest.raises(OLMoConfigurationError, match="only supports Ulysses"):
        attn.apply_cp(
            fake_mesh,
            ring=RingContextParallelStyle(load_balancer=RingAttentionZigZagLoadBalancer),
        )


def test_landmark_factory_wiring():
    config = TransformerConfig.qwen3_4B(vocab_size=1000, landmark=True, mem_freq=63)
    mixer = config.block.sequence_mixer
    assert mixer.name == AttentionType.landmark
    assert mixer.mem_freq == 63

    # mem_freq without landmark is rejected.
    with pytest.raises(OLMoConfigurationError):
        TransformerConfig.llama_like(d_model=64, vocab_size=256, n_layers=1, n_heads=8, mem_freq=3)

    # A small landmark model actually builds.
    small = TransformerConfig.llama_like(
        d_model=64,
        vocab_size=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=2,
        qk_norm=True,
        landmark=True,
        mem_freq=3,
    )
    model = small.build()
    n_landmark = sum(1 for m in model.modules() if isinstance(m, LandmarkAttention))
    assert n_landmark == 2
