"""Tests for :class:`SharedVectorLandmarkAttention`.

The fused-kernel (``use_kernel=True``) ``head_dim`` output is exercised by the existing fast-landmark
kernel tests; these CPU tests cover the eager reference path and the (kernel-shared) ``vec_dim`` tail
computation:

* the eager ``head_dim`` output equals ordinary (non-compressive) landmark attention;
* the tail from :meth:`SharedVectorLandmarkAttention._shared_vector_tail` (used by *both* paths)
  equals a dense brute-force tail derived from the full landmark grouped-softmax probabilities.

``cu_doc_lens`` sequence packing is covered by the ``test_packing_*`` tests below: a packed forward
(one sequence, ``cu_doc_lens``) must equal gradient accumulation over the same documents fed
one-at-a-time, for both the ``head_dim`` output and the (doc-aware) ``vec_dim`` tail.
"""

from typing import Optional

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    AttentionConfig,
    AttentionType,
    LandmarkAttention,
    SharedVectorLandmarkAttention,
)
from olmo_core.nn.attention.landmark import (
    build_block_doc_id,
    build_landmark_masks,
    landmark_grouped_softmax,
)
from olmo_core.nn.attention.landmark_kernel import has_landmark_kernel
from olmo_core.testing import requires_gpu


def _build(
    *,
    d_model: int = 64,
    n_heads: int = 4,
    n_kv_heads: int = 4,
    head_dim: int = 16,
    mem_freq: int = 15,
    vec_dim: int = 32,
    dtype: torch.dtype = torch.float32,
) -> SharedVectorLandmarkAttention:
    m = SharedVectorLandmarkAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        mem_freq=mem_freq,
        vec_dim=vec_dim,
        use_kernel=False,
        bias=False,
        dtype=dtype,
    )
    # Randomize the (zero-initialized) base and w_out_vec so tests exercise a non-trivial tail path.
    with torch.no_grad():
        m.base.normal_(std=0.1)
        m.w_out_vec.weight.normal_(std=0.1)
    return m


def _qkv(m: SharedVectorLandmarkAttention, B: int, T: int, dtype: torch.dtype):
    g = torch.Generator().manual_seed(0)
    shape = (B, m.n_heads, T, m.head_dim)
    q = torch.randn(shape, generator=g, dtype=dtype)
    k = torch.randn(shape, generator=g, dtype=dtype)
    v = torch.randn(shape, generator=g, dtype=dtype)
    return q, k, v


def _dense_tail_reference(
    m: SharedVectorLandmarkAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_doc_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Brute-force tail from the dense landmark grouped-softmax probabilities ``P``.

    ``tail_i = sum_j P_ij * (base if j in query i's block else e_{block(j)})``. With ``cu_doc_lens``,
    ``P`` (from :func:`build_landmark_masks`) already has cross-document entries at exactly zero, so
    the ``cross`` / ``same_mass`` split below naturally excludes another document's blocks.
    """
    B, H, T, _ = q.shape
    Lb = m.block_size
    attn_mask, is_mem, last_section_mask = build_landmark_masks(
        T, Lb, q.device, q.dtype, cu_doc_lens=cu_doc_lens, batch_size=B
    )
    attn = torch.matmul(q, k.transpose(-1, -2)) * m.softmax_scale + attn_mask
    attn = torch.maximum(attn, torch.tensor(torch.finfo(attn.dtype).min, dtype=attn.dtype))
    P = landmark_grouped_softmax(
        attn,
        dim=-1,
        is_mem=is_mem.expand(B, H, T, T),
        last_section_mask=last_section_mask.expand(B, 1, T, T),
    )

    mem_pos = torch.arange(Lb - 1, T, Lb)
    v_lm = v[:, :, mem_pos, :]  # (B, H, nb, D)
    e = torch.einsum("bhnd,hde->bhne", v_lm, m.weight_landmark)  # (B, H, nb, vec)
    block_of = torch.arange(T) // Lb
    E_full = e[:, :, block_of, :]  # (B, H, T, vec)
    same_block = (block_of.view(T, 1) == block_of.view(1, T)).to(P.dtype)  # (T, T)

    cross = torch.matmul(P * (1.0 - same_block), E_full)  # (B, H, T, vec)
    same_mass = (P * same_block).sum(dim=-1)  # (B, H, T)
    return cross + same_mass.unsqueeze(-1) * m.base.view(1, H, 1, m.vec_dim)


def test_eager_main_matches_non_compressive_landmark():
    m = _build()
    T = m.block_size * 3
    q, k, v = _qkv(m, B=2, T=T, dtype=torch.float32)

    main = m._main_dense(q, k, v)

    ref = LandmarkAttention(
        d_model=m.d_model,
        n_heads=m.n_heads,
        n_kv_heads=m.n_kv_heads,
        head_dim=m.head_dim,
        mem_freq=m.mem_freq,
        bias=False,
    )
    ref_out = ref._eager_forward(q, k, v)
    torch.testing.assert_close(main, ref_out)


def test_tail_matches_dense_bruteforce():
    m = _build()
    with torch.no_grad():
        m.weight_landmark.normal_(std=0.3)  # non-trivial landmark map
    T = m.block_size * 4
    q, k, v = _qkv(m, B=2, T=T, dtype=torch.float32)

    tail = m._shared_vector_tail(q, k, v)
    ref = _dense_tail_reference(m, q, k, v)
    torch.testing.assert_close(tail, ref, atol=1e-4, rtol=1e-4)


def test_tail_matches_dense_bruteforce_packed():
    """The doc-aware tail (``doc_id`` given) must match the dense brute-force reference computed
    with the matching ``cu_doc_lens``, which independently derives cross-document masking via the
    full grouped-softmax probability matrix rather than the block-gate shortcut."""
    m = _build()
    with torch.no_grad():
        m.weight_landmark.normal_(std=0.3)
        m.base.normal_(std=0.3)
    Lb = m.block_size
    doc_lens = [2 * Lb, 3 * Lb]
    T = sum(doc_lens)
    q, k, v = _qkv(m, B=2, T=T, dtype=torch.float32)
    # Flattened-over-batch: both rows share the same [doc_lens[0], doc_lens[1]] layout.
    cu_doc_lens = torch.tensor([0, doc_lens[0], T, T + doc_lens[0], 2 * T], dtype=torch.int32)
    doc_id = build_block_doc_id(cu_doc_lens, batch_size=2, seq_len=T, block_size=Lb)

    tail = m._shared_vector_tail(q, k, v, doc_id=doc_id)
    ref = _dense_tail_reference(m, q, k, v, cu_doc_lens=cu_doc_lens)
    torch.testing.assert_close(tail, ref, atol=1e-4, rtol=1e-4)


def test_packing_no_cross_document_attention():
    """A query in the second document's tail must be unaffected by perturbing the first document's
    keys/values -- the direct analogue of ``test_landmark_packing_no_cross_document_attention``."""
    m = _build()
    with torch.no_grad():
        m.weight_landmark.normal_(std=0.3)
        m.base.normal_(std=0.3)
    Lb = m.block_size
    doc_lens = [2 * Lb, 3 * Lb]
    T = sum(doc_lens)
    q, k, v = _qkv(m, B=1, T=T, dtype=torch.float32)
    cu_doc_lens = torch.tensor([0, *torch.tensor(doc_lens).cumsum(0).tolist()], dtype=torch.int32)
    doc_id = build_block_doc_id(cu_doc_lens, batch_size=1, seq_len=T, block_size=Lb)

    tail = m._shared_vector_tail(q, k, v, doc_id=doc_id)

    k2, v2 = k.clone(), v.clone()
    k2[:, :, : doc_lens[0]] += torch.randn_like(k2[:, :, : doc_lens[0]])
    v2[:, :, : doc_lens[0]] += torch.randn_like(v2[:, :, : doc_lens[0]])
    tail2 = m._shared_vector_tail(q, k2, v2, doc_id=doc_id)

    # Second document's queries (positions doc_lens[0]:) are untouched by the first document's edit.
    torch.testing.assert_close(
        tail[:, :, doc_lens[0] :], tail2[:, :, doc_lens[0] :], atol=1e-6, rtol=1e-6
    )
    # Sanity: the edit *did* change the first document's own tail (the perturbation is not a no-op).
    assert not torch.allclose(tail[:, :, : doc_lens[0]], tail2[:, :, : doc_lens[0]])


def _packing_equivalence_check(*, mem_freq: int = 15, doc_lens, batch_size: int = 1):
    """Run a packed forward/backward (one sequence with ``cu_doc_lens``) through the full module and
    compare against gradient accumulation over the same documents fed one-at-a-time. Outputs and all
    parameter gradients must match -- the invariant packed SFT relies on. Mirrors
    ``landmark_test.py::_packing_equivalence_check``, using a coarser tolerance because (unlike plain
    ``LandmarkAttention``) the tail is always computed in float32 internally (see ``_tail_query_chunk``
    module docs), independent of packing.
    """
    block_size = mem_freq + 1
    assert all(L % block_size == 0 for L in doc_lens)
    T = sum(doc_lens)
    torch.manual_seed(0)

    m = SharedVectorLandmarkAttention(
        d_model=64,
        n_heads=4,
        n_kv_heads=4,
        head_dim=16,
        mem_freq=mem_freq,
        vec_dim=8,
        use_kernel=False,
        bias=False,
        dtype=torch.float64,
    )
    with torch.no_grad():
        m.weight_landmark.normal_(std=0.3)
        m.base.normal_(std=0.1)
        m.w_out_vec.weight.normal_(std=0.1)
    m.train()

    x_packed = torch.randn(batch_size, T, 64, dtype=torch.float64, requires_grad=True)
    # Flattened-over-batch boundaries: every row shares the same ``doc_lens`` layout.
    flat = []
    running = 0
    for _ in range(batch_size):
        for L in doc_lens:
            running += L
            flat.append(running)
    cu_doc_lens = torch.tensor([0, *flat], dtype=torch.int32)

    out_packed = m(x_packed, cu_doc_lens=cu_doc_lens)
    out_packed.pow(2).sum().backward()
    packed_grads = {n: p.grad.clone() for n, p in m.named_parameters()}

    for p in m.parameters():
        p.grad = None
    x_unpacked = x_packed.detach().clone().requires_grad_(True)
    rows_out = []
    for b in range(batch_size):
        start = 0
        doc_outs = []
        for L in doc_lens:
            doc_outs.append(m(x_unpacked[b : b + 1, start : start + L, :]))
            start += L
        rows_out.append(torch.cat(doc_outs, dim=1))
    out_accum = torch.cat(rows_out, dim=0)
    out_accum.pow(2).sum().backward()
    accum_grads = {n: p.grad.clone() for n, p in m.named_parameters()}

    torch.testing.assert_close(out_packed, out_accum, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(x_packed.grad, x_unpacked.grad, atol=1e-5, rtol=1e-4)
    for name in packed_grads:
        torch.testing.assert_close(
            packed_grads[name], accum_grads[name], atol=1e-5, rtol=1e-4, msg=name
        )


def test_packing_matches_grad_accumulation():
    _packing_equivalence_check(doc_lens=[32, 48])


def test_packing_matches_grad_accumulation_three_docs():
    _packing_equivalence_check(doc_lens=[16, 32, 32])


def test_packing_matches_grad_accumulation_batched():
    _packing_equivalence_check(doc_lens=[16, 32], batch_size=2)


def test_packing_rejects_unaligned_document_boundary():
    m = _build()
    T = m.block_size * 3
    x = torch.randn(1, T, m.d_model)
    unaligned = torch.tensor([0, m.block_size + 2, T], dtype=torch.int32)
    with pytest.raises(ValueError, match="LandmarkPackingInstanceSource"):
        m(x, cu_doc_lens=unaligned)


def test_tail_chunking_matches_single_chunk_and_grads():
    """Query-chunking (with per-chunk checkpointing under autograd) must reproduce the single-chunk
    dense result and its gradients -- each query's tail is independent, so chunking only bounds the
    fp32 working set. Uses a chunk of 2 blocks over 5 blocks to exercise an uneven final chunk."""
    m = _build()
    with torch.no_grad():
        m.weight_landmark.normal_(std=0.3)
        m.base.normal_(std=0.3)
    T = m.block_size * 5
    q, k, v = _qkv(m, B=2, T=T, dtype=torch.float32)

    # Reference: single chunk (chunk >= T), with grads.
    m._tail_query_chunk = 10**9
    q1 = q.clone().requires_grad_(True)
    ref = m._shared_vector_tail(q1, k, v)
    ref.sum().backward()
    ref_q_grad = q1.grad.clone()
    ref_wl_grad = m.weight_landmark.grad.clone()
    ref_base_grad = m.base.grad.clone()
    m.zero_grad(set_to_none=True)

    # Also match the independent dense brute-force reference (chunking didn't just match itself).
    torch.testing.assert_close(
        ref, _dense_tail_reference(m, q1.detach(), k, v), atol=1e-4, rtol=1e-4
    )

    # Multi-chunk (2 blocks/chunk -> chunks of 2,2,1 blocks) with checkpointing (grad enabled).
    m._tail_query_chunk = 2 * m.block_size
    q2 = q.clone().requires_grad_(True)
    out = m._shared_vector_tail(q2, k, v)
    out.sum().backward()

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(q2.grad, ref_q_grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(m.weight_landmark.grad, ref_wl_grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(m.base.grad, ref_base_grad, atol=1e-5, rtol=1e-5)


def test_tail_cp_head_slicing_matches_full(monkeypatch):
    """Under Ulysses CP the tail runs head-parallel: after the cp2hp all-to-all each rank holds only
    ``n_heads / cp_degree`` heads (global slice ``[rank*H : (rank+1)*H]``), while ``weight_landmark`` /
    ``base`` stay replicated at full ``n_heads``. The forward must select this rank's parameter slice.

    Compute the tail on each rank's head slice with ``cp_enabled`` and check it (a) does not raise the
    head-size einsum mismatch the un-sliced params caused and (b) reproduces the matching slice of the
    full-head (non-CP) tail. The other landmark variants have no such per-head param, so this guards
    the one class that needed the fix.
    """
    import olmo_core.nn.attention.landmark_shared_vector as lsv

    n_heads = 8
    m = _build(n_heads=n_heads, n_kv_heads=n_heads)
    with torch.no_grad():
        m.weight_landmark.normal_(std=0.3)  # non-trivial per-head map
        m.base.normal_(std=0.3)
    T = m.block_size * 4
    q, k, v = _qkv(m, B=2, T=T, dtype=torch.float32)

    # Reference: full-head tail, CP disabled (slice is a no-op).
    assert not m.cp_enabled
    full = m._shared_vector_tail(q, k, v)  # (B, n_heads, T, vec)

    cp_degree = 4
    h_local = n_heads // cp_degree
    m._cp_pg = object()  # non-None -> cp_enabled; dist.get_rank is monkeypatched per rank below
    assert m.cp_enabled

    parts = []
    for rank in range(cp_degree):
        monkeypatch.setattr(lsv.dist, "get_rank", lambda pg, r=rank: r)
        h0 = rank * h_local
        part = m._shared_vector_tail(
            q[:, h0 : h0 + h_local], k[:, h0 : h0 + h_local], v[:, h0 : h0 + h_local]
        )  # (B, h_local, T, vec)
        assert part.shape == (2, h_local, T, m.vec_dim)
        torch.testing.assert_close(part, full[:, h0 : h0 + h_local], atol=1e-4, rtol=1e-4)
        parts.append(part)

    # Concatenating the per-rank head slices (what the hp2cp all-to-all gathers) rebuilds the full tail.
    torch.testing.assert_close(torch.cat(parts, dim=1), full, atol=1e-4, rtol=1e-4)


def test_first_block_tail_is_base():
    """Queries in block 0 have no past block, so their entire tail is the learned base vector."""
    m = _build()
    with torch.no_grad():
        m.weight_landmark.normal_(std=0.3)
    T = m.block_size * 3
    q, k, v = _qkv(m, B=2, T=T, dtype=torch.float32)

    tail = m._shared_vector_tail(q, k, v)  # (B, H, T, vec)
    Lb = m.block_size
    first_block = tail[:, :, :Lb, :]  # queries in block 0
    expected = m.base.view(1, m.n_heads, 1, m.vec_dim).expand_as(first_block)
    torch.testing.assert_close(first_block, expected, atol=1e-5, rtol=1e-4)


def test_shapes_and_split_projection():
    m = _build(vec_dim=8)
    B, T = 2, m.block_size * 3
    q, k, v = _qkv(m, B=B, T=T, dtype=torch.float32)

    # head_dim output branch is unchanged; the vec branch has its own projection.
    assert m._attn_core(q, k, v).shape == (B, m.n_heads, T, m.head_dim)
    assert m._shared_vector_tail(q, k, v).shape == (B, m.n_heads, T, m.vec_dim)
    assert m.w_out.in_features == m.n_heads * m.head_dim  # base shape -> loads from base checkpoint
    assert m.w_out_vec.in_features == m.n_heads * m.vec_dim

    x = torch.randn(B, T, m.d_model)
    out = m(x)
    assert out.shape == (B, T, m.d_model)


def test_zero_init_tail_reproduces_plain_landmark():
    """With the default (zero) w_out_vec/base init, the vec tail contributes nothing, so the module
    equals a plain fast-landmark model sharing the same q/k/v/out weights."""
    m = SharedVectorLandmarkAttention(
        d_model=64, n_heads=4, n_kv_heads=4, head_dim=16, mem_freq=15, use_kernel=False, bias=False
    )  # default init: w_out_vec = 0, base = 0
    x = torch.randn(2, m.block_size * 3, m.d_model)
    out = m(x)

    ref = LandmarkAttention(
        d_model=64, n_heads=4, n_kv_heads=4, head_dim=16, mem_freq=15, bias=False
    )
    # Share the loaded/base weights (q/k/v/out) between the two modules.
    ref_keys = set(ref.state_dict().keys())
    ref.load_state_dict({k: v for k, v in m.state_dict().items() if k in ref_keys}, strict=False)
    torch.testing.assert_close(out, ref(x), atol=1e-5, rtol=1e-4)


def test_decode_matches_training_per_position():
    """Per-position decode (main + the new tail) reproduces the training forward at each query."""
    m = _build()
    with torch.no_grad():
        m.weight_landmark.normal_(std=0.3)
    T = m.block_size * 3
    q, k, v = _qkv(m, B=1, T=T, dtype=torch.float32)

    main_train = m._attn_core(q, k, v)  # (1, H, T, head_dim)
    tail_train = m._shared_vector_tail(q, k, v)  # (1, H, T, vec)

    for qpos in range(T):
        probs, v_used, ss = m._decode_probs(
            q[:, :, qpos : qpos + 1, :], k[:, :, : qpos + 1, :], v[:, :, : qpos + 1, :], qpos
        )
        main_dec = torch.matmul(probs.to(v_used.dtype), v_used)  # (1, H, 1, head_dim)
        tail_dec = m._decode_tail(probs, v_used, ss)  # (1, H, 1, vec)
        torch.testing.assert_close(main_dec[:, :, 0], main_train[:, :, qpos], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(tail_dec[:, :, 0], tail_train[:, :, qpos], atol=1e-4, rtol=1e-4)


def test_prefill_generate_matches_training_forward():
    """The KV-cached prefill path (two-branch projection) equals the plain forward."""
    m = _build()
    with torch.no_grad():
        m.weight_landmark.normal_(std=0.3)
    T = m.block_size * 3
    x = torch.randn(1, T, m.d_model)

    out_train = m(x)
    m.init_kv_cache_manager(batch_size=1, max_seq_len=T)
    out_prefill = m(x)
    torch.testing.assert_close(out_train, out_prefill, atol=1e-5, rtol=1e-4)


def test_backward_populates_new_param_grads():
    m = _build()
    B, T = 1, m.block_size * 3
    x = torch.randn(B, T, m.d_model)
    m(x).sum().backward()
    assert m.weight_landmark.grad is not None and torch.isfinite(m.weight_landmark.grad).all()
    assert m.base.grad is not None and torch.isfinite(m.base.grad).all()


def test_gate_rejected():
    from olmo_core.nn.attention import GateConfig, GateGranularity

    with pytest.raises(OLMoConfigurationError):
        SharedVectorLandmarkAttention(
            d_model=64,
            n_heads=4,
            head_dim=16,
            mem_freq=15,
            use_kernel=False,
            gate=GateConfig(granularity=GateGranularity.elementwise),
        )


def test_config_build_and_validation():
    # vec_dim without shared_vector_landmark -> error
    with pytest.raises(OLMoConfigurationError):
        AttentionConfig(
            name=AttentionType.fast_landmark, n_heads=4, head_dim=16, mem_freq=15, vec_dim=8
        ).build(d_model=64, layer_idx=0, n_layers=1)

    # valid build produces the class with the configured vec_dim and widened w_out
    m = AttentionConfig(
        name=AttentionType.shared_vector_landmark,
        n_heads=4,
        n_kv_heads=4,
        head_dim=16,
        mem_freq=15,
        vec_dim=8,
        bias=False,
    ).build(d_model=64, layer_idx=0, n_layers=1)
    assert isinstance(m, SharedVectorLandmarkAttention)
    assert m.vec_dim == 8
    assert m.w_out.in_features == 4 * 16  # unchanged base shape
    assert m.w_out_vec.in_features == 4 * 8


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
def test_kernel_packing_head_dim_matches_eager():
    """The fused-kernel ``head_dim`` output (``_attn_core`` with ``doc_id``) must match the eager
    dense ``head_dim`` output (``_main_dense`` with the equivalent ``cu_doc_lens``), on a batch of two
    rows with distinct document layouts. (``_shared_vector_tail`` is shared by both paths -- see the
    CPU ``test_packing_*`` tests above for its doc-aware correctness -- so only the ``head_dim``
    branch differs here and needs a kernel-specific check.)
    """
    torch.manual_seed(0)
    mem_freq = 15
    block_size = mem_freq + 1
    m = SharedVectorLandmarkAttention(
        d_model=64,
        n_heads=4,
        n_kv_heads=4,
        head_dim=16,
        mem_freq=mem_freq,
        vec_dim=8,
        use_kernel=True,
        bias=False,
    ).cuda()

    B, T = 2, block_size * 4
    # Row 0: docs [2 blocks, 2 blocks]; row 1: docs [1 block, 3 blocks].
    cu_doc_lens = torch.tensor(
        [0, 2 * block_size, T, T + block_size, 2 * T], dtype=torch.int32, device="cuda"
    )
    doc_id = build_block_doc_id(cu_doc_lens, B, T, block_size)

    q, k, v = _qkv(m, B=B, T=T, dtype=torch.bfloat16)
    q, k, v = q.cuda(), k.cuda(), v.cuda()

    main_kernel = m._attn_core(q, k, v, doc_id=doc_id)
    main_eager = m._main_dense(q, k, v, cu_doc_lens=cu_doc_lens)

    torch.testing.assert_close(main_kernel, main_eager, rtol=1e-2, atol=1e-2)
