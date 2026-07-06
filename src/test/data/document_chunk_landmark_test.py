"""Tests for the document-chunked data primitives (special-token boundaries + first-fit window
packing) and their end-to-end consistency with runtime chunk_id reconstruction and the
DocumentLandmarkAttention / DocumentChunkedAttention models."""

import torch

from olmo_core.data.document_chunk_landmark import (
    ChunkSegment,
    emit_document_chunk_dense,
    emit_document_chunk_landmark,
    find_chunk_spans,
)
from olmo_core.nn.attention.chunked_mask import (
    FREE_CHUNK_ID,
    PAD_CHUNK_ID,
    build_chunk_ids_from_tokens,
)

# Marker / special ids used by the tests (mirroring real Qwen3 box_start/box_end + reserved landmark).
DS, DE, MID, PAD, EOS = 301, 302, 300, 303, 304


def _emit(segs, mem_freq):
    return emit_document_chunk_landmark(segs, mem_freq=mem_freq, mem_id=MID, pad_id=PAD)


def test_find_chunk_spans():
    ids = [DS, 10, 11, DE, 99, DS, 12, DE, 7]
    assert find_chunk_spans(ids, DS, DE) == [(0, 3), (5, 7)]


def test_dense_emit_flattens_segments():
    segs = [
        ChunkSegment([5, 6], [False, False], False),  # FREE instruction
        ChunkSegment([DS, 10, DE], [False, False, False], True),  # context doc (markers included)
        ChunkSegment([40, 41], [True, True], False),  # FREE answer
    ]
    ids, mask = emit_document_chunk_dense(segs)
    assert ids == [5, 6, DS, 10, DE, 40, 41]
    assert mask == [False, False, False, False, False, True, True]
    # roles reconstruct: FREE, then a 1-doc context span (markers incl), then FREE answer.
    roles = build_chunk_ids_from_tokens(torch.tensor([ids + [EOS]]), DS, DE, EOS)[0].tolist()
    assert roles == [-1, -1, 0, 0, 0, -1, -1, -1]


def test_landmark_emit_block_aligned_and_landmarked():
    block = 4  # mem_freq=3
    ids, mask = _emit(
        [
            ChunkSegment([DS, 10, 11, DE], [False] * 4, True),
            ChunkSegment([40], [True], False),
        ],
        mem_freq=3,
    )
    assert len(ids) % block == 0 and len(mask) == len(ids)
    assert all(t == MID for t in ids[block - 1 :: block])  # landmark at every block-end
    assert not any(m for t, m in zip(ids, mask) if t in (MID, PAD))  # landmark/pad never supervised


def test_landmark_packs_small_docs_into_one_window():
    # mem_freq=6: two 3-token docs fit together in one window.
    ids, _ = _emit(
        [
            ChunkSegment([DS, 10, DE], [False] * 3, True),
            ChunkSegment([DS, 11, DE], [False] * 3, True),
            ChunkSegment([40], [True], False),
        ],
        mem_freq=6,
    )
    block = 7
    assert len(ids) == 2 * block
    # Window 0 holds BOTH docs (their 6 tokens) then a landmark; window 1 holds the FREE answer.
    assert ids[:block] == [DS, 10, DE, DS, 11, DE, MID]
    roles = build_chunk_ids_from_tokens(torch.tensor([ids + [EOS]]), DS, DE, EOS, pad_id=PAD)[
        0
    ].tolist()
    # Two distinct documents share the window but keep separate chunk ids.
    assert roles[0:3] == [0, 0, 0] and roles[3:6] == [1, 1, 1]
    assert roles[8:13] == [PAD_CHUNK_ID] * 5  # window-1 fill padding -> PAD (non-attendable)


def test_landmark_overflow_starts_new_window():
    # mem_freq=4: a 3-token doc, then a 3-token doc -> the second cannot fit (3+3>4) -> new window.
    ids, _ = _emit(
        [
            ChunkSegment([DS, 10, DE], [False] * 3, True),
            ChunkSegment([DS, 11, DE], [False] * 3, True),
        ],
        mem_freq=4,
    )
    block = 5
    assert len(ids) == 2 * block
    assert ids[:3] == [DS, 10, DE] and ids[3] == PAD  # window 0: doc0 + pad fill
    assert ids[block : block + 3] == [DS, 11, DE]  # window 1: doc1


def test_landmark_large_doc_spans_whole_windows_from_boundary():
    # mem_freq=3: a 6-token doc (> one window) starts at a boundary and spans two whole windows.
    ids, _ = _emit([ChunkSegment([DS, 10, 11, 12, 13, DE], [False] * 6, True)], mem_freq=3)
    block = 4
    assert len(ids) == 2 * block
    assert ids == [DS, 10, 11, MID, 12, 13, DE, MID]
    roles = build_chunk_ids_from_tokens(torch.tensor([ids + [EOS]]), DS, DE, EOS, pad_id=PAD)[
        0
    ].tolist()
    # The whole doc span (box_start..box_end, incl. the interior landmark at pos 3) is chunk 0; only
    # the final landmark at pos 7 (after box_end) is FREE.
    assert roles[0:7] == [0] * 7 and roles[7] == FREE_CHUNK_ID


def test_full_pipeline_landmark_trains_without_nan():
    from olmo_core.nn.transformer import TransformerConfig

    vocab = 320
    ids, _ = _emit(
        [
            ChunkSegment([5, 6], [False, False], False),
            ChunkSegment([DS, 10, 11, DE], [False] * 4, True),
            ChunkSegment([DS, 20, DE], [False] * 3, True),
            ChunkSegment([40, 41, 42], [True, True, True], False),
        ],
        mem_freq=3,
    )
    seq = ids + [EOS]
    block = 4
    while len(seq) % block != 0:
        seq.append(PAD)
    input_ids = torch.tensor([seq])

    cfg = TransformerConfig.llama_like(
        d_model=32,
        vocab_size=vocab,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        document_landmark=True,
        mem_freq=3,
    )
    model = cfg.build(init_device="cpu")
    model.enable_document_chunk_attention(doc_start_id=DS, doc_end_id=DE, eos_id=EOS, pad_id=PAD)
    out = model(input_ids, labels=input_ids.clone())
    loss = out.loss if hasattr(out, "loss") else out
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_full_pipeline_dense_trains_without_nan():
    from olmo_core.nn.transformer import TransformerConfig

    vocab = 320
    ids, _ = emit_document_chunk_dense(
        [
            ChunkSegment([5, 6], [False, False], False),
            ChunkSegment([DS, 10, 11, DE], [False] * 4, True),
            ChunkSegment([DS, 20, DE], [False] * 3, True),
            ChunkSegment([40, 41], [True, True], False),
        ]
    )
    input_ids = torch.tensor([ids + [EOS]])

    cfg = TransformerConfig.llama_like(
        d_model=32,
        vocab_size=vocab,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        document_chunked=True,
        cross_doc_mode="chunked",
    )
    model = cfg.build(init_device="cpu")
    model.enable_document_chunk_attention(doc_start_id=DS, doc_end_id=DE, eos_id=EOS)
    out = model(input_ids, labels=input_ids.clone())
    loss = out.loss if hasattr(out, "loss") else out
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------
# Test-time (inference) mask application: the chunked mask is reconstructed from the boundary tokens
# and applied on every forward() -- the exact path the native eval harness uses (gm.model(input_ids)),
# not just during training.
# ---------------------------------------------------------------------------

DSx, DEx, EOSx = 301, 302, 304  # box_start / box_end / eos ids for the inference tests


def _doc_model(variant, vocab=320, mem_freq=3, enable=True):
    from olmo_core.nn.transformer import TransformerConfig

    kw = dict(d_model=32, vocab_size=vocab, n_layers=2, n_heads=4, n_kv_heads=2)
    if variant == "dense":
        cfg = TransformerConfig.llama_like(document_chunked=True, cross_doc_mode="chunked", **kw)
    elif variant == "landmark":
        cfg = TransformerConfig.llama_like(document_landmark=True, mem_freq=mem_freq, **kw)
    else:  # full (standard attention baseline)
        cfg = TransformerConfig.llama_like(**kw)
    model = cfg.build(init_device="cpu")
    if enable and variant != "full":
        model.enable_document_chunk_attention(
            doc_start_id=DSx, doc_end_id=DEx, eos_id=EOSx, pad_id=(PAD if variant == "landmark" else None)
        )
    model.eval()
    return model


def test_dense_mask_isolation_at_inference():
    # In eval()/no_grad (the native-eval forward path), editing a token of document A must NOT change a
    # token of document B (chunks isolated), but MUST change the trailing FREE token (bridge).
    # Layout: [free, <s>,docA,<e>, <s>,docB,<e>, free] -> pos 5 is docB content, pos 7 is FREE.
    seq = [5, DSx, 10, DEx, DSx, 20, DEx, 6]
    model = _doc_model("dense")
    with torch.no_grad():
        base = model(torch.tensor([seq]))
        alt = seq.copy()
        alt[2] = 11  # edit document A's content token
        edited = model(torch.tensor([alt]))
    assert torch.allclose(base[0, 5], edited[0, 5], atol=1e-5)  # docB isolated from docA
    assert not torch.allclose(base[0, 7], edited[0, 7], atol=1e-4)  # FREE query bridges


def test_full_attention_baseline_is_not_isolated():
    # The control: with standard attention (no chunked mask) the same docA edit DOES reach docB.
    seq = [5, DSx, 10, DEx, DSx, 20, DEx, 6]
    model = _doc_model("full")
    with torch.no_grad():
        base = model(torch.tensor([seq]))
        alt = seq.copy()
        alt[2] = 11
        edited = model(torch.tensor([alt]))
    assert not torch.allclose(base[0, 5], edited[0, 5], atol=1e-4)  # full attention: docB sees docA


def test_chunk_mask_changes_inference_output_dense_and_landmark():
    # Enabling the document-chunk mask changes the model's inference output vs the same weights with the
    # mask OFF -- i.e. the mask is genuinely applied at test time (for both variants).
    for variant in ("dense", "landmark"):
        model = _doc_model(variant)
        if variant == "dense":
            seq = [5, DSx, 10, DEx, DSx, 20, DEx, 6]
        else:  # landmark: build the real block-aligned packed layout
            seq, _ = emit_document_chunk_landmark(
                [
                    ChunkSegment([5], [False], False),
                    ChunkSegment([DSx, 10, DEx], [False] * 3, True),
                    ChunkSegment([DSx, 20, DEx], [False] * 3, True),
                    ChunkSegment([6], [False], False),
                ],
                mem_freq=3,
                mem_id=MID,
                pad_id=PAD,
            )
        ids = torch.tensor([seq])
        with torch.no_grad():
            on = model(ids)
            model._document_chunk_attention = None  # mask OFF: chunk_ids no longer reconstructed
            off = model(ids)
        assert torch.isfinite(on).all() and torch.isfinite(off).all()
        assert not torch.allclose(on, off, atol=1e-4), f"{variant}: mask had no effect at inference"


# ---------------------------------------------------------------------------
# Top-k landmark retrieval at inference (DocumentLandmarkAttention / eager LandmarkAttention).
# ---------------------------------------------------------------------------


def test_topk_landmark_retrieval_masks_all_but_best():
    # Deterministic: with top_k=1 only the highest-scoring landmark COLUMN survives per query; the
    # others are set to finfo.min (so the grouped softmax zeroes their blocks).
    from olmo_core.nn.attention import AttentionConfig, AttentionType, DocumentLandmarkAttention
    from olmo_core.nn.rope import RoPEConfig, RoPEType

    cfg = AttentionConfig(
        name=AttentionType.document_landmark, n_heads=2, n_kv_heads=1, head_dim=8, bias=False,
        mem_freq=3, cross_doc_mode="chunked", rope=RoPEConfig(name=RoPEType.default, theta=10_000),
    )
    attn = cfg.build(16, layer_idx=0, n_layers=1)
    assert isinstance(attn, DocumentLandmarkAttention)
    attn.set_eval_top_k(1)
    T = 12  # block_size 4 -> landmark columns at 3, 7, 11
    a = torch.zeros(1, 2, T, T)
    a[..., 3] = 1.0
    a[..., 7] = 5.0  # highest -> kept
    a[..., 11] = 2.0
    out = attn._apply_topk_landmark_retrieval(a.clone(), T)
    fmin = torch.finfo(out.dtype).min
    assert (out[..., 7] == 5.0).all()  # best landmark kept
    assert (out[..., 3] == fmin).all() and (out[..., 11] == fmin).all()  # others masked
    # content columns are untouched (the grouped softmax gates them via the landmarks)
    assert (out[..., 0] == 0.0).all()


def test_landmark_topk_noop_when_k_ge_blocks_and_changes_when_small():
    # k >= num_landmarks is exactly the exact path; k=1 changes the inference output. Train unaffected.
    seq, _ = emit_document_chunk_landmark(
        [ChunkSegment([5], [False], False)]
        + [ChunkSegment([DS, 10 + i, DE], [False] * 3, True) for i in range(4)]
        + [ChunkSegment([6], [False], False)],
        mem_freq=3,
        mem_id=MID,
        pad_id=PAD,
    )
    ids = torch.tensor([seq])
    model = _doc_model("landmark")
    with torch.no_grad():
        exact = model(ids)
        assert model.set_landmark_eval_top_k(1000) >= 1  # k huge -> no-op
        big = model(ids)
        model.set_landmark_eval_top_k(1)
        one = model(ids)
        model.set_landmark_eval_top_k(None)
        restored = model(ids)
    assert torch.allclose(exact, big, atol=1e-5)  # k >= n_blocks == exact
    assert torch.allclose(exact, restored, atol=1e-5)  # None restores exact
    assert torch.isfinite(one).all()
    assert not torch.allclose(exact, one, atol=1e-4)  # k=1 genuinely changes the output


@__import__("pytest").mark.gpu
def test_document_landmark_fused_kernel_matches_eager_fwd_and_grad():
    # The fused kernel path (use_kernel=True) must match the eager grouped-softmax + chunked-mask path
    # for BOTH the forward and the input gradients (the regression that gates enabling the kernel).
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    from olmo_core.nn.attention.landmark import landmark_grouped_softmax
    from olmo_core.nn.attention.landmark_fast import fused_landmark_attention_fast, has_landmark_kernel

    if not has_landmark_kernel():
        pytest.skip("fused landmark kernel unavailable")
    from olmo_core.nn.attention import AttentionConfig, AttentionType
    from olmo_core.nn.rope import RoPEConfig, RoPEType

    torch.manual_seed(0)
    dev = "cuda"
    B, Hh, Dd, mem_freq = 1, 4, 32, 15  # head_dim>=16 and block_size>=16 (kernel tl.dot constraint)
    bs = mem_freq + 1
    T = bs * 6
    q0, k0, v0 = (torch.randn(B, Hh, T, Dd, device=dev) * 0.5 for _ in range(3))
    scale = Dd**-0.5
    is_mem = torch.arange(T, device=dev) % bs == bs - 1
    cids = torch.full((B, T), -1, dtype=torch.long, device=dev)
    cids[0, 0:32] = 0
    cids[0, 32:64] = 1
    go = torch.randn(B, Hh, T, Dd, device=dev)

    qk, kk, vk = (x.clone().requires_grad_() for x in (q0, k0, v0))
    ok = fused_landmark_attention_fast(qk, kk, vk, is_mem, scale, bs, chunk_ids=cids)
    (ok * go).sum().backward()

    qe, ke, ve = (x.clone().requires_grad_() for x in (q0, k0, v0))
    cfg = AttentionConfig(
        name=AttentionType.document_landmark, n_heads=Hh, n_kv_heads=Hh, head_dim=Dd, bias=False,
        mem_freq=mem_freq, cross_doc_mode="chunked", rope=RoPEConfig(name=RoPEType.default, theta=10000),
    )
    mod = cfg.build(Hh * Dd, layer_idx=0, n_layers=1).to(dev)
    mod._chunk_ids = cids
    am, ism, lsm = mod._landmark_masks(T, torch.device(dev), torch.float32, batch_size=B)
    attn = (qe @ ke.transpose(-1, -2)) * scale + am
    attn = torch.maximum(attn, torch.tensor(torch.finfo(attn.dtype).min, device=dev))
    oe = landmark_grouped_softmax(attn, -1, ism.expand(B, Hh, T, T), lsm.expand(B, 1, T, T)) @ ve
    (oe * go).sum().backward()

    assert torch.allclose(ok, oe, atol=1e-4), (ok - oe).abs().max()
    for gk, ge in ((qk.grad, qe.grad), (kk.grad, ke.grad), (vk.grad, ve.grad)):
        assert torch.allclose(gk, ge, atol=1e-3, rtol=1e-2), (gk - ge).abs().max()
