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
