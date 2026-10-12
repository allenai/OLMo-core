from pathlib import Path

import pytest

from olmo_core.data.composable.landmark_packing_instance_source import (
    LandmarkPackingInstanceSource,
)
from olmo_core.data.composable.token_source import InMemoryDocumentSource
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.data.types import LandmarkPackingStrategy
from olmo_core.exceptions import OLMoConfigurationError


@pytest.fixture
def tokenizer() -> TokenizerConfig:
    # No BOS; EOS marks document boundaries (Qwen3-style).
    return TokenizerConfig(vocab_size=100, eos_token_id=99, pad_token_id=98)


def _packed(tokens, tmp_path, tokenizer, **kw):
    docs = InMemoryDocumentSource(tokens=tokens, tokenizer=tokenizer, work_dir=tmp_path)
    return LandmarkPackingInstanceSource(
        docs,
        sequence_length=kw.pop("sequence_length", 16),
        mem_freq=kw.pop("mem_freq", 3),
        mem_id=kw.pop("mem_id", 50),
        pad_id=tokenizer.pad_token_id,
        work_dir=tmp_path,
        **kw,
    )


def test_block_aligned_doc_lens_and_landmarks(tmp_path: Path, tokenizer: TokenizerConfig):
    eos, pad, mem = tokenizer.eos_token_id, tokenizer.pad_token_id, 50
    mem_freq, block_size, seq_len = 3, 4, 16
    # Doc A: 3 content tokens (+EOS=4) -> ceil(4/3)=2 blocks -> 8 landmark tokens.
    # Doc B: 2 content tokens (+EOS=3) -> ceil(3/3)=1 block -> 4 landmark tokens.
    tokens = [1, 2, 3, eos] + [4, 5, eos]
    src = _packed(
        tokens, tmp_path, tokenizer, sequence_length=seq_len, mem_freq=mem_freq, mem_id=mem
    )

    assert len(src) == 1  # 8 + 4 = 12 <= 16, both docs in one window
    inst = src[0]
    ids, mask, doc_lens = (
        list(inst["input_ids"]),
        list(inst["label_mask"]),
        list(inst["doc_lens"]),
    )

    assert len(ids) == seq_len
    assert sum(doc_lens) == seq_len
    assert all(dl % block_size == 0 for dl in doc_lens)
    # Doc A (8) + Doc B (4) + tail pad (4).
    assert doc_lens == [8, 4, 4]

    # Landmark token sits at every periodic position; label_mask is False there.
    for p in range(seq_len):
        if p % block_size == block_size - 1:
            assert ids[p] == mem, p
            assert mask[p] is False, p

    # Doc A: [1,2,3,eos] padded to 6 (mem_freq*2) then landmarks every 3.
    assert ids[:8] == [1, 2, 3, mem, eos, pad, pad, mem]
    assert mask[:8] == [True, True, True, False, True, False, False, False]
    # Doc B: [4,5,eos] -> one block of 3 + landmark.
    assert ids[8:12] == [4, 5, eos, mem]
    assert mask[8:12] == [True, True, True, False]
    # Tail pad block (landmark kept at the block end for the periodic is_mem invariant).
    assert ids[12:16] == [pad, pad, pad, mem]
    assert mask[12:16] == [False, False, False, False]


@pytest.mark.parametrize("num_landmarks", [2, 3])
def test_multi_landmark_blocks(tmp_path: Path, tokenizer: TokenizerConfig, num_landmarks: int):
    """``num_landmarks`` landmark tokens per block, matching MultiCompressiveLandmarkAttention."""
    eos, pad, mem = tokenizer.eos_token_id, tokenizer.pad_token_id, 50
    mem_freq = 3
    block_size = mem_freq + num_landmarks
    seq_len = 4 * block_size
    # Doc A: 3 content (+EOS=4) -> ceil(4/3)=2 blocks. Doc B: 2 content (+EOS=3) -> 1 block.
    tokens = [1, 2, 3, eos] + [4, 5, eos]
    src = _packed(
        tokens,
        tmp_path,
        tokenizer,
        sequence_length=seq_len,
        mem_freq=mem_freq,
        mem_id=mem,
        num_landmarks=num_landmarks,
    )

    assert len(src) == 1
    inst = src[0]
    ids, mask, doc_lens = (
        list(inst["input_ids"]),
        list(inst["label_mask"]),
        list(inst["doc_lens"]),
    )

    assert len(ids) == seq_len
    assert sum(doc_lens) == seq_len
    # Doc A (2 blocks) + Doc B (1 block) + tail pad (1 block).
    assert doc_lens == [2 * block_size, block_size, block_size]

    # The last ``num_landmarks`` positions of every block are landmark tokens, loss-free.
    for p in range(seq_len):
        if p % block_size >= mem_freq:
            assert ids[p] == mem, p
            assert mask[p] is False, p
        else:
            assert ids[p] != mem, p

    lm = [mem] * num_landmarks
    lm_mask = [False] * num_landmarks
    assert ids[: 2 * block_size] == [1, 2, 3] + lm + [eos, pad, pad] + lm
    assert mask[: 2 * block_size] == [True, True, True] + lm_mask + [True, False, False] + lm_mask
    assert ids[2 * block_size : 3 * block_size] == [4, 5, eos] + lm
    assert ids[3 * block_size :] == [pad] * mem_freq + lm
    assert mask[3 * block_size :] == [False] * block_size


def test_multi_landmark_doc_lens_consumable_as_cu_doc_lens(
    tmp_path: Path, tokenizer: TokenizerConfig
):
    """Multi-landmark doc_lens stay aligned to the model's block_size = mem_freq + num_landmarks."""
    import torch

    from olmo_core.data.utils import get_cumulative_document_lengths
    from olmo_core.nn.attention.landmark import build_block_doc_id

    eos = tokenizer.eos_token_id
    mem_freq, num_landmarks = 3, 2
    block_size = mem_freq + num_landmarks  # 5
    seq_len = 4 * block_size  # 20
    src = _packed(
        [1, 2, 3, eos] + [4, 5, eos],
        tmp_path,
        tokenizer,
        sequence_length=seq_len,
        mem_freq=mem_freq,
        mem_id=50,
        num_landmarks=num_landmarks,
    )
    doc_lens = torch.tensor([list(src[0]["doc_lens"])], dtype=torch.int32)
    cu = get_cumulative_document_lengths(doc_lens)
    assert cu[-1].item() == seq_len
    doc_id = build_block_doc_id(cu, batch_size=1, seq_len=seq_len, block_size=block_size)
    assert doc_id.tolist() == [[0, 0, 1, 2]]  # blocks: docA, docA, docB, tail-pad


def test_rejects_bad_num_landmarks(tmp_path: Path, tokenizer: TokenizerConfig):
    with pytest.raises(OLMoConfigurationError):
        _packed(
            [1, 2, 3, tokenizer.eos_token_id],
            tmp_path,
            tokenizer,
            sequence_length=16,
            mem_freq=3,
            num_landmarks=0,
        )
    # sequence_length must be a multiple of mem_freq + num_landmarks (= 5), not mem_freq + 1.
    with pytest.raises(OLMoConfigurationError):
        _packed(
            [1, 2, 3, tokenizer.eos_token_id],
            tmp_path,
            tokenizer,
            sequence_length=16,
            mem_freq=3,
            num_landmarks=2,
        )


def test_greedy_packing_opens_new_window(tmp_path: Path, tokenizer: TokenizerConfig):
    eos = tokenizer.eos_token_id
    # Each doc: 3 content (+EOS=4) -> 2 blocks -> 8 landmark tokens. seq_len=16 fits exactly 2 docs.
    doc = [1, 2, 3, eos]
    src = _packed(doc * 3, tmp_path, tokenizer, sequence_length=16, mem_freq=3, mem_id=50)
    assert len(src) == 2  # docs [0,1] in window 0, doc [2] in window 1
    assert list(src[0]["doc_lens"]) == [8, 8]  # exactly fills, no tail pad
    assert list(src[1]["doc_lens"]) == [8, 8]  # one real doc (8) + tail pad (8)


def test_drops_overlong_documents_with_warning(tmp_path: Path, tokenizer: TokenizerConfig, caplog):
    eos = tokenizer.eos_token_id
    # A 20-content-token doc -> ceil(20/3)=7 blocks -> 28 landmark tokens > seq_len 16: dropped.
    long_doc = list(range(1, 21)) + [eos]
    short_doc = [1, 2, eos]
    import logging

    with caplog.at_level(logging.WARNING):
        src = _packed(
            long_doc + short_doc, tmp_path, tokenizer, sequence_length=16, mem_freq=3, mem_id=50
        )
    assert src._num_dropped == 1
    assert any(
        "dropped" in r.message and "increasing sequence_length" in r.message for r in caplog.records
    )
    # Only the short doc survives.
    assert len(src) == 1
    assert list(src[0]["doc_lens"]) == [4, 12]


def test_rejects_unaligned_sequence_length(tmp_path: Path, tokenizer: TokenizerConfig):
    with pytest.raises(OLMoConfigurationError):
        _packed(
            [1, 2, 3, tokenizer.eos_token_id], tmp_path, tokenizer, sequence_length=10, mem_freq=3
        )


def test_best_fit_decreasing_beats_next_fit_on_a_bad_order(
    tmp_path: Path, tokenizer: TokenizerConfig
):
    """Next-fit closes a window as soon as a document misses; BFD comes back and fills it."""
    eos = tokenizer.eos_token_id
    # mem_freq=3, block_size=4, seq_len=16 -> 4 blocks per window.
    # Doc sizes in blocks, in this order: 3, 2, 1. Total 6 blocks -> 2 windows is optimal.
    big = list(range(1, 9)) + [eos]  # 9 content tokens -> ceil(9/3)=3 blocks
    mid = [1, 2, 3, 4, eos]  # 5 content tokens -> 2 blocks
    small = [7, eos]  # 2 content tokens -> 1 block
    tokens = big + mid + small

    next_fit = _packed(
        tokens, tmp_path / "nf", tokenizer, sequence_length=16, mem_freq=3, mem_id=50
    )
    # next-fit: [big] (3 blocks, mid doesn't fit) | [mid, small] -> 2 windows, 1 wasted block.
    assert len(next_fit) == 2
    assert list(next_fit[0]["doc_lens"]) == [12, 4]  # big + tail pad

    bfd = _packed(
        tokens,
        tmp_path / "bfd",
        tokenizer,
        sequence_length=16,
        mem_freq=3,
        mem_id=50,
        packing_strategy=LandmarkPackingStrategy.best_fit_decreasing,
    )
    # BFD: big(3) + small(1) fills a window exactly; mid(2) takes the second.
    assert len(bfd) == 2
    assert list(bfd[0]["doc_lens"]) == [12, 4]  # big + small, no tail pad
    assert list(bfd[0]["input_ids"])[12:16] == [7, eos, tokenizer.pad_token_id, 50]


def test_best_fit_decreasing_invariants(tmp_path: Path, tokenizer: TokenizerConfig):
    """Every document appears once, windows never overflow, and doc_lens stay block-aligned."""
    import random

    eos = tokenizer.eos_token_id
    mem_freq, block_size, seq_len = 3, 4, 64
    random.seed(11)
    tokens: list = []
    n_docs = 200
    for _ in range(n_docs):
        # Up to 15 blocks of content -- always short enough to fit a 16-block window.
        tokens.extend([random.randint(1, 40) for _ in range(random.randint(1, 44))] + [eos])

    src = _packed(
        tokens,
        tmp_path,
        tokenizer,
        sequence_length=seq_len,
        mem_freq=mem_freq,
        mem_id=50,
        packing_strategy=LandmarkPackingStrategy.best_fit_decreasing,
    )
    assert src._num_dropped == 0

    packed_docs = [d for window in src._windows for d in window]
    assert sorted(packed_docs) == list(range(n_docs))  # each document exactly once

    for i in range(len(src)):
        inst = src[i]
        doc_lens = list(inst["doc_lens"])
        assert len(inst["input_ids"]) == seq_len
        assert sum(doc_lens) == seq_len
        assert all(dl % block_size == 0 and dl > 0 for dl in doc_lens)

    # BFD must not need more windows than next-fit on the same documents.
    next_fit = _packed(
        tokens, tmp_path / "nf", tokenizer, sequence_length=seq_len, mem_freq=mem_freq, mem_id=50
    )
    assert len(src) <= len(next_fit)


def test_packing_strategy_changes_the_fingerprint(tmp_path: Path, tokenizer: TokenizerConfig):
    """BFD must not reuse next-fit's cached shuffle indices -- but next-fit's hash can't move."""
    tokens = [1, 2, 3, tokenizer.eos_token_id] * 4
    default = _packed(tokens, tmp_path / "a", tokenizer, sequence_length=16, mem_freq=3, mem_id=50)
    explicit_next_fit = _packed(
        tokens,
        tmp_path / "b",
        tokenizer,
        sequence_length=16,
        mem_freq=3,
        mem_id=50,
        packing_strategy=LandmarkPackingStrategy.next_fit,
    )
    bfd = _packed(
        tokens,
        tmp_path / "c",
        tokenizer,
        sequence_length=16,
        mem_freq=3,
        mem_id=50,
        packing_strategy=LandmarkPackingStrategy.best_fit_decreasing,
    )
    assert default.fingerprint == explicit_next_fit.fingerprint
    assert bfd.fingerprint != default.fingerprint


def test_doc_lens_consumable_as_cu_doc_lens(tmp_path: Path, tokenizer: TokenizerConfig):
    """The emitted doc_lens must form a valid block-aligned cu_doc_lens for landmark attention."""
    import torch

    from olmo_core.data.utils import get_cumulative_document_lengths
    from olmo_core.nn.attention.landmark import build_block_doc_id

    eos = tokenizer.eos_token_id
    src = _packed(
        [1, 2, 3, eos] + [4, 5, eos], tmp_path, tokenizer, sequence_length=16, mem_freq=3, mem_id=50
    )
    doc_lens = torch.tensor([list(src[0]["doc_lens"])], dtype=torch.int32)
    cu = get_cumulative_document_lengths(doc_lens)
    assert cu[-1].item() == 16
    # build_block_doc_id validates block alignment (block_size = 4) and totals.
    doc_id = build_block_doc_id(cu, batch_size=1, seq_len=16, block_size=4)
    assert doc_id.tolist() == [[0, 0, 1, 2]]  # blocks: docA, docA, docB, tail-pad
