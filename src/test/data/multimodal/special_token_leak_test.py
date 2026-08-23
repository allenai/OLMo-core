"""Corpus text containing special-token strings must never reach the loss.

Regression tests for the failure that killed the first 8-GPU ``single-image-only-v11``
production run at step ~171: two OmniScience captions (of 150,000) describe
``<im_start>``/``<im_end>`` tags in prose, ``tokenizer.encode`` parsed those runs as the
control tokens 151936/151937, and because a caption is a *supervised* response the ids
survived the ``loss_masks > 0`` filter and tripped
``Assertion 'cur_target >= 0 && cur_target < n_classes' failed`` inside
``F.cross_entropy``, aborting the CUDA context on all 8 ranks.
"""

import numpy as np
import pytest

from olmo_core.data.multimodal.sequence_builder import (
    OutOfRangeLabelError,
    check_supervised_labels,
)
from olmo_core.nn.vision.molmo2_tokens import (
    IM_END_ID,
    IM_PATCH_ID,
    IM_START_ID,
    IMAGE_TOKEN_IDS,
    LM_VOCAB_SIZE,
    NON_LM_TOKEN_IDS,
)

# Verbatim from the OmniScience caption that took the run down.
POISONED_CAPTION = (
    "The visual layout uses clear demarcation with <im_start> and <im_end> tags to "
    "define sections, and the entire prompt is presented in a monospaced font."
)


def test_non_lm_token_ids_are_all_outside_the_lm_head():
    # The premise of the whole bug: the tokenizer is wider than the LM head.
    assert NON_LM_TOKEN_IDS
    assert all(i >= LM_VOCAB_SIZE for i in NON_LM_TOKEN_IDS)
    # Every image-block token is one of them, which is why they are only safe at
    # loss-weight-0 positions.
    assert IMAGE_TOKEN_IDS <= NON_LM_TOKEN_IDS


def test_guard_rejects_supervised_control_token():
    input_ids = np.array([10, 11, IM_START_ID, 12], dtype=np.int64)
    labels = np.array([11, IM_START_ID, 12, 13], dtype=np.int64)
    loss_masks = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    with pytest.raises(OutOfRangeLabelError, match="outside the LM head"):
        check_supervised_labels(input_ids, labels, loss_masks, source="omniscience")


def test_guard_allows_control_tokens_at_unsupervised_positions():
    # The normal case: an image block sits in the prompt at loss weight 0. The ids are
    # in `labels` by construction but never reach the loss, so this must not raise.
    input_ids = np.array([IM_START_ID, IM_PATCH_ID, IM_END_ID, 42, 43], dtype=np.int64)
    labels = np.array([IM_PATCH_ID, IM_END_ID, 42, 43, 44], dtype=np.int64)
    loss_masks = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    check_supervised_labels(input_ids, labels, loss_masks, source="pixmo_cap")


def test_guard_rejects_stray_control_token_in_input_ids():
    # A control token outside an image block corrupts token_type_ids even though it
    # never reaches the loss.
    stray = max(NON_LM_TOKEN_IDS - IMAGE_TOKEN_IDS)
    input_ids = np.array([10, stray, 12], dtype=np.int64)
    labels = np.array([stray, 12, 13], dtype=np.int64)
    loss_masks = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(OutOfRangeLabelError, match="token_type_ids"):
        check_supervised_labels(input_ids, labels, loss_masks, source="finevision")


@pytest.mark.parametrize(
    "text",
    [
        "A bar chart showing revenue by quarter, with Q3 highest.",
        "The answer is 42. <think>reasoning</think> Final: 42",
        "Solve for x: 3x + 5 = 20, so x = 5.",
    ],
)
def test_encode_corpus_text_is_identical_for_ordinary_text(text):
    # Parity guarantee: the fix must be a no-op for text with no special-token strings,
    # otherwise it would perturb every mm_olmo-parity source.
    tok = pytest.importorskip("transformers").AutoTokenizer.from_pretrained(
        "allenai/Molmo2-4B", trust_remote_code=True
    )
    from olmo_core.data.multimodal.sft_common import encode_corpus_text

    assert encode_corpus_text(tok, text) == tok.encode(text, add_special_tokens=False)


def test_encode_corpus_text_neutralizes_the_poisoned_caption():
    tok = pytest.importorskip("transformers").AutoTokenizer.from_pretrained(
        "allenai/Molmo2-4B", trust_remote_code=True
    )
    from olmo_core.data.multimodal.sft_common import encode_corpus_text

    # The old call emits the out-of-range control tokens ...
    before = tok.encode(POISONED_CAPTION, add_special_tokens=False)
    assert {IM_START_ID, IM_END_ID} <= set(before)

    # ... the new one encodes them as ordinary text.
    after = encode_corpus_text(tok, POISONED_CAPTION)
    assert not (set(after) & NON_LM_TOKEN_IDS)
    assert all(i < LM_VOCAB_SIZE for i in after)
    # Same text, just spelled out in more (ordinary) tokens.
    assert len(after) > len(before)
