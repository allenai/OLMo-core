"""Bucketed sequence padding.

The crop budget usually closes a pack well before the token budget does, so a batch's real
content can be a fraction of ``pad_sequence_length`` -- and every padded position still
costs a full pass through QKV, MLP, norms and RoPE. ``pad_multiple_of`` pads to a bucket
instead, bounding the number of distinct shapes so torch.compile sees a small reusable set
of graphs.
"""

import numpy as np
import pytest

from olmo_core.data.multimodal import MultimodalCollatorConfig

_SEQ = 16384
_PATCH_DIM = 588
_N_PATCHES = 729


def _example(n_tokens: int, n_crops: int = 1):
    return dict(
        input_ids=np.ones(n_tokens, dtype=np.int64),
        labels=np.ones(n_tokens, dtype=np.int64),
        loss_masks=np.ones(n_tokens, dtype=np.float32),
        position_ids=np.arange(n_tokens, dtype=np.int64),
        token_type_ids=np.zeros(n_tokens, dtype=np.int64),
        images=np.zeros((n_crops, _N_PATCHES, _PATCH_DIM), dtype=np.float32),
        pooled_patches_idx=np.full((n_crops, 4), -1, dtype=np.int64),
        example_ids=np.zeros(n_tokens, dtype=np.int64),
    )


def _collate(examples, bucket=None):
    return MultimodalCollatorConfig(
        pad_token_id=0, pad_sequence_length=_SEQ, pad_multiple_of=bucket
    ).build()(examples)


def test_default_still_pads_to_the_fixed_length():
    assert _collate([_example(1000)])["input_ids"].shape[1] == _SEQ


@pytest.mark.parametrize(
    "real,bucket,expected",
    [
        (1000, 2048, 2048),
        (2048, 2048, 2048),  # exact multiple must not round up a whole bucket
        (2049, 2048, 4096),
        (5000, 4096, 8192),
        (100, 16384, 16384),
    ],
)
def test_rounds_up_to_the_bucket(real, bucket, expected):
    assert _collate([_example(real)], bucket=bucket)["input_ids"].shape[1] == expected


def test_never_exceeds_pad_sequence_length():
    """A near-full pack must still land exactly on the cap, not a bucket above it."""
    batch = _collate([_example(_SEQ - 5)], bucket=4096)
    assert batch["input_ids"].shape[1] == _SEQ


def test_over_length_example_is_truncated_not_grown():
    batch = _collate([_example(_SEQ + 500)], bucket=4096)
    assert batch["input_ids"].shape[1] == _SEQ


def test_batch_uses_the_longest_example():
    batch = _collate([_example(500), _example(3000), _example(900)], bucket=2048)
    assert batch["input_ids"].shape[1] == 4096


def test_bucketing_preserves_real_tokens_and_pad_marking():
    """Shrinking the sequence must not change content, only how much padding follows."""
    full = _collate([_example(1500)])
    bucketed = _collate([_example(1500)], bucket=2048)

    np.testing.assert_array_equal(
        full["input_ids"][0, :1500].numpy(), bucketed["input_ids"][0, :1500].numpy()
    )
    # Same real-token count; only the padded tail differs in length.
    assert (
        int((full["example_ids"] >= 0).sum()) == int((bucketed["example_ids"] >= 0).sum()) == 1500
    )
    assert int((bucketed["example_ids"] < 0).sum()) == 2048 - 1500
