"""Occupancy accounting: the padded-vs-real bookkeeping SpeedMonitor reports.

Reported TPS counts padding, because the collator pads every pack to a fixed
``pad_sequence_length`` and the ViT runs on a crop tensor padded to the rank-batch max.
These tests pin the two signals that make the waste recoverable downstream: pad tokens
carry ``example_ids == -1``, and ``n_real_crops`` records the unpadded crop count.
"""

import numpy as np

from olmo_core.data.multimodal import MultimodalCollatorConfig

_SEQ = 512
_PATCH_DIM = 588
_N_PATCHES = 729


def _example(n_tokens: int, n_crops: int, example_ids: np.ndarray | None = None):
    return dict(
        input_ids=np.ones(n_tokens, dtype=np.int64),
        labels=np.ones(n_tokens, dtype=np.int64),
        loss_masks=np.ones(n_tokens, dtype=np.float32),
        position_ids=np.arange(n_tokens, dtype=np.int64),
        token_type_ids=np.zeros(n_tokens, dtype=np.int64),
        images=np.zeros((n_crops, _N_PATCHES, _PATCH_DIM), dtype=np.float32),
        pooled_patches_idx=np.full((max(n_crops, 0), 4), -1, dtype=np.int64),
        example_ids=(
            example_ids if example_ids is not None else np.zeros(n_tokens, dtype=np.int64)
        ),
    )


def _collate(examples):
    return MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=_SEQ).build()(examples)


def test_n_real_crops_records_unpadded_counts():
    """The crop axis is padded to the batch max, so one big example inflates the rest."""
    batch = _collate([_example(100, 2), _example(100, 9), _example(100, 0)])

    # Every example is padded up to the largest crop count in the batch...
    assert tuple(batch["images"].shape) == (3, 9, _N_PATCHES, _PATCH_DIM)
    # ...but the real counts survive, so the waste is measurable.
    np.testing.assert_array_equal(batch["n_real_crops"].numpy(), [2, 9, 0])

    real, padded = int(batch["n_real_crops"].sum()), 3 * 9
    assert real == 11 and padded == 27
    assert abs(real / padded - 11 / 27) < 1e-9


def test_text_only_batch_still_pays_for_one_dummy_crop():
    """A wholly text-only batch keeps the vision path alive on a zero crop."""
    batch = _collate([_example(64, 0), _example(64, 0)])

    assert tuple(batch["images"].shape) == (2, 1, _N_PATCHES, _PATCH_DIM)
    # The dummy crop is padding by construction: no example contributed one.
    np.testing.assert_array_equal(batch["n_real_crops"].numpy(), [0, 0])


def test_token_occupancy_is_recoverable_from_example_ids():
    """Pad positions are ``-1``, which is what SpeedMonitor counts against."""
    batch = _collate([_example(100, 1), _example(300, 1)])

    assert tuple(batch["input_ids"].shape) == (2, _SEQ)
    useful = int((batch["example_ids"] >= 0).sum())
    assert useful == 400  # not 2 * 512
    assert abs(useful / batch["input_ids"].numel() - 400 / 1024) < 1e-9


def test_packed_rows_count_every_example_not_just_the_row():
    """With packing, one row holds several examples; occupancy is still per-token."""
    packed_ids = np.concatenate([np.zeros(120, dtype=np.int64), np.ones(80, dtype=np.int64)])
    batch = _collate([_example(200, 4, example_ids=packed_ids)])

    useful = int((batch["example_ids"] >= 0).sum())
    assert useful == 200
    # max(example_ids)+1 is the per-row example count SpeedMonitor uses.
    assert int(batch["example_ids"].amax()) + 1 == 2
