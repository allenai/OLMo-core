"""Resuming a packed mixture must not rebuild the batches it already consumed.

Before this, ``MixtureDataLoader`` resumed by pulling and discarding ``batches_processed``
real batches -- full image decode, preprocess and 2D-knapsack packing -- so resume cost
grew linearly with the resume step. On the multi-image tier that outran the 60-minute
collective timeout: a 32-GPU ``image-only-v9`` run resumed at step 10000, emitted no steps
for two hours, and died on
``Watchdog caught collective operation timeout ... ran for 3600061 milliseconds``.

The fix skips the *ref* stream instead, which is pure RNG arithmetic, and never touches the
examples those refs point at. These tests pin the two properties that makes correct:
the skip is cheap, and it lands on the same refs the replay would have.
"""

import numpy as np
import pytest

from olmo_core.data.multimodal.packed_mixture_iterable import (
    _CountingRefIter,
    iter_rank_mixture_refs,
)
from olmo_core.data.multimodal.packing import REF_CURSOR_KEY


def _refs(n, *, dp_rank=0, dp_world_size=1):
    return list(
        __import__("itertools").islice(
            iter_rank_mixture_refs(
                seed=7,
                epoch=1,
                weights=[0.5, 0.5],
                sizes=[1000, 1000],
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                epoch_instances=4000,
            ),
            n,
        )
    )


def test_counting_ref_iter_counts_and_skips():
    it = _CountingRefIter(iter(range(100)))
    assert it.count == 0
    next(it), next(it)
    assert it.count == 2
    it.skip(10)
    assert it.count == 12
    # skip() must actually consume, not just bump the counter
    assert next(it) == 12


def test_skip_lands_on_the_same_ref_as_replay():
    # The whole point: fast-forwarding N refs must reach exactly where consuming N one by
    # one would have. If these diverge, a resumed run silently reads different data.
    replayed = _CountingRefIter(iter(_refs(50)))
    for _ in range(30):
        next(replayed)
    skipped = _CountingRefIter(iter(_refs(50)))
    skipped.skip(30)
    assert replayed.count == skipped.count == 30
    assert next(replayed) == next(skipped)


def test_skip_does_not_touch_the_underlying_examples():
    # Cheapness is the property under test: skipping must not pull examples. A ref stream
    # that explodes if its *examples* are loaded stands in for image decode/preprocess.
    loaded = []

    def example_stream(ref_iter):
        for ref in ref_iter:
            loaded.append(ref)
            yield ref

    counter = _CountingRefIter(iter(range(1000)))
    counter.skip(500)
    assert loaded == []  # nothing loaded during the skip
    gen = example_stream(counter)
    assert next(gen) == 500  # and we resume at the right place
    assert loaded == [500]


@pytest.mark.parametrize("dp_world_size", [1, 4])
def test_ref_stream_is_deterministic_across_calls(dp_world_size):
    # Skipping is only sound because the ref stream is reproducible for a given
    # (seed, epoch): the resumed process regenerates it rather than reading it back.
    a = _refs(40, dp_rank=1 % dp_world_size, dp_world_size=dp_world_size)
    b = _refs(40, dp_rank=1 % dp_world_size, dp_world_size=dp_world_size)
    assert a == b


def test_ref_cursor_key_is_not_a_model_input_name():
    # The cursor rides on the pack and through the collator, so its key must not collide
    # with anything the train module forwards to the model.
    assert REF_CURSOR_KEY.startswith("_")
    for reserved in (
        "input_ids",
        "labels",
        "loss_masks",
        "position_ids",
        "token_type_ids",
        "images",
        "pooled_patches_idx",
        "subsegment_ids",
        "example_ids",
    ):
        assert REF_CURSOR_KEY != reserved


def test_collator_passes_cursor_through_and_takes_the_max():
    from olmo_core.data.multimodal.collator import MultimodalCollator

    def ex(n_tok, cursor=None):
        d = {
            "input_ids": np.arange(n_tok, dtype=np.int64),
            "labels": np.arange(n_tok, dtype=np.int64),
            "loss_masks": np.ones(n_tok, dtype=np.float32),
            "position_ids": np.arange(n_tok, dtype=np.int64),
            "token_type_ids": np.zeros(n_tok, dtype=np.int64),
            "images": np.zeros((1, 2, 3), dtype=np.float32),
            "pooled_patches_idx": np.zeros((1, 2), dtype=np.int64),
        }
        if cursor is not None:
            d[REF_CURSOR_KEY] = cursor
        return d

    collator = MultimodalCollator(pad_token_id=0, pad_sequence_length=8)
    batch = collator([ex(4, cursor=11), ex(6, cursor=17)])
    # Max, because the loader must resume past everything any worker already consumed.
    assert batch[REF_CURSOR_KEY] == 17

    # Absent on every pack (e.g. the single-process path) => key omitted entirely.
    assert REF_CURSOR_KEY not in collator([ex(4), ex(6)])
