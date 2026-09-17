"""SpeedMonitorCallback's padding-aware accounting.

The collator's side of this is covered in
``src/test/data/multimodal/pack_occupancy_test.py``; these drive the callback itself,
which is where the arithmetic that reaches W&B actually lives.
"""

import numpy as np
import torch

from olmo_core.train.callbacks.speed_monitor import SpeedMonitorCallback

_SEQ = 512


def _batch(rows, pad_to=_SEQ, with_example_ids=True, n_real_crops=None, padded_crops=None):
    """A minimal batch: `rows` is a list of real-token counts, one per row."""
    b = len(rows)
    ids = torch.zeros(b, pad_to, dtype=torch.long)
    out = {"input_ids": ids}
    if with_example_ids:
        eid = torch.full((b, pad_to), -1, dtype=torch.long)
        for i, n in enumerate(rows):
            eid[i, :n] = 0
        out["example_ids"] = eid
    if n_real_crops is not None:
        out["n_real_crops"] = torch.tensor(n_real_crops, dtype=torch.long)
        out["images"] = torch.zeros(b, padded_crops, 4, 4)
    return out


class _StubTrainModule:
    """Not a TransformerTrainModule and no ``extra_flops_per_batch``, so the FLOPs
    branch of ``pre_step`` short-circuits and only the token accounting runs."""


class _StubTrainer:
    train_module = _StubTrainModule()


def _cb():
    cb = SpeedMonitorCallback()
    cb._trainer = _StubTrainer()  # type: ignore[assignment]
    cb._first_step = False  # skip the "don't record the first batch" guard
    return cb


def test_counts_only_non_pad_tokens():
    cb = _cb()
    cb.pre_step(_batch([100, 60]))
    assert cb._step_tokens == 2 * _SEQ  # padded
    assert cb._step_useful_tokens == 160  # real
    assert cb._step_useful_tokens / cb._step_tokens == 160 / 1024


def test_absent_example_ids_reports_nothing_rather_than_full():
    """Without example_ids we cannot see padding -- claiming 100% would be a lie."""
    cb = _cb()
    cb.pre_step(_batch([100, 60], with_example_ids=False))
    assert cb._step_useful_tokens is None
    assert cb._total_useful_tokens == 0


def test_totals_accumulate_only_measurable_steps():
    cb = _cb()
    cb.pre_step(_batch([100, 60]))
    cb.pre_step(_batch([100, 60], with_example_ids=False))  # unmeasurable, must not count
    cb.pre_step(_batch([40, 40]))
    assert cb._total_useful_tokens == 160 + 80


def test_stale_values_do_not_carry_into_the_next_step():
    """A step that cannot be measured must not re-log the previous step's number."""
    cb = _cb()
    cb.pre_step(_batch([100, 60], n_real_crops=[2, 3], padded_crops=4))
    assert cb._step_crop_occupancy == 5 / 8
    cb.pre_step(_batch([100, 60]))  # no crops in this batch
    assert cb._step_crop_occupancy is None
    assert cb._step_useful_tokens == 160


def test_crop_occupancy_against_the_padded_tensor():
    cb = _cb()
    cb.pre_step(_batch([10, 10, 10], n_real_crops=[2, 9, 0], padded_crops=9))
    assert cb._step_crop_occupancy == 11 / 27


def test_text_only_batch_reports_its_dummy_crop_as_waste():
    """The collator always emits one zero crop; it is padding and should read as such."""
    cb = _cb()
    cb.pre_step(_batch([10, 10], n_real_crops=[0, 0], padded_crops=1))
    assert cb._step_crop_occupancy == 0.0


def test_parallel_degree_divides_useful_tokens_like_padded_tokens():
    cb = _cb()
    cb._parallel_degree = 2
    cb.pre_step(_batch([100, 60]))
    assert cb._step_tokens == (2 * _SEQ) // 2
    assert cb._step_useful_tokens == 160 // 2


def test_crop_occupancy_is_measured_before_the_dp_wide_pad_round():
    """Documents a known bias rather than pinning a wrong value as correct.

    ``MultimodalLM._encode_images`` does its own DP-wide ``all_reduce(MAX)`` and pads
    every rank's crop axis to the busiest rank's width -- but that happens during the
    forward pass, after ``pre_step`` has already measured occupancy from the collated
    batch. So the metric is a lower bound on true padding (an optimistic occupancy
    figure), not the exact fraction the ViT executes. This locks in that the metric
    reflects the batch as collated, not as the model additionally pads it, so a future
    change that tried to "fix" this without updating the metric's documented meaning
    would be caught here.
    """
    cb = _cb()
    # Two ranks' worth of already-DP-padded crops would look identical to two ranks that
    # were never padded at all -- pre_step has no way to tell, because it only sees the
    # collator's rank-local batch.
    cb.pre_step(_batch([10, 10], n_real_crops=[2, 2], padded_crops=4))
    pre_dp_pad = cb._step_crop_occupancy

    cb.pre_step(_batch([10, 10], n_real_crops=[2, 2], padded_crops=4))  # same collated shape
    assert cb._step_crop_occupancy == pre_dp_pad == 4 / 8
