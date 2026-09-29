"""The scratchpad decision tokens can be re-weighted without touching the body."""

import numpy as np

from olmo_core.data.multimodal.sft_common import (
    THINK_CLOSE_ID,
    THINK_OPEN_ID,
    boost_think_tokens,
)


def _seq():
    labels = np.array([-100, THINK_OPEN_ID, 10, 11, 12, THINK_CLOSE_ID, 20, THINK_OPEN_ID])
    lm = np.array([0.0, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.0])
    return {"labels": labels, "loss_masks": lm}


def test_boost_raises_only_supervised_delimiters():
    out = boost_think_tokens(_seq(), 1.0)
    lm = out["loss_masks"]
    assert lm[1] == 1.0 and lm[5] == 1.0  # <think>, </think>
    assert np.allclose(lm[[2, 3, 4, 6]], 0.04)  # body and answer untouched
    assert lm[7] == 0.0  # an unsupervised <think> stays unsupervised


def test_boost_is_max_not_overwrite_and_noop_when_unset():
    s = _seq()
    s["loss_masks"][1] = 2.0
    out = boost_think_tokens(s, 1.0)
    assert out["loss_masks"][1] == 2.0
    s2 = _seq()
    assert np.array_equal(boost_think_tokens(s2, None)["loss_masks"], _seq()["loss_masks"])
    assert np.array_equal(boost_think_tokens(_seq(), 0.0)["loss_masks"], _seq()["loss_masks"])
