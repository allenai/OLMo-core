"""``ResidualStream.masked_dropout`` — mm_olmo's ``response_residual_dropout``.

Molmo2 drops a fraction of the residual stream on *response* tokens only, leaving prompt and
image tokens untouched (mm_olmo's ``Dropout(mask_p=...)``).
"""

import pytest
import torch

from olmo_core.nn.residual_stream import ResidualStream

B, T, D = 1, 4000, 8


def _inputs():
    x = torch.ones(B, T, D)
    residual = torch.zeros(B, T, D)
    drop_mask = torch.zeros(B, T, dtype=torch.bool)
    drop_mask[:, T // 2 :] = True  # second half = response tokens
    return residual, x, drop_mask


def test_masked_dropout_only_drops_masked_tokens():
    torch.manual_seed(0)
    residual, x, drop_mask = _inputs()
    out = ResidualStream(dropout=0.0, masked_dropout=0.1).train()(residual, x, drop_mask=drop_mask)

    unmasked, masked = out[:, : T // 2], out[:, T // 2 :]
    assert (unmasked == 1.0).all(), "prompt/image tokens must never be dropped"
    dropped = (masked == 0).float().mean().item()
    assert 0.06 < dropped < 0.14, dropped  # ~0.10


def test_masked_dropout_is_unbiased_and_inverted():
    torch.manual_seed(0)
    residual, x, drop_mask = _inputs()
    out = ResidualStream(dropout=0.0, masked_dropout=0.1).train()(residual, x, drop_mask=drop_mask)
    masked = out[:, T // 2 :]
    # Surviving activations are scaled by 1/keep_prob, so the expectation is preserved.
    torch.testing.assert_close(masked[masked != 0][0], torch.tensor(1 / 0.9), rtol=1e-5, atol=1e-5)
    assert abs(masked.mean().item() - 1.0) < 0.05


def test_no_dropout_in_eval_mode():
    residual, x, drop_mask = _inputs()
    out = ResidualStream(dropout=0.0, masked_dropout=0.1).eval()(residual, x, drop_mask=drop_mask)
    assert (out == 1.0).all()


def test_zero_masked_dropout_matches_the_plain_path():
    residual, x, drop_mask = _inputs()
    torch.manual_seed(1)
    expected = ResidualStream(dropout=0.0).train()(residual, x)
    torch.manual_seed(1)
    actual = ResidualStream(dropout=0.0, masked_dropout=0.0).train()(
        residual, x, drop_mask=drop_mask
    )
    assert torch.equal(expected, actual)


def test_drop_mask_required_when_masked_dropout_is_set():
    residual, x, _ = _inputs()
    with pytest.raises(ValueError, match="drop_mask"):
        ResidualStream(dropout=0.0, masked_dropout=0.1).train()(residual, x)


def test_rejects_invalid_rate():
    with pytest.raises(ValueError, match="masked_dropout"):
        ResidualStream(masked_dropout=1.0)
