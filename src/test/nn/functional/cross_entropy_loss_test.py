import importlib

import pytest
import torch

from olmo_core.nn.functional import cross_entropy_loss, fused_linear_cross_entropy_loss
from olmo_core.testing import DEVICES


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_cross_entropy_loss(device, reduction):
    vocab_size = 50257
    N = 32

    logits = torch.randn(N, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (N,), device=device)

    ce_loss, z_loss = cross_entropy_loss(logits, labels, reduction=reduction, compute_z_loss=True)
    assert ce_loss.shape == tuple()
    assert ce_loss.numel() == 1
    assert z_loss is not None
    assert z_loss.shape == tuple()
    assert z_loss.numel() == 1

    # Now add some masked values to logits and labels and make sure we get the same result.
    logits_padded = torch.cat([logits, torch.rand(3, vocab_size, device=device)], dim=0)
    labels_padded = torch.cat([labels, torch.tensor([-100] * 3, device=device)], dim=0)
    ce_loss1, z_loss1 = cross_entropy_loss(
        logits_padded, labels_padded, reduction=reduction, compute_z_loss=True
    )
    torch.testing.assert_close(ce_loss, ce_loss1)
    torch.testing.assert_close(z_loss, z_loss1)


@pytest.mark.parametrize(
    ("compute_z_loss", "expected_lse_square_scale"),
    [
        pytest.param(False, 0.0, id="disabled"),
        pytest.param(True, 0.125, id="enabled"),
    ],
)
def test_fused_linear_cross_entropy_loss_z_loss_contract(
    monkeypatch, compute_z_loss, expected_lse_square_scale
):
    loss_module = importlib.import_module("olmo_core.nn.functional.cross_entropy_loss")
    liger_call = {}
    ce_loss = torch.tensor(1.5)
    kernel_z_loss = torch.tensor(0.25)

    def mock_fused_loss(
        _input,
        _weight,
        _labels,
        _bias,
        _ce_weight,
        _ignore_index,
        lse_square_scale,
        _label_smoothing,
        _reduction,
        _softcap,
        return_z_loss,
        _accum_dtype,
    ):
        liger_call["lse_square_scale"] = lse_square_scale
        liger_call["return_z_loss"] = return_z_loss
        return ce_loss, kernel_z_loss, torch.tensor(1.0)

    monkeypatch.setattr(loss_module, "_fused_linear_cross_entropy_loss", mock_fused_loss)

    actual_ce_loss, actual_z_loss = fused_linear_cross_entropy_loss(
        torch.zeros(1, 2),
        torch.zeros(3, 2),
        torch.zeros(1, dtype=torch.long),
        compute_z_loss=compute_z_loss,
        z_loss_multiplier=0.125,
    )

    assert liger_call["lse_square_scale"] == expected_lse_square_scale
    assert liger_call["return_z_loss"] is compute_z_loss
    assert actual_ce_loss is ce_loss
    if compute_z_loss:
        assert actual_z_loss is kernel_z_loss
    else:
        assert actual_z_loss is None
