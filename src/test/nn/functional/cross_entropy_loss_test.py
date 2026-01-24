import pytest
import torch

from olmo_core.nn.functional import cross_entropy_loss, cute_cross_entropy_loss
from olmo_core.testing import DEVICES, requires_quack
from olmo_core.utils import get_default_device


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


@pytest.mark.parametrize("reduction", ["sum", "mean"])
@requires_quack
def test_cute_cross_entropy_loss(reduction):
    vocab_size = 50257
    N = 32
    device = get_default_device()

    logits = torch.randn(N, vocab_size, device=device, requires_grad=True)
    labels = torch.randint(0, vocab_size, (N,), device=device)

    # Add some masked values.
    logits = torch.cat([logits, torch.rand(3, vocab_size, device=device)], dim=0).unsqueeze(0)
    labels = torch.cat([labels, torch.tensor([-100] * 3, device=device)], dim=0)

    logits1 = logits.clone()

    ce_loss, z_loss = cute_cross_entropy_loss(
        logits, labels, reduction=reduction, compute_z_loss=True
    )
    assert z_loss is not None
    loss = ce_loss + z_loss
    loss.backward()
    assert logits.grad is not None

    # Make sure cute and reference implementation give the same results.
    ce_loss1, z_loss1 = cross_entropy_loss(
        logits1, labels, reduction=reduction, compute_z_loss=True
    )
    assert z_loss1 is not None
    loss1 = ce_loss1 + z_loss1
    loss1.backward()
    assert logits1.grad is not None
    torch.testing.assert_close(ce_loss, ce_loss1)
    torch.testing.assert_close(z_loss, z_loss1)
    torch.testing.assert_close(logits.grad, logits1.grad)
