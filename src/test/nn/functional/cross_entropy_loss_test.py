import pytest
import torch

from olmo_core.nn.functional import cross_entropy_loss
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


@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize("ignore_index", [-100, -1])
def test_z_loss_ignored_labels_and_gradients(reduction, ignore_index):
    torch.manual_seed(0)
    logits = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([1, ignore_index, 3, ignore_index])
    valid = labels != ignore_index
    ce, z = cross_entropy_loss(
        logits, labels, ignore_index=ignore_index, reduction=reduction, compute_z_loss=True
    )
    assert z is not None
    ref_logits = logits.detach()[valid].clone().requires_grad_()
    ref_ce, ref_z = cross_entropy_loss(
        ref_logits, labels[valid], compute_z_loss=True, reduction="sum"
    )
    assert ref_z is not None
    if reduction == "none":
        torch.testing.assert_close(ce[~valid], torch.zeros_like(ce[~valid]))
        torch.testing.assert_close(z[~valid], torch.zeros_like(z[~valid]))
        ce, z = ce.sum(), z.sum()
    elif reduction == "mean":
        ref_ce, ref_z = ref_ce / valid.sum(), ref_z / valid.sum()
    torch.testing.assert_close(ce, ref_ce)
    torch.testing.assert_close(z, ref_z)
    (ce + z).backward()
    (ref_ce + ref_z).backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad[~valid], torch.zeros_like(logits.grad[~valid]))
    torch.testing.assert_close(logits.grad[valid], ref_logits.grad)


def test_unreduced_z_loss_all_ignored():
    logits = torch.zeros(2, 4, requires_grad=True)
    ce, z = cross_entropy_loss(
        logits, torch.full((2,), -100), reduction="none", compute_z_loss=True
    )
    assert z is not None
    torch.testing.assert_close(z, torch.zeros_like(z))
    (ce + z).sum().backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
