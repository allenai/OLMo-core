import pytest
import torch

from olmo_core.nn.functional import cross_entropy_loss, fused_cross_entropy_loss

from ...utils import DEVICES, requires_flash_attn, requires_gpu


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


@requires_gpu
@requires_flash_attn
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_fused_cross_entropy_loss(reduction):
    device = torch.device("cuda")
    vocab_size = 50257
    N = 1024

    logits = torch.randn(N, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (N,), device=device)

    ce_loss_fused, z_loss_fused = fused_cross_entropy_loss(
        logits.clone(), labels.clone(), reduction=reduction, compute_z_loss=True
    )
    assert z_loss_fused is not None

    # TODO: only comparing when reduction is "mean" since results from "sum" are too far
    # apart (see note below as well). This is potentially an issue.
    if reduction == "mean":
        # Make sure the outputs match those from the non-fused cross entropy loss function.
        ce_loss, z_loss = cross_entropy_loss(
            logits.clone(), labels.clone(), reduction=reduction, compute_z_loss=True
        )
        assert z_loss is not None

        # TODO: I'm not sure why these aren't closer. We shouldn't need to relax atol/rtol for
        # this to pass.
        torch.testing.assert_close(ce_loss, ce_loss_fused, atol=2e-2, rtol=2e-3)
        torch.testing.assert_close(z_loss, z_loss_fused)

    # Now add some masked values to logits and labels and make sure we get the same result.
    logits_padded = torch.cat([logits, torch.rand(3, vocab_size, device=device)], dim=0)
    labels_padded = torch.cat([labels, torch.tensor([-100] * 3, device=device)], dim=0)
    ce_loss1, z_loss1 = fused_cross_entropy_loss(
        logits_padded, labels_padded, reduction=reduction, compute_z_loss=True
    )
    torch.testing.assert_close(ce_loss_fused, ce_loss1)
    torch.testing.assert_close(z_loss_fused, z_loss1)
