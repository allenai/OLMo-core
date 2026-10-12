"""Tests for the sparse-label fast path on the LM head (``sparse_loss_positions``)."""

from typing import Optional

import pytest
import torch

from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation

D_MODEL = 64
VOCAB = 512


def _build(sparse: bool, loss_impl: LMLossImplementation, device: torch.device):
    cfg = LMHeadConfig(
        layer_norm=LayerNormConfig(name="rms"),
        loss_implementation=loss_impl,
        sparse_loss_positions=sparse,
    )
    torch.manual_seed(0)
    head = cfg.build(d_model=D_MODEL, vocab_size=VOCAB, init_device=str(device))
    return head.to(device)


def _labels(B: int, T: int, n_keep: Optional[int], device: torch.device) -> torch.Tensor:
    """Labels with ``n_keep`` supervised positions per row, the rest ignored."""
    labels = torch.full((B, T), -100, dtype=torch.long, device=device)
    if n_keep:
        real = torch.randint(0, VOCAB, (B, n_keep), device=device)
        labels[:, -n_keep:] = real
    return labels


@pytest.mark.parametrize("loss_impl", [LMLossImplementation.default])
@pytest.mark.parametrize("n_keep", [1, 4, 32])
@pytest.mark.parametrize("loss_reduction", ["sum", "mean"])
def test_sparse_matches_dense(loss_impl, n_keep, loss_reduction):
    """Gathering to loss positions changes neither the loss nor the gradients."""
    device = torch.device("cpu")
    B, T = 2, 64
    torch.manual_seed(1234)
    x = torch.randn(B, T, D_MODEL, device=device)
    labels = _labels(B, T, n_keep, device)

    losses = {}
    grads = {}
    for sparse in (False, True):
        head = _build(sparse, loss_impl, device)
        inp = x.clone().requires_grad_(True)
        out = head(
            inp,
            labels=labels,
            loss_reduction=loss_reduction,
            return_logits=False,
        )
        out.loss.backward()
        losses[sparse] = out.loss.detach()
        grads[sparse] = inp.grad.detach().clone()

    torch.testing.assert_close(losses[True], losses[False], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grads[True], grads[False], rtol=1e-5, atol=1e-6)


def test_all_positions_masked_is_zero_loss():
    """A row with no supervised position must not produce NaN, and must not move the weights."""
    device = torch.device("cpu")
    B, T = 2, 16
    torch.manual_seed(7)
    x = torch.randn(B, T, D_MODEL, device=device, requires_grad=True)
    labels = _labels(B, T, None, device)

    head = _build(True, LMLossImplementation.default, device)
    out = head(x, labels=labels, loss_reduction="sum", return_logits=False)
    out.loss.backward()

    assert torch.isfinite(out.loss), out.loss
    assert out.loss.item() == 0.0
    assert torch.count_nonzero(x.grad) == 0


def test_none_reduction_keeps_per_position_shape():
    """The eval path asks for per-position loss; the fast path must decline to fire."""
    device = torch.device("cpu")
    B, T = 2, 16
    x = torch.randn(B, T, D_MODEL, device=device)
    labels = _labels(B, T, 4, device)

    head = _build(True, LMLossImplementation.default, device)
    out = head(x, labels=labels, loss_reduction="none", return_logits=False)
    assert out.loss.shape == (B, T), out.loss.shape
