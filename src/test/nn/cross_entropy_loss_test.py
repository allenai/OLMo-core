from typing import Literal, Optional

import pytest
import torch

from olmo_core.nn.cross_entropy_loss import LMCrossEntropyLoss

from ..utils import requires_gpu


@pytest.mark.parametrize(
    "fused, compile, reduction",
    [
        pytest.param(False, False, "sum", id="default-sum"),
        pytest.param(False, False, "none", id="default-none"),
    ],
)
@requires_gpu
def test_cross_entropy_loss_parallel(
    fused: bool,
    compile: bool,
    reduction: Literal["sum", "mean", "none"],
    z_loss_multiplier: Optional[float] = None,
):
    B, S, V = 4, 16, 256

    loss_fn = LMCrossEntropyLoss(
        reduction=reduction, compile=compile, fused=fused, z_loss_multiplier=z_loss_multiplier
    )

    input_ids = torch.randint(0, V, (B, S), device="cuda")
    logits = torch.randn(B, S, V, device="cuda", requires_grad=True)
    labels = input_ids.clone()[..., 1:].contiguous()
    labels[0][2] = -100
    labels[2][9] = -100
    labels[3][12] = -100

    batch_num_tokens_for_loss = (labels != -100).sum()
    ce_loss, z_loss = loss_fn(logits, labels)
    ce_loss.div_(batch_num_tokens_for_loss)
    if z_loss is not None:
        z_loss.div_(batch_num_tokens_for_loss)

    loss = ce_loss
    if z_loss is not None:
        loss += z_loss

    if reduction != "none":
        assert loss.shape == tuple()
    else:
        assert loss.shape == labels.shape
        loss = loss.sum()

    # Trigger backward pass.
    loss.backward()
    assert logits.grad is not None
