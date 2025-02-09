from typing import Literal, Optional

import pytest
import torch
from torch.distributed import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from olmo_core.distributed.utils import get_world_size
from olmo_core.nn.cross_entropy_loss import CrossEntropyLoss
from olmo_core.utils import get_default_device

from ..distributed.utils import requires_multi_gpu, run_distributed_test


def run_cross_entropy_loss_parallel(
    fused: bool,
    compile: bool,
    reduction: Literal["sum", "mean", "none"],
    z_loss_multiplier: Optional[float],
    logits: torch.Tensor,
    labels: torch.Tensor,
    batch_num_tokens_for_loss: torch.Tensor,
):
    tp_mesh = init_device_mesh("cuda", (get_world_size(),), mesh_dim_names=("tp",))

    logits = distribute_tensor(
        logits.to(device=get_default_device()), device_mesh=tp_mesh, placements=(Shard(1),)
    )
    labels = distribute_tensor(
        labels.to(device=get_default_device()), device_mesh=tp_mesh, placements=(Shard(1),)
    )
    batch_num_tokens_for_loss = batch_num_tokens_for_loss.to(device=get_default_device())

    loss_fn = CrossEntropyLoss(
        reduction=reduction, compile=compile, fused=fused, z_loss_multiplier=z_loss_multiplier
    )
    loss_fn.apply_tp(tp_mesh)

    ce_loss, z_loss = loss_fn(logits[..., :-1, :].contiguous(), labels)
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

    # Trigger backward pass.
    loss.backward()
    assert logits.grad is not None


@pytest.mark.parametrize(
    "fused, compile, reduction",
    [
        pytest.param(False, False, "sum", id="default-sum"),
        pytest.param(False, False, "none", id="default-none"),
    ],
)
@requires_multi_gpu
def test_cross_entropy_loss_parallel(
    fused: bool,
    compile: bool,
    reduction: Literal["sum", "mean", "none"],
    z_loss_multiplier: Optional[float] = None,
):
    B, S, V = 4, 16, 256

    loss_fn = CrossEntropyLoss(
        reduction=reduction, compile=compile, fused=fused, z_loss_multiplier=z_loss_multiplier
    )

    labels = torch.randint(0, V, (B, S), device="cuda")
    logits = torch.randn(B, S, V, device="cuda", requires_grad=True)
    labels[0][2] = -100
    labels[2][9] = -100
    labels[3][12] = -100
    batch_num_tokens_for_loss = (labels != -100).sum()

    # Get losses.
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

    run_distributed_test(
        run_cross_entropy_loss_parallel,
        world_size=2,
        backend="nccl",
        start_method="spawn",
        func_args=(
            fused,
            compile,
            reduction,
            z_loss_multiplier,
            logits.detach().cpu(),
            labels.detach().cpu(),
            batch_num_tokens_for_loss.detach().cpu(),
        ),
    )
