from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_local_tensor, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.lm_head import LMHeadConfig, LMHeadType, LMLossImplementation
from olmo_core.utils import get_default_device, seed_all

from ..distributed.utils import requires_multi_gpu, run_distributed_test


def test_lm_head_builder_config():
    lm_head = LMHeadConfig(name=LMHeadType.default).build(d_model=64, vocab_size=128)
    assert lm_head.w_out.bias is not None

    lm_head = LMHeadConfig(name=LMHeadType.default, bias=False).build(d_model=64, vocab_size=128)
    assert lm_head.w_out.bias is None

    with pytest.raises(OLMoConfigurationError):
        LMHeadConfig(name=LMHeadType.normalized, bias=True).build(d_model=64, vocab_size=128)


def run_lm_head_tp(
    *,
    checkpoint_dir: Path,
    config: LMHeadConfig,
    d_model: int,
    vocab_size: int,
    loss_reduction: str,
    loss_div_factor: float,
    z_loss_multiplier: float,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    ce_loss: torch.Tensor,
    z_loss: torch.Tensor,
    grad: torch.Tensor,
):
    seed_all(42)
    device = get_default_device()
    tp_mesh = dist.init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("tp",))

    lm_head = config.build(d_model=d_model, vocab_size=vocab_size, init_device="meta")
    lm_head.apply_tp(tp_mesh, input_layouts=(Shard(1), Replicate()))
    lm_head.to_empty(device=device)

    load_model_and_optim_state(checkpoint_dir, lm_head)

    local_inputs = distribute_tensor(
        inputs.to(device=device), device_mesh=tp_mesh, placements=(Shard(1),)
    )
    local_output = lm_head(
        local_inputs,
        labels=labels,
        loss_div_factor=loss_div_factor,
        loss_reduction=loss_reduction,
        z_loss_multiplier=z_loss_multiplier,
    )
    local_ce_loss, local_z_loss = local_output.ce_loss, local_output.z_loss
    (local_ce_loss + local_z_loss).backward()
    assert local_inputs.grad is not None

    torch.testing.assert_close(get_local_tensor(local_ce_loss), ce_loss.to(device=device))
    torch.testing.assert_close(get_local_tensor(local_z_loss), z_loss.to(device=device))
    torch.testing.assert_close(
        get_local_tensor(local_inputs.grad),
        get_local_tensor(
            distribute_tensor(grad.to(device=device), device_mesh=tp_mesh, placements=(Shard(1),))
        ),
    )


@requires_multi_gpu
@pytest.mark.parametrize("head_type", [LMHeadType.default])
@pytest.mark.parametrize("loss_implementation", [LMLossImplementation.default])
@pytest.mark.parametrize("d_model", [64])
@pytest.mark.parametrize("vocab_size", [128])
@pytest.mark.parametrize("loss_reduction", ["sum"])
def test_lm_head_tp(
    tmp_path: Path,
    head_type: LMHeadType,
    loss_implementation: LMLossImplementation,
    d_model: int,
    vocab_size: int,
    loss_reduction: str,
    z_loss_multiplier: float = 1e-2,
):
    seed_all(42)
    device = torch.device("cuda")

    checkpoint_dir = tmp_path / "checkpoint"

    config = LMHeadConfig(name=head_type, loss_implementation=loss_implementation, bias=False)
    lm_head = config.build(d_model=d_model, vocab_size=vocab_size)
    save_model_and_optim_state(checkpoint_dir, lm_head)

    B, S = 2, 32
    inputs = torch.randn(B, S, d_model, device=device, requires_grad=True)
    labels = torch.randint(0, vocab_size, (B, S), device=device)
    loss_div_factor = B * S

    output = lm_head(
        inputs,
        labels=labels,
        loss_div_factor=loss_div_factor,
        loss_reduction=loss_reduction,
        z_loss_multiplier=z_loss_multiplier,
    )
    ce_loss, z_loss = output.ce_loss, output.z_loss
    (ce_loss + z_loss).backward()
    assert inputs.grad is not None

    run_distributed_test(
        run_lm_head_tp,
        backend="nccl",
        start_method="spawn",
        func_kwargs=dict(
            checkpoint_dir=checkpoint_dir,
            config=config,
            d_model=d_model,
            vocab_size=vocab_size,
            loss_reduction=loss_reduction,
            loss_div_factor=loss_div_factor,
            z_loss_multiplier=z_loss_multiplier,
            inputs=inputs.detach().cpu(),
            labels=labels.detach().cpu(),
            ce_loss=ce_loss.detach().cpu(),
            z_loss=z_loss.detach().cpu(),
            grad=inputs.grad.detach().cpu(),
        ),
    )
