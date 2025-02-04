import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard, distribute_tensor

from olmo_core.config import DType
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.parallel import (
    ExpertParallelConfig,
    build_expert_parallel_mesh,
)
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.nn.moe import MoEBase, MoEConfig, MoERouterConfig, MoEType
from olmo_core.utils import get_default_device, seed_all

from ...distributed.utils import requires_multi_gpu, run_distributed_test
from ...utils import requires_gpu


def init_mlp_weights(moe: MoEBase):
    for w in (moe.experts.mlp.w1, moe.experts.mlp.w2, moe.experts.mlp.w3):
        torch.nn.init.normal_(w, std=0.02)  # type: ignore


@requires_gpu
@pytest.mark.parametrize("moe_type", [MoEType.dropless, MoEType.default])
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="BF16")])
def test_moe(moe_type, dtype):
    seed_all(42)

    d_model = 128
    config = MoEConfig(
        name=moe_type,
        num_experts=4,
        hidden_size=256,
        router=MoERouterConfig(top_k=1, dtype=DType.from_pt(dtype)),
        z_loss_weight=0.1,
        dtype=DType.from_pt(dtype),
    )
    moe = config.build(d_model=d_model, num_layers=1, init_device="cuda")
    init_mlp_weights(moe)

    # Check num params calculation.
    num_params = 0
    for p in moe.parameters():
        num_params += p.numel()
    if config.num_params(d_model) != num_params:
        # For debugging...
        for n, p in moe.named_parameters():
            print(f"{n}: {p.shape}")
    assert config.num_params(d_model) == num_params

    # Run forward pass.
    B, S = 2, 16
    x = torch.randn(B, S, d_model, dtype=dtype, device="cuda", requires_grad=True)

    output = moe(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert (output > 0).any()

    losses = moe.compute_losses(B * S)
    lb_loss = losses["load balancing loss"]
    assert math.isfinite(lb_loss.item())
    z_loss = losses["router Z loss"]
    assert math.isfinite(z_loss.item())
    loss = lb_loss + z_loss

    # Run backward pass.
    loss.backward()
    assert x.grad is not None


def run_moe_with_expert_parallelism(
    checkpoint_dir: Path,
    config: MoEConfig,
    d_model: int,
    batch: torch.Tensor,
    expected_output: torch.Tensor,
):
    seed_all(42)

    ep_mesh = build_expert_parallel_mesh(ExpertParallelConfig(degree=min(dist.get_world_size(), 2)))

    moe = config.build(d_model=d_model, num_layers=1, init_device="meta")
    moe.apply_ep(ep_mesh)
    moe.to_empty(device=get_default_device())

    # Load checkpoint.
    load_model_and_optim_state(checkpoint_dir, moe)

    # Split batch and expected output across process group.
    total_tokens = batch.shape[0] * batch.shape[1]
    batch = get_local_tensor(
        distribute_tensor(
            batch.to(device=get_default_device()), device_mesh=ep_mesh, placements=(Shard(0),)
        )
    )
    batch.requires_grad_(True)
    expected_output = get_local_tensor(
        distribute_tensor(
            expected_output.to(device=get_default_device()),
            device_mesh=ep_mesh,
            placements=(Shard(0),),
        )
    )

    # Run forward pass.
    output = moe(batch)
    assert output.shape == batch.shape
    if dist.get_rank() == 0:
        print(f"{output=}")
        print(f"{expected_output=}")
    torch.testing.assert_close(output, expected_output)

    losses = moe.compute_losses(total_tokens // ep_mesh.size())
    lb_loss = losses["load balancing loss"]
    assert math.isfinite(lb_loss.item())

    z_loss = losses["router Z loss"]
    assert math.isfinite(z_loss.item())
    loss = lb_loss + z_loss

    # Run backward pass.
    loss.backward()
    assert batch.grad is not None


@requires_multi_gpu
@pytest.mark.parametrize("moe_type", [MoEType.dropless, MoEType.default])
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="BF16")])
def test_moe_with_expert_parallelism(tmp_path: Path, moe_type: MoEType, dtype: torch.dtype):
    """
    Test that we get the same result when we run an MoE on a single device as we do when
    we run it across multiple devices with expert parallelism.
    """
    seed_all(42)

    device = torch.device("cuda")

    d_model = 128
    config = MoEConfig(
        name=moe_type,
        num_experts=4,
        hidden_size=256,
        router=MoERouterConfig(
            top_k=1,
            uniform_expert_assignment=moe_type == MoEType.default,
            dtype=DType.from_pt(dtype),
        ),
        z_loss_weight=0.1,
        dtype=DType.from_pt(dtype),
    )
    moe = config.build(d_model=d_model, num_layers=1, init_device="cpu")
    moe.to(device=device)
    init_mlp_weights(moe)

    # Save state so when we spawn distributed processes they can load the same weights.
    save_model_and_optim_state(tmp_path, moe)

    # Create batch and run forward pass.
    B, S = 4, 16
    batch = torch.randn(B, S, d_model, dtype=dtype, device=device, requires_grad=True)
    output = moe(batch)
    assert output.shape == batch.shape
    assert torch.isfinite(output).all()
    assert (output > 0).any()

    # Get losses.
    losses = moe.compute_losses(B * S)
    lb_loss = losses["load balancing loss"]
    assert math.isfinite(lb_loss.item())

    z_loss = losses["router Z loss"]
    assert math.isfinite(z_loss.item())
    loss = lb_loss + z_loss

    # Run backward pass.
    loss.backward()
    assert batch.grad is not None

    run_distributed_test(
        run_moe_with_expert_parallelism,
        backend="nccl",
        start_method="spawn",
        func_args=(tmp_path, config, d_model, batch.detach().cpu(), output.detach().cpu()),
    )
