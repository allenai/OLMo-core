import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from olmo_core.config import DType
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.parallel import (
    ExpertParallelConfig,
    build_expert_parallel_mesh,
    get_ep_mesh,
)
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.moe import (
    MoEConfig,
    MoELoadBalancingLossGranularity,
    MoERouterConfig,
    MoEType,
)
from olmo_core.testing import (
    has_grouped_gemm,
    requires_gpu,
    requires_grouped_gemm,
    requires_multi_gpu,
    run_distributed_test,
)
from olmo_core.utils import get_default_device, record_flops, seed_all


@requires_gpu
@pytest.mark.parametrize("moe_type", [MoEType.dropless, MoEType.default])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="BF16")])
def test_moe(moe_type: MoEType, shared: bool, dtype: torch.dtype):
    seed_all(42)

    d_model = 128
    config = MoEConfig(
        name=moe_type,
        num_experts=4,
        hidden_size=256,
        router=MoERouterConfig(top_k=1),
        shared_mlp=None if not shared else FeedForwardConfig(hidden_size=256),
        z_loss_weight=0.1,
        dtype=DType.from_pt(dtype),
    )
    moe = config.build(d_model=d_model, init_device="cuda")

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

    # Check auxiliary losses.
    metrics = moe.compute_metrics()
    lb_loss, _ = metrics["load balancing loss"]
    assert math.isfinite(lb_loss.item())
    z_loss, _ = metrics["router Z loss"]
    assert math.isfinite(z_loss.item())

    # Trigger backwards pass.
    output.sum().backward()
    assert x.grad is not None


def run_moe_with_expert_parallelism(
    checkpoint_dir: Path,
    config: MoEConfig,
    d_model: int,
    batch: torch.Tensor,
    expected_output: torch.Tensor,
    expected_lb_loss: torch.Tensor,
    expected_z_loss: torch.Tensor,
):
    seed_all(42)

    world_mesh = build_expert_parallel_mesh(ExpertParallelConfig(degree=dist.get_world_size()))
    ep_mesh = get_ep_mesh(world_mesh)

    moe = config.build(d_model=d_model, init_device="meta")
    moe.apply_ep(ep_mesh)
    moe.to_empty(device=get_default_device())

    # Load checkpoint.
    load_model_and_optim_state(checkpoint_dir, moe)

    # Split batch and expected output across process group.
    batch = get_local_tensor(
        distribute_tensor(
            batch.to(device=get_default_device()),
            device_mesh=world_mesh,
            placements=(
                Replicate(),
                Shard(0),
            ),
        )
    )
    batch.requires_grad_(True)
    expected_output = get_local_tensor(
        distribute_tensor(
            expected_output.to(device=get_default_device()),
            device_mesh=world_mesh,
            placements=(
                Replicate(),
                Shard(0),
            ),
        )
    )

    # Run forward pass.
    output = moe(batch)
    assert output.shape == batch.shape
    torch.testing.assert_close(output, expected_output)

    metrics = moe.compute_metrics()

    # Check load balancing loss.
    lb_loss, _ = metrics["load balancing loss"]
    assert math.isfinite(lb_loss.item())
    if config.lb_loss_granularity != MoELoadBalancingLossGranularity.local_batch:
        total_lb_loss = lb_loss.detach() / dist.get_world_size()
        dist.all_reduce(total_lb_loss)
        torch.testing.assert_close(total_lb_loss, expected_lb_loss.to(total_lb_loss.device))

    # Check Z loss.
    z_loss, _ = metrics["router Z loss"]
    assert math.isfinite(z_loss.item())
    total_z_loss = z_loss.detach() / dist.get_world_size()
    dist.all_reduce(total_z_loss)
    torch.testing.assert_close(total_z_loss, expected_z_loss.to(total_z_loss.device))

    # Run backward pass.
    output.sum().backward()
    assert batch.grad is not None


@requires_multi_gpu
@pytest.mark.parametrize("moe_type", [MoEType.dropless, MoEType.default])
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="BF16")])
@pytest.mark.parametrize(
    "lb_granularity",
    [
        pytest.param(MoELoadBalancingLossGranularity.local_batch, id="local-batch-LB"),
        pytest.param(MoELoadBalancingLossGranularity.instance, id="instance-LB"),
    ],
)
def test_moe_with_expert_parallelism(
    tmp_path: Path,
    moe_type: MoEType,
    dtype: torch.dtype,
    lb_granularity: MoELoadBalancingLossGranularity,
):
    """
    Test that we get the same result when we run an MoE on a single device as we do when
    we run it across multiple devices with expert parallelism.
    """
    seed_all(42)

    device = torch.device("cuda")

    d_model = 8
    config = MoEConfig(
        name=moe_type,
        num_experts=4,
        hidden_size=256,
        router=MoERouterConfig(
            top_k=1,
            uniform_expert_assignment=moe_type
            == MoEType.default,  # EP results may be different otherwise
            dtype=DType.from_pt(dtype),
        ),
        lb_loss_granularity=lb_granularity,
        z_loss_weight=0.1,
        dtype=DType.from_pt(dtype),
    )
    moe = config.build(d_model=d_model, init_device="cpu")
    moe.to(device=device)

    # Save state so when we spawn distributed processes they can load the same weights.
    save_model_and_optim_state(tmp_path, moe)

    # Create batch and run forward pass.
    B, S = 2, 4
    batch = torch.randn(B, S, d_model, dtype=dtype, device=device, requires_grad=True)
    output = moe(batch)
    assert output.shape == batch.shape
    assert torch.isfinite(output).all()
    assert (output > 0).any()

    # Get losses.
    metrics = moe.compute_metrics()
    lb_loss, _ = metrics["load balancing loss"]
    assert math.isfinite(lb_loss.item())

    z_loss, _ = metrics["router Z loss"]
    assert math.isfinite(z_loss.item())

    # Run backward pass.
    output.sum().backward()
    assert batch.grad is not None

    run_distributed_test(
        run_moe_with_expert_parallelism,
        backend="nccl",
        start_method="spawn",
        func_args=(
            tmp_path,
            config,
            d_model,
            batch.detach().cpu(),
            output.detach().cpu(),
            lb_loss.detach().cpu(),
            z_loss.detach().cpu(),
        ),
    )


@requires_gpu
@requires_grouped_gemm
@pytest.mark.parametrize("shared", [False, True], ids=["no_shared_expert", "with_shared_expert"])
def test_moe_num_flops_per_token(shared: bool):
    if has_grouped_gemm:
        pytest.skip("Pytorch flop recording is not supported for custom kernel grouped_gemm")

    seed_all(0)

    d_model = 128
    hidden_size = 256
    seq_len = 32
    batch_size = 1

    config = MoEConfig(
        #  Idealized FLOPs differ too much from actual FLOPs for default MoE implementation
        # (due to padding experts to a fixed capacity). So we use the dropless MoE implementation.
        name=MoEType.dropless,
        num_experts=16,
        hidden_size=hidden_size,
        router=MoERouterConfig(top_k=2),
        shared_mlp=None if not shared else FeedForwardConfig(hidden_size=hidden_size),
    )
    moe = config.build(d_model=d_model, init_device="cuda")

    x = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)

    actual_flops = record_flops(moe, x, with_backward=True)
    actual_flops_per_token = actual_flops // seq_len

    estimated_flops_per_token = moe.num_flops_per_token(seq_len)

    tolerance = 0.02
    relative_error = (
        abs(estimated_flops_per_token - actual_flops_per_token) / actual_flops_per_token
    )
    assert relative_error < tolerance, (
        f"Estimated FLOPs ({estimated_flops_per_token}) differs too much from actual ({actual_flops_per_token}), "
        f"{relative_error=:.2%}, {tolerance=:.2%}"
    )
