import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import init_device_mesh

from olmo_core.config import DType
from olmo_core.nn.moe import MoEConfig, MoEMLPConfig, MoERouterConfig, MoEType
from olmo_core.utils import get_default_device

from ...distributed.utils import requires_multi_gpu, run_distributed_test
from ...utils import requires_gpu, requires_grouped_gemm


@requires_gpu
@requires_grouped_gemm
@pytest.mark.parametrize("moe_type", [MoEType.dropless])
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="BF16")])
def test_moe(moe_type, dtype):
    d_model = 128
    config = MoEConfig(
        name=moe_type,
        num_experts=4,
        hidden_size=256,
        router=MoERouterConfig(top_k=1, dtype=DType.from_pt(dtype)),
        mlp=MoEMLPConfig(dtype=DType.from_pt(dtype)),
        z_loss_weight=0.1,
    )
    moe = config.build(d_model=d_model, num_layers=1, init_device="cuda")

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

    losses = moe.compute_losses(B * S)
    lb_loss = losses["load balancing loss"]
    z_loss = losses["router Z loss"]
    loss = lb_loss + z_loss

    # Run backward pass.
    loss.backward()
    assert x.grad is not None


def run_moe_with_expert_parallelism(moe_type, dtype):
    ep_mesh = init_device_mesh(get_default_device().type, (dist.get_world_size(),))

    d_model = 128
    config = MoEConfig(
        name=moe_type,
        num_experts=4,
        hidden_size=256,
        router=MoERouterConfig(top_k=1, dtype=DType.from_pt(dtype)),
        mlp=MoEMLPConfig(dtype=DType.from_pt(dtype)),
        z_loss_weight=0.1,
    )
    moe = config.build(d_model=d_model, num_layers=1, init_device="meta")
    moe.apply_ep(ep_mesh)
    moe.to_empty(device=get_default_device())

    # Run forward pass.
    B, S = 2, 16
    x = torch.randn(B, S, d_model, dtype=dtype, device="cuda", requires_grad=True)

    output = moe(x)
    assert output.shape == x.shape

    losses = moe.compute_losses(B * S)
    lb_loss = losses["load balancing loss"]
    z_loss = losses["router Z loss"]
    loss = lb_loss + z_loss

    # Run backward pass.
    loss.backward()
    assert x.grad is not None


@requires_multi_gpu
@requires_grouped_gemm
@pytest.mark.parametrize("moe_type", [MoEType.dropless])
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="BF16")])
def test_moe_with_expert_parallelism(moe_type, dtype):
    run_distributed_test(
        run_moe_with_expert_parallelism,
        backend="nccl",
        start_method="spawn",
        func_args=(moe_type, dtype),
    )
