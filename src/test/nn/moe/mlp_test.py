import torch
import torch.distributed as dist

from olmo_core.distributed.parallel import (
    ExpertParallelConfig,
    build_expert_parallel_mesh,
)
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.nn.moe.mlp import DroplessMoEMLP, MoEMLP
from olmo_core.utils import get_default_device

from ...distributed.utils import requires_multi_gpu, run_distributed_test
from ...utils import requires_gpu, requires_grouped_gemm


@requires_gpu
def test_mlp():
    mlp = MoEMLP(
        d_model=128, hidden_size=256, num_experts=2, init_device="cuda", dtype=torch.bfloat16
    )
    x = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16)
    tokens_per_expert = torch.tensor([3, 3], device="cuda")
    out = mlp(x, tokens_per_expert)
    assert out.shape == (6, 128)


@requires_gpu
@requires_grouped_gemm
def test_dropless_mlp():
    mlp = DroplessMoEMLP(
        d_model=128, hidden_size=256, num_experts=2, init_device="cuda", dtype=torch.bfloat16
    )
    x = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16)
    tokens_per_expert = torch.tensor([3, 2], device="cuda")
    out = mlp(x, tokens_per_expert)
    assert out.shape == (5, 128)


def run_mlp_with_expert_parallelism():
    ep_mesh = build_expert_parallel_mesh(ExpertParallelConfig(degree=dist.get_world_size()))

    mlp = MoEMLP(
        d_model=128,
        hidden_size=256,
        num_experts=dist.get_world_size() * 2,
        init_device="meta",
        dtype=torch.bfloat16,
    )
    mlp.apply_ep(ep_mesh)
    mlp.to_empty(device=get_default_device())
    assert get_local_tensor(mlp.w1).shape == (2, 256, 128)

    x = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16)
    tokens_per_expert = torch.tensor([3, 3], device="cuda")
    out = mlp(x, tokens_per_expert)

    assert out.shape == (6, 128)


@requires_multi_gpu
def test_mlp_with_expert_parallelism():
    run_distributed_test(run_mlp_with_expert_parallelism, backend="nccl", start_method="spawn")


def run_dropless_mlp_with_expert_parallelism():
    ep_mesh = build_expert_parallel_mesh(ExpertParallelConfig(degree=dist.get_world_size()))

    mlp = MoEMLP(
        d_model=128,
        hidden_size=256,
        num_experts=dist.get_world_size() * 2,
        init_device="meta",
        dtype=torch.bfloat16,
    )
    mlp.apply_ep(ep_mesh)
    mlp.to_empty(device=get_default_device())
    assert get_local_tensor(mlp.w1).shape == (2, 256, 128)

    x = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16)
    tokens_per_expert = torch.tensor([2, 3], device="cuda")
    out = mlp(x, tokens_per_expert)

    assert out.shape == (5, 128)


@requires_multi_gpu
def test_dropless_mlp_with_expert_parallelism():
    run_distributed_test(
        run_dropless_mlp_with_expert_parallelism, backend="nccl", start_method="spawn"
    )
