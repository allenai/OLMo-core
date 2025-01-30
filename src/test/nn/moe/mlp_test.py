import torch
import torch.distributed as dist
from torch.distributed.tensor import init_device_mesh

from olmo_core.nn.moe.mlp import MoEMLP
from olmo_core.utils import get_default_device

from ...distributed.utils import requires_multi_gpu, run_distributed_test
from ...utils import requires_gpu


@requires_gpu
def test_mlp():
    mlp = MoEMLP(d_model=128, hidden_size=256, num_experts=2, init_device="cuda")
    x = torch.randn(5, 128, device="cuda")
    tokens_per_expert = torch.tensor([3, 2], device="cuda")
    out = mlp(x, tokens_per_expert)
    assert out.shape == (5, 128)


def run_mlp_with_expert_parallelism():
    mlp = MoEMLP(d_model=128, hidden_size=256, num_experts=4, init_device="meta")
    ep_mesh = init_device_mesh(get_default_device().type, (dist.get_world_size(),))
    mlp.apply_ep(ep_mesh)
    mlp.to_empty(device=get_default_device())
    x = torch.randn(5, 128, device="cuda")
    tokens_per_expert = torch.tensor([3, 2], device="cuda")
    out = mlp(x, tokens_per_expert)
    assert out.shape == (5, 128)


@requires_multi_gpu
def test_mlp_with_expert_parallelism():
    run_distributed_test(run_mlp_with_expert_parallelism, backend="nccl", start_method="spawn")
