import torch

from olmo_core.nn.moe.mlp import MoEMLP

from ...utils import requires_gpu


@requires_gpu
def test_mlp():
    mlp = MoEMLP(d_model=128, hidden_size=256, num_experts=2, init_device="cuda")
    x = torch.randn(5, 128, device="cuda")
    tokens_per_expert = torch.tensor([3, 2], device="cuda")
    out = mlp(x, tokens_per_expert)
    assert out.shape == (5, 128)
