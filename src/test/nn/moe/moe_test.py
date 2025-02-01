import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.moe import MoEConfig, MoEMLPConfig, MoERouterConfig, MoEType

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
    x = torch.randn(2, 16, d_model, dtype=dtype, device="cuda", requires_grad=True)
    output, lb_loss, z_loss = moe(x)
    assert output.shape == x.shape
    assert lb_loss is not None
    assert z_loss is not None
    loss = lb_loss + z_loss

    # Run backward pass.
    loss.backward()
    assert x.grad is not None
