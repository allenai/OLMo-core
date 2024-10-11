import pytest
import torch

from olmo_core.nn.moe import MoEConfig, MoEType

from ..utils import requires_gpu, requires_megablocks


@requires_gpu
@requires_megablocks
@pytest.mark.parametrize("moe_type", [MoEType.default, MoEType.dropless])
def test_moe(moe_type):
    d_model = 128
    config = MoEConfig(name=moe_type, hidden_size=512, num_experts=4)
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
    moe(torch.randn(2, 16, d_model, device="cuda"))
