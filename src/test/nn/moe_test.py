import pytest
import torch

from olmo_core.nn.moe import MoEConfig, MoEType

from ..utils import requires_gpu, requires_megablocks


@requires_gpu
@requires_megablocks
@pytest.mark.parametrize("moe_type", [MoEType.default, MoEType.dropless])
def test_moe(moe_type):
    d_model = 128
    config = MoEConfig(name=moe_type, hidden_size=256)
    moe = config.build(d_model=d_model, init_device="cuda")
    print(moe)
    moe(torch.randn(2, d_model, device="cuda"))
