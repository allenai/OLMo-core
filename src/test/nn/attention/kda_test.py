import pytest
import torch

from olmo_core.nn.attention import KimiDeltaAttentionConfig
from olmo_core.testing import requires_gpu
from olmo_core.testing.utils import requires_fla


@requires_fla
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(KimiDeltaAttentionConfig(n_heads=8), id="default"),
        pytest.param(KimiDeltaAttentionConfig(n_heads=8, head_dim=32), id="head_dim=32"),
        pytest.param(KimiDeltaAttentionConfig(n_heads=8, conv_size=8), id="conv_size=8"),
    ],
)
def test_kimi_delta_attention_config_num_params(config: KimiDeltaAttentionConfig):
    d_model = 512
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")
    assert config.num_params(d_model) == sum(p.numel() for p in module.parameters())


@requires_fla
@requires_gpu
def test_kimi_delta_attention_fwd_bwd():
    device = "cuda"
    dtype = torch.bfloat16
    d_model, seq_len, batch_size = 256, 64, 2
    config = KimiDeltaAttentionConfig(n_heads=2, head_dim=128)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    cu_doc_lens = torch.tensor([0, 32, 64, 96, 128], dtype=torch.int32, device=device)

    with torch.autocast(device_type=device, dtype=dtype):
        y = module(x, cu_doc_lens=cu_doc_lens)
        assert y.shape == x.shape
        y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
