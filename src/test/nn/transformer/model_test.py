import logging

import pytest
import torch
import torch.nn as nn

from olmo_core.nn.layer_norm import LayerNorm
from olmo_core.nn.transformer import TransformerConfig

from ...utils import GPU_MARKS

log = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cuda", id="cpu->cuda", marks=GPU_MARKS),
        pytest.param("cpu", "cpu", id="cpu->cpu"),
    ],
)
def test_small_llama2_config_builder(init_device, device):
    config = TransformerConfig.llama2_271M(vocab_size=50257)
    log.info(config)
    model = config.build(init_device=init_device, device=torch.device(device))

    # Make sure num params estimate is correct.
    num_actual_params = 0
    for _, p in model.named_parameters():
        num_actual_params += p.numel()
    assert config.num_params == num_actual_params
    assert model.num_params == num_actual_params

    for module in model.modules():
        # Make sure there are no biases anywhere and layer norm weights are all 1.
        if isinstance(module, (nn.Linear, LayerNorm)):
            assert module.bias is None
        if isinstance(module, LayerNorm):
            assert module.weight is not None
            assert (module.weight == 1).all()

    # Make sure block_idx is set correctly.
    assert model.blocks[0].block_idx == 0
    assert model.blocks[-1].block_idx == len(model.blocks) - 1


@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cuda", id="cpu->cuda", marks=GPU_MARKS),
        pytest.param("cpu", "cpu", id="cpu->cpu"),
    ],
)
def test_small_ngpt_config_builder(init_device, device):
    config = TransformerConfig.ngpt_271M(vocab_size=50257)
    model = config.build(init_device=init_device, device=torch.device(device))

    # Make sure num params estimate is correct.
    num_actual_params = 0
    for _, p in model.named_parameters():
        num_actual_params += p.numel()
    assert config.num_params == num_actual_params
    assert model.num_params == num_actual_params

    # Make sure block_idx is set correctly.
    assert model.blocks[0].block_idx == 0
    assert model.blocks[-1].block_idx == len(model.blocks) - 1

    # Make sure all weights are normalized in the embedding dimension.
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert module.bias is None
            w = module.weight
            if w.shape[1] == config.d_model and "attention.w_out" not in name:
                pass
            elif w.shape[0] == config.d_model:
                w = w.transpose(0, 1)
            else:
                continue

            log.info(f"Checking norm for '{name}'")
            norm = torch.linalg.vector_norm(w, dim=1)
            torch.testing.assert_close(norm, torch.ones_like(norm))
