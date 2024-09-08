import torch.nn as nn

from olmo_core.nn.layer_norm import LayerNorm
from olmo_core.nn.transformer import TransformerConfig


def test_small_llama2_config_builder():
    config = TransformerConfig.llama2_271M(vocab_size=50257)
    model = config.build()
    model.init_weights()

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
