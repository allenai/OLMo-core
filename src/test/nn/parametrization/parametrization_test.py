import logging
from typing import Optional, Type

import pytest
import torch

from olmo_core.nn.parametrization import (
    MupScalingStrategy,
    ParametrizationConfig,
    ParametrizationOptimizerType,
)
from olmo_core.nn.parametrization.config import ParametrizationType
from olmo_core.nn.transformer.config import TransformerBlockType, TransformerConfig
from olmo_core.optim.adam import AdamConfig
from olmo_core.optim.adamw import AdamWConfig, SkipStepAdamWConfig
from olmo_core.optim.config import OptimConfig, OptimGroupOverride

log = logging.getLogger(__name__)


def get_transformer_config(
    parametrization_config: Optional[ParametrizationConfig] = None,
    vocab_size: int = 100,
    init_seed: int = 0,
    d_model_multiplier: int = 1,
    n_heads_multiplier: int = 1,
    **kwargs,
) -> TransformerConfig:
    return TransformerConfig.llama_like(
        d_model=8 * d_model_multiplier,
        vocab_size=vocab_size,
        n_layers=2,
        n_heads=2 * n_heads_multiplier,
        hidden_size_multiplier=1,
        hidden_size_multiple_of=16,
        block_name=TransformerBlockType.reordered_norm,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
        fused_ops=False,
        use_flash=False,
        parametrization=parametrization_config,
        init_seed=init_seed,
        **kwargs,
    )


def get_transformer_inputs(vocab_size: int = 100) -> torch.Tensor:
    return torch.arange(0, vocab_size).unsqueeze(0)


@pytest.mark.parametrize(
    "parametrization_optimizer_type, parametrization_scaling_strategy, expected_multiplier",
    [
        pytest.param(ParametrizationOptimizerType.adam, MupScalingStrategy.constant_inputs, None),
        pytest.param(
            ParametrizationOptimizerType.adam,
            MupScalingStrategy.constant_lr,
            1 / 5,
        ),
        pytest.param(
            ParametrizationOptimizerType.adam,
            MupScalingStrategy.constant_init_std,
            1 / 5,
        ),
    ],
)
def test_mup_input_multiplier_scaling_input(
    parametrization_optimizer_type, parametrization_scaling_strategy, expected_multiplier
):
    parametrization_config = ParametrizationConfig(
        name=ParametrizationType.mup,
        optimizer=parametrization_optimizer_type,
        scaling_strategy=parametrization_scaling_strategy,
    )
    parametrization = parametrization_config.build(input_dim=5, output_dim=1)

    torch.testing.assert_close(parametrization.input_multiplier, expected_multiplier)


@pytest.mark.parametrize(
    "parametrization_optimizer_type, parametrization_scaling_strategy, expected_multiplier",
    [
        pytest.param(ParametrizationOptimizerType.adam, MupScalingStrategy.constant_inputs, None),
        pytest.param(ParametrizationOptimizerType.adam, MupScalingStrategy.constant_lr, 1 / 5),
        pytest.param(
            ParametrizationOptimizerType.adam,
            MupScalingStrategy.constant_init_std,
            1 / (5 ** (1 / 2)),
        ),
    ],
)
def test_mup_input_multiplier_scaling_input_and_output(
    parametrization_optimizer_type, parametrization_scaling_strategy, expected_multiplier
):
    parametrization_config = ParametrizationConfig(
        name=ParametrizationType.mup,
        optimizer=parametrization_optimizer_type,
        scaling_strategy=parametrization_scaling_strategy,
    )
    parametrization = parametrization_config.build(input_dim=5, output_dim=2)

    torch.testing.assert_close(parametrization.input_multiplier, expected_multiplier)


@pytest.mark.parametrize(
    "parametrization_optimizer_type, parametrization_scaling_strategy, expected_multiplier",
    [
        pytest.param(
            ParametrizationOptimizerType.adam,
            MupScalingStrategy.constant_inputs,
            1 / 5,
        ),
        pytest.param(ParametrizationOptimizerType.adam, MupScalingStrategy.constant_lr, None),
        pytest.param(
            ParametrizationOptimizerType.adam,
            MupScalingStrategy.constant_init_std,
            None,
        ),
    ],
)
def test_mup_init_std_multiplier_scaling_input(
    parametrization_optimizer_type, parametrization_scaling_strategy, expected_multiplier
):
    parametrization_config = ParametrizationConfig(
        name=ParametrizationType.mup,
        optimizer=parametrization_optimizer_type,
        scaling_strategy=parametrization_scaling_strategy,
    )
    parametrization = parametrization_config.build(input_dim=5, output_dim=1)

    torch.testing.assert_close(parametrization.init_std_multiplier, expected_multiplier)


@pytest.mark.parametrize(
    "parametrization_optimizer_type, parametrization_scaling_strategy, expected_multiplier",
    [
        pytest.param(
            ParametrizationOptimizerType.adam,
            MupScalingStrategy.constant_inputs,
            1 / (5 ** (1 / 2)),
        ),
        pytest.param(
            ParametrizationOptimizerType.adam,
            MupScalingStrategy.constant_lr,
            5 ** (1 / 2),
        ),
        pytest.param(
            ParametrizationOptimizerType.adam,
            MupScalingStrategy.constant_init_std,
            None,
        ),
    ],
)
def test_mup_init_std_multiplier_scaling_input_and_output(
    parametrization_optimizer_type, parametrization_scaling_strategy, expected_multiplier
):
    parametrization_config = ParametrizationConfig(
        name=ParametrizationType.mup,
        optimizer=parametrization_optimizer_type,
        scaling_strategy=parametrization_scaling_strategy,
    )
    parametrization = parametrization_config.build(input_dim=5, output_dim=2)

    torch.testing.assert_close(parametrization.init_std_multiplier, expected_multiplier)


@pytest.mark.parametrize(
    "parametrization_optimizer_type",
    [
        pytest.param(parametrization_optimizer_type)
        for parametrization_optimizer_type in ParametrizationOptimizerType
    ],
)
def test_standard_parametrization_same_output(parametrization_optimizer_type):
    model_config = get_transformer_config()

    parametrization_config = ParametrizationConfig(
        name=ParametrizationType.default,
        optimizer=parametrization_optimizer_type,
    )
    parametrization_model_config = get_transformer_config(parametrization_config)

    model = model_config.build()
    parametrization_model = parametrization_model_config.build()

    model.init_weights()
    parametrization_model.to_empty(device=model.device)
    parametrization_model.load_state_dict(model.state_dict())

    input_ids = get_transformer_inputs()
    logits = model(input_ids=input_ids)
    parametrization_logits = parametrization_model(input_ids=input_ids)

    torch.testing.assert_close(logits, parametrization_logits)


@pytest.mark.parametrize(
    "optim_config_cls",
    [
        pytest.param(optim_config_cls)
        for optim_config_cls in [AdamConfig, AdamWConfig, SkipStepAdamWConfig]
    ],
)
def test_standard_parametrization_same_optim_groups(optim_config_cls: Type[OptimConfig]):
    model_config = get_transformer_config()

    parametrization_optimizer_type = optim_config_cls.parametrization_optimizer_type()
    assert parametrization_optimizer_type is not None, "Optimizer does not support parametrization"

    parametrization_config = ParametrizationConfig(
        name=ParametrizationType.default,
        optimizer=parametrization_optimizer_type,
    )
    parametrization_model_config = get_transformer_config(parametrization_config)

    model = model_config.build()
    parametrization_model = parametrization_model_config.build()

    optim_config = optim_config_cls(
        lr=1e-2,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
    )

    model.init_weights()
    parametrization_model.to_empty(device=model.device)
    parametrization_model.load_state_dict(model.state_dict())

    optim: torch.optim.Optimizer = optim_config.build(model)
    parametrization_optim: torch.optim.Optimizer = optim_config.build(parametrization_model)

    torch.testing.assert_close(optim.param_groups, parametrization_optim.param_groups)


@pytest.mark.parametrize(
    "parametrization_optimizer_type",
    [
        pytest.param(parametrization_optimizer_type)
        for parametrization_optimizer_type in ParametrizationOptimizerType
    ],
)
def test_mup_no_input_scaling_same_output(parametrization_optimizer_type):
    model_config = get_transformer_config()

    parametrization_config = ParametrizationConfig(
        name=ParametrizationType.mup,
        optimizer=parametrization_optimizer_type,
        scaling_strategy=MupScalingStrategy.constant_inputs,
    )
    # Set softmax scale to 1.0 to make mup attention match that of standard parametrization
    parametrization_model_config = get_transformer_config(parametrization_config, softmax_scale=1.0)

    model = model_config.build()
    parametrization_model = parametrization_model_config.build()

    model.init_weights()
    parametrization_model.to_empty(device=model.device)
    parametrization_model.load_state_dict(model.state_dict())

    input_ids = get_transformer_inputs()
    logits = model(input_ids=input_ids)
    parametrization_logits = parametrization_model(input_ids=input_ids)

    torch.testing.assert_close(logits, parametrization_logits)


@pytest.mark.parametrize(
    "optim_config_cls",
    [
        pytest.param(AdamConfig),
        pytest.param(AdamWConfig),
        pytest.param(SkipStepAdamWConfig),
    ],
)
def test_mup_no_lr_scaling_same_optim_groups(optim_config_cls):
    model_config = get_transformer_config()

    parametrization_optimizer_type = optim_config_cls.parametrization_optimizer_type()
    assert parametrization_optimizer_type is not None, "Optimizer does not support parametrization"

    parametrization_config = ParametrizationConfig(
        name=ParametrizationType.mup,
        optimizer=parametrization_optimizer_type,
        scaling_strategy=MupScalingStrategy.constant_lr,
    )
    parametrization_model_config = get_transformer_config(parametrization_config)

    model = model_config.build()
    parametrization_model = parametrization_model_config.build()

    optim_config = optim_config_cls(
        lr=1e-2,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
    )

    model.init_weights()
    parametrization_model.to_empty(device=model.device)
    parametrization_model.load_state_dict(model.state_dict())

    optim: torch.optim.Optimizer = optim_config.build(model)
    parametrization_optim: torch.optim.Optimizer = optim_config.build(parametrization_model)

    torch.testing.assert_close(optim.param_groups, parametrization_optim.param_groups)
