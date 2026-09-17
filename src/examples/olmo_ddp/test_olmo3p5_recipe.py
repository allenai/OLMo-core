"""CPU-only drift guards for the reusable production recipe."""

from dataclasses import replace

import pytest
from olmo3p5_models import GEOMETRIES, build_model_config
from olmo3p5_recipe import (
    MEDIUM_64,
    MEDIUM_128,
    SMALL,
    build_trainer,
    optimization_environment,
)

from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.ddp import OLMoDDPTransformerBlockConfig


@pytest.mark.parametrize("size", GEOMETRIES)
@pytest.mark.parametrize("use_emo", [False, True])
def test_model_shape(size, use_emo):
    model = build_model_config(size, eos_token_id=100257, use_emo=use_emo)
    assert model.num_params == GEOMETRIES[size].expected_total_params
    assert isinstance(model.block, OLMoDDPTransformerBlockConfig)
    assert model.block.routed_experts_router is not None
    assert (model.block.routed_experts_router.emo is not None) == use_emo
    assert model.block_overrides is not None
    for index in GEOMETRIES[size].full_attention_layers:
        attention = model.block_overrides[index].sequence_mixer
        assert isinstance(attention, AttentionConfig) and attention.qk_norm_per_head_gains


def test_topology_and_cadence():
    assert [s.gradient_accumulation_steps for s in (SMALL, MEDIUM_64, MEDIUM_128)] == [8, 16, 8]
    config = build_trainer(SMALL, save_folder="/checkpoints/test", work_dir="/tmp/test")
    assert config.max_duration.value == 834466
    cp = config.callbacks["checkpointer"]
    assert cp.save_async is False and cp.max_checkpoints is None and cp.remove == "never"
    assert 18000 in cp.fixed_steps and 18250 in cp.fixed_steps and 60000 in cp.fixed_steps
    assert 18100 not in cp.fixed_steps and cp.save_interval == 500
    assert (
        optimization_environment(ep_degree=1, torch_version="2.13.0")["OLMO_PROFILE_EMO_TOP16"]
        == "0"
    )
    assert (
        optimization_environment(ep_degree=1, torch_version="2.11.0")["OLMO_PROFILE_EMO_TOP16"]
        == "1"
    )
    with pytest.raises(ValueError):
        replace(SMALL, batch_tokens=1).validate()
