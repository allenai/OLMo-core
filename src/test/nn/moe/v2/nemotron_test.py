"""Config-construction and CPU build/forward tests for the Nemotron-3 Nano variant."""

import torch

from olmo_core.config import DType
from olmo_core.nn.attention import NemotronMamba2Config
from olmo_core.nn.layer_norm import LayerNormType
from olmo_core.nn.moe.v2.nemotron import (
    NemotronBlockConfig,
    NemotronBlockKind,
    build_debug_nemotron3_nano_config,
    build_nemotron3_nano_config,
    build_nemotron3_nano_config_from_hf_config,
)
from olmo_core.nn.transformer.config import TransformerConfig


def test_nemotron_block_config_wires_mixer_as_sequence_mixer():
    block = NemotronBlockConfig(
        kind=NemotronBlockKind.mamba,
        mixer=NemotronMamba2Config(mamba_num_heads=4, mamba_head_dim=16, n_groups=1),
    )
    # The base config validates on `sequence_mixer`; __post_init__ mirrors `mixer` into it.
    assert block.sequence_mixer is block.mixer
    assert block.num_params(128) > 0


def test_build_nemotron3_nano_config_uses_named_blocks_and_pattern():
    pattern = ("mamba", "moe", "attention")
    config = build_nemotron3_nano_config(
        vocab_size=128,
        d_model=128,
        n_layers=3,
        layers_block_type=pattern,
        num_attention_heads=4,
        num_key_value_heads=1,
        attention_head_dim=32,
        mamba_num_heads=4,
        mamba_head_dim=32,
        n_groups=1,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=256,
    )
    assert isinstance(config, TransformerConfig)
    assert isinstance(config.block, dict)
    assert set(config.block) == {"mamba", "attention", "moe"}
    assert config.block_pattern == list(pattern)
    # The block-level norm is the Nemotron RMSNorm variant.
    mamba_block = config.block["mamba"]
    assert isinstance(mamba_block, NemotronBlockConfig)
    assert mamba_block.norm.name == LayerNormType.nemotron_rms


def test_build_nemotron3_nano_config_from_hf_config_reads_text_config():
    hf_config = {
        "text_config": {
            "vocab_size": 128,
            "hidden_size": 128,
            "layers_block_type": ["mamba", "moe", "attention"],
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "mamba_num_heads": 4,
            "mamba_head_dim": 32,
            "ssm_state_size": 16,
            "conv_kernel": 4,
            "n_groups": 1,
            "chunk_size": 64,
            "time_step_min": 0.001,
            "time_step_max": 0.1,
            "time_step_floor": 0.0001,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 128,
            "moe_shared_expert_intermediate_size": 256,
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 2.5,
            "norm_topk_prob": True,
            "norm_eps": 1e-5,
        }
    }
    config = build_nemotron3_nano_config_from_hf_config(hf_config)
    assert config.n_layers == 3
    assert config.block_pattern == ["mamba", "moe", "attention"]


def test_nemotron_debug_model_builds_and_runs_on_cpu():
    config = build_debug_nemotron3_nano_config(
        vocab_size=128, n_layers=6, d_model=128, dtype=DType.float32
    )
    model = config.build(init_device="cpu")
    # Six layers spanning all three block kinds (mamba/moe/attention).
    assert len(model.blocks) == 6
    model.init_weights(device=torch.device("cpu"))

    input_ids = torch.randint(0, 128, (1, 16))
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (1, 16, 128)
