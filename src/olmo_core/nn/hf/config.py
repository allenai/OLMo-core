from transformers import (
    Olmo2Config,
    Olmo2RetrofitConfig,
    Olmo3Config,
    Olmoe2Config,
    PretrainedConfig,
)

from olmo_core.doc_utils import beta_feature
from olmo_core.nn.attention import Attention
from olmo_core.nn.moe import MoEMLP, MoERouterGatingFunction
from olmo_core.nn.rope import RoPEScalingConfig
from olmo_core.nn.transformer.block import (
    MoEHybridReorderedNormTransformerBlock,
    ReorderedNormTransformerBlock,
)
from olmo_core.nn.transformer.model import (
    MoETransformer,
    NormalizedTransformer,
    Transformer,
)


def _get_moe_hf_config(model: MoETransformer) -> PretrainedConfig:
    moe_block_keys: list[str] = []
    regular_blocks_keys: list[str] = []
    for key, block in model.blocks.items():
        if isinstance(block, MoEHybridReorderedNormTransformerBlock):
            moe_block_keys.append(key)
        elif isinstance(block, ReorderedNormTransformerBlock):
            regular_blocks_keys.append(key)
        else:
            raise NotImplementedError(
                f"Block is not a {MoEHybridReorderedNormTransformerBlock.__name__} or {ReorderedNormTransformerBlock.__name__}, unable to build HF config for {model.__class__.__name__}"
            )

    for block in model.blocks.values():
        assert isinstance(
            block, (MoEHybridReorderedNormTransformerBlock, ReorderedNormTransformerBlock)
        )

        if not isinstance(block.attention, Attention):
            raise NotImplementedError(
                f"Attention is not a {Attention.__name__}, unable to build HF config for {model.__class__.__name__}"
            )

        if block.attention.rope is None:
            raise NotImplementedError(
                f"Attention does not use rope, unable to build HF config for {model.__class__.__name__}"
            )

        if not block.attention.use_head_qk_norm:
            raise NotImplementedError(
                f"Head qk norm is not enabled, unable to build HF config for {model.__class__.__name__}"
            )

    for key in moe_block_keys:
        block = model.blocks[key]
        assert isinstance(block, MoEHybridReorderedNormTransformerBlock)

        if not isinstance(block.experts.mlp, MoEMLP):
            raise NotImplementedError(
                f"MoE mlp is not a {MoEMLP.__name__}, unable to build HF config for {model.__class__.__name__}"
            )

        if block.router.normalize_expert_weights is not None:
            raise NotImplementedError(
                f"Expert weight normalization is not supported, unable to build HF config for {model.__class__.__name__}"
            )

        if block.router.gating_function != MoERouterGatingFunction.sigmoid:
            raise NotImplementedError(
                f"Only sigmoid gating function is supported for MoE router, unable to build HF config for {model.__class__.__name__}"
            )

    moe_block = model.blocks[moe_block_keys[0]]
    assert isinstance(moe_block, MoEHybridReorderedNormTransformerBlock)
    assert isinstance(moe_block.attention, Attention)

    if len(regular_blocks_keys) > 0:
        regular_block = model.blocks[regular_blocks_keys[0]]
        assert isinstance(regular_block, ReorderedNormTransformerBlock)
        intermediate_size = regular_block.feed_forward.hidden_size
    else:
        intermediate_size = moe_block.feed_forward.hidden_size

    # Handle RoPE scaling validation
    all_blocks = list(model.blocks.values())
    rope_scaling = _validate_rope_scaling_config(all_blocks)

    config: PretrainedConfig = Olmoe2Config(
        vocab_size=model.vocab_size,
        hidden_size=model.d_model,
        intermediate_size=intermediate_size,
        num_hidden_layers=model.n_layers,
        num_attention_heads=moe_block.attention.n_heads,
        num_key_value_heads=moe_block.attention.n_kv_heads,
        hidden_act="silu",
        max_position_embeddings=-1,
        attention_bias=moe_block.attention.w_out.bias is not None,
        rope_theta=moe_block.attention.rope.theta,
        rope_scaling=rope_scaling,
        pad_token_id=None,  # type: ignore
        bos_token_id=None,
        eos_token_id=None,  # type: ignore
        rms_norm_eps=moe_block.feed_forward_norm.eps,
        num_experts_per_tok=moe_block.feed_forward_moe.router.top_k,
        num_experts=moe_block.feed_forward_moe.router.num_experts,
        tie_word_embeddings=False,
        moe_intermediate_size=moe_block.feed_forward_moe.experts.mlp.hidden_size,
        shared_mlp_intermediate_size=moe_block.feed_forward.hidden_size,
        mlp_only_layers=sorted([int(key) for key in regular_blocks_keys]),
        sliding_window=max(block.attention.window_size[0] + 1 for block in model.blocks.values()),
        layer_types=[
            "sliding_attention" if block.attention.window_size != (-1, -1) else "full_attention"
            for block in model.blocks.values()
        ],
    )
    return config


@beta_feature
def get_hf_config(model: Transformer) -> PretrainedConfig:
    if isinstance(model, MoETransformer):
        return _get_moe_hf_config(model)

    if isinstance(model, NormalizedTransformer):
        raise NotImplementedError(
            f"Building HF config not implemented for {model.__class__.__name__}"
        )

    blocks = list(model.blocks.values())
    if not isinstance(blocks[0], ReorderedNormTransformerBlock):
        raise NotImplementedError(
            f"Block is not a {ReorderedNormTransformerBlock.__name__}, unable to build HF config for {model.__class__.__name__}"
        )

    if not isinstance(blocks[0].attention, Attention):
        raise NotImplementedError(
            f"Attention is not a {Attention.__name__}, unable to build HF config for {model.__class__.__name__}"
        )
    if blocks[0].attention.rope is None:
        raise NotImplementedError(
            f"Attention does not use rope, unable to build HF config for {model.__class__.__name__}"
        )

    rope_scaling = _get_and_validate_rope_scaling_config(blocks)

    if blocks[0].attention.use_head_qk_norm:
        return Olmo3Config(
            vocab_size=model.vocab_size,
            hidden_size=model.d_model,
            intermediate_size=blocks[0].feed_forward.hidden_size,
            num_hidden_layers=model.n_layers,
            num_attention_heads=blocks[0].attention.n_heads,
            num_key_value_heads=blocks[0].attention.n_kv_heads,
            hidden_act="silu",
            max_position_embeddings=-1,
            attention_bias=blocks[0].attention.w_out.bias is not None,
            rope_theta=blocks[0].attention.rope.theta,
            rope_scaling=rope_scaling,
            pad_token_id=None,  # type: ignore
            bos_token_id=None,
            eos_token_id=None,  # type: ignore
            rms_norm_eps=blocks[0].feed_forward_norm.eps,
            tie_word_embeddings=False,
            sliding_window=max(block.attention.window_size[0] + 1 for block in blocks),
            layer_types=[
                "sliding_attention" if block.attention.window_size != (-1, -1) else "full_attention"
                for block in blocks
            ],
        )
    else:
        if any(block.attention.window_size != (-1, -1) for block in blocks):
            return Olmo2RetrofitConfig(
                vocab_size=model.vocab_size,
                hidden_size=model.d_model,
                intermediate_size=blocks[0].feed_forward.hidden_size,
                num_hidden_layers=model.n_layers,
                num_attention_heads=blocks[0].attention.n_heads,
                num_key_value_heads=blocks[0].attention.n_kv_heads,
                hidden_act="silu",
                max_position_embeddings=-1,
                attention_bias=blocks[0].attention.w_out.bias is not None,
                rope_theta=blocks[0].attention.rope.theta,
                rope_scaling=rope_scaling,
                pad_token_id=None,  # type: ignore
                bos_token_id=None,
                eos_token_id=None,  # type: ignore
                rms_norm_eps=blocks[0].feed_forward_norm.eps,
                tie_word_embeddings=False,
                sliding_window=max(block.attention.window_size[0] + 1 for block in blocks),
                layer_types=[
                    "sliding_attention"
                    if block.attention.window_size != (-1, -1)
                    else "full_attention"
                    for block in blocks
                ],
            )

        if rope_scaling is not None:
            raise NotImplementedError(
                f"OLMo 2 does not support Llama3 RoPE, unable to build HF config for {model.__class__.__name__}"
            )

        return Olmo2Config(
            vocab_size=model.vocab_size,
            hidden_size=model.d_model,
            intermediate_size=blocks[0].feed_forward.hidden_size,
            num_hidden_layers=model.n_layers,
            num_attention_heads=blocks[0].attention.n_heads,
            num_key_value_heads=blocks[0].attention.n_kv_heads,
            hidden_act="silu",
            max_position_embeddings=-1,
            attention_bias=blocks[0].attention.w_out.bias is not None,
            rope_theta=blocks[0].attention.rope.theta,
            pad_token_id=None,  # type: ignore
            bos_token_id=None,
            eos_token_id=None,  # type: ignore
            rms_norm_eps=blocks[0].feed_forward_norm.eps,
            tie_word_embeddings=False,
        )


def _get_and_validate_rope_scaling_config(blocks) -> dict | None:
    """
    Validate RoPE scaling configuration across transformer blocks.

    :param blocks: The list of transformer blocks to validate.
    :returns: The validated RoPE scaling config dict for HF, or None if no scaling.
    :raises NotImplementedError: If RoPE scaling is applied to sliding window layers or if
                               full attention layers have different RoPE scaling configs.
    """
    # Separate full attention layers from sliding window layers
    full_attention_layers = [
        (idx, block)
        for idx, block in enumerate(blocks)
        if block.attention.sliding_window is None
        or not blocks.attention.sliding_window.should_use_swa(idx, len(blocks))
    ]
    sliding_window_layers = [
        (idx, block)
        for idx, block in enumerate(blocks)
        if (
            block.attention.sliding_window is not None
            and block.attention.sliding_window.should_use_swa(idx, len(blocks))
        )
    ]

    # Check for RoPE scaling on sliding window layers (not allowed)
    sliding_with_scaling = [
        (idx, block)
        for idx, block in sliding_window_layers
        if block.attention.rope.scaling is not None
    ]
    if sliding_with_scaling:
        sliding_indices = [idx for idx, _ in sliding_with_scaling]
        raise NotImplementedError(
            f"RoPE scaling is configured on sliding window attention layers {sliding_indices}, "
            f"but HuggingFace only supports RoPE scaling on full attention layers. "
            f"Please remove RoPE scaling from sliding window layers or convert them to full attention."
        )

    # Collect RoPE scaling configs from full attention layers only
    full_layers_with_scaling = [
        (idx, block)
        for idx, block in full_attention_layers
        if block.attention.rope.scaling is not None
    ]
    if not full_layers_with_scaling:
        return None

    rope_scaling_configs: list[RoPEScalingConfig] = [
        block.attention.rope.scaling for _, block in full_layers_with_scaling
    ]

    # Validate that all full attention layers with RoPE scaling use the same configuration
    first_config = rope_scaling_configs[0]
    first_config_dict = first_config.to_hf_dict()

    for i, rope_config in enumerate(rope_scaling_configs[1:], 1):
        config_dict = rope_config.to_hf_dict()
        if config_dict != first_config_dict:
            scaling_indices = [idx for idx, _ in full_layers_with_scaling]
            raise NotImplementedError(
                f"Full attention layers have different RoPE scaling configurations but HuggingFace "
                f"only supports model-wide RoPE scaling. Full attention layers with scaling: {scaling_indices}. "
                f"First config: {first_config_dict}, Different config at layer {i}: {config_dict}"
            )

    return first_config_dict
