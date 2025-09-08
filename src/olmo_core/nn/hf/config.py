from transformers import Olmo2Config, PretrainedConfig

from olmo_core.doc_utils import beta_feature
from olmo_core.nn.attention import Attention
from olmo_core.nn.transformer.block import ReorderedNormTransformerBlock
from olmo_core.nn.transformer.model import (
    MoETransformer,
    NormalizedTransformer,
    Transformer,
)

try:
    from transformers import Olmo3Config  # type: ignore
except ImportError:
    Olmo3Config = None


@beta_feature
def get_hf_config(model: Transformer) -> PretrainedConfig:
    if isinstance(model, (MoETransformer, NormalizedTransformer)):
        raise NotImplementedError(
            f"Building HF config not implemented for {model.__class__.__name__}"
        )

    for block in model.blocks.values():
        if not isinstance(block, ReorderedNormTransformerBlock):
            raise NotImplementedError(
                f"Block is not a {ReorderedNormTransformerBlock.__name__}, unable to build HF config for {model.__class__.__name__}"
            )

        if not isinstance(block.attention, Attention):
            raise NotImplementedError(
                f"Attention is not a {Attention.__name__}, unable to build HF config for {model.__class__.__name__}"
            )
        if block.attention.rope is None:
            raise NotImplementedError(
                f"Attention does not use rope, unable to build HF config for {model.__class__.__name__}"
            )

    blocks = list(model.blocks.values())
    first_block = blocks[0]
    assert isinstance(first_block, ReorderedNormTransformerBlock)
    attention = first_block.attention
    assert isinstance(attention, Attention)
    rope = attention.rope
    assert rope is not None

    if any(block.attention.window_size != (-1, -1) for block in blocks):
        # Has sliding window, should be an Olmo3 model.
        if Olmo3Config is None:
            raise RuntimeError("The current transformers version does not support Olmo3.")

        return Olmo3Config(
            vocab_size=model.vocab_size,
            hidden_size=model.d_model,
            intermediate_size=first_block.feed_forward.hidden_size,
            num_hidden_layers=model.n_layers,
            num_attention_heads=attention.n_heads,
            num_key_value_heads=attention.n_kv_heads,
            hidden_act="silu",
            max_position_embeddings=-1,
            attention_bias=attention.w_out.bias is not None,
            rope_theta=rope.theta,
            pad_token_id=None,  # type: ignore
            bos_token_id=None,
            eos_token_id=None,  # type: ignore
            rms_norm_eps=first_block.feed_forward_norm.eps,
            tie_word_embeddings=False,
            sliding_window=max(block.attention.window_size[0] + 1 for block in blocks),
            layer_types=[
                "sliding_attention" if block.attention.window_size != (-1, -1) else "full_attention"
                for block in blocks
            ],
        )

    return Olmo2Config(
        vocab_size=model.vocab_size,
        hidden_size=model.d_model,
        intermediate_size=first_block.feed_forward.hidden_size,
        num_hidden_layers=model.n_layers,
        num_attention_heads=attention.n_heads,
        num_key_value_heads=attention.n_kv_heads,
        hidden_act="silu",
        max_position_embeddings=-1,
        attention_bias=attention.w_out.bias is not None,
        rope_theta=rope.theta,
        pad_token_id=None,  # type: ignore
        bos_token_id=None,
        eos_token_id=None,  # type: ignore
        rms_norm_eps=first_block.feed_forward_norm.eps,
        tie_word_embeddings=False,
    )
