import typing

from transformers import Olmo2Config, PretrainedConfig

from olmo_core.doc_utils import beta_feature
from olmo_core.nn.attention import Attention
from olmo_core.nn.transformer.block import ReorderedNormTransformerBlock
from olmo_core.nn.transformer.model import (
    MoETransformer,
    NormalizedTransformer,
    Transformer,
)


@beta_feature
def get_hf_config(model: Transformer) -> PretrainedConfig:
    if isinstance(model, (MoETransformer, NormalizedTransformer)):
        raise NotImplementedError(
            f"Building HF config not implemented for {model.__class__.__name__}"
        )

    blocks = [block for _, block in sorted(model.blocks.items(), key=lambda item: int(item[0]))]
    if not isinstance(blocks[0], ReorderedNormTransformerBlock):
        raise NotImplementedError(
            f"Block is not a {ReorderedNormTransformerBlock.__name__}, unable to build HF config for {model.__class__.__name__}"
        )
    blocks = typing.cast(list[ReorderedNormTransformerBlock], blocks)

    if not isinstance(blocks[0].attention, Attention):
        raise NotImplementedError(
            f"Attention is not a {Attention.__name__}, unable to build HF config for {model.__class__.__name__}"
        )
    if blocks[0].attention.rope is None:
        raise NotImplementedError(
            f"Attention does not use rope, unable to build HF config for {model.__class__.__name__}"
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
        # Our sliding window is a pair of offsets (left/right), HF's is the window size
        sliding_window=max(block.attention.window_size[0] + 1 for block in blocks),  # type: ignore
        layer_types=[
            "sliding_attention" if block.attention.window_size[0] != -1 else "full_attention"  # type: ignore
            for block in blocks
        ],
    )
