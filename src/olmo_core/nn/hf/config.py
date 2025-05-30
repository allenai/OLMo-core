from transformers import Olmo2Config, PretrainedConfig, LlamaConfig

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
    block = next(iter(model.blocks.values()))

    return LlamaConfig(
        vocab_size=model.vocab_size,
        hidden_size=model.d_model,
        intermediate_size=block.feed_forward.hidden_size,
        num_hidden_layers=model.n_layers,
        num_attention_heads=block.attention.n_heads,
        num_key_value_heads=block.attention.n_kv_heads,
        hidden_act="silu",
        max_position_embeddings=-1,
        attention_bias=block.attention.w_out.bias is not None,
        rope_theta=block.attention.rope.theta,
        pad_token_id=None,  # type: ignore
        bos_token_id=None,
        eos_token_id=None,  # type: ignore
        rms_norm_eps=block.feed_forward_norm.eps,
        tie_word_embeddings=False,
    )
