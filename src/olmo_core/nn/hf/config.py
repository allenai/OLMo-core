from transformers import LlamaConfig, Olmo2Config, PretrainedConfig
from transformers.models.olmpool.configuration_olmpool import (
    Llama3ReorderedConfig,
    Olmo2HeadwiseQKNormConfig,
    Olmo3PreorderConfig,
    Olmo3PreorderHeadwiseQKNormConfig,
)

from olmo_core.doc_utils import beta_feature
from olmo_core.nn.attention import Attention
from olmo_core.nn.transformer.block import ReorderedNormTransformerBlock, TransformerBlock
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

    block = next(iter(model.blocks.values()))
    if not isinstance(block, (ReorderedNormTransformerBlock, TransformerBlock)):
        raise NotImplementedError(
            f"Block is not a {TransformerBlock.__name__} or {ReorderedNormTransformerBlock.__name__}, "
            f"unable to build HF config for {model.__class__.__name__}"
        )

    if not isinstance(block.attention, Attention):
        raise NotImplementedError(
            f"Attention is not a {Attention.__name__}, unable to build HF config for {model.__class__.__name__}"
        )
    if block.attention.rope is None:
        raise NotImplementedError(
            f"Attention does not use rope, unable to build HF config for {model.__class__.__name__}"
        )

    has_qk_norm = block.attention.q_norm is not None
    has_headwise_qk_norm = has_qk_norm and getattr(block.attention, "use_head_qk_norm", False)

    common = dict(
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

    # ReorderedNormTransformerBlock is a subclass of TransformerBlock — check it first.
    if isinstance(block, ReorderedNormTransformerBlock):
        if has_qk_norm:
            if has_headwise_qk_norm:
                return Olmo2HeadwiseQKNormConfig(**common)
            return Olmo2Config(**common)
        else:
            return Llama3ReorderedConfig(**common)
    else:  # TransformerBlock (default pre-norm)
        if has_qk_norm:
            if has_headwise_qk_norm:
                return Olmo3PreorderHeadwiseQKNormConfig(**common)
            return Olmo3PreorderConfig(**common)
        else:
            return LlamaConfig(**common)
