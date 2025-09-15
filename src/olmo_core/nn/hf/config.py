from transformers import Olmo2Config, Olmo3Config, PretrainedConfig

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

    if blocks[0].attention.use_head_qk_norm:
        config = Olmo3Config(
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
            sliding_window=max(block.attention.window_size[0] + 1 for block in blocks),
            layer_types=[
                "sliding_attention" if block.attention.window_size != (-1, -1) else "full_attention"
                for block in blocks
            ],
        )
        
        if blocks[0].attention.use_sinks:
            config.use_sinks = True
            if hasattr(blocks[0].attention, 'sinks') and blocks[0].attention.sinks is not None:
                sinks_shape = blocks[0].attention.sinks.shape
                if len(sinks_shape) == 1:
                    config.sinks_shape = "per_head"
                elif len(sinks_shape) == 2:
                    config.sinks_shape = "per_head_per_token"
                    config.num_sink_tokens = sinks_shape[1]
            
        return config
    else:
        if any(block.attention.window_size != (-1, -1) for block in blocks):
            raise NotImplementedError(
                f"Sliding window attention is not supported without head qk norm, unable to build HF config for {model.__class__.__name__}"
            )

        config = Olmo2Config(
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
         
        if blocks[0].attention.use_sinks:
            config.use_sinks = True
            if hasattr(blocks[0].attention, 'sinks') and blocks[0].attention.sinks is not None:
                sinks_shape = blocks[0].attention.sinks.shape
                if len(sinks_shape) == 1:
                    config.sinks_shape = "per_head" 
                elif len(sinks_shape) == 2:
                    config.sinks_shape = "per_head_per_token"
                    config.num_sink_tokens = sinks_shape[1]
            
        return config