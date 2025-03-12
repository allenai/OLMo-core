from typing import Dict

from transformers import PretrainedConfig

from olmo_core.doc_utils import beta_feature

BLOCK_STR = "[block]"

# Map of Hugging Face keys to olmo_core keys. Different HF models may use different
# names for a given olmo_core state, but we assume for now that the same HF name does not
# refer to different olmo_core states in different models. That is, we assume that
# the mapping from HF state names to olmo_core state names is many to one.
# TODO: Update this comment
HF_TO_OLMO_CORE_MAPPINGS: Dict[str, str] = {
    "model.embed_tokens.weight": "embeddings.weight",
    "model.norm.weight": "lm_head.norm.weight",
    "lm_head.weight": "lm_head.w_out.weight",
    # Attention.
    f"model.layers.{BLOCK_STR}.self_attn.q_proj.weight": f"blocks.{BLOCK_STR}.attention.w_q.weight",
    f"model.layers.{BLOCK_STR}.self_attn.k_proj.weight": f"blocks.{BLOCK_STR}.attention.w_k.weight",
    f"model.layers.{BLOCK_STR}.self_attn.v_proj.weight": f"blocks.{BLOCK_STR}.attention.w_v.weight",
    f"model.layers.{BLOCK_STR}.self_attn.o_proj.weight": f"blocks.{BLOCK_STR}.attention.w_out.weight",
    # MLP.
    f"model.layers.{BLOCK_STR}.mlp.gate_proj.weight": f"blocks.{BLOCK_STR}.feed_forward.w1.weight",
    f"model.layers.{BLOCK_STR}.mlp.down_proj.weight": f"blocks.{BLOCK_STR}.feed_forward.w2.weight",
    f"model.layers.{BLOCK_STR}.mlp.up_proj.weight": f"blocks.{BLOCK_STR}.feed_forward.w3.weight",
    # Layer norms.
    f"model.layers.{BLOCK_STR}.input_layernorm.weight": f"blocks.{BLOCK_STR}.attention_norm.weight",
    f"model.layers.{BLOCK_STR}.post_attention_layernorm.weight": f"blocks.{BLOCK_STR}.attention_norm.weight",
    f"model.layers.{BLOCK_STR}.post_feedforward_layernorm.weight": f"blocks.{BLOCK_STR}.feed_forward_norm.weight",
    f"model.layers.{BLOCK_STR}.self_attn.q_norm.weight": f"blocks.{BLOCK_STR}.attention.q_norm.weight",
    f"model.layers.{BLOCK_STR}.self_attn.k_norm.weight": f"blocks.{BLOCK_STR}.attention.k_norm.weight",
}

MODEL_SPECIFIC_HF_TO_OLMO_CORE_MAPPINGS: Dict[str, Dict[str, str]] = {
    "meta-llama/Llama-3.2-1B": {
        f"model.layers.{BLOCK_STR}.post_attention_layernorm.weight": f"blocks.{BLOCK_STR}.feed_forward_norm.weight"
    }
}

OLMO_CORE_TO_HF_MAPPINGS: Dict[str, str] = {
    "embeddings.weight": "model.embed_tokens.weight",
    "lm_head.norm.weight": "model.norm.weight",
    "lm_head.w_out.weight": "lm_head.weight",
    # Attention.
    f"blocks.{BLOCK_STR}.attention.w_q.weight": f"model.layers.{BLOCK_STR}.self_attn.q_proj.weight",
    f"blocks.{BLOCK_STR}.attention.w_k.weight": f"model.layers.{BLOCK_STR}.self_attn.k_proj.weight",
    f"blocks.{BLOCK_STR}.attention.w_v.weight": f"model.layers.{BLOCK_STR}.self_attn.v_proj.weight",
    f"blocks.{BLOCK_STR}.attention.w_out.weight": f"model.layers.{BLOCK_STR}.self_attn.o_proj.weight",
    # MLP.
    f"blocks.{BLOCK_STR}.feed_forward.w1.weight": f"model.layers.{BLOCK_STR}.mlp.gate_proj.weight",
    f"blocks.{BLOCK_STR}.feed_forward.w2.weight": f"model.layers.{BLOCK_STR}.mlp.down_proj.weight",
    f"blocks.{BLOCK_STR}.feed_forward.w3.weight": f"model.layers.{BLOCK_STR}.mlp.up_proj.weight",
    # Layer norms.
    f"blocks.{BLOCK_STR}.attention_norm.weight": f"model.layers.{BLOCK_STR}.post_attention_layernorm.weight",
    f"blocks.{BLOCK_STR}.feed_forward_norm.weight": f"model.layers.{BLOCK_STR}.post_feedforward_layernorm.weight",
    f"blocks.{BLOCK_STR}.attention.q_norm.weight": f"model.layers.{BLOCK_STR}.self_attn.q_norm.weight",
    f"blocks.{BLOCK_STR}.attention.k_norm.weight": f"model.layers.{BLOCK_STR}.self_attn.k_norm.weight",
}


@beta_feature
def get_key_mapping_from_hf(
    config: PretrainedConfig, *, model_id: str | None = None
) -> Dict[str, str]:
    if not hasattr(config, "num_hidden_layers"):
        raise ValueError(f"Number of hidden layers missing in HF config: {config}")
    n_layers: int = config.num_hidden_layers

    mapping = {
        k.replace(BLOCK_STR, str(i)): v.replace(BLOCK_STR, str(i))
        for i in range(n_layers)
        for k, v in HF_TO_OLMO_CORE_MAPPINGS.items()
    }

    if model_id is not None:
        model_specific_mapping = {
            k.replace(BLOCK_STR, str(i)): v.replace(BLOCK_STR, str(i))
            for i in range(n_layers)
            for k, v in MODEL_SPECIFIC_HF_TO_OLMO_CORE_MAPPINGS.get(model_id, {}).items()
        }
        mapping.update(model_specific_mapping)

    return mapping


@beta_feature
def get_key_mapping_to_hf(config: PretrainedConfig) -> Dict[str, str]:
    if not hasattr(config, "num_hidden_layers"):
        raise ValueError(f"Number of hidden layers missing in HF config: {config}")
    n_layers: int = config.num_hidden_layers

    return {
        k.replace(BLOCK_STR, str(i)): v.replace(BLOCK_STR, str(i))
        for i in range(n_layers)
        for k, v in OLMO_CORE_TO_HF_MAPPINGS.items()
    }
