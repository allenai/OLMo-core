import copy
from typing import Dict, Iterable, List

from transformers import PretrainedConfig

from olmo_core.doc_utils import beta_feature
from olmo_core.nn.conversion.key_mapping import (
    TemplateMapping,
    TemplatePlaceholder,
    TensorMapping,
)

LAYER = TemplatePlaceholder.LAYER
EXPERT = TemplatePlaceholder.EXPERT


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
    f"model.layers.{LAYER}.self_attn.q_proj.weight": f"blocks.{LAYER}.attention.w_q.weight",
    f"model.layers.{LAYER}.self_attn.k_proj.weight": f"blocks.{LAYER}.attention.w_k.weight",
    f"model.layers.{LAYER}.self_attn.v_proj.weight": f"blocks.{LAYER}.attention.w_v.weight",
    f"model.layers.{LAYER}.self_attn.o_proj.weight": f"blocks.{LAYER}.attention.w_out.weight",
    # MLP.
    f"model.layers.{LAYER}.mlp.gate_proj.weight": f"blocks.{LAYER}.feed_forward.w1.weight",
    f"model.layers.{LAYER}.mlp.down_proj.weight": f"blocks.{LAYER}.feed_forward.w2.weight",
    f"model.layers.{LAYER}.mlp.up_proj.weight": f"blocks.{LAYER}.feed_forward.w3.weight",
    # Layer norms.
    f"model.layers.{LAYER}.input_layernorm.weight": f"blocks.{LAYER}.attention_norm.weight",
    f"model.layers.{LAYER}.post_attention_layernorm.weight": f"blocks.{LAYER}.attention_norm.weight",
    f"model.layers.{LAYER}.post_feedforward_layernorm.weight": f"blocks.{LAYER}.feed_forward_norm.weight",
    f"model.layers.{LAYER}.self_attn.q_norm.weight": f"blocks.{LAYER}.attention.q_norm.weight",
    f"model.layers.{LAYER}.self_attn.k_norm.weight": f"blocks.{LAYER}.attention.k_norm.weight",
    # MoEMLP.
    f"model.layers.{LAYER}.mlp.gate.weight": f"blocks.{LAYER}.feed_forward_moe.router.weight",
    f"model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight": f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1",
    f"model.layers.{LAYER}.mlp.experts.{EXPERT}.down_proj.weight": f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2",
    f"model.layers.{LAYER}.mlp.experts.{EXPERT}.up_proj.weight": f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3",
}


MODEL_SPECIFIC_HF_TO_OLMO_CORE_MAPPINGS: Dict[str, Dict[str, str]] = {
    "meta-llama/Llama-3.2-1B": {
        f"model.layers.{LAYER}.post_attention_layernorm.weight": f"blocks.{LAYER}.feed_forward_norm.weight"
    }
}

OLMO_CORE_TO_HF_MAPPINGS: Dict[str, str] = {
    "embeddings.weight": "model.embed_tokens.weight",
    "lm_head.norm.weight": "model.norm.weight",
    "lm_head.w_out.weight": "lm_head.weight",
    # Attention.
    f"blocks.{LAYER}.attention.w_q.weight": f"model.layers.{LAYER}.self_attn.q_proj.weight",
    f"blocks.{LAYER}.attention.w_k.weight": f"model.layers.{LAYER}.self_attn.k_proj.weight",
    f"blocks.{LAYER}.attention.w_v.weight": f"model.layers.{LAYER}.self_attn.v_proj.weight",
    f"blocks.{LAYER}.attention.w_out.weight": f"model.layers.{LAYER}.self_attn.o_proj.weight",
    # MLP.
    f"blocks.{LAYER}.feed_forward.w1.weight": f"model.layers.{LAYER}.mlp.gate_proj.weight",
    f"blocks.{LAYER}.feed_forward.w2.weight": f"model.layers.{LAYER}.mlp.down_proj.weight",
    f"blocks.{LAYER}.feed_forward.w3.weight": f"model.layers.{LAYER}.mlp.up_proj.weight",
    # Layer norms.
    f"blocks.{LAYER}.attention_norm.weight": f"model.layers.{LAYER}.post_attention_layernorm.weight",
    f"blocks.{LAYER}.feed_forward_norm.weight": f"model.layers.{LAYER}.post_feedforward_layernorm.weight",
    f"blocks.{LAYER}.attention.q_norm.weight": f"model.layers.{LAYER}.self_attn.q_norm.weight",
    f"blocks.{LAYER}.attention.k_norm.weight": f"model.layers.{LAYER}.self_attn.k_norm.weight",
    # MoEMLP.
    f"blocks.{LAYER}.feed_forward_moe.router.weight": f"model.layers.{LAYER}.mlp.gate.weight",
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1": f"model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight",
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2": f"model.layers.{LAYER}.mlp.experts.{EXPERT}.down_proj.weight",
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3": f"model.layers.{LAYER}.mlp.experts.{EXPERT}.up_proj.weight",
}


HF_TO_OLMO_CORE_TEMPLATE_MAPPING: Dict[str, TemplateMapping] = {
    f"model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight": TemplateMapping(
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight",
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1",
        source_key_per_expert=True,
        source_concat_dim=1,
        dims_permutation=(1, 0),
        dest_chunk_dim=0,
    ),
    f"model.layers.{LAYER}.mlp.experts.{EXPERT}.down_proj.weight": TemplateMapping(
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.down_proj.weight",
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2",
        source_key_per_expert=True,
        source_concat_dim=1,
        dims_permutation=(1, 0),
        dest_chunk_dim=0,
    ),
    f"model.layers.{LAYER}.mlp.experts.{EXPERT}.up_proj.weight": TemplateMapping(
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.up_proj.weight",
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3",
        source_key_per_expert=True,
        source_concat_dim=1,
        dims_permutation=(1, 0),
        dest_chunk_dim=0,
    ),
    f"model.layers.{LAYER}.mlp.gate.weight": TemplateMapping(
        f"model.layers.{LAYER}.mlp.gate.weight",
        f"blocks.{LAYER}.feed_forward_moe.router.weight",
        flatten_dims=(0, 1),
    ),
}

OLMO_CORE_TO_HF_TEMPLATE_MAPPING: Dict[str, TemplateMapping] = {
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1": TemplateMapping(
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1",
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight",
        dest_key_per_expert=True,
        source_concat_dim=0,
        dims_permutation=(1, 0),
        dest_chunk_dim=1,
    ),
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2": TemplateMapping(
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2",
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.down_proj.weight",
        dest_key_per_expert=True,
        source_concat_dim=0,
        dims_permutation=(1, 0),
        dest_chunk_dim=1,
    ),
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3": TemplateMapping(
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3",
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.up_proj.weight",
        dest_key_per_expert=True,
        source_concat_dim=0,
        dims_permutation=(1, 0),
        dest_chunk_dim=1,
    ),
    f"blocks.{LAYER}.feed_forward_moe.router.weight": TemplateMapping(
        f"blocks.{LAYER}.feed_forward_moe.router.weight",
        f"model.layers.{LAYER}.mlp.gate.weight",
        unflatten_dim=(0, (TemplatePlaceholder.EXPERT, -1)),
    ),
}


def _get_hf_to_olmo_core_mapping(
    hf_key_template: str,
    olmo_core_key_template: str,
    i_block: int,
    n_blocks: int,
    i_expert: int | None = None,
    n_experts: int = 0,
) -> TensorMapping | None:
    template_mapping = HF_TO_OLMO_CORE_TEMPLATE_MAPPING.get(
        hf_key_template, TemplateMapping(hf_key_template, olmo_core_key_template)
    )

    mapping = template_mapping.to_mapping(i_block, n_blocks, i_expert, n_experts)
    if len(mapping.source_keys) == 0 or len(mapping.dest_keys) == 0:
        return None

    return mapping


@beta_feature
def get_key_mapping_from_hf(
    config: PretrainedConfig, hf_state_keys: Iterable[str], *, model_id: str | None = None
) -> List[TensorMapping]:
    if not hasattr(config, "num_hidden_layers"):
        raise ValueError(f"Number of hidden layers missing in HF config: {config}")
    n_blocks: int = config.num_hidden_layers
    n_experts: int = getattr(config, "num_experts", 0)

    template_mappings = copy.deepcopy(HF_TO_OLMO_CORE_MAPPINGS)
    if model_id is not None and model_id in MODEL_SPECIFIC_HF_TO_OLMO_CORE_MAPPINGS:
        template_mappings.update(copy.deepcopy(MODEL_SPECIFIC_HF_TO_OLMO_CORE_MAPPINGS[model_id]))

    mappings = [
        _get_hf_to_olmo_core_mapping(
            k,
            v,
            i_block,
            n_blocks,
            i_expert=None if n_experts == 0 else i_expert,
            n_experts=n_experts,
        )
        for i_block in range(n_blocks)
        for i_expert in range(n_experts or 1)
        for k, v in template_mappings.items()
    ]

    hf_state_keys = set(hf_state_keys)
    return [
        mapping
        for mapping in mappings
        if mapping and all(k in hf_state_keys for k in mapping.source_keys)
    ]


def _get_olmo_core_to_hf_mapping(
    olmo_core_key_template: str,
    hf_key_template: str,
    i_block: int,
    n_blocks: int,
    i_expert: int | None = None,
    n_experts: int = 0,
) -> TensorMapping | None:
    template_mapping = OLMO_CORE_TO_HF_TEMPLATE_MAPPING.get(
        olmo_core_key_template, TemplateMapping(olmo_core_key_template, hf_key_template)
    )

    mapping = template_mapping.to_mapping(i_block, n_blocks, i_expert, n_experts)
    if len(mapping.source_keys) == 0 or len(mapping.dest_keys) == 0:
        return None

    return mapping


@beta_feature
def get_key_mapping_to_hf(
    config: PretrainedConfig, olmo_state_keys: Iterable[str]
) -> List[TensorMapping]:
    if not hasattr(config, "num_hidden_layers"):
        raise ValueError(f"Number of hidden layers missing in HF config: {config}")
    n_blocks: int = config.num_hidden_layers
    n_experts: int = getattr(config, "num_experts", 0)

    mappings = [
        _get_olmo_core_to_hf_mapping(
            k,
            v,
            i_block,
            n_blocks,
            i_expert=None if n_experts == 0 else i_expert,
            n_experts=n_experts,
        )
        for i_block in range(n_blocks)
        for i_expert in range(n_experts or 1)
        for k, v in OLMO_CORE_TO_HF_MAPPINGS.items()
    ]

    olmo_state_keys = set(olmo_state_keys)
    return [
        mapping
        for mapping in mappings
        if mapping and all(k in olmo_state_keys for k in mapping.source_keys)
    ]
