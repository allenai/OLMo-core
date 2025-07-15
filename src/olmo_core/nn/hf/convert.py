from typing import Any, Dict, List

from transformers import PretrainedConfig

from olmo_core.doc_utils import beta_feature
from olmo_core.nn.conversion.state_converter import StateConverter
from olmo_core.nn.conversion.state_mapping import (
    StateMappingTemplate,
    StateType,
    TemplatePlaceholder,
)

LAYER = TemplatePlaceholder.LAYER
EXPERT = TemplatePlaceholder.EXPERT


#: Map of Hugging Face keys to OLMo Core keys, that is used to determine how HF state
#: maps to OLMo Core state. Different HF models may use different names for a given OLMo
#: Core state. You may configure this to change how HF state maps to OLMo Core state.
#:
#: This map only captures one-to-one mappings from HF to OLMo Core. For many-to-many mappings
#: or mappings that require additional manipulation of state, see
#: :data:`HF_TO_OLMO_CORE_TEMPLATE_MAPPINGS`. If a given HF key can refer to different OLMo Core
#: states depending on the HF model, see :data:`MODEL_SPECIFIC_HF_TO_OLMO_CORE_MAPPINGS`.
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
    f"model.layers.{LAYER}.mlp.shared_mlp.gate_proj.weight": f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w1.weight",
    f"model.layers.{LAYER}.mlp.shared_mlp.down_proj.weight": f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w2.weight",
    f"model.layers.{LAYER}.mlp.shared_mlp.up_proj.weight": f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w3.weight",
}


#: Map of Hugging Face keys to OLMo Core keys. This map captures overrides of the standard
#: one-to-one mappings in :data:`HF_TO_OLMO_CORE_MAPPINGS`, in case a given HF key can refer to
#: different OLMo Core states depending on the HF model. You may configure this to change
#: how HF state maps to OLMo Core state.
MODEL_SPECIFIC_HF_TO_OLMO_CORE_MAPPINGS: Dict[str, Dict[str, str]] = {
    "meta-llama/Llama-3.2-1B": {
        f"model.layers.{LAYER}.post_attention_layernorm.weight": f"blocks.{LAYER}.feed_forward_norm.weight"
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        f"model.layers.{LAYER}.post_attention_layernorm.weight": f"blocks.{LAYER}.feed_forward_norm.weight"
    },
}


#: Map of Hugging Face keys to OLMo Core keys, that is used to determine how HF state
#: maps to OLMo Core state. Different HF models may use different names for a given OLMo
#: Core state. You may configure this to change how HF state maps to OLMo Core state.
#:
#: This map captures many-to-many mappings from HF to OLMo Core and mappings that require
#: additional manipulation of state (e.g. merging dimensions).
#: For simple one-to-one mappings from HF to OLMo Core, see
#: :data:`HF_TO_OLMO_CORE_MAPPINGS`.
HF_TO_OLMO_CORE_TEMPLATE_MAPPINGS: Dict[str, StateMappingTemplate] = {
    f"model.layers.{LAYER}.mlp.block_sparse_moe.experts.{EXPERT}.gate_proj.weight": StateMappingTemplate(
        f"model.layers.{LAYER}.mlp.block_sparse_moe.experts.{EXPERT}.gate_proj.weight",
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1",
        source_key_per_placeholder=TemplatePlaceholder.EXPERT,
        source_concat_dim=1,
        dims_permutation=(1, 0),
        dest_chunk_dim=0,
    ),
    f"model.layers.{LAYER}.mlp.block_sparse_moe.experts.{EXPERT}.down_proj.weight": StateMappingTemplate(
        f"model.layers.{LAYER}.mlp.block_sparse_moe.experts.{EXPERT}.down_proj.weight",
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2",
        source_key_per_placeholder=TemplatePlaceholder.EXPERT,
        source_concat_dim=1,
        dims_permutation=(1, 0),
        dest_chunk_dim=0,
    ),
    f"model.layers.{LAYER}.mlp.block_sparse_moe.experts.{EXPERT}.up_proj.weight": StateMappingTemplate(
        f"model.layers.{LAYER}.mlp.block_sparse_moe.experts.{EXPERT}.up_proj.weight",
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3",
        source_key_per_placeholder=TemplatePlaceholder.EXPERT,
        source_concat_dim=1,
        dims_permutation=(1, 0),
        dest_chunk_dim=0,
    ),
    f"model.layers.{LAYER}.mlp.block_sparse_moe.gate.weight": StateMappingTemplate(
        f"model.layers.{LAYER}.mlp.block_sparse_moe.gate.weight",
        f"blocks.{LAYER}.feed_forward_moe.router.weight",
        flatten_dims=(0, 1),
    ),
}


#: Map of OLMo Core weight keys to Hugging Face weight keys, that is used to determine how OLMo Core state
#: maps to HF state. You may configure this to change how OLMo Core state maps to HF state.
#:
#: This map only captures one-to-one mappings from OLMo Core to HF. For many-to-many mappings
#: or mappings that require additional manipulation of state, see :data:`OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS`.
OLMO_CORE_TO_HF_WEIGHT_MAPPINGS: Dict[str, str] = {
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
    # f"blocks.{LAYER}.feed_forward_moe.router.weight": f"model.layers.{LAYER}.mlp.gate.weight",
    f"blocks.{LAYER}.feed_forward_moe_norm.weight": f"model.layers.{LAYER}.post_moe_norm.weight",
    f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w1.weight": f"model.layers.{LAYER}.mlp.shared_mlp.gate_proj.weight",
    f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w2.weight": f"model.layers.{LAYER}.mlp.shared_mlp.down_proj.weight",
    f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w3.weight": f"model.layers.{LAYER}.mlp.shared_mlp.up_proj.weight",
}


#: Map of OLMo Core module keys to Hugging Face module keys, that is used to determine how OLMo Core state
#: maps to HF state. You may configure this to change how OLMo Core state maps to HF state.
#:
#: This map only captures one-to-one mappings from OLMo Core to HF. For many-to-many mappings
#: or mappings that require additional manipulation of state, see :data:`OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS`.
OLMO_CORE_TO_HF_MODULE_MAPPINGS: Dict[str, str] = {
    "embeddings": "model.embed_tokens",
    "lm_head.norm": "model.norm",
    "lm_head.w_out": "lm_head",
    # Attention.
    f"blocks.{LAYER}.attention.w_q": f"model.layers.{LAYER}.self_attn.q_proj",
    f"blocks.{LAYER}.attention.w_k": f"model.layers.{LAYER}.self_attn.k_proj",
    f"blocks.{LAYER}.attention.w_v": f"model.layers.{LAYER}.self_attn.v_proj",
    f"blocks.{LAYER}.attention.w_out": f"model.layers.{LAYER}.self_attn.o_proj",
    # MLP.
    f"blocks.{LAYER}.feed_forward": f"model.layers.{LAYER}.mlp",
    f"blocks.{LAYER}.feed_forward.w1": f"model.layers.{LAYER}.mlp.gate_proj",
    f"blocks.{LAYER}.feed_forward.w2": f"model.layers.{LAYER}.mlp.down_proj",
    f"blocks.{LAYER}.feed_forward.w3": f"model.layers.{LAYER}.mlp.up_proj",
    # Layer norms.
    f"blocks.{LAYER}.attention_norm": f"model.layers.{LAYER}.post_attention_layernorm",
    f"blocks.{LAYER}.feed_forward_norm": f"model.layers.{LAYER}.post_feedforward_layernorm",
    f"blocks.{LAYER}.attention.q_norm": f"model.layers.{LAYER}.self_attn.q_norm",
    f"blocks.{LAYER}.attention.k_norm": f"model.layers.{LAYER}.self_attn.k_norm",
    # MoEMLP.
    f"blocks.{LAYER}.feed_forward_moe": f"model.layers.{LAYER}.block_sparse_moe",
    f"blocks.{LAYER}.feed_forward_moe_norm": f"model.layers.{LAYER}.post_moe_norm",
    f"blocks.{LAYER}.feed_forward_moe.router": f"model.layers.{LAYER}.block_sparse_moe.gate",
    f"blocks.{LAYER}.feed_forward_moe.shared_mlp": f"model.layers.{LAYER}.block_sparse_moe.shared_mlp",
    f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w1": f"model.layers.{LAYER}.block_sparse_moe.shared_mlp.gate_proj",
    f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w2": f"model.layers.{LAYER}.block_sparse_moe.shared_mlp.down_proj",
    f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w3": f"model.layers.{LAYER}.block_sparse_moe.shared_mlp.up_proj",
}


#: Map of OLMo Core keys to Hugging Face keys, that is used to determine how OLMo Core state
#: maps to HF state. You may configure this to change how OLMo Core state maps to HF state.
#:
#: This map captures many-to-many mappings from OLMo Core to HF and mappings that require
#: additional manipulation of state (e.g. merging dimensions).
#: For simple one-to-one mappings from OLMo Core to HF, see
#: :data:`OLMO_CORE_TO_HF_MAPPINGS`.
OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS: Dict[str, StateMappingTemplate] = {
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1": StateMappingTemplate(
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1",
        f"model.layers.{LAYER}.block_sparse_moe.experts.{EXPERT}.gate_proj.weight",
        dest_key_per_placeholder=TemplatePlaceholder.EXPERT,
        source_concat_dim=0,
        dims_permutation=(1, 0),
        dest_chunk_dim=1,
    ),
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2": StateMappingTemplate(
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2",
        f"model.layers.{LAYER}.block_sparse_moe.experts.{EXPERT}.down_proj.weight",
        dest_key_per_placeholder=TemplatePlaceholder.EXPERT,
        source_concat_dim=0,
        dims_permutation=(1, 0),
        dest_chunk_dim=1,
    ),
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3": StateMappingTemplate(
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3",
        f"model.layers.{LAYER}.block_sparse_moe.experts.{EXPERT}.up_proj.weight",
        dest_key_per_placeholder=TemplatePlaceholder.EXPERT,
        source_concat_dim=0,
        dims_permutation=(1, 0),
        dest_chunk_dim=1,
    ),
    f"blocks.{LAYER}.feed_forward_moe.router.weight": StateMappingTemplate(
        f"blocks.{LAYER}.feed_forward_moe.router.weight",
        f"model.layers.{LAYER}.block_sparse_moe.gate.weight",
        unflatten_dim=(0, (TemplatePlaceholder.EXPERT, -1)),
    ),
}


def _get_hf_model_to_olmo_core_one_to_one_templates(
    model_id: str | None = None,
) -> List[StateMappingTemplate]:
    mapping_templates = {
        hf_key: StateMappingTemplate(hf_key, olmo_core_key)
        for hf_key, olmo_core_key in HF_TO_OLMO_CORE_MAPPINGS.items()
    }
    if model_id in MODEL_SPECIFIC_HF_TO_OLMO_CORE_MAPPINGS:
        model_specific_mapping_templates = {
            hf_key: StateMappingTemplate(hf_key, olmo_core_key)
            for hf_key, olmo_core_key in MODEL_SPECIFIC_HF_TO_OLMO_CORE_MAPPINGS[model_id].items()
        }
        mapping_templates.update(model_specific_mapping_templates)

    return list(mapping_templates.values())


def _get_converter_from_hf(model_id: str | None = None) -> StateConverter:
    mapping_templates = _get_hf_model_to_olmo_core_one_to_one_templates(model_id)
    mapping_templates += list(HF_TO_OLMO_CORE_TEMPLATE_MAPPINGS.values())
    return StateConverter(mapping_templates)


@beta_feature
def get_converter_from_hf(model_id: str | None = None) -> StateConverter:
    return _get_converter_from_hf(model_id=model_id)


@beta_feature
def convert_state_from_hf(
    config: PretrainedConfig,
    hf_state: Dict[str, Any],
    *,
    model_id: str | None = None,
) -> Dict[str, Any]:
    """
    Converts a model state dict in Hugging Face transformers format into an unsharded state dict of
    OLMo Core format.

    :param config: The Hugging Face config for the model
    :param hf_state: A model state dict in HF format.
    :param model_id: The model id of the HF model in HF Hub.
    """

    converter = _get_converter_from_hf(model_id=model_id)

    if not hasattr(config, "num_hidden_layers"):
        raise ValueError(f"Number of hidden layers missing in HF config: {config}")
    n_layers: int = config.num_hidden_layers
    n_experts: int | None = getattr(config, "num_experts", None)

    placeholder_bounds = {
        TemplatePlaceholder.LAYER: n_layers,
    }
    if n_experts:
        placeholder_bounds[TemplatePlaceholder.EXPERT] = n_experts

    return converter.convert(hf_state, placeholder_bounds)


def _get_converter_to_hf() -> StateConverter:
    mapping_templates = [
        StateMappingTemplate(olmo_core_key, hf_key, state_type=StateType.module)
        for olmo_core_key, hf_key in OLMO_CORE_TO_HF_MODULE_MAPPINGS.items()
    ]
    mapping_templates += [
        StateMappingTemplate(olmo_core_key, hf_key, state_type=StateType.weight)
        for olmo_core_key, hf_key in OLMO_CORE_TO_HF_WEIGHT_MAPPINGS.items()
    ]
    mapping_templates += list(OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS.values())
    return StateConverter(mapping_templates)


@beta_feature
def get_converter_to_hf() -> StateConverter:
    return _get_converter_to_hf()


@beta_feature
def convert_state_to_hf(
    config: PretrainedConfig, olmo_core_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Converts an *unsharded* model state dict of OLMo Core format into Hugging Face transformers format.

    :param config: The Hugging Face config for the model
    :param olmo_core_state: An unsharded OLMo Core model state dict. None of the states can be
        :class:`DTensor` or :class:`ShardedTensor`
    """

    converter = _get_converter_to_hf()

    if not hasattr(config, "num_hidden_layers"):
        raise ValueError(f"Number of hidden layers missing in HF config: {config}")
    n_layers: int = config.num_hidden_layers
    n_experts: int | None = getattr(config, "num_experts", None)

    placeholder_bounds = {
        TemplatePlaceholder.LAYER: n_layers,
    }
    if n_experts:
        placeholder_bounds[TemplatePlaceholder.EXPERT] = n_experts

    return converter.convert(olmo_core_state, placeholder_bounds)
