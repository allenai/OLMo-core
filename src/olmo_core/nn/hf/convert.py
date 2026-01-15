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


#: Map of Hugging Face weight keys to OLMo Core weight keys, that is used to determine how HF state
#: maps to OLMo Core state. Different HF models may use different names for a given OLMo
#: Core state. You may configure this to change how HF state maps to OLMo Core state.
#:
#: This map only captures one-to-one mappings from HF to OLMo Core. For many-to-many mappings
#: or mappings that require additional manipulation of state, see
#: :data:`HF_TO_OLMO_CORE_TEMPLATE_MAPPINGS`. If a given HF key can refer to different OLMo Core
#: states depending on the HF model, see :data:`MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_WEIGHT_MAPPINGS`.
HF_TO_OLMO_CORE_WEIGHT_MAPPINGS: Dict[str, str] = {
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


#: Map of Hugging Face module keys to OLMo Core module keys, that is used to determine how HF state
#: maps to OLMo Core state. Different HF models may use different names for a given OLMo
#: Core state. You may configure this to change how HF state maps to OLMo Core state.
#:
#: This map only captures one-to-one mappings from HF to OLMo Core. For many-to-many mappings
#: or mappings that require additional manipulation of state, see
#: :data:`HF_TO_OLMO_CORE_TEMPLATE_MAPPINGS`. If a given HF key can refer to different OLMo Core
#: states depending on the HF model, see :data:`MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_MODULE_MAPPINGS`.
HF_TO_OLMO_CORE_MODULE_MAPPINGS: Dict[str, str] = {
    "model.embed_tokens": "embeddings",
    "model.norm": "lm_head.norm",
    "lm_head": "lm_head.w_out",
    # Attention.
    f"model.layers.{LAYER}.self_attn.q_proj": f"blocks.{LAYER}.attention.w_q",
    f"model.layers.{LAYER}.self_attn.k_proj": f"blocks.{LAYER}.attention.w_k",
    f"model.layers.{LAYER}.self_attn.v_proj": f"blocks.{LAYER}.attention.w_v",
    f"model.layers.{LAYER}.self_attn.o_proj": f"blocks.{LAYER}.attention.w_out",
    # MLP.
    f"model.layers.{LAYER}.mlp.gate_proj": f"blocks.{LAYER}.feed_forward.w1",
    f"model.layers.{LAYER}.mlp.down_proj": f"blocks.{LAYER}.feed_forward.w2",
    f"model.layers.{LAYER}.mlp.up_proj": f"blocks.{LAYER}.feed_forward.w3",
    # Layer norms.
    f"model.layers.{LAYER}.input_layernorm": f"blocks.{LAYER}.attention_norm",
    f"model.layers.{LAYER}.post_attention_layernorm": f"blocks.{LAYER}.attention_norm",
    f"model.layers.{LAYER}.post_feedforward_layernorm": f"blocks.{LAYER}.feed_forward_norm",
    f"model.layers.{LAYER}.self_attn.q_norm": f"blocks.{LAYER}.attention.q_norm",
    f"model.layers.{LAYER}.self_attn.k_norm": f"blocks.{LAYER}.attention.k_norm",
    # MoEMLP.
    f"model.layers.{LAYER}.mlp": f"blocks.{LAYER}.feed_forward_moe",
    f"model.layers.{LAYER}.post_moe_norm": f"blocks.{LAYER}.feed_forward_moe_norm",
    f"model.layers.{LAYER}.mlp.gate": f"blocks.{LAYER}.feed_forward_moe.router",
    # Indices are not part of the original OLMo Core state but can be introduced during conversion for aide debugging.
    f"model.layers.{LAYER}.mlp.gate.indices": f"blocks.{LAYER}.feed_forward_moe.router.indices",
    f"model.layers.{LAYER}.mlp.shared_mlp": f"blocks.{LAYER}.feed_forward_moe.shared_mlp",
    f"model.layers.{LAYER}.mlp.shared_mlp.gate_proj": f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w1",
    f"model.layers.{LAYER}.mlp.shared_mlp.down_proj": f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w2",
    f"model.layers.{LAYER}.mlp.shared_mlp.up_proj": f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w3",
}


#: Map of Hugging Face weight keys to OLMo Core weight keys. This map captures overrides of the standard
#: one-to-one mappings in :data:`HF_TO_OLMO_CORE_WEIGHT_MAPPINGS`, in case a given HF key can refer to
#: different OLMo Core states depending on the HF model architecture. You may configure this to change
#: how HF state maps to OLMo Core state.
MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_WEIGHT_MAPPINGS: Dict[str, Dict[str, str]] = {
    "llama": {
        f"model.layers.{LAYER}.post_attention_layernorm.weight": f"blocks.{LAYER}.feed_forward_norm.weight"
    },
}

#: Map of Hugging Face module keys to OLMo Core module keys. This map captures overrides of the standard
#: one-to-one mappings in :data:`HF_TO_OLMO_CORE_MODULE_MAPPINGS`, in case a given HF key can refer to
#: different OLMo Core states depending on the HF model architecture. You may configure this to change
#: how HF state maps to OLMo Core state.
MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_MODULE_MAPPINGS: Dict[str, Dict[str, str]] = {
    "llama": {
        f"model.layers.{LAYER}.post_attention_layernorm": f"blocks.{LAYER}.feed_forward_norm"
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
    f"model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight": StateMappingTemplate(
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight",
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1",
        source_key_per_placeholder=TemplatePlaceholder.EXPERT,
        source_concat_dim=1,
        dims_permutation=(1, 0),
    ),
    f"model.layers.{LAYER}.mlp.experts.{EXPERT}.down_proj.weight": StateMappingTemplate(
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.down_proj.weight",
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2",
        source_key_per_placeholder=TemplatePlaceholder.EXPERT,
        source_concat_dim=1,
        dims_permutation=(1, 0),
    ),
    f"model.layers.{LAYER}.mlp.experts.{EXPERT}.up_proj.weight": StateMappingTemplate(
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.up_proj.weight",
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3",
        source_key_per_placeholder=TemplatePlaceholder.EXPERT,
        source_concat_dim=1,
        dims_permutation=(1, 0),
    ),
    f"model.layers.{LAYER}.mlp.gate.weight": StateMappingTemplate(
        f"model.layers.{LAYER}.mlp.gate.weight",
        f"blocks.{LAYER}.feed_forward_moe.router.weight",
        flatten_dims=(0, 1),
    ),
}


#: Map of Hugging Face keys to OLMo Core keys. This map captures overrides of the standard
#: mappings in :data:`HF_TO_OLMO_CORE_TEMPLATE_MAPPINGS`, in case a given HF key can refer to
#: different OLMo Core states depending on the HF model. You may configure this to change how HF
#: state maps to OLMo Core state.
MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_TEMPLATE_MAPPINGS: Dict[
    str, Dict[str, StateMappingTemplate]
] = {
    "flex_olmo": {
        # FlexOlmo uses fused gate_up_proj and stacks all experts in single tensors.
        # Split gate_up_proj into w1 (gate) and w3 (up):
        # - gate_up_proj: (num_experts, 2 * hidden_size, d_model)
        # - w1: (num_experts * d_model, hidden_size)
        # - w3: (num_experts * d_model, hidden_size)
        f"model.layers.{LAYER}.mlp.experts.gate_up_proj": StateMappingTemplate(
            f"model.layers.{LAYER}.mlp.experts.gate_up_proj",
            (
                f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1",
                f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3",
            ),
            unflatten_dim=(1, (2, -1)),  # (num_experts, 2, hidden_size, d_model)
            dims_permutation=(1, 0, 3, 2),  # (2, num_experts, d_model, hidden_size)
            flatten_dims=(0, 2),  # (2 * num_experts * d_model, hidden_size)
            dest_chunk_dim=0,  # chunk into w1 and w3
        ),
        # Convert down_proj to w2:
        # - down_proj: (num_experts, d_model, hidden_size)
        # - w2: (num_experts * hidden_size, d_model)
        f"model.layers.{LAYER}.mlp.experts.down_proj": StateMappingTemplate(
            f"model.layers.{LAYER}.mlp.experts.down_proj",
            f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2",
            dims_permutation=(0, 2, 1),  # (num_experts, hidden_size, d_model)
            flatten_dims=(0, 1),  # (num_experts * hidden_size, d_model)
        ),
        f"model.layers.{LAYER}.mlp.gate.weight": StateMappingTemplate(
            f"model.layers.{LAYER}.mlp.gate.weight",
            f"blocks.{LAYER}.feed_forward_moe.router.weight",
            flatten_dims=(0, 1),
        ),
    }
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
    f"blocks.{LAYER}.feed_forward_moe.router.weight": f"model.layers.{LAYER}.mlp.gate.weight",
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1": f"model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight",
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2": f"model.layers.{LAYER}.mlp.experts.{EXPERT}.down_proj.weight",
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3": f"model.layers.{LAYER}.mlp.experts.{EXPERT}.up_proj.weight",
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
    f"blocks.{LAYER}.feed_forward_moe": f"model.layers.{LAYER}.mlp",
    f"blocks.{LAYER}.feed_forward_moe_norm": f"model.layers.{LAYER}.post_moe_norm",
    f"blocks.{LAYER}.feed_forward_moe.router": f"model.layers.{LAYER}.mlp.gate",
    # Indices are not part of the original OLMo Core state but can be introduced during conversion for aide debugging.
    f"blocks.{LAYER}.feed_forward_moe.router.indices": f"model.layers.{LAYER}.mlp.gate.indices",
    f"blocks.{LAYER}.feed_forward_moe.shared_mlp": f"model.layers.{LAYER}.mlp.shared_mlp",
    f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w1": f"model.layers.{LAYER}.mlp.shared_mlp.gate_proj",
    f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w2": f"model.layers.{LAYER}.mlp.shared_mlp.down_proj",
    f"blocks.{LAYER}.feed_forward_moe.shared_mlp.w3": f"model.layers.{LAYER}.mlp.shared_mlp.up_proj",
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
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj.weight",
        dest_key_per_placeholder=TemplatePlaceholder.EXPERT,
        dims_permutation=(1, 0),
        dest_chunk_dim=1,
    ),
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2": StateMappingTemplate(
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2",
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.down_proj.weight",
        dest_key_per_placeholder=TemplatePlaceholder.EXPERT,
        dims_permutation=(1, 0),
        dest_chunk_dim=1,
    ),
    f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3": StateMappingTemplate(
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3",
        f"model.layers.{LAYER}.mlp.experts.{EXPERT}.up_proj.weight",
        dest_key_per_placeholder=TemplatePlaceholder.EXPERT,
        dims_permutation=(1, 0),
        dest_chunk_dim=1,
    ),
    f"blocks.{LAYER}.feed_forward_moe.router.weight": StateMappingTemplate(
        f"blocks.{LAYER}.feed_forward_moe.router.weight",
        f"model.layers.{LAYER}.mlp.gate.weight",
        unflatten_dim=(0, (TemplatePlaceholder.EXPERT, -1)),
    ),
}


#: Map of OLMo Core keys to Hugging Face keys, that is used to determine how OLMo Core state
#: maps to HF state. This map captures overrides of the standard mappings in
#: :data:`OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS`, in case a given OLMo Core key can refer to
#: different HF states depending on the HF model. You may configure this to change how OLMo Core
#: state maps to HF state.
# FlexOlmo uses fused gate_up_proj and stacks all experts in single tensors.
# This mapping combines w1 (gate) and w3 (up) into fused gate_up_proj:
# - w1: (num_experts * d_model, hidden_size)
# - w3: (num_experts * d_model, hidden_size)
# - gate_up_proj: (num_experts, 2 * hidden_size, d_model)
_FLEX_OLMO_GATE_UP_PROJ_MAPPING = StateMappingTemplate(
    (
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1",
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3",
    ),
    f"model.layers.{LAYER}.mlp.experts.gate_up_proj",
    source_concat_dim=0,  # concat to (2 * num_experts * d_model, hidden_size)
    unflatten_dim=(0, (2, TemplatePlaceholder.EXPERT, -1)),  # (2, num_experts, d_model, hidden_size)
    dims_permutation=(1, 0, 3, 2),  # (num_experts, 2, hidden_size, d_model)
    flatten_dims=(1, 2),  # (num_experts, 2 * hidden_size, d_model)
)

MODEL_TYPE_SPECIFIC_OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS: Dict[
    str, Dict[str, StateMappingTemplate]
] = {
    "flex_olmo": {
        # Both w1 and w3 entries point to the same combined mapping to override default per-expert mappings
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w1": _FLEX_OLMO_GATE_UP_PROJ_MAPPING,
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w3": _FLEX_OLMO_GATE_UP_PROJ_MAPPING,
        # Convert w2 (down) to down_proj:
        # - w2: (num_experts * hidden_size, d_model)
        # - down_proj: (num_experts, d_model, hidden_size)
        f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2": StateMappingTemplate(
            f"blocks.{LAYER}.feed_forward_moe.experts.mlp.w2",
            f"model.layers.{LAYER}.mlp.experts.down_proj",
            unflatten_dim=(0, (TemplatePlaceholder.EXPERT, -1)),  # (num_experts, hidden_size, d_model)
            dims_permutation=(0, 2, 1),  # (num_experts, d_model, hidden_size)
        ),
        f"blocks.{LAYER}.feed_forward_moe.router.weight": StateMappingTemplate(
            f"blocks.{LAYER}.feed_forward_moe.router.weight",
            f"model.layers.{LAYER}.mlp.gate.weight",
            unflatten_dim=(0, (TemplatePlaceholder.EXPERT, -1)),
        ),
    }
}


def _get_hf_model_to_olmo_core_one_to_one_templates(
    model_type: str | None = None,
) -> List[StateMappingTemplate]:
    mapping_templates = {
        hf_key: StateMappingTemplate(hf_key, olmo_core_key, state_type=StateType.weight)
        for hf_key, olmo_core_key in HF_TO_OLMO_CORE_WEIGHT_MAPPINGS.items()
    }

    for hf_key, olmo_core_key in HF_TO_OLMO_CORE_MODULE_MAPPINGS.items():
        mapping_templates[hf_key] = StateMappingTemplate(
            hf_key, olmo_core_key, state_type=StateType.module
        )

    if model_type in MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_WEIGHT_MAPPINGS:
        model_type_specific_mapping_templates = {
            hf_key: StateMappingTemplate(hf_key, olmo_core_key, state_type=StateType.weight)
            for hf_key, olmo_core_key in MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_WEIGHT_MAPPINGS[
                model_type
            ].items()
        }
        mapping_templates.update(model_type_specific_mapping_templates)

    if model_type in MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_MODULE_MAPPINGS:
        model_type_specific_mapping_templates = {
            hf_key: StateMappingTemplate(hf_key, olmo_core_key, state_type=StateType.module)
            for hf_key, olmo_core_key in MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_MODULE_MAPPINGS[
                model_type
            ].items()
        }
        mapping_templates.update(model_type_specific_mapping_templates)

    return list(mapping_templates.values())


def _get_converter_from_hf(model_type: str | None = None) -> StateConverter:
    mapping_templates = _get_hf_model_to_olmo_core_one_to_one_templates(model_type)

    # Use model-type-specific template mappings if available, otherwise use default
    if model_type and model_type in MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_TEMPLATE_MAPPINGS:
        mapping_templates += list(
            MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_TEMPLATE_MAPPINGS[model_type].values()
        )
    else:
        mapping_templates += list(HF_TO_OLMO_CORE_TEMPLATE_MAPPINGS.values())

    return StateConverter(mapping_templates)


@beta_feature
def get_converter_from_hf(model_type: str | None = None) -> StateConverter:
    return _get_converter_from_hf(model_type=model_type)


def _convert_state(
    config: PretrainedConfig,
    state: Dict[str, Any],
    converter: StateConverter,
) -> Dict[str, Any]:
    if not hasattr(config, "num_hidden_layers"):
        raise ValueError(f"Number of hidden layers missing in HF config: {config}")
    n_layers: int = config.num_hidden_layers
    n_experts: int | None = getattr(config, "num_experts", None)

    placeholder_bounds = {
        TemplatePlaceholder.LAYER: n_layers,
    }
    if n_experts:
        placeholder_bounds[TemplatePlaceholder.EXPERT] = n_experts

    return converter.convert(state, placeholder_bounds)


@beta_feature
def convert_state_from_hf(
    config: PretrainedConfig,
    hf_state: Dict[str, Any],
    *,
    model_type: str | None = None,
) -> Dict[str, Any]:
    """
    Converts a model state dict in Hugging Face transformers format into an unsharded state dict of
    OLMo Core format.

    :param config: The Hugging Face config for the model
    :param hf_state: A model state dict in HF format.
    :param model_type: The model type of the HF model.
    """

    converter = _get_converter_from_hf(model_type=model_type)

    return _convert_state(config, hf_state, converter)


def _get_converter_to_hf(model_type: str | None = None) -> StateConverter:
    mapping_templates = {
        olmo_core_key: StateMappingTemplate(olmo_core_key, hf_key, state_type=StateType.module)
        for olmo_core_key, hf_key in OLMO_CORE_TO_HF_MODULE_MAPPINGS.items()
    }
    mapping_templates.update(
        {
            olmo_core_key: StateMappingTemplate(olmo_core_key, hf_key, state_type=StateType.weight)
            for olmo_core_key, hf_key in OLMO_CORE_TO_HF_WEIGHT_MAPPINGS.items()
        }
    )
    mapping_templates.update(OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS)

    if model_type:
        mapping_templates.update(
            MODEL_TYPE_SPECIFIC_OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS.get(model_type, {})
        )

    return StateConverter(list(mapping_templates.values()))


@beta_feature
def get_converter_to_hf(model_type: str | None = None) -> StateConverter:
    return _get_converter_to_hf(model_type)


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

    converter = _get_converter_to_hf(getattr(config, "model_type", None))

    return _convert_state(config, olmo_core_state, converter)
