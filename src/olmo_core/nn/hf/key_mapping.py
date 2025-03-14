import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from transformers import PretrainedConfig

from olmo_core.doc_utils import beta_feature

BLOCK_STR = "[block]"
EXPERT_STR = "[expert]"

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
    # MoEMLP.
    f"model.layers.{BLOCK_STR}.mlp.gate.weight": f"blocks.{BLOCK_STR}.feed_forward_moe.router.weight",
    f"model.layers.{BLOCK_STR}.mlp.experts.{EXPERT_STR}.gate_proj.weight": f"blocks.{BLOCK_STR}.feed_forward_moe.experts.mlp.w1",
    f"model.layers.{BLOCK_STR}.mlp.experts.{EXPERT_STR}.down_proj.weight": f"blocks.{BLOCK_STR}.feed_forward_moe.experts.mlp.w2",
    f"model.layers.{BLOCK_STR}.mlp.experts.{EXPERT_STR}.up_proj.weight": f"blocks.{BLOCK_STR}.feed_forward_moe.experts.mlp.w3",
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
    # MoEMLP.
    f"blocks.{BLOCK_STR}.feed_forward_moe.router.weight": f"model.layers.{BLOCK_STR}.mlp.gate.weight",
    f"blocks.{BLOCK_STR}.feed_forward_moe.experts.mlp.w1": f"model.layers.{BLOCK_STR}.mlp.experts.{EXPERT_STR}.gate_proj.weight",
    f"blocks.{BLOCK_STR}.feed_forward_moe.experts.mlp.w2": f"model.layers.{BLOCK_STR}.mlp.experts.{EXPERT_STR}.down_proj.weight",
    f"blocks.{BLOCK_STR}.feed_forward_moe.experts.mlp.w3": f"model.layers.{BLOCK_STR}.mlp.experts.{EXPERT_STR}.up_proj.weight",
}


@dataclass
class TemplateMapping:
    source_template_keys: str | Tuple[str, ...]
    dest_template_keys: str | Tuple[str, ...]

    source_key_per_block: bool = False
    source_key_per_expert: bool = False
    dest_key_per_block: bool = False
    dest_key_per_expert: bool = False

    source_concat_dim: int = 0
    dims_permutation: Tuple[int, ...] | None = None
    unflatten_dim: Tuple[int, Tuple[str | int, ...]] | None = None
    flatten_dims: Tuple[int, int] | None = None
    dest_chunk_dim: int = 0

    def __post_init__(self):
        if (self.source_key_per_block or self.source_key_per_expert) and isinstance(
            self.source_template_keys, tuple
        ):
            raise ValueError(
                "Having a key per block or expert is not supported with multiple template keys"
            )

        if self.source_key_per_block and self.source_key_per_expert:
            raise ValueError("Can only have a key per block or per expert, not both")

        if (self.dest_key_per_block or self.dest_key_per_expert) and isinstance(
            self.dest_template_keys, tuple
        ):
            raise ValueError(
                "Having a key per block or expert is not supported with multiple template keys"
            )

        if self.dest_key_per_block and self.dest_key_per_expert:
            raise ValueError("Can only have a key per block or per expert, not both")

    def _templates_to_keys(
        self,
        templates: str | Tuple[str, ...],
        key_per_block: bool,
        key_per_expert: bool,
        i_block: int,
        n_blocks: int,
        i_expert: int | None = None,
        n_experts: int = 0,
    ) -> Tuple[str, ...]:
        if key_per_block:
            assert isinstance(templates, str)
            templates = tuple(templates.replace(BLOCK_STR, str(i)) for i in range(n_blocks))
        elif key_per_expert:
            assert isinstance(templates, str)
            templates = tuple(templates.replace(EXPERT_STR, str(i)) for i in range(n_experts))
        elif isinstance(templates, str):
            templates = (templates,)

        assert isinstance(templates, tuple)

        return tuple(
            template.replace(BLOCK_STR, str(i_block)).replace(EXPERT_STR, str(i_expert))
            for template in templates
        )

    def to_mapping(
        self, i_block: int, n_blocks: int, i_expert: int | None = None, n_experts: int = 0
    ):
        source_keys = self._templates_to_keys(
            self.source_template_keys,
            self.source_key_per_block,
            self.source_key_per_expert,
            i_block,
            n_blocks,
            i_expert,
            n_experts,
        )
        dest_keys = self._templates_to_keys(
            self.dest_template_keys,
            self.dest_key_per_block,
            self.dest_key_per_expert,
            i_block,
            n_blocks,
            i_expert,
            n_experts,
        )

        unflatten_dim = None
        if self.unflatten_dim is not None:
            unflatten_dim_shape = tuple(
                n_blocks if dim == BLOCK_STR else n_experts if dim == EXPERT_STR else int(dim)
                for dim in self.unflatten_dim[1]
            )
            unflatten_dim = (self.unflatten_dim[0], unflatten_dim_shape)

        return Mapping(
            source_keys,
            dest_keys,
            self.source_concat_dim,
            self.dims_permutation,
            unflatten_dim,
            self.flatten_dims,
            self.dest_chunk_dim,
        )


HF_TO_OLMO_CORE_TEMPLATE_MAPPING: Dict[str, TemplateMapping] = {
    f"model.layers.{BLOCK_STR}.mlp.experts.{EXPERT_STR}.gate_proj.weight": TemplateMapping(
        f"model.layers.{BLOCK_STR}.mlp.experts.{EXPERT_STR}.gate_proj.weight",
        f"blocks.{BLOCK_STR}.feed_forward_moe.experts.mlp.w1",
        source_key_per_expert=True,
        source_concat_dim=1,
        dims_permutation=(1, 0),
        dest_chunk_dim=0,
    ),
    f"model.layers.{BLOCK_STR}.mlp.experts.{EXPERT_STR}.down_proj.weight": TemplateMapping(
        f"model.layers.{BLOCK_STR}.mlp.experts.{EXPERT_STR}.down_proj.weight",
        f"blocks.{BLOCK_STR}.feed_forward_moe.experts.mlp.w2",
        source_key_per_expert=True,
        source_concat_dim=1,
        dims_permutation=(1, 0),
        dest_chunk_dim=0,
    ),
    f"model.layers.{BLOCK_STR}.mlp.experts.{EXPERT_STR}.up_proj.weight": TemplateMapping(
        f"model.layers.{BLOCK_STR}.mlp.experts.{EXPERT_STR}.up_proj.weight",
        f"blocks.{BLOCK_STR}.feed_forward_moe.experts.mlp.w3",
        source_key_per_expert=True,
        source_concat_dim=1,
        dims_permutation=(1, 0),
        dest_chunk_dim=0,
    ),
    f"model.layers.{BLOCK_STR}.mlp.gate.weight": TemplateMapping(
        f"model.layers.{BLOCK_STR}.mlp.gate.weight",
        f"blocks.{BLOCK_STR}.feed_forward_moe.router.weight",
        flatten_dims=(0, 1),
    ),
}

OLMO_CORE_TO_HF_TEMPLATE_MAPPING: Dict[str, TemplateMapping] = {
    f"blocks.{BLOCK_STR}.feed_forward_moe.experts.mlp.w1": TemplateMapping(
        f"blocks.{BLOCK_STR}.feed_forward_moe.experts.mlp.w1",
        f"model.layers.{BLOCK_STR}.mlp.experts.{EXPERT_STR}.gate_proj.weight",
        dest_key_per_expert=True,
        source_concat_dim=0,
        dims_permutation=(1, 0),
        dest_chunk_dim=1,
    ),
    f"blocks.{BLOCK_STR}.feed_forward_moe.experts.mlp.w2": TemplateMapping(
        f"blocks.{BLOCK_STR}.feed_forward_moe.experts.mlp.w2",
        f"model.layers.{BLOCK_STR}.mlp.experts.{EXPERT_STR}.down_proj.weight",
        dest_key_per_expert=True,
        source_concat_dim=0,
        dims_permutation=(1, 0),
        dest_chunk_dim=1,
    ),
    f"blocks.{BLOCK_STR}.feed_forward_moe.experts.mlp.w3": TemplateMapping(
        f"blocks.{BLOCK_STR}.feed_forward_moe.experts.mlp.w3",
        f"model.layers.{BLOCK_STR}.mlp.experts.{EXPERT_STR}.up_proj.weight",
        dest_key_per_expert=True,
        source_concat_dim=0,
        dims_permutation=(1, 0),
        dest_chunk_dim=1,
    ),
    f"blocks.{BLOCK_STR}.feed_forward_moe.router.weight": TemplateMapping(
        f"blocks.{BLOCK_STR}.feed_forward_moe.router.weight",
        f"model.layers.{BLOCK_STR}.mlp.gate.weight",
        unflatten_dim=(0, (EXPERT_STR, -1))
    ),
}


@dataclass
class Mapping:
    source_keys: Tuple[str, ...]
    dest_keys: Tuple[str, ...]

    source_concat_dim: int = 0
    dims_permutation: Tuple[int, ...] | None = None
    unflatten_dim: Tuple[int, Tuple[int, ...]] | None = None
    flatten_dims: Tuple[int, int] | None = None
    dest_chunk_dim: int = 0


# def _get_keys(
#     key_template: str,
#     chunking_key: str | None,
#     i_block: int,
#     n_blocks: int,
#     i_expert: int | None = None,
#     n_experts: int = 0,
# ) -> Tuple[str, ...]:
#     if chunking_key is None:
#         return (key_template.replace(BLOCK_STR, str(i_block)).replace(EXPERT_STR, str(i_expert)),)

#     if chunking_key == EXPERT_STR:
#         n_chunks = n_experts or 1
#     elif chunking_key == BLOCK_STR:
#         n_chunks = n_blocks
#     else:
#         raise NotImplementedError(chunking_key)

#     key_templates = [key_template.replace(chunking_key, str(i)) for i in range(n_chunks)]
#     return tuple(
#         template.replace(BLOCK_STR, str(i_block)).replace(EXPERT_STR, str(i_expert))
#         for template in key_templates
#     )


def _get_hf_to_olmo_core_mapping(
    hf_key_template: str,
    olmo_core_key_template: str,
    i_block: int,
    n_blocks: int,
    i_expert: int | None = None,
    n_experts: int = 0,
) -> Mapping | None:
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
) -> List[Mapping]:
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
) -> Mapping | None:
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
) -> List[Mapping]:
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
