import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from olmo_core.config import StrEnum


class TemplatePlaceholder(StrEnum):
    LAYER = "[layer]"
    EXPERT = "[expert]"


@dataclass
class StateMappingTemplate:
    """
    A
    """

    source_template_keys: str | Tuple[str, ...]
    dest_template_keys: str | Tuple[str, ...]

    source_key_per_placeholder: TemplatePlaceholder | None = None
    dest_key_per_placeholder: TemplatePlaceholder | None = None

    source_concat_dim: int = 0
    dims_permutation: Tuple[int, ...] | None = None
    unflatten_dim: Tuple[int, Tuple[TemplatePlaceholder | int, ...]] | None = None
    flatten_dims: Tuple[int, int] | None = None
    dest_chunk_dim: int = 0

    def __post_init__(self):
        if self.source_key_per_placeholder and isinstance(self.source_template_keys, tuple):
            raise ValueError(
                f"Having a key per {self.source_key_per_placeholder} is not supported with multiple template keys"
            )

        if self.dest_key_per_placeholder and isinstance(self.dest_template_keys, tuple):
            raise ValueError(
                f"Having a key per {self.dest_key_per_placeholder} is not supported with multiple template keys"
            )

    def _templates_to_keys(
        self,
        templates: str | Tuple[str, ...],
        placeholder_values: Dict[TemplatePlaceholder, Any],
        *,
        key_per_placeholder: TemplatePlaceholder | None = None,
        key_per_placeholder_values: List[Any] | None = None,
    ) -> Tuple[str, ...] | None:
        if key_per_placeholder:
            if key_per_placeholder_values is None:
                return None

            assert isinstance(templates, str)
            assert key_per_placeholder in templates
            assert key_per_placeholder_values is not None
            templates = tuple(
                templates.replace(key_per_placeholder, str(value))
                for value in key_per_placeholder_values
            )
        elif isinstance(templates, str):
            templates = (templates,)

        assert isinstance(templates, tuple)

        keys = []
        for template in templates:
            key = template
            for placeholder, value in placeholder_values.items():
                if placeholder in template and value is not None:
                    key = key.replace(placeholder, str(value))
                elif placeholder not in template and value is None:
                    pass
                else:
                    # If a placeholder is given a value but is not present,
                    # we treat the placeholder values as invalid.
                    # Similarly, if a placeholder is not given a value but is present,
                    # we treat the placeholder values as invalid.
                    return None

            keys.append(key)

        return tuple(keys)

    def to_mapping(
        self,
        placeholder_values: Dict[TemplatePlaceholder, int | None],
        placeholder_bounds: Dict[TemplatePlaceholder, int],
    ) -> Optional["StateMapping"]:
        required_placeholders: Set[TemplatePlaceholder | None] = set()
        if self.source_key_per_placeholder:
            required_placeholders.add(self.source_key_per_placeholder)
        if self.dest_key_per_placeholder:
            required_placeholders.add(self.dest_key_per_placeholder)
        if self.unflatten_dim:
            required_placeholders.update(
                [dim for dim in self.unflatten_dim[1] if isinstance(dim, TemplatePlaceholder)]
            )

        missing_required_placeholders = required_placeholders.difference(placeholder_bounds.keys())
        if missing_required_placeholders:
            return None

        source_keys = self._templates_to_keys(
            self.source_template_keys,
            placeholder_values,
            key_per_placeholder=self.source_key_per_placeholder,
            key_per_placeholder_values=list(
                range(placeholder_bounds[self.source_key_per_placeholder])
            )
            if self.source_key_per_placeholder
            and placeholder_values[self.source_key_per_placeholder] is None
            else None,
        )
        dest_keys = self._templates_to_keys(
            self.dest_template_keys,
            placeholder_values,
            key_per_placeholder=self.dest_key_per_placeholder,
            key_per_placeholder_values=list(
                range(placeholder_bounds[self.dest_key_per_placeholder])
            )
            if self.dest_key_per_placeholder
            and placeholder_values[self.dest_key_per_placeholder] is None
            else None,
        )

        if source_keys is None or dest_keys is None:
            return None

        unflatten_dim = None
        if self.unflatten_dim is not None:
            unflatten_dim_shape = tuple(
                placeholder_bounds[dim] if isinstance(dim, TemplatePlaceholder) else int(dim)
                for dim in self.unflatten_dim[1]
            )
            unflatten_dim = (self.unflatten_dim[0], unflatten_dim_shape)

        return StateMapping(
            source_keys,
            dest_keys,
            self.source_concat_dim,
            self.dims_permutation,
            unflatten_dim,
            self.flatten_dims,
            self.dest_chunk_dim,
        )


@dataclass
class StateMapping:
    source_keys: Tuple[str, ...]
    dest_keys: Tuple[str, ...]

    source_concat_dim: int = 0
    dims_permutation: Tuple[int, ...] | None = None
    unflatten_dim: Tuple[int, Tuple[int, ...]] | None = None
    flatten_dims: Tuple[int, int] | None = None
    dest_chunk_dim: int = 0


class StateConverter:
    def __init__(self, mapping_templates: List[StateMappingTemplate]) -> None:
        self.mapping_templates = mapping_templates

    def _fill_placeholders(
        self,
        mapping: StateMappingTemplate,
        placeholder_values: Dict[TemplatePlaceholder, int | None],
        placeholder_bounds: Dict[TemplatePlaceholder, int],
    ) -> StateMapping | None:
        return mapping.to_mapping(placeholder_values, placeholder_bounds)

    def _get_mappings(
        self, state_dict: Dict[str, Any], placeholder_bounds: Dict[TemplatePlaceholder, int]
    ) -> List[StateMapping]:
        # We consider all combinations of placeholders, including allowing each placeholder to not be set.
        # If a placeholder is set when not need, the combination will be treated as invalid
        # and so ignored.
        placeholder_value_combinations: List[Dict[TemplatePlaceholder, int | None]] = list(
            map(
                dict,
                itertools.product(
                    *[
                        [(placeholder, i) for i in range(bound)] + [(placeholder, None)]
                        for placeholder, bound in placeholder_bounds.items()
                    ]
                ),
            )
        )

        # Fill in the placeholders in the mapping templates
        state_mappings = [
            self._fill_placeholders(
                mapping_template,
                placeholder_value_combination,
                placeholder_bounds,
            )
            for mapping_template in self.mapping_templates
            for placeholder_value_combination in placeholder_value_combinations
        ]
        state_mappings = [mapping for mapping in state_mappings if mapping is not None]

        # Filter for mappings that are relevant to the given state dict
        state_keys = set(state_dict.keys())
        state_mappings = [
            mapping
            for mapping in state_mappings
            if mapping and all(k in state_keys for k in mapping.source_keys)
        ]

        return state_mappings

    def get_mappings(
        self, state_dict: Dict[str, Any], placeholder_bounds: Dict[TemplatePlaceholder, int]
    ) -> List[StateMapping]:
        return self._get_mappings(state_dict, placeholder_bounds)

    def convert(
        self, state_dict: Dict[str, Any], placeholder_bounds: Dict[TemplatePlaceholder, int]
    ) -> Dict[str, Any]:
        state_mappings = self._get_mappings(state_dict, placeholder_bounds)

        unused_original_keys = set(state_dict.keys())
        converted_state_dict = {}
        for mapping in state_mappings:
            original_keys = mapping.source_keys
            converted_keys = mapping.dest_keys
            if isinstance(state_dict[original_keys[0]], torch.Tensor):
                original_state = torch.cat(
                    [state_dict[key] for key in original_keys],
                    dim=mapping.source_concat_dim,
                )

                if mapping.unflatten_dim is not None:
                    original_state = original_state.unflatten(*mapping.unflatten_dim)
                if mapping.dims_permutation is not None:
                    original_state = original_state.permute(*mapping.dims_permutation)
                if mapping.flatten_dims is not None:
                    original_state = original_state.flatten(*mapping.flatten_dims)

                state_chunks = torch.chunk(
                    original_state, chunks=len(converted_keys), dim=mapping.dest_chunk_dim
                )
                for hf_key, state_chunk in zip(converted_keys, state_chunks):
                    converted_state_dict[hf_key] = state_chunk.contiguous()
            else:
                raise RuntimeError(
                    f"Attempting to map {len(original_keys)} non-tensor states to {len(converted_keys)} keys"
                )

            unused_original_keys -= set(original_keys)

        if len(unused_original_keys) > 0:
            raise RuntimeError(
                f"Some state keys were not converted: {sorted(unused_original_keys)}"
            )

        return converted_state_dict
