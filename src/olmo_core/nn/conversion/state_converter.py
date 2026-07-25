import itertools
from dataclasses import dataclass
from typing import Any

import torch

from olmo_core.doc_utils import beta_feature
from olmo_core.nn.conversion.state_mapping import (
    StateMapping,
    StateMappingTemplate,
    StateType,
    TemplatePlaceholder,
)


@beta_feature
@dataclass
class StateConverter:
    """
    A class for converting state from one format to another format (e.g. OLMo Core to HF).
    """

    mapping_templates: list[StateMappingTemplate]

    def _fill_placeholders(
        self,
        mapping: StateMappingTemplate,
        placeholder_values: dict[TemplatePlaceholder, int | None],
        placeholder_bounds: dict[TemplatePlaceholder, int],
    ) -> StateMapping | None:
        return mapping.to_mapping(placeholder_values, placeholder_bounds)

    def _get_mappings(
        self,
        state_dict: dict[str, Any],
        placeholder_bounds: dict[TemplatePlaceholder, int],
        state_type: StateType = StateType.weight,
    ) -> list[StateMapping]:
        # We consider all combinations of placeholders, including allowing each placeholder to not be set.
        # If a placeholder is set when not need, the combination will be treated as invalid
        # and so ignored.
        placeholder_value_combinations: list[dict[TemplatePlaceholder, int | None]] = list(
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
            if mapping_template.state_type == state_type
            for placeholder_value_combination in placeholder_value_combinations
        ]

        # Filter for mappings that are relevant to the given state dict
        state_keys = set(state_dict.keys())
        relevant_state_mappings = [
            mapping
            for mapping in state_mappings
            if mapping is not None and all(k in state_keys for k in mapping.source_keys)
        ]

        return relevant_state_mappings

    def get_mappings(
        self,
        state_dict: dict[str, Any],
        placeholder_bounds: dict[TemplatePlaceholder, int],
        state_type: StateType = StateType.weight,
    ) -> list[StateMapping]:
        """
        Gets the state mapping from the given state dict to the converted format,
        without performing conversion.

        :param state_dict: The state dictionary in unconverted format.
        :param placeholder_bounds: Upper bound values for any relevant placeholders
            (e.g. for ``TemplatePlaceholder.EXPERT``, the number of experts).
        :param state_type: The type of state this state dict corresponds to. Defaults to ``StateType.weight``.
        """

        return self._get_mappings(state_dict, placeholder_bounds, state_type=state_type)

    def convert(
        self,
        state_dict: dict[str, Any],
        placeholder_bounds: dict[TemplatePlaceholder, int],
        state_type: StateType = StateType.weight,
    ) -> dict[str, Any]:
        """
        Converts a state dict to another format. This currently only supports tensor values.

        :param state_dict: The state dictionary to convert.
        :param placeholder_bounds: Upper bound values for any relevant placeholders
            (e.g. for ``TemplatePlaceholder.EXPERT``, the number of experts).
        :param state_type: The type of state this state dict corresponds to. Defaults to ``StateType.weight``.
        """

        state_mappings = self._get_mappings(state_dict, placeholder_bounds, state_type=state_type)

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
                raise RuntimeError(  # noqa: TRY004
                    f"Attempting to map {len(original_keys)} non-tensor states to {len(converted_keys)} keys"
                )

            unused_original_keys -= set(original_keys)

        if len(unused_original_keys) > 0:
            raise RuntimeError(
                f"Some state keys were not converted: {sorted(unused_original_keys)}"
            )

        return converted_state_dict
