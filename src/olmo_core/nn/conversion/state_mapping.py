from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

from olmo_core.config import StrEnum


class TemplatePlaceholder(StrEnum):
    """
    A placeholder that can be used in the templates of :class:`StateMappingTemplate`.
    """

    LAYER = "[layer]"
    """"""

    EXPERT = "[expert]"
    """"""

    LOCAL_ENCODER_LAYER = "[local_encoder_layer]"
    """"""

    LOCAL_DECODER_LAYER = "[local_decoder_layer]"
    """"""


class StateType(StrEnum):
    """
    The category the state being converted corresponds to.
    """

    weight = "weight"
    """
    The state being converted corresponds to a weight. This is useful for converting between checkpoints,
    where the state is the weight itself.
    """

    module = "module"
    """
    The state being converted corresponds to a modules. This can be useful for comparing activations between
    different implementations of the same model, where the states are the activations of submodules.
    """


@dataclass
class StateMappingTemplate:
    """
    The template for a mapping state from one format to another format (e.g. OLMo Core to HF).
    These mappings are 'templates' since they support keys and other metadata having placeholders
    for information like the layer number or number of MoE experts. This class can be converted
    to a :class:`StateMapping` by providing the placeholder information.

    The most standard mapping is a one-to-one state mapping, which corresponds to a single
    string entry for both :data:`source_template_keys` and :data:`dest_template_keys`. The class also supports
    more complicated mappings, like many-to-many mappings or mappings that also require further
    manipulations of state like permuting dimensions.
    """

    source_template_keys: str | Tuple[str, ...]
    """
    The key or keys of the state(s) being mapping from.
    """
    dest_template_keys: str | Tuple[str, ...]
    """
    The key or keys of the state(s) being mapping to.
    """

    state_type: StateType = StateType.weight

    source_key_per_placeholder: TemplatePlaceholder | None = None
    """
    A placeholder in :data:`source_template_keys` for which this mapping should map all valid placeholder
    values, rather than 1 specific value. For example, this enables mapping states from all experts
    (using ``TemplatePlaceholder.EXPERT``) to a single state.

    When provided, :data:`source_template_keys` must be a string.
    """
    dest_key_per_placeholder: TemplatePlaceholder | None = None
    """
    A placeholder in :data:`dest_template_keys` for which this mapping should map all valid placeholder
    values, rather than 1 specific value. For example, this enables mapping from a single state to
    states from all experts (using ``TemplatePlaceholder.EXPERT``).

    When provided, :data:`dest_template_keys` must be a string.
    """

    source_concat_dim: int = 0
    """
    When many states are being mapping from, this specifies the dimension on which to combine them.
    """
    unflatten_dim: Tuple[int, Tuple[TemplatePlaceholder | int, ...]] | None = None
    """
    This specifies that the given dimension (``unflatten_dim[0]``) should be unflattened using the shape
    given in ``unflatten_dim[1]``. A placeholder can be given instead of a number, to represent its
    corresponding upper bound (e.g. ``TemplatePlaceholder.EXPERT`` represents the number of experts). 
    """
    dims_permutation: Tuple[int, ...] | None = None
    """
    This specifies the permutation that should be applied to the dimensions of the state after any
    unflattening from :data:`unflatten_dim` has occurred.
    """
    flatten_dims: Tuple[int, int] | None = None
    """
    This specifies that all the dimensions between the 2 given dimensions (inclusive) should be flattened,
    after any permutations from :data:`dims_permutation` have been applied.
    """
    dest_chunk_dim: int = 0
    """
    When many states are being mapping to, this specifies the dimension on which to (evenly) chunk them.
    """

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
        placeholder_values: Dict[TemplatePlaceholder, Any],
        placeholder_bounds: Dict[TemplatePlaceholder, int],
        *,
        source: bool,
    ) -> Tuple[str, ...] | None:
        if source:
            templates = self.source_template_keys
            key_per_placeholder = self.source_key_per_placeholder
        else:
            templates = self.dest_template_keys
            key_per_placeholder = self.dest_key_per_placeholder

        if key_per_placeholder:
            if not isinstance(templates, str):
                raise ValueError(
                    "Invalid template; template must be a string when expanding a placeholder"
                )
            template = templates

            if key_per_placeholder not in template:
                raise ValueError(
                    f"Invalid template; placeholder {key_per_placeholder} is being expanded but is not present in template {template}"
                )

            if key_per_placeholder not in placeholder_bounds:
                raise ValueError(
                    f"Invalid bounds; placeholder {key_per_placeholder} does not have a bound"
                )
            key_per_placeholder_values = list(range(placeholder_bounds[key_per_placeholder]))

            templates = tuple(
                template.replace(key_per_placeholder, str(value))
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

        for key in keys:
            if any(placeholder in key for placeholder in TemplatePlaceholder):
                # If a placeholder has not been filled, its key was not provided.
                return None

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
            # This may be, say, an MoE mapping for which we do not have any expert values.
            # This is ok; we simply discard this mapping.
            return None

        source_keys = self._templates_to_keys(
            placeholder_values,
            placeholder_bounds,
            source=True,
        )
        dest_keys = self._templates_to_keys(
            placeholder_values,
            placeholder_bounds,
            source=False,
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
            state_type=self.state_type,
            source_concat_dim=self.source_concat_dim,
            unflatten_dim=unflatten_dim,
            dims_permutation=self.dims_permutation,
            flatten_dims=self.flatten_dims,
            dest_chunk_dim=self.dest_chunk_dim,
        )


@dataclass
class StateMapping:
    """
    A mapping from state from one format to another format (e.g. OLMo Core to HF).

    The most standard mapping is a one-to-one state mapping, which corresponds to a single
    string entry for both :data:`source_keys` and :data:`dest_keys`. The class also supports
    more complicated mappings, like many-to-many mappings or mappings that also require further
    manipulations of state like permuting dimensions.
    """

    source_keys: Tuple[str, ...]
    """
    The key(s) of the state(s) being mapping from.
    """

    dest_keys: Tuple[str, ...]
    """
    The key or keys of the state(s) being mapping to.
    """

    state_type: StateType = StateType.weight

    source_concat_dim: int = 0
    """
    When many states are being mapping from, this specifies the dimension on which to combine them.
    """
    unflatten_dim: Tuple[int, Tuple[int, ...]] | None = None
    """
    This specifies that the given dimension (``unflatten_dim[0]``) should be unflattened using the shape
    given in ``unflatten_dim[1]``.
    """
    dims_permutation: Tuple[int, ...] | None = None
    """
    This specifies the permutation that should be applied to the dimensions of the state after any
    unflattening from :data:`unflatten_dim` has occurred.
    """
    flatten_dims: Tuple[int, int] | None = None
    """
    This specifies that all the dimensions between the 2 given dimensions (inclusive) should be flattened,
    after any permutations from :data:`dims_permutation` have been applied.
    """
    dest_chunk_dim: int = 0
    """
    When many states are being mapping to, this specifies the dimension on which to (evenly) chunk them.
    """
