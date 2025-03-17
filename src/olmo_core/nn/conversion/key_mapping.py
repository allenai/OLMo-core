from dataclasses import dataclass
from typing import Tuple

from olmo_core.config import StrEnum


class TemplatePlaceholder(StrEnum):
    LAYER = "[layer]"
    EXPERT = "[expert]"


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
    unflatten_dim: Tuple[int, Tuple[TemplatePlaceholder | int, ...]] | None = None
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
            templates = tuple(
                templates.replace(TemplatePlaceholder.LAYER, str(i)) for i in range(n_blocks)
            )
        elif key_per_expert:
            assert isinstance(templates, str)
            templates = tuple(
                templates.replace(TemplatePlaceholder.EXPERT, str(i)) for i in range(n_experts)
            )
        elif isinstance(templates, str):
            templates = (templates,)

        assert isinstance(templates, tuple)

        return tuple(
            template.replace(TemplatePlaceholder.LAYER, str(i_block)).replace(
                TemplatePlaceholder.EXPERT, str(i_expert)
            )
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
                n_blocks
                if dim == TemplatePlaceholder.LAYER
                else n_experts
                if dim == TemplatePlaceholder.EXPERT
                else int(dim)
                for dim in self.unflatten_dim[1]
            )
            unflatten_dim = (self.unflatten_dim[0], unflatten_dim_shape)

        return TensorMapping(
            source_keys,
            dest_keys,
            self.source_concat_dim,
            self.dims_permutation,
            unflatten_dim,
            self.flatten_dims,
            self.dest_chunk_dim,
        )


@dataclass
class TensorMapping:
    source_keys: Tuple[str, ...]
    dest_keys: Tuple[str, ...]

    source_concat_dim: int = 0
    dims_permutation: Tuple[int, ...] | None = None
    unflatten_dim: Tuple[int, Tuple[int, ...]] | None = None
    flatten_dims: Tuple[int, int] | None = None
    dest_chunk_dim: int = 0
