import functools as ft
import hashlib
from typing import Optional

from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError

from .instance_source import Instance, InstanceSource
from .utils import build_global_indices, resolve_seed


class SlicedInstanceSource(InstanceSource):
    """
    An instance source that provides a slice of another instance source.
    """

    def __init__(
        self,
        source: InstanceSource,
        source_slice: slice,
        *,
        seed: Optional[int] = None,
        work_dir: PathOrStr,
    ):
        super().__init__(
            work_dir=work_dir,
            sequence_length=source.sequence_length,
            max_sequence_length=source.max_sequence_length,
            label=source.label,
        )
        self._source = source
        self._slice = source_slice
        self._seed = resolve_seed(seed)
        self._sliced_indices = build_global_indices(
            len(source),
            sequence_length=self.sequence_length,
            max_sequence_length=self.max_sequence_length,
            seed=self.seed,
        )[source_slice]
        if self._sliced_indices.size == 0:
            raise OLMoConfigurationError(
                f"{self.__class__.__name__} created with an empty slice ({source_slice}) from source "
                f"with {len(source):,d} instances."
            )

    @property
    def source(self) -> InstanceSource:
        return self._source

    @property
    def source_slice(self) -> slice:
        return self._slice

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @property
    def num_instances(self) -> int:
        chunk_size = self.max_sequence_length // self.sequence_length
        return chunk_size * (len(self._sliced_indices) // chunk_size)

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        chunk_size = self.max_sequence_length // self.sequence_length
        sha256_hash.update(
            (
                f"class={self.__class__.__name__},"
                f"{self.seed=},"
                f"{self.max_sequence_length=},"
                f"slice_start={self.source_slice.start // chunk_size if self.source_slice.start is not None else 0},"
                f"slice_stop={self.source_slice.stop // chunk_size if self.source_slice.stop is not None else -1},"
                f"slice_step={self.source_slice.step // chunk_size if self.source_slice.step is not None else 1},"
                f"source={self.source.fingerprint},"
            ).encode()
        )
        return sha256_hash.hexdigest()

    def __len__(self) -> int:
        return self.num_instances

    def __getitem__(self, idx: int) -> Instance:
        idx = int(self._sliced_indices[self.validate_index(idx)])
        return self.source[idx]

    def children(self):
        return [self.source]
