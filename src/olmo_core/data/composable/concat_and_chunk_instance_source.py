import functools as ft
import hashlib
from typing import Optional, Sequence

from olmo_core.aliases import PathOrStr

from .instance_source import Instance, InstanceSource
from .token_source import TokenSource


class ConcatAndChunkInstanceSource(InstanceSource):
    """
    The basic instance source that simply chunks up token sources without regard for
    document boundaries, just like the :class:`~olmo_core.data.numpy_dataset.NumpyFSLDataset`.
    """

    def __init__(
        self,
        *,
        sources: Sequence[TokenSource],
        sequence_length: int,
        work_dir: PathOrStr,
        max_sequence_length: Optional[int] = None,
    ):
        super().__init__(
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
            work_dir=work_dir,
        )
        self._sources = tuple(sources)

    @property
    def sources(self) -> tuple[TokenSource]:
        return self._sources

    @ft.cached_property
    def num_instances(self) -> int:
        return sum(
            (
                (source.num_tokens // self.max_sequence_length)
                * (self.max_sequence_length // self.sequence_length)
                for source in self.sources
            )
        )

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        # NOTE: counterintuitively, we don't take 'self.sequence_length' into account when computing
        # the fingerprint because only 'self.max_sequence_length' will affect data order, and this source
        # allows for changing the sequence length in the middle of an epoch.
        sha256_hash.update(
            (f"class={self.__class__.__name__},{self.max_sequence_length=},").encode()
        )
        for source in self.sources:
            sha256_hash.update(f"source={source.fingerprint},".encode())
        return sha256_hash.hexdigest()

    def __len__(self) -> int:
        return self.num_instances

    def __getitem__(self, idx: int) -> Instance:
        if idx < 0:
            idx = self.num_instances + idx
        assert 0 <= idx < self.num_instances
        source_start_offset = 0
        for source in self.sources:
            source_end_offset = source_start_offset + (
                source.num_tokens // self.max_sequence_length
            ) * (self.max_sequence_length // self.sequence_length)

            if source_start_offset <= idx < source_end_offset:
                return source.get_token_range(
                    (idx - source_start_offset) * self.sequence_length, self.sequence_length
                )

            source_start_offset = source_end_offset
        raise IndexError(f"Index {idx} out of range for {self.num_instances} instances.")
