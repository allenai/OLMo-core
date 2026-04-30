import dataclasses
import functools as ft
import hashlib
import typing
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError

from ..tokenizer import TokenizerConfig
from ..types import NumpyUIntTypes
from ..utils import get_rng
from .instance_source import Instance, InstanceSource, InstanceSourceConfig
from .utils import SEED_NOT_SET, resolve_seed


@dataclass
class RandomInstanceSourceConfig(InstanceSourceConfig):
    """Config for :class:`RandomInstanceSource`."""

    tokenizer: TokenizerConfig
    sequence_length: int
    avg_document_length: int
    seed: int = dataclasses.field(default_factory=lambda: resolve_seed(SEED_NOT_SET))
    num_instances: Optional[int] = None
    num_tokens: Optional[int] = None
    max_sequence_length: Optional[int] = None
    label: Optional[str] = None

    def build(self, work_dir: PathOrStr) -> "RandomInstanceSource":
        return RandomInstanceSource(
            work_dir=work_dir,
            **self.as_dict(recurse=False),
        )


class RandomInstanceSource(InstanceSource):
    """
    An instance source that generates random instances. Useful for benchmarking.
    """

    Config = RandomInstanceSourceConfig

    DISPLAY_ICON = "\uedec"

    def __init__(
        self,
        *,
        tokenizer: TokenizerConfig,
        sequence_length: int,
        avg_document_length: int,
        seed: int = SEED_NOT_SET,
        num_instances: Optional[int] = None,
        num_tokens: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
        label: Optional[str] = None,
        work_dir: PathOrStr,
    ):
        if (num_tokens is None) == (num_instances is None):
            raise OLMoConfigurationError(
                "Either num_tokens or num_instances must be set, but not both."
            )
        elif num_tokens is not None:
            assert num_tokens > 0
            num_instances = num_tokens // sequence_length
        elif num_instances is not None:
            assert num_instances > 0
            num_tokens = num_instances * sequence_length

        assert num_tokens is not None

        super().__init__(
            work_dir=work_dir,
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
            label=label,
        )
        self._num_tokens = self.max_sequence_length * (num_tokens // self.max_sequence_length)
        self._tokenizer = tokenizer
        self._avg_document_length = avg_document_length
        seed = resolve_seed(seed)
        assert seed is not None
        self._seed = seed

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._tokenizer.bos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def avg_document_length(self) -> int:
        return self._avg_document_length

    @ft.cached_property
    def non_special_token(self) -> int:
        for token_id in range(self.vocab_size):
            if token_id not in (self.eos_token_id, self.bos_token_id, self.pad_token_id):
                return token_id
        raise RuntimeError("No non-special token found in the vocabulary.")

    @property
    def dtype(self) -> NumpyUIntTypes:
        for dtype in (np.uint8, np.uint16, np.uint32):
            if np.iinfo(dtype).max >= self.vocab_size:
                return dtype
        return np.uint64

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            (
                f"class={self.__class__.__name__},"
                f"num_tokens={self.num_tokens},"
                f"seed={self.seed},"
                f"max_sequence_length={self.max_sequence_length},"
                f"eos_token_id={self.eos_token_id},"
                f"bos_token_id={self.bos_token_id},"
                f"pad_token_id={self.pad_token_id},"
                f"vocab_size={self.vocab_size},"
            ).encode()
        )
        return sha256_hash.hexdigest()

    def __len__(self) -> int:
        return self._num_tokens // self.sequence_length

    def __getitem__(self, idx: int) -> Instance:
        idx = self.validate_index(idx)

        if self.sequence_length < self.max_sequence_length:
            base_idx = idx // (self.max_sequence_length // self.sequence_length)
        else:
            base_idx = idx

        seed = self.seed + base_idx
        rng = get_rng(seed)

        # Generate random tokens.
        tokens = rng.integers(0, self.vocab_size, self.max_sequence_length, dtype=self.dtype)

        # Replace special tokens with non-special tokens.
        tokens[tokens == self.eos_token_id] = self.non_special_token
        tokens[tokens == self.pad_token_id] = self.non_special_token
        if self.bos_token_id is not None:
            tokens[tokens == self.bos_token_id] = self.non_special_token

        # Inject random document boundaries.
        num_docs = max(
            1, round(rng.integers(-3, 3) + self.max_sequence_length / self.avg_document_length)
        )
        tokens[-1] = self.eos_token_id
        if self.bos_token_id is not None:
            tokens[0] = self.bos_token_id
        if num_docs > 1:
            buffer = 1 if self.bos_token_id is None else 2
            doc_boundaries = (
                buffer + rng.permutation(self.max_sequence_length - 2 * buffer - 1)[: num_docs - 1]
            )
            tokens[doc_boundaries] = self.eos_token_id
            if self.bos_token_id is not None:
                tokens[doc_boundaries + 1] = self.bos_token_id

        # Pull out sub-sequence if needed.
        if self.sequence_length < self.max_sequence_length:
            start_offset = (
                idx % (self.max_sequence_length // self.sequence_length)
            ) * self.sequence_length
            tokens = tokens[start_offset : start_offset + self.sequence_length]

        return {"input_ids": typing.cast(Sequence[int], tokens)}

    def children(self):
        return []
