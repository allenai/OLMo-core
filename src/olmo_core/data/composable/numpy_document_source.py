import functools as ft
import hashlib
import logging
import random
import typing
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

import olmo_core.distributed.utils as dist_utils
import olmo_core.io as io
from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import log_once

from ..mixes import DataMix, DataMixBase
from ..tokenizer import TokenizerConfig
from ..types import NumpyDatasetDType, NumpyUIntTypes
from ..utils import chunked, iter_document_indices, load_array_slice
from .token_source import DocumentSource, DocumentSourceConfig, TokenRange
from .utils import path_map, resolve_seed

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class NumpyDocumentSourceConfigBase(DocumentSourceConfig):
    """Base config class for :class:`NumpyDocumentSourceConfig` and :class:`NumpyDocumentSourceMixConfig`."""

    tokenizer: TokenizerConfig
    """The config of the tokenizer that was used to tokenize the source files."""
    dtype: Optional[NumpyDatasetDType] = None
    """The numpy datatype of the token ID arrays in the source paths."""
    source_permutation_seed: Optional[int] = None
    """Used to shuffle the source files before grouping/building the document sources."""
    source_group_size: int = 1
    """The number of npy source files to group together into a single source."""
    label: Optional[str] = None
    """An optional to assign for logging and debugging."""

    def __post_init__(self):
        if self.source_group_size < -1 or self.source_group_size == 0:
            raise OLMoConfigurationError("'source_group_size' must be -1 or a positive integer.")
        self.source_permutation_seed = resolve_seed(self.source_permutation_seed)

    def get_dtype(self) -> NumpyUIntTypes:
        if self.dtype is not None:
            return NumpyDatasetDType(self.dtype).as_np_dtype()

        for dtype in (
            NumpyDatasetDType.uint8,
            NumpyDatasetDType.uint16,
            NumpyDatasetDType.uint32,
            NumpyDatasetDType.uint64,
        ):
            if (self.tokenizer.vocab_size - 1) <= np.iinfo(dtype.as_np_dtype()).max:
                log_once(log, f"Assuming dtype '{dtype}' based on vocab size")
                return dtype.as_np_dtype()

        raise ValueError("vocab size too big!")


@dataclass(kw_only=True)
class NumpyDocumentSourceConfig(NumpyDocumentSourceConfigBase):
    """Config class for building one or more :class:`NumpyDocumentSource` directly from source paths."""

    source_paths: List[str]
    """The paths/URLs to the numpy token ID arrays."""
    label_mask_paths: Optional[List[str]] = None
    """The paths/URLs to numpy bool files indicating which tokens should be masked."""
    expand_glob: Optional[bool] = None
    """If true, treat source/label paths as glob patterns and expand them when building the sources."""

    @classmethod
    def from_source_groups(
        cls,
        source_path_groups: Dict[str, List[PathOrStr]],
        *,
        tokenizer: TokenizerConfig,
        label_mask_path_groups: Optional[Dict[str, List[PathOrStr]]] = None,
        expand_glob: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, "NumpyDocumentSourceConfig"]:
        """
        A more efficient way to create multiple configs from groups of source paths.
        This will use a thread pool to expand all globs concurrently, which can be substantially
        faster especially when some of the globs point to cloud storage URLs.

        :param source_path_groups: Groups of source paths to use. Each group will be put into its own config
            with the corresponding label.
        :param tokenizer: The tokenizer config to use.
        :param label_mask_path_groups: Optional groups of label mask paths to use. Each group should
            correspond to the group in ``source_paths`` at the same key.
        """
        if label_mask_path_groups is not None:
            assert len(source_path_groups) == len(label_mask_path_groups)
            assert set(source_path_groups.keys()) == set(label_mask_path_groups.keys())
            for k in source_path_groups.keys():
                assert len(source_path_groups[k]) == len(label_mask_path_groups[k])

        if expand_glob is None:
            expand_glob = any(
                ["*" in str(p) for group in source_path_groups.values() for p in group]
            )

        source_paths_to_use: Dict[str, List[str]] = {}
        mask_paths_to_use: Optional[Dict[str, List[str]]] = None
        if expand_glob:
            _, src_pattern_to_expanded = cls._expand_globs(
                [p for group in source_path_groups.values() for p in group]
            )
            for k, group in source_path_groups.items():
                expanded_group = []
                for p in group:
                    expanded_group.extend(src_pattern_to_expanded[p])
                source_paths_to_use[k] = expanded_group

            if label_mask_path_groups is not None:
                mask_paths_to_use = {}
                _, mask_pattern_to_expanded = cls._expand_globs(
                    [p for group in label_mask_path_groups.values() for p in group]
                )
                for k, group in label_mask_path_groups.items():
                    expanded_group = []
                    for p in group:
                        expanded_group.extend(mask_pattern_to_expanded[p])
                    mask_paths_to_use[k] = expanded_group
        else:
            source_paths_to_use = {
                k: [str(p) for p in group] for k, group in source_path_groups.items()
            }
            if label_mask_path_groups is not None:
                mask_paths_to_use = {
                    k: [str(p) for p in group] for k, group in label_mask_path_groups.items()
                }

        configs: Dict[str, NumpyDocumentSourceConfig] = {}
        for k, src_group in source_paths_to_use.items():
            configs[k] = cls(
                source_paths=src_group,
                label_mask_paths=mask_paths_to_use[k] if mask_paths_to_use is not None else None,
                tokenizer=tokenizer,
                expand_glob=False,
                label=k,
                **kwargs,
            )
        return configs

    def build(self, work_dir: PathOrStr) -> List["NumpyDocumentSource"]:  # type: ignore[override]
        """
        Build the sources.

        .. note::
            The number of sources returned depends on the length of :data:`source_paths` and the
            value of :data:`~NumpyDocumentSourceConfigBase.source_group_size`.
        """
        dtype = self.get_dtype()
        label = self.label

        if label is None:
            if len(self.source_paths) == 1:
                label = self.source_paths[0]
            else:
                label = "various paths"

        expand_glob = self.expand_glob
        if self.expand_glob is None:
            expand_glob = any(["*" in p for p in self.source_paths])

        if expand_glob:
            source_paths, _ = self._expand_globs(self.source_paths)
            mask_paths = (
                None
                if self.label_mask_paths is None
                else self._expand_globs(self.label_mask_paths)[0]
            )
        else:
            source_paths = self.source_paths
            mask_paths = self.label_mask_paths

        if self.source_permutation_seed is not None:
            source_order = list(range(len(self.source_paths)))
            rng = random.Random(self.source_permutation_seed)
            rng.shuffle(source_order)
            source_paths = [source_paths[i] for i in source_order]
            mask_paths = None if mask_paths is None else [mask_paths[i] for i in source_order]

        # NOTE: we always create a single main source first, then split it up if needed.
        # This way is more efficient because we can query for the size of all source files concurrently.
        main_source = NumpyDocumentSource(
            source_paths=source_paths,
            label_mask_paths=mask_paths,
            tokenizer=self.tokenizer,
            dtype=dtype,
            work_dir=work_dir,
            label=label,
        )

        if self.source_group_size > 0:
            return main_source.split_by_source(self.source_group_size)
        else:
            return [main_source]

    @classmethod
    def _expand_globs(
        cls, patterns: Sequence[PathOrStr]
    ) -> Tuple[List[str], Dict[PathOrStr, List[str]]]:
        log.info("Expanding globs...")
        results: List[List[str]] = []
        if dist_utils.get_rank() == 0:
            results = path_map(cls._expand_glob, patterns)
        else:
            results = []
        results = dist_utils.broadcast_object(results)

        expanded: List[str] = []
        pattern_to_expanded: Dict[PathOrStr, List[str]] = {}
        for pattern, matches in zip(patterns, results):
            if not matches:
                raise FileNotFoundError(pattern)
            if len(matches) <= 5:
                summary = "\n".join([f"- '{match}'" for match in matches])
            else:
                summary = "\n".join(
                    [
                        f"- '{matches[0]}'",
                        f"- '{matches[1]}'",
                        "⋮",
                        f"- '{matches[-2]}'",
                        f"- '{matches[-1]}'",
                    ]
                )
            log.info(f"Expanded '{pattern}' into {len(matches):,d} paths:\n{summary}")
            expanded.extend(matches)
            pattern_to_expanded[pattern] = matches

        return expanded, pattern_to_expanded

    @classmethod
    def _expand_glob(cls, pattern: PathOrStr) -> List[str]:
        pattern = str(pattern)
        if "*" in pattern:
            return sorted(io.glob_directory(pattern))
        else:
            return [pattern]


@dataclass(kw_only=True)
class NumpyDocumentSourceMixConfig(NumpyDocumentSourceConfigBase):
    """Config class for building one or more :class:`NumpyDocumentSource` from a predefined source mix."""

    mix: Union[str, DataMixBase]
    """The name of a data mix (e.g. ``"dolma17"``)."""
    mix_base_dir: str
    """The base directory of the data mix."""

    def build(self, work_dir: PathOrStr) -> List["NumpyDocumentSource"]:  # type: ignore[override]
        """
        Build the sources.

        .. note::
            The number of sources returned depends on the number of paths in the mix and the
            value of :data:`~NumpyDocumentSourceConfigBase.source_group_size`.
        """
        if self.tokenizer.identifier is None:
            raise OLMoConfigurationError(
                "Missing tokenizer identifier required to construct data mix"
            )
        mix = self.mix
        if not isinstance(mix, DataMixBase):
            mix = DataMix(mix)
        source_paths, _ = mix.build(self.mix_base_dir, self.tokenizer.identifier)
        kwargs = self.as_dict(recurse=False, exclude={"mix", "mix_base_dir", "label"})
        return NumpyDocumentSourceConfig(
            source_paths=source_paths, label=self.label or self.mix, **kwargs
        ).build(work_dir=work_dir)


class NumpyDocumentSource(DocumentSource):
    """
    A :class:`DocumentSource` that reads tokens from one or more tokenized numpy source files.

    .. important::
        There's some overhead when instantiating this class because it needs to query the sizes of
        all the source files. If you want to create multiple sources from the same set of files,
        consider first creating a single source and then splitting it up using :meth:`split_by_source()`,
        which will be much more efficient than creating multiple sources directly since the sizes
        of the source files will only need to be queried once and will be done so concurrently with a
        thread pool.

    :param source_paths: The paths/URLs to the numpy token ID arrays.
    :param dtype: The numpy datatype of the token ID arrays in the source paths.
    :param tokenizer: The config of the tokenizer that was used to tokenize the source files.
    :param label_mask_paths: The paths/URLs to numpy bool files indicating which tokens should be masked.
    """

    Config = NumpyDocumentSourceConfig

    MixConfig = NumpyDocumentSourceMixConfig

    def __init__(
        self,
        *,
        source_paths: Sequence[PathOrStr],
        dtype: NumpyUIntTypes,
        work_dir: PathOrStr,
        tokenizer: TokenizerConfig,
        label_mask_paths: Optional[Sequence[PathOrStr]] = None,
        label: Optional[str] = None,
        _source_sizes: Optional[Sequence[int]] = None,
        _label_mask_sizes: Optional[Sequence[int]] = None,
    ):
        super().__init__(work_dir=work_dir, label=label)

        if not source_paths:
            raise OLMoConfigurationError("'source_paths' must contain at least one path.")

        if label_mask_paths is not None and len(label_mask_paths) != len(source_paths):
            raise OLMoConfigurationError(
                "'label_mask_paths' should have the same length as 'source_paths'."
            )

        self._source_paths = tuple((io.normalize_path(p) for p in source_paths))
        self._label_mask_paths = (
            None
            if label_mask_paths is None
            else tuple((io.normalize_path(p) for p in label_mask_paths))
        )
        self._dtype = dtype
        self._tokenizer = tokenizer

        source_sizes: Sequence[int]
        if _source_sizes is not None:
            source_sizes = tuple(_source_sizes)
        else:
            if self.rank == 0:
                item_size = self.dtype(0).itemsize
                source_sizes = path_map(
                    lambda p: io.get_file_size(p) // item_size, self.source_paths
                )
            else:
                source_sizes = []
            source_sizes = dist_utils.broadcast_object(source_sizes)
        assert len(source_sizes) == len(self.source_paths)
        self._source_sizes = tuple(source_sizes)

        self._label_mask_sizes: Optional[Tuple[int, ...]] = None
        if self.label_mask_paths is not None:
            label_mask_sizes: Sequence[int]
            if _label_mask_sizes is not None:
                label_mask_sizes = tuple(_label_mask_sizes)
            else:
                if self.rank == 0:
                    item_size = np.bool_(0).itemsize
                    label_mask_sizes = path_map(
                        lambda p: io.get_file_size(p) // item_size, self.label_mask_paths
                    )
                else:
                    label_mask_sizes = []
                label_mask_sizes = dist_utils.broadcast_object(label_mask_sizes)

            assert len(label_mask_sizes) == len(self.label_mask_paths)
            self._label_mask_sizes = tuple(label_mask_sizes)

            for label_path, label_mask_size, source_path, source_size in zip(
                self.label_mask_paths, label_mask_sizes, self.source_paths, self.source_sizes
            ):
                if label_mask_size != source_size:
                    raise OLMoConfigurationError(
                        "Each file in 'label_mask_paths' should have the same number of items as the corresponding file in 'source_paths', "
                        f"but found {label_mask_size:,d} in '{label_path}' vs {source_size:,d} in '{source_path}'.",
                    )

    @property
    def source_paths(self) -> Tuple[str, ...]:
        return self._source_paths

    @property
    def source_sizes(self) -> Tuple[int, ...]:
        return self._source_sizes

    @property
    def label_mask_paths(self) -> Optional[Tuple[str, ...]]:
        return self._label_mask_paths

    @property
    def label_mask_sizes(self) -> Optional[Tuple[int, ...]]:
        return self._label_mask_sizes

    @property
    def dtype(self) -> NumpyUIntTypes:
        return self._dtype

    @property
    def tokenizer(self) -> TokenizerConfig:
        return self._tokenizer

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.tokenizer.bos_token_id

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            (
                f"class={self.__class__.__name__},"
                f"{self.dtype=},"
                f"{self.eos_token_id=},"
                f"{self.bos_token_id=},"
            ).encode()
        )
        # NOTE: it's too expensive to hash the contents of the source files, so we take a shortcut
        # by hashing their paths and sizes instead. This should be sufficient to detect changes 99.99% of the time.
        for path, size in zip(self.source_paths, self.source_sizes):
            sha256_hash.update(f"{path=},{size=},".encode())
        if self.label_mask_paths is not None:
            for label_path, size in zip(self.label_mask_paths, self.source_sizes):
                sha256_hash.update(f"{label_path=},{size=},".encode())
        return sha256_hash.hexdigest()

    @ft.cached_property
    def num_tokens(self) -> int:
        return sum(self.source_sizes)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.source_paths}"

    def split_by_source(self, group_size: int = 1) -> List["NumpyDocumentSource"]:
        """
        Split the source up into multiple smaller sources from groups of source files.
        """
        assert group_size >= 1
        source_paths_groups = chunked(self.source_paths, group_size)
        source_size_groups = chunked(self.source_sizes, group_size)
        label_mask_paths_groups = (
            chunked(self.label_mask_paths, group_size)
            if self.label_mask_paths is not None
            else [None for _ in chunked(self.source_paths, group_size)]  # type: ignore[misc]
        )
        label_mask_size_groups = (
            chunked(self.label_mask_sizes, group_size)
            if self.label_mask_sizes is not None
            else [None for _ in chunked(self.source_sizes, group_size)]  # type: ignore[misc]
        )
        return [
            self.__class__(
                source_paths=source_paths,
                dtype=self.dtype,
                work_dir=self.work_dir,
                tokenizer=self.tokenizer,
                label_mask_paths=label_mask_paths,
                label=self.label,
                _source_sizes=source_sizes,
                _label_mask_sizes=label_mask_sizes,
            )
            for source_paths, label_mask_paths, source_sizes, label_mask_sizes in zip(
                source_paths_groups,
                label_mask_paths_groups,
                source_size_groups,
                label_mask_size_groups,
            )
        ]

    def get_token_range(self, start_idx: int, end_idx: int) -> TokenRange:
        start_idx, end_idx = self.validate_indices(start_idx, end_idx)

        token_chunks: List[np.ndarray] = []
        mask_chunks: List[np.ndarray] = []
        source_start_offset = 0
        for i, (source_path, source_size) in enumerate(zip(self.source_paths, self.source_sizes)):
            source_end_offset = source_start_offset + source_size

            if source_start_offset <= start_idx < source_end_offset:
                token_chunk = load_array_slice(
                    source_path,
                    start_idx - source_start_offset,
                    min(end_idx - source_start_offset, source_size),
                    self.dtype,
                )
                token_chunks.append(token_chunk)

                if self.label_mask_paths is not None:
                    mask_path = self.label_mask_paths[i]
                    mask_chunk = load_array_slice(
                        mask_path,
                        start_idx - source_start_offset,
                        min(end_idx - source_start_offset, source_size),
                        np.bool_,
                    )
                    mask_chunks.append(mask_chunk)

                if end_idx - source_start_offset <= source_size:
                    break
                else:
                    start_idx = source_end_offset

            source_start_offset = source_end_offset
        else:
            raise IndexError(f"Failed to find tokens in range {start_idx}->{end_idx}.")

        input_ids = np.concatenate(token_chunks)
        out: TokenRange = {"input_ids": typing.cast(Sequence[int], input_ids)}
        if mask_chunks:
            out["label_mask"] = typing.cast(Sequence[bool], np.concatenate(mask_chunks))

        return out

    def get_document_offsets(self) -> Iterable[tuple[int, int]]:
        start_offset = 0
        for source_path, source_size in zip(self.source_paths, self.source_sizes):
            last_doc_end = 0
            for doc_start, doc_end in iter_document_indices(
                source_path,
                eos_token_id=self.eos_token_id,
                bos_token_id=self.bos_token_id,
                dtype=self.dtype,
            ):
                assert doc_start == last_doc_end  # API assumes consecutive documents
                yield doc_start + start_offset, doc_end + start_offset
                last_doc_end = doc_end

            # To avoid unexpected results, we ALWAYS treat the end of a source file as the end of
            # a document, even if it doesn't end with an EOS token ID. This *should* always be the case
            # anyway, but just to be careful.
            if last_doc_end != source_size:
                yield last_doc_end + start_offset, source_size + start_offset

            start_offset += source_size

    def children(self):
        return []
