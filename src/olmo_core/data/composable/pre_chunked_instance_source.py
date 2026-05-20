import functools as ft
import hashlib
import logging
import re
import typing
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

import olmo_core.distributed.utils as dist_utils
import olmo_core.io as io
from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError

from ..types import NumpyDatasetDType, NumpyUIntTypes
from ..utils import load_array_slice
from .instance_source import Instance, InstanceSource, InstanceSourceConfig
from .utils import path_map

log = logging.getLogger(__name__)


def _extract_prefix(path: str, suffix: str) -> str:
    """Extract the prefix from a path by removing the known suffix (e.g. '-tokens.npy')."""
    basename = path.rsplit("/", 1)[-1] if "/" in path else path
    if not basename.endswith(suffix):
        raise OLMoConfigurationError(f"Expected path ending with '{suffix}', got '{path}'")
    return basename[: -len(suffix)]


def _match_parallel_paths(
    token_paths: Sequence[str],
    pos_ids_paths: Sequence[str],
    vis_limit_paths: Sequence[str],
    label_mask_paths: Sequence[str] = (),
) -> List[Tuple[str, str, str, Optional[str]]]:
    """
    Match parallel file groups by extracting shared prefixes.

    Returns a sorted list of (token_path, pos_ids_path, vis_limit_path,
    label_mask_path_or_None) tuples. If ``label_mask_paths`` is empty,
    every label_mask slot is None. If non-empty, every prefix must have
    a matching label_mask file.
    """
    token_by_prefix: Dict[str, str] = {}
    for p in token_paths:
        prefix = _extract_prefix(p, "-tokens.npy")
        token_by_prefix[prefix] = p

    pos_ids_by_prefix: Dict[str, str] = {}
    for p in pos_ids_paths:
        prefix = _extract_prefix(p, "-pos_ids.npy")
        pos_ids_by_prefix[prefix] = p

    vis_limit_by_prefix: Dict[str, str] = {}
    for p in vis_limit_paths:
        prefix = _extract_prefix(p, "-vis_limit.npy")
        vis_limit_by_prefix[prefix] = p

    label_mask_by_prefix: Dict[str, str] = {}
    for p in label_mask_paths:
        prefix = _extract_prefix(p, "-label_mask.npy")
        label_mask_by_prefix[prefix] = p

    required_sets = [
        set(token_by_prefix.keys()),
        set(pos_ids_by_prefix.keys()),
        set(vis_limit_by_prefix.keys()),
    ]
    if label_mask_paths:
        required_sets.append(set(label_mask_by_prefix.keys()))
    common_prefixes = sorted(set.intersection(*required_sets))

    if not common_prefixes:
        raise OLMoConfigurationError(
            "No matching file groups found. Token files, pos_ids files, vis_limit files "
            "(and optionally label_mask files) must share a common prefix (e.g. "
            "'part-03-00042-tokens.npy' matches 'part-03-00042-pos_ids.npy', "
            "'part-03-00042-vis_limit.npy', and 'part-03-00042-label_mask.npy')."
        )

    missing_pos = set(token_by_prefix.keys()) - set(pos_ids_by_prefix.keys())
    missing_vis = set(token_by_prefix.keys()) - set(vis_limit_by_prefix.keys())
    if missing_pos:
        log.warning(f"Token files without matching pos_ids: {sorted(missing_pos)[:5]}")
    if missing_vis:
        log.warning(f"Token files without matching vis_limit: {sorted(missing_vis)[:5]}")
    if label_mask_paths:
        missing_lm = set(token_by_prefix.keys()) - set(label_mask_by_prefix.keys())
        if missing_lm:
            log.warning(f"Token files without matching label_mask: {sorted(missing_lm)[:5]}")

    return [
        (
            token_by_prefix[prefix],
            pos_ids_by_prefix[prefix],
            vis_limit_by_prefix[prefix],
            label_mask_by_prefix.get(prefix) if label_mask_paths else None,
        )
        for prefix in common_prefixes
    ]


@dataclass
class PreChunkedInstanceSourceConfig(InstanceSourceConfig):
    """
    Config for :class:`PreChunkedInstanceSource`.

    Reads pre-aligned parallel ``.npy`` files where every ``sequence_length`` tokens
    is one instance, preserving window boundaries. Used for idealized overlap data
    that includes custom position IDs and visibility limits alongside tokens.
    """

    token_paths: List[str]
    """Glob patterns for token .npy files (e.g. ``"*-tokens.npy"``)."""
    pos_ids_paths: List[str]
    """Glob patterns for position ID .npy files (e.g. ``"*-pos_ids.npy"``)."""
    vis_limit_paths: List[str]
    """Glob patterns for visibility limit .npy files (e.g. ``"*-vis_limit.npy"``)."""
    sequence_length: int
    """The number of tokens per instance."""
    label_mask_paths: List[str] = field(default_factory=list)
    """Optional glob patterns for label mask .npy files (e.g. ``"*-label_mask.npy"``).
    When provided, every matched token shard must have a corresponding label_mask
    shard. False entries map to ``ignore_index`` in the labels — used by tree-mask
    data sources to drop the loss at "artificial fold-start" positions whose
    design context is empty (cross-pair anchor starts, within-pair adjacent starts
    with LCP=0)."""
    token_dtype: Optional[NumpyDatasetDType] = None
    """The numpy dtype for token arrays. Defaults to uint16."""
    label: Optional[str] = None
    """An optional label for this source."""

    def build(self, work_dir: PathOrStr) -> "PreChunkedInstanceSource":
        token_dtype = (
            self.token_dtype.as_np_dtype() if self.token_dtype is not None else np.uint16
        )

        # Expand globs (distributed: only rank 0 expands, then broadcasts).
        token_paths = self._expand_all_globs(self.token_paths)
        pos_ids_paths = self._expand_all_globs(self.pos_ids_paths)
        vis_limit_paths = self._expand_all_globs(self.vis_limit_paths)
        label_mask_paths = (
            self._expand_all_globs(self.label_mask_paths)
            if self.label_mask_paths
            else []
        )

        # Match by prefix.
        groups = _match_parallel_paths(
            token_paths, pos_ids_paths, vis_limit_paths, label_mask_paths
        )
        matched_token_paths = [t[0] for t in groups]
        matched_pos_ids_paths = [t[1] for t in groups]
        matched_vis_limit_paths = [t[2] for t in groups]
        matched_label_mask_paths: Optional[List[str]] = (
            [typing.cast(str, t[3]) for t in groups] if label_mask_paths else None
        )

        return PreChunkedInstanceSource(
            token_paths=matched_token_paths,
            pos_ids_paths=matched_pos_ids_paths,
            vis_limit_paths=matched_vis_limit_paths,
            label_mask_paths=matched_label_mask_paths,
            token_dtype=token_dtype,
            sequence_length=self.sequence_length,
            work_dir=work_dir,
            label=self.label,
        )

    @staticmethod
    def _expand_all_globs(patterns: Sequence[str]) -> List[str]:
        """Expand glob patterns, distributing the work from rank 0."""
        expanded: List[str]
        if dist_utils.get_rank() == 0:
            result: List[str] = []
            for pattern in patterns:
                if "*" in pattern:
                    matches = io.deterministic_glob_directory(pattern)
                    if not matches:
                        raise FileNotFoundError(f"No files matched pattern: {pattern}")
                    result.extend(matches)
                else:
                    result.append(pattern)
            expanded = result
        else:
            expanded = []
        expanded = dist_utils.broadcast_object(expanded)
        return expanded


class PreChunkedInstanceSource(InstanceSource):
    """
    An :class:`InstanceSource` that reads pre-chunked instances from parallel ``.npy`` files.

    Each set of files (tokens, pos_ids, vis_limit) must have the same number of tokens,
    and the total tokens in each file must be a multiple of ``sequence_length``.

    Instances are read at ``sequence_length``-aligned boundaries, preserving the original
    window structure of the data.
    """

    Config = PreChunkedInstanceSourceConfig
    DISPLAY_ICON = "\uf51e"

    def __init__(
        self,
        *,
        token_paths: Sequence[str],
        pos_ids_paths: Sequence[str],
        vis_limit_paths: Sequence[str],
        token_dtype: NumpyUIntTypes,
        sequence_length: int,
        work_dir: PathOrStr,
        label: Optional[str] = None,
        label_mask_paths: Optional[Sequence[str]] = None,
    ):
        super().__init__(sequence_length=sequence_length, work_dir=work_dir, label=label)
        self._token_paths = tuple(token_paths)
        self._pos_ids_paths = tuple(pos_ids_paths)
        self._vis_limit_paths = tuple(vis_limit_paths)
        self._label_mask_paths: Optional[Tuple[str, ...]] = (
            tuple(label_mask_paths) if label_mask_paths is not None else None
        )
        self._token_dtype = token_dtype

        # Query file sizes (rank 0 queries, then broadcasts).
        token_item_size = token_dtype(0).itemsize
        pos_ids_item_size = np.uint32(0).itemsize
        vis_limit_item_size = np.uint32(0).itemsize
        label_mask_item_size = np.uint8(0).itemsize

        if self.rank == 0:
            token_sizes = path_map(
                lambda p: io.get_file_size(p) // token_item_size, self._token_paths
            )
            pos_ids_sizes = path_map(
                lambda p: io.get_file_size(p) // pos_ids_item_size, self._pos_ids_paths
            )
            vis_limit_sizes = path_map(
                lambda p: io.get_file_size(p) // vis_limit_item_size, self._vis_limit_paths
            )
            label_mask_sizes = (
                path_map(
                    lambda p: io.get_file_size(p) // label_mask_item_size,
                    self._label_mask_paths,
                )
                if self._label_mask_paths is not None
                else []
            )
        else:
            token_sizes = []
            pos_ids_sizes = []
            vis_limit_sizes = []
            label_mask_sizes = []

        self._token_sizes: Tuple[int, ...] = tuple(dist_utils.broadcast_object(token_sizes))
        self._pos_ids_sizes: Tuple[int, ...] = tuple(dist_utils.broadcast_object(pos_ids_sizes))
        self._vis_limit_sizes: Tuple[int, ...] = tuple(dist_utils.broadcast_object(vis_limit_sizes))
        self._label_mask_sizes: Tuple[int, ...] = tuple(
            dist_utils.broadcast_object(label_mask_sizes)
        )

        # Validate sizes match.
        for i, (ts, ps, vs) in enumerate(
            zip(self._token_sizes, self._pos_ids_sizes, self._vis_limit_sizes)
        ):
            if ts != ps or ts != vs:
                raise OLMoConfigurationError(
                    f"File group {i}: token size ({ts}) != pos_ids size ({ps}) or vis_limit size ({vs}). "
                    f"All parallel arrays must have the same number of elements."
                )
        if self._label_mask_paths is not None:
            for i, (ts, ls) in enumerate(
                zip(self._token_sizes, self._label_mask_sizes)
            ):
                if ts != ls:
                    raise OLMoConfigurationError(
                        f"File group {i}: token size ({ts}) != label_mask size ({ls}). "
                        f"All parallel arrays must have the same number of elements."
                    )

        log.info(
            f"PreChunkedInstanceSource: {len(self._token_paths)} file groups, "
            f"{sum(self._token_sizes):,d} total tokens, "
            f"{len(self):,d} instances of length {sequence_length}"
            f"{' (with label_mask)' if self._label_mask_paths is not None else ''}"
        )

    @property
    def rank(self) -> int:
        return self._rank

    @ft.cached_property
    def _instances_per_file(self) -> Tuple[int, ...]:
        return tuple(size // self.sequence_length for size in self._token_sizes)

    @ft.cached_property
    def num_instances(self) -> int:
        return sum(self._instances_per_file)

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            f"class={self.__class__.__name__},seq_len={self.sequence_length},".encode()
        )
        for tp, pp, vp in zip(self._token_paths, self._pos_ids_paths, self._vis_limit_paths):
            sha256_hash.update(f"token={tp},pos_ids={pp},vis_limit={vp},".encode())
        if self._label_mask_paths is not None:
            for lp in self._label_mask_paths:
                sha256_hash.update(f"label_mask={lp},".encode())
        return sha256_hash.hexdigest()

    def __len__(self) -> int:
        return self.num_instances

    def __getitem__(self, idx: int) -> Instance:
        idx = self.validate_index(idx)

        # Find which file and offset within that file.
        file_start = 0
        for file_idx, n_instances in enumerate(self._instances_per_file):
            file_end = file_start + n_instances
            if file_start <= idx < file_end:
                local_idx = idx - file_start
                start_token = local_idx * self.sequence_length
                end_token = start_token + self.sequence_length

                input_ids = load_array_slice(
                    self._token_paths[file_idx], start_token, end_token, self._token_dtype
                )
                pos_ids = load_array_slice(
                    self._pos_ids_paths[file_idx], start_token, end_token, np.uint32
                )
                vis_limit = load_array_slice(
                    self._vis_limit_paths[file_idx], start_token, end_token, np.uint32
                )

                instance: Dict = {
                    "input_ids": typing.cast(Sequence[int], input_ids),
                    "pos_ids": typing.cast(Sequence[int], pos_ids),
                    "vis_limit": typing.cast(Sequence[int], vis_limit),
                }
                if self._label_mask_paths is not None:
                    label_mask = load_array_slice(
                        self._label_mask_paths[file_idx],
                        start_token,
                        end_token,
                        np.uint8,
                    ).astype(bool)
                    instance["label_mask"] = typing.cast(Sequence[bool], label_mask)
                return instance

            file_start = file_end

        raise IndexError(f"Index {idx} out of range for {self.num_instances} instances.")

    def children(self) -> Iterable["PreChunkedInstanceSource"]:
        return []
