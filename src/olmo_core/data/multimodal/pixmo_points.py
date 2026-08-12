"""PixMo pointing / counting + CoSyn pointing datasets for Molmo2 stage-1.

Ports mm_olmo's pointing data sources (``olmo/data/pixmo_datasets.py``):

* :class:`PixMoPointsDataset` — ``points-pointing`` / ``points-counting`` (or both);
  each row has several ``(label, points)`` annotations → a multi-branch example, each
  branch a ``pointing`` or ``point_count`` Q/A over the shared image.
* :class:`PixMoCountDataset` — ``count``; single-annotation, alternating ``point_count``
  / ``pointing`` style; points are pixel-space (normalized by image size).
* :class:`CoSynPointDataset` — ``cosyn-point``; each row has several ``(question, points,
  name)`` annotations → multi-branch pointing.

All answers use the html-v2 grounding format (see :mod:`.grounding`). Sequences are
assembled with :func:`~olmo_core.data.multimodal.sequence_builder.build_branched_sequence`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.nn.vision.molmo2_tokens import (
    N_PATCHES_SQ,
    PATCH_DIM,
    POOL_H,
    POOL_W,
    Molmo2TokenIds,
)

from .grounding import normalize_points, pointing_answer
from .message_sequence import encode_sft_example
from .sft_common import (
    SftMessageFormat,
    sft_example_rng,
    truncate_example,
    validate_sft_message_format,
)
from .sft_formatter import SftFormatter

__all__ = [
    "CoSynPointDataset",
    "CoSynPointDatasetConfig",
    "PixMoCountDataset",
    "PixMoCountDatasetConfig",
    "PixMoPointsDataset",
    "PixMoPointsDatasetConfig",
]

from .paths import PIXMO_DATASETS

_CONTENT_FINGERPRINT_VERSION = "pixmo-perception-adapter-v1"
_CONTENT_FINGERPRINT_DOMAIN = b"pixmo-perception-adapter-v1\0"
_SCALAR_COUNT_PROMPT = "How many {label} are there?"
_TOKEN_FIELDS = ("input_ids", "labels", "loss_masks", "position_ids", "token_type_ids")
_PERCENT_POINT_CLAMP_TOLERANCE = 2.0


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_arrow_fingerprint(dataset: Any, *, source_name: str, split: str) -> dict[str, Any]:
    """Return the stable identity of one selected Arrow source split."""
    fingerprint = getattr(dataset, "_fingerprint", None)
    if callable(fingerprint):
        fingerprint = fingerprint()
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(
            f"{source_name} split {split!r} does not expose a stable Arrow fingerprint"
        )
    return {
        "arrow_fingerprint": fingerprint,
        "num_rows": len(dataset),
        "source_name": source_name,
        "split": split,
    }


def _index_sha256(index: Sequence[tuple[int, list[int]]]) -> str:
    digest = hashlib.sha256()
    for row_index, label_indices in index:
        digest.update(
            _canonical_bytes(
                {"label_indices": [int(i) for i in label_indices], "row_index": int(row_index)}
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _adapter_fingerprint(
    adapter_name: str,
    config: Config,
    source_descriptors: Sequence[Mapping[str, Any]],
    *,
    derived_index_sha256: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "adapter": adapter_name,
        "config": asdict(config),
        "sources": list(source_descriptors),
        "version": _CONTENT_FINGERPRINT_VERSION,
    }
    if derived_index_sha256 is not None:
        payload["derived_index_sha256"] = derived_index_sha256
    return hashlib.sha256(_CONTENT_FINGERPRINT_DOMAIN + _canonical_bytes(payload)).hexdigest()


def _available_columns(dataset: Any) -> set[str] | None:
    columns = getattr(dataset, "column_names", None)
    if columns is None:
        return None
    return {str(column) for column in columns}


def _annotation_rows(dataset: Any, required_columns: Sequence[str]):
    """Iterate annotations without decoding the source's image column."""
    columns = _available_columns(dataset)
    if columns is not None:
        missing = sorted(set(required_columns) - columns)
        if missing:
            raise ValueError(f"Dataset lacks required annotation columns: {missing}")
    selected = dataset
    select_columns = getattr(dataset, "select_columns", None)
    if callable(select_columns):
        selected = select_columns(list(required_columns))
    for index in range(len(selected)):
        row = selected[index]
        if not isinstance(row, Mapping):
            raise TypeError(f"Annotation row {index} must be a mapping, got {type(row)}")
        missing = [column for column in required_columns if column not in row]
        if missing:
            raise ValueError(f"Annotation row {index} lacks required columns: {missing}")
        yield index, row


def _require_columns(dataset: Any, required_columns: Sequence[str]) -> None:
    columns = _available_columns(dataset)
    if columns is None:
        return
    missing = sorted(set(required_columns) - columns)
    if missing:
        raise ValueError(f"Dataset lacks required columns: {missing}")


def _require_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value


def _require_nonnegative_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return value


def _require_sequence(value: Any, *, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    return value


def _require_percent_point(point: Any, *, field_name: str) -> None:
    if not isinstance(point, Mapping) or "x" not in point or "y" not in point:
        raise ValueError(f"{field_name} must contain x/y coordinates")
    try:
        x, y = float(point["x"]), float(point["y"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} coordinates must be numeric") from error
    low, high = -_PERCENT_POINT_CLAMP_TOLERANCE, 100.0 + _PERCENT_POINT_CLAMP_TOLERANCE
    if not np.isfinite([x, y]).all() or not (low <= x <= high and low <= y <= high):
        raise ValueError(
            f"{field_name} coordinates must be finite percentages within the preprocessing "
            f"clamp tolerance [{low:g}, {high:g}]"
        )


def _require_xy_mapping(
    points: Any,
    *,
    field_name: str,
    percent_coordinates: bool,
) -> np.ndarray:
    if not isinstance(points, Mapping) or "x" not in points or "y" not in points:
        raise ValueError(f"{field_name} must contain x/y coordinate arrays")
    xs = _require_sequence(points["x"], field_name=f"{field_name}.x")
    ys = _require_sequence(points["y"], field_name=f"{field_name}.y")
    if len(xs) != len(ys):
        raise ValueError(f"{field_name}.x and {field_name}.y must have the same length")
    try:
        xy = np.asarray([xs, ys], dtype=np.float64).T.reshape(-1, 2)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} coordinates must be numeric") from error
    if not np.isfinite(xy).all():
        raise ValueError(f"{field_name} coordinates must be finite")
    if percent_coordinates and xy.size and np.any((xy < 0.0) | (xy > 100.0)):
        raise ValueError(f"{field_name} coordinates must be percentages in [0, 100]")
    return xy


def _validate_dataset_rows(dataset_name: str, rows, validate_row: Callable[[Any], None]) -> None:
    invalid_count = 0
    first_errors: list[str] = []
    row_count = 0
    for row_index, row in rows:
        row_count += 1
        try:
            validate_row(row)
        except (TypeError, ValueError) as error:
            invalid_count += 1
            if len(first_errors) < 8:
                first_errors.append(f"{row_index}: {error}")
    if invalid_count:
        raise ValueError(
            f"{dataset_name} has {invalid_count} invalid annotation rows out of {row_count}; "
            f"first errors: {first_errors}"
        )


def _validate_serialized_example(
    example: dict[str, np.ndarray],
    *,
    token_ids: Molmo2TokenIds,
    max_sequence_length: int | None,
    max_output_crops: int,
) -> None:
    """Fail before packing when a serialized image example violates model geometry."""
    missing = sorted((set(_TOKEN_FIELDS) | {"images", "pooled_patches_idx"}) - set(example))
    if missing:
        raise ValueError(f"Serialized example lacks required fields: {missing}")
    n_tokens: int | None = None
    for field_name in _TOKEN_FIELDS:
        value = example[field_name]
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise ValueError(f"{field_name} must be a rank-1 NumPy array")
        if n_tokens is None:
            n_tokens = len(value)
        elif len(value) != n_tokens:
            raise ValueError("All serialized token fields must have identical lengths")
    assert n_tokens is not None
    if n_tokens <= 0:
        raise ValueError("Serialized examples must contain at least one token")
    if max_sequence_length is not None and n_tokens > max_sequence_length:
        raise ValueError(
            f"Serialized example has {n_tokens} tokens, exceeding {max_sequence_length}"
        )
    for field_name in ("input_ids", "labels", "position_ids", "token_type_ids"):
        if not np.issubdtype(example[field_name].dtype, np.integer):
            raise ValueError(f"{field_name} must have an integer dtype")
    loss_masks = example["loss_masks"]
    if not np.issubdtype(loss_masks.dtype, np.floating):
        raise ValueError("loss_masks must have a floating dtype")
    if not np.isfinite(loss_masks).all() or np.any(loss_masks < 0):
        raise ValueError("loss_masks must be finite and nonnegative")
    if not np.any(loss_masks > 0):
        raise ValueError("Serialized example contains no supervised tokens")
    if np.any(example["position_ids"] < 0):
        raise ValueError("position_ids must be nonnegative")
    expected_token_types = np.isin(
        example["input_ids"], np.fromiter(token_ids.image_token_ids, dtype=np.int64)
    ).astype(np.int64)
    if not np.array_equal(example["token_type_ids"], expected_token_types):
        raise ValueError("token_type_ids do not exactly mark the configured image tokens")
    if "subsegment_ids" in example:
        subsegments = example["subsegment_ids"]
        if (
            not isinstance(subsegments, np.ndarray)
            or subsegments.ndim != 1
            or len(subsegments) != n_tokens
            or not np.issubdtype(subsegments.dtype, np.integer)
        ):
            raise ValueError("subsegment_ids must be a rank-1 integer array matching tokens")

    images = example["images"]
    if (
        not isinstance(images, np.ndarray)
        or images.dtype != np.float32
        or images.ndim != 3
        or images.shape[1:] != (N_PATCHES_SQ, PATCH_DIM)
    ):
        raise ValueError(f"images must be float32 with shape (crops, {N_PATCHES_SQ}, {PATCH_DIM})")
    if not 1 <= images.shape[0] <= max_output_crops:
        raise ValueError(
            f"images has {images.shape[0]} crops; expected between 1 and {max_output_crops}"
        )
    if not np.isfinite(images).all():
        raise ValueError("images must contain only finite values")

    pooled = example["pooled_patches_idx"]
    if (
        not isinstance(pooled, np.ndarray)
        or pooled.dtype != np.int64
        or pooled.ndim != 2
        or pooled.shape[1] != POOL_H * POOL_W
    ):
        raise ValueError(
            f"pooled_patches_idx must be int64 with shape (pooled_tokens, {POOL_H * POOL_W})"
        )
    valid = pooled >= 0
    if pooled.shape[0] == 0 or not np.all(valid.any(axis=1)):
        raise ValueError("Every pooled-patch row must contain at least one valid patch index")
    if np.any(pooled < -1) or np.any(pooled[valid] >= images.shape[0] * N_PATCHES_SQ):
        raise ValueError("pooled_patches_idx contains an out-of-range patch index")
    n_image_tokens = int(np.count_nonzero(example["input_ids"] == token_ids.im_patch_id))
    if n_image_tokens != pooled.shape[0]:
        raise ValueError(
            f"Serialized example has {n_image_tokens} <im_patch> tokens but "
            f"{pooled.shape[0]} pooled rows"
        )


def _finalize_example(
    example: dict[str, np.ndarray],
    *,
    strict_validation: bool,
    max_sequence_length: int | None,
    max_crops: int,
    high_res_max_crops: int,
    p_high_res: float,
    loss_token_weighting: str,
    token_ids: Molmo2TokenIds,
) -> dict[str, np.ndarray]:
    if not strict_validation:
        return example
    original_length = len(example["input_ids"])
    if max_sequence_length is not None:
        example = truncate_example(
            example,
            max_sequence_length,
            image_patch_token_id=token_ids.im_patch_id,
            image_token_ids=token_ids.image_token_ids,
            recompute_root_subsegments=loss_token_weighting
            in ("root_subsegments", "root_subsegments_root_tokens"),
        )
    effective_high_res = max_crops if p_high_res <= 0 else max(max_crops, high_res_max_crops)
    # The processor returns one global crop in addition to at most ``max_crops`` tiles.
    _validate_serialized_example(
        example,
        token_ids=token_ids,
        max_sequence_length=max_sequence_length,
        max_output_crops=effective_high_res + 1,
    )
    example["metadata"] = {
        **example.get("metadata", {}),
        "original_length": original_length,
        "truncated": max_sequence_length is not None and original_length > max_sequence_length,
    }
    return example


def _build_example(
    tokenizer,
    pil_image,
    build_branches: Callable[[np.random.RandomState], list[tuple[str, str]]],
    *,
    max_crops: int,
    strict_validation: bool = False,
    high_res_max_crops: int = 24,
    max_sequence_length: int | None = None,
    loss_token_weighting: str,
    token_ids: Molmo2TokenIds,
    message_weight: float | None = None,
    p_high_res: float = 0.0,
    message_format: SftMessageFormat = "qwen3",
    rng: np.random.RandomState,
) -> dict[str, np.ndarray]:
    """Format and assemble a (possibly multi-branch) pointing example.

    :param build_branches: Builds ``(user_question, assistant_answer)`` strings before image
        augmentation, preserving Molmo2's random-number consumption order.
    """
    branches_text = list(build_branches(rng))
    example = encode_sft_example(
        tokenizer,
        pil_image,
        branches_text,
        max_crops=max_crops,
        high_res_max_crops=high_res_max_crops,
        p_high_res=p_high_res,
        loss_token_weighting=loss_token_weighting,
        token_ids=token_ids,
        message_format=message_format,
        message_weight=message_weight,
        shuffle_rng=rng,
    )
    return _finalize_example(
        example,
        strict_validation=strict_validation,
        max_sequence_length=max_sequence_length,
        max_crops=max_crops,
        high_res_max_crops=high_res_max_crops,
        p_high_res=p_high_res,
        loss_token_weighting=loss_token_weighting,
        token_ids=token_ids,
    )


def _load_split(path: str, split: str, *, require_split: bool):
    from .dataset_compat import load_from_disk_compat

    ds = load_from_disk_compat(path)
    if hasattr(ds, "keys") and split in ds:
        return ds[split]
    if require_split:
        raise ValueError(f"Dataset {path!r} lacks required split {split!r}")
    return ds


def _open_image(p):
    from PIL import Image

    return p if isinstance(p, Image.Image) else Image.open(p)


# ---------------------------------------------------------------------------
# PixMo points (pointing / counting)
# ---------------------------------------------------------------------------


@dataclass
class PixMoPointsDatasetConfig(Config):
    """Configure PixMo's multi-annotation pointing/counting source.

    ``kind="basic"`` selects ``points-pointing`` and ``kind="high_frequency"`` selects
    ``points-counting``. ``kind="both"`` concatenates them in that order. When
    ``max_sequence_length`` is set, examples are safely truncated and validated before
    they can reach packing or collation.
    """

    split: str = "train"
    require_split: bool = False
    """Require ``split`` to exist in the saved Arrow ``DatasetDict``."""
    kind: str = "both"  # "basic" (points-pointing) | "high_frequency" (points-counting) | "both"
    counting: str | bool = "both"  # "both" randomly selects; bool fixes one style
    both_mode: Literal["per_annotation", "duplicate"] = "per_annotation"
    """How ``counting='both'`` is sampled. Stage 2 follows mm_olmo and samples a style
    per annotation. The Stage 1 recipe explicitly selects ``duplicate`` for compatibility
    with its existing two-style dataset expansion."""
    max_points: int = 60
    max_total_points_per_example: int = 60
    max_crops: int = 8
    high_res_max_crops: int = 24
    max_sequence_length: int | None = None
    loss_token_weighting: str = "root_subsegments"
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    message_weight: float | None = None
    p_high_res: float = 0.0
    message_format: SftMessageFormat = "qwen3"
    seed: int = 0

    def build(self, tokenizer) -> PixMoPointsDataset:
        return PixMoPointsDataset(self, tokenizer)


class PixMoPointsDataset:
    """Map-style PixMo points adapter with audited annotations and stable identity."""

    content_fingerprint_version = _CONTENT_FINGERPRINT_VERSION

    def __init__(self, config: PixMoPointsDatasetConfig, tokenizer):
        if config.kind not in ("basic", "high_frequency", "both"):
            raise ValueError(f"Unknown PixMo points kind {config.kind!r}")
        if config.counting not in (False, True, "both"):
            raise ValueError(f"Unknown PixMo points counting mode {config.counting!r}")
        if config.both_mode not in ("per_annotation", "duplicate"):
            raise ValueError(f"Unknown PixMo points both_mode {config.both_mode!r}")
        if not isinstance(config.split, str) or not config.split:
            raise ValueError("PixMo points split must be a non-empty string")
        if config.max_points < 0 or config.max_total_points_per_example <= 0:
            raise ValueError("PixMo point limits must be nonnegative with a positive total")
        if config.max_crops <= 0 or config.high_res_max_crops <= 0:
            raise ValueError("PixMo crop limits must be positive")
        if not 0.0 <= config.p_high_res <= 1.0:
            raise ValueError("p_high_res must be in [0, 1]")
        if config.max_sequence_length is not None and config.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive when set")
        validate_sft_message_format(
            config.message_format,
            tokenizer=tokenizer,
            token_ids=config.token_ids,
        )
        self.config = config
        self.tokenizer = tokenizer
        sub = {
            "basic": ["points-pointing"],
            "high_frequency": ["points-counting"],
            "both": ["points-counting", "points-pointing"],
        }[config.kind]
        from datasets import concatenate_datasets

        self._sources = [
            _load_split(
                f"{PIXMO_DATASETS}/{source_name}",
                config.split,
                require_split=config.require_split,
            )
            for source_name in sub
        ]
        self._data = concatenate_datasets(self._sources)
        # Pre-split each row's labels into sub-batches with <= max_total_points (mm_olmo).
        self._index = self._build_sub_index()
        if config.require_split:
            source_descriptors = [
                _require_arrow_fingerprint(source, source_name=name, split=config.split)
                for name, source in zip(sub, self._sources)
            ]
            self.content_fingerprint = _adapter_fingerprint(
                type(self).__name__,
                config,
                source_descriptors,
                derived_index_sha256=_index_sha256(self._index),
            )
        self._annotations_validated = False

    def _build_sub_index(self) -> list[tuple[int, list[int]]]:
        cfg = self.config
        counts = self._data["count"]
        labels = self._data["label"] if cfg.require_split else None
        index: list[tuple[int, list[int]]] = []
        skipped_blank_labels = 0
        skipped_over_max_points = 0
        for row, point_counts in enumerate(counts):
            row_labels = labels[row] if labels is not None else None
            if row_labels is not None and len(point_counts) != len(row_labels):
                raise ValueError(
                    f"PixMo points row {row} has {len(point_counts)} counts but "
                    f"{len(row_labels)} labels"
                )
            on: list[int] = []
            total = 0
            for li, n in enumerate(point_counts):
                if row_labels is not None and (
                    not isinstance(row_labels[li], str) or not row_labels[li].strip()
                ):
                    skipped_blank_labels += 1
                    continue
                if n > cfg.max_points:
                    skipped_over_max_points += 1
                    continue
                if on and total + n > cfg.max_total_points_per_example:
                    index.append((row, on))
                    on, total = [], 0
                on.append(li)
                total += n
            if on:
                index.append((row, on))
        self.annotation_filter_stats = {
            "blank_labels": skipped_blank_labels,
            "over_max_points": skipped_over_max_points,
        }
        return index

    def __len__(self) -> int:
        size = len(self._index)
        if self.config.counting == "both" and self.config.both_mode == "duplicate":
            return size * 2
        return size

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        return self.get(i, 0)

    def raw_image_references(self, index: int) -> Tuple[Any, ...]:
        """Return the source image reference for one expanded logical example."""
        example_index = (
            index // 2
            if self.config.counting == "both" and self.config.both_mode == "duplicate"
            else index
        )
        row_index, _ = self._index[example_index]
        return (self._data[row_index]["image"],)

    def annotation_content_sha256(self) -> str:
        """Hash every ordered logical annotation retained by the derived row index."""
        digest = hashlib.sha256()
        for logical_index, (row_index, label_indices) in enumerate(self._index):
            row = self._data[row_index]
            digest.update(
                _canonical_bytes(
                    {
                        "collection_method": [
                            row["collection_method"][index] for index in label_indices
                        ],
                        "count": [row["count"][index] for index in label_indices],
                        "label": [row["label"][index] for index in label_indices],
                        "label_indices": list(label_indices),
                        "logical_index": logical_index,
                        "image": row["image"],
                        "points": [row["points"][index] for index in label_indices],
                        "row_index": row_index,
                    }
                )
                + b"\n"
            )
        return digest.hexdigest()

    def validate_required_annotations(self) -> None:
        """Exhaustively validate the non-image annotations used by this adapter.

        Images are deliberately not decoded during this scan. Their serialized crop and
        pooling geometry is checked on every built example before packing.

        :raises ValueError: If any row has missing, malformed, or out-of-range annotations.
        """
        if self._annotations_validated:
            return
        _require_columns(
            self._data,
            ("image", "label", "points", "count", "collection_method"),
        )

        retained_by_row: dict[int, set[int]] = {}
        for row_index, label_indices in self._index:
            retained_by_row.setdefault(row_index, set()).update(label_indices)

        def retained_rows():
            rows = _annotation_rows(
                self._data,
                ("label", "points", "count", "collection_method"),
            )
            for row_index, row in rows:
                retained = sorted(retained_by_row.get(row_index, ()))
                if not retained:
                    continue
                yield row_index, {
                    field_name: [row[field_name][index] for index in retained]
                    for field_name in ("label", "points", "count", "collection_method")
                }

        def validate_row(row: Mapping[str, Any]) -> None:
            labels = _require_sequence(row["label"], field_name="label")
            point_groups = _require_sequence(row["points"], field_name="points")
            counts = _require_sequence(row["count"], field_name="count")
            methods = _require_sequence(row["collection_method"], field_name="collection_method")
            sizes = {len(labels), len(point_groups), len(counts), len(methods)}
            if len(sizes) != 1 or not labels:
                raise ValueError(
                    "label, points, count, and collection_method must be equally sized and nonempty"
                )
            for annotation_index, (label, points, count, method) in enumerate(
                zip(labels, point_groups, counts, methods)
            ):
                prefix = f"annotation {annotation_index}"
                _require_text(label, field_name=f"{prefix}.label")
                count = _require_nonnegative_integer(count, field_name=f"{prefix}.count")
                points = _require_sequence(points, field_name=f"{prefix}.points")
                if len(points) != count:
                    raise ValueError(
                        f"{prefix}.count={count} does not match {len(points)} point annotations"
                    )
                method = _require_text(method, field_name=f"{prefix}.collection_method")
                if method not in ("pointing", "counting"):
                    raise ValueError(f"{prefix}.collection_method has unknown value {method!r}")
                for point_index, point in enumerate(points):
                    _require_percent_point(
                        point,
                        field_name=f"{prefix}.points[{point_index}]",
                    )

        _validate_dataset_rows(
            "PixMo points",
            retained_rows(),
            validate_row,
        )
        self._annotations_validated = True

    def get(self, i: int, epoch: int = 0) -> dict[str, np.ndarray]:
        """Build one deterministically augmented example for a source epoch."""
        fixed_style = None
        if self.config.counting == "both" and self.config.both_mode == "duplicate":
            example_idx = i // 2
            fixed_style = "point_count" if i % 2 == 0 else "pointing"
        else:
            example_idx = i
        row_idx, label_idxs = self._index[example_idx]
        rng = sft_example_rng(self.config.seed, i, epoch, self.config.message_format)
        row = self._data[row_idx]
        fmt = SftFormatter(seed=self.config.seed)
        specs: list[tuple[str, str, Any]] = []
        for li in label_idxs:
            label = row["label"][li]
            pts = row["points"][li]
            if fixed_style is not None:
                style = fixed_style
            elif self.config.counting == "both":
                style = rng.choice(["point_count", "pointing"])
            else:
                style = "point_count" if self.config.counting else "pointing"
            specs.append((style, label, pts))

        def build_branches(branch_rng: np.random.RandomState) -> list[tuple[str, str]]:
            branches: list[tuple[str, str]] = []
            for branch_style, label, points in specs:
                sub = {
                    "style": branch_style,
                    "label": label,
                    "points": points,
                    "point_scale": 100,
                }
                prompt, answer = fmt.format_turns(sub, index=i, rng=branch_rng)[0]
                branches.append((prompt, answer))
            return branches

        return _build_example(
            self.tokenizer,
            _open_image(row["image"]),
            build_branches,
            max_crops=self.config.max_crops,
            strict_validation=self.config.require_split,
            high_res_max_crops=self.config.high_res_max_crops,
            max_sequence_length=self.config.max_sequence_length,
            loss_token_weighting=self.config.loss_token_weighting,
            token_ids=self.config.token_ids,
            message_weight=self.config.message_weight,
            p_high_res=self.config.p_high_res,
            message_format=self.config.message_format,
            rng=rng,
        )


# ---------------------------------------------------------------------------
# PixMo count (single annotation, alternating point_count / pointing)
# ---------------------------------------------------------------------------


@dataclass
class PixMoCountDatasetConfig(Config):
    """Configure PixMo Count grounding or scalar-count document continuations."""

    split: str = "train"
    require_split: bool = False
    """Require ``split`` to exist in the saved Arrow ``DatasetDict``."""
    mode: Literal["grounded", "scalar_count"] = "grounded"
    """``scalar_count`` always supervises the declared integer using document layout."""
    counting: str | bool = "both"  # "both" interleaves point_count (even) / pointing (odd)
    max_crops: int = 8
    high_res_max_crops: int = 24
    max_sequence_length: int | None = None
    loss_token_weighting: str = "root_subsegments"
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    message_weight: float | None = None
    p_high_res: float = 0.0
    message_format: SftMessageFormat = "qwen3"
    seed: int = 0

    def build(self, tokenizer) -> PixMoCountDataset:
        return PixMoCountDataset(self, tokenizer)


class PixMoCountDataset:
    """Map-style PixMo Count adapter with an explicit scalar-count mode."""

    content_fingerprint_version = _CONTENT_FINGERPRINT_VERSION

    def __init__(self, config: PixMoCountDatasetConfig, tokenizer):
        if config.mode not in ("grounded", "scalar_count"):
            raise ValueError(f"Unknown PixMo Count mode {config.mode!r}")
        if config.counting not in (False, True, "both"):
            raise ValueError(f"Unknown PixMo Count counting mode {config.counting!r}")
        if config.mode == "scalar_count":
            if config.message_format != "document":
                raise ValueError("PixMo scalar_count mode requires message_format='document'")
            if config.counting != "both":
                raise ValueError(
                    "PixMo scalar_count mode does not use grounded counting styles; leave "
                    "counting='both'"
                )
        if not isinstance(config.split, str) or not config.split:
            raise ValueError("PixMo Count split must be a non-empty string")
        if config.max_crops <= 0 or config.high_res_max_crops <= 0:
            raise ValueError("PixMo crop limits must be positive")
        if not 0.0 <= config.p_high_res <= 1.0:
            raise ValueError("p_high_res must be in [0, 1]")
        if config.max_sequence_length is not None and config.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive when set")
        validate_sft_message_format(
            config.message_format,
            tokenizer=tokenizer,
            token_ids=config.token_ids,
        )
        self.config = config
        self.tokenizer = tokenizer
        self._data = _load_split(
            f"{PIXMO_DATASETS}/count",
            config.split,
            require_split=config.require_split,
        )
        self._n = len(self._data)
        if config.require_split:
            descriptor = _require_arrow_fingerprint(
                self._data,
                source_name="count",
                split=config.split,
            )
            self.content_fingerprint = _adapter_fingerprint(
                type(self).__name__,
                config,
                [descriptor],
            )
        self._annotations_validated = False

    def __len__(self) -> int:
        if self.config.mode == "scalar_count":
            return self._n
        return self._n * 2 if self.config.counting == "both" else self._n

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        return self.get(i, 0)

    def raw_image_references(self, index: int) -> Tuple[Any, ...]:
        """Return the source image reference for one scalar or grounded logical example."""
        row_index = (
            index // 2
            if self.config.mode == "grounded" and self.config.counting == "both"
            else index
        )
        return (self._data[row_index]["image"],)

    def annotation_content_sha256(self) -> str:
        """Hash every ordered scalar/grounding annotation without opening images."""
        digest = hashlib.sha256()
        for row_index, row in _annotation_rows(self._data, ("label", "count", "points")):
            digest.update(
                _canonical_bytes(
                    {
                        "image": self._data[row_index]["image"],
                        "index": row_index,
                        **dict(row),
                    }
                )
                + b"\n"
            )
        return digest.hexdigest()

    def validate_required_annotations(self) -> None:
        """Validate every declared count, label, and optional grounding coordinate array.

        The scalar-count mode intentionally treats ``count`` as authoritative on every
        split; validation rows may therefore carry empty point arrays for positive counts.

        :raises ValueError: If a required annotation is absent or malformed.
        """
        if self._annotations_validated:
            return
        _require_columns(self._data, ("image", "label", "count", "points"))

        def validate_row(row: Mapping[str, Any]) -> None:
            _require_text(row["label"], field_name="label")
            _require_nonnegative_integer(row["count"], field_name="count")
            _require_xy_mapping(
                row["points"],
                field_name="points",
                percent_coordinates=False,
            )

        _validate_dataset_rows(
            "PixMo Count",
            _annotation_rows(self._data, ("label", "count", "points")),
            validate_row,
        )
        self._annotations_validated = True

    def get(self, i: int, epoch: int = 0) -> dict[str, np.ndarray]:
        """Build one deterministically augmented example for a source epoch."""
        if self.config.mode == "scalar_count":
            row_idx, style = i, "scalar_count"
        elif self.config.counting == "both":
            row_idx, style = i // 2, ("point_count" if i % 2 == 0 else "pointing")
        else:
            row_idx, style = i, ("point_count" if self.config.counting else "pointing")
        row = self._data[row_idx]
        label = row["label"]
        count = int(row["count"])
        pil = _open_image(row["image"])
        pts = row.get("points") or {"x": [], "y": []}
        rng = sft_example_rng(self.config.seed, i, epoch, self.config.message_format)
        fmt = SftFormatter(seed=self.config.seed)
        if self.config.require_split:
            xy = _require_xy_mapping(
                pts,
                field_name=f"row {row_idx}.points",
                percent_coordinates=False,
            )
            if style != "scalar_count" and xy.size:
                width, height = pil.size
                if width <= 0 or height <= 0:
                    raise ValueError(f"row {row_idx} image has invalid size {pil.size!r}")
                if np.any(xy[:, 0] < 0) or np.any(xy[:, 0] > width):
                    raise ValueError(f"row {row_idx} contains an x coordinate outside the image")
                if np.any(xy[:, 1] < 0) or np.any(xy[:, 1] > height):
                    raise ValueError(f"row {row_idx} contains a y coordinate outside the image")
        else:
            xy = np.array([pts["x"], pts["y"]], dtype=np.float64).T.reshape(-1, 2)
        sub = {
            "style": style,
            "label": label,
            "points": xy,
            "point_scale": None,
            "image_size": pil.size,
        }

        def build_branches(branch_rng: np.random.RandomState) -> list[tuple[str, str]]:
            if style == "scalar_count":
                return [(_SCALAR_COUNT_PROMPT.format(label=label), str(count))]
            # PixMo Count validation/test retain the declared count but omit point
            # annotations. Those rows support count-only evaluation, not grounding.
            if len(xy) == 0 and count > 0:
                return [
                    (
                        f"How many {label} are there?",
                        pointing_answer(xy, label, "count", count=count),
                    )
                ]
            prompt, answer = fmt.format_turns(sub, index=i, rng=branch_rng)[0]
            return [(prompt, answer)]

        return _build_example(
            self.tokenizer,
            pil,
            build_branches,
            max_crops=self.config.max_crops,
            strict_validation=self.config.require_split,
            high_res_max_crops=self.config.high_res_max_crops,
            max_sequence_length=self.config.max_sequence_length,
            loss_token_weighting=self.config.loss_token_weighting,
            token_ids=self.config.token_ids,
            message_weight=self.config.message_weight,
            p_high_res=self.config.p_high_res,
            message_format=self.config.message_format,
            rng=rng,
        )


# ---------------------------------------------------------------------------
# CoSyn point (document pointing; multi-branch, prompt = the question)
# ---------------------------------------------------------------------------


@dataclass
class CoSynPointDatasetConfig(Config):
    """Configure CoSyn pointing with an explicit Arrow split and sequence bound."""

    split: str = "train"
    require_split: bool = False
    """Require ``split`` to exist in the saved Arrow ``DatasetDict``."""
    max_crops: int = 8
    high_res_max_crops: int = 24
    max_sequence_length: int | None = None
    loss_token_weighting: str = "root_subsegments"
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    message_weight: float | None = None
    p_high_res: float = 0.0
    message_format: SftMessageFormat = "qwen3"
    seed: int = 0

    def build(self, tokenizer) -> CoSynPointDataset:
        return CoSynPointDataset(self, tokenizer)


class CoSynPointDataset:
    """Map-style CoSyn pointing adapter with audited percent-coordinate geometry."""

    content_fingerprint_version = _CONTENT_FINGERPRINT_VERSION

    def __init__(self, config: CoSynPointDatasetConfig, tokenizer):
        if not isinstance(config.split, str) or not config.split:
            raise ValueError("CoSyn Point split must be a non-empty string")
        if config.max_crops <= 0 or config.high_res_max_crops <= 0:
            raise ValueError("CoSyn crop limits must be positive")
        if not 0.0 <= config.p_high_res <= 1.0:
            raise ValueError("p_high_res must be in [0, 1]")
        if config.max_sequence_length is not None and config.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive when set")
        validate_sft_message_format(
            config.message_format,
            tokenizer=tokenizer,
            token_ids=config.token_ids,
        )
        self.config = config
        self.tokenizer = tokenizer
        self._data = _load_split(
            f"{PIXMO_DATASETS}/cosyn-point",
            config.split,
            require_split=config.require_split,
        )
        if config.require_split:
            descriptor = _require_arrow_fingerprint(
                self._data,
                source_name="cosyn-point",
                split=config.split,
            )
            self.content_fingerprint = _adapter_fingerprint(
                type(self).__name__,
                config,
                [descriptor],
            )
        self._annotations_validated = False

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        return self.get(i, 0)

    def raw_image_references(self, index: int) -> Tuple[Any, ...]:
        """Return the source image reference for one CoSyn logical example."""
        return (self._data[index]["image"],)

    def annotation_content_sha256(self) -> str:
        """Hash every ordered CoSyn question/name/point annotation."""
        digest = hashlib.sha256()
        for row_index, row in _annotation_rows(self._data, ("questions", "answer_points", "names")):
            digest.update(
                _canonical_bytes(
                    {
                        "image": self._data[row_index]["image"],
                        "index": row_index,
                        **dict(row),
                    }
                )
                + b"\n"
            )
        return digest.hexdigest()

    def validate_required_annotations(self) -> None:
        """Validate every question, answer point, and object name without decoding images.

        :raises ValueError: If branches are missing, misaligned, or outside percent geometry.
        """
        if self._annotations_validated:
            return
        _require_columns(self._data, ("image", "questions", "answer_points", "names"))

        def validate_row(row: Mapping[str, Any]) -> None:
            questions = _require_sequence(row["questions"], field_name="questions")
            answer_points = _require_sequence(row["answer_points"], field_name="answer_points")
            names = _require_sequence(row["names"], field_name="names")
            sizes = {len(questions), len(answer_points), len(names)}
            if len(sizes) != 1 or not questions:
                raise ValueError(
                    "questions, answer_points, and names must be equally sized and nonempty"
                )
            for branch_index, (question, points, name) in enumerate(
                zip(questions, answer_points, names)
            ):
                prefix = f"branch {branch_index}"
                _require_text(question, field_name=f"{prefix}.question")
                _require_text(name, field_name=f"{prefix}.name")
                xy = _require_xy_mapping(
                    points,
                    field_name=f"{prefix}.answer_points",
                    percent_coordinates=True,
                )
                if len(xy) == 0:
                    raise ValueError(f"{prefix}.answer_points must be nonempty")

        _validate_dataset_rows(
            "CoSyn Point",
            _annotation_rows(self._data, ("questions", "answer_points", "names")),
            validate_row,
        )
        self._annotations_validated = True

    def get(self, i: int, epoch: int = 0) -> dict[str, np.ndarray]:
        """Build one deterministically augmented example for a source epoch."""
        row = self._data[i]
        if self.config.require_split:
            questions = _require_sequence(row["questions"], field_name=f"row {i}.questions")
            answer_points = _require_sequence(
                row["answer_points"], field_name=f"row {i}.answer_points"
            )
            names = _require_sequence(row["names"], field_name=f"row {i}.names")
            if len({len(questions), len(answer_points), len(names)}) != 1 or not questions:
                raise ValueError(
                    f"row {i} questions, answer_points, and names must be equally sized and "
                    "nonempty"
                )
        else:
            questions = row["questions"]
            answer_points = row["answer_points"]
            names = row["names"]
        branches: list[tuple[str, str]] = []
        for branch_index, (question, points, name) in enumerate(
            zip(questions, answer_points, names)
        ):
            if self.config.require_split:
                question = _require_text(question, field_name=f"row {i}.question[{branch_index}]")
                name = _require_text(name, field_name=f"row {i}.name[{branch_index}]")
                xy = _require_xy_mapping(
                    points,
                    field_name=f"row {i}.answer_points[{branch_index}]",
                    percent_coordinates=True,
                )
                if len(xy) == 0:
                    raise ValueError(f"row {i}.answer_points[{branch_index}] must be nonempty")
            else:
                xy = np.array([points["x"], points["y"]], dtype=np.float64).T.reshape(-1, 2)
            norm = normalize_points(xy, point_scale=100, image_size=None)
            # cosyn_point uses the "pointing" answer (just the points tag), label = name.
            answer = pointing_answer(norm, name.lower(), "pointing", count=len(norm))
            branches.append((question, answer))
        return _build_example(
            self.tokenizer,
            _open_image(row["image"]),
            lambda branch_rng: branches,
            max_crops=self.config.max_crops,
            strict_validation=self.config.require_split,
            high_res_max_crops=self.config.high_res_max_crops,
            max_sequence_length=self.config.max_sequence_length,
            loss_token_weighting=self.config.loss_token_weighting,
            token_ids=self.config.token_ids,
            message_weight=self.config.message_weight,
            p_high_res=self.config.p_high_res,
            message_format=self.config.message_format,
            rng=sft_example_rng(self.config.seed, i, epoch, self.config.message_format),
        )
