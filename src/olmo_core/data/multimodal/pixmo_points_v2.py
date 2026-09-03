"""Audited PixMo pointing / counting sources for Molmo2 stage-1.

Ports mm_olmo's second-generation pointing data (``olmo/data/pixmo_datasets.py``
``PixMoPointV2`` and ``PixMoCountConfigV2``), the sources its molmo3 stage-1 mixture uses in
place of ``pixmo_points_train`` / ``pixmo_points_high_freq_train`` / ``pixmo_count_train``
(``launch_scripts/train_molmo3_stage1.py``, ``_base_mixture``). Compared to the v1 builds in
:mod:`.pixmo_points`, every row is one *image* carrying all of its annotations, and:

* every point set has an ``audit_result`` -- a VLM re-evaluation: ``correct`` / ``unsure`` /
  ``error`` / ``clear_error`` (``n/a`` for empty sets). A failed audit (``error`` or
  ``clear_error``) can either be dropped (``filter_audit=True``) or trained on behind a marker
  style (``audit_style``, e.g. ``aux_pointing``) that differs from the base style only in its
  ``"<style>:"`` token, so those less reliable targets do not dilute the primary
  ``pointing`` / ``point_count`` distributions. Both work, per the mm_olmo team.
* :class:`PixMoPointsV2Dataset` rows also ship absence queries for training the
  ``"There are none."`` refusal: ``easy_negatives`` (unrelated labels) and
  ``paired_negatives`` / ``paired_negatives_v2`` (hard, near-miss labels for the image's own
  objects). These are meant to be *sub-sampled* per epoch (``n_easy_samples``,
  ``p_paired_negatives``); training on all of them makes the model over-refuse.

Both datasets format through :class:`~.sft_formatter.SftFormatter` and the html-v2 grounding
format like their v1 siblings, and assemble with the shared ``_build_example`` (image
preprocessing + branched sequence). Not ported from ``PixMoPointV2``: the segmentation-mask
branches (``include_masks`` / ``pixmo_seg``) and the 3D / surface-normal renderings, none of
which Molmo2 can emit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError

from .paths import PIXMO_DATASETS, PIXMO_POINTS_V2
from .pixmo_points import _build_example, _load_split, _open_image
from .sequence_builder import example_rng
from .sft_formatter import SftFormatter

__all__ = [
    "FAILED_AUDIT_RESULTS",
    "PixMoPointsV2DatasetConfig",
    "PixMoPointsV2Dataset",
    "PixMoCountV2DatasetConfig",
    "PixMoCountV2Dataset",
]

log = logging.getLogger(__name__)

#: ``audit_result`` values mm_olmo treats as a failed audit (``PixMoPointV2._keep``).
FAILED_AUDIT_RESULTS = frozenset({"error", "clear_error"})

# mm_olmo ``PixMoPointV2.kind`` -> the ``source`` column value it keeps.
_KIND_TO_SOURCE = {"basic": "pointing", "high_frequency": "counting", "both": None}


def _failed_audit(audit_result: Optional[str]) -> bool:
    return audit_result in FAILED_AUDIT_RESULTS


def _choose(rng: np.random.RandomState, options: Sequence[str]) -> str:
    """``rng.choice`` over a style tuple, as a plain ``str``."""
    return str(rng.choice(list(options)))


def _rows_with_any(column, keep_fn) -> np.ndarray:
    """Per-row ``any(keep_fn(annotation))`` over a list-of-struct Arrow column.

    Works chunk by chunk on the raw Arrow buffers so that, unlike materialising the column
    with ``dataset["annotations"]``, the (large) mask RLE strings are never decoded: the
    full 224k-row PixMo-Points scan takes ~0.1 s. ``keep_fn`` receives the flattened
    struct array of one chunk and returns a boolean numpy mask over it.
    """
    import pyarrow.compute as pc

    out: List[np.ndarray] = []
    for chunk in column.chunks:
        flat = pc.list_flatten(chunk)
        keep = np.asarray(keep_fn(flat), dtype=bool)
        parents = pc.list_parent_indices(chunk).to_numpy(zero_copy_only=False)
        row_any = np.zeros(len(chunk), dtype=bool)
        if len(parents):
            row_any[np.unique(parents[keep])] = True
        out.append(row_any)
    return np.concatenate(out) if out else np.zeros(0, dtype=bool)


def _bool_np(arr) -> np.ndarray:
    """Arrow boolean array -> numpy bool, nulls as False."""
    import pyarrow.compute as pc

    return pc.fill_null(arr, False).to_numpy(zero_copy_only=False).astype(bool)


# ---------------------------------------------------------------------------
# PixMo points v2: audited, image-grouped pointing + counting with absence queries
# ---------------------------------------------------------------------------


@dataclass
class PixMoPointsV2DatasetConfig(Config):
    """mm_olmo ``PixMoPointV2`` (its pointing leg). Field names follow mm_olmo's.

    Rows are images; each carries several ``(label, points, audit_result)`` annotations plus
    per-image negative labels. Every annotation kept by :meth:`PixMoPointsV2Dataset.keep`
    becomes one branch of a multi-branch example; annotated empty point sets and the
    sub-sampled negatives become ``"There are none."`` branches.
    """

    dataset_path: str = PIXMO_POINTS_V2
    """A flat HF ``Dataset`` saved with ``save_to_disk`` (columns ``image``, ``source``,
    ``annotations``, ``easy_negatives``, ``paired_negatives``, ``paired_negatives_v2``)."""

    kind: str = "both"
    """``"basic"`` keeps the PixMo-Points rows (``source == "pointing"``), ``"high_frequency"``
    the PixMo-Count-style rows (``"counting"``), ``"both"`` everything."""

    style: Tuple[str, ...] = ("point_count", "pointing")
    """Styles drawn uniformly per annotation and per negative (mm_olmo ``style``)."""

    max_points: int = 60
    """Annotations with more points are skipped (they are also the least reliable)."""

    min_points: int = 0
    """Annotations with fewer points are skipped; ``0`` keeps the annotated empty sets, which
    render as ``"There are none."``."""

    filter_audit: bool = False
    """Drop annotations whose ``audit_result`` is ``error`` / ``clear_error``."""

    audit_style: Optional[Tuple[str, ...]] = None
    """Styles drawn for audit-failed annotations that are kept, e.g.
    ``("aux_point_count", "aux_pointing")``. ``None`` renders them like everything else."""

    n_easy_samples: int = 2
    """Easy negatives sampled per image and epoch (mm_olmo samples a fixed 2 regardless of
    this field's value; we honour the field, which is identical at the default)."""

    n_hard_negatives: int = 0
    """Negatives sampled from ``easy_negatives + paired_negatives`` (the v1 pool); mm_olmo's
    stage 1 leaves this at 0."""

    p_paired_negatives: float = 0.0
    """Expected fraction of the image's paired (hard) negatives used per epoch; the count is
    the stochastically-rounded ``len * p``. mm_olmo's stage 1 uses 0.25."""

    v2_paired_negatives: bool = True
    """Sample from ``paired_negatives_v2`` rather than ``paired_negatives``."""

    negative_weight: Optional[float] = None
    """Loss multiplier for the paired and annotated negatives. Easy negatives always weigh 1
    (mm_olmo ``PixMoPointV2.format_example``)."""

    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments"
    message_weight: Optional[float] = None
    p_high_res: float = 0.0
    seed: int = 0
    prompt_templates: str = "uber_model_v2"
    """Prompt family for the question text; stage 1 uses ``"none"`` (bare label)."""
    system_prompt: str = "demo_or_style_v2"
    """Prompt family for the style prefix; stage 1 uses ``"style_and_length_v2"``."""

    def validate(self):
        if self.kind not in _KIND_TO_SOURCE:
            raise OLMoConfigurationError(
                f"kind={self.kind!r} is not one of {tuple(_KIND_TO_SOURCE)}"
            )
        if not self.style:
            raise OLMoConfigurationError("style must name at least one style")
        if self.audit_style is not None and not self.audit_style:
            raise OLMoConfigurationError("audit_style must be None or name at least one style")
        if not 0.0 <= self.p_paired_negatives <= 1.0:
            raise OLMoConfigurationError("p_paired_negatives must be in [0, 1]")
        if self.min_points < 0 or self.max_points < self.min_points:
            raise OLMoConfigurationError("need 0 <= min_points <= max_points")
        if self.n_easy_samples < 0 or self.n_hard_negatives < 0:
            raise OLMoConfigurationError("n_easy_samples / n_hard_negatives must be >= 0")

    def build(self, tokenizer) -> "PixMoPointsV2Dataset":
        self.validate()
        return PixMoPointsV2Dataset(self, tokenizer)


class PixMoPointsV2Dataset:
    """Map-style dataset over the images of :class:`PixMoPointsV2DatasetConfig` that have at
    least one trainable annotation."""

    def __init__(self, config: PixMoPointsV2DatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._data = _load_split(config.dataset_path, "train")
        self._index = self._build_index()
        log.info(
            "PixMoPointsV2 (%s): %d of %d images have a trainable annotation",
            config.dataset_path,
            len(self._index),
            len(self._data),
        )

    # -- selection -------------------------------------------------------------------------

    def keep(self, anno: Dict[str, Any]) -> bool:
        """Whether to train on this annotation (mm_olmo ``PixMoPointV2._keep``, pointing leg)."""
        cfg = self.config
        label = anno.get("label")
        if not label or not label.strip():
            return False
        if cfg.filter_audit and _failed_audit(anno.get("audit_result")):
            return False
        return cfg.min_points <= len(anno["points"]) <= cfg.max_points

    def _build_index(self) -> np.ndarray:
        """Row indices with ``kind``'s source and at least one annotation passing :meth:`keep`
        -- mm_olmo's build-time ``ds.filter``, computed in memory instead of as an Arrow cache
        file next to the (shared) data."""
        import pyarrow as pa
        import pyarrow.compute as pc

        cfg = self.config
        table = self._data.data

        def _keep_flat(flat) -> np.ndarray:
            n_points = pc.list_value_length(flat.field("points")).to_numpy(zero_copy_only=False)
            n_points = np.nan_to_num(n_points.astype(np.float64), nan=-1)
            has_label = _bool_np(
                pc.invert(pc.equal(pc.utf8_trim_whitespace(flat.field("label")), ""))
            )
            keep = has_label & (n_points >= cfg.min_points) & (n_points <= cfg.max_points)
            if cfg.filter_audit:
                failed = _bool_np(
                    pc.is_in(
                        flat.field("audit_result"),
                        value_set=pa.array(sorted(FAILED_AUDIT_RESULTS)),
                    )
                )
                keep &= ~failed
            return keep

        rows = _rows_with_any(table.column("annotations"), _keep_flat)
        source = _KIND_TO_SOURCE[cfg.kind]
        if source is not None:
            rows &= _bool_np(pc.equal(table.column("source"), source))
        return np.flatnonzero(rows)

    def __len__(self) -> int:
        return len(self._index)

    # -- formatting ------------------------------------------------------------------------

    def format_row(
        self, row: Dict[str, Any], rng: np.random.RandomState
    ) -> Tuple[List[Dict[str, Any]], List[Optional[float]]]:
        """One image's branch messages and per-branch loss weights (mm_olmo
        ``PixMoPointV2.format_example``, in its draw order: annotation styles, then the hard,
        easy and paired negative samples, then the negatives' styles).

        Each message is a formatter sub-example: ``style``, ``label``, ``points`` (``(N, 2)``
        in 0-100 percent coordinates, clipped) -- ``N == 0`` renders ``"There are none."``.
        """
        cfg = self.config
        messages: List[Dict[str, Any]] = []
        weights: List[Optional[float]] = []
        negatives: List[str] = []

        for anno in row["annotations"]:
            if not self.keep(anno):
                continue
            label = anno["label"].strip()
            raw = anno["points"]
            if len(raw) == 0:
                negatives.append(label)  # an annotated absence
                continue
            # Rows carry (x, y, depth) per point; only x, y are rendered.
            points = np.asarray(raw, dtype=np.float64).reshape(len(raw), -1)[:, :2]
            if cfg.audit_style and _failed_audit(anno.get("audit_result")):
                style = _choose(rng, cfg.audit_style)
            else:
                style = _choose(rng, cfg.style)
            messages.append(
                dict(style=style, label=label, points=points, point_scale=100, clip_points=True)
            )
            weights.append(None)

        easy = [str(x) for x in row.get("easy_negatives") or []]
        if cfg.n_hard_negatives > 0:
            pool = easy + [str(x) for x in row.get("paired_negatives") or []]
            if len(pool) > cfg.n_hard_negatives:
                pool = [str(x) for x in rng.choice(pool, size=cfg.n_hard_negatives, replace=False)]
            negatives += pool
        if cfg.n_easy_samples > 0:
            sample = easy
            if len(sample) > cfg.n_easy_samples:
                sample = [
                    str(x) for x in rng.choice(sample, size=cfg.n_easy_samples, replace=False)
                ]
            negatives += sample
        if cfg.p_paired_negatives > 0:
            key = "paired_negatives_v2" if cfg.v2_paired_negatives else "paired_negatives"
            paired = [str(x) for x in row.get(key) or []]
            expected = len(paired) * cfg.p_paired_negatives
            n_paired = int(expected) + int((expected - int(expected)) > rng.random())
            if len(paired) > n_paired:
                paired = [str(x) for x in rng.choice(paired, size=n_paired, replace=False)]
            negatives += paired

        for label in negatives:
            messages.append(
                dict(
                    style=_choose(rng, cfg.style),
                    label=label,
                    points=np.zeros((0, 2), dtype=np.float64),
                    point_scale=100,
                    clip_points=True,
                )
            )
            weights.append(1.0 if label in easy else cfg.negative_weight)

        if not messages:
            raise ValueError("PixMoPointsV2 row has no trainable annotation")
        return messages, weights

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._data[int(self._index[i])]
        rng = example_rng(cfg.seed, i)
        messages, weights = self.format_row(row, rng)
        fmt = SftFormatter(
            seed=cfg.seed, prompt_templates=cfg.prompt_templates, system_prompt=cfg.system_prompt
        )
        branches = [fmt.format_turns(msg, index=i, rng=rng)[0] for msg in messages]
        return _build_example(
            self.tokenizer,
            _open_image(row["image"]),
            branches,
            max_crops=cfg.max_crops,
            loss_token_weighting=cfg.loss_token_weighting,
            message_weight=cfg.message_weight,
            p_high_res=cfg.p_high_res,
            shuffle_rng=rng,
            branch_weights=weights,
        )


# ---------------------------------------------------------------------------
# PixMo count v2: audited PixMo-Count grouped by image
# ---------------------------------------------------------------------------


@dataclass
class PixMoCountV2DatasetConfig(Config):
    """mm_olmo ``PixMoCountConfigV2``: PixMo-Count re-grouped by image with a VLM audit per
    point set. Points are pixel coordinates (normalised by the image size, as in the v1
    ``count`` build); unlike v1 the two styles are drawn per annotation rather than doubling
    the dataset.
    """

    dataset_path: str = f"{PIXMO_DATASETS}/count-v2"
    """HF ``DatasetDict`` saved with ``save_to_disk`` (``train`` / ``validation`` / ``test``)."""

    split: str = "train"

    style: Tuple[str, ...] = ("point_count", "pointing")
    """Styles drawn uniformly per annotation (mm_olmo ``style="both"``)."""

    filter_audit: bool = False
    """Drop point sets whose ``audit_result`` is ``error`` / ``clear_error``."""

    audit_style: Optional[Tuple[str, ...]] = None
    """Styles drawn for audit-failed point sets that are kept, e.g.
    ``("aux_point_count", "aux_pointing")``."""

    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments"
    message_weight: Optional[float] = None
    p_high_res: float = 0.0
    seed: int = 0
    prompt_templates: str = "uber_model_v2"
    """Prompt family for the question text; stage 1 uses ``"none"`` (bare label)."""
    system_prompt: str = "demo_or_style_v2"
    """Prompt family for the style prefix; stage 1 uses ``"style_and_length_v2"``."""

    def validate(self):
        if not self.style:
            raise OLMoConfigurationError("style must name at least one style")
        if self.audit_style is not None and not self.audit_style:
            raise OLMoConfigurationError("audit_style must be None or name at least one style")

    def build(self, tokenizer) -> "PixMoCountV2Dataset":
        self.validate()
        return PixMoCountV2Dataset(self, tokenizer)


class PixMoCountV2Dataset:
    """Map-style dataset over the images of :class:`PixMoCountV2DatasetConfig` that keep at
    least one point set."""

    def __init__(self, config: PixMoCountV2DatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._data = _load_split(config.dataset_path, config.split)
        self._index = self._build_index()
        log.info(
            "PixMoCountV2 (%s/%s): %d of %d images have a trainable point set",
            config.dataset_path,
            config.split,
            len(self._index),
            len(self._data),
        )

    def keep(self, anno: Dict[str, Any]) -> bool:
        """Whether to train on this point set (mm_olmo ``PixMoCountConfigV2``)."""
        return not (self.config.filter_audit and _failed_audit(anno.get("audit_result")))

    def _build_index(self) -> np.ndarray:
        """Rows with at least one kept point set (mm_olmo's ``filter_audit`` build-time
        filter; without it every row qualifies)."""
        import pyarrow as pa
        import pyarrow.compute as pc

        def _keep_flat(flat) -> np.ndarray:
            keep = np.ones(len(flat), dtype=bool)
            if self.config.filter_audit:
                keep &= ~_bool_np(
                    pc.is_in(
                        flat.field("audit_result"),
                        value_set=pa.array(sorted(FAILED_AUDIT_RESULTS)),
                    )
                )
            return keep

        return np.flatnonzero(_rows_with_any(self._data.data.column("points"), _keep_flat))

    def __len__(self) -> int:
        return len(self._index)

    def format_row(
        self, row: Dict[str, Any], rng: np.random.RandomState, image_size: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """One image's branch messages (mm_olmo ``PixMoCountConfigV2.format_example``)."""
        cfg = self.config
        messages: List[Dict[str, Any]] = []
        for anno in row["points"]:
            if not self.keep(anno):
                continue
            if cfg.audit_style and _failed_audit(anno.get("audit_result")):
                style = _choose(rng, cfg.audit_style)
            else:
                style = _choose(rng, cfg.style)
            xy = np.asarray(anno["points"], dtype=np.float64).reshape(-1, 2)
            messages.append(
                dict(
                    style=style,
                    label=anno["label"],
                    points=xy,
                    point_scale=None,
                    image_size=image_size,
                )
            )
        if not messages:
            raise ValueError("PixMoCountV2 row has no trainable point set")
        return messages

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._data[int(self._index[i])]
        rng = example_rng(cfg.seed, i)
        pil = _open_image(row["image"])
        messages = self.format_row(row, rng, pil.size)
        fmt = SftFormatter(
            seed=cfg.seed, prompt_templates=cfg.prompt_templates, system_prompt=cfg.system_prompt
        )
        branches = [fmt.format_turns(msg, index=i, rng=rng)[0] for msg in messages]
        return _build_example(
            self.tokenizer,
            pil,
            branches,
            max_crops=cfg.max_crops,
            loss_token_weighting=cfg.loss_token_weighting,
            message_weight=cfg.message_weight,
            p_high_res=cfg.p_high_res,
            shuffle_rng=rng,
        )
