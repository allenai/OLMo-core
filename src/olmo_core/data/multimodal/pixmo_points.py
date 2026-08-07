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

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds

from .grounding import normalize_points, pointing_answer
from .document_layout import branch_context_ids, image_prefix_ids, response_ids
from .rng import make_random_state
from .sequence_builder import build_branched_sequence
from .sft_formatter import SftFormatter

__all__ = [
    "PixMoPointsDatasetConfig",
    "PixMoPointsDataset",
    "PixMoCountDatasetConfig",
    "PixMoCountDataset",
    "CoSynPointDatasetConfig",
    "CoSynPointDataset",
]

from .paths import PIXMO_DATASETS


def _build_example(
    tokenizer,
    pil_image,
    build_branches: Callable[[np.random.RandomState], List[Tuple[str, str]]],
    *,
    max_crops: int,
    loss_token_weighting: str,
    token_ids: Molmo2TokenIds,
    message_weight: float | None = None,
    p_high_res: float = 0.0,
    rng: np.random.RandomState,
) -> Dict[str, np.ndarray]:
    """Format and assemble a (possibly multi-branch) pointing example.

    :param build_branches: Builds ``(user_question, assistant_answer)`` strings before image
        augmentation, preserving Molmo2's random-number consumption order.
    """
    import torch

    from olmo_core.nn.vision.molmo2_image_processor import preprocess_image_molmo2

    branches_text = list(build_branches(rng))
    if len(branches_text) > 1:
        order = np.arange(len(branches_text))
        rng.shuffle(order)
        branches_text = [branches_text[i] for i in order]

    images_t, pooling_t, image_grid = preprocess_image_molmo2(
        pil_image,
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_crops=max_crops,
        p_high_res=p_high_res,
        is_training=True,
        rng=rng,
    )
    prefix = image_prefix_ids(tokenizer, image_grid, token_ids=token_ids)
    branches = [
        (
            branch_context_ids(tokenizer, question),
            response_ids(tokenizer, answer),
        )
        for question, answer in branches_text
    ]
    from olmo_core.data.multimodal.message_weight import apply_message_weight_to_loss_masks

    seq = build_branched_sequence(
        prefix,
        branches,
        eos_id=tokenizer.eos_token_id,
        image_token_ids=token_ids.image_token_ids,
        loss_token_weighting=loss_token_weighting,
    )
    subsegment_ids = seq.get("subsegment_ids")
    from olmo_core.data.multimodal.message_weight import MessageWeight

    mw = MessageWeight.from_string(loss_token_weighting).with_overrides(message_weight)
    seq["loss_masks"] = apply_message_weight_to_loss_masks(
        seq["loss_masks"], subsegment_ids, mw, branch_scaling_already_applied=True
    )
    seq["images"] = images_t[0].numpy()
    seq["pooled_patches_idx"] = pooling_t[0].numpy()
    return seq


def _load_split(path: str, split: str):
    from .dataset_compat import load_from_disk_compat

    ds = load_from_disk_compat(path)
    return ds[split] if hasattr(ds, "keys") and split in ds else ds


def _open_image(p):
    from PIL import Image

    return p if isinstance(p, Image.Image) else Image.open(p)


# ---------------------------------------------------------------------------
# PixMo points (pointing / counting)
# ---------------------------------------------------------------------------


@dataclass
class PixMoPointsDatasetConfig(Config):
    """``pixmo_points_train`` (kind=basic) / ``pixmo_points_high_freq_train`` (high_frequency)."""

    kind: str = "both"  # "basic" (points-pointing) | "high_frequency" (points-counting) | "both"
    counting: str = "both"  # "both" duplicates each example in point_count/pointing styles
    max_points: int = 60
    max_total_points_per_example: int = 60
    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments"
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    message_weight: float | None = None
    p_high_res: float = 0.0
    seed: int = 0

    def build(self, tokenizer) -> "PixMoPointsDataset":
        return PixMoPointsDataset(self, tokenizer)


class PixMoPointsDataset:
    def __init__(self, config: PixMoPointsDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        sub = {"basic": ["points-pointing"], "high_frequency": ["points-counting"]}.get(
            config.kind, ["points-counting", "points-pointing"]
        )
        from datasets import concatenate_datasets

        self._data = concatenate_datasets(
            [_load_split(f"{PIXMO_DATASETS}/{s}", "train") for s in sub]
        )
        # Pre-split each row's labels into sub-batches with <= max_total_points (mm_olmo).
        self._index = self._build_sub_index()

    def _build_sub_index(self) -> List[Tuple[int, List[int]]]:
        cfg = self.config
        counts = self._data["count"]
        index: List[Tuple[int, List[int]]] = []
        for row, point_counts in enumerate(counts):
            on: List[int] = []
            total = 0
            for li, n in enumerate(point_counts):
                if n > cfg.max_points:
                    continue
                if on and total + n > cfg.max_total_points_per_example:
                    index.append((row, on))
                    on, total = [], 0
                on.append(li)
                total += n
            if on:
                index.append((row, on))
        return index

    def __len__(self) -> int:
        size = len(self._index)
        return size * 2 if self.config.counting == "both" else size

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        return self.get(i, 0)

    def get(self, i: int, epoch: int = 0) -> Dict[str, np.ndarray]:
        """Build one deterministically augmented example for a source epoch."""
        if self.config.counting == "both":
            example_idx = i // 2
            style = "point_count" if i % 2 == 0 else "pointing"
        else:
            example_idx = i
            style = "point_count" if self.config.counting else "pointing"
        row_idx, label_idxs = self._index[example_idx]
        rng = make_random_state(self.config.seed + i, epoch)
        row = self._data[row_idx]
        fmt = SftFormatter(seed=self.config.seed)
        specs: List[Tuple[str, str, Any]] = []
        for li in label_idxs:
            label = row["label"][li]
            pts = row["points"][li]
            specs.append((style, label, pts))

        def build_branches(branch_rng: np.random.RandomState) -> List[Tuple[str, str]]:
            branches: List[Tuple[str, str]] = []
            for branch_style, label, points in specs:
                sub = {
                    "style": branch_style,
                    "label": label,
                    "points": points,
                    "point_scale": 100,
                }
                prompt, answer = fmt.format_turns(sub, index=i, rng=branch_rng)[0]
                branches.append((f"{branch_style}: {prompt}", answer))
            return branches

        return _build_example(
            self.tokenizer,
            _open_image(row["image"]),
            build_branches,
            max_crops=self.config.max_crops,
            loss_token_weighting=self.config.loss_token_weighting,
            token_ids=self.config.token_ids,
            message_weight=self.config.message_weight,
            p_high_res=self.config.p_high_res,
            rng=rng,
        )


# ---------------------------------------------------------------------------
# PixMo count (single annotation, alternating point_count / pointing)
# ---------------------------------------------------------------------------


@dataclass
class PixMoCountDatasetConfig(Config):
    counting: str = "both"  # "both" interleaves point_count (even) / pointing (odd)
    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments"
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    message_weight: float | None = None
    p_high_res: float = 0.0
    seed: int = 0

    def build(self, tokenizer) -> "PixMoCountDataset":
        return PixMoCountDataset(self, tokenizer)


class PixMoCountDataset:
    def __init__(self, config: PixMoCountDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._data = _load_split(f"{PIXMO_DATASETS}/count", "train")
        self._n = len(self._data)

    def __len__(self) -> int:
        return self._n * 2 if self.config.counting == "both" else self._n

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        return self.get(i, 0)

    def get(self, i: int, epoch: int = 0) -> Dict[str, np.ndarray]:
        """Build one deterministically augmented example for a source epoch."""
        if self.config.counting == "both":
            row_idx, style = i // 2, ("point_count" if i % 2 == 0 else "pointing")
        else:
            row_idx, style = i, ("point_count" if self.config.counting else "pointing")
        row = self._data[row_idx]
        label = row["label"]
        count = int(row["count"])
        pil = _open_image(row["image"])
        pts = row.get("points") or {"x": [], "y": []}
        rng = make_random_state(self.config.seed + i, epoch)
        fmt = SftFormatter(seed=self.config.seed)
        xy = np.array([pts["x"], pts["y"]], dtype=np.float64).T.reshape(-1, 2)
        sub = {
            "style": style,
            "label": label,
            "points": xy,
            "point_scale": None,
            "image_size": pil.size,
            "count": count,
        }

        def build_branches(branch_rng: np.random.RandomState) -> List[Tuple[str, str]]:
            prompt, answer = fmt.format_turns(sub, index=i, rng=branch_rng)[0]
            return [(f"{style}: {prompt}", answer)]

        return _build_example(
            self.tokenizer,
            pil,
            build_branches,
            max_crops=self.config.max_crops,
            loss_token_weighting=self.config.loss_token_weighting,
            token_ids=self.config.token_ids,
            message_weight=self.config.message_weight,
            p_high_res=self.config.p_high_res,
            rng=rng,
        )


# ---------------------------------------------------------------------------
# CoSyn point (document pointing; multi-branch, prompt = the question)
# ---------------------------------------------------------------------------


@dataclass
class CoSynPointDatasetConfig(Config):
    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments"
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    message_weight: float | None = None
    p_high_res: float = 0.0
    seed: int = 0

    def build(self, tokenizer) -> "CoSynPointDataset":
        return CoSynPointDataset(self, tokenizer)


class CoSynPointDataset:
    def __init__(self, config: CoSynPointDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._data = _load_split(f"{PIXMO_DATASETS}/cosyn-point", "train")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        return self.get(i, 0)

    def get(self, i: int, epoch: int = 0) -> Dict[str, np.ndarray]:
        """Build one deterministically augmented example for a source epoch."""
        row = self._data[i]
        branches: List[Tuple[str, str]] = []
        for question, points, name in zip(row["questions"], row["answer_points"], row["names"]):
            xy = np.array([points["x"], points["y"]], dtype=np.float64).T.reshape(-1, 2)
            norm = normalize_points(xy, point_scale=100, image_size=None)
            # cosyn_point uses the "pointing" answer (just the points tag), label = name.
            answer = pointing_answer(norm, name.lower(), "pointing", count=len(norm))
            branches.append((f"cosyn_point: {question}", answer))
        return _build_example(
            self.tokenizer,
            _open_image(row["image"]),
            lambda branch_rng: branches,
            max_crops=self.config.max_crops,
            loss_token_weighting=self.config.loss_token_weighting,
            token_ids=self.config.token_ids,
            message_weight=self.config.message_weight,
            p_high_res=self.config.p_high_res,
            rng=make_random_state(self.config.seed + i, epoch),
        )
