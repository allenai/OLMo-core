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

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from olmo_core.config import Config

from .grounding import (
    POINT_COUNT_PROMPTS,
    POINTING_PROMPTS,
    normalize_points,
    pointing_answer,
)
from .sequence_builder import build_branched_sequence

__all__ = [
    "PixMoPointsDatasetConfig",
    "PixMoPointsDataset",
    "PixMoCountDatasetConfig",
    "PixMoCountDataset",
    "CoSynPointDatasetConfig",
    "CoSynPointDataset",
]

_B = "/weka/oe-training-default/mm-olmo/torch_datasets/pixmo_datasets"


def _build_user_turn(tokenizer, question: str) -> List[int]:
    """Encode a single user turn + assistant header (no BOS/image): the non-loss context
    of a branch. Matches the molmo2 chat layout for a turn after the shared image prefix."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(text, add_special_tokens=False)


def _image_prefix(tokenizer, image_grid) -> List[int]:
    """BOS + expanded image-token block (the shared prefix for a pointing example)."""
    from olmo_core.nn.vision.molmo2_tokens import build_image_token_ids

    resized_h, resized_w, h, w = (int(image_grid[i]) for i in range(4))
    bos = tokenizer.bos_token_id or tokenizer.eos_token_id
    return [bos] + build_image_token_ids(resized_h, resized_w, h, w)


def _build_example(
    tokenizer,
    pil_image,
    branches_text: List[Tuple[str, str]],
    *,
    max_crops: int,
    loss_token_weighting: str,
) -> Dict[str, np.ndarray]:
    """Preprocess the image and assemble a (possibly multi-branch) pointing example.

    :param branches_text: list of ``(user_question, assistant_answer)`` strings.
    """
    import torch

    from olmo_core.nn.vision.molmo2_image_processor import preprocess_image_molmo2

    images_t, pooling_t, image_grid = preprocess_image_molmo2(
        pil_image, dtype=torch.float32, device=torch.device("cpu"), max_crops=max_crops
    )
    prefix = _image_prefix(tokenizer, image_grid)
    branches = [
        (_build_user_turn(tokenizer, q), tokenizer.encode(a, add_special_tokens=False))
        for q, a in branches_text
    ]
    seq = build_branched_sequence(
        prefix,
        branches,
        eos_id=tokenizer.eos_token_id,
        loss_token_weighting=loss_token_weighting,
    )
    seq["images"] = images_t[0].numpy()
    seq["pooled_patches_idx"] = pooling_t[0].numpy()
    return seq


def _load_split(path: str, split: str):
    from datasets import load_from_disk

    ds = load_from_disk(path)
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
    counting: str = "both"  # "both" -> random pointing/point_count per branch
    max_points: int = 60
    max_total_points_per_example: int = 60
    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments"
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

        self._data = concatenate_datasets([_load_split(f"{_B}/{s}", "train") for s in sub])
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
        return len(self._index)

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        row_idx, label_idxs = self._index[i]
        rng = np.random.RandomState(self.config.seed + i)
        row = self._data[row_idx]
        branches: List[Tuple[str, str]] = []
        for li in label_idxs:
            label = row["label"][li]
            pts = row["points"][li]
            xy = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float64).reshape(-1, 2)
            norm = normalize_points(xy, point_scale=100, image_size=None)
            if self.config.counting == "both":
                style = rng.choice(["point_count", "pointing"])
            else:
                style = "point_count" if self.config.counting else "pointing"
            pool = POINT_COUNT_PROMPTS if style == "point_count" else POINTING_PROMPTS
            prompt = pool[rng.randint(len(pool))].format(label=label)
            answer = pointing_answer(norm, label.lower(), style, count=len(norm))
            branches.append((prompt, answer))
        return _build_example(
            self.tokenizer,
            _open_image(row["image"]),
            branches,
            max_crops=self.config.max_crops,
            loss_token_weighting=self.config.loss_token_weighting,
        )


# ---------------------------------------------------------------------------
# PixMo count (single annotation, alternating point_count / pointing)
# ---------------------------------------------------------------------------


@dataclass
class PixMoCountDatasetConfig(Config):
    counting: str = "both"  # "both" interleaves point_count (even) / pointing (odd)
    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments"
    seed: int = 0

    def build(self, tokenizer) -> "PixMoCountDataset":
        return PixMoCountDataset(self, tokenizer)


class PixMoCountDataset:
    def __init__(self, config: PixMoCountDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._data = _load_split(f"{_B}/count", "train")
        self._n = len(self._data)

    def __len__(self) -> int:
        return self._n * 2 if self.config.counting == "both" else self._n

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        if self.config.counting == "both":
            row_idx, style = i // 2, ("point_count" if i % 2 == 0 else "pointing")
        else:
            row_idx, style = i, ("point_count" if self.config.counting else "pointing")
        row = self._data[row_idx]
        label = row["label"]
        count = int(row["count"])
        pil = _open_image(row["image"])
        pts = row.get("points") or {"x": [], "y": []}
        xy = np.array([pts["x"], pts["y"]], dtype=np.float64).T.reshape(-1, 2)
        norm = normalize_points(xy, point_scale=None, image_size=pil.size)  # pixel -> /(w,h)
        pool = POINT_COUNT_PROMPTS if style == "point_count" else POINTING_PROMPTS
        rng = np.random.RandomState(self.config.seed + i)
        prompt = pool[rng.randint(len(pool))].format(label=label)
        answer = pointing_answer(norm, label.lower(), style, count=count)
        return _build_example(
            self.tokenizer,
            pil,
            [(prompt, answer)],
            max_crops=self.config.max_crops,
            loss_token_weighting=self.config.loss_token_weighting,
        )


# ---------------------------------------------------------------------------
# CoSyn point (document pointing; multi-branch, prompt = the question)
# ---------------------------------------------------------------------------


@dataclass
class CoSynPointDatasetConfig(Config):
    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments"
    seed: int = 0

    def build(self, tokenizer) -> "CoSynPointDataset":
        return CoSynPointDataset(self, tokenizer)


class CoSynPointDataset:
    def __init__(self, config: CoSynPointDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._data = _load_split(f"{_B}/cosyn-point", "train")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        row = self._data[i]
        branches: List[Tuple[str, str]] = []
        for question, points, name in zip(row["questions"], row["answer_points"], row["names"]):
            xy = np.array([points["x"], points["y"]], dtype=np.float64).T.reshape(-1, 2)
            norm = normalize_points(xy, point_scale=100, image_size=None)
            # cosyn_point uses the "pointing" answer (just the points tag), label = name.
            answer = pointing_answer(norm, name, "pointing", count=len(norm))
            branches.append((question, answer))
        return _build_example(
            self.tokenizer,
            _open_image(row["image"]),
            branches,
            max_crops=self.config.max_crops,
            loss_token_weighting=self.config.loss_token_weighting,
        )
