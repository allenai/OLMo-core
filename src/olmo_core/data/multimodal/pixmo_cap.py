"""PixMoCap caption-pretraining dataset for Molmo2 "stage 1" training.

A dependency-free (no ``mm_olmo``) map-style :class:`torch.utils.data.Dataset` that
turns PixMoCap image-caption examples into packed Molmo2 training sequences. Each
example produces a shared prefix (BOS + image block + user prompt + assistant
header) that branches into one or more assistant responses (a long caption and/or a
spoken transcript), assembled by
:func:`~olmo_core.data.multimodal.sequence_builder.build_packed_sequence`.

Three data sources are supported via ``dataset_path``:

* ``"synthetic"`` — random RGB images + random short responses (for smoke tests).
* a ``.jsonl`` file — one object per line with ``image`` (path or URL),
  ``caption`` (str) and optional ``transcripts`` (list[str]).
* a HuggingFace Arrow directory (``datasets.load_from_disk``) — the canonical
  PixMoCap layout (``image``, ``caption``, ``transcripts``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from olmo_core.config import Config

from .sequence_builder import build_packed_sequence

__all__ = ["PixMoCapDataset", "PixMoCapDatasetConfig", "CAPTION_PROMPTS", "TRANSCRIPT_PROMPTS"]

# Prompt pools mirroring mm_olmo's ``GENERAL_PROMPTS_V1`` (data_formatter.py); one is
# sampled per example (seeded) so the user turn matches the caption-pretraining mix.
CAPTION_PROMPTS = (
    "Describe this image.",
    "Describe this image",
    "Write a long description of this image.",
    "Construct a long caption for this image",
    "Generate a caption",
    "Create a detailed caption",
    "Write a long caption",
    "Describe this image in detail",
)
TRANSCRIPT_PROMPTS = (
    "Describe this image as if you are a person speaking",
    "Imagine you are a person talking about this image. Generate a transcript of what you would say.",
    "Generate an audio transcript of a person describing this image",
    "Create a transcript of a human describing this image out load",
)

_MODES = ("caption", "transcript", "transcript_and_caption")


@dataclass
class PixMoCapDatasetConfig(Config):
    """Configuration for :class:`PixMoCapDataset`."""

    dataset_path: str
    """``"synthetic"``, a ``.jsonl`` file, or a HF Arrow directory."""

    split: str = "train"
    mode: str = "transcript_and_caption"
    """One of ``"caption"``, ``"transcript"``, ``"transcript_and_caption"``."""

    image_root: Optional[str] = None
    """Optional prefix joined to relative image paths from a jsonl source."""

    max_crops: int = 8
    max_sequence_length: int = 5248
    loss_token_weighting: str = "root_subsegments"
    fixed_prompt: Optional[str] = None
    """If set, always use this user prompt instead of sampling from the pools.
    Useful for deterministic parity tests."""

    seed: int = 0
    synthetic_size: int = 64
    """Number of examples to generate when ``dataset_path == "synthetic"``."""

    def build(self, tokenizer) -> "PixMoCapDataset":
        return PixMoCapDataset(self, tokenizer)


class PixMoCapDataset:
    """Map-style dataset yielding packed Molmo2 caption-pretraining examples."""

    def __init__(self, config: PixMoCapDatasetConfig, tokenizer):
        if config.mode not in _MODES:
            raise ValueError(f"Unknown mode {config.mode!r}; expected one of {_MODES}")
        self.config = config
        self.tokenizer = tokenizer
        self._rows: Optional[List[Dict[str, Any]]] = None
        self._hf = None

        path = config.dataset_path
        if path == "synthetic":
            self._kind = "synthetic"
        elif path.endswith(".jsonl"):
            self._kind = "jsonl"
            self._rows = self._load_jsonl(path)
        else:
            self._kind = "arrow"
            from datasets import load_from_disk

            ds = load_from_disk(path)
            self._hf = ds[config.split] if config.split in ds else ds

        self._eos_id = tokenizer.eos_token_id
        self._bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id

    # -- length -----------------------------------------------------------------

    def __len__(self) -> int:
        if self._kind == "synthetic":
            return self.config.synthetic_size
        if self._kind == "jsonl":
            assert self._rows is not None
            return len(self._rows)
        return len(self._hf)

    # -- loading helpers --------------------------------------------------------

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict[str, Any]]:
        import json

        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _get_row(self, index: int) -> Dict[str, Any]:
        if self._kind == "jsonl":
            assert self._rows is not None
            return self._rows[index]
        return self._hf[index]

    def _load_image(self, row: Dict[str, Any]):
        from PIL import Image

        img = row["image"]
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, str):
            path = img
            if self.config.image_root is not None and not img.startswith(
                ("http://", "https://", "/")
            ):
                import os

                path = os.path.join(self.config.image_root, img)
            return Image.open(path)
        raise TypeError(f"Unsupported image field type: {type(img)}")

    # -- core -------------------------------------------------------------------

    def _select_responses(self, row: Dict[str, Any], rng: np.random.RandomState) -> List[str]:
        """Pick the response branch text(s) for this example per ``mode``."""
        caption = row.get("caption", "")
        transcripts = row.get("transcripts") or []
        mode = self.config.mode
        if mode == "caption":
            return [caption]
        if mode == "transcript":
            if not transcripts:
                return [caption]
            return [transcripts[rng.randint(len(transcripts))]]
        # transcript_and_caption: caption first, then a random transcript (if any).
        responses = [caption]
        if transcripts:
            responses.append(transcripts[rng.randint(len(transcripts))])
        return responses

    def _build_prefix_ids(self, prompt: str, image_grid: Optional[np.ndarray]) -> List[int]:
        """BOS + image block + user turn + assistant header (native Molmo2 layout)."""
        from olmo_core.nn.vision.molmo2_tokens import build_image_token_ids

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids: List[int] = self.tokenizer.encode(text, add_special_tokens=False)
        if image_grid is not None:
            resized_h, resized_w, h, w = (int(image_grid[i]) for i in range(4))
            ids = build_image_token_ids(resized_h, resized_w, h, w) + ids
        return [self._bos_id] + ids

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        from olmo_core.nn.vision.molmo2_image_processor import preprocess_image_molmo2

        cfg = self.config
        rng = np.random.RandomState(cfg.seed + index)

        import torch

        if self._kind == "synthetic":
            from PIL import Image

            arr = rng.randint(0, 256, size=(64, 96, 3), dtype=np.uint8)
            pil = Image.fromarray(arr)
            n_words = rng.randint(8, 24)
            caption = " ".join(f"word{rng.randint(1000)}" for _ in range(n_words))
            transcripts = [" ".join(f"tok{rng.randint(1000)}" for _ in range(n_words))]
            row: Dict[str, Any] = {"caption": caption, "transcripts": transcripts}
        else:
            row = self._get_row(index)
            pil = self._load_image(row)

        images_t, pooling_t, image_grid = preprocess_image_molmo2(
            pil, dtype=torch.float32, device=torch.device("cpu"), max_crops=cfg.max_crops
        )
        images = images_t[0].numpy()  # (n_crops, n_patches, patch_dim)
        pooled = pooling_t[0].numpy()  # (n_pool, pool_size)

        if cfg.fixed_prompt is not None:
            prompt = cfg.fixed_prompt
        else:
            pool = TRANSCRIPT_PROMPTS if cfg.mode == "transcript" else CAPTION_PROMPTS
            prompt = pool[rng.randint(len(pool))]

        prefix_ids = self._build_prefix_ids(prompt, image_grid)
        responses = self._select_responses(row, rng)
        response_ids = [self.tokenizer.encode(r, add_special_tokens=False) for r in responses]

        seq = build_packed_sequence(
            prefix_ids,
            response_ids,
            eos_id=self._eos_id,
            loss_token_weighting=cfg.loss_token_weighting,
        )

        # Truncate to max_sequence_length, never cutting an <im_patch> token.
        if len(seq["input_ids"]) > cfg.max_sequence_length:
            seq = _truncate(seq, cfg.max_sequence_length)

        seq["images"] = images
        seq["pooled_patches_idx"] = pooled
        return seq


def _truncate(seq: Dict[str, np.ndarray], max_len: int) -> Dict[str, np.ndarray]:
    """Right-truncate all per-token fields to ``max_len`` (asserting no image token cut)."""
    from olmo_core.nn.vision.molmo2_tokens import IM_PATCH_ID

    keep = max_len
    if np.any(seq["input_ids"][keep:] == IM_PATCH_ID):
        raise ValueError(
            "max_sequence_length too small: truncation would drop <im_patch> tokens "
            "(the image block must fit entirely within the sequence)."
        )
    out = {}
    for k, v in seq.items():
        out[k] = v[:keep] if v.ndim == 1 and len(v) >= keep else v
    return out
