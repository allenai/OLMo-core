"""SynthDoG synthetic-document transcription for Molmo2 stage-1.

``naver-clova-ix/synthdog-en`` is the English half-million-page SynthDoG corpus that Donut was
pretrained on (Kim et al., ECCV 2022): synthetic document pages rendered from Wikipedia text over
scanned-paper backgrounds, paired with the text that was rendered into them. Each parquet row is
an ``image`` plus a ``ground_truth`` JSON string of the shape::

    {"gt_parse": {"text_sequence": "Dares Wins Vol. 5 Tommy's Heroes ..."}}

Only ``text_sequence`` is supervision; the rest of the envelope is Donut's task wrapper.

The prompt is the OCR transcription tag, :data:`~.olmocr.OLMOCR_STYLE` (``"olmocr:"``), the same
one the olmOCR-mix pages use: this is the same task, so it is the same tag.

One property of the corpus to keep in mind when weighting it. The targets carry SynthDoG's
line-wrap artefacts: the generator breaks a word that does not fit at the line end without a
hyphen, and the ground truth then joins the pieces with a space, so a page transcribes as
``"Closin g Time"`` / ``"connectio n"`` / ``"Su perm an 's"``. Measured on 4,000 rows of the first
shard, 99.8% contain at least one such split. That is faithful to what the renderer drew, but it
is not the convention the olmOCR-mix pages use (clean words, markdown tables), so under the shared
tag the model sees both. ``style`` is settable if that turns out to need separating.

Layout on disk (an ``hf download`` of the repo, i.e. ``data/<split>-*.parquet``)::

    <SYNTHDOG_EN>/data/train-00000-of-00084-<hash>.parquet   x84, 500,000 rows
    <SYNTHDOG_EN>/data/validation-00000-of-00001-<hash>.parquet    500 rows

.. note::
    The first load converts the parquet to a memory-mapped Arrow cache under
    ``HF_DATASETS_CACHE`` (:func:`~.sft_common.load_hf_dataset`), which for this corpus is a
    second ~39 GB copy and takes ~85 s. Point ``HF_DATASETS_CACHE`` at shared storage so a job's
    ranks and later runs reuse one cache instead of each rebuilding it.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError

from .message_sequence import encode_sft_example
from .olmocr import OLMOCR_STYLE
from .paths import SYNTHDOG_EN
from .pixmo_cap import STYLE_TAG_FAMILIES, style_tag_prompt
from .sft_common import (
    EpochSeededExamples,
    decode_pil_image,
    get_example_with_skip,
    load_hf_dataset,
    truncate_example,
)

__all__ = [
    "SYNTHDOG_SPLITS",
    "SynthDogDatasetConfig",
    "SynthDogDataset",
    "extract_text_sequence",
]

log = logging.getLogger(__name__)

SYNTHDOG_SPLITS = ("train", "validation")

#: Columns the dataset reads (the repo has no others).
_COLUMNS = ["image", "ground_truth"]


def extract_text_sequence(ground_truth: str) -> str:
    """The supervised text of one row's ``ground_truth`` envelope.

    The whole target is the value of ``text_sequence``; everything around it is Donut's task
    wrapper. The envelope is uniformly ``{"gt_parse": {"text_sequence": ...}}`` (checked on 6,000
    rows spread over 12 shards), and a top-level ``text_sequence`` is accepted as a fallback.

    :param ground_truth: The row's raw JSON string.

    :returns: ``gt_parse.text_sequence``, stripped.

    :raises ValueError: If the envelope does not parse, lacks the expected keys, or the text is
        blank -- all of which leave nothing to train on, so the caller skips the row.
    """
    try:
        parsed = json.loads(ground_truth)
    except (TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"ground_truth is not JSON: {e}") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"ground_truth is not a JSON object but {type(parsed).__name__}")
    envelope = parsed.get("gt_parse")
    text = (
        envelope.get("text_sequence") if isinstance(envelope, dict) else parsed.get("text_sequence")
    )
    if not isinstance(text, str):
        raise ValueError("ground_truth has no gt_parse.text_sequence string")
    text = text.strip()
    if not text:
        raise ValueError("empty text_sequence")
    return text


@dataclass
class SynthDogDatasetConfig(Config):
    """One split of the SynthDoG parquet build: transcribe a synthetic document page."""

    dataset_path: str = SYNTHDOG_EN
    """Repo root holding ``data/<split>-*.parquet`` (an ``hf download`` of the dataset repo)."""

    split: str = "train"
    """``train`` (500,000 pages) or ``validation`` (500)."""

    style: str = OLMOCR_STYLE
    """Style tag shown in the user turn: the OCR transcription tag, shared with the olmOCR-mix
    pages because it is the same task. Settable if this corpus's line-wrap artefacts (see the
    module docstring) ever need their own tag."""

    max_crops: int = 8
    max_sequence_length: Optional[int] = None
    """Tail-truncate the built sequence to this many tokens; set it to the training sequence
    length. Pages are short (median 286 characters of target text), so this rarely binds."""

    loss_token_weighting: str = "root_subsegments"
    message_weight: Optional[float] = None
    """Scalar loss multiplier for this source."""

    seed: int = 0

    system_prompt: str = "style_and_length_v3"
    """Prefix family for the style tag (:func:`~.pixmo_cap.style_tag_prompt`); the default renders
    the bare ``"<style>:"``, as the rest of the OCR group does."""

    def validate(self):
        if not self.dataset_path:
            raise OLMoConfigurationError("dataset_path must point at a SynthDoG repo root")
        if self.split not in SYNTHDOG_SPLITS:
            raise OLMoConfigurationError(
                f"split={self.split!r} is not one of {list(SYNTHDOG_SPLITS)}"
            )
        if not self.style:
            raise OLMoConfigurationError("style must be a non-empty style name")
        if self.system_prompt not in STYLE_TAG_FAMILIES:
            raise OLMoConfigurationError(
                f"system_prompt={self.system_prompt!r} is not one of {sorted(STYLE_TAG_FAMILIES)}"
            )

    def build(self, tokenizer) -> "SynthDogDataset":
        self.validate()
        return SynthDogDataset(self, tokenizer)


class SynthDogDataset(EpochSeededExamples):
    """Map-style dataset over one SynthDoG split."""

    def __init__(self, config: SynthDogDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        # One glob per split: the shards share a `data/` directory, so globbing `*.parquet`
        # would silently pool train with validation.
        self.shard_glob = os.path.join(config.dataset_path, "data", f"{config.split}-*.parquet")
        self._data = load_hf_dataset(self.shard_glob, split="train", keep_columns=_COLUMNS)
        self._warned = 0
        log.info("SynthDoG %s/%s: %d pages", config.dataset_path, config.split, len(self._data))

    def __len__(self) -> int:
        return len(self._data)

    @staticmethod
    def transcription(row: Dict[str, Any]) -> str:
        """This page's target text (:func:`extract_text_sequence`)."""
        return extract_text_sequence(row["ground_truth"])

    def user_prompt(self, text: str, rng: np.random.RandomState) -> str:
        """The user turn: only the style tag, as for the other transcription sources."""
        return style_tag_prompt(self.config.style, text, rng, self.config.system_prompt)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Build page ``index``, deterministically skipping unusable rows.

        A row whose envelope does not parse, whose image will not decode, or whose truncation
        leaves no loss tokens must not raise out of here: it would spend the mixture loader's
        error budget and a run of them would abort training. Same policy as the other OCR
        sources; see :func:`~olmo_core.data.multimodal.sft_common.get_example_with_skip`.
        """
        return get_example_with_skip(self, index, len(self))

    def _build(self, i: int) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._data[i]
        text = self.transcription(row)
        image = decode_pil_image(row["image"]).convert("RGB")
        rng = self.epoch_rng(i)
        prompt = self.user_prompt(text, rng)
        seq = encode_sft_example(
            self.tokenizer,
            image,
            [(prompt, text)],
            max_crops=cfg.max_crops,
            loss_token_weighting=cfg.loss_token_weighting,
            message_weight=cfg.message_weight,
            shuffle_rng=rng,
        )
        if cfg.max_sequence_length is not None:
            seq = truncate_example(seq, cfg.max_sequence_length)
        return seq
