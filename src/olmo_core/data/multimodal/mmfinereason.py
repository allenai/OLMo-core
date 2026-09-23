"""MMFineReason-SFT multimodal reasoning SFT dataset.

Reads ``OpenDataArena/MMFineReason-SFT-586K-Qwen3-VL-235B-Thinking`` (and its siblings, e.g.
the 123K subset or the unfiltered 1.8M parent) — the difficulty-filtered "hardest 33%" of
MMFineReason, where a 4B thinking model fails at least once out of four attempts, annotated
with long-form chain-of-thought from Qwen3-VL-235B-A22B-Thinking.

Each row is one image + one question + one answer, so every example is a single-turn
conversation over an image prefix, assembled by
:func:`~olmo_core.data.multimodal.message_sequence.encode_sft_example` (the same qwen3
layout as every other stage-2 source: no BOS, image block inside the user turn).

**Supervision target.** The answer column (``original_answer`` by default) comes in two
shapes, roughly half the corpus each:

* ``<think>…</think><answer>…</answer>`` — a private reasoning trace followed by the
  user-facing answer. Only the ``<answer>`` content is supervised; the trace is dropped
  (it is ~2/3 of the raw characters, and training on it would teach the model to emit
  reasoning it was not asked for).
* plain prose with no tags — supervised in full.

See :func:`extract_answer_text` for the exact rule, including malformed-tag fallbacks.

.. note::
    ``question`` carries an inline ``<image>`` marker (always exactly one). It is stripped
    and the image is supplied as an explicit token block instead — see
    :func:`~olmo_core.data.multimodal.sft_common.strip_image_placeholders`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from olmo_core.config import Config

from .message_sequence import encode_sft_example
from .sequence_builder import example_rng
from .sft_common import (
    decode_pil_image,
    extract_reasoning_scratchpad,
    extract_reasoning_text,
    get_example_with_skip,
    load_hf_dataset,
    strip_image_placeholders,
    truncate_example,
)

__all__ = ["MMFineReasonDatasetConfig", "MMFineReasonDataset", "extract_answer_text"]

log = logging.getLogger(__name__)

DATA_PATH = (
    "/weka/oe-training-default/mm-olmo/hf_datasets/"
    "OpenDataArena___MMFineReason-SFT-586K-Qwen3-VL-235B-Thinking/data"
)
"""Default location of the parquet shards on weka."""

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_STRAY_TAGS_RE = re.compile(r"</?(?:think|answer)>")


def extract_answer_text(raw: Optional[str]) -> str:
    """Extract the user-facing supervision target from a raw MMFineReason answer.

    The rule, in order:

    1. If one or more well-formed ``<answer>…</answer>`` blocks are present, return the
       contents of the **last** one. (Last, not first, so a reasoning trace that merely
       *mentions* ``<answer>`` cannot hijack the result.)
    2. Otherwise drop any ``<think>…</think>`` block and return the remaining text — this is
       the common no-tag case, where the whole answer is the target.
    3. Any leftover unpaired ``<think>`` / ``<answer>`` tag is removed so stray markup never
       reaches the loss.

    :param raw: The raw answer text (``None`` / empty is allowed).

    :returns: The supervision text, stripped. Empty if nothing survives, which the dataset
        treats as an unusable row.
    """
    if not raw:
        return ""
    blocks = _ANSWER_RE.findall(raw)
    for block in reversed(blocks):
        text = block.strip()
        if text:
            return _STRAY_TAGS_RE.sub("", text).strip()
    cleaned = _THINK_RE.sub("", raw)
    return _STRAY_TAGS_RE.sub("", cleaned).strip()


@dataclass
class MMFineReasonDatasetConfig(Config):
    """Configuration for :class:`MMFineReasonDataset`."""

    dataset_path: str = DATA_PATH
    """Parquet shard directory / glob / file, or a ``save_to_disk`` Arrow directory."""

    split: str = "train"

    question_column: str = "question"
    """Column holding the prompt."""

    answer_column: str = "original_answer"
    """Column holding the supervision target, parsed by :func:`extract_answer_text`."""

    image_column: str = "image"

    sources: Optional[List[str]] = None
    """If set, keep only rows whose ``source`` is in this list (the corpus mixes MMR1,
    GameQA, BMMR, LLaVA-CoT, several FineVision subsets, ...). ``None`` keeps everything."""

    max_pass_rate: Optional[float] = None
    """If set, keep only rows with ``pass_rate <=`` this (smaller = harder)."""

    require_consistent: Optional[bool] = None
    """If set, keep only rows whose ``is_consistent`` flag matches."""

    supervise_cot: bool = False
    """Supervise the ``<think>`` derivation instead of only the ``<answer>`` content.

    The trace is already in :attr:`answer_column`, so this costs no staging -- unlike
    ChartVerse, whose derivation had to be dropped to keep the staged corpus to 1.19 TB.
    That makes this the cheapest available test of whether process supervision helps at
    all, against a control curve (the answer-only ``mmfinereason`` 10k diet) that differs
    in nothing but this flag.

    Only ~49% of rows carry a ``<think>`` block; the rest fall back to the answer text, so
    the realized mixture is roughly half derivations. Traces average ~1.1k tokens (p99
    ~3.4k), about a quarter of ChartVerse's, so the packing cost is correspondingly milder.
    """

    max_crops: int = 8
    max_sequence_length: int = 4096
    loss_token_weighting: str = "root_subsegments"
    seed: int = 0

    short_answer_column: str = "answer"
    """Column holding the short verifiable answer (p50 **4 characters**), as opposed to
    :attr:`answer_column`'s verbose ``original_answer`` (p50 1,516). Used only by
    :attr:`cot_scratchpad`, which needs a *committed* answer short enough to be graded on
    its own -- CharXiv descriptive gold is p50 5 characters, and Qwen-Thinking's graded
    span is p50 24 against a p50 639 trace. Falls back to the parsed verbose answer when
    this column is missing or empty (~4% of rows)."""

    cot_scratchpad: bool = False
    """With :attr:`supervise_cot`, keep the trace inside ``<think>…</think>`` and follow it
    with the bare answer, instead of stripping the tags and supervising the derivation as
    prose. Only meaningful paired with olmo-eval's ``strip_reasoning_trace``, which removes
    the scratchpad before grading; without that the tags reach the judge. Rows with no
    ``<think>`` block (~51%) fall back to the answer-only target either way."""

    skip_overlong: bool = False
    """Skip over-budget rows instead of right-truncating them (see
    :attr:`~olmo_core.data.multimodal.chartverse.ChartVerseDatasetConfig.skip_overlong`).
    Defaults on with :attr:`supervise_cot`."""

    def __post_init__(self):
        if self.supervise_cot:
            self.skip_overlong = True

    def build(self, tokenizer) -> "MMFineReasonDataset":
        """Construct the dataset.

        :param tokenizer: A Molmo2 chat tokenizer.
        """
        return MMFineReasonDataset(self, tokenizer)


class MMFineReasonDataset:
    """Map-style dataset yielding packed MMFineReason SFT examples."""

    def __init__(self, config: MMFineReasonDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        needed = [config.question_column, config.answer_column, config.image_column]
        if config.cot_scratchpad and config.short_answer_column:
            needed.append(config.short_answer_column)
        filter_cols = ["source", "pass_rate", "is_consistent"]
        self._data = load_hf_dataset(
            config.dataset_path,
            config.split,
            keep_columns=needed + filter_cols,
        )
        self._index = self._build_index()
        self._warned = 0

    def _build_index(self) -> Optional[np.ndarray]:
        """Row positions kept after the ``sources`` / ``pass_rate`` / consistency filters.

        Returns ``None`` when no filter is active (so ``__getitem__`` indexes directly).
        """
        cfg = self.config
        if cfg.sources is None and cfg.max_pass_rate is None and cfg.require_consistent is None:
            return None

        cols = self._data.column_names
        keep = np.ones(len(self._data), dtype=bool)
        if cfg.sources is not None and "source" in cols:
            wanted = set(cfg.sources)
            keep &= np.array(
                [s in wanted for s in self._data["source"]],
                dtype=bool,
            )
        if cfg.max_pass_rate is not None and "pass_rate" in cols:
            rates = np.array(
                [np.inf if r is None else float(r) for r in self._data["pass_rate"]],
                dtype=np.float64,
            )
            keep &= rates <= cfg.max_pass_rate
        if cfg.require_consistent is not None and "is_consistent" in cols:
            keep &= np.array(
                [bool(v) == cfg.require_consistent for v in self._data["is_consistent"]],
                dtype=bool,
            )
        index = np.nonzero(keep)[0]
        log.info("MMFineReason: kept %d / %d rows after filtering", len(index), len(self._data))
        return index

    def __len__(self) -> int:
        return len(self._data) if self._index is None else len(self._index)

    def _row(self, i: int) -> Dict:
        pos = int(i if self._index is None else self._index[i])
        return self._data[pos]

    def _build(self, i: int) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._row(i)

        question = strip_image_placeholders(row[cfg.question_column])
        raw = row[cfg.answer_column]
        answer = extract_answer_text(raw)
        if cfg.supervise_cot:
            # Both extractors fall back to the answer-only target when there is no <think>
            # block, so the ~51% of rows without a trace behave exactly as they do today.
            if cfg.cot_scratchpad:
                # Grade the short verifiable answer, not the verbose one: everything the
                # model needs to *say* goes inside <think>, and only this is scored.
                short = (row.get(cfg.short_answer_column) or "").strip()
                answer = extract_reasoning_scratchpad(raw, final_answer=short or answer) or answer
            else:
                answer = extract_reasoning_text(raw, final_answer=answer) or answer
        if not question or not answer:
            raise ValueError("empty question or answer after parsing")

        image = row.get(cfg.image_column)
        pil_images = [decode_pil_image(image)] if image is not None else []

        seq = encode_sft_example(
            self.tokenizer,
            pil_images,
            [[(question, answer)]],
            max_crops=cfg.max_crops,
            loss_token_weighting=cfg.loss_token_weighting,
            shuffle_rng=example_rng(cfg.seed, i),
        )
        if cfg.skip_overlong and len(seq["input_ids"]) > cfg.max_sequence_length:
            raise ValueError(
                f"MMFineReason row {i}: {len(seq['input_ids'])} tokens exceeds "
                f"max_sequence_length={cfg.max_sequence_length}"
            )
        return truncate_example(seq, cfg.max_sequence_length)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Build the example at ``index``, skipping ahead over unusable rows.

        See :func:`~olmo_core.data.multimodal.sft_common.get_example_with_skip`.
        """
        return get_example_with_skip(self, index, len(self))
