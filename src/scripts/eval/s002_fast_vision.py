"""Evaluate native s002 multimodal checkpoints on fixed PixMo validation samples.

This evaluator keeps checkpoints in OLMo-core's distributed format, runs the model with
EP8, and reports response-token-weighted CE/PPL for matched caption, count-only,
basic-pointing, and grounded point-counting samples. Count-only evaluation also reports
candidate-normalized 2-10 classification and response-format diagnostics. Grounded
point-counting evaluation separately scores those same candidates at the teacher-forced final
count position in each serialized response. The same explicit sample indices and OLMo 3 chat
serializer are used for every checkpoint in a comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from s002_downstream import (
    DEFAULT_OUTPUT_ROOT,
    _build_model_and_module_config,
    _checkpoint_state_dir,
    _config_path,
    _git_revision,
)

from olmo_core.data.multimodal import (
    PIXMO_DATASETS,
    MultimodalCollator,
    MultimodalDataLoader,
    PixMoCapDatasetConfig,
    PixMoCountDatasetConfig,
    PixMoPointsDatasetConfig,
)
from olmo_core.data.multimodal.sft_common import (
    SFT_MESSAGE_FORMATS,
    MaxSequenceLengthDataset,
    SftMessageFormat,
    validate_sft_message_format,
)
from olmo_core.distributed.utils import all_reduce_value, get_rank, get_world_size
from olmo_core.eval import MultimodalLMEvaluator
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.vision import prepare_molmo2_tokenizer
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.utils import gc_cuda, move_to_device

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = (
    "/weka/oe-training-default/rustin/experiments/vision-moe/checkpoints/"
    "s002-stage2-v9-pilot-bounded-errors-5a81c40c/step200"
)
DEFAULT_TOKENIZER = (
    "/weka/oe-training-default/robertb/olmo3moe-post-training/checkpoints/"
    "s002-olmo3moe-instruct-sft-resume-to1000-fused-20260727-hf"
)
DEFAULT_HF_CACHE = "/weka/oe-training-default/rustin/hf-cache/hub"
DEFAULT_EXAMPLES = 512
DEFAULT_SAMPLE_SEED = 6198
DEFAULT_MAX_SEQUENCE_LENGTH = 16384
DEFAULT_MAX_CROPS = 8
DEFAULT_RANK_BATCH_INSTANCES = 1
TASK_NAMES = ("caption", "count", "points", "point_count")
TASK_SEED_OFFSETS = {"caption": 0, "count": 1, "points": 2, "point_count": 2}
NUMERIC_COUNT_VALUES = tuple(range(2, 11))
GROUNDED_COUNT_TARGET_VALUES = tuple(range(61))


@dataclass(frozen=True)
class TaskSpec:
    """A named validation dataset and its deterministic source indices."""

    name: str
    dataset: Any
    indices: Sequence[int]


@dataclass(frozen=True)
class NumericCountTokenProtocol:
    """Single-token answer candidates and competing response-format prefixes."""

    values: Sequence[int]
    candidate_token_ids: Sequence[int]
    eos_token_id: int
    counting_prefix_token_id: int
    points_prefix_token_id: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "candidate_token_ids": list(self.candidate_token_ids),
            "eos_token_id": self.eos_token_id,
            "counting_prefix": {
                "text": "Counting",
                "first_token_id": self.counting_prefix_token_id,
            },
            "points_prefix": {
                "text": "<points",
                "first_token_id": self.points_prefix_token_id,
            },
            "candidate_scoring": (
                "softmax over first-response-token logits for the single-token answers 2-10"
            ),
        }


@dataclass(frozen=True)
class GroundedFinalCountTokenProtocol:
    """Token templates for locating final counts in grounded point-count responses."""

    candidate_values: Sequence[int]
    candidate_token_ids: Sequence[int]
    eos_token_id: int
    target_values: Sequence[int]
    positive_suffix_token_ids: dict[int, Sequence[int]]
    none_response_token_ids: Sequence[int]
    candidate_token_offsets: dict[int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_values": list(self.candidate_values),
            "candidate_token_ids": list(self.candidate_token_ids),
            "eos_token_id": self.eos_token_id,
            "supported_target_values": list(self.target_values),
            "target_serialization": {
                "positive_template": "Counting the <points...> shows a total of N.",
                "matched_terminal_template": " shows a total of N.",
                "zero_text": "There are none.",
                "zero_token_ids": list(self.none_response_token_ids),
                "positive_suffix_token_ids": {
                    str(value): list(self.positive_suffix_token_ids[value])
                    for value in self.target_values
                    if value > 0
                },
                "candidate_token_offsets": {
                    str(value): self.candidate_token_offsets[value]
                    for value in self.candidate_values
                },
            },
            "slot_selection": (
                "within each supervised assistant subsegment, match the exact terminal target "
                "serialization immediately before EOS and select its final count label row; "
                "numeric tokens inside the preceding <points> coordinates are never candidates"
            ),
            "logit_alignment": (
                "response logits and labels use the same flattened row-major loss-mask order"
            ),
            "candidate_scoring": (
                "softmax over token logits for the single-token counts 2-10 at the "
                "teacher-forced grounded final-count position"
            ),
            "aggregation_unit": "eligible grounded assistant response (target count 2-10)",
            "excluded_targets": (
                "0, 1, and 11-60 are detected and audited but rejected from candidate scoring"
            ),
            "predeclared_step200_go_rule": {
                "final_count_top1": "step200 >= max(Stage1, step50) - 0.03",
                "final_count_nll": "step200 <= min(Stage1, step50) + 0.10 nat",
                "grounded_ce_vs_stage1": "step200 <= 0.95 * Stage1",
                "grounded_ce_vs_step50": "step200 <= 1.05 * step50",
            },
        }


class IndexedDataset:
    """Expose a fixed source-index selection while preserving source-epoch semantics."""

    def __init__(self, dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self.config = getattr(dataset, "config", None)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0):
        source_index = self.indices[index]
        getter = getattr(self.dataset, "get", None)
        return getter(source_index, epoch) if getter is not None else self.dataset[source_index]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", help="Config JSON (defaults to CHECKPOINT/config.json).")
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES, default=list(TASK_NAMES))
    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--ep-degree", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--rank-batch-instances", type=int, default=DEFAULT_RANK_BATCH_INSTANCES)
    parser.add_argument("--max-crops", type=int, default=DEFAULT_MAX_CROPS)
    parser.add_argument(
        "--message-format",
        choices=SFT_MESSAGE_FORMATS,
        default="olmo3_chat",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--hf-cache", default=DEFAULT_HF_CACHE)
    parser.add_argument("--work-dir", help="Local evaluator work directory.")
    parser.add_argument("--output", help="Result JSON path under Rustin's eval root by default.")
    parser.add_argument("--checkpoint-load-threads", type=int, default=8)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, world_size: int) -> None:
    if args.examples <= 0:
        raise ValueError("--examples must be positive")
    if args.ep_degree <= 0 or world_size % args.ep_degree != 0:
        raise ValueError(
            f"WORLD_SIZE ({world_size}) must be divisible by positive --ep-degree "
            f"({args.ep_degree})"
        )
    if args.max_sequence_length <= 0:
        raise ValueError("--max-sequence-length must be positive")
    if args.rank_batch_instances <= 0:
        raise ValueError("--rank-batch-instances must be positive")
    if args.max_crops <= 0:
        raise ValueError("--max-crops must be positive")
    if args.checkpoint_load_threads <= 0:
        raise ValueError("--checkpoint-load-threads must be positive")
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError("--tasks must not contain duplicates")


def _representative_indices(size: int, examples: int, *, seed: int) -> list[int]:
    """Select a deterministic, without-replacement permutation prefix."""
    if size <= 0:
        raise ValueError("dataset size must be positive")
    if examples > size:
        raise ValueError(f"Requested {examples} examples from a dataset of size {size}")
    return np.random.RandomState(seed).permutation(size)[:examples].astype(int).tolist()


def _indices_sha256(indices: Sequence[int]) -> str:
    encoded = json.dumps(list(indices), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _numeric_count_token_protocol(tokenizer) -> NumericCountTokenProtocol:
    candidate_token_ids = []
    for value in NUMERIC_COUNT_VALUES:
        encoded = list(tokenizer.encode(str(value), add_special_tokens=False))
        if len(encoded) != 1:
            raise ValueError(
                f"Numeric count candidate {value} must encode to one token, got {encoded}"
            )
        candidate_token_ids.append(int(encoded[0]))
    if len(set(candidate_token_ids)) != len(candidate_token_ids):
        raise ValueError("Numeric count candidates must have distinct token IDs")
    if tokenizer.eos_token_id is None:
        raise ValueError("Numeric count diagnostics require a tokenizer EOS token")

    def first_token(text: str) -> int:
        encoded = list(tokenizer.encode(text, add_special_tokens=False))
        if not encoded:
            raise ValueError(f"Response-format prefix {text!r} encoded to no tokens")
        return int(encoded[0])

    return NumericCountTokenProtocol(
        values=NUMERIC_COUNT_VALUES,
        candidate_token_ids=tuple(candidate_token_ids),
        eos_token_id=int(tokenizer.eos_token_id),
        counting_prefix_token_id=first_token("Counting"),
        points_prefix_token_id=first_token("<points"),
    )


def _grounded_final_count_token_protocol(
    tokenizer,
    numeric_protocol: NumericCountTokenProtocol,
) -> GroundedFinalCountTokenProtocol:
    """Build exact terminal-response templates for grounded count-slot selection."""

    positive_suffix_token_ids: dict[int, Sequence[int]] = {}
    candidate_token_offsets: dict[int, int] = {}
    candidate_id_by_value = dict(zip(numeric_protocol.values, numeric_protocol.candidate_token_ids))
    positive_suffix_prefix = tuple(
        int(token_id)
        for token_id in tokenizer.encode(" shows a total of ", add_special_tokens=False)
    )
    positive_suffix_punctuation = tuple(
        int(token_id) for token_id in tokenizer.encode(".", add_special_tokens=False)
    )
    if not positive_suffix_prefix or not positive_suffix_punctuation:
        raise ValueError("Grounded count suffix framing encoded to no tokens")
    for value in GROUNDED_COUNT_TARGET_VALUES:
        if value == 0:
            continue
        encoded = tuple(
            int(token_id)
            for token_id in tokenizer.encode(
                f" shows a total of {value}.",
                add_special_tokens=False,
            )
        )
        if not encoded:
            raise ValueError(f"Grounded count suffix for {value} encoded to no tokens")
        positive_suffix_token_ids[value] = encoded
        if value in candidate_id_by_value:
            candidate_token_id = candidate_id_by_value[value]
            expected = positive_suffix_prefix + (candidate_token_id,) + positive_suffix_punctuation
            if encoded != expected:
                raise ValueError(
                    f"Grounded count suffix for {value} must serialize its final count as "
                    f"exactly the single candidate token {candidate_token_id}; expected "
                    f"{expected}, got {encoded}"
                )
            candidate_token_offsets[value] = len(positive_suffix_prefix)

    if len(set(positive_suffix_token_ids.values())) != len(positive_suffix_token_ids):
        raise ValueError("Grounded positive count suffix tokenizations must be distinct")
    none_response_token_ids = tuple(
        int(token_id) for token_id in tokenizer.encode("There are none.", add_special_tokens=False)
    )
    if not none_response_token_ids:
        raise ValueError("Grounded zero-count response encoded to no tokens")

    return GroundedFinalCountTokenProtocol(
        candidate_values=tuple(numeric_protocol.values),
        candidate_token_ids=tuple(numeric_protocol.candidate_token_ids),
        eos_token_id=numeric_protocol.eos_token_id,
        target_values=GROUNDED_COUNT_TARGET_VALUES,
        positive_suffix_token_ids=positive_suffix_token_ids,
        none_response_token_ids=none_response_token_ids,
        candidate_token_offsets=candidate_token_offsets,
    )


def _match_grounded_final_count_target(
    response_labels: Sequence[int],
    protocol: GroundedFinalCountTokenProtocol,
) -> tuple[int, int | None]:
    """Identify a grounded response target and its eligible final-count label offset."""

    labels = tuple(int(label) for label in response_labels)
    if not labels or labels[-1] != protocol.eos_token_id:
        raise ValueError(
            "Grounded point-count response must end in the configured EOS token; got tail "
            f"{list(labels[-12:])}"
        )
    body = labels[:-1]
    matches: list[tuple[int, int | None]] = []
    if body == tuple(protocol.none_response_token_ids):
        matches.append((0, None))
    for value, suffix_ids in protocol.positive_suffix_token_ids.items():
        suffix = tuple(suffix_ids)
        if len(body) >= len(suffix) and body[-len(suffix) :] == suffix:
            candidate_position = None
            if value in protocol.candidate_token_offsets:
                candidate_position = (
                    len(body) - len(suffix) + protocol.candidate_token_offsets[value]
                )
            matches.append((value, candidate_position))
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one grounded final-count target serialization per assistant "
            f"response, found {len(matches)} for tail {list(labels[-16:])}"
        )
    return matches[0]


def _numeric_count_batch_statistics(
    batch: dict[str, Any],
    logits: torch.Tensor,
    protocol: NumericCountTokenProtocol,
) -> torch.Tensor:
    """Return additive count-classification and response-format statistics for one batch."""
    labels = batch.get("labels")
    if labels is None:
        raise ValueError("Numeric count diagnostics require labels")
    # MultimodalLM uses this exact flattened row-major mask to select response logits.
    # Check ignored labels separately so label filtering cannot silently change row alignment.
    response_mask = batch["loss_masks"] > 0
    if bool(torch.any(labels.masked_select(response_mask) == -100)):
        raise ValueError("Numeric count loss positions must not have ignored labels")
    response_counts = response_mask.sum(dim=1)
    if not bool(torch.all(response_counts == 2)):
        raise ValueError(
            "Numeric count diagnostics require exactly two supervised tokens per example "
            f"(number and EOS), got {response_counts.detach().cpu().tolist()}"
        )

    batch_size = int(labels.shape[0])
    response_labels = labels.masked_select(response_mask).reshape(batch_size, 2)
    if logits.ndim == 3:
        response_logits = logits.reshape(-1, logits.shape[-1])[response_mask.reshape(-1)]
    elif logits.ndim == 2 and logits.shape[0] == batch_size * 2:
        response_logits = logits
    else:
        raise ValueError(
            "Numeric count diagnostics expected response-only logits with shape "
            f"({batch_size * 2}, vocab), got {tuple(logits.shape)}"
        )
    response_logits = response_logits.reshape(batch_size, 2, -1).float()
    first_logits, eos_logits = response_logits[:, 0], response_logits[:, 1]
    first_labels, eos_labels = response_labels[:, 0], response_labels[:, 1]

    candidate_ids = torch.tensor(
        protocol.candidate_token_ids,
        dtype=torch.long,
        device=first_logits.device,
    )
    candidate_matches = first_labels.unsqueeze(1) == candidate_ids.unsqueeze(0)
    if not bool(torch.all(candidate_matches.sum(dim=1) == 1)):
        raise ValueError(
            "Numeric count labels must be one of the configured candidate token IDs; got "
            f"{first_labels.detach().cpu().tolist()}"
        )
    if not bool(torch.all(eos_labels == protocol.eos_token_id)):
        raise ValueError(
            "Numeric count responses must end in the configured EOS token; got "
            f"{eos_labels.detach().cpu().tolist()}"
        )

    gold_candidate_indices = candidate_matches.to(torch.int64).argmax(dim=1)
    candidate_logits = first_logits.index_select(1, candidate_ids)
    candidate_log_probs = F.log_softmax(candidate_logits, dim=1)
    candidate_nll = -candidate_log_probs.gather(1, gold_candidate_indices.unsqueeze(1)).squeeze(1)
    candidate_predictions = candidate_logits.argmax(dim=1)

    first_log_normalizer = torch.logsumexp(first_logits, dim=1)
    eos_log_normalizer = torch.logsumexp(eos_logits, dim=1)
    raw_digit_nll = first_log_normalizer - first_logits.gather(
        1, first_labels.unsqueeze(1)
    ).squeeze(1)
    raw_eos_nll = eos_log_normalizer - eos_logits.gather(1, eos_labels.unsqueeze(1)).squeeze(1)
    numeric_log_mass = torch.logsumexp(candidate_logits, dim=1) - first_log_normalizer
    counting_log_mass = first_logits[:, protocol.counting_prefix_token_id] - first_log_normalizer
    points_log_mass = first_logits[:, protocol.points_prefix_token_id] - first_log_normalizer
    numeric_mass = numeric_log_mass.exp()
    counting_mass = counting_log_mass.exp()
    points_mass = points_log_mass.exp()

    # These conditional shares are stable even when all named formats have low raw mass.
    numeric_vs_counting = torch.sigmoid(numeric_log_mass - counting_log_mass)
    structured_log_mass = torch.logaddexp(counting_log_mass, points_log_mass)
    numeric_vs_structured = torch.sigmoid(numeric_log_mass - structured_log_mass)

    target_histogram = F.one_hot(gold_candidate_indices, num_classes=len(protocol.values)).sum(
        dim=0
    )
    prediction_histogram = F.one_hot(candidate_predictions, num_classes=len(protocol.values)).sum(
        dim=0
    )
    scalar_sums = torch.stack(
        [
            candidate_nll.sum(),
            (candidate_predictions == gold_candidate_indices).sum(),
            raw_digit_nll.sum(),
            raw_eos_nll.sum(),
            (first_logits.argmax(dim=1) == first_labels).sum(),
            numeric_mass.sum(),
            counting_mass.sum(),
            points_mass.sum(),
            numeric_vs_counting.sum(),
            numeric_vs_structured.sum(),
        ]
    )
    return torch.cat(
        [
            scalar_sums.to(torch.float64),
            target_histogram.to(torch.float64),
            prediction_histogram.to(torch.float64),
            scalar_sums.new_tensor([batch_size], dtype=torch.float64),
        ]
    )


def _numeric_count_metrics(
    statistics: torch.Tensor,
    protocol: NumericCountTokenProtocol,
) -> dict[str, Any]:
    """Convert globally summed numeric-count statistics into JSON-ready metrics."""
    n_candidates = len(protocol.values)
    expected_size = 10 + 2 * n_candidates + 1
    if statistics.numel() != expected_size:
        raise ValueError(
            f"Expected {expected_size} numeric count statistics, got {statistics.numel()}"
        )
    values = statistics.detach().cpu().to(torch.float64)
    examples = int(values[-1].item())
    if examples <= 0:
        raise ValueError("Numeric count diagnostics evaluated no examples")
    denominator = float(examples)
    target_start = 10
    prediction_start = target_start + n_candidates
    target_histogram = {
        str(value): int(values[target_start + index].item())
        for index, value in enumerate(protocol.values)
    }
    prediction_histogram = {
        str(value): int(values[prediction_start + index].item())
        for index, value in enumerate(protocol.values)
    }
    return {
        "examples": examples,
        "candidate_values": list(protocol.values),
        "candidate_token_ids": list(protocol.candidate_token_ids),
        "target_histogram": target_histogram,
        "candidate_top1_prediction_histogram": prediction_histogram,
        "metrics": {
            "candidate-normalized first-token NLL": float(values[0].item() / denominator),
            "candidate top-1 accuracy": float(values[1].item() / denominator),
            "raw digit NLL": float(values[2].item() / denominator),
            "raw teacher-forced EOS NLL": float(values[3].item() / denominator),
            "raw two-token CE": float((values[2].item() + values[3].item()) / (2 * denominator)),
            "full-vocabulary digit top-1 accuracy": float(values[4].item() / denominator),
            "first-token numeric candidate mass": float(values[5].item() / denominator),
            "first-token Counting-prefix mass": float(values[6].item() / denominator),
            "first-token points-prefix mass": float(values[7].item() / denominator),
            "first-token numeric share vs Counting": float(values[8].item() / denominator),
            "first-token numeric share vs structured prefixes": float(
                values[9].item() / denominator
            ),
        },
    }


def _grounded_final_count_batch_statistics(
    batch: dict[str, Any],
    logits: torch.Tensor,
    protocol: GroundedFinalCountTokenProtocol,
) -> torch.Tensor:
    """Return additive final-count statistics from grounded point-count responses."""

    labels = batch.get("labels")
    if labels is None:
        raise ValueError("Grounded final-count diagnostics require labels")
    response_mask = batch["loss_masks"] > 0
    if bool(torch.any(labels.masked_select(response_mask) == -100)):
        raise ValueError("Grounded final-count loss positions must not have ignored labels")
    response_counts = response_mask.sum(dim=1)
    total_response_tokens = int(response_counts.sum().item())
    if logits.ndim == 3:
        response_logits = logits.reshape(-1, logits.shape[-1])[response_mask.reshape(-1)]
    elif logits.ndim == 2 and logits.shape[0] == total_response_tokens:
        response_logits = logits
    else:
        raise ValueError(
            "Grounded final-count diagnostics expected response-only logits with shape "
            f"({total_response_tokens}, vocab), got {tuple(logits.shape)}"
        )
    response_logits = response_logits.float()
    response_labels = labels.masked_select(response_mask).detach().cpu()

    subsegment_ids = batch.get("subsegment_ids")
    response_subsegments = (
        subsegment_ids.masked_select(response_mask).detach().cpu()
        if subsegment_ids is not None
        else None
    )
    candidate_values = tuple(int(value) for value in protocol.candidate_values)
    candidate_index_by_value = {value: index for index, value in enumerate(candidate_values)}
    target_index_by_value = {
        int(value): index for index, value in enumerate(protocol.target_values)
    }
    target_histogram = [0] * len(protocol.target_values)
    selected_response_rows: list[int] = []
    gold_candidate_indices: list[int] = []
    examples_with_scored_targets = 0
    grounded_response_slots = 0

    response_offset = 0
    for row_index, row_count_tensor in enumerate(response_counts.detach().cpu()):
        row_count = int(row_count_tensor.item())
        row_labels = response_labels[response_offset : response_offset + row_count]
        if response_subsegments is None:
            row_subsegments = torch.zeros(row_count, dtype=torch.long)
        else:
            row_subsegments = response_subsegments[response_offset : response_offset + row_count]
        ordered_subsegments = list(dict.fromkeys(int(value) for value in row_subsegments.tolist()))
        row_scored_targets = 0
        for subsegment_id in ordered_subsegments:
            group_positions = torch.nonzero(
                row_subsegments == subsegment_id,
                as_tuple=False,
            ).flatten()
            group_labels = row_labels.index_select(0, group_positions).tolist()
            target_value, candidate_group_position = _match_grounded_final_count_target(
                group_labels,
                protocol,
            )
            if target_value not in target_index_by_value:
                raise ValueError(
                    f"Grounded target {target_value} is outside configured audit values"
                )
            grounded_response_slots += 1
            target_histogram[target_index_by_value[target_value]] += 1
            if candidate_group_position is not None:
                selected_response_rows.append(
                    response_offset + int(group_positions[candidate_group_position].item())
                )
                gold_candidate_indices.append(candidate_index_by_value[target_value])
                row_scored_targets += 1
        if row_scored_targets:
            examples_with_scored_targets += 1
        response_offset += row_count
    if response_offset != total_response_tokens:
        raise RuntimeError(
            f"Consumed {response_offset} response rows, expected {total_response_tokens}"
        )

    candidate_ids = torch.tensor(
        protocol.candidate_token_ids,
        dtype=torch.long,
        device=response_logits.device,
    )
    if selected_response_rows:
        selected_logits = response_logits.index_select(
            0,
            torch.tensor(
                selected_response_rows,
                dtype=torch.long,
                device=response_logits.device,
            ),
        )
        gold_indices = torch.tensor(
            gold_candidate_indices,
            dtype=torch.long,
            device=response_logits.device,
        )
        candidate_logits = selected_logits.index_select(1, candidate_ids)
        candidate_log_probs = F.log_softmax(candidate_logits, dim=1)
        candidate_nll = -candidate_log_probs.gather(1, gold_indices.unsqueeze(1)).squeeze(1)
        candidate_predictions = candidate_logits.argmax(dim=1)
        gold_token_ids = candidate_ids.index_select(0, gold_indices)
        log_normalizer = torch.logsumexp(selected_logits, dim=1)
        raw_nll = log_normalizer - selected_logits.gather(1, gold_token_ids.unsqueeze(1)).squeeze(1)
        candidate_mass = (torch.logsumexp(candidate_logits, dim=1) - log_normalizer).exp()
        prediction_histogram = F.one_hot(
            candidate_predictions,
            num_classes=len(candidate_values),
        ).sum(dim=0)
        candidate_nll_sum = candidate_nll.sum()
        candidate_correct_sum = (candidate_predictions == gold_indices).sum()
        raw_nll_sum = raw_nll.sum()
        full_vocab_correct_sum = (selected_logits.argmax(dim=1) == gold_token_ids).sum()
        candidate_mass_sum = candidate_mass.sum()
    else:
        prediction_histogram = torch.zeros(
            len(candidate_values),
            dtype=torch.long,
            device=response_logits.device,
        )
        candidate_nll_sum = response_logits.new_zeros(())
        candidate_correct_sum = response_logits.new_zeros((), dtype=torch.long)
        raw_nll_sum = response_logits.new_zeros(())
        full_vocab_correct_sum = response_logits.new_zeros((), dtype=torch.long)
        candidate_mass_sum = response_logits.new_zeros(())

    scalar_sums = torch.stack(
        [
            candidate_nll_sum,
            candidate_correct_sum,
            raw_nll_sum,
            full_vocab_correct_sum,
            candidate_mass_sum,
            response_logits.new_tensor(float(grounded_response_slots)),
            response_logits.new_tensor(float(len(selected_response_rows))),
            response_logits.new_tensor(float(labels.shape[0])),
            response_logits.new_tensor(float(examples_with_scored_targets)),
        ]
    ).to(torch.float64)
    return torch.cat(
        [
            scalar_sums,
            torch.tensor(
                target_histogram,
                dtype=torch.float64,
                device=response_logits.device,
            ),
            prediction_histogram.to(torch.float64),
        ]
    )


def _grounded_final_count_metrics(
    statistics: torch.Tensor,
    protocol: GroundedFinalCountTokenProtocol,
) -> dict[str, Any]:
    """Convert globally summed grounded final-count statistics to JSON-ready metrics."""

    n_scalars = 9
    n_targets = len(protocol.target_values)
    n_candidates = len(protocol.candidate_values)
    expected_size = n_scalars + n_targets + n_candidates
    if statistics.numel() != expected_size:
        raise ValueError(
            f"Expected {expected_size} grounded final-count statistics, "
            f"got {statistics.numel()}"
        )
    values = statistics.detach().cpu().to(torch.float64)
    grounded_response_slots = int(values[5].item())
    scored_response_slots = int(values[6].item())
    examples = int(values[7].item())
    examples_with_scored_targets = int(values[8].item())
    if grounded_response_slots <= 0 or examples <= 0:
        raise ValueError("Grounded final-count diagnostics evaluated no responses")
    if scored_response_slots <= 0:
        raise ValueError("Grounded final-count diagnostics found no targets in 2-10")

    target_start = n_scalars
    prediction_start = target_start + n_targets
    target_histogram = {
        str(value): int(values[target_start + index].item())
        for index, value in enumerate(protocol.target_values)
    }
    prediction_histogram = {
        str(value): int(values[prediction_start + index].item())
        for index, value in enumerate(protocol.candidate_values)
    }
    denominator = float(scored_response_slots)
    return {
        "examples": examples,
        "examples_with_scored_targets": examples_with_scored_targets,
        "grounded_response_slots": grounded_response_slots,
        "scored_response_slots": scored_response_slots,
        "excluded_response_slots": grounded_response_slots - scored_response_slots,
        "candidate_values": list(protocol.candidate_values),
        "candidate_token_ids": list(protocol.candidate_token_ids),
        "target_histogram": target_histogram,
        "candidate_top1_prediction_histogram": prediction_histogram,
        "metrics": {
            "candidate-normalized final-count NLL": float(values[0].item() / denominator),
            "final-count candidate top-1 accuracy": float(values[1].item() / denominator),
            "raw final-count NLL": float(values[2].item() / denominator),
            "full-vocabulary final-count top-1 accuracy": float(values[3].item() / denominator),
            "final-count numeric candidate mass": float(values[4].item() / denominator),
        },
    }


def _build_task_datasets(
    tokenizer,
    token_ids: Molmo2TokenIds,
    *,
    message_format: SftMessageFormat,
    max_sequence_length: int,
    max_crops: int,
    sample_seed: int,
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "max_crops": max_crops,
        "loss_token_weighting": "none",
        "token_ids": token_ids,
        "message_format": message_format,
        "seed": sample_seed,
    }
    return {
        "caption": PixMoCapDatasetConfig(
            dataset_path=f"{PIXMO_DATASETS}/cap",
            split="validation",
            mode="caption",
            max_sequence_length=max_sequence_length,
            **common,
        ).build(tokenizer),
        "count": PixMoCountDatasetConfig(
            split="validation",
            counting=True,
            **common,
        ).build(tokenizer),
        "points": PixMoPointsDatasetConfig(
            split="validation",
            kind="basic",
            counting=False,
            **common,
        ).build(tokenizer),
        "point_count": PixMoPointsDatasetConfig(
            split="validation",
            kind="basic",
            counting=True,
            **common,
        ).build(tokenizer),
    }


def _build_task_specs(
    datasets: dict[str, Any],
    tasks: Sequence[str],
    *,
    examples: int,
    sample_seed: int,
) -> list[TaskSpec]:
    shared_point_indices = None
    if any(name in tasks for name in ("points", "point_count")):
        points_size = len(datasets["points"])
        point_count_size = len(datasets["point_count"])
        if points_size != point_count_size:
            raise ValueError(
                "Pointing and point-counting datasets must expose the same source index space, "
                f"got {points_size} and {point_count_size}"
            )
        shared_point_indices = _representative_indices(
            points_size,
            examples,
            seed=sample_seed + TASK_SEED_OFFSETS["points"],
        )

    specs = []
    for name in tasks:
        dataset = datasets[name]
        if name in ("points", "point_count"):
            assert shared_point_indices is not None
            indices = shared_point_indices
        else:
            indices = _representative_indices(
                len(dataset),
                examples,
                seed=sample_seed + TASK_SEED_OFFSETS[name],
            )
        specs.append(TaskSpec(name=name, dataset=dataset, indices=indices))
    return specs


def _default_output(checkpoint: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_name = (
        checkpoint.parent.name if checkpoint.name == "model_and_optim" else checkpoint.name
    )
    return Path(DEFAULT_OUTPUT_ROOT) / checkpoint_name / f"fast-vision-complete-{stamp}.json"


def _evaluate_task(
    train_module,
    task: TaskSpec,
    collator: MultimodalCollator,
    *,
    work_dir: Path,
    max_sequence_length: int,
    rank_batch_instances: int,
    sample_seed: int,
    token_ids: Molmo2TokenIds,
    dp_world_size: int,
    dp_rank: int,
    numeric_count_protocol: NumericCountTokenProtocol | None = None,
    grounded_final_count_protocol: GroundedFinalCountTokenProtocol | None = None,
) -> dict[str, Any]:
    global_batch_instances = rank_batch_instances * dp_world_size
    if len(task.indices) % global_batch_instances != 0:
        raise ValueError(
            f"Task {task.name!r} has {len(task.indices)} examples, which is not divisible by "
            f"the global data-parallel batch of {global_batch_instances}"
        )

    selected = IndexedDataset(task.dataset, task.indices)
    bounded = MaxSequenceLengthDataset(
        selected,
        max_sequence_length,
        token_ids=token_ids,
    )
    loader = MultimodalDataLoader(
        bounded,
        collator,
        work_dir=work_dir / task.name,
        global_batch_size=global_batch_instances * max_sequence_length,
        seed=sample_seed,
        shuffle=False,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
    )
    evaluator = MultimodalLMEvaluator(
        name=f"pixmo-{task.name}-validation",
        batches=loader,
        device=train_module.device,
        process_group=train_module.dp_process_group,
        deterministic=True,
    )
    evaluator.reset_metrics()

    started = time.monotonic()
    batches = 0
    local_examples = 0
    numeric_count_statistics: torch.Tensor | None = None
    grounded_final_count_statistics: torch.Tensor | None = None
    for batch in evaluator:
        batches += 1
        local_examples += int(batch["input_ids"].shape[0])
        batch = move_to_device(batch, train_module.device)
        retain_response_logits = (
            numeric_count_protocol is not None or grounded_final_count_protocol is not None
        )
        if not retain_response_logits:
            output = train_module.eval_batch(dict(batch))
        else:
            output = train_module.eval_batch(dict(batch), return_response_logits=True)
        if not isinstance(output, LMOutputWithLoss):
            raise TypeError(f"Expected LMOutputWithLoss, got {type(output).__name__}")
        evaluator.update_metrics(batch, output.ce_loss, output.logits)
        if numeric_count_protocol is not None:
            if output.logits is None:
                raise RuntimeError("Numeric count diagnostics require response logits")
            batch_statistics = _numeric_count_batch_statistics(
                batch,
                output.logits,
                numeric_count_protocol,
            )
            if numeric_count_statistics is None:
                numeric_count_statistics = batch_statistics
            else:
                numeric_count_statistics += batch_statistics
        if grounded_final_count_protocol is not None:
            if output.logits is None:
                raise RuntimeError("Grounded final-count diagnostics require response logits")
            batch_statistics = _grounded_final_count_batch_statistics(
                batch,
                output.logits,
                grounded_final_count_protocol,
            )
            if grounded_final_count_statistics is None:
                grounded_final_count_statistics = batch_statistics
            else:
                grounded_final_count_statistics += batch_statistics
        if get_rank() == 0 and (batches == 1 or batches % 20 == 0):
            log.info("[%s] batch %d/%d", task.name, batches, len(loader))

    expected_local_examples = len(task.indices) // dp_world_size
    if local_examples != expected_local_examples:
        raise RuntimeError(
            f"Task {task.name!r} evaluated {local_examples} local examples, expected "
            f"{expected_local_examples}"
        )

    # MeanMetric.compute() reduces its internal tensors in place, so preserve the local
    # weight before computing the globally reduced CE/PPL.
    local_response_token_weight = evaluator.ce_loss.weight.detach().clone()
    metrics = {
        name: float(value.detach().cpu().item())
        for name, value in evaluator.compute_metrics().items()
    }
    response_token_weight = all_reduce_value(
        local_response_token_weight,
        train_module.device,
        group=train_module.dp_process_group,
    )
    result = {
        "metrics": metrics,
        "dataset_size": len(task.dataset),
        "examples": len(task.indices),
        "sample_indices": list(task.indices),
        "sample_indices_sha256": _indices_sha256(task.indices),
        "batches_per_dp_rank": batches,
        "examples_per_dp_rank": local_examples,
        "response_token_weight": float(response_token_weight.detach().cpu().item()),
        "elapsed_seconds": time.monotonic() - started,
    }
    if numeric_count_protocol is not None:
        if numeric_count_statistics is None:
            raise RuntimeError("Numeric count diagnostics did not observe any batches")
        global_statistics = all_reduce_value(
            numeric_count_statistics,
            train_module.device,
            group=train_module.dp_process_group,
        )
        result["numeric_count_diagnostics"] = _numeric_count_metrics(
            global_statistics,
            numeric_count_protocol,
        )
    if grounded_final_count_protocol is not None:
        if grounded_final_count_statistics is None:
            raise RuntimeError("Grounded final-count diagnostics did not observe any batches")
        global_statistics = all_reduce_value(
            grounded_final_count_statistics,
            train_module.device,
            group=train_module.dp_process_group,
        )
        result["grounded_final_count_diagnostics"] = _grounded_final_count_metrics(
            global_statistics,
            grounded_final_count_protocol,
        )
    if get_rank() == 0:
        log.info("Finished %s: %s", task.name, metrics)
    del evaluator, loader, bounded, selected
    gc_cuda()
    return result


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _validate_args(args, world_size)

    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepare_training_environment()

    try:
        from transformers import GPT2Tokenizer

        checkpoint = Path(args.checkpoint).resolve()
        config_path = _config_path(checkpoint, args.config)
        with config_path.open() as config_file:
            raw_config = json.load(config_file)

        model, module_config, detected_checkpoint_kind = _build_model_and_module_config(
            raw_config,
            ep_degree=args.ep_degree,
            max_sequence_length=args.max_sequence_length,
            rank_batch_size=args.rank_batch_instances * args.max_sequence_length,
        )
        if detected_checkpoint_kind != "multimodal_stage1":
            raise ValueError("Fast vision evaluation requires a multimodal checkpoint")
        # Count diagnostics retain logits only at supervised response positions.
        # Never materialize full 16k-sequence logits.
        module_config.response_logits_only = True
        train_module = module_config.build(model, eval_only=True)
        state_dir = _checkpoint_state_dir(checkpoint)
        log.info("Loading multimodal checkpoint from %s", state_dir)
        train_module.load_state_dict_direct(
            state_dir,
            process_group=dist.group.WORLD,
            thread_count=args.checkpoint_load_threads,
            load_optim_state=False,
        )

        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer, cache_dir=args.hf_cache)
        token_ids = prepare_molmo2_tokenizer(tokenizer, model_vocab_size=100352)
        message_format = validate_sft_message_format(
            args.message_format,
            tokenizer=tokenizer,
            token_ids=token_ids,
        )
        if raw_config["model"]["image_patch_token_id"] != token_ids.im_patch_id:
            raise ValueError(
                "Tokenizer image-patch ID does not match checkpoint model config: "
                f"{token_ids.im_patch_id} != {raw_config['model']['image_patch_token_id']}"
            )
        if tokenizer.pad_token_id is None:
            raise ValueError(f"Tokenizer {args.tokenizer!r} does not define a pad token")
        numeric_count_protocol = _numeric_count_token_protocol(tokenizer)
        grounded_final_count_protocol = _grounded_final_count_token_protocol(
            tokenizer,
            numeric_count_protocol,
        )

        collator = MultimodalCollator(
            pad_token_id=int(tokenizer.pad_token_id),
            label_ignore_index=-100,
            pad_sequence_length=args.max_sequence_length,
        )
        datasets = _build_task_datasets(
            tokenizer,
            token_ids,
            message_format=message_format,
            max_sequence_length=args.max_sequence_length,
            max_crops=args.max_crops,
            sample_seed=args.sample_seed,
        )
        tasks = _build_task_specs(
            datasets,
            args.tasks,
            examples=args.examples,
            sample_seed=args.sample_seed,
        )

        dp_process_group = train_module.dp_process_group
        dp_world_size = get_world_size(dp_process_group)
        dp_rank = get_rank(dp_process_group)
        global_batch_instances = args.rank_batch_instances * dp_world_size
        if args.examples % global_batch_instances != 0:
            raise ValueError(
                f"--examples ({args.examples}) must be divisible by the global data-parallel batch "
                f"({global_batch_instances})"
            )

        output = Path(args.output) if args.output else _default_output(checkpoint)
        work_dir = (
            Path(args.work_dir)
            if args.work_dir
            else Path(os.environ.get("RESULTS_DIR", "/tmp")) / "s002-fast-vision"
        )
        results = {
            task.name: _evaluate_task(
                train_module,
                task,
                collator,
                work_dir=work_dir,
                max_sequence_length=args.max_sequence_length,
                rank_batch_instances=args.rank_batch_instances,
                sample_seed=args.sample_seed,
                token_ids=token_ids,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                numeric_count_protocol=(numeric_count_protocol if task.name == "count" else None),
                grounded_final_count_protocol=(
                    grounded_final_count_protocol if task.name == "point_count" else None
                ),
            )
            for task in tasks
        }

        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "checkpoint_state_dir": str(state_dir),
            # The shared detector labels every native multimodal s002 config
            # ``multimodal_stage1`` because both training stages have the same tensor
            # architecture. Expose that detector result without mislabeling Stage 2.
            "model_family": "multimodal_s002",
            "detected_checkpoint_kind": detected_checkpoint_kind,
            "config": str(config_path),
            "git": _git_revision(),
            "protocol": {
                "harness": "native-olmo-core-fast-vision",
                "tasks": list(args.tasks),
                "dataset_split": "validation",
                "examples_per_task": args.examples,
                "sample_seed": args.sample_seed,
                "task_seed_offsets": TASK_SEED_OFFSETS,
                "sample_selection": (
                    "numpy.RandomState(sample_seed + task_seed_offset)."
                    "permutation(dataset_size)[:examples]"
                ),
                "source_epoch": 0,
                "message_format": message_format,
                "tokenizer": args.tokenizer,
                "token_ids": token_ids.as_config_dict(),
                "max_sequence_length": args.max_sequence_length,
                "max_high_resolution_crops": args.max_crops,
                "image_tensor_includes_global_crop": True,
                "loss_token_weighting": "none",
                "metric_aggregation": "response-token-weighted mean cross entropy",
                "rank_batch_instances": args.rank_batch_instances,
                "global_batch_instances": global_batch_instances,
                "world_size": world_size,
                "ep_degree": args.ep_degree,
                "ep_dp_degree": world_size // args.ep_degree,
                "dp_process_group_size": dp_world_size,
                "attention_backend": "flex",
                "numeric_count_scoring": numeric_count_protocol.as_dict(),
                "grounded_final_count_scoring": grounded_final_count_protocol.as_dict(),
            },
            "results": results,
        }
        if get_rank() == 0:
            _write_json_atomic(output, payload)
            log.info("Wrote results to %s", output)
        dist.barrier()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
