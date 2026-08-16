"""Certify native vision-alignment checkpoints on fixed academic validation examples.

This evaluator is intentionally separate from the pinned alignment evaluators.  A CPU-only
``build-manifest`` command freezes validation example IDs, exact source and image identities,
the deterministic shuffled-image mapping, and contamination against an explicit alignment
training-image inventory.  The distributed ``evaluate`` command then validates that manifest
and scores the exact same rows with a native EP8 checkpoint under three image conditions:
correct, within-task shuffled, and blank.

The first protocol version supports a deterministic 512-example panel from each validation split
of VQAv2, TextVQA, DocVQA, ChartQA, AI2D, and A-OKVQA multiple choice.  It uses the
checkpoint's pre-SFT document interface.  Free-answer tasks use 24-token greedy generation and
their standard task metric; multiple-choice tasks use candidate-normalized option-letter
likelihood and accuracy.  All artifacts are canonical JSON and write-once.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import re
import stat
import string
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from s002_downstream import (
    _build_model_and_module_config,
    _checkpoint_state_dir,
    _config_path,
)

from olmo_core.data.multimodal.academic import registry as academic_registry
from olmo_core.data.multimodal.document_layout import document_prompt_ids, response_ids
from olmo_core.data.multimodal.vision_alignment_sources import (
    load_pinned_vision_alignment_tokenizer,
)
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
from olmo_core.nn.vision.molmo2_image_processor import (
    _select_tiling,
    preprocess_image_molmo2,
)
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds, build_image_token_ids
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.utils import gc_cuda

log = logging.getLogger(__name__)

PROTOCOL_NAME = "vision-alignment-external-academic-ep8-v1"
MANIFEST_FORMAT = "vision_alignment_external_academic_manifest"
RECEIPT_FORMAT = "vision_alignment_external_academic_receipt"
SCHEMA_VERSION = 1
EP_DEGREE = 8
DEFAULT_TASKS = (
    "vqav2",
    "textvqa",
    "docvqa",
    "chartqa",
    "ai2d",
    "a_okvqa_mc",
)
CONTROLS = ("correct", "shuffled", "blank")
EMPTY_OPTION_DISPLAY = "<empty>"
DEFAULT_SELECTION_SEED = 6198
DEFAULT_EXAMPLES_PER_TASK = 512
DEFAULT_MAX_SEQUENCE_LENGTH = 8192
DEFAULT_MAX_CROPS = 8
DEFAULT_MAX_NEW_TOKENS = 24
DEFAULT_SEQUENCE_BUCKET_SIZE = 128
EXPECTED_EOS_TOKEN_ID = 100257
EXPECTED_PAD_TOKEN_ID = 100277
EXPECTED_MOLMO2_TOKEN_IDS_SHA256 = (
    "79c201abd6c61d6031c28f7bf4d4b5abd99d21ec43fa995ae6ef5c7246f67d21"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}")

EXPECTED_JOINT_CONFIG_SHA256 = "64b302865831b5aaf11e86e142a85b3467a06b93d6c214fb67f7f94a45c4ddc8"
EXPECTED_JOINT_CHECKPOINT_PARENT = Path(
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints/"
    "vision-alignment-joint-v1"
)


class _MatchedWrongReceiptPin(TypedDict):
    path: Path
    sha256: str


EXPECTED_MATCHED_WRONG_RECEIPTS: dict[int, _MatchedWrongReceiptPin] = {
    12000: {
        "path": Path(
            "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/"
            "joint-v1-matched-wrong-v1/step12000-41860467.json"
        ),
        "sha256": "c7f960975ade934ecf8c9c0c7f39f417d1e1d983d9cbbe1d7f56afef3f00ce64",
    },
    16000: {
        "path": Path(
            "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/"
            "joint-v1-matched-wrong-v1/step16000-41860467.json"
        ),
        "sha256": "c9307835a3597331add4a800a8a4baa7f2dd4df89f6a44b348766899782b5ccc",
    },
}
EXPECTED_CHECKPOINT_MARKER_SHA256 = (
    "77dfdeec42fe7990f4b3b9c4eeecd480edcf5066c110603b115920af38423d03"
)
EXPECTED_DCP_METADATA_SHA256 = {
    12000: "44cc94aa5b69bb774e45561062476d4e97a3d6ef3ff6e5ab40f53591a42a651f",
    16000: "a377447e5cea89c8d204df5a3d95810bd860bd6111d55dbb52bbe951aa6f4ff2",
}
EXPECTED_TRAIN_IMAGE_INVENTORY = Path(
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/artifacts/"
    "perception-provenance-v2/inventories/train-union-unique.sha256"
)
EXPECTED_TRAIN_IMAGE_INVENTORY_SHA256 = (
    "57e00fb00205f2757bddcf29ce55dbdd3506860c38fed5e77eaba7f7c295dcf0"
)
EXPECTED_TRAIN_IMAGE_INVENTORY_COUNT = 1_206_685
EXPECTED_TRAIN_IMAGE_INVENTORY_BYTES = 78_434_525
EXPECTED_ANSWER_TOKEN_COVERAGE = {
    "vqav2": {
        "selected": 512,
        "max_shortest_response_tokens": 4,
        "max_shortest_response_tokens_with_eos": 5,
        "rows_exceeding_cap": 0,
        "rows_without_eos_room": 0,
        "rows_over_8_response_tokens": 0,
        "ordered_rows_sha256": ("30234f6015c10cc511fe98f6e74a0ba5a60f8432fdea9897c0492ce49bf9cda6"),
    },
    "textvqa": {
        "selected": 512,
        "max_shortest_response_tokens": 7,
        "max_shortest_response_tokens_with_eos": 8,
        "rows_exceeding_cap": 0,
        "rows_without_eos_room": 0,
        "rows_over_8_response_tokens": 0,
        "ordered_rows_sha256": ("5fbdf5959ef944a68a8793652b2ece4cafdcbdeca6f33a81b27b4ea9dcb424b1"),
    },
    "docvqa": {
        "selected": 512,
        "max_shortest_response_tokens": 19,
        "max_shortest_response_tokens_with_eos": 20,
        "rows_exceeding_cap": 0,
        "rows_without_eos_room": 0,
        "rows_over_8_response_tokens": 33,
        "ordered_rows_sha256": ("b3e918b62495e706490d7e180785476d88d0c055b92ab2e1f775ed386442bbf2"),
    },
    "chartqa": {
        "selected": 512,
        "max_shortest_response_tokens": 8,
        "max_shortest_response_tokens_with_eos": 9,
        "rows_exceeding_cap": 0,
        "rows_without_eos_room": 0,
        "rows_over_8_response_tokens": 0,
        "ordered_rows_sha256": ("c9e468c0cc7833abe7240674a2fd6b6ab2a5800ecdce205793d8321d8844bed4"),
    },
}


@dataclass(frozen=True)
class AcademicExample:
    """Canonical runtime representation of one academic validation question."""

    task: str
    example_id: str
    source_position: str
    visual: Any
    image_reference: Any
    question: str
    answers: tuple[str, ...] = ()
    options: tuple[str, ...] = ()
    answer_index: int | None = None
    stratum: str | None = None

    def annotation(self) -> dict[str, Any]:
        """Return the image-independent canonical annotation projection."""
        return {
            "task": self.task,
            "example_id": self.example_id,
            "source_position": self.source_position,
            "question": self.question,
            "answers": list(self.answers),
            "options": list(self.options),
            "answer_index": self.answer_index,
            "stratum": self.stratum,
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON key {key!r}")
        output[key] = value
    return output


def _exact_mapping(value: Any, fields: Sequence[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    expected = set(fields)
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{name} fields differ; missing={missing}, extra={extra}")
    return value


def _validate_timestamp(value: Any, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} is not ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")


def _attach_content_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "content_sha256" in payload:
        raise ValueError("content_sha256 must be attached exactly once")
    output = dict(payload)
    output["content_sha256"] = _canonical_sha256(payload)
    return output


def _verify_content_sha256(payload: Mapping[str, Any], *, name: str) -> None:
    expected = payload.get("content_sha256")
    if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
        raise ValueError(f"{name} lacks a lowercase SHA-256 content identity")
    content = dict(payload)
    del content["content_sha256"]
    actual = _canonical_sha256(content)
    if actual != expected:
        raise ValueError(f"{name} content SHA-256 differs: expected {expected}, got {actual}")


def _load_json_strict(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Could not read strict JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object in {path}")
    return value


def _artifact_path(path: Path, *, name: str) -> Path:
    """Return a lexical absolute path after rejecting symlinked existing components."""
    absolute = Path(os.path.abspath(path.expanduser()))
    for component in (*reversed(absolute.parents), absolute):
        if component == Path(component.anchor) or not component.exists():
            continue
        if stat.S_ISLNK(component.lstat().st_mode):
            raise ValueError(f"{name} contains a symlinked component: {component}")
    return absolute


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _write_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish canonical JSON through an owned, no-follow temporary without replacement."""
    path = _artifact_path(path, name="canonical output")
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _artifact_path(path, name="canonical output")
    raw = _canonical_bytes(payload) + b"\n"
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(path.parent, directory_flags)
    temporary_name = f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    temporary_fd = -1
    temporary_identity: tuple[int, int, int] | None = None
    created = False
    try:
        directory_before = os.fstat(directory_fd)
        if not stat.S_ISDIR(directory_before.st_mode) or directory_before.st_uid != os.geteuid():
            raise ValueError("Canonical output directory is not an owned directory")
        temporary_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        temporary_fd = os.open(
            temporary_name,
            temporary_flags,
            0o600,
            dir_fd=directory_fd,
        )
        created = True
        temporary_before = os.fstat(temporary_fd)
        if (
            not stat.S_ISREG(temporary_before.st_mode)
            or temporary_before.st_uid != os.geteuid()
            or temporary_before.st_nlink != 1
        ):
            raise ValueError("Canonical output temporary is not an owned private regular file")
        temporary_identity = _stat_signature(temporary_before)[:3]
        view = memoryview(raw)
        while view:
            view = view[os.write(temporary_fd, view) :]
        os.fsync(temporary_fd)
        temporary_after = os.fstat(temporary_fd)
        if _stat_signature(temporary_before)[:3] != _stat_signature(temporary_after)[
            :3
        ] or temporary_after.st_size != len(raw):
            raise RuntimeError("Canonical output temporary changed while written")
        named_temporary = os.stat(temporary_name, dir_fd=directory_fd, follow_symlinks=False)
        if _stat_signature(named_temporary) != _stat_signature(temporary_after):
            raise RuntimeError("Canonical output temporary path was replaced")
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite canonical artifact {path}") from error
        temporary_after_link = os.fstat(temporary_fd)
        destination = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            _stat_signature(destination) != _stat_signature(temporary_after_link)
            or destination.st_uid != os.geteuid()
        ):
            raise RuntimeError("Canonical output hard link differs from its exact temporary")
        os.fsync(directory_fd)
        directory_after = os.fstat(directory_fd)
        directory_current = path.parent.lstat()
        if (
            _stat_signature(directory_before)[:3] != _stat_signature(directory_after)[:3]
            or _stat_signature(directory_before)[:3] != _stat_signature(directory_current)[:3]
        ):
            raise RuntimeError("Canonical output directory changed during publication")
    finally:
        if created:
            named: os.stat_result | None
            try:
                named = os.stat(temporary_name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                named = None
            if (
                named is not None
                and temporary_identity is not None
                and _stat_signature(named)[:3] == temporary_identity
            ):
                os.unlink(temporary_name, dir_fd=directory_fd)
        if temporary_fd >= 0:
            os.close(temporary_fd)
        os.close(directory_fd)
    expected = hashlib.sha256(raw).hexdigest()
    if _sha256_file_stable(path) != expected:
        raise RuntimeError("Published canonical output bytes differ")


def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as error:
        raise ValueError(f"Could not stat required file {path}: {error}") from error
    if not path.is_file() or stat.st_size <= 0:
        raise ValueError(f"Required file is not a non-empty regular file: {path}")
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev)


def _sha256_file_stable(path: Path) -> str:
    before = _file_signature(path)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file_handle:
            opened = os.fstat(file_handle.fileno())
            opened_signature = (
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_ino,
                opened.st_dev,
            )
            if opened_signature != before:
                raise ValueError(f"Required file changed before hashing: {path}")
            while chunk := file_handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"Could not hash required file {path}: {error}") from error
    if _file_signature(path) != before:
        raise ValueError(f"Required file changed while hashing: {path}")
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    size = _file_signature(path)[0]
    return {"path": str(path), "bytes": size, "sha256": _sha256_file_stable(path)}


def _image_reference_sha256(reference: Any) -> str:
    if isinstance(reference, Mapping):
        embedded = reference.get("bytes")
        if isinstance(embedded, (bytes, bytearray, memoryview)) and embedded:
            return hashlib.sha256(bytes(embedded)).hexdigest()
        reference = reference.get("path")
    if isinstance(reference, (bytes, bytearray, memoryview)):
        if not reference:
            raise ValueError("Image byte reference is empty")
        return hashlib.sha256(bytes(reference)).hexdigest()
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"Unsupported exact image reference {type(reference)!r}")
    path = Path(reference).expanduser().resolve()
    return _sha256_file_stable(path)


def _image_dimensions(reference: Any) -> tuple[int, int]:
    if isinstance(reference, Image.Image):
        width, height = reference.size
    elif isinstance(reference, np.ndarray):
        if reference.ndim < 2:
            raise ValueError("Image array has fewer than two dimensions")
        height, width = reference.shape[:2]
    else:
        if isinstance(reference, Mapping):
            embedded = reference.get("bytes")
            if isinstance(embedded, (bytes, bytearray, memoryview)) and embedded:
                import io

                with Image.open(io.BytesIO(bytes(embedded))) as image:
                    width, height = image.size
            else:
                reference = reference.get("path")
                if not isinstance(reference, (str, os.PathLike)):
                    raise ValueError("Image mapping has neither encoded bytes nor a path")
                with Image.open(reference) as image:
                    width, height = image.size
        elif isinstance(reference, (str, os.PathLike)):
            with Image.open(reference) as image:
                width, height = image.size
        else:
            raise TypeError(f"Unsupported image reference {type(reference)!r}")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    return int(height), int(width)


def _molmo2_grid_signature(
    reference: Any, *, max_crops: int = DEFAULT_MAX_CROPS
) -> tuple[int, ...]:
    """Derive Molmo2's pooled grid from dimensions without materializing crop tensors."""
    height, width = _image_dimensions(reference)
    patch_size = 14
    crop_patch_size = 378 // patch_size
    overlap_patches = 8
    crop_window_patches = crop_patch_size - overlap_patches
    crop_window_size = crop_window_patches * patch_size
    tiling = _select_tiling(
        height - overlap_patches * patch_size,
        width - overlap_patches * patch_size,
        crop_window_size,
        max_crops,
    )
    high_resolution_height = int(tiling[0]) * crop_window_patches + overlap_patches
    high_resolution_width = int(tiling[1]) * crop_window_patches + overlap_patches
    pooled_low_resolution = (crop_patch_size + 1) // 2
    return (
        pooled_low_resolution,
        pooled_low_resolution,
        (high_resolution_height + 1) // 2,
        (high_resolution_width + 1) // 2,
    )


def _ordered_projection_sha256(examples: Sequence[AcademicExample]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(_canonical_bytes(example.annotation()))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    return value.strip()


def _option_text(value: Any, *, name: str, allow_blank: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    output = value.strip()
    if not output and not allow_blank:
        raise ValueError(f"{name} must be non-blank")
    return output


def _answers(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise TypeError(f"{name} must be a string or string sequence")
    output = tuple(_validate_text(item, name=f"{name} answer") for item in values)
    if not output:
        raise ValueError(f"{name} contains no answers")
    return output


def _validate_examples(task: str, examples: Sequence[AcademicExample]) -> None:
    if not examples:
        raise ValueError(f"Academic task {task!r} is empty")
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Academic task {task!r} contains duplicate example IDs")
    for example in examples:
        if example.task != task:
            raise ValueError(f"Academic task projection drifted for {example.example_id!r}")
        _validate_text(example.question, name=f"{task}/{example.example_id} question")
        if task in ("ai2d", "a_okvqa_mc"):
            if not 2 <= len(example.options) <= len(string.ascii_uppercase):
                raise ValueError(f"{task}/{example.example_id} has an invalid option count")
            if example.answer_index is None or not 0 <= example.answer_index < len(example.options):
                raise ValueError(f"{task}/{example.example_id} has an invalid answer index")
        elif not example.answers:
            raise ValueError(f"{task}/{example.example_id} contains no gold answers")


def _ai2d_examples_from_raw(raw: Any) -> list[AcademicExample]:
    """Project AI2D rows without eagerly decoding and retaining validation images."""
    output = []
    for index in range(len(raw)):
        row = raw[index]
        output.append(
            AcademicExample(
                task="ai2d",
                example_id=_validate_text(row["question_id"], name="AI2D question_id"),
                source_position=str(index),
                visual=row["image"],
                image_reference=row["image"],
                question=_validate_text(row["question"], name="AI2D question"),
                options=tuple(
                    _option_text(value, name="AI2D option", allow_blank=True)
                    for value in row["answer_texts"]
                ),
                answer_index=int(row["correct_answer"]),
                stratum=("transparent" if bool(row.get("has_transparent_box")) else "standard"),
            )
        )
    return output


def _ai2d_examples() -> list[AcademicExample]:
    from datasets import Image as HFImage

    from olmo_core.data.multimodal.dataset_compat import load_from_disk_compat

    root = Path(academic_registry.ACADEMIC_DATASETS) / "ai2d"
    raw = load_from_disk_compat(root)["validation"].cast_column("image", HFImage(decode=False))
    return _ai2d_examples_from_raw(raw)


def _load_task_examples(task: str) -> list[AcademicExample]:
    output: list[AcademicExample] = []
    if task == "vqav2":
        rows = academic_registry.ACADEMIC_REGISTRY["coco_2014_vqa_multi"].loader("validation")
        for image_index, row in enumerate(rows):
            for question_index, message in enumerate(row["messages"]):
                output.append(
                    AcademicExample(
                        task=task,
                        example_id=str(message["question_id"]),
                        source_position=f"{image_index}:{question_index}",
                        visual=row["image"],
                        image_reference=row["image"],
                        question=_validate_text(message["question"], name="VQAv2 question"),
                        answers=_answers(message["answers"], name="VQAv2"),
                    )
                )
    elif task in ("textvqa", "docvqa", "chartqa", "a_okvqa_mc"):
        registry_name = {
            "textvqa": "text_vqa",
            "docvqa": "doc_qa",
            "chartqa": "chart_qa_weighted",
            "a_okvqa_mc": "a_okvqa_mc",
        }[task]
        rows = academic_registry.ACADEMIC_REGISTRY[registry_name].loader("validation")
        for index, row in enumerate(rows):
            metadata = row.get("metadata", {})
            raw_id = metadata.get("example_id")
            if task == "chartqa":
                kind = "human" if bool(metadata.get("is_human")) else "augmented"
                example_id = f"{kind}:{index:06d}"
            else:
                example_id = str(raw_id)
                if not example_id or example_id == "None":
                    raise ValueError(f"{task} row {index} lacks an example ID")
            common = {
                "task": task,
                "example_id": example_id,
                "source_position": str(index),
                "visual": row["image"],
                "image_reference": row["image"],
                "question": _validate_text(row["question"], name=f"{task} question"),
            }
            if task == "a_okvqa_mc":
                output.append(
                    AcademicExample(
                        **common,
                        options=tuple(
                            _option_text(value, name="A-OKVQA option", allow_blank=True)
                            for value in row["options"]
                        ),
                        answer_index=int(row["answer_idx"]),
                    )
                )
            else:
                output.append(
                    AcademicExample(
                        **common,
                        answers=_answers(row["answers"], name=task),
                        stratum=(
                            ("human" if bool(metadata.get("is_human")) else "augmented")
                            if task == "chartqa"
                            else None
                        ),
                    )
                )
    elif task == "ai2d":
        output = _ai2d_examples()
    else:
        raise ValueError(f"Unsupported academic task {task!r}")
    _validate_examples(task, output)
    return output


def _source_files(task: str) -> list[Path]:
    if task == "vqav2":
        return [Path(academic_registry.VQA2_SOURCE) / "molmo_val.json"]
    if task == "textvqa":
        return [Path(academic_registry.TEXT_VQA_SOURCE) / "TextVQA_0.5.1_val.json"]
    if task == "docvqa":
        return [Path(academic_registry.DOCQA_SOURCE) / "val_v1.0_withQT.json"]
    if task == "chartqa":
        root = Path(academic_registry.CHARTQA_SOURCE) / "val"
        return [root / "val_human.json", root / "val_augmented.json"]
    if task == "a_okvqa_mc":
        return [Path(academic_registry.A_OKVQA_SOURCE) / "aokvqa_v1p0_val.json"]
    if task == "ai2d":
        root = Path(academic_registry.ACADEMIC_DATASETS) / "ai2d"
        files = [root / "dataset_dict.json"]
        files.extend(sorted((root / "validation").glob("data-*.arrow")))
        files.extend(
            [root / "validation" / "dataset_info.json", root / "validation" / "state.json"]
        )
        return files
    raise ValueError(f"Unsupported academic task {task!r}")


def _source_identity(task: str, examples: Sequence[AcademicExample]) -> dict[str, Any]:
    files = [_file_identity(path) for path in _source_files(task)]
    return {
        "split": "validation",
        "examples": len(examples),
        "ordered_annotation_projection_sha256": _ordered_projection_sha256(examples),
        "files": files,
        "files_sha256": _canonical_sha256(files),
    }


_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ARTICLES = frozenset(("a", "an", "the"))
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(?<=\d)(,)+(?=\d)")
_PUNCTUATION = (
    ";",
    "/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
)


def _normalize_vqa_answer(value: str) -> str:
    text = value.replace("\n", " ").replace("\t", " ").strip()
    processed = text
    for punctuation in _PUNCTUATION:
        if (
            punctuation + " " in text
            or " " + punctuation in text
            or _COMMA_STRIP.search(text) is not None
        ):
            processed = processed.replace(punctuation, "")
        else:
            processed = processed.replace(punctuation, " ")
    processed = _PERIOD_STRIP.sub("", processed)
    tokens = []
    for token in processed.lower().split():
        token = _NUMBER_MAP.get(token, token)
        if token not in _ARTICLES:
            tokens.append(_CONTRACTIONS.get(token, token))
    return " ".join(tokens)


def _normalize_textvqa_answer(value: str) -> str:
    """Apply TextVQA's word-tokenize pass before VQA answer normalization."""
    tokenized = value.lower().replace(",", "").replace("?", "").replace("'s", " 's").strip()
    return _normalize_vqa_answer(tokenized)


def _vqa_consensus_accuracy(
    prediction: str,
    answers: Sequence[str],
    *,
    normalizer: Any,
) -> float:
    predicted = normalizer(prediction)
    normalized = [normalizer(answer) for answer in answers]
    if not normalized:
        raise ValueError("VQA accuracy requires at least one reference answer")
    scores = []
    for index in range(len(normalized)):
        matches = sum(
            answer == predicted
            for other_index, answer in enumerate(normalized)
            if other_index != index
        )
        scores.append(min(1.0, float(matches) / 3.0))
    return float(sum(scores) / len(scores))


def _vqa_accuracy(prediction: str, answers: Sequence[str]) -> float:
    return _vqa_consensus_accuracy(
        prediction,
        answers,
        normalizer=_normalize_vqa_answer,
    )


def _textvqa_accuracy(prediction: str, answers: Sequence[str]) -> float:
    return _vqa_consensus_accuracy(
        prediction,
        answers,
        normalizer=_normalize_textvqa_answer,
    )


def _levenshtein_distance(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, right_character in enumerate(right, start=1):
        current = [row]
        for column, left_character in enumerate(left, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _anls(prediction: str, answers: Sequence[str], *, threshold: float = 0.5) -> float:
    predicted = prediction.lower().strip()
    if not predicted:
        return 0.0
    similarities = []
    for answer in answers:
        target = answer.lower().strip()
        denominator = max(len(predicted), len(target))
        similarity = (
            1.0
            if denominator == 0
            else 1.0 - _levenshtein_distance(predicted, target) / denominator
        )
        similarities.append(similarity)
    score = max(similarities)
    return float(score if score > threshold else 0.0)


def _chartqa_relaxed_accuracy(
    prediction: str,
    target: str,
    *,
    max_relative_change: float = 0.05,
) -> float:
    def to_float(text: str) -> float | None:
        text = text.strip()
        try:
            return float(text[:-1]) / 100.0 if text.endswith("%") else float(text)
        except ValueError:
            return None

    predicted_float = to_float(prediction)
    target_float = to_float(target)
    if predicted_float is not None and target_float is not None and target_float != 0.0:
        return float(abs(predicted_float - target_float) / abs(target_float) <= max_relative_change)
    return float(prediction.strip().lower() == target.strip().lower())


def _metric_name(task: str) -> str:
    if task in ("vqav2", "textvqa"):
        return "vqa_accuracy"
    if task == "docvqa":
        return "anls"
    if task == "chartqa":
        return "relaxed_accuracy"
    if task in ("ai2d", "a_okvqa_mc"):
        return "multiple_choice_accuracy"
    raise ValueError(f"Unsupported academic task {task!r}")


def _score_prediction(
    example: AcademicExample,
    *,
    prediction: str,
    predicted_index: int | None,
) -> float:
    if example.task == "vqav2":
        return _vqa_accuracy(prediction, example.answers)
    if example.task == "textvqa":
        return _textvqa_accuracy(prediction, example.answers)
    if example.task == "docvqa":
        return _anls(prediction, example.answers)
    if example.task == "chartqa":
        return _chartqa_relaxed_accuracy(prediction, example.answers[0])
    if example.task == "ai2d":
        if predicted_index is None or example.answer_index is None:
            raise ValueError("Multiple-choice scoring requires prediction and target indices")
        return float(predicted_index == example.answer_index)
    if example.task == "a_okvqa_mc":
        if predicted_index is None or example.answer_index is None:
            raise ValueError("Multiple-choice scoring requires prediction and target indices")
        return float(example.options[predicted_index] == example.options[example.answer_index])
    raise ValueError(f"Unsupported academic task {example.task!r}")


def _load_train_inventory(path: Path) -> tuple[set[str], dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Configured alignment train-image inventory is missing: {path}")
    try:
        values = tuple(path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not read train-image inventory {path}: {error}") from error
    if not values or any(_SHA256_RE.fullmatch(value) is None for value in values):
        raise ValueError("Train-image inventory contains an invalid SHA-256 row")
    if tuple(sorted(set(values))) != values:
        raise ValueError("Train-image inventory must be sorted and unique")
    identity = _file_identity(path)
    identity["count"] = len(values)
    return set(values), identity


def _select_examples(
    examples: Sequence[AcademicExample],
    *,
    task: str,
    seed: int,
    maximum: int,
) -> list[AcademicExample]:
    if maximum < 0:
        raise ValueError("examples-per-task must be non-negative")
    if maximum == 0 or maximum >= len(examples):
        return list(examples)
    ranked = sorted(
        examples,
        key=lambda example: (
            hashlib.sha256(f"{seed}\0{task}\0{example.example_id}".encode()).digest(),
            example.example_id,
        ),
    )
    return ranked[:maximum]


def _ai2d_base_diagram_id(example_id: str) -> str:
    """Return the source-diagram identity shared by AI2D question/render variants."""
    match = re.fullmatch(r"(.+\.png)-[0-9]+(?:_transparent)?", example_id)
    if match is None:
        raise ValueError(f"AI2D example ID does not encode a base diagram: {example_id!r}")
    return match.group(1)


def _donor_pairing_key(
    example: AcademicExample, grid_signature: tuple[int, ...]
) -> tuple[tuple[int, ...], str | None]:
    return grid_signature, example.stratum


def _donor_content_group(example: AcademicExample, image_hash: str) -> str:
    return _ai2d_base_diagram_id(example.example_id) if example.task == "ai2d" else image_hash


def _grid_compatible_selection(
    examples: Sequence[AcademicExample],
    *,
    task: str,
    seed: int,
    maximum: int,
    resolve_image: Any,
) -> tuple[list[AcademicExample], dict[str, str], dict[str, tuple[int, ...]], dict[str, Any]]:
    """Exclude nonviable donor-pairing strata and backfill from the ranked tail."""
    if maximum <= 0 or maximum > len(examples):
        raise ValueError("Grid-compatible selection requires a positive in-range task cap")
    ranked = sorted(
        examples,
        key=lambda example: (
            hashlib.sha256(f"{seed}\0{task}\0{example.example_id}".encode()).digest(),
            example.example_id,
        ),
    )
    initial = ranked[:maximum]
    image_hashes: dict[str, str] = {}
    grid_signatures: dict[str, tuple[int, ...]] = {}

    def resolve(example: AcademicExample) -> None:
        image_hash, grid_signature = resolve_image(example)
        image_hashes[example.example_id] = image_hash
        grid_signatures[example.example_id] = grid_signature

    for example in initial:
        resolve(example)
    hashes_by_pairing: dict[tuple[tuple[int, ...], str | None], set[str]] = {}
    content_groups_by_pairing: dict[tuple[tuple[int, ...], str | None], set[str]] = {}
    for example in initial:
        image_hash = image_hashes[example.example_id]
        pairing_key = _donor_pairing_key(example, grid_signatures[example.example_id])
        hashes_by_pairing.setdefault(pairing_key, set()).add(image_hash)
        content_groups_by_pairing.setdefault(pairing_key, set()).add(
            _donor_content_group(example, image_hash)
        )
    viable_pairings = {
        pairing_key
        for pairing_key, hashes in hashes_by_pairing.items()
        if len(hashes) >= 2 and len(content_groups_by_pairing[pairing_key]) >= 2
    }
    excluded = [
        example
        for example in initial
        if _donor_pairing_key(example, grid_signatures[example.example_id]) not in viable_pairings
    ]
    selected = [
        example
        for example in initial
        if _donor_pairing_key(example, grid_signatures[example.example_id]) in viable_pairings
    ]
    backfilled = []
    remaining_by_stratum: dict[str | None, int] = {}
    for example in excluded:
        remaining_by_stratum[example.stratum] = remaining_by_stratum.get(example.stratum, 0) + 1
    for example in ranked[maximum:]:
        if len(selected) == maximum:
            break
        resolve(example)
        if (
            remaining_by_stratum.get(example.stratum, 0) > 0
            and _donor_pairing_key(example, grid_signatures[example.example_id]) in viable_pairings
        ):
            selected.append(example)
            backfilled.append(example)
            remaining_by_stratum[example.stratum] -= 1
    if len(selected) != maximum:
        raise ValueError(f"Task {task!r} cannot backfill a grid-compatible panel of {maximum}")
    selected_ids = {example.example_id for example in selected}
    image_hashes = {key: value for key, value in image_hashes.items() if key in selected_ids}
    grid_signatures = {key: value for key, value in grid_signatures.items() if key in selected_ids}

    def stratum_counts(values: Sequence[AcademicExample]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in values:
            label = value.stratum if value.stratum is not None else "<none>"
            counts[label] = counts.get(label, 0) + 1
        return {label: counts[label] for label in sorted(counts)}

    audit = {
        "rule": (
            "rank by sha256(seed\\0task\\0example_id); pair within exact Molmo2 pooled-grid "
            "and task stratum; exclude initial rows whose pairing stratum has fewer than two "
            "unique image hashes or donor content groups; backfill in rank order only from "
            "already viable pairing strata while preserving initial task-stratum counts; "
            "AI2D content group is source base diagram"
        ),
        "initial_count": len(initial),
        "excluded_nonviable_pairing_count": len(excluded),
        "excluded_nonviable_pairing_ids": [example.example_id for example in excluded],
        "backfilled_count": len(backfilled),
        "backfilled_ids": [example.example_id for example in backfilled],
        "final_count": len(selected),
        "initial_stratum_counts": stratum_counts(initial),
        "final_stratum_counts": stratum_counts(selected),
    }
    if audit["initial_stratum_counts"] != audit["final_stratum_counts"]:
        raise AssertionError("Grid-compatible backfill changed task-stratum counts")
    return selected, image_hashes, grid_signatures, audit


def _shuffle_donors(
    selected: Sequence[AcademicExample],
    image_hashes: Mapping[str, str],
    grid_signatures: Mapping[str, tuple[int, ...]],
) -> dict[str, str]:
    first_by_pairing_hash: dict[tuple[tuple[tuple[int, ...], str | None], str], str] = {}
    for example in selected:
        pairing_key = _donor_pairing_key(example, grid_signatures[example.example_id])
        key = (pairing_key, image_hashes[example.example_id])
        first_by_pairing_hash.setdefault(key, example.example_id)
    hashes_by_pairing: dict[tuple[tuple[int, ...], str | None], list[str]] = {}
    for pairing_key, image_hash in first_by_pairing_hash:
        hashes_by_pairing.setdefault(pairing_key, []).append(image_hash)
    selected_by_id = {example.example_id: example for example in selected}
    donors = {}
    for example in selected:
        example_id = example.example_id
        image_hash = image_hashes[example_id]
        pairing_key = _donor_pairing_key(example, grid_signatures[example_id])
        unique_hashes = sorted(hashes_by_pairing[pairing_key])
        if len(unique_hashes) < 2:
            raise ValueError(
                "Shuffled-image control requires two unique images in every pairing stratum"
            )
        start = unique_hashes.index(image_hash)
        recipient_group = _donor_content_group(example, image_hash)
        for offset in range(1, len(unique_hashes)):
            donor_hash = unique_hashes[(start + offset) % len(unique_hashes)]
            donor_id = first_by_pairing_hash[(pairing_key, donor_hash)]
            donor = selected_by_id[donor_id]
            if _donor_content_group(donor, donor_hash) != recipient_group:
                donors[example_id] = donor_id
                break
        else:
            raise ValueError(
                "Shuffled-image control lacks a distinct donor content group in one stratum"
            )
    if any(
        image_hashes[recipient] == image_hashes[donor]
        or grid_signatures[recipient] != grid_signatures[donor]
        or selected_by_id[recipient].stratum != selected_by_id[donor].stratum
        or _donor_content_group(selected_by_id[recipient], image_hashes[recipient])
        == _donor_content_group(selected_by_id[donor], image_hashes[donor])
        for recipient, donor in donors.items()
    ):
        raise AssertionError("Deterministic shuffled-image mapping changed pairing semantics")
    return donors


def _repo_relative_file_identity(path: Path, repo_root: Path) -> dict[str, Any]:
    """Return a checkout-portable identity for one repository file."""
    path = path.resolve()
    repo_root = repo_root.resolve()
    try:
        relative = path.relative_to(repo_root).as_posix()
    except ValueError as error:
        raise ValueError(f"Implementation file {path} is outside repository {repo_root}") from error
    identity = _file_identity(path)
    return {
        "repo_relative_path": relative,
        "bytes": identity["bytes"],
        "sha256": identity["sha256"],
    }


def _implementation_identity() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    source_root = repo_root / "src"
    paths = {
        "evaluator": Path(__file__).resolve(),
        "academic_registry": source_root / "olmo_core/data/multimodal/academic/registry.py",
        "document_layout": source_root / "olmo_core/data/multimodal/document_layout.py",
        "image_processor": source_root / "olmo_core/nn/vision/molmo2_image_processor.py",
        "checkpoint_loader": Path(__file__).resolve().with_name("s002_downstream.py"),
        "saved_endpoint_validator": Path(__file__)
        .resolve()
        .with_name("vision_alignment_joint_matched_wrong_saved_steps_validate.py"),
    }
    files = {name: _repo_relative_file_identity(path, repo_root) for name, path in paths.items()}
    return {"files": files, "files_sha256": _canonical_sha256(files)}


def _build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.tasks) != len(set(args.tasks)) or any(
        task not in DEFAULT_TASKS for task in args.tasks
    ):
        raise ValueError(f"--tasks must be unique names from {DEFAULT_TASKS}")
    if args.selection_seed < 0:
        raise ValueError("--selection-seed must be non-negative")
    if args.examples_per_task < 2:
        raise ValueError("--examples-per-task must be at least two")
    git_identity = _git_revision()
    _validate_git_identity(git_identity)
    train_hashes, train_identity = _load_train_inventory(Path(args.train_image_inventory))
    requested_confirmatory_panel = (
        tuple(args.tasks) == DEFAULT_TASKS
        and args.examples_per_task == DEFAULT_EXAMPLES_PER_TASK
        and args.selection_seed == DEFAULT_SELECTION_SEED
    )
    expected_train_identity = {
        "path": str(EXPECTED_TRAIN_IMAGE_INVENTORY.resolve()),
        "bytes": EXPECTED_TRAIN_IMAGE_INVENTORY_BYTES,
        "sha256": EXPECTED_TRAIN_IMAGE_INVENTORY_SHA256,
        "count": EXPECTED_TRAIN_IMAGE_INVENTORY_COUNT,
    }
    if requested_confirmatory_panel and train_identity != expected_train_identity:
        raise ValueError("Confirmatory panel requires the exact alignment train-image inventory")
    tasks: dict[str, Any] = {}
    image_hash_cache: dict[str, str] = {}
    grid_signature_cache: dict[str, tuple[int, ...]] = {}
    sampled = False
    for task in args.tasks:
        log.info("Loading and hashing %s validation", task)
        examples = _load_task_examples(task)
        source = _source_identity(task, examples)

        def resolve_image(example: AcademicExample) -> tuple[str, tuple[int, ...]]:
            cache_key = (
                str(Path(example.image_reference).resolve())
                if isinstance(example.image_reference, str)
                else f"embedded:{example.task}:{example.example_id}"
            )
            image_hash = image_hash_cache.get(cache_key)
            if image_hash is None:
                image_hash = _image_reference_sha256(example.image_reference)
                image_hash_cache[cache_key] = image_hash
            grid_signature = grid_signature_cache.get(image_hash)
            if grid_signature is None:
                grid_signature = _molmo2_grid_signature(example.image_reference)
                grid_signature_cache[image_hash] = grid_signature
            return image_hash, grid_signature

        selected, image_hashes, grid_signatures, grid_selection = _grid_compatible_selection(
            examples,
            task=task,
            seed=args.selection_seed,
            maximum=args.examples_per_task,
            resolve_image=resolve_image,
        )
        sampled |= len(selected) != len(examples)
        donors = _shuffle_donors(selected, image_hashes, grid_signatures)
        records = []
        for example in selected:
            donor_id = donors[example.example_id]
            records.append(
                {
                    "example_id": example.example_id,
                    "source_position": example.source_position,
                    "annotation_sha256": _canonical_sha256(example.annotation()),
                    "image_sha256": image_hashes[example.example_id],
                    "image_grid_signature": list(grid_signatures[example.example_id]),
                    "image_token_count": len(
                        build_image_token_ids(*grid_signatures[example.example_id])
                    ),
                    "alignment_train_image_overlap": image_hashes[example.example_id]
                    in train_hashes,
                    "shuffled_donor_id": donor_id,
                    "shuffled_image_sha256": image_hashes[donor_id],
                    "shuffled_image_grid_signature": list(grid_signatures[donor_id]),
                    "shuffled_alignment_train_image_overlap": image_hashes[donor_id]
                    in train_hashes,
                }
            )
        tasks[task] = {
            "source": source,
            "grid_selection": grid_selection,
            "selection_count": len(records),
            "selection_sha256": _canonical_sha256(records),
            "records": records,
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "format": MANIFEST_FORMAT,
        "protocol_name": PROTOCOL_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": git_identity,
        "implementation": _implementation_identity(),
        "selection": {
            "split": "validation",
            "tasks": list(args.tasks),
            "seed": args.selection_seed,
            "examples_per_task_limit": args.examples_per_task,
            "partial": sampled,
            "panel_status": "confirmatory" if requested_confirmatory_panel else "diagnostic",
            "ranking": "sha256(seed\\0task\\0example_id), then example_id",
        },
        "controls": {
            "names": list(CONTROLS),
            "shuffled": (
                "next lexicographically sorted unique image SHA-256 within exact task and "
                "Molmo2 pooled-grid/task-stratum pairing, skipping the recipient's donor "
                "content group (AI2D source base diagram)"
            ),
            "blank": "solid RGB(0,0,0) image at the recipient image dimensions",
        },
        "contamination": {
            "method": "exact encoded-image-byte SHA-256 intersection",
            "alignment_train_image_inventory": train_identity,
            "reported_subset": "exact_byte_nonoverlap",
            "limitation": "exact-byte non-overlap is not semantic contamination cleanliness",
        },
        "tasks": tasks,
    }
    return _attach_content_sha256(payload)


def _validate_manifest_and_load_examples(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, AcademicExample]], dict[str, Any]]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _load_json_strict(manifest_path)
    _verify_content_sha256(manifest, name="academic selection manifest")
    _exact_mapping(
        manifest,
        (
            "schema_version",
            "format",
            "protocol_name",
            "created_at",
            "git",
            "implementation",
            "selection",
            "controls",
            "contamination",
            "tasks",
            "content_sha256",
        ),
        name="academic selection manifest",
    )
    _validate_timestamp(manifest["created_at"], name="academic selection manifest created_at")
    git = _exact_mapping(
        manifest["git"], ("revision", "dirty"), name="academic selection manifest git"
    )
    _validate_git_identity(git)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("format") != MANIFEST_FORMAT
        or manifest.get("protocol_name") != PROTOCOL_NAME
    ):
        raise ValueError("Academic selection manifest protocol identity differs")
    if manifest.get("implementation") != _implementation_identity():
        raise ValueError("Academic selection manifest evaluator implementation differs")
    selection = _exact_mapping(
        manifest.get("selection"),
        (
            "split",
            "tasks",
            "seed",
            "examples_per_task_limit",
            "partial",
            "panel_status",
            "ranking",
        ),
        name="academic selection manifest selection",
    )
    if selection.get("split") != "validation":
        raise ValueError("Academic selection manifest must pin the validation split")
    seed = selection.get("seed")
    maximum = selection.get("examples_per_task_limit")
    if type(seed) is not int or seed < 0 or type(maximum) is not int or maximum < 2:
        raise ValueError("Academic selection manifest has invalid selection parameters")
    task_names = selection.get("tasks")
    if (
        not isinstance(task_names, list)
        or not task_names
        or len(task_names) != len(set(task_names))
        or any(task not in DEFAULT_TASKS for task in task_names)
    ):
        raise ValueError("Academic selection manifest task list is invalid")
    requested_confirmatory_panel = (
        tuple(task_names) == DEFAULT_TASKS
        and seed == DEFAULT_SELECTION_SEED
        and maximum == DEFAULT_EXAMPLES_PER_TASK
    )
    expected_panel_status = "confirmatory" if requested_confirmatory_panel else "diagnostic"
    if (
        selection.get("panel_status") != expected_panel_status
        or selection.get("ranking") != "sha256(seed\\0task\\0example_id), then example_id"
    ):
        raise ValueError("Academic selection manifest panel status or ranking differs")

    controls = _exact_mapping(
        manifest.get("controls"),
        ("names", "shuffled", "blank"),
        name="academic selection manifest controls",
    )
    if controls != {
        "names": list(CONTROLS),
        "shuffled": (
            "next lexicographically sorted unique image SHA-256 within exact task and "
            "Molmo2 pooled-grid/task-stratum pairing, skipping the recipient's donor "
            "content group (AI2D source base diagram)"
        ),
        "blank": "solid RGB(0,0,0) image at the recipient image dimensions",
    }:
        raise ValueError("Academic selection manifest controls differ")

    contamination = _exact_mapping(
        manifest.get("contamination"),
        ("method", "alignment_train_image_inventory", "reported_subset", "limitation"),
        name="academic selection manifest contamination",
    )
    if (
        contamination.get("method") != "exact encoded-image-byte SHA-256 intersection"
        or contamination.get("reported_subset") != "exact_byte_nonoverlap"
        or contamination.get("limitation")
        != "exact-byte non-overlap is not semantic contamination cleanliness"
    ):
        raise ValueError("Academic selection manifest contamination protocol differs")
    expected_inventory = contamination.get("alignment_train_image_inventory")
    if not isinstance(expected_inventory, dict) or not isinstance(
        expected_inventory.get("path"), str
    ):
        raise TypeError("Academic selection manifest lacks a train-image inventory path")
    train_hashes, actual_inventory = _load_train_inventory(Path(expected_inventory["path"]))
    if actual_inventory != expected_inventory:
        raise ValueError("Alignment train-image inventory identity differs from the manifest")
    if requested_confirmatory_panel and actual_inventory != {
        "path": str(EXPECTED_TRAIN_IMAGE_INVENTORY.resolve()),
        "bytes": EXPECTED_TRAIN_IMAGE_INVENTORY_BYTES,
        "sha256": EXPECTED_TRAIN_IMAGE_INVENTORY_SHA256,
        "count": EXPECTED_TRAIN_IMAGE_INVENTORY_COUNT,
    }:
        raise ValueError("Confirmatory manifest does not bind the exact train-image inventory")

    manifest_tasks = manifest.get("tasks")
    if not isinstance(manifest_tasks, dict) or set(manifest_tasks) != set(task_names):
        raise ValueError("Academic selection manifest task payloads differ from its task list")
    loaded: dict[str, dict[str, AcademicExample]] = {}
    image_hash_cache: dict[str, str] = {}
    grid_signature_cache: dict[str, tuple[int, ...]] = {}
    partial = False
    for task in task_names:
        log.info("Revalidating %s against the immutable selection manifest", task)
        examples = _load_task_examples(task)
        task_payload = manifest_tasks[task]
        task_payload = _exact_mapping(
            task_payload,
            ("source", "grid_selection", "selection_count", "selection_sha256", "records"),
            name=f"academic manifest task {task!r}",
        )
        if task_payload.get("source") != _source_identity(task, examples):
            raise ValueError(f"Academic task {task!r} source identity differs")

        def resolve_image(example: AcademicExample) -> tuple[str, tuple[int, ...]]:
            cache_key = (
                str(Path(example.image_reference).resolve())
                if isinstance(example.image_reference, str)
                else f"embedded:{example.task}:{example.example_id}"
            )
            image_hash = image_hash_cache.get(cache_key)
            if image_hash is None:
                image_hash = _image_reference_sha256(example.image_reference)
                image_hash_cache[cache_key] = image_hash
            grid_signature = grid_signature_cache.get(image_hash)
            if grid_signature is None:
                grid_signature = _molmo2_grid_signature(example.image_reference)
                grid_signature_cache[image_hash] = grid_signature
            return image_hash, grid_signature

        (
            expected_selected,
            expected_image_hashes,
            expected_grid_signatures,
            expected_grid_selection,
        ) = _grid_compatible_selection(
            examples,
            task=task,
            seed=seed,
            maximum=maximum,
            resolve_image=resolve_image,
        )
        if task_payload.get("grid_selection") != expected_grid_selection:
            raise ValueError(f"Academic task {task!r} grid selection was not rederived")
        records = task_payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"Academic task {task!r} has no selection records")
        if task_payload.get("selection_count") != len(records) or task_payload.get(
            "selection_sha256"
        ) != _canonical_sha256(records):
            raise ValueError(f"Academic task {task!r} selection identity differs")
        partial |= len(records) != len(examples)
        by_id = {example.example_id: example for example in examples}
        selected: dict[str, AcademicExample] = {}
        record_ids = []
        record_by_id: dict[str, Mapping[str, Any]] = {}
        selected_image_hashes: dict[str, str] = {}
        for raw_record in records:
            raw_record = _exact_mapping(
                raw_record,
                (
                    "example_id",
                    "source_position",
                    "annotation_sha256",
                    "image_sha256",
                    "image_grid_signature",
                    "image_token_count",
                    "alignment_train_image_overlap",
                    "shuffled_donor_id",
                    "shuffled_image_sha256",
                    "shuffled_image_grid_signature",
                    "shuffled_alignment_train_image_overlap",
                ),
                name=f"academic task {task!r} selection row",
            )
            example_id = raw_record.get("example_id")
            if not isinstance(example_id, str) or example_id not in by_id:
                raise ValueError(f"Academic task {task!r} selected an unavailable example")
            if example_id in selected:
                raise ValueError(f"Academic task {task!r} selected a duplicate example")
            example = by_id[example_id]
            if raw_record.get("source_position") != example.source_position or raw_record.get(
                "annotation_sha256"
            ) != _canonical_sha256(example.annotation()):
                raise ValueError(f"Academic task {task!r}/{example_id} annotation differs")
            image_hash = expected_image_hashes[example_id]
            if raw_record.get("image_sha256") != image_hash:
                raise ValueError(f"Academic task {task!r}/{example_id} image differs")
            grid_signature = expected_grid_signatures[example_id]
            if raw_record.get("image_grid_signature") != list(grid_signature) or raw_record.get(
                "image_token_count"
            ) != len(build_image_token_ids(*grid_signature)):
                raise ValueError(f"Academic task {task!r}/{example_id} grid identity differs")
            if raw_record.get("alignment_train_image_overlap") != (image_hash in train_hashes):
                raise ValueError(f"Academic task {task!r}/{example_id} contamination flag differs")
            selected[example_id] = example
            selected_image_hashes[example_id] = image_hash
            record_ids.append(example_id)
            record_by_id[example_id] = raw_record
        expected_ids = [example.example_id for example in expected_selected]
        if record_ids != expected_ids:
            raise ValueError(f"Academic task {task!r} selected-ID order was not rederived")
        expected_donors = _shuffle_donors(
            [selected[example_id] for example_id in record_ids],
            selected_image_hashes,
            expected_grid_signatures,
        )
        for example_id in record_ids:
            record = record_by_id[example_id]
            donor_id = record.get("shuffled_donor_id")
            if (
                not isinstance(donor_id, str)
                or donor_id not in selected
                or donor_id != expected_donors[example_id]
            ):
                raise ValueError(f"Academic task {task!r}/{example_id} donor is unavailable")
            donor_hash = record_by_id[donor_id].get("image_sha256")
            donor_grid = record_by_id[donor_id].get("image_grid_signature")
            if (
                donor_hash != record.get("shuffled_image_sha256")
                or donor_hash == record.get("image_sha256")
                or donor_grid != record.get("shuffled_image_grid_signature")
                or donor_grid != record.get("image_grid_signature")
                or record.get("shuffled_alignment_train_image_overlap")
                != (donor_hash in train_hashes)
            ):
                raise ValueError(f"Academic task {task!r}/{example_id} donor identity differs")
        loaded[task] = selected
    if selection.get("partial") is not partial:
        raise ValueError("Academic selection manifest partial-coverage flag differs")
    return manifest, loaded, _file_identity(manifest_path)


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _aggregate_task_outputs(task: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_name = _metric_name(task)
    controls: dict[str, Any] = {}
    for control in CONTROLS:
        all_values = [float(row["controls"][control]["score"]) for row in rows]
        exact_byte_nonoverlap_values = [
            float(row["controls"][control]["score"])
            for row in rows
            if not (
                row["shuffled_alignment_train_image_overlap"]
                if control == "shuffled"
                else row["alignment_train_image_overlap"]
            )
        ]
        metrics: dict[str, Any] = {
            metric_name: _mean(all_values),
            f"exact_byte_nonoverlap_{metric_name}": _mean(exact_byte_nonoverlap_values),
            "examples": len(all_values),
            "exact_byte_nonoverlap_examples": len(exact_byte_nonoverlap_values),
        }
        strata = (
            ("human", "augmented")
            if task == "chartqa"
            else (("standard", "transparent") if task == "ai2d" else ())
        )
        for stratum in strata:
            stratum_values = [
                float(row["controls"][control]["score"])
                for row in rows
                if row["stratum"] == stratum
            ]
            exact_byte_nonoverlap_stratum_values = [
                float(row["controls"][control]["score"])
                for row in rows
                if row["stratum"] == stratum
                and not (
                    row["shuffled_alignment_train_image_overlap"]
                    if control == "shuffled"
                    else row["alignment_train_image_overlap"]
                )
            ]
            metrics[f"{metric_name}_{stratum}"] = _mean(stratum_values)
            metrics[f"exact_byte_nonoverlap_{metric_name}_{stratum}"] = _mean(
                exact_byte_nonoverlap_stratum_values
            )
        controls[control] = metrics
    correct = controls["correct"]
    exact_byte_nonoverlap_pair_values = {
        control: [
            float(row["controls"][control]["score"])
            for row in rows
            if not row["alignment_train_image_overlap"]
            and not row["shuffled_alignment_train_image_overlap"]
        ]
        for control in ("correct", "shuffled")
    }
    exact_byte_nonoverlap_blank_values = {
        control: [
            float(row["controls"][control]["score"])
            for row in rows
            if not row["alignment_train_image_overlap"]
        ]
        for control in ("correct", "blank")
    }
    exact_pair_correct = _mean(exact_byte_nonoverlap_pair_values["correct"])
    exact_pair_shuffled = _mean(exact_byte_nonoverlap_pair_values["shuffled"])
    exact_blank_correct = _mean(exact_byte_nonoverlap_blank_values["correct"])
    exact_blank = _mean(exact_byte_nonoverlap_blank_values["blank"])
    return {
        "metric": metric_name,
        "controls": controls,
        "image_control_deltas": {
            f"{metric_name}_correct_minus_shuffled": (
                correct[metric_name] - controls["shuffled"][metric_name]
            ),
            f"{metric_name}_correct_minus_blank": (
                correct[metric_name] - controls["blank"][metric_name]
            ),
            f"exact_byte_nonoverlap_{metric_name}_correct_minus_shuffled": (
                exact_pair_correct - exact_pair_shuffled
                if exact_pair_correct is not None and exact_pair_shuffled is not None
                else None
            ),
            f"exact_byte_nonoverlap_{metric_name}_correct_minus_blank": (
                exact_blank_correct - exact_blank
                if exact_blank_correct is not None and exact_blank is not None
                else None
            ),
            "exact_byte_nonoverlap_correct_minus_shuffled_examples": len(
                exact_byte_nonoverlap_pair_values["correct"]
            ),
            "exact_byte_nonoverlap_correct_minus_blank_examples": len(
                exact_byte_nonoverlap_blank_values["correct"]
            ),
        },
    }


def _generation_stop_counts(task: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if task in ("ai2d", "a_okvqa_mc"):
        return {}
    return {
        control: {
            "eos": sum(row["controls"][control]["stop_reason"] == "eos" for row in rows),
            "max_tokens": sum(
                row["controls"][control]["stop_reason"] == "max_tokens" for row in rows
            ),
        }
        for control in CONTROLS
    }


def _answer_token_coverage(
    loaded: Mapping[str, Mapping[str, AcademicExample]], tokenizer: Any
) -> dict[str, Any]:
    """Measure whether the frozen generation cap can represent every selected free answer."""
    coverage = {}
    for task in ("vqav2", "textvqa", "docvqa", "chartqa"):
        examples = loaded.get(task)
        if not isinstance(examples, Mapping) or not examples:
            raise ValueError(f"Answer-token coverage lacks frozen task {task!r}")
        rows = []
        response_lengths: list[int] = []
        response_lengths_with_eos: list[int] = []
        for example in examples.values():
            if not example.answers:
                raise ValueError(
                    f"Answer-token coverage lacks golds for {task}/{example.example_id}"
                )
            shortest = min(len(response_ids(tokenizer, answer)) for answer in example.answers)
            response_lengths.append(shortest)
            response_lengths_with_eos.append(shortest + 1)
            rows.append(
                {
                    "example_id": example.example_id,
                    "shortest_response_tokens": shortest,
                    "shortest_response_tokens_with_eos": shortest + 1,
                }
            )
        coverage[task] = {
            "selected": len(rows),
            "max_shortest_response_tokens": max(response_lengths),
            "max_shortest_response_tokens_with_eos": max(response_lengths_with_eos),
            "rows_exceeding_cap": sum(
                length > DEFAULT_MAX_NEW_TOKENS for length in response_lengths
            ),
            "rows_without_eos_room": sum(
                length > DEFAULT_MAX_NEW_TOKENS for length in response_lengths_with_eos
            ),
            "rows_over_8_response_tokens": sum(length > 8 for length in response_lengths),
            "ordered_rows_sha256": _canonical_sha256(rows),
        }
    return coverage


def _as_rgb_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB").copy()
    if isinstance(value, np.ndarray):
        return Image.fromarray(value.astype("uint8")).convert("RGB")
    if isinstance(value, Mapping):
        embedded = value.get("bytes")
        if isinstance(embedded, (bytes, bytearray, memoryview)) and embedded:
            import io

            with Image.open(io.BytesIO(bytes(embedded))) as image:
                return image.convert("RGB").copy()
        value = value.get("path")
    if isinstance(value, (str, os.PathLike)):
        with Image.open(value) as image:
            return image.convert("RGB").copy()
    raise ValueError(f"Unsupported runtime image value {type(value)!r}")


def _build_mc_prompt(question: str, options: Sequence[str]) -> str:
    if not 2 <= len(options) <= len(string.ascii_uppercase):
        raise ValueError("Multiple-choice prompt requires between 2 and 26 options")
    option_text = "\n".join(
        f"{letter}. {option if option else EMPTY_OPTION_DISPLAY}"
        for letter, option in zip(string.ascii_uppercase, options)
    )
    return (
        f"Question: {question}\nOptions:\n{option_text}\n"
        "Answer with only the option letter.\nAnswer:"
    )


def _free_answer_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


class _NativeAcademicInference:
    def __init__(
        self,
        train_module: Any,
        tokenizer: Any,
        token_ids: Molmo2TokenIds,
        *,
        max_sequence_length: int,
        max_crops: int,
        max_new_tokens: int,
        sequence_bucket_size: int,
    ) -> None:
        self.train_module = train_module
        self.model = train_module.model_parts[0]
        self.tokenizer = tokenizer
        self.token_ids = token_ids
        self.max_sequence_length = max_sequence_length
        self.max_crops = max_crops
        self.max_new_tokens = max_new_tokens
        self.sequence_bucket_size = sequence_bucket_size
        self.text_vocab_size = min(token_ids.image_token_ids)

    @property
    def device(self) -> torch.device:
        return self.train_module.device

    def _buffer_length(self, required: int) -> int:
        rounded = (
            (required + self.sequence_bucket_size - 1) // self.sequence_bucket_size
        ) * self.sequence_bucket_size
        return min(rounded, self.max_sequence_length)

    def _prepare_visual(
        self, image: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], tuple[int, ...]]:
        images, pooling, grid = preprocess_image_molmo2(
            image,
            dtype=torch.float32,
            device=torch.device("cpu"),
            max_crops=self.max_crops,
            is_training=False,
        )
        resized_h, resized_w, height, width = (int(grid[index]) for index in range(4))
        image_ids = build_image_token_ids(
            resized_h,
            resized_w,
            height,
            width,
            token_ids=self.token_ids,
        )
        return images, pooling, image_ids, (resized_h, resized_w, height, width)

    def _inputs(
        self,
        prompt_ids: Sequence[int],
        *,
        required: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if required > self.max_sequence_length:
            raise ValueError(
                f"Academic request needs {required} tokens, exceeding the frozen "
                f"maximum {self.max_sequence_length}"
            )
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Academic evaluator tokenizer has no pad or EOS token")
        buffer_length = self._buffer_length(required)
        input_ids = torch.full(
            (1, buffer_length),
            int(pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        input_ids[0, : len(prompt_ids)] = torch.tensor(prompt_ids, device=self.device)
        token_type_ids = torch.zeros_like(input_ids)
        prompt_tensor = input_ids[0, : len(prompt_ids)]
        for image_token_id in self.token_ids.image_token_ids:
            token_type_ids[0, : len(prompt_ids)] |= prompt_tensor.eq(image_token_id)
        position_ids = torch.arange(buffer_length, device=self.device).unsqueeze(0)
        return input_ids, token_type_ids, position_ids

    def _consensus_token(self, token_id: int) -> int:
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return token_id
        extrema = torch.tensor([token_id, token_id], dtype=torch.int64, device=self.device)
        dist.all_reduce(extrema[:1], op=dist.ReduceOp.MIN)
        dist.all_reduce(extrema[1:], op=dist.ReduceOp.MAX)
        if int(extrema[0]) != int(extrema[1]):
            raise RuntimeError(
                "Native EP ranks predicted different tokens: "
                f"min={int(extrema[0])}, max={int(extrema[1])}"
            )
        return int(extrema[0])

    def predict(
        self,
        example: AcademicExample,
        visual: Image.Image,
    ) -> dict[str, Any]:
        images, pooling, image_ids, grid_signature = self._prepare_visual(visual)
        is_multiple_choice = bool(example.options)
        prompt = (
            _build_mc_prompt(example.question, example.options)
            if is_multiple_choice
            else _free_answer_prompt(example.question)
        )
        prompt_ids = document_prompt_ids(self.tokenizer, prompt, image_ids=image_ids)
        with torch.inference_mode():
            encoded_features = self.model.encode_images(images, pooling)
        if is_multiple_choice:
            candidate_encodings = [
                response_ids(self.tokenizer, letter)
                for letter in string.ascii_uppercase[: len(example.options)]
            ]
            if any(len(ids) != 1 for ids in candidate_encodings):
                raise ValueError("Every academic option letter must encode to one response token")
            input_ids, token_type_ids, position_ids = self._inputs(
                prompt_ids,
                required=len(prompt_ids),
            )
            logits_position = torch.tensor(
                [[len(prompt_ids) - 1]], dtype=torch.long, device=self.device
            )
            with torch.inference_mode():
                logits = self.train_module.model_forward_no_pipeline(
                    input_ids,
                    encoded_image_features=encoded_features,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    logits_to_keep=logits_position,
                )
            if not isinstance(logits, torch.Tensor):
                raise TypeError(f"Expected academic logits tensor, got {type(logits).__name__}")
            candidate_ids = torch.tensor(
                [ids[0] for ids in candidate_encodings], dtype=torch.long, device=self.device
            )
            candidate_logits = logits[0, 0, candidate_ids].float()
            predicted_index = self._consensus_token(int(candidate_logits.argmax().item()))
            candidate_log_probabilities = candidate_logits.log_softmax(dim=-1).cpu().tolist()
            return {
                "prediction": string.ascii_uppercase[predicted_index],
                "predicted_index": predicted_index,
                "candidate_log_probabilities": {
                    letter: float(value)
                    for letter, value in zip(
                        string.ascii_uppercase[: len(example.options)],
                        candidate_log_probabilities,
                    )
                },
                "image_grid_signature": list(grid_signature),
                "image_token_count": len(image_ids),
                "image_token_ids_sha256": _canonical_sha256(image_ids),
                "input_tokens": len(prompt_ids),
                "output_tokens": 1,
            }

        required = len(prompt_ids) + self.max_new_tokens
        input_ids, token_type_ids, position_ids = self._inputs(prompt_ids, required=required)
        generated: list[int] = []
        stop_reason = "max_tokens"
        with torch.inference_mode():
            for _ in range(self.max_new_tokens):
                current_length = len(prompt_ids) + len(generated)
                logits_position = torch.tensor(
                    [[current_length - 1]], dtype=torch.long, device=self.device
                )
                logits = self.train_module.model_forward_no_pipeline(
                    input_ids,
                    encoded_image_features=encoded_features,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    logits_to_keep=logits_position,
                )
                if not isinstance(logits, torch.Tensor):
                    raise TypeError(f"Expected academic logits tensor, got {type(logits).__name__}")
                next_token = int(logits[0, 0, : self.text_vocab_size].argmax().item())
                next_token = self._consensus_token(next_token)
                generated.append(next_token)
                if next_token == self.tokenizer.eos_token_id:
                    stop_reason = "eos"
                    break
                input_ids[0, current_length] = next_token
        prediction = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {
            "prediction": prediction,
            "predicted_index": None,
            "generated_token_ids": generated,
            "stop_reason": stop_reason,
            "image_grid_signature": list(grid_signature),
            "image_token_count": len(image_ids),
            "image_token_ids_sha256": _canonical_sha256(image_ids),
            "input_tokens": len(prompt_ids),
            "output_tokens": len(generated),
        }


def _evaluate_manifest(
    inference: _NativeAcademicInference,
    manifest: Mapping[str, Any],
    loaded: Mapping[str, Mapping[str, AcademicExample]],
) -> dict[str, Any]:
    task_results: dict[str, Any] = {}
    task_names = manifest["selection"]["tasks"]
    for task in task_names:
        started = time.monotonic()
        manifest_task = manifest["tasks"][task]
        records = manifest_task["records"]
        examples = loaded[task]
        output_rows = []
        for row_index, record in enumerate(records):
            example = examples[record["example_id"]]
            donor = examples[record["shuffled_donor_id"]]
            control_outputs: dict[str, Any] = {}
            for control in CONTROLS:
                if control == "correct":
                    visual = _as_rgb_image(example.visual)
                elif control == "shuffled":
                    visual = _as_rgb_image(donor.visual)
                else:
                    recipient = _as_rgb_image(example.visual)
                    visual = Image.new("RGB", recipient.size, color=(0, 0, 0))
                    recipient.close()
                try:
                    prediction = inference.predict(example, visual)
                finally:
                    visual.close()
                if (
                    prediction["image_grid_signature"] != record["image_grid_signature"]
                    or prediction["image_token_count"] != record["image_token_count"]
                ):
                    raise ValueError(
                        f"{task}/{example.example_id}/{control} image-token layout differs"
                    )
                score = _score_prediction(
                    example,
                    prediction=prediction["prediction"],
                    predicted_index=prediction["predicted_index"],
                )
                control_outputs[control] = {**prediction, "score": score}
            if len({output["image_token_ids_sha256"] for output in control_outputs.values()}) != 1:
                raise ValueError(
                    f"{task}/{example.example_id} correct/shuffled/blank image IDs differ"
                )
            output_rows.append(
                {
                    "example_id": example.example_id,
                    "source_position": example.source_position,
                    "annotation_sha256": record["annotation_sha256"],
                    "image_sha256": record["image_sha256"],
                    "image_grid_signature": record["image_grid_signature"],
                    "image_token_count": record["image_token_count"],
                    "alignment_train_image_overlap": record["alignment_train_image_overlap"],
                    "shuffled_donor_id": record["shuffled_donor_id"],
                    "shuffled_image_sha256": record["shuffled_image_sha256"],
                    "shuffled_image_grid_signature": record["shuffled_image_grid_signature"],
                    "shuffled_alignment_train_image_overlap": record[
                        "shuffled_alignment_train_image_overlap"
                    ],
                    "question": example.question,
                    "gold_answers": list(example.answers),
                    "options": list(example.options),
                    "gold_answer_index": example.answer_index,
                    "stratum": example.stratum,
                    "controls": control_outputs,
                }
            )
            if get_rank() == 0 and (row_index == 0 or (row_index + 1) % 100 == 0):
                log.info("[%s] evaluated %d/%d examples", task, row_index + 1, len(records))
        aggregates = _aggregate_task_outputs(task, output_rows)
        task_results[task] = {
            "source": manifest_task["source"],
            "selection_count": len(records),
            "selection_sha256": manifest_task["selection_sha256"],
            "alignment_train_image_overlap_count": sum(
                bool(row["alignment_train_image_overlap"]) for row in output_rows
            ),
            "generation_stop_counts": _generation_stop_counts(task, output_rows),
            "elapsed_seconds": time.monotonic() - started,
            **aggregates,
            "examples": output_rows,
        }
        if get_rank() == 0:
            log.info("Finished %s: %s", task, aggregates)
        if dist.is_initialized():
            dist.barrier()
        gc_cuda()
    return task_results


def _git_revision() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _validate_git_identity(identity: Mapping[str, Any]) -> None:
    """Require a clean, immutable Git revision for a certification receipt."""
    revision = identity.get("revision")
    if not isinstance(revision, str) or _GIT_REVISION_RE.fullmatch(revision) is None:
        raise ValueError("Certification requires a lowercase 40-hex Git revision")
    if identity.get("dirty") is not False:
        raise ValueError("Certification refuses a dirty Git checkout")


def _checkpoint_identity(checkpoint: Path, config_path: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    state_dir = Path(_checkpoint_state_dir(checkpoint)).resolve()
    if not state_dir.is_dir():
        raise ValueError(f"Checkpoint state directory is missing: {state_dir}")
    metadata_path = state_dir / ".metadata"
    if not metadata_path.is_file():
        raise ValueError(f"Checkpoint DCP metadata is missing: {metadata_path}")
    state_files: list[dict[str, Any]] = []
    for path in sorted(state_dir.iterdir(), key=lambda value: value.name):
        if not path.is_file():
            raise ValueError(f"Checkpoint state directory contains a non-file entry: {path}")
        state_files.append({"name": path.name, "bytes": path.stat().st_size})
    marker_path = checkpoint / ".metadata.json"
    if not marker_path.is_file():
        raise ValueError(f"Checkpoint marker is missing: {marker_path}")
    return {
        "checkpoint": str(checkpoint),
        "config": _file_identity(config_path),
        "checkpoint_marker": _file_identity(marker_path),
        "state_dir": str(state_dir),
        "dcp_metadata": _file_identity(metadata_path),
        "state_file_inventory": state_files,
        "state_file_inventory_sha256": _canonical_sha256(state_files),
        "state_file_count": len(state_files),
        "state_bytes": sum(int(row["bytes"]) for row in state_files),
    }


def _joint_step(checkpoint: Path) -> int:
    checkpoint = checkpoint.expanduser().resolve()
    match = re.fullmatch(r"step([0-9]+)", checkpoint.name)
    step = int(match.group(1)) if match is not None else None
    if step not in EXPECTED_MATCHED_WRONG_RECEIPTS:
        raise ValueError("External academic certification admits only joint steps 12000 and 16000")
    expected = (EXPECTED_JOINT_CHECKPOINT_PARENT / f"step{step}").resolve()
    if checkpoint != expected:
        raise ValueError("Checkpoint is outside the exact permanent joint-v1 endpoint lineage")
    assert step is not None
    return step


def _load_saved_endpoint_validator() -> Any:
    name = "_vision_alignment_joint_saved_endpoint_validator_for_external_academic"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = (
        Path(__file__)
        .resolve()
        .with_name("vision_alignment_joint_matched_wrong_saved_steps_validate.py")
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load saved-endpoint receipt validator {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _validate_joint_checkpoint_and_prior_receipt(
    checkpoint: Path,
    config_path: Path,
    *,
    verify_live_checkpoint: bool,
) -> dict[str, Any]:
    """Bind certification to one exact joint endpoint and its prior V2 visual receipt."""
    checkpoint = checkpoint.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    step = _joint_step(checkpoint)
    if config_path != checkpoint / "config.json":
        raise ValueError("Certification requires the checkpoint-local config.json")
    config_identity = _file_identity(config_path)
    if config_identity["sha256"] != EXPECTED_JOINT_CONFIG_SHA256:
        raise ValueError("Checkpoint config is not the reviewed joint-v1 config identity")
    expected = EXPECTED_MATCHED_WRONG_RECEIPTS[step]
    receipt_path = Path(expected["path"]).resolve()
    receipt_identity = _file_identity(receipt_path)
    if receipt_identity["sha256"] != expected["sha256"]:
        raise ValueError(f"step{step} prior matched/wrong V2 receipt raw SHA-256 differs")
    validator = _load_saved_endpoint_validator()
    validator.validate_evaluator_receipt(
        receipt_path,
        expected_sha256=expected["sha256"],
        step=step,
        verify_live_checkpoint=verify_live_checkpoint,
    )
    receipt = _load_json_strict(receipt_path)
    return {
        "step": step,
        "path": receipt_identity["path"],
        "bytes": receipt_identity["bytes"],
        "sha256": receipt_identity["sha256"],
        "content_sha256": receipt.get("content_sha256"),
        "format": receipt.get("format"),
        "version": receipt.get("version"),
        "protocol_name": receipt.get("protocol", {}).get("name"),
    }


def _set_model_parts_eval(train_module: Any) -> None:
    for model_part in train_module.model_parts:
        model_part.eval()


def _validate_runtime_args(args: argparse.Namespace) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if world_size != EP_DEGREE or local_world_size != EP_DEGREE:
        raise ValueError(
            f"The certification protocol requires WORLD_SIZE=LOCAL_WORLD_SIZE={EP_DEGREE}; "
            f"got WORLD_SIZE={world_size}, LOCAL_WORLD_SIZE={local_world_size}"
        )
    if args.max_sequence_length != DEFAULT_MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"The certification protocol requires --max-sequence-length="
            f"{DEFAULT_MAX_SEQUENCE_LENGTH}"
        )
    if args.max_crops != DEFAULT_MAX_CROPS:
        raise ValueError(f"The certification protocol requires --max-crops={DEFAULT_MAX_CROPS}")
    if args.max_new_tokens != DEFAULT_MAX_NEW_TOKENS:
        raise ValueError(
            f"The certification protocol requires --max-new-tokens={DEFAULT_MAX_NEW_TOKENS}"
        )
    if args.sequence_bucket_size != DEFAULT_SEQUENCE_BUCKET_SIZE:
        raise ValueError(
            f"The certification protocol requires --sequence-bucket-size="
            f"{DEFAULT_SEQUENCE_BUCKET_SIZE}"
        )
    if args.checkpoint_load_threads <= 0:
        raise ValueError("--checkpoint-load-threads must be positive")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite canonical artifact {output}")


def _protocol_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen scientific and runtime protocol recorded in every receipt."""
    task_counts = {
        task: {
            "selected": int(manifest["tasks"][task]["selection_count"]),
            "available": int(manifest["tasks"][task]["source"]["examples"]),
        }
        for task in DEFAULT_TASKS
    }
    free_answer_examples = sum(
        task_counts[task]["selected"] for task in ("vqav2", "textvqa", "docvqa", "chartqa")
    )
    multiple_choice_examples = sum(task_counts[task]["selected"] for task in ("ai2d", "a_okvqa_mc"))
    return {
        "claim": "confirmatory sampled validation evidence for joint step selection",
        "benchmark_scope": (
            "locally cached validation projections; descriptive checkpoint comparison, not an "
            "official leaderboard submission"
        ),
        "split": "validation",
        "tasks": list(DEFAULT_TASKS),
        "controls": list(CONTROLS),
        "control_pairing": (
            "shuffled images preserve each recipient's exact Molmo2 pooled grid, image-token "
            "layout, and declared task stratum; AI2D donors also differ in source base diagram; "
            "blank images preserve recipient dimensions"
        ),
        "contamination_reporting": (
            "exact-byte-only; correct-minus-shuffled nonoverlap deltas require both recipient "
            "and donor absent from train inventory, while blank deltas require recipient absent"
        ),
        "message_format": "document",
        "prompt": {
            "free_answer": "Question: {question}\\nAnswer:",
            "multiple_choice": (
                "Question: {question}\\nOptions:\\n{letter}. {option}\\n"
                "Answer with only the option letter.\\nAnswer:"
            ),
            "empty_option_display": EMPTY_OPTION_DISPLAY,
            "response_separator": "single leading space",
        },
        "generation": "greedy full-sequence recompute without KV cache",
        "answer_token_coverage": EXPECTED_ANSWER_TOKEN_COVERAGE,
        "multiple_choice": (
            "candidate-normalized next-token option-letter likelihood; deterministic argmax "
            "with earliest option winning exact ties"
        ),
        "metrics": {
            "vqav2": "VQA leave-one-out consensus accuracy with vendored EvalAI normalization",
            "textvqa": (
                "TextVQA word-tokenize normalization followed by VQA leave-one-out consensus"
            ),
            "docvqa": "ANLS, case-insensitive, similarity strictly greater than 0.5",
            "chartqa": "relaxed accuracy, 5% relative numeric tolerance",
            "ai2d": (
                "outer option-label index accuracy on local validation and transparent-box "
                "augmentation strata"
            ),
            "a_okvqa_mc": "predicted choice-text equals correct choice-text",
        },
        "sample_coverage": {
            "examples_per_task": DEFAULT_EXAMPLES_PER_TASK,
            "selection_seed": DEFAULT_SELECTION_SEED,
            "ranking": "sha256(seed\\0task\\0example_id), then example_id",
            "panel_status": manifest["selection"]["panel_status"],
            "complete_validation_splits": False,
            "ai2d_strata": ["standard", "transparent"],
            "tasks": task_counts,
        },
        "compute_budget": {
            "free_answer_examples": free_answer_examples,
            "multiple_choice_examples": multiple_choice_examples,
            "image_controls_per_example": len(CONTROLS),
            "maximum_free_answer_forwards": (
                free_answer_examples * len(CONTROLS) * DEFAULT_MAX_NEW_TOKENS
            ),
            "multiple_choice_forwards": multiple_choice_examples * len(CONTROLS),
            "maximum_total_model_forwards": (
                free_answer_examples * len(CONTROLS) * DEFAULT_MAX_NEW_TOKENS
                + multiple_choice_examples * len(CONTROLS)
            ),
            "maximum_image_encoding_calls": (
                (free_answer_examples + multiple_choice_examples) * len(CONTROLS)
            ),
        },
        "world_size": EP_DEGREE,
        "ep_degree": EP_DEGREE,
        "expert_parallel_path": ExpertParallelPath.sync_1d.value,
        "logical_eval_replicas": 1,
        "max_sequence_length": DEFAULT_MAX_SEQUENCE_LENGTH,
        "max_high_resolution_crops": DEFAULT_MAX_CROPS,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "sequence_bucket_size": DEFAULT_SEQUENCE_BUCKET_SIZE,
        "attention_backend": "flex",
    }


def _validate_prediction_output(
    task: str,
    example: AcademicExample,
    value: Any,
    *,
    name: str,
    tokenizer: Any,
    text_vocab_size: int,
) -> dict[str, Any]:
    fields = {
        "prediction",
        "predicted_index",
        "input_tokens",
        "output_tokens",
        "score",
        "image_grid_signature",
        "image_token_count",
        "image_token_ids_sha256",
    }
    is_multiple_choice = task in ("ai2d", "a_okvqa_mc")
    if is_multiple_choice:
        fields.add("candidate_log_probabilities")
    else:
        fields.update(("generated_token_ids", "stop_reason"))
    output = _exact_mapping(value, tuple(fields), name=name)
    prediction = output["prediction"]
    input_tokens = output["input_tokens"]
    output_tokens = output["output_tokens"]
    score = output["score"]
    grid_signature = output["image_grid_signature"]
    if not isinstance(prediction, str):
        raise TypeError(f"{name} prediction must be a string")
    if (
        type(input_tokens) is not int
        or not 0 < input_tokens <= DEFAULT_MAX_SEQUENCE_LENGTH
        or type(output_tokens) is not int
        or not 0 < output_tokens <= DEFAULT_MAX_NEW_TOKENS
        or not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not np.isfinite(float(score))
    ):
        raise ValueError(f"{name} token counts or score are invalid")
    if (
        not isinstance(grid_signature, list)
        or len(grid_signature) != 4
        or any(type(item) is not int or item <= 0 for item in grid_signature)
        or type(output["image_token_count"]) is not int
        or output["image_token_count"] <= 0
        or not isinstance(output["image_token_ids_sha256"], str)
        or _SHA256_RE.fullmatch(output["image_token_ids_sha256"]) is None
    ):
        raise ValueError(f"{name} image-token layout identity is invalid")
    if is_multiple_choice:
        predicted_index = output["predicted_index"]
        if (
            type(predicted_index) is not int
            or not 0 <= predicted_index < len(example.options)
            or output_tokens != 1
            or prediction != string.ascii_uppercase[predicted_index]
        ):
            raise ValueError(f"{name} multiple-choice prediction is inconsistent")
        probabilities = output["candidate_log_probabilities"]
        expected_letters = list(string.ascii_uppercase[: len(example.options)])
        if not isinstance(probabilities, dict) or list(probabilities) != expected_letters:
            raise ValueError(f"{name} candidate log-probability keys differ")
        log_probabilities = list(probabilities.values())
        if any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not np.isfinite(float(item))
            for item in log_probabilities
        ):
            raise ValueError(f"{name} candidate log probabilities are invalid")
        probability_sum = float(np.exp(np.asarray(log_probabilities, dtype=np.float64)).sum())
        if not np.isclose(probability_sum, 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError(f"{name} candidate log probabilities are not normalized")
        if predicted_index != int(np.argmax(np.asarray(log_probabilities, dtype=np.float64))):
            raise ValueError(f"{name} prediction is not the deterministic probability argmax")
    else:
        if output["predicted_index"] is not None:
            raise ValueError(f"{name} free-answer predicted_index must be null")
        generated = output["generated_token_ids"]
        if (
            not isinstance(generated, list)
            or len(generated) != output_tokens
            or any(
                type(token_id) is not int or not 0 <= token_id < text_vocab_size
                for token_id in generated
            )
        ):
            raise ValueError(f"{name} generated-token identity differs")
        if output["stop_reason"] not in ("eos", "max_tokens") or (
            output["stop_reason"] == "max_tokens" and output_tokens != DEFAULT_MAX_NEW_TOKENS
        ):
            raise ValueError(f"{name} generation stop reason differs")
        if (
            output["stop_reason"] == "eos"
            and (generated[-1] != EXPECTED_EOS_TOKEN_ID or EXPECTED_EOS_TOKEN_ID in generated[:-1])
        ) or (output["stop_reason"] == "max_tokens" and EXPECTED_EOS_TOKEN_ID in generated):
            raise ValueError(f"{name} generation stop token differs")
        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if decoded != prediction:
            raise ValueError(f"{name} decoded generated tokens differ from prediction")
    expected_score = _score_prediction(
        example,
        prediction=prediction,
        predicted_index=output["predicted_index"],
    )
    if float(score) != expected_score:
        raise ValueError(f"{name} score was not rederived from prediction and gold")
    return output


def _receipt_example_from_row(task: str, row: Mapping[str, Any]) -> AcademicExample:
    answers = row.get("gold_answers")
    options = row.get("options")
    if (
        not isinstance(answers, list)
        or any(not isinstance(answer, str) for answer in answers)
        or not isinstance(options, list)
        or any(not isinstance(option, str) for option in options)
    ):
        raise ValueError(f"Receipt task {task!r} has invalid gold answers or options")
    answer_index = row.get("gold_answer_index")
    if answer_index is not None and type(answer_index) is not int:
        raise ValueError(f"Receipt task {task!r} has an invalid gold answer index")
    stratum = row.get("stratum")
    if stratum is not None and not isinstance(stratum, str):
        raise ValueError(f"Receipt task {task!r} has an invalid stratum")
    return AcademicExample(
        task=task,
        example_id=_validate_text(row.get("example_id"), name=f"{task} receipt example ID"),
        source_position=_validate_text(
            row.get("source_position"), name=f"{task} receipt source position"
        ),
        visual=None,
        image_reference=None,
        question=_validate_text(row.get("question"), name=f"{task} receipt question"),
        answers=tuple(answers),
        options=tuple(options),
        answer_index=answer_index,
        stratum=stratum,
    )


def _validate_receipt_tasks(
    tasks: Any,
    *,
    manifest: Mapping[str, Any],
    loaded: Mapping[str, Mapping[str, AcademicExample]] | None,
    tokenizer: Any,
    text_vocab_size: int,
) -> dict[str, Any]:
    if text_vocab_size <= EXPECTED_EOS_TOKEN_ID:
        raise ValueError("External academic receipt text-vocabulary boundary is invalid")
    if not isinstance(tasks, dict) or set(tasks) != set(DEFAULT_TASKS):
        raise ValueError("External academic receipt tasks differ from the frozen panel")
    manifest_tasks = manifest["tasks"]
    for task in DEFAULT_TASKS:
        result = _exact_mapping(
            tasks[task],
            (
                "source",
                "selection_count",
                "selection_sha256",
                "alignment_train_image_overlap_count",
                "generation_stop_counts",
                "elapsed_seconds",
                "metric",
                "controls",
                "image_control_deltas",
                "examples",
            ),
            name=f"external academic receipt task {task}",
        )
        manifest_task = manifest_tasks[task]
        records = manifest_task["records"]
        if (
            result["source"] != manifest_task["source"]
            or result["selection_count"] != len(records)
            or result["selection_sha256"] != manifest_task["selection_sha256"]
            or not isinstance(result["elapsed_seconds"], (int, float))
            or isinstance(result["elapsed_seconds"], bool)
            or not np.isfinite(float(result["elapsed_seconds"]))
            or float(result["elapsed_seconds"]) < 0.0
        ):
            raise ValueError(f"External academic receipt task {task!r} envelope differs")
        rows = result["examples"]
        if not isinstance(rows, list) or len(rows) != len(records):
            raise ValueError(f"External academic receipt task {task!r} row count differs")
        for index, (row_value, record) in enumerate(zip(rows, records)):
            row = _exact_mapping(
                row_value,
                (
                    "example_id",
                    "source_position",
                    "annotation_sha256",
                    "image_sha256",
                    "image_grid_signature",
                    "image_token_count",
                    "alignment_train_image_overlap",
                    "shuffled_donor_id",
                    "shuffled_image_sha256",
                    "shuffled_image_grid_signature",
                    "shuffled_alignment_train_image_overlap",
                    "question",
                    "gold_answers",
                    "options",
                    "gold_answer_index",
                    "stratum",
                    "controls",
                ),
                name=f"external academic receipt {task} row {index}",
            )
            for field in (
                "example_id",
                "source_position",
                "annotation_sha256",
                "image_sha256",
                "image_grid_signature",
                "image_token_count",
                "alignment_train_image_overlap",
                "shuffled_donor_id",
                "shuffled_image_sha256",
                "shuffled_image_grid_signature",
                "shuffled_alignment_train_image_overlap",
            ):
                if row[field] != record[field]:
                    raise ValueError(f"External academic receipt {task}/{index} {field} differs")
            example = _receipt_example_from_row(task, row)
            if example.annotation() != {
                "task": task,
                "example_id": row["example_id"],
                "source_position": row["source_position"],
                "question": row["question"],
                "answers": row["gold_answers"],
                "options": row["options"],
                "answer_index": row["gold_answer_index"],
                "stratum": row["stratum"],
            }:
                raise AssertionError("Receipt annotation projection construction drifted")
            if _canonical_sha256(example.annotation()) != record["annotation_sha256"]:
                raise ValueError(f"External academic receipt {task}/{index} annotation differs")
            if (
                loaded is not None
                and loaded[task][example.example_id].annotation() != example.annotation()
            ):
                raise ValueError(
                    f"External academic receipt {task}/{index} live annotation differs"
                )
            controls = row["controls"]
            if not isinstance(controls, dict) or set(controls) != set(CONTROLS):
                raise ValueError(f"External academic receipt {task}/{index} controls differ")
            for control in CONTROLS:
                output = _validate_prediction_output(
                    task,
                    example,
                    controls[control],
                    name=f"external academic receipt {task}/{index}/{control}",
                    tokenizer=tokenizer,
                    text_vocab_size=text_vocab_size,
                )
                if (
                    output["image_grid_signature"] != record["image_grid_signature"]
                    or output["image_token_count"] != record["image_token_count"]
                ):
                    raise ValueError(
                        f"External academic receipt {task}/{index}/{control} grid differs"
                    )
            if len({controls[control]["image_token_ids_sha256"] for control in CONTROLS}) != 1:
                raise ValueError(f"External academic receipt {task}/{index} image-token IDs differ")
        overlap_count = sum(bool(row["alignment_train_image_overlap"]) for row in rows)
        if result["alignment_train_image_overlap_count"] != overlap_count:
            raise ValueError(f"External academic receipt task {task!r} overlap count differs")
        if result["generation_stop_counts"] != _generation_stop_counts(task, rows):
            raise ValueError(f"External academic receipt task {task!r} stop counts differ")
        aggregates = _aggregate_task_outputs(task, rows)
        for field in ("metric", "controls", "image_control_deltas"):
            if result[field] != aggregates[field]:
                raise ValueError(f"External academic receipt task {task!r} {field} differs")
    return tasks


def _load_receipt_manifest(
    reference: Any,
    *,
    verify_live_sources: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, AcademicExample]] | None]:
    manifest_reference = _exact_mapping(
        reference,
        ("path", "bytes", "sha256", "content_sha256", "partial", "panel_status"),
        name="external academic receipt manifest reference",
    )
    path_value = manifest_reference["path"]
    if not isinstance(path_value, str):
        raise TypeError("External academic receipt manifest path is invalid")
    path = Path(path_value).expanduser().resolve()
    identity = _file_identity(path)
    if any(identity[field] != manifest_reference[field] for field in ("path", "bytes", "sha256")):
        raise ValueError("External academic receipt manifest raw identity differs")
    if verify_live_sources:
        manifest, loaded, validated_identity = _validate_manifest_and_load_examples(path)
        if validated_identity != identity:
            raise ValueError("External academic receipt manifest changed during validation")
    else:
        manifest = _load_json_strict(path)
        _verify_content_sha256(manifest, name="external academic receipt manifest")
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("format") != MANIFEST_FORMAT
            or manifest.get("protocol_name") != PROTOCOL_NAME
            or manifest.get("implementation") != _implementation_identity()
        ):
            raise ValueError("External academic receipt manifest envelope differs")
        loaded = None
    if (
        manifest_reference["content_sha256"] != manifest.get("content_sha256")
        or manifest_reference["partial"] != manifest.get("selection", {}).get("partial")
        or manifest_reference["panel_status"] != manifest.get("selection", {}).get("panel_status")
    ):
        raise ValueError("External academic receipt manifest semantic identity differs")
    return manifest, loaded


def _load_exact_receipt_tokenizer(
    config_identity: Mapping[str, Any], tokenizer_provenance: Mapping[str, Any]
) -> tuple[Any, Molmo2TokenIds]:
    """Reload the pinned tokenizer used to rederive serialized generation semantics."""
    config_path = Path(str(config_identity["path"]))
    if _file_identity(config_path) != config_identity:
        raise ValueError("External academic receipt live config identity differs")
    raw_config = _load_json_strict(config_path)
    artifacts = raw_config.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("External academic receipt config lacks tokenizer artifacts")
    expected_provenance = {
        "identifier": artifacts.get("tokenizer_id"),
        "revision": artifacts.get("tokenizer_revision"),
        "fingerprint": artifacts.get("tokenizer_fingerprint"),
    }
    if any(not isinstance(value, str) or not value for value in expected_provenance.values()):
        raise ValueError("External academic receipt config tokenizer provenance is incomplete")
    if any(
        tokenizer_provenance.get(field) != expected_provenance[field]
        for field in expected_provenance
    ):
        raise ValueError("External academic receipt tokenizer differs from checkpoint config")
    cache_dir = artifacts.get("hf_cache_dir")
    model_config = raw_config.get("model")
    if (
        not isinstance(cache_dir, str)
        or not cache_dir
        or not isinstance(model_config, dict)
        or not isinstance(model_config.get("lm"), dict)
        or type(model_config["lm"].get("vocab_size")) is not int
    ):
        raise ValueError("External academic receipt tokenizer runtime config is incomplete")
    tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
        identifier=expected_provenance["identifier"],
        revision=expected_provenance["revision"],
        expected_fingerprint=expected_provenance["fingerprint"],
        cache_dir=cache_dir,
        model_vocab_size=model_config["lm"]["vocab_size"],
    )
    if (
        tokenizer.eos_token_id != EXPECTED_EOS_TOKEN_ID
        or tokenizer.pad_token_id != EXPECTED_PAD_TOKEN_ID
        or token_ids.as_config_dict() != tokenizer_provenance.get("token_ids")
        or _canonical_sha256(token_ids.as_config_dict())
        != tokenizer_provenance.get("token_ids_sha256")
    ):
        raise ValueError("External academic receipt live tokenizer identities differ")
    return tokenizer, token_ids


def _validate_external_academic_receipt_payload(
    receipt: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    loaded: Mapping[str, Mapping[str, AcademicExample]] | None,
) -> dict[str, Any]:
    value = _exact_mapping(
        receipt,
        (
            "schema_version",
            "format",
            "protocol_name",
            "created_at",
            "git",
            "implementation",
            "manifest",
            "checkpoint",
            "prior_matched_wrong_v2",
            "artifact_policy",
            "tokenizer",
            "protocol",
            "tasks",
            "content_sha256",
        ),
        name="external academic receipt",
    )
    _verify_content_sha256(value, name="external academic receipt")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["format"] != RECEIPT_FORMAT
        or value["protocol_name"] != PROTOCOL_NAME
    ):
        raise ValueError("External academic receipt protocol identity differs")
    _validate_timestamp(value["created_at"], name="external academic receipt created_at")
    git = _exact_mapping(value["git"], ("revision", "dirty"), name="external academic receipt git")
    _validate_git_identity(git)
    if git != manifest.get("git"):
        raise ValueError("External academic receipt Git identity differs from its manifest")
    if value["implementation"] != _implementation_identity():
        raise ValueError("External academic receipt implementation differs")
    manifest_reference = _exact_mapping(
        value["manifest"],
        ("path", "bytes", "sha256", "content_sha256", "partial", "panel_status"),
        name="external academic receipt manifest reference",
    )
    if (
        manifest_reference["content_sha256"] != manifest.get("content_sha256")
        or manifest_reference["partial"] != manifest.get("selection", {}).get("partial")
        or manifest_reference["panel_status"] != manifest.get("selection", {}).get("panel_status")
    ):
        raise ValueError("External academic receipt manifest reference differs")
    manifest_path = manifest_reference["path"]
    if not isinstance(manifest_path, str):
        raise TypeError("External academic receipt manifest path is invalid")
    live_manifest_identity = _file_identity(Path(manifest_path))
    if any(
        manifest_reference[field] != live_manifest_identity[field]
        for field in ("path", "bytes", "sha256")
    ):
        raise ValueError("External academic receipt manifest raw identity differs")
    if value["protocol"] != _protocol_payload(manifest):
        raise ValueError("External academic receipt frozen protocol differs")
    checkpoint = _exact_mapping(
        value["checkpoint"],
        (
            "checkpoint",
            "config",
            "checkpoint_marker",
            "state_dir",
            "dcp_metadata",
            "state_file_inventory",
            "state_file_inventory_sha256",
            "state_file_count",
            "state_bytes",
        ),
        name="external academic receipt checkpoint",
    )
    config = _exact_mapping(
        checkpoint["config"], ("path", "bytes", "sha256"), name="checkpoint config identity"
    )
    if config["sha256"] != EXPECTED_JOINT_CONFIG_SHA256:
        raise ValueError("External academic receipt config identity differs")
    checkpoint_path = checkpoint["checkpoint"]
    if not isinstance(checkpoint_path, str):
        raise TypeError("External academic receipt checkpoint path is invalid")
    step = _joint_step(Path(checkpoint_path))
    if config["path"] != str(Path(checkpoint_path).resolve() / "config.json"):
        raise ValueError("External academic receipt checkpoint config path differs")
    marker = _exact_mapping(
        checkpoint["checkpoint_marker"],
        ("path", "bytes", "sha256"),
        name="external academic receipt checkpoint marker",
    )
    metadata = _exact_mapping(
        checkpoint["dcp_metadata"],
        ("path", "bytes", "sha256"),
        name="external academic receipt DCP metadata",
    )
    expected_state_dir = str(Path(checkpoint_path).resolve() / "model_and_optim")
    if (
        marker["path"] != str(Path(checkpoint_path).resolve() / ".metadata.json")
        or marker["sha256"] != EXPECTED_CHECKPOINT_MARKER_SHA256
        or checkpoint["state_dir"] != expected_state_dir
        or metadata["path"] != str(Path(expected_state_dir) / ".metadata")
        or metadata["sha256"] != EXPECTED_DCP_METADATA_SHA256[step]
    ):
        raise ValueError("External academic receipt checkpoint endpoint identity differs")
    state_inventory = checkpoint["state_file_inventory"]
    if (
        not isinstance(state_inventory, list)
        or checkpoint["state_file_count"] != len(state_inventory)
        or checkpoint["state_file_inventory_sha256"] != _canonical_sha256(state_inventory)
        or any(
            not isinstance(row, dict)
            or set(row) != {"name", "bytes"}
            or not isinstance(row["name"], str)
            or not row["name"]
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
            for row in state_inventory
        )
        or len({row["name"] for row in state_inventory}) != len(state_inventory)
        or checkpoint["state_bytes"] != sum(row["bytes"] for row in state_inventory)
    ):
        raise ValueError("External academic receipt checkpoint file inventory differs")
    prior = _exact_mapping(
        value["prior_matched_wrong_v2"],
        ("step", "path", "bytes", "sha256", "content_sha256", "format", "version", "protocol_name"),
        name="external academic receipt prior matched/wrong reference",
    )
    expected_prior = EXPECTED_MATCHED_WRONG_RECEIPTS[step]
    if (
        prior["step"] != step
        or prior["path"] != str(Path(expected_prior["path"]).resolve())
        or prior["sha256"] != expected_prior["sha256"]
    ):
        raise ValueError("External academic receipt prior matched/wrong identity differs")
    prior_identity = _file_identity(Path(prior["path"]))
    prior_payload = _load_json_strict(Path(prior["path"]))
    _verify_content_sha256(prior_payload, name="prior matched/wrong V2 receipt")
    if (
        any(prior[field] != prior_identity[field] for field in ("path", "bytes", "sha256"))
        or prior["content_sha256"] != prior_payload.get("content_sha256")
        or prior["format"] != "vision_alignment_joint_matched_wrong_receipt"
        or prior["version"] != 2
        or prior["protocol_name"]
        != "vision-alignment-joint-native-matched-wrong-saved-endpoints-v2"
    ):
        raise ValueError("External academic receipt prior matched/wrong envelope differs")
    artifact_policy = _exact_mapping(
        value["artifact_policy"],
        ("descriptive_only", "promotion_eligible", "checkpoint_selection_evidence"),
        name="external academic receipt artifact policy",
    )
    if artifact_policy != {
        "descriptive_only": True,
        "promotion_eligible": False,
        "checkpoint_selection_evidence": True,
    }:
        raise ValueError("External academic receipt artifact policy differs")
    tokenizer_provenance = _exact_mapping(
        value["tokenizer"],
        (
            "identifier",
            "revision",
            "fingerprint",
            "eos_token_id",
            "pad_token_id",
            "token_ids",
            "token_ids_sha256",
        ),
        name="external academic receipt tokenizer",
    )
    if any(
        not isinstance(tokenizer_provenance[field], str) or not tokenizer_provenance[field]
        for field in ("identifier", "revision", "fingerprint")
    ) or not isinstance(tokenizer_provenance["token_ids"], dict):
        raise ValueError("External academic receipt tokenizer provenance is invalid")
    prior_tokenizer = prior_payload.get("tokenizer")
    if not isinstance(prior_tokenizer, dict) or {
        field: tokenizer_provenance[field]
        for field in ("identifier", "revision", "fingerprint", "token_ids")
    } != {
        "identifier": prior_tokenizer.get("id"),
        "revision": prior_tokenizer.get("revision"),
        "fingerprint": prior_tokenizer.get("fingerprint"),
        "token_ids": prior_tokenizer.get("token_ids"),
    }:
        raise ValueError("External academic receipt tokenizer differs from prior V2 evidence")
    if (
        tokenizer_provenance["eos_token_id"] != EXPECTED_EOS_TOKEN_ID
        or tokenizer_provenance["pad_token_id"] != EXPECTED_PAD_TOKEN_ID
        or tokenizer_provenance["token_ids_sha256"] != EXPECTED_MOLMO2_TOKEN_IDS_SHA256
        or tokenizer_provenance["token_ids_sha256"] != prior_tokenizer.get("token_ids_sha256")
        or tokenizer_provenance["token_ids_sha256"]
        != _canonical_sha256(tokenizer_provenance["token_ids"])
    ):
        raise ValueError("External academic receipt tokenizer token identities differ")
    runtime_tokenizer, runtime_token_ids = _load_exact_receipt_tokenizer(
        config,
        tokenizer_provenance,
    )
    _validate_receipt_tasks(
        value["tasks"],
        manifest=manifest,
        loaded=loaded,
        tokenizer=runtime_tokenizer,
        text_vocab_size=min(runtime_token_ids.image_token_ids),
    )
    return value


def validate_external_academic_receipt(
    path: str | Path,
    expected_sha256: str,
    *,
    verify_live: bool = True,
) -> dict[str, Any]:
    """Strictly reload and rederive an external-academic receipt.

    :param path: Canonical receipt path.
    :param expected_sha256: Independently supplied raw receipt SHA-256.
    :param verify_live: Rehash academic sources, selected images, checkpoint, and prior V2 receipt.
    :returns: The validated receipt.
    """
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError("Expected receipt SHA-256 must be lowercase hex")
    receipt_path = Path(path).expanduser().resolve()
    identity = _file_identity(receipt_path)
    if identity["sha256"] != expected_sha256:
        raise ValueError("External academic receipt raw SHA-256 differs")
    receipt = _load_json_strict(receipt_path)
    manifest, loaded = _load_receipt_manifest(
        receipt.get("manifest"), verify_live_sources=verify_live
    )
    _validate_external_academic_receipt_payload(receipt, manifest=manifest, loaded=loaded)
    checkpoint_path = Path(receipt["checkpoint"]["checkpoint"])
    config_path = Path(receipt["checkpoint"]["config"]["path"])
    if verify_live:
        if _checkpoint_identity(checkpoint_path, config_path) != receipt["checkpoint"]:
            raise ValueError("External academic receipt live checkpoint identity differs")
        prior = _validate_joint_checkpoint_and_prior_receipt(
            checkpoint_path,
            config_path,
            verify_live_checkpoint=True,
        )
        if prior != receipt["prior_matched_wrong_v2"]:
            raise ValueError("External academic receipt live prior V2 identity differs")
    return receipt


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    _validate_runtime_args(args)
    manifest, loaded, manifest_file_identity = _validate_manifest_and_load_examples(
        Path(args.manifest)
    )
    selection = manifest["selection"]
    if (
        selection.get("panel_status") != "confirmatory"
        or selection.get("tasks") != list(DEFAULT_TASKS)
        or selection.get("seed") != DEFAULT_SELECTION_SEED
        or selection.get("examples_per_task_limit") != DEFAULT_EXAMPLES_PER_TASK
        or selection.get("partial") is not True
    ):
        raise ValueError("Certification evaluate requires the frozen 512-per-task panel")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config_path = Path(_config_path(checkpoint, args.config)).resolve()
    prior_matched_wrong = _validate_joint_checkpoint_and_prior_receipt(
        checkpoint,
        config_path,
        verify_live_checkpoint=False,
    )
    raw_config = _load_json_strict(config_path)
    if raw_config.get("phase") != "joint":
        raise ValueError("External academic certification requires an exact joint checkpoint")
    data_config = raw_config.get("data")
    if not isinstance(data_config, dict) or data_config.get("message_format") != "document":
        raise ValueError("External academic certification requires message_format='document'")
    checkpoint_identity = _checkpoint_identity(checkpoint, config_path)
    git_identity = _git_revision()
    _validate_git_identity(git_identity)
    if git_identity != manifest["git"]:
        raise ValueError("Evaluation Git identity differs from the manifest builder")
    implementation_identity = _implementation_identity()

    if args.hf_cache:
        os.environ["HF_HOME"] = str(Path(args.hf_cache).expanduser().resolve())
    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepare_training_environment()
    try:
        model, module_config, checkpoint_kind = _build_model_and_module_config(
            raw_config,
            ep_degree=EP_DEGREE,
            max_sequence_length=args.max_sequence_length,
            rank_batch_size=args.max_sequence_length,
            ep_path=ExpertParallelPath.sync_1d,
        )
        if checkpoint_kind != "multimodal_stage1":
            raise ValueError(
                "External academic certification requires a native multimodal checkpoint; "
                f"detected {checkpoint_kind!r}"
            )
        train_module = module_config.build(model, eval_only=True)
        state_dir = Path(_checkpoint_state_dir(checkpoint)).resolve()
        train_module.load_state_dict_direct(
            state_dir,
            process_group=dist.group.WORLD,
            thread_count=args.checkpoint_load_threads,
            load_optim_state=False,
        )
        _set_model_parts_eval(train_module)

        artifacts = raw_config.get("artifacts")
        if not isinstance(artifacts, dict):
            raise TypeError("Vision-alignment checkpoint lacks pinned tokenizer artifacts")
        tokenizer_id = artifacts.get("tokenizer_id")
        tokenizer_revision = artifacts.get("tokenizer_revision")
        tokenizer_fingerprint = artifacts.get("tokenizer_fingerprint")
        hf_cache_dir = args.hf_cache or artifacts.get("hf_cache_dir")
        if not all(
            isinstance(value, str) and value
            for value in (
                tokenizer_id,
                tokenizer_revision,
                tokenizer_fingerprint,
                hf_cache_dir,
            )
        ):
            raise ValueError("Vision-alignment checkpoint tokenizer provenance is incomplete")
        model_vocab_size = int(raw_config["model"]["lm"]["vocab_size"])
        tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
            identifier=tokenizer_id,
            revision=tokenizer_revision,
            expected_fingerprint=tokenizer_fingerprint,
            cache_dir=hf_cache_dir,
            model_vocab_size=model_vocab_size,
        )
        if int(raw_config["model"]["image_patch_token_id"]) != token_ids.im_patch_id:
            raise ValueError("Checkpoint image-patch ID differs from the pinned tokenizer")
        if tokenizer.pad_token_id != int(raw_config["collator"]["pad_token_id"]):
            raise ValueError("Checkpoint collator pad ID differs from the pinned tokenizer")
        if (
            tokenizer.eos_token_id != EXPECTED_EOS_TOKEN_ID
            or tokenizer.pad_token_id != EXPECTED_PAD_TOKEN_ID
            or _canonical_sha256(token_ids.as_config_dict()) != EXPECTED_MOLMO2_TOKEN_IDS_SHA256
        ):
            raise ValueError("Pinned tokenizer token identities differ from the frozen protocol")
        answer_token_coverage = _answer_token_coverage(loaded, tokenizer)
        if answer_token_coverage != EXPECTED_ANSWER_TOKEN_COVERAGE:
            raise ValueError(
                "Frozen answer-token coverage differs from the audited 24-token protocol"
            )

        inference = _NativeAcademicInference(
            train_module,
            tokenizer,
            token_ids,
            max_sequence_length=args.max_sequence_length,
            max_crops=args.max_crops,
            max_new_tokens=args.max_new_tokens,
            sequence_bucket_size=args.sequence_bucket_size,
        )
        task_results = _evaluate_manifest(inference, manifest, loaded)

        (
            closing_manifest,
            closing_loaded,
            closing_manifest_identity,
        ) = _validate_manifest_and_load_examples(Path(args.manifest))
        if (
            closing_manifest != manifest
            or closing_manifest_identity != manifest_file_identity
            or {task: tuple(examples) for task, examples in closing_loaded.items()}
            != {task: tuple(examples) for task, examples in loaded.items()}
        ):
            raise ValueError("Manifest, academic sources, or selected image identities changed")
        closing_checkpoint_identity = _checkpoint_identity(checkpoint, config_path)
        if closing_checkpoint_identity != checkpoint_identity:
            raise ValueError("Checkpoint identity changed during external academic evaluation")
        closing_prior = _validate_joint_checkpoint_and_prior_receipt(
            checkpoint,
            config_path,
            verify_live_checkpoint=True,
        )
        if closing_prior != prior_matched_wrong:
            raise ValueError("Prior matched/wrong receipt identity changed during evaluation")
        if _implementation_identity() != implementation_identity:
            raise ValueError("Evaluator implementation changed during evaluation")
        closing_git_identity = _git_revision()
        _validate_git_identity(closing_git_identity)
        if closing_git_identity != git_identity:
            raise ValueError("Git identity changed during external academic evaluation")

        payload = {
            "schema_version": SCHEMA_VERSION,
            "format": RECEIPT_FORMAT,
            "protocol_name": PROTOCOL_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git": git_identity,
            "implementation": implementation_identity,
            "manifest": {
                **manifest_file_identity,
                "content_sha256": manifest["content_sha256"],
                "partial": manifest["selection"]["partial"],
                "panel_status": manifest["selection"]["panel_status"],
            },
            "checkpoint": checkpoint_identity,
            "prior_matched_wrong_v2": prior_matched_wrong,
            "artifact_policy": {
                "descriptive_only": True,
                "promotion_eligible": False,
                "checkpoint_selection_evidence": True,
            },
            "tokenizer": {
                "identifier": tokenizer_id,
                "revision": tokenizer_revision,
                "fingerprint": tokenizer_fingerprint,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "token_ids": token_ids.as_config_dict(),
                "token_ids_sha256": _canonical_sha256(token_ids.as_config_dict()),
            },
            "protocol": _protocol_payload(manifest),
            "tasks": task_results,
        }
        receipt = _attach_content_sha256(payload)
        _validate_external_academic_receipt_payload(
            receipt,
            manifest=manifest,
            loaded=loaded,
        )
        publication: list[Any] = [None]
        if get_rank() == 0:
            try:
                output_path = Path(args.output)
                _write_json_no_overwrite(output_path, receipt)
                raw_sha256 = _sha256_file_stable(_artifact_path(output_path, name="receipt"))
                validate_external_academic_receipt(
                    output_path,
                    raw_sha256,
                    verify_live=False,
                )
                publication[0] = {"ok": True, "sha256": raw_sha256}
            except Exception as error:  # noqa: BLE001 - propagate rank-zero persistence failure.
                publication[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(publication, src=0)
        publication_result = publication[0]
        if not isinstance(publication_result, Mapping) or publication_result.get("ok") is not True:
            detail = (
                publication_result.get("error")
                if isinstance(publication_result, Mapping)
                else repr(publication_result)
            )
            raise RuntimeError(f"Could not persist external-academic receipt: {detail}")
        if get_rank() == 0:
            log.info(
                "Wrote and strictly reloaded canonical external-academic receipt %s (sha256=%s)",
                args.output,
                publication_result["sha256"],
            )
        return receipt
    finally:
        teardown_training_environment()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build-manifest",
        help="Freeze validation IDs, source hashes, controls, and contamination on CPU.",
    )
    build.add_argument("--output", required=True)
    build.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS, default=list(DEFAULT_TASKS))
    build.add_argument(
        "--examples-per-task",
        type=int,
        default=DEFAULT_EXAMPLES_PER_TASK,
        help=(
            f"Deterministic per-task cap (confirmatory default: {DEFAULT_EXAMPLES_PER_TASK}); "
            "diagnostic manifests also require at least two examples per task."
        ),
    )
    build.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    build.add_argument("--train-image-inventory", required=True)

    evaluate = commands.add_parser(
        "evaluate",
        help="Evaluate one native multimodal checkpoint against an immutable manifest.",
    )
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--config")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--hf-cache")
    evaluate.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    evaluate.add_argument("--max-crops", type=int, default=DEFAULT_MAX_CROPS)
    evaluate.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    evaluate.add_argument("--sequence-bucket-size", type=int, default=DEFAULT_SEQUENCE_BUCKET_SIZE)
    evaluate.add_argument("--checkpoint-load-threads", type=int, default=8)

    validate = commands.add_parser(
        "validate-receipt",
        help="Strictly reload a receipt and rederive live sources, checkpoint, rows, and metrics.",
    )
    validate.add_argument("--receipt", required=True)
    validate.add_argument("--expected-sha256", required=True)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parser().parse_args()
    if args.command == "build-manifest":
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite canonical artifact {output}")
        manifest = _build_manifest(args)
        _write_json_no_overwrite(output, manifest)
        log.info("Wrote canonical academic selection manifest to %s", output)
        return
    if args.command == "evaluate":
        _evaluate(args)
        return
    if args.command == "validate-receipt":
        validate_external_academic_receipt(
            args.receipt,
            args.expected_sha256,
            verify_live=True,
        )
        log.info("Validated canonical external-academic receipt %s", args.receipt)
        return
    raise AssertionError(f"Unknown command {args.command!r}")


if __name__ == "__main__":
    main()
