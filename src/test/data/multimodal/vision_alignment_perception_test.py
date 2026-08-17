"""Focused tests for perception-phase document datasets."""

from __future__ import annotations

import zlib
from typing import Any, Dict, List

import numpy as np
import pytest

from olmo_core.data.multimodal import finevision
from olmo_core.data.multimodal import vision_alignment_perception as perception
from olmo_core.data.multimodal.pixmo_points import PixMoCountDatasetConfig
from olmo_core.data.multimodal.vision_alignment_perception_sources import (
    VisionAlignmentPerceptionSourceSpec,
    build_vision_alignment_perception_dataset_config,
)
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds


class _Tokenizer:
    eos_token_id = 100257
    bos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return [(zlib.crc32(word.encode()) % 10000) + 10 for word in text.split()]


def _encoded_stub() -> Dict[str, np.ndarray]:
    return {
        "input_ids": np.array([1, 2, 3], dtype=np.int64),
        "labels": np.array([2, 3, 4], dtype=np.int64),
        "loss_masks": np.array([0.0, 1.0, 1.0], dtype=np.float32),
        "position_ids": np.arange(3, dtype=np.int64),
        "token_type_ids": np.zeros(3, dtype=np.int64),
        "images": np.zeros((1, 1, 1), dtype=np.float32),
        "pooled_patches_idx": np.zeros((1, 1), dtype=np.int64),
    }


def test_ocr_document_uses_fixed_native_prompt_and_modal_answer(monkeypatch, tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"not decoded in this test")
    rows = {
        "text_vqa": [
            {
                "image": str(image),
                "question": "  What word is visible?  ",
                "answers": ["Blue", "blue", "red"],
            }
        ],
        "doc_qa": [
            {
                "image": str(image),
                "question": "What is the total?",
                "answers": ["42"],
            }
        ],
    }
    monkeypatch.setattr(
        perception,
        "build_academic_data",
        lambda name, split: rows[name],
    )
    monkeypatch.setattr(perception, "decode_pil_image", lambda value: value)
    captured: List[Any] = []

    def encode(_tokenizer, image_value, turns, **kwargs):
        captured.append((image_value, turns, kwargs))
        return _encoded_stub()

    monkeypatch.setattr(perception, "encode_sft_example", encode)
    config = perception.VisionAlignmentOcrDocumentDatasetConfig(
        source_names=("text_vqa", "doc_qa"),
        max_sequence_length=3,
    )
    dataset = config.build(_Tokenizer())
    dataset.validate_required_annotations()

    assert len(dataset) == 2
    assert dataset[0]["input_ids"].tolist() == [1, 2, 3]
    assert captured[0][1] == [("Question: What word is visible?\nAnswer:", "Blue")]
    assert captured[0][2]["message_format"] == "document"
    assert captured[0][2]["max_images"] == 1
    assert len(dataset.content_fingerprint) == 64


def test_ocr_fingerprint_binds_annotations_and_validation_fails_closed(monkeypatch, tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"x")
    rows = [
        {"image": str(image), "question": "Q", "answers": ["A"]},
        {"image": str(tmp_path / "missing.png"), "question": "", "answers": []},
    ]
    monkeypatch.setattr(perception, "build_academic_data", lambda _name, split: rows)
    config = perception.VisionAlignmentOcrDocumentDatasetConfig(source_names=("text_vqa",))
    first = config.build(_Tokenizer())
    first_fingerprint = first.content_fingerprint

    with pytest.raises(ValueError, match="2 invalid rows|1 invalid rows"):
        first.validate_required_annotations()

    rows[0] = {**rows[0], "answers": ["changed"]}
    second = config.build(_Tokenizer())
    assert second.content_fingerprint != first_fingerprint


def test_ocr_rejects_chat_layout_and_unreviewed_sources(monkeypatch):
    monkeypatch.setattr(perception, "build_academic_data", lambda _name, split: [])
    with pytest.raises(ValueError, match="message_format='document'"):
        perception.VisionAlignmentOcrDocumentDatasetConfig(
            source_names=("text_vqa",), message_format="qwen3"
        ).build(_Tokenizer())
    with pytest.raises(ValueError, match="selected from"):
        perception.VisionAlignmentOcrDocumentDatasetConfig(source_names=("science_qa_img",)).build(
            _Tokenizer()
        )


def _finevision_arrow(*, texts, images, formatting=4, visual=4, relevance=4):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "texts": texts,
            "images": images,
            "formatting_min": [formatting] * len(texts),
            "visual_dependency_min": [visual] * len(texts),
            "image_correspondence_min": [4] * len(texts),
            "relevance_min": [relevance] * len(texts),
        }
    )


def test_finevision_strict_annotations_and_fingerprint(monkeypatch):
    valid = _finevision_arrow(
        texts=[[{"user": "Describe it.", "assistant": "A triangle."}]],
        images=[[{"bytes": b"image", "path": None}]],
    )
    monkeypatch.setattr(finevision, "load_hf_dataset", lambda *args, **kwargs: valid)
    config = finevision.FineVisionDatasetConfig(
        dataset_path="/synthetic/reviewed",
        message_format="document",
        min_formatting=4,
        min_visual_dependency=4,
        min_relevance=4,
        require_quality_columns=True,
        strict_annotations=True,
    )
    dataset = config.build(_Tokenizer())
    dataset.validate_required_annotations()
    assert len(dataset.content_fingerprint) == 64

    invalid = _finevision_arrow(
        texts=[[{"user": "", "assistant": "answer"}]],
        images=[[]],
    )
    monkeypatch.setattr(finevision, "load_hf_dataset", lambda *args, **kwargs: invalid)
    dataset = config.build(_Tokenizer())
    with pytest.raises(ValueError, match="exactly one image"):
        dataset.validate_required_annotations()


def test_finevision_required_quality_column_fails_closed(monkeypatch):
    arrow = _finevision_arrow(
        texts=[[{"user": "Q", "assistant": "A"}]],
        images=[[{"bytes": b"image", "path": None}]],
    ).remove_columns("relevance_min")
    monkeypatch.setattr(finevision, "load_hf_dataset", lambda *args, **kwargs: arrow)
    config = finevision.FineVisionDatasetConfig(
        dataset_path="/synthetic/reviewed",
        min_relevance=4,
        require_quality_columns=True,
    )
    with pytest.raises(ValueError, match="relevance_min"):
        config.build(_Tokenizer())


class _AlignmentChild:
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.content_fingerprint = f"fingerprint-{name}"
        self.validated = False

    def __len__(self):
        return self.size

    def validate_required_annotations(self):
        self.validated = True

    def get(self, index: int, epoch: int = 0):
        return {"source": self.name, "index": index, "epoch": epoch}


def test_audited_alignment_combines_reviewed_sources(monkeypatch):
    built = []

    def build(config, _tokenizer):
        child = _AlignmentChild(config.config_name, 2 if "visualweb" in config.config_name else 1)
        built.append((config, child))
        return child

    monkeypatch.setattr(finevision.FineVisionDatasetConfig, "build", build)
    config = perception.VisionAlignmentAuditedAlignmentDatasetConfig()
    dataset = config.build(_Tokenizer())
    dataset.validate_required_annotations()

    assert len(dataset) == 3
    assert dataset.get(0, 7)["source"] == "visualwebinstruct(filtered)"
    assert dataset.get(2, 7)["source"] == "geo170k(align)"
    assert all(child.validated for _, child in built)
    assert all(child_config.strict_annotations for child_config, _ in built)
    assert all(child_config.message_format == "document" for child_config, _ in built)
    assert all(child_config.max_images == 1 for child_config, _ in built)
    assert len(dataset.content_fingerprint) == 64


def test_source_registry_builds_all_missing_perception_adapters():
    spec = VisionAlignmentPerceptionSourceSpec(
        phase="perception",
        pixmo_cap_path="/pixmo-cap",
        sequence_length=2560,
        max_crops=8,
        message_format="document",
        loss_token_weighting="root_subsegments_root_tokens",
        caption_prompt="Description:",
        transcript_prompt="Transcript:",
        require_transcript=True,
    )
    token_ids = Molmo2TokenIds()

    scalar = build_vision_alignment_perception_dataset_config(
        spec, token_ids, "scalar_count", split="validation"
    )
    ocr = build_vision_alignment_perception_dataset_config(
        spec, token_ids, "ocr_document", split="validation"
    )
    alignment = build_vision_alignment_perception_dataset_config(
        spec, token_ids, "audited_alignment", split="train"
    )

    assert isinstance(scalar, PixMoCountDatasetConfig)
    assert scalar.mode == "scalar_count"
    assert scalar.split == "validation"
    assert scalar.max_sequence_length == 2560
    assert isinstance(ocr, perception.VisionAlignmentOcrDocumentDatasetConfig)
    assert ocr.source_names == perception.VISION_ALIGNMENT_OCR_SOURCES
    assert ocr.message_format == "document"
    assert isinstance(alignment, perception.VisionAlignmentAuditedAlignmentDatasetConfig)
    assert alignment.min_formatting == 4
    assert alignment.min_visual_dependency == 4
    assert alignment.min_relevance == 4
