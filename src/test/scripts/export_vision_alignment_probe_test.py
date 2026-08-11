"""Tests for canonical exact-runtime Vision Alignment probe export."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

from olmo_core.data.multimodal.vision_alignment_sources import VisionAlignmentSourceSpec


def _load_script(name: str, filename: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / "data" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeRuntimeDataset:
    content_fingerprint = "fake-runtime-dataset-v1"

    def __init__(self, size: int = 8):
        self.size = size
        self.validated_annotations = 0
        self.requests: List[Tuple[int, int]] = []

    def __len__(self):
        return self.size

    def validate_required_annotations(self):
        self.validated_annotations += 1

    def get(self, index: int, epoch: int = 0):
        self.requests.append((index, epoch))
        return {
            "input_ids": np.array([index + 1, index + 2], dtype=np.int64),
            "labels": np.array([index + 2, -100], dtype=np.int64),
            "loss_masks": np.array([1.0, 0.0], dtype=np.float32),
            "position_ids": np.array([0, 1], dtype=np.int64),
            "token_type_ids": np.array([0, 0], dtype=np.int64),
            "images": np.full((1, 2, 3), index, dtype=np.float32),
            "pooled_patches_idx": np.array([[0, 1, 0, 1]], dtype=np.int64),
        }


def test_canonical_export_binds_indices_and_exact_serialized_rows(tmp_path: Path):
    exporter = _load_script(
        "_export_vision_alignment_probe_test_module", "export_vision_alignment_probe.py"
    )
    auditor = _load_script(
        "_audit_vision_alignment_probe_test_module", "audit_vision_alignment_mix.py"
    )
    dataset = _FakeRuntimeDataset()
    probe_path = tmp_path / "pixmo_caption.jsonl"

    source_entry = exporter.export_source_probe(
        dataset,
        source_name="pixmo_caption",
        output_path=probe_path,
        num_examples=4,
        seed=17,
    )

    assert dataset.validated_annotations == 1
    assert dataset.requests == [(index, 0) for index in source_entry["probe_indices"]]
    records = [json.loads(line) for line in probe_path.read_text().splitlines()]
    assert [record["probe_index"] for record in records] == source_entry["probe_indices"]
    assert source_entry["serialized_row_hashes_sha256"] == exporter._canonical_sha256(
        [record["serialized_row_sha256"] for record in records]
    )

    spec = VisionAlignmentSourceSpec(
        phase="bridge",
        pixmo_cap_path="synthetic",
        sequence_length=2560,
        max_crops=8,
        message_format="document",
        loss_token_weighting="root_subsegments_root_tokens",
        caption_prompt="Description:",
        transcript_prompt="Transcript:",
        require_transcript=True,
    )
    catalog = exporter.build_probe_catalog(
        spec=spec,
        source_entries=[source_entry],
        image_manifest_sha256="a" * 64,
        probe_seed=17,
        examples_per_source=4,
    )
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_bytes(exporter._canonical_bytes(catalog) + b"\n")

    report = auditor.audit_source_catalog(catalog_path, {"pixmo_caption": 1.0})

    assert report["status"] == "ok"
    assert report["source_catalog_version"] == 2
    assert report["inputs"]["pixmo_caption"]["probe_indices"] == source_entry["probe_indices"]
    assert report["inputs"]["pixmo_caption"]["serialized_row_hashes"] == [
        record["serialized_row_sha256"] for record in records
    ]

    records[0]["input_ids"][0] += 1
    probe_path.write_bytes(
        b"".join(exporter._canonical_bytes(record) + b"\n" for record in records)
    )
    catalog["sources"][0]["sha256"] = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    catalog_path.write_bytes(exporter._canonical_bytes(catalog) + b"\n")

    tampered_report = auditor.audit_source_catalog(catalog_path, {"pixmo_caption": 1.0})
    assert tampered_report["status"] == "failed"
    assert any("serialized row hashes differ" in failure for failure in tampered_report["failures"])
