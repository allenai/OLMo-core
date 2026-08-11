import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2] / "scripts" / "data" / "audit_vision_alignment_mix.py"
    )
    spec = importlib.util.spec_from_file_location("_audit_vision_alignment_mix_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def auditor():
    return _load_module()


def _write_jsonl(path: Path, records) -> str:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_catalog(path: Path, auditor, sources) -> Path:
    sources = [
        {
            "dataset_fingerprint": f"test-dataset-{index}",
            "dataset_size": 1024,
            **source,
        }
        for index, source in enumerate(sources)
    ]
    path.write_text(
        json.dumps(
            {
                "format": auditor.SOURCE_CATALOG_FORMAT,
                "version": auditor.LEGACY_SOURCE_CATALOG_VERSION,
                "recipe_version": 1,
                "formatter_version": "vision-alignment-document-v1",
                "image_manifest_sha256": "a" * 64,
                "preprocessing_config_sha256": "b" * 64,
                "sources": sources,
            },
            sort_keys=True,
        )
    )
    return path


def test_jsonl_audit_reports_exact_metrics_and_loss_mass_sampling(tmp_path: Path, auditor):
    caption_path = tmp_path / "caption.jsonl"
    caption_sha = _write_jsonl(
        caption_path,
        [
            {
                "input_ids": [10, 11, 12, 13],
                "loss_masks": [0.0, 0.0, 1.0, 1.0],
                "image_crops": 2,
                "truncated": False,
            },
            {
                "input_ids": [20, 21, 22],
                "loss_masks": [0.0, 0.5, 0.5],
                "images": [[[0.0]]],
                "metadata": {"truncated": True},
            },
        ],
    )
    transcript_path = tmp_path / "transcript.jsonl"
    transcript_sha = _write_jsonl(
        transcript_path,
        [
            {
                "input_ids": [30, 31, 32, 33],
                "loss_masks": [0.0, 1.0, 1.0, 1.0],
                "image_crops": 1,
            }
        ],
    )
    catalog_path = _write_catalog(
        tmp_path / "catalog.json",
        auditor,
        [
            {
                "name": "pixmo_transcript",
                "format": "jsonl",
                "path": transcript_path.name,
                "sha256": transcript_sha,
            },
            {
                "name": "pixmo_caption",
                "format": "jsonl",
                "path": caption_path.name,
                "sha256": caption_sha,
            },
        ],
    )

    report = auditor.audit_source_catalog(
        catalog_path,
        {"pixmo_caption": 0.75, "pixmo_transcript": 0.25},
        phase="bridge",
    )

    assert report["status"] == "ok"
    assert report["recipe_version"] == 1
    assert report["formatter_version"] == "vision-alignment-document-v1"
    assert report["image_manifest_sha256"] == "a" * 64
    assert report["preprocessing_config_sha256"] == "b" * 64
    caption = report["sources"]["pixmo_caption"]
    assert caption["examples"] == {"seen": 2, "valid": 2, "errors": 0}
    assert caption["raw_input_tokens"] == {"total": 7, "mean": 3.5, "min": 3, "max": 4}
    assert caption["positive_supervised_tokens"]["total"] == 4
    assert caption["summed_loss_weight"]["total"] == 3.0
    assert caption["mean_sum_loss_masks"] == 1.5
    assert caption["image_crops"]["total"] == 3
    assert caption["truncated_examples"] == 1
    assert caption["zero_loss_examples"] == 0
    assert report["mean_loss_weight"] == {
        "pixmo_caption": 1.5,
        "pixmo_transcript": 3.0,
    }
    assert report["sampling_probabilities"] == pytest.approx(
        {"pixmo_caption": 6 / 7, "pixmo_transcript": 1 / 7}
    )
    assert report["expected_loss_mass"] == pytest.approx(
        {"pixmo_caption": 0.75, "pixmo_transcript": 0.25}
    )
    assert report["inputs"]["pixmo_caption"]["sha256"] == caption_sha
    assert report["inputs"]["pixmo_caption"]["dataset_fingerprint"] == "test-dataset-1"
    assert report["inputs"]["pixmo_caption"]["dataset_size"] == 1024
    assert len(report["input_content_sha256"]) == 64
    fingerprint = report.pop("fingerprint")
    assert fingerprint == hashlib.sha256(auditor._canonical_bytes(report)).hexdigest()


def test_npz_fixed_shape_examples_are_audited(tmp_path: Path, auditor):
    source_path = tmp_path / "native.npz"
    np.savez(
        source_path,
        input_ids=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
        loss_masks=np.array([[1.0, 1.0, 1.0], [0.0, 0.25, 0.25]], dtype=np.float32),
        image_crops=np.array([0, 2], dtype=np.int64),
        truncated=np.array([False, True], dtype=np.bool_),
    )
    catalog_path = _write_catalog(
        tmp_path / "catalog.json",
        auditor,
        [{"name": "native_text_replay", "format": "npz", "path": source_path.name}],
    )

    report = auditor.audit_source_catalog(catalog_path, {"native_text_replay": 1.0})

    assert report["status"] == "ok"
    source = report["sources"]["native_text_replay"]
    assert source["raw_input_tokens"]["total"] == 6
    assert source["positive_supervised_tokens"]["total"] == 5
    assert source["summed_loss_weight"]["total"] == 3.5
    assert source["mean_sum_loss_masks"] == 1.75
    assert source["image_crops"]["total"] == 2
    assert source["truncated_examples"] == 1
    assert report["sampling_probabilities"] == {"native_text_replay": 1.0}


def test_malformed_record_emits_failed_artifact_and_nonzero_exit(tmp_path: Path, auditor, capsys):
    source_path = tmp_path / "caption.jsonl"
    source_path.write_text(
        json.dumps({"input_ids": [1, 2], "loss_masks": [0.0, 1.0], "image_crops": 1})
        + "\n"
        + "not json\n"
        + json.dumps({"input_ids": [3, 4], "loss_masks": [0.0]})
        + "\n"
    )
    catalog_path = _write_catalog(
        tmp_path / "catalog.json",
        auditor,
        [{"name": "pixmo_caption", "format": "jsonl", "path": source_path.name}],
    )
    output_path = tmp_path / "audit.json"

    exit_code = auditor.main(
        [
            str(catalog_path),
            "--target-loss-mass",
            "pixmo_caption=1",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "audit failed" in captured.err
    report = json.loads(output_path.read_text())
    assert report["status"] == "failed"
    assert report["sources"]["pixmo_caption"]["examples"] == {
        "seen": 3,
        "valid": 1,
        "errors": 2,
    }
    assert len(report["sources"]["pixmo_caption"]["error_samples"]) == 2
    assert report["sampling_probabilities"] is None
    assert report["expected_loss_mass"] is None


def test_zero_loss_and_target_catalog_mismatch_cannot_calibrate(tmp_path: Path, auditor):
    source_path = tmp_path / "caption.jsonl"
    _write_jsonl(
        source_path,
        [{"input_ids": [1, 2], "loss_masks": [0.0, 0.0], "image_crops": 1}],
    )
    catalog_path = _write_catalog(
        tmp_path / "catalog.json",
        auditor,
        [{"name": "pixmo_caption", "format": "jsonl", "path": source_path.name}],
    )

    report = auditor.audit_source_catalog(
        catalog_path,
        {"pixmo_caption": 0.7, "pixmo_transcript": 0.3},
    )

    assert report["status"] == "failed"
    assert report["sources"]["pixmo_caption"]["zero_loss_examples"] == 1
    assert any("not positive" in failure for failure in report["failures"])
    assert any("target/catalog source mismatch" in failure for failure in report["failures"])


def test_declared_source_hash_is_verified(tmp_path: Path, auditor):
    source_path = tmp_path / "caption.jsonl"
    _write_jsonl(source_path, [{"input_ids": [1], "loss_masks": [1.0]}])
    catalog_path = _write_catalog(
        tmp_path / "catalog.json",
        auditor,
        [
            {
                "name": "pixmo_caption",
                "format": "jsonl",
                "path": source_path.name,
                "sha256": "0" * 64,
            }
        ],
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        auditor.load_source_catalog(catalog_path)


def test_phase_cli_writes_byte_identical_canonical_artifacts(tmp_path: Path, auditor):
    sources = []
    for name, weight in (("pixmo_caption", 2.0), ("pixmo_transcript", 1.0)):
        source_path = tmp_path / f"{name}.jsonl"
        _write_jsonl(
            source_path,
            [{"input_ids": [1, 2], "loss_masks": [0.0, weight], "image_crops": 1}],
        )
        sources.append({"name": name, "format": "jsonl", "path": source_path.name})
    catalog_path = _write_catalog(tmp_path / "catalog.json", auditor, sources)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    assert auditor.main([str(catalog_path), "--phase", "bridge", "--output", str(first)]) == 0
    assert auditor.main([str(catalog_path), "--phase", "bridge", "--output", str(second)]) == 0

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")
    assert first.read_bytes().count(b"\n") == 1
    report = json.loads(first.read_text())
    assert report["phase"] == "bridge"
    assert report["target_loss_mass"] == pytest.approx(
        {"pixmo_caption": 0.7, "pixmo_transcript": 0.3}
    )
