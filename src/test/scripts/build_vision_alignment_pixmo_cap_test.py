import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict, load_from_disk


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "data"
        / "build_vision_alignment_pixmo_cap.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_build_vision_alignment_pixmo_cap_test_module", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_launcher():
    path = Path(__file__).resolve().parents[2] / "scripts" / "train" / "Vision-Alignment.py"
    spec = importlib.util.spec_from_file_location("_vision_alignment_builder_parity_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    return str(path.resolve())


def _save_source(tmp_path: Path, *, blank_caption=False, duplicate_validation=False):
    images = tmp_path / "images"
    images.mkdir(parents=True)
    train_images = [
        _write_image(images / "train-a", b"train-a"),
        _write_image(images / "train-overlap", b"validation-overlap"),
        str((images / "train-overlap").resolve()),
        _write_image(images / "train-c", b"train-duplicate"),
        _write_image(images / "train-d", b"train-duplicate"),
    ]
    validation_images = [
        _write_image(images / "validation-a", b"validation-overlap"),
        _write_image(images / "validation-b", b"validation-b"),
    ]
    if duplicate_validation:
        validation_images.append(_write_image(images / "validation-c", b"validation-overlap"))

    def split(paths, prefix):
        return Dataset.from_dict(
            {
                "image": paths,
                "caption": [
                    "" if blank_caption and index == 0 else f"{prefix} caption {index}"
                    for index in range(len(paths))
                ],
                "transcripts": [[f"{prefix} transcript {index}"] for index in range(len(paths))],
                "image_url": [
                    f"https://example.invalid/{prefix}/{index}" for index in range(len(paths))
                ],
            }
        )

    source_path = tmp_path / "source"
    DatasetDict(
        {
            "train": split(train_images, "train"),
            "validation": split(validation_images, "validation"),
        }
    ).save_to_disk(str(source_path))
    loaded = load_from_disk(str(source_path))
    pins = {
        "expected_train_fingerprint": loaded["train"]._fingerprint,
        "expected_train_examples": len(loaded["train"]),
        "expected_validation_fingerprint": loaded["validation"]._fingerprint,
        "expected_validation_examples": len(loaded["validation"]),
    }
    return source_path, pins, train_images, validation_images


def _build(module, source_path, output_path, pins, **kwargs):
    return module.build_pixmo_cap_artifact(
        source_dataset_path=str(source_path),
        output_dir=str(output_path),
        workers=2,
        scan_batch_size=2,
        max_shard_size="1MB",
        **pins,
        **kwargs,
    )


def test_row_image_path_digest_matches_portable_vector():
    module = _load_module()
    assert (
        module.row_image_paths_sha256(["/images/a", "/images/b", "/images/a"])
        == "cfcbbb377fe84bb193946d59a44d0ddec6dd583a7ea1090eb569807f98743c75"
    )


def test_builder_filters_all_train_overlap_and_publishes_atomic_v3_artifact(tmp_path, monkeypatch):
    module = _load_module()
    source, pins, train_images, validation_images = _save_source(tmp_path)
    output = tmp_path / "artifact"
    calls = []
    original_hash = module._sha256_file_stable

    def tracked_hash(path, signature):
        calls.append(str(path))
        return original_hash(path, signature)

    monkeypatch.setattr(module, "_sha256_file_stable", tracked_hash)
    manifest = _build(module, source, output, pins)

    assert len(calls) == len(set(train_images + validation_images))
    assert not (tmp_path / ".artifact.building").exists()
    assert manifest["format"] == "vision_alignment_validation_manifest"
    assert manifest["version"] == 3
    assert manifest["builder"]["script"] == module.BUILDER_SCRIPT
    assert manifest["filtering"] == {
        "source_overlap_unique_images": 1,
        "removed_train_examples": 2,
        "validation_duplicate_examples": 0,
        "output_overlap_unique_images": 0,
    }
    assert manifest["output"]["splits"]["train"]["examples"] == 3
    assert manifest["output"]["splits"]["validation"]["examples"] == 2
    assert (
        manifest["source"]["splits"]["validation"]["row_image_paths_sha256"]
        == manifest["output"]["splits"]["validation"]["row_image_paths_sha256"]
    )

    saved = load_from_disk(str(output / "dataset"))
    assert saved["train"]["image"] == [train_images[index] for index in (0, 3, 4)]
    assert saved["validation"]["image"] == validation_images
    for split in ("train", "validation"):
        inventory = (output / f"{split}-images.sha256").read_text().splitlines()
        rows = (output / f"{split}-row-images.sha256").read_text().splitlines()
        assert inventory == sorted(set(inventory))
        assert set(rows) == set(inventory)
        assert len(rows) == len(saved[split])
        assert (output / f"{split}-images.sha256").read_bytes().endswith(b"\n")
        assert (output / f"{split}-row-images.sha256").read_bytes().endswith(b"\n")
        assert hashlib.sha256((output / f"{split}-images.sha256").read_bytes()).hexdigest() == (
            manifest["inventories"][split]["sha256"]
        )

    manifest_path = output / "vision-alignment-validation-manifest.json"
    assert json.loads(manifest_path.read_text()) == manifest
    assert (output / "COMPLETE").read_text().strip() == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    launcher = _load_launcher()
    monkeypatch.setattr(
        launcher,
        "_CANONICAL_PIXMO_SOURCE_DATASET",
        str(source.resolve()),
    )
    monkeypatch.setattr(
        launcher,
        "_CANONICAL_PIXMO_SOURCE_SPLITS",
        {
            "train": (
                pins["expected_train_fingerprint"],
                pins["expected_train_examples"],
            ),
            "validation": (
                pins["expected_validation_fingerprint"],
                pins["expected_validation_examples"],
            ),
        },
    )
    source_audit = {
        "image_manifest_sha256": manifest["inventories"]["train"]["sha256"],
        "inputs": {
            name: {
                "dataset_fingerprint": manifest["output"]["splits"]["train"]["dataset_fingerprint"],
                "dataset_size": manifest["output"]["splits"]["train"]["examples"],
            }
            for name in ("pixmo_caption", "pixmo_transcript")
        },
    }
    config = SimpleNamespace(
        data=SimpleNamespace(
            allow_unpinned_synthetic_smoke=False,
            pixmo_cap_path=str(output / "dataset"),
        ),
        evaluation=SimpleNamespace(
            validation_manifest_path=str(manifest_path),
            validation_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            examples_per_source=2,
        ),
    )
    assert launcher._validate_validation_manifest(config, source_audit) == manifest
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _build(module, source, output, pins)


def test_builder_fails_closed_on_source_pin_and_blank_annotation(tmp_path):
    module = _load_module()
    source, pins, _, _ = _save_source(tmp_path / "pin")
    bad_pins = dict(pins)
    bad_pins["expected_train_examples"] += 1
    with pytest.raises(ValueError, match="source pin mismatch"):
        _build(module, source, tmp_path / "pin-artifact", bad_pins)

    blank_source, blank_pins, _, _ = _save_source(tmp_path / "blank", blank_caption=True)
    blank_output = tmp_path / "blank-artifact"
    with pytest.raises(ValueError, match="blank or non-string caption"):
        _build(module, blank_source, blank_output, blank_pins)
    assert not blank_output.exists()


def test_builder_preserves_and_records_duplicate_validation_content(tmp_path):
    module = _load_module()
    source, pins, _, validation_images = _save_source(tmp_path, duplicate_validation=True)
    output = tmp_path / "artifact"
    manifest = _build(module, source, output, pins)
    saved = load_from_disk(str(output / "dataset"))
    assert saved["validation"]["image"] == validation_images
    assert manifest["filtering"]["validation_duplicate_examples"] == 1
    assert manifest["output"]["splits"]["validation"]["examples"] == 3
    assert manifest["output"]["splits"]["validation"]["unique_image_content"] == 2


def test_exact_resume_reuses_durable_path_hash_cache(tmp_path, monkeypatch):
    module = _load_module()
    source, pins, train_images, validation_images = _save_source(tmp_path)
    output = tmp_path / "artifact"
    original_save = module._save_dataset_dict
    monkeypatch.setattr(
        module,
        "_save_dataset_dict",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated interruption")),
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _build(module, source, output, pins)
    staging = tmp_path / ".artifact.building"
    assert staging.is_dir()
    assert (staging / "image-hash-cache.sqlite3").is_file()

    monkeypatch.setattr(module, "_save_dataset_dict", original_save)
    fresh_hashes = []
    original_hash = module._sha256_file_stable

    def tracked_hash(path, signature):
        fresh_hashes.append(str(path))
        return original_hash(path, signature)

    monkeypatch.setattr(module, "_sha256_file_stable", tracked_hash)
    _build(module, source, output, pins, resume=True)
    assert fresh_hashes == []
    assert len(set(train_images + validation_images)) > 0
    assert output.is_dir()
