"""Focused end-to-end tests for the perception provenance builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from olmo_core.data.multimodal import (
    vision_alignment_perception_provenance as provenance,
)
from olmo_core.data.multimodal.finevision import FINEVISION_ROOT
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_SOURCE_NAMES,
    load_perception_provenance_manifest,
)
from olmo_core.data.multimodal.vision_alignment_perception_sources import (
    VisionAlignmentPerceptionSourceSpec,
)


def _load_builder():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "data"
        / "build_vision_alignment_perception_provenance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_build_vision_alignment_perception_provenance_test_module",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _RawDataset:
    def __init__(self, images, fingerprint):
        self.images = tuple(images)
        self.content_fingerprint = fingerprint
        self.validation_calls = 0

    def __len__(self):
        return len(self.images)

    def raw_image_references(self, index):
        return (self.images[index],)

    def validate_required_annotations(self):
        self.validation_calls += 1

    def annotation_content_sha256(self):
        digest = hashlib.sha256()
        for image in self.images:
            digest.update(image if isinstance(image, bytes) else str(image).encode())
            digest.update(b"\0")
        return digest.hexdigest()


def _production_spec(tmp_path: Path) -> VisionAlignmentPerceptionSourceSpec:
    return VisionAlignmentPerceptionSourceSpec(
        phase="perception",
        pixmo_cap_path=str((tmp_path / "pixmo-cap").resolve()),
        sequence_length=2560,
        max_crops=8,
        message_format="document",
        loss_token_weighting="root_subsegments_root_tokens",
        caption_prompt="Description:",
        transcript_prompt="Transcript:",
        require_transcript=True,
        finevision_root=FINEVISION_ROOT,
        finevision_visualweb_path=str((tmp_path / "finevision-visualweb").resolve()),
        finevision_geo170k_path=str((tmp_path / "finevision-geo170k").resolve()),
        finevision_visualweb_fingerprint="a" * 64,
        finevision_geo170k_fingerprint="b" * 64,
    )


def _write_image(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path.resolve())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mock_sources(tmp_path: Path):
    images = tmp_path / "images"
    validation_shared = _write_image(images / "validation-shared", b"validation-shared")
    # Different path, identical bytes: filtering must be content-based across sources.
    train_alias = _write_image(images / "train-alias", b"validation-shared")
    datasets = {}
    for source_ordinal, source_name in enumerate(PERCEPTION_SOURCE_NAMES):
        train = [
            _write_image(
                images / f"{source_name}-train-{index}",
                f"{source_name}-train-{index}".encode(),
            )
            for index in range(5)
        ]
        validation = [
            _write_image(
                images / f"{source_name}-validation-{index}",
                f"{source_name}-validation-{index}".encode(),
            )
            for index in range(2)
        ]
        if source_name == "pixmo_caption":
            validation[0] = validation_shared
        if source_name == "scalar_count":
            train[2] = train_alias
        if source_name == "pixmo_transcript":
            # Same raw path within a source exercises the path cache.
            train[4] = train[3]
        datasets[(source_name, "train")] = _RawDataset(
            train,
            hashlib.sha256(f"{source_name}:train".encode()).hexdigest(),
        )
        datasets[(source_name, "validation")] = _RawDataset(
            validation,
            hashlib.sha256(f"{source_name}:validation".encode()).hexdigest(),
        )
        assert source_ordinal >= 0
    return datasets, train_alias, validation_shared


def _fake_materialization(module, source_spec):
    manifest_path = Path(source_spec.finevision_visualweb_path).parent / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    raw = b'{"synthetic":"finevision-materialization"}\n'
    manifest_path.write_bytes(raw)
    return module.FineVisionMaterialization(
        manifest_path=manifest_path,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        content_sha256="d" * 64,
        source_root=Path(source_spec.finevision_root),
        visualweb_path=Path(source_spec.finevision_visualweb_path),
        geo170k_path=Path(source_spec.finevision_geo170k_path),
        visualweb_fingerprint=source_spec.finevision_visualweb_fingerprint,
        geo170k_fingerprint=source_spec.finevision_geo170k_fingerprint,
    )


def _run_builder(module, tmp_path, monkeypatch, *, resume=False, created_at=None):
    monkeypatch.setattr(module, "VALIDATION_IMAGE_CONTENTS_PER_SOURCE", 2)
    monkeypatch.setattr(provenance, "VALIDATION_IMAGE_CONTENTS_PER_SOURCE", 2)
    monkeypatch.setattr(
        provenance,
        "validate_finevision_materialization",
        lambda root, value, spec: provenance.FineVisionMaterializationReference(
            path=Path(root) / value["path"],
            sha256=value["sha256"],
            content_sha256=value["content_sha256"],
            visualweb_fingerprint=value["visualweb_fingerprint"],
            geo170k_fingerprint=value["geo170k_fingerprint"],
        ),
    )
    monkeypatch.setattr(
        module,
        "load_perception_provenance_manifest",
        lambda path, **_kwargs: type(
            "_Validated",
            (),
            {"content_sha256": json.loads(Path(path).read_text())["content_sha256"]},
        )(),
    )
    source_spec = _production_spec(tmp_path)
    datasets, train_alias, validation_shared = _mock_sources(tmp_path)
    inventory = module.vision_alignment_perception_implementation_inventory()
    registry_sha = module.vision_alignment_perception_source_registry_sha256()

    def build_dataset(_spec, _tokenizer, _token_ids, name, *, split, **_kwargs):
        if name == "audited_alignment" and split == "validation":
            raise AssertionError("audited_alignment must derive validation from train")
        return datasets[(name, split)]

    monkeypatch.setattr(module, "build_vision_alignment_perception_dataset", build_dataset)
    output = tmp_path / "artifact"
    manifest = module.build_vision_alignment_perception_provenance(
        source_spec=source_spec,
        expected_source_spec_sha256=source_spec.preprocessing_sha256,
        expected_source_registry_sha256=registry_sha,
        expected_implementation_inventory=inventory,
        expected_implementation_inventory_sha256=module._canonical_sha256(inventory),
        output_dir=output,
        tokenizer=object(),
        token_ids=object(),
        finevision_materialization=_fake_materialization(module, source_spec),
        resume=resume,
        created_at=created_at,
    )
    return manifest, output, datasets, train_alias, validation_shared


def test_builder_filters_cross_source_alias_and_loader_accepts_exact_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_builder()
    created_at = "2026-08-12T00:00:00Z"
    manifest, output, datasets, train_alias, validation_shared = _run_builder(
        module,
        tmp_path,
        monkeypatch,
        created_at=created_at,
    )

    manifest_path = output / module.MANIFEST_NAME
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    loaded = load_perception_provenance_manifest(
        manifest_path,
        expected_sha256=manifest_sha,
    )
    assert loaded.content_sha256 == manifest["content_sha256"]
    assert manifest["created_at"] == created_at
    assert manifest["unions"]["overlap_count"] == 0
    assert not set(
        (output / manifest["unions"]["train_unique_image_content"]["path"]).read_text().splitlines()
    ).intersection(
        (output / manifest["unions"]["validation_unique_image_content"]["path"])
        .read_text()
        .splitlines()
    )
    alias_hash = hashlib.sha256(Path(train_alias).read_bytes()).hexdigest()
    assert alias_hash == hashlib.sha256(Path(validation_shared).read_bytes()).hexdigest()
    assert alias_hash not in loaded.selection("scalar_count", "train").row_image_content_sha256
    assert manifest["filtering"]["scalar_count"] == {
        "candidate_train_examples": 5,
        "removed_train_examples": 1,
        "output_train_examples": 4,
    }
    assert all(
        dataset.validation_calls == 2
        for (source_name, split), dataset in datasets.items()
        if not (source_name == "audited_alignment" and split == "validation")
    )
    assert (output / "COMPLETE").read_text().strip() == manifest_sha
    assert not (tmp_path / ".artifact.building").exists()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.build_vision_alignment_perception_provenance(
            source_spec=loaded.source_spec,
            expected_source_spec_sha256=loaded.source_spec_sha256,
            expected_source_registry_sha256=module.vision_alignment_perception_source_registry_sha256(),
            expected_implementation_inventory=module.vision_alignment_perception_implementation_inventory(),
            expected_implementation_inventory_sha256=module._canonical_sha256(
                module.vision_alignment_perception_implementation_inventory()
            ),
            output_dir=output,
            tokenizer=object(),
            token_ids=object(),
            finevision_materialization=_fake_materialization(module, loaded.source_spec),
        )


def test_scan_rejects_dataset_identity_mutation_during_image_walk(tmp_path: Path):
    module = _load_builder()
    image = _write_image(tmp_path / "image", b"stable")

    class _MutatingDataset(_RawDataset):
        def raw_image_references(self, index):
            references = super().raw_image_references(index)
            if index == 0:
                self.content_fingerprint = "f" * 64
            return references

    dataset = _MutatingDataset([image, image], "e" * 64)
    cache = module.ImageHashCache(tmp_path / "cache.sqlite3", "1" * 64)
    try:
        with pytest.raises(ValueError, match="identity changed during image scan"):
            module._scan_dataset(
                dataset,
                source_name="pixmo_caption",
                physical_split="train",
                hasher=module.ImageHasher(cache),
            )
    finally:
        cache.close()


def test_final_snapshot_rejects_annotation_mutation_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_builder()
    dataset = _RawDataset([b"image"], "e" * 64)
    annotation_sha = dataset.annotation_content_sha256()
    scan = module.RawSplitScan(
        dataset=dataset,
        physical_split="train",
        base_fingerprint=dataset.content_fingerprint,
        base_annotation_sha256=annotation_sha,
        row_hashes=(hashlib.sha256(b"image").hexdigest(),),
    )
    dataset.images = (b"changed-annotation",)
    monkeypatch.setattr(
        module,
        "build_vision_alignment_perception_dataset",
        lambda *_args, **_kwargs: dataset,
    )

    with pytest.raises(ValueError, match="identity changed before publication"):
        module._validate_raw_scans_unchanged(
            {("pixmo_caption", "train"): scan},
            source_spec=_production_spec(tmp_path),
            tokenizer=object(),
            token_ids=object(),
        )


def test_final_snapshot_rechecks_path_signatures(tmp_path: Path):
    module = _load_builder()
    path = Path(_write_image(tmp_path / "image", b"initial"))
    cache = module.ImageHashCache(tmp_path / "cache.sqlite3", "1" * 64)
    hasher = module.ImageHasher(cache)
    try:
        hasher.hash_reference(str(path))
        path.write_bytes(b"mutated")
        with pytest.raises(ValueError, match="changed after hashing"):
            hasher.validate_paths_unchanged()
    finally:
        cache.close()


def test_validation_selection_is_deterministic_and_chooses_one_row_per_content():
    module = _load_builder()
    a = "a" * 64
    b = "b" * 64
    c = "c" * 64
    rows = (a, a, b, c)

    first = module._validation_representative_indices(
        rows,
        source_name="audited_alignment",
        source_spec_sha256="1" * 64,
        target_image_contents=2,
    )
    second = module._validation_representative_indices(
        rows,
        source_name="audited_alignment",
        source_spec_sha256="1" * 64,
        target_image_contents=2,
    )

    assert first == second
    assert len(first) == len({rows[index] for index in first}) == 2

    with pytest.raises(ValueError, match="only 3 distinct image contents"):
        module._validation_representative_indices(
            rows,
            source_name="audited_alignment",
            source_spec_sha256="1" * 64,
            target_image_contents=4,
        )


def test_production_validation_selects_exactly_512_distinct_hashes_source_independently():
    module = _load_builder()
    rows = tuple(hashlib.sha256(str(index).encode()).hexdigest() for index in range(513))

    first = module._validation_representative_indices(
        rows,
        source_name="pixmo_caption",
        source_spec_sha256="1" * 64,
        target_image_contents=module.VALIDATION_IMAGE_CONTENTS_PER_SOURCE,
    )
    second = module._validation_representative_indices(
        rows,
        source_name="scalar_count",
        source_spec_sha256="1" * 64,
        target_image_contents=module.VALIDATION_IMAGE_CONTENTS_PER_SOURCE,
    )

    assert module.VALIDATION_IMAGE_CONTENTS_PER_SOURCE == 512
    assert first == second
    assert len(first) == len({rows[index] for index in first}) == 512


def test_production_build_emits_512_distinct_validation_contents_for_every_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_builder()
    real_load_manifest = module.load_perception_provenance_manifest
    monkeypatch.setattr(
        provenance,
        "validate_finevision_materialization",
        lambda root, value, spec: provenance.FineVisionMaterializationReference(
            path=Path(root) / value["path"],
            sha256=value["sha256"],
            content_sha256=value["content_sha256"],
            visualweb_fingerprint=value["visualweb_fingerprint"],
            geo170k_fingerprint=value["geo170k_fingerprint"],
        ),
    )
    monkeypatch.setattr(
        module,
        "load_perception_provenance_manifest",
        lambda path, **_kwargs: type(
            "_Validated",
            (),
            {"content_sha256": json.loads(Path(path).read_text())["content_sha256"]},
        )(),
    )
    source_spec = _production_spec(tmp_path)
    datasets = {}
    for source_name in PERCEPTION_SOURCE_NAMES:
        datasets[(source_name, "train")] = _RawDataset(
            [f"{source_name}:train:{index}".encode() for index in range(513)],
            hashlib.sha256(f"{source_name}:train".encode()).hexdigest(),
        )
        datasets[(source_name, "validation")] = _RawDataset(
            [f"{source_name}:validation:{index}".encode() for index in range(513)],
            hashlib.sha256(f"{source_name}:validation".encode()).hexdigest(),
        )

    def build_dataset(_spec, _tokenizer, _token_ids, name, *, split, **_kwargs):
        if name == "audited_alignment" and split == "validation":
            raise AssertionError("audited_alignment must derive validation from train")
        return datasets[(name, split)]

    monkeypatch.setattr(module, "build_vision_alignment_perception_dataset", build_dataset)
    inventory = module.vision_alignment_perception_implementation_inventory()
    output = tmp_path / "artifact"
    module.build_vision_alignment_perception_provenance(
        source_spec=source_spec,
        expected_source_spec_sha256=source_spec.preprocessing_sha256,
        expected_source_registry_sha256=module.vision_alignment_perception_source_registry_sha256(),
        expected_implementation_inventory=inventory,
        expected_implementation_inventory_sha256=module._canonical_sha256(inventory),
        output_dir=output,
        tokenizer=object(),
        token_ids=object(),
        finevision_materialization=_fake_materialization(module, source_spec),
        created_at="2026-08-12T00:00:00Z",
    )

    copied = output / "upstream" / module.FINEVISION_MATERIALIZATION_MANIFEST
    assert (
        copied.read_bytes() == _fake_materialization(module, source_spec).manifest_path.read_bytes()
    )
    loaded = real_load_manifest(output / module.MANIFEST_NAME)
    for source_name in PERCEPTION_SOURCE_NAMES:
        selection = loaded.selection(source_name, "validation")
        assert len(selection.indices) == 512
        assert len(selection.row_image_content_sha256) == 512
        assert len(selection.unique_image_content_sha256) == 512


def test_interruption_keeps_only_staging_and_exact_resume_reuses_hash_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_builder()
    monkeypatch.setattr(module, "VALIDATION_IMAGE_CONTENTS_PER_SOURCE", 2)
    monkeypatch.setattr(
        module,
        "load_perception_provenance_manifest",
        lambda path, **_kwargs: type(
            "_Validated",
            (),
            {"content_sha256": json.loads(Path(path).read_text())["content_sha256"]},
        )(),
    )
    source_spec = _production_spec(tmp_path)
    datasets, _, _ = _mock_sources(tmp_path)
    inventory = module.vision_alignment_perception_implementation_inventory()
    registry_sha = module.vision_alignment_perception_source_registry_sha256()
    monkeypatch.setattr(
        module,
        "build_vision_alignment_perception_dataset",
        lambda _spec, _tokenizer, _token_ids, name, *, split, **_kwargs: datasets[(name, split)],
    )
    output = tmp_path / "artifact"
    original_write_lines = module._write_lines
    calls = 0

    def interrupt(path, values):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected interruption")
        return original_write_lines(path, values)

    monkeypatch.setattr(module, "_write_lines", interrupt)
    kwargs = {
        "source_spec": source_spec,
        "expected_source_spec_sha256": source_spec.preprocessing_sha256,
        "expected_source_registry_sha256": registry_sha,
        "expected_implementation_inventory": inventory,
        "expected_implementation_inventory_sha256": module._canonical_sha256(inventory),
        "output_dir": output,
        "tokenizer": object(),
        "token_ids": object(),
        "finevision_materialization": _fake_materialization(module, source_spec),
    }
    with pytest.raises(RuntimeError, match="injected interruption"):
        module.build_vision_alignment_perception_provenance(**kwargs)
    staging = tmp_path / ".artifact.building"
    assert staging.is_dir()
    assert not output.exists()
    assert not (staging / module.MANIFEST_NAME).exists()
    assert (staging / "image-hash-cache.sqlite3").is_file()
    resumed_created_at = json.loads((staging / "build-state.json").read_text())["created_at"]

    monkeypatch.setattr(module, "_write_lines", original_write_lines)
    fresh_hashes = []
    original_hash = module._sha256_path_stable

    def tracked_hash(path, signature):
        fresh_hashes.append(path)
        return original_hash(path, signature)

    monkeypatch.setattr(module, "_sha256_path_stable", tracked_hash)
    manifest = module.build_vision_alignment_perception_provenance(**kwargs, resume=True)
    assert fresh_hashes == []
    assert output.is_dir()
    assert manifest["status"] == "verified"
    assert manifest["created_at"] == resumed_created_at


def test_resume_recovers_empty_pre_state_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_builder()
    (tmp_path / ".artifact.building").mkdir()

    manifest, output, _, _, _ = _run_builder(
        module,
        tmp_path,
        monkeypatch,
        resume=True,
        created_at="2026-08-12T00:00:00Z",
    )

    assert output.is_dir()
    assert manifest["created_at"] == "2026-08-12T00:00:00Z"


def test_atomic_publish_never_replaces_existing_directory(tmp_path: Path):
    module = _load_builder()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module._rename_directory_no_replace(source, destination)

    assert source.is_dir()
    assert destination.is_dir()


def test_finevision_binding_rejects_raw_root_fallback(tmp_path: Path):
    module = _load_builder()
    artifact_root = tmp_path / "materialized"
    visualweb = artifact_root / "visualwebinstruct-filtered"
    geo = artifact_root / "geo170k-align"
    visualweb.mkdir(parents=True)
    geo.mkdir()
    materialization = module.FineVisionMaterialization(
        manifest_path=artifact_root / "manifest.json",
        raw_sha256="a" * 64,
        content_sha256="b" * 64,
        source_root=module.CANONICAL_FINEVISION_SOURCE_ROOT,
        visualweb_path=visualweb.resolve(),
        geo170k_path=geo.resolve(),
        visualweb_fingerprint="a" * 64,
        geo170k_fingerprint="b" * 64,
    )
    spec = replace(
        _production_spec(tmp_path),
        finevision_root=str(module.CANONICAL_FINEVISION_SOURCE_ROOT),
        finevision_visualweb_path=None,
        finevision_geo170k_path=None,
    )
    bound = module._bind_finevision_materialization(spec, materialization)
    assert bound.finevision_visualweb_path == str(visualweb.resolve())
    assert bound.finevision_geo170k_path == str(geo.resolve())

    with pytest.raises(ValueError, match="differs from the FineVision artifact"):
        module._bind_finevision_materialization(
            replace(spec, finevision_visualweb_path=str(tmp_path / "raw-fallback")),
            materialization,
        )

    with pytest.raises(ValueError, match="does not bind the verified FineVision artifact"):
        module.build_vision_alignment_perception_provenance(
            source_spec=replace(bound, finevision_visualweb_fingerprint="f" * 64),
            expected_source_spec_sha256=replace(
                bound, finevision_visualweb_fingerprint="f" * 64
            ).preprocessing_sha256,
            expected_source_registry_sha256=module.vision_alignment_perception_source_registry_sha256(),
            expected_implementation_inventory=module.vision_alignment_perception_implementation_inventory(),
            expected_implementation_inventory_sha256=module._canonical_sha256(
                module.vision_alignment_perception_implementation_inventory()
            ),
            output_dir=tmp_path / "unbound-artifact",
            tokenizer=object(),
            token_ids=object(),
            finevision_materialization=materialization,
        )


def test_finevision_materialization_validation_binds_current_bytes_and_live_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_builder()
    source_root = tmp_path / "raw"
    artifact = tmp_path / "materialized"
    artifact.mkdir()
    source_specs = (
        ("visualwebinstruct(filtered)", "visualwebinstruct-filtered", 1, 1),
        ("geo170k(align)", "geo170k-align", 1, 1),
    )
    monkeypatch.setattr(module, "CANONICAL_FINEVISION_SOURCE_ROOT", source_root)
    monkeypatch.setattr(module, "CANONICAL_FINEVISION_SOURCES", source_specs)

    sources = []
    outputs = []
    live_fingerprints = {}
    for ordinal, (name, output_name, shard_count, rows) in enumerate(source_specs):
        source_shard = source_root / name / "train-00000.parquet"
        source_shard.parent.mkdir(parents=True)
        source_shard.write_bytes(f"raw-{ordinal}".encode())
        output_dir = artifact / output_name
        output_dir.mkdir()
        output_shard = output_dir / "data-00000-of-00001.arrow"
        output_shard.write_bytes(f"arrow-{ordinal}".encode())
        (output_dir / "dataset_info.json").write_text("{}\n")
        dataset_info_sha = _sha256(output_dir / "dataset_info.json")
        receipt = {
            "source_sha256": _sha256(source_shard),
            "output_sha256": _sha256(output_shard),
            "rows": rows,
        }
        output_shard.with_suffix(".receipt.json").write_text(json.dumps(receipt) + "\n")
        sources.append(
            {
                "name": name,
                "output_name": output_name,
                "shards": [
                    {
                        "path": source_shard.relative_to(source_root).as_posix(),
                        "bytes": source_shard.stat().st_size,
                        "rows": rows,
                        "sha256": _sha256(source_shard),
                    }
                ],
                "shard_count": shard_count,
                "rows": rows,
                "physical_schema_sha256": "1" * 64,
                "source_metadata_sha256": "2" * 64,
            }
        )
        output = {
            "name": name,
            "path": output_name,
            "rows": rows,
            "physical_schema_sha256": "1" * 64,
            "dataset_info_sha256": dataset_info_sha,
            "shards": [
                {
                    "path": output_shard.relative_to(artifact).as_posix(),
                    "bytes": output_shard.stat().st_size,
                    "rows": rows,
                    "sha256": _sha256(output_shard),
                }
            ],
        }
        from scripts.data.materialize_vision_alignment_finevision import (
            output_dataset_fingerprint,
        )

        output["dataset_fingerprint"] = output_dataset_fingerprint(
            source_name=name,
            rows=rows,
            physical_schema_sha256=output["physical_schema_sha256"],
            shards=output["shards"],
            dataset_info_sha256=dataset_info_sha,
        )
        live_fingerprints[str(output_dir.resolve())] = output["dataset_fingerprint"]
        outputs.append(output)

    materializer = Path(module.__file__).resolve().with_name(module.FINEVISION_MATERIALIZER_SCRIPT)
    manifest = {
        "format": module.FINEVISION_MATERIALIZATION_FORMAT,
        "version": module.FINEVISION_MATERIALIZATION_VERSION,
        "builder_sha256": _sha256(materializer),
        "source_root": str(source_root.resolve()),
        "sources": sources,
        "status": "verified",
        "created_at": "2026-08-12T00:00:00Z",
        "outputs": outputs,
    }
    manifest["content_sha256"] = module._canonical_sha256(manifest)
    manifest_path = artifact / module.FINEVISION_MATERIALIZATION_MANIFEST
    manifest_path.write_bytes(module._canonical_bytes(manifest) + b"\n")
    manifest_sha = _sha256(manifest_path)
    (artifact / "COMPLETE").write_text(manifest_sha + "\n")

    class _Live:
        def __init__(self, fingerprint):
            self._fingerprint = fingerprint

        def __len__(self):
            return 1

    from olmo_core.data.multimodal import dataset_compat

    monkeypatch.setattr(
        dataset_compat,
        "load_from_disk_compat",
        lambda path: _Live(live_fingerprints[str(Path(path).resolve())]),
    )
    verified = module._validate_finevision_materialization(manifest_path, manifest_sha)
    assert verified.visualweb_path == (artifact / "visualwebinstruct-filtered").resolve()
    assert verified.geo170k_path == (artifact / "geo170k-align").resolve()

    provenance_root = tmp_path / "provenance"
    copied = provenance_root / "upstream" / module.FINEVISION_MATERIALIZATION_MANIFEST
    copied.parent.mkdir(parents=True)
    copied.write_bytes(manifest_path.read_bytes())
    source_spec = replace(
        _production_spec(tmp_path),
        finevision_root=str(source_root.resolve()),
        finevision_visualweb_path=str(verified.visualweb_path),
        finevision_geo170k_path=str(verified.geo170k_path),
        finevision_visualweb_fingerprint=verified.visualweb_fingerprint,
        finevision_geo170k_fingerprint=verified.geo170k_fingerprint,
    )
    monkeypatch.setattr(provenance, "_FINEVISION_CANONICAL_SOURCES", source_specs)
    monkeypatch.setattr(provenance, "FINEVISION_ROOT", str(source_root.resolve()))
    runtime_reference = provenance.validate_finevision_materialization(
        provenance_root,
        {
            "path": f"upstream/{module.FINEVISION_MATERIALIZATION_MANIFEST}",
            "sha256": manifest_sha,
            "content_sha256": verified.content_sha256,
            "visualweb_fingerprint": verified.visualweb_fingerprint,
            "geo170k_fingerprint": verified.geo170k_fingerprint,
        },
        source_spec,
    )
    assert runtime_reference.sha256 == manifest_sha
    original_hash_file = provenance._sha256_file

    def reject_repeat_arrow_hash(path):
        if Path(path).suffix == ".arrow":
            raise AssertionError("cached verifier rehashed an unchanged Arrow shard")
        return original_hash_file(path)

    monkeypatch.setattr(provenance, "_sha256_file", reject_repeat_arrow_hash)
    assert (
        provenance.validate_finevision_materialization(
            provenance_root,
            {
                "path": f"upstream/{module.FINEVISION_MATERIALIZATION_MANIFEST}",
                "sha256": manifest_sha,
                "content_sha256": verified.content_sha256,
                "visualweb_fingerprint": verified.visualweb_fingerprint,
                "geo170k_fingerprint": verified.geo170k_fingerprint,
            },
            source_spec,
        )
        == runtime_reference
    )
    monkeypatch.setattr(provenance, "_sha256_file", original_hash_file)

    (artifact / "geo170k-align" / "data-00000-of-00001.arrow").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="output shard bytes differ"):
        module._validate_finevision_materialization(manifest_path, manifest_sha)
    with pytest.raises(ValueError, match="live output shard differs"):
        provenance.validate_finevision_materialization(
            provenance_root,
            {
                "path": f"upstream/{module.FINEVISION_MATERIALIZATION_MANIFEST}",
                "sha256": manifest_sha,
                "content_sha256": verified.content_sha256,
                "visualweb_fingerprint": verified.visualweb_fingerprint,
                "geo170k_fingerprint": verified.geo170k_fingerprint,
            },
            source_spec,
        )
