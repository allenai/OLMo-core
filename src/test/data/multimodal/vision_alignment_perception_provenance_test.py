"""Adversarial tests for perception provenance and atomic probe export."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Sequence

import pytest

from olmo_core.data.multimodal import (
    vision_alignment_perception_provenance as provenance,
)
from olmo_core.data.multimodal.finevision import FINEVISION_ROOT
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_SOURCE_NAMES,
    PerceptionSplitSelection,
    SelectedVisionAlignmentDataset,
    image_reference_sha256,
    load_perception_provenance_manifest,
    selected_dataset_fingerprint,
)
from olmo_core.data.multimodal.vision_alignment_perception_sources import (
    VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
    VisionAlignmentPerceptionSourceSpec,
    vision_alignment_perception_implementation_inventory,
    vision_alignment_perception_source_registry_sha256,
)


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_lines(root: Path, relative_path: str, values: Sequence[Any]) -> Dict[str, Any]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(f"{value}\n".encode("utf-8") for value in values)
    path.write_bytes(raw)
    return {"path": relative_path, "sha256": _sha256_bytes(raw), "count": len(values)}


def _write_selection(root: Path, relative_path: str, indices: Sequence[int]) -> Dict[str, Any]:
    reference = _write_lines(root, relative_path, indices)
    reference["indices_sha256"] = _canonical_sha256(list(indices))
    return reference


def _write_manifest(path: Path, value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("content_sha256", None)
    payload["content_sha256"] = _canonical_sha256(payload)
    raw = _canonical_bytes(payload) + b"\n"
    path.write_bytes(raw)
    raw_sha256 = _sha256_bytes(raw)
    (path.parent / "COMPLETE").write_text(raw_sha256 + "\n")
    return raw_sha256


@dataclass
class _ManifestArtifact:
    path: Path
    raw_sha256: str
    value: Dict[str, Any]
    builder_path: Path


def _production_source_spec(root: Path) -> VisionAlignmentPerceptionSourceSpec:
    return VisionAlignmentPerceptionSourceSpec(
        phase="perception",
        pixmo_cap_path=str(root / "pixmo-cap"),
        sequence_length=2560,
        max_crops=8,
        message_format="document",
        loss_token_weighting="root_subsegments_root_tokens",
        caption_prompt="Description:",
        transcript_prompt="Transcript:",
        require_transcript=True,
        finevision_root=FINEVISION_ROOT,
        finevision_visualweb_path=str(root / "finevision-visualweb"),
        finevision_geo170k_path=str(root / "finevision-geo170k"),
        finevision_visualweb_fingerprint="a" * 64,
        finevision_geo170k_fingerprint="b" * 64,
    )


def _build_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    overlap: bool = False,
) -> _ManifestArtifact:
    root = tmp_path / "provenance"
    root.mkdir(parents=True)
    builder_path = tmp_path / "build_vision_alignment_perception_provenance.py"
    builder_path.write_bytes(b"# pinned synthetic builder\n")
    monkeypatch.setattr(provenance, "_builder_script_path", lambda: builder_path)

    source_spec = _production_source_spec(root)
    upstream = root / "upstream" / provenance._FINEVISION_MATERIALIZATION_MANIFEST
    upstream.parent.mkdir()
    upstream_raw = b'{"synthetic":"finevision-materialization"}\n'
    upstream.write_bytes(upstream_raw)
    monkeypatch.setattr(
        provenance,
        "validate_finevision_materialization",
        lambda _root, _value, _spec: provenance.FineVisionMaterializationReference(
            path=upstream,
            sha256=_sha256_bytes(upstream_raw),
            content_sha256="c" * 64,
            visualweb_fingerprint="a" * 64,
            geo170k_fingerprint="b" * 64,
        ),
    )
    # Round-trip to the representation a JSON manifest actually contains.
    source_spec_value = json.loads(json.dumps(source_spec.as_canonical_dict()))
    source_spec_sha = source_spec.preprocessing_sha256
    sources: Dict[str, Any] = {}
    filtering: Dict[str, Any] = {}
    union_values: Dict[str, set[str]] = {"train": set(), "validation": set()}

    first_train_hash: str | None = None
    for source_ordinal, source_name in enumerate(PERCEPTION_SOURCE_NAMES):
        split_values: Dict[str, Any] = {}
        for logical_split in ("train", "validation"):
            if source_name == "audited_alignment":
                physical_split = "train"
                base_examples = 514
                indices = (512, 513) if logical_split == "train" else tuple(range(512))
                base_fingerprint = hashlib.sha256(f"{source_name}:train:base".encode()).hexdigest()[
                    :16
                ]
            else:
                physical_split = logical_split
                base_examples = 2 if logical_split == "train" else 512
                indices = (0, 1) if logical_split == "train" else tuple(range(512))
                base_fingerprint = hashlib.sha256(
                    f"{source_name}:{logical_split}:base".encode()
                ).hexdigest()[:16]
            row_hashes = [
                hashlib.sha256(
                    f"{source_name}:{logical_split}:image:{ordinal}".encode()
                ).hexdigest()
                for ordinal in range(len(indices))
            ]
            if logical_split == "train" and first_train_hash is None:
                first_train_hash = row_hashes[0]
            if overlap and logical_split == "validation" and source_ordinal == 0:
                assert first_train_hash is not None
                row_hashes[0] = first_train_hash
            prefix = f"sources/{source_name}/{logical_split}"
            selection = _write_selection(root, f"{prefix}/indices.txt", indices)
            unique_hashes = sorted(set(row_hashes))
            row_inventory = _write_lines(root, f"{prefix}/row-images.txt", row_hashes)
            unique_inventory = _write_lines(root, f"{prefix}/unique-images.txt", unique_hashes)
            runtime_fingerprint = selected_dataset_fingerprint(
                source_name=source_name,
                logical_split=logical_split,
                physical_split=physical_split,
                base_fingerprint=base_fingerprint,
                selection_indices_sha256=selection["indices_sha256"],
                source_spec_sha256=source_spec_sha,
            )
            split_values[logical_split] = {
                "physical_split": physical_split,
                "base_annotation_sha256": hashlib.sha256(
                    f"{source_name}:{physical_split}:annotations".encode()
                ).hexdigest(),
                "base_dataset_fingerprint": base_fingerprint,
                "base_examples": base_examples,
                "selection": selection,
                "runtime_dataset_fingerprint": runtime_fingerprint,
                "runtime_examples": len(indices),
                "row_image_content": row_inventory,
                "unique_image_content": unique_inventory,
            }
            union_values[logical_split].update(unique_hashes)

        components = {
            "audited_alignment": ["visualwebinstruct(filtered)", "geo170k(align)"],
            "ocr_document": list(source_spec.ocr_source_names),
        }.get(source_name, [source_name])
        sources[source_name] = {"components": components, **split_values}
        candidate_train = 514 if source_name == "audited_alignment" else 2
        filtering[source_name] = {
            "candidate_train_examples": candidate_train,
            "removed_train_examples": candidate_train - 2,
            "output_train_examples": 2,
        }

    union_refs = {
        f"{logical_split}_unique_image_content": _write_lines(
            root,
            f"unions/{logical_split}.txt",
            sorted(union_values[logical_split]),
        )
        for logical_split in ("train", "validation")
    }
    signature_image = tmp_path / "signature-image.bin"
    signature_image.write_bytes(b"signature image")
    signature_stat = signature_image.stat()
    image_path_signatures = _write_lines(
        root,
        "inventories/image-path-signatures.jsonl",
        [
            _canonical_bytes(
                {
                    "path": str(signature_image.resolve()),
                    "size_bytes": signature_stat.st_size,
                    "mtime_ns": signature_stat.st_mtime_ns,
                    "ctime_ns": signature_stat.st_ctime_ns,
                    "inode": signature_stat.st_ino,
                    "device": signature_stat.st_dev,
                    "sha256": _sha256_bytes(signature_image.read_bytes()),
                }
            ).decode("ascii")
        ],
    )
    value: Dict[str, Any] = {
        "format": provenance.PERCEPTION_PROVENANCE_FORMAT,
        "version": provenance.PERCEPTION_PROVENANCE_VERSION,
        "status": "verified",
        "phase": "perception",
        "created_at": "2026-08-12T00:00:00Z",
        "builder": {
            "name": "build_vision_alignment_perception_provenance",
            "version": 1,
            "script_sha256": _sha256_bytes(builder_path.read_bytes()),
        },
        "source_spec": source_spec_value,
        "source_spec_sha256": source_spec_sha,
        "source_registry_version": VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
        "source_registry_sha256": vision_alignment_perception_source_registry_sha256(),
        "source_implementation_inventory": (vision_alignment_perception_implementation_inventory()),
        "finevision_materialization": {
            "path": f"upstream/{provenance._FINEVISION_MATERIALIZATION_MANIFEST}",
            "sha256": _sha256_bytes(upstream_raw),
            "content_sha256": "c" * 64,
            "visualweb_fingerprint": "a" * 64,
            "geo170k_fingerprint": "b" * 64,
        },
        "image_path_signatures": image_path_signatures,
        "validation_selection": {
            "algorithm": "sha256-ranked-distinct-content-representatives-v1",
            "image_contents_per_source": 512,
        },
        "sources": sources,
        "unions": {**union_refs, "overlap_count": 0},
        "filtering": filtering,
    }
    manifest_path = root / provenance.PERCEPTION_PROVENANCE_MANIFEST
    raw_sha = _write_manifest(manifest_path, value)
    value = json.loads(manifest_path.read_text())
    return _ManifestArtifact(manifest_path, raw_sha, value, builder_path)


def _rewrite_artifact(
    artifact: _ManifestArtifact,
    mutate: Callable[[Dict[str, Any]], None],
) -> str:
    value = deepcopy(artifact.value)
    mutate(value)
    return _write_manifest(artifact.path, value)


def test_manifest_accepts_real_16_hex_arrow_fingerprints(tmp_path: Path, monkeypatch):
    artifact = _build_manifest(tmp_path, monkeypatch)

    loaded = load_perception_provenance_manifest(artifact.path, expected_sha256=artifact.raw_sha256)

    assert loaded.raw_sha256 == artifact.raw_sha256
    assert all(
        len(selection.base_dataset_fingerprint) == 16 for selection in loaded.selections.values()
    )
    assert loaded.source_spec.finevision_visualweb_path == str(
        artifact.path.parent / "finevision-visualweb"
    )
    assert loaded.source_spec.finevision_geo170k_path == str(
        artifact.path.parent / "finevision-geo170k"
    )


def test_manifest_requires_the_external_raw_sha_pin(tmp_path: Path, monkeypatch):
    artifact = _build_manifest(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="raw SHA mismatch"):
        load_perception_provenance_manifest(artifact.path, expected_sha256="0" * 64)


@pytest.mark.parametrize("marker", [None, "0" * 64 + "\n", "{sha}\nextra\n"])
def test_manifest_requires_exact_complete_marker(tmp_path: Path, monkeypatch, marker: str | None):
    artifact = _build_manifest(tmp_path, monkeypatch)
    complete = artifact.path.parent / "COMPLETE"
    if marker is None:
        complete.unlink()
    else:
        complete.write_text(marker.format(sha=artifact.raw_sha256))

    with pytest.raises(ValueError, match="COMPLETE"):
        load_perception_provenance_manifest(
            artifact.path,
            expected_sha256=artifact.raw_sha256,
        )

    loaded = load_perception_provenance_manifest(
        artifact.path,
        expected_sha256=artifact.raw_sha256,
        require_complete=False,
    )
    assert loaded.raw_sha256 == artifact.raw_sha256


def test_nonzero_rank_manifest_load_skips_the_rank0_signature_payload(tmp_path: Path, monkeypatch):
    artifact = _build_manifest(tmp_path, monkeypatch)
    signature_ref = artifact.value["image_path_signatures"]
    signature_path = artifact.path.parent / signature_ref["path"]
    signature_path.write_text("rank zero already rejected or approved this payload\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_perception_provenance_manifest(
            artifact.path,
            expected_sha256=artifact.raw_sha256,
        )

    loaded = load_perception_provenance_manifest(
        artifact.path,
        expected_sha256=artifact.raw_sha256,
        verify_finevision_materialization=False,
        load_image_path_signatures=False,
    )
    assert loaded.image_path_signatures == ()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value.__setitem__("version", True),
            "provenance version must be an integer",
        ),
        (
            lambda value: value["source_spec"].__setitem__("unknown", "field"),
            "source_spec fields",
        ),
        (
            lambda value: value["source_spec"].__setitem__("ocr_source_names", 7),
            "ocr_source_names must be a string list",
        ),
        (
            lambda value: value["sources"]["ocr_document"].__setitem__("components", ["text_vqa"]),
            "components differ",
        ),
        (
            lambda value: value["builder"].__setitem__("name", "unreviewed_builder"),
            "builder identity differs",
        ),
        (
            lambda value: value["builder"].__setitem__("version", True),
            "builder.version must be an integer",
        ),
        (
            lambda value: value["builder"].__setitem__("script_sha256", "0" * 64),
            "builder bytes differ",
        ),
        (
            lambda value: value.__setitem__("source_registry_version", True),
            "source_registry_version must be an integer",
        ),
        (
            lambda value: value.__setitem__("source_registry_sha256", "0" * 64),
            "source implementation identity differs",
        ),
        (
            lambda value: value["source_implementation_inventory"]["files"].__setitem__(
                "pixmo_points.py", "0" * 64
            ),
            "source implementation identity differs",
        ),
    ],
)
def test_manifest_rejects_schema_component_and_current_code_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[Dict[str, Any]], None],
    match: str,
):
    artifact = _build_manifest(tmp_path, monkeypatch)
    raw_sha = _rewrite_artifact(artifact, mutate)

    with pytest.raises(ValueError, match=match):
        load_perception_provenance_manifest(artifact.path, expected_sha256=raw_sha)


def test_source_spec_requires_explicit_absolute_finevision_artifacts(tmp_path: Path):
    spec = _production_source_spec(tmp_path)
    spec.validate_production_contract()

    with pytest.raises(ValueError, match="absolute finevision_visualweb_path"):
        replace(spec, finevision_visualweb_path=None).validate_production_contract()
    with pytest.raises(ValueError, match="absolute finevision_geo170k_path"):
        replace(spec, finevision_geo170k_path="relative/path").validate_production_contract()


def test_manifest_rejects_unsorted_selection(tmp_path: Path, monkeypatch):
    artifact = _build_manifest(tmp_path, monkeypatch)
    value = deepcopy(artifact.value)
    source_name = "pixmo_caption"
    split = value["sources"][source_name]["train"]
    selection_path = artifact.path.parent / split["selection"]["path"]
    raw = b"1\n0\n"
    selection_path.write_bytes(raw)
    split["selection"]["sha256"] = _sha256_bytes(raw)
    split["selection"]["indices_sha256"] = _canonical_sha256([1, 0])
    split["runtime_dataset_fingerprint"] = selected_dataset_fingerprint(
        source_name=source_name,
        logical_split="train",
        physical_split="train",
        base_fingerprint=split["base_dataset_fingerprint"],
        selection_indices_sha256=split["selection"]["indices_sha256"],
        source_spec_sha256=value["source_spec_sha256"],
    )
    raw_sha = _write_manifest(artifact.path, value)

    with pytest.raises(ValueError, match="sorted unique"):
        load_perception_provenance_manifest(artifact.path, expected_sha256=raw_sha)


@pytest.mark.parametrize(
    ("source_name", "logical_split", "physical_split"),
    [
        ("pixmo_caption", "train", "validation"),
        ("pixmo_caption", "validation", "train"),
        ("audited_alignment", "validation", "validation"),
    ],
)
def test_manifest_rejects_wrong_physical_split_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_name: str,
    logical_split: str,
    physical_split: str,
):
    artifact = _build_manifest(tmp_path, monkeypatch)

    def mutate(value):
        split = value["sources"][source_name][logical_split]
        split["physical_split"] = physical_split
        split["runtime_dataset_fingerprint"] = selected_dataset_fingerprint(
            source_name=source_name,
            logical_split=logical_split,
            physical_split=physical_split,
            base_fingerprint=split["base_dataset_fingerprint"],
            selection_indices_sha256=split["selection"]["indices_sha256"],
            source_spec_sha256=value["source_spec_sha256"],
        )

    raw_sha = _rewrite_artifact(artifact, mutate)
    with pytest.raises(ValueError, match="must use physical split"):
        load_perception_provenance_manifest(artifact.path, expected_sha256=raw_sha)


def test_manifest_rejects_audited_alignment_base_identity_drift(tmp_path: Path, monkeypatch):
    artifact = _build_manifest(tmp_path, monkeypatch)

    def mutate(value):
        split = value["sources"]["audited_alignment"]["validation"]
        split["base_dataset_fingerprint"] = "f" * 16
        split["runtime_dataset_fingerprint"] = selected_dataset_fingerprint(
            source_name="audited_alignment",
            logical_split="validation",
            physical_split="train",
            base_fingerprint=split["base_dataset_fingerprint"],
            selection_indices_sha256=split["selection"]["indices_sha256"],
            source_spec_sha256=value["source_spec_sha256"],
        )

    raw_sha = _rewrite_artifact(artifact, mutate)
    with pytest.raises(ValueError, match="share one physical base dataset identity"):
        load_perception_provenance_manifest(artifact.path, expected_sha256=raw_sha)


def test_manifest_rejects_validation_selection_policy_drift(tmp_path: Path, monkeypatch):
    artifact = _build_manifest(tmp_path, monkeypatch)
    raw_sha = _rewrite_artifact(
        artifact,
        lambda value: value["validation_selection"].__setitem__("image_contents_per_source", 511),
    )

    with pytest.raises(ValueError, match="selection policy differs"):
        load_perception_provenance_manifest(artifact.path, expected_sha256=raw_sha)


def test_manifest_rejects_union_overlap(tmp_path: Path, monkeypatch):
    artifact = _build_manifest(tmp_path, monkeypatch, overlap=True)

    with pytest.raises(ValueError, match="unions overlap"):
        load_perception_provenance_manifest(artifact.path, expected_sha256=artifact.raw_sha256)


def test_manifest_rejects_inventory_tamper_and_path_escape(tmp_path: Path, monkeypatch):
    artifact = _build_manifest(tmp_path, monkeypatch)
    train_union = artifact.value["unions"]["train_unique_image_content"]
    (artifact.path.parent / train_union["path"]).write_text("0" * 64 + "\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_perception_provenance_manifest(artifact.path, expected_sha256=artifact.raw_sha256)

    artifact = _build_manifest(tmp_path / "escape-case", monkeypatch)
    raw_sha = _rewrite_artifact(
        artifact,
        lambda value: value["unions"]["train_unique_image_content"].__setitem__(
            "path", "../outside.txt"
        ),
    )
    with pytest.raises(ValueError, match="escapes the provenance artifact root"):
        load_perception_provenance_manifest(artifact.path, expected_sha256=raw_sha)


def test_image_reference_hash_matches_runtime_bytes_precedence(tmp_path: Path):
    image_path = tmp_path / "image.bin"
    image_path.write_bytes(b"path bytes")
    embedded = b"embedded bytes"

    assert image_reference_sha256({"path": str(image_path), "bytes": embedded}) == (
        _sha256_bytes(embedded)
    )
    assert image_reference_sha256({"path": str(image_path), "bytes": b""}) == (
        _sha256_bytes(b"path bytes")
    )
    assert image_reference_sha256(memoryview(embedded)) == _sha256_bytes(embedded)


class _RawImageDataset:
    def __init__(self, images: Sequence[Path], fingerprint: str, annotation_sha256: str):
        self.images = tuple(images)
        self.content_fingerprint = fingerprint
        self._annotation_sha256 = annotation_sha256

    def __len__(self) -> int:
        return len(self.images)

    def get(self, index: int, epoch: int = 0) -> Dict[str, int]:
        return {"index": index, "epoch": epoch}

    def raw_image_references(self, index: int):
        return (str(self.images[index]),)

    def annotation_content_sha256(self):
        return self._annotation_sha256


def test_selected_dataset_detects_current_image_byte_drift(tmp_path: Path):
    images = [tmp_path / "0.img", tmp_path / "1.img"]
    images[0].write_bytes(b"zero")
    images[1].write_bytes(b"one")
    base_fingerprint = "0123456789abcdef"
    selection_indices_sha = _canonical_sha256([0, 1])
    selection = PerceptionSplitSelection(
        physical_split="train",
        base_annotation_sha256="2" * 64,
        base_dataset_fingerprint=base_fingerprint,
        base_examples=2,
        indices=(0, 1),
        selection_indices_sha256=selection_indices_sha,
        runtime_dataset_fingerprint=selected_dataset_fingerprint(
            source_name="pixmo_caption",
            logical_split="train",
            physical_split="train",
            base_fingerprint=base_fingerprint,
            selection_indices_sha256=selection_indices_sha,
            source_spec_sha256="1" * 64,
        ),
        row_image_content_sha256=(_sha256_bytes(b"zero"), _sha256_bytes(b"one")),
        unique_image_content_sha256=tuple(sorted((_sha256_bytes(b"zero"), _sha256_bytes(b"one")))),
    )
    selected = SelectedVisionAlignmentDataset(
        _RawImageDataset(images, base_fingerprint, "2" * 64),
        source_name="pixmo_caption",
        logical_split="train",
        selection=selection,
    )

    assert len(selected.validate_image_content()) == 64
    assert selected.get(1, 7) == {"index": 1, "epoch": 7}
    images[1].write_bytes(b"mutated")

    with pytest.raises(ValueError, match="image bytes differ"):
        selected.validate_image_content()


def _load_exporter():
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "data"
        / ("export_vision_alignment_perception_probe.py")
    )
    spec = importlib.util.spec_from_file_location(
        "_vision_alignment_perception_exporter_adversarial_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exporter_requires_exact_manifest_sha_argument(tmp_path: Path):
    exporter = _load_exporter()

    with pytest.raises(SystemExit) as error:
        exporter._parser().parse_args(
            [
                "--image-provenance-manifest",
                str(tmp_path / "manifest.json"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )

    assert error.value.code == 2


def test_exporter_failure_leaves_no_final_or_partial_output(tmp_path: Path, monkeypatch):
    exporter = _load_exporter()
    source_spec = _production_source_spec(tmp_path)
    expected_manifest_sha = "a" * 64
    fake_manifest = SimpleNamespace(
        path=tmp_path / "manifest.json",
        raw_sha256=expected_manifest_sha,
        content_sha256="b" * 64,
        source_spec=source_spec,
        source_spec_sha256=source_spec.preprocessing_sha256,
        selection=lambda _name, _split: SimpleNamespace(physical_split="train", base_examples=1),
    )
    observed: Dict[str, Any] = {}

    def load_manifest(path, *, expected_sha256):
        observed["manifest_path"] = path
        observed["expected_sha256"] = expected_sha256
        return fake_manifest

    class _Mixture:
        def __init__(self, phase: str):
            assert phase == "perception"

        def resolved_targets(self):
            return {name: 1.0 for name in PERCEPTION_SOURCE_NAMES}

    calls: List[str] = []

    def fail_during_export(
        dataset,
        *,
        source_name,
        output_path,
        num_examples,
        seed,
        epochs,
        unbounded_dataset,
        max_sequence_length,
    ):
        del dataset, num_examples, seed, epochs, unbounded_dataset, max_sequence_length
        calls.append(source_name)
        output_path.write_bytes(b"staged partial bytes\n")
        if len(calls) == 2:
            raise ValueError("injected source failure")
        return {"name": source_name}

    monkeypatch.setattr(exporter, "load_perception_provenance_manifest", load_manifest)
    monkeypatch.setattr(exporter, "VisionAlignmentMixtureConfig", _Mixture)
    monkeypatch.setattr(
        exporter, "load_pinned_vision_alignment_tokenizer", lambda **_: (object(), object())
    )

    class _OneRowDataset:
        def __len__(self):
            return 1

    monkeypatch.setattr(
        exporter,
        "build_selected_perception_dataset",
        lambda *_, **__: SimpleNamespace(_dataset=_OneRowDataset()),
    )

    monkeypatch.setattr(
        exporter,
        "build_vision_alignment_perception_dataset",
        lambda *_, **__: _OneRowDataset(),
    )
    monkeypatch.setattr(exporter, "export_source_probe", fail_during_export)
    output_dir = tmp_path / "perception-probes"

    with pytest.raises(SystemExit) as error:
        exporter.main(
            [
                "--image-provenance-manifest",
                str(fake_manifest.path),
                "--expected-image-provenance-sha256",
                expected_manifest_sha,
                "--output-dir",
                str(output_dir),
            ]
        )

    assert error.value.code == 2
    assert observed["expected_sha256"] == expected_manifest_sha
    assert not output_dir.exists()
    assert not list(tmp_path.glob(f".{output_dir.name}.*.building"))
