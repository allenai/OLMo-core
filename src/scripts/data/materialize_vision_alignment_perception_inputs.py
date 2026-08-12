#!/usr/bin/env python
"""Materialize the immutable input bundle for perception image provenance.

The bundle binds a pinned source-spec template to one strictly validated FineVision
materialization, captures the current transitive perception implementation, and records the
exact downstream artifact plan. Publication is deterministic, fail-closed, and atomic: the
repository must be clean on the reviewed branch, every input is raw-SHA-pinned, code identity is
rechecked immediately before publication, and an existing output is never replaced.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from olmo_core.data.multimodal.dataset_compat import load_from_disk_compat
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_PROVENANCE_MANIFEST,
)
from olmo_core.data.multimodal.vision_alignment_perception_sources import (
    VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
    VisionAlignmentPerceptionSourceSpec,
    vision_alignment_perception_implementation_inventory,
    vision_alignment_perception_source_registry_sha256,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    VISION_ALIGNMENT_PIXMO_ROW_PATH_INVENTORY_ALGORITHM,
    pixmo_row_path_inventory,
    runtime_dataset_fingerprint,
)
from scripts.data.build_vision_alignment_perception_provenance import (
    FineVisionMaterialization,
    _bind_finevision_materialization,
    _canonical_bytes,
    _publish_staging,
    _strict_json_object,
    _validate_finevision_materialization,
)

BUNDLE_FORMAT = "vision_alignment_perception_provenance_inputs"
BUNDLE_VERSION = 2
BUNDLE_STATUS = "verified"
DEFAULT_EXPECTED_BRANCH = "vision-moe"
DEFAULT_HF_CACHE_DIR = "/weka/oe-training-default/rustin/hf-cache/hub"
SOURCE_SPEC_NAME = "source-spec.json"
IMPLEMENTATION_INVENTORY_NAME = "implementation-inventory.json"
PINS_NAME = "pins.json"
COMPLETE_NAME = "COMPLETE"
SOURCE_CATALOG_NAME = "vision-alignment-perception-source-catalog.json"
INPUT_OUTPUT_NAME = "perception-provenance-inputs-v2"
PROVENANCE_OUTPUT_NAME = "perception-provenance-v2"
PROBE_OUTPUT_NAME = "perception-probe-v2"
AUDIT_OUTPUT_NAME = "perception-source-audit-v2.json"
PIXMO_CAP_VALIDATION_FORMAT = "vision_alignment_validation_manifest"
PIXMO_CAP_VALIDATION_VERSION = 3
PIXMO_CAP_VALIDATION_NAME = "vision-alignment-validation-manifest.json"
PIXMO_CAP_BUILDER_PATH = "src/scripts/data/build_vision_alignment_pixmo_cap.py"
PIXMO_CAP_BUILDER_FORMAT = "vision_alignment_pixmo_cap_builder"
PIXMO_CAP_FILTER_ALGORITHM = "preserve-validation-drop-train-content-overlap-v1"
PIXMO_CAP_ROW_CONTENT_ALGORITHM = "sha256-lines-v1"
INPUT_MATERIALIZER_PATH = "src/scripts/data/materialize_vision_alignment_perception_inputs.py"
PIPELINE_SCRIPT_PATHS = {
    "finevision_materializer": "src/scripts/data/materialize_vision_alignment_finevision.py",
    "mix_auditor": "src/scripts/data/audit_vision_alignment_perception_mix.py",
    "pixmo_cap_builder": PIXMO_CAP_BUILDER_PATH,
    "probe_exporter": "src/scripts/data/export_vision_alignment_perception_probe.py",
    "provenance_builder": "src/scripts/data/build_vision_alignment_perception_provenance.py",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DATASET_FINGERPRINT_RE = re.compile(r"[0-9a-f]{16,64}")
_PIXMO_SPLITS = frozenset({"train", "validation"})
_PIXMO_ROOT_FIELDS = {
    "format",
    "version",
    "builder",
    "source",
    "output",
    "inventories",
    "filtering",
}
_PIXMO_BUILDER_FIELDS = {
    "format",
    "version",
    "script",
    "script_sha256",
    "filter_algorithm",
    "image_hash_algorithm",
    "row_image_paths_algorithm",
    "row_image_content_algorithm",
}
_PIXMO_OUTPUT_FIELDS = {"dataset_path", "splits"}
_PIXMO_OUTPUT_SPLIT_FIELDS = {
    "dataset_fingerprint",
    "examples",
    "row_image_content_path",
    "row_image_content_sha256",
    "row_image_paths_sha256",
    "unique_image_content",
    "unique_image_paths",
}


@dataclass(frozen=True)
class GitIdentity:
    """Clean repository branch and commit identity."""

    branch: str
    commit: str


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _lower_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _lower_commit(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{name} must be a lowercase 40-character Git commit")
    return value


def _git(repository_root: Path, *arguments: str, allow_failure: bool = False) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode and not allow_failure:
        detail = process.stderr.strip() or process.stdout.strip() or "git command failed"
        raise ValueError(detail)
    return process.stdout.strip()


def _git_identity(repository_root: Path, expected_commit: str) -> GitIdentity:
    repository_root = repository_root.resolve()
    top_level = Path(_git(repository_root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != repository_root:
        raise ValueError(
            f"Input materializer must run from repository root {repository_root}, got {top_level}"
        )
    branch = _git(
        repository_root,
        "symbolic-ref",
        "--quiet",
        "--short",
        "HEAD",
        allow_failure=True,
    )
    if branch != DEFAULT_EXPECTED_BRANCH:
        raise ValueError(
            f"Input materializer requires branch {DEFAULT_EXPECTED_BRANCH!r}, got {branch!r}"
        )
    commit = _git(repository_root, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Repository HEAD is not an exact hexadecimal commit")
    if commit != expected_commit:
        raise ValueError(f"Input materializer requires commit {expected_commit}, got {commit}")
    dirty = _git(repository_root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ValueError("Input materializer requires a clean repository worktree")
    return GitIdentity(branch=branch, commit=commit)


def _pipeline_script_hashes(repository_root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, relative_path in sorted(PIPELINE_SCRIPT_PATHS.items()):
        path = repository_root / relative_path
        if not path.is_file():
            raise ValueError(f"Required pipeline script is absent: {path}")
        result[name] = {"path": relative_path, "sha256": _sha256_file(path)}
    return result


def _input_materializer_identity(repository_root: Path) -> dict[str, str]:
    path = repository_root / INPUT_MATERIALIZER_PATH
    if path.resolve() != Path(__file__).resolve() or not path.is_file():
        raise ValueError("Input materializer is not running from its canonical repository path")
    return {"path": INPUT_MATERIALIZER_PATH, "sha256": _sha256_file(path)}


def _read_pinned_file(path: Path, expected_sha256: str, *, name: str) -> bytes:
    if not path.is_file():
        raise ValueError(f"{name} is not a file: {path}")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Could not read {name}: {path}") from error
    actual = hashlib.sha256(raw).hexdigest()
    if actual != _lower_sha256(expected_sha256, name=f"expected {name} SHA-256"):
        raise ValueError(f"{name} SHA-256 differs: expected {expected_sha256}, got {actual}")
    return raw


def _validate_pinned_file(path: Path, expected_sha256: str, *, name: str) -> str:
    raw = _read_pinned_file(path, expected_sha256, name=name)
    return hashlib.sha256(raw).hexdigest()


def _parse_strict_json(raw: bytes, *, name: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid {name} JSON") from error


def _parse_source_spec(raw: bytes) -> VisionAlignmentPerceptionSourceSpec:
    value = _parse_strict_json(raw, name="source-spec template")
    if not isinstance(value, Mapping):
        raise TypeError("Perception source-spec template must be a JSON object")
    mapping = dict(value)
    registry_version = mapping.pop("source_registry_version", None)
    expected_fields = {field.name for field in fields(VisionAlignmentPerceptionSourceSpec)}
    if (
        registry_version != VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION
        or set(mapping) != expected_fields
    ):
        raise ValueError("Perception source-spec template fields or registry version differ")
    if not isinstance(mapping.get("ocr_source_names"), list):
        raise TypeError("Perception source-spec template ocr_source_names must be a JSON list")
    mapping["ocr_source_names"] = tuple(mapping["ocr_source_names"])
    source_spec = VisionAlignmentPerceptionSourceSpec(**mapping)
    if source_spec.as_canonical_dict() != value:
        raise ValueError("Perception source-spec template is not its canonical representation")
    return source_spec


def _exact_mapping(value: Any, expected_fields: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected_fields - actual)}, "
            f"extra={sorted(actual - expected_fields)}"
        )
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_live_pixmo_cap(
    dataset_path: Path,
    *,
    builder: Mapping[str, Any],
    output_splits: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    try:
        live = load_from_disk_compat(dataset_path)
    except Exception as error:
        raise ValueError(
            f"Could not load canonical PixMoCap output {dataset_path}: {error}"
        ) from error
    if not hasattr(live, "keys") or set(live.keys()) != _PIXMO_SPLITS:
        raise ValueError("Canonical PixMoCap output must contain exactly train/validation")
    validated: dict[str, dict[str, Any]] = {}
    for split in sorted(_PIXMO_SPLITS):
        expected = _exact_mapping(
            output_splits[split],
            _PIXMO_OUTPUT_SPLIT_FIELDS,
            name=f"PixMoCap output.splits.{split}",
        )
        fingerprint = expected["dataset_fingerprint"]
        if (
            not isinstance(fingerprint, str)
            or _DATASET_FINGERPRINT_RE.fullmatch(fingerprint) is None
        ):
            raise ValueError(f"PixMoCap output {split} fingerprint is invalid")
        examples = _positive_integer(expected["examples"], name=f"PixMoCap {split} examples")
        unique_paths = _positive_integer(
            expected["unique_image_paths"],
            name=f"PixMoCap {split} unique_image_paths",
        )
        row_paths_sha256 = _lower_sha256(
            expected["row_image_paths_sha256"],
            name=f"PixMoCap {split} row_image_paths_sha256",
        )
        unique_content = _positive_integer(
            expected["unique_image_content"],
            name=f"PixMoCap {split} unique_image_content",
        )
        _lower_sha256(
            expected["row_image_content_sha256"],
            name=f"PixMoCap {split} row_image_content_sha256",
        )
        row_content_path = expected["row_image_content_path"]
        if (
            not isinstance(row_content_path, str)
            or not row_content_path
            or Path(row_content_path).is_absolute()
            or ".." in Path(row_content_path).parts
        ):
            raise ValueError(f"PixMoCap output {split} row-content path is invalid")
        if unique_paths > examples or unique_content > examples:
            raise ValueError(f"PixMoCap output {split} unique counts exceed examples")
        live_split = live[split]
        inventory = pixmo_row_path_inventory(live_split)
        if (
            runtime_dataset_fingerprint(live_split) != fingerprint
            or len(live_split) != examples
            or inventory.get("algorithm") != builder["row_image_paths_algorithm"]
            or inventory.get("rows") != examples
            or inventory.get("unique_paths") != unique_paths
            or inventory.get("sha256") != row_paths_sha256
        ):
            raise ValueError(f"Live canonical PixMoCap {split} split differs from its manifest")
        validated[split] = {
            "dataset_fingerprint": fingerprint,
            "examples": examples,
            "row_image_paths_sha256": row_paths_sha256,
            "unique_image_paths": unique_paths,
        }
    return validated


def _validate_pixmo_cap_manifest(
    manifest_path: Path,
    expected_sha256: str,
    *,
    expected_dataset_path: Path,
    expected_builder_sha256: str,
) -> Mapping[str, Any]:
    if manifest_path.name != PIXMO_CAP_VALIDATION_NAME:
        raise ValueError(
            f"PixMoCap validation manifest must use canonical name {PIXMO_CAP_VALIDATION_NAME!r}"
        )
    raw = _read_pinned_file(
        manifest_path,
        expected_sha256,
        name="PixMoCap validation manifest",
    )
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    manifest = _parse_strict_json(raw, name="PixMoCap validation manifest")
    manifest = _exact_mapping(
        manifest,
        _PIXMO_ROOT_FIELDS,
        name="PixMoCap validation manifest",
    )
    if (
        manifest.get("format") != PIXMO_CAP_VALIDATION_FORMAT
        or manifest.get("version") != PIXMO_CAP_VALIDATION_VERSION
    ):
        raise ValueError("PixMoCap validation manifest identity differs")
    builder = _exact_mapping(
        manifest["builder"],
        _PIXMO_BUILDER_FIELDS,
        name="PixMoCap validation manifest builder",
    )
    output = _exact_mapping(
        manifest["output"],
        _PIXMO_OUTPUT_FIELDS,
        name="PixMoCap validation manifest output",
    )
    if (
        builder["format"] != PIXMO_CAP_BUILDER_FORMAT
        or builder["version"] != 1
        or builder["script"] != PIXMO_CAP_BUILDER_PATH
        or builder["script_sha256"] != expected_builder_sha256
        or builder["filter_algorithm"] != PIXMO_CAP_FILTER_ALGORITHM
        or builder["image_hash_algorithm"] != "sha256"
        or builder["row_image_paths_algorithm"]
        != VISION_ALIGNMENT_PIXMO_ROW_PATH_INVENTORY_ALGORITHM
        or builder["row_image_content_algorithm"] != PIXMO_CAP_ROW_CONTENT_ALGORITHM
    ):
        raise ValueError("PixMoCap validation manifest builder identity differs")
    dataset_value = output["dataset_path"]
    if dataset_value != "dataset":
        raise ValueError("PixMoCap output dataset path must be relative to its artifact")
    dataset_path = (manifest_path.parent / dataset_value).resolve()
    if dataset_path != expected_dataset_path.resolve() or not dataset_path.is_dir():
        raise ValueError("PixMoCap validation manifest does not bind the source-spec dataset")
    output_splits = output["splits"]
    if not isinstance(output_splits, Mapping) or set(output_splits) != _PIXMO_SPLITS:
        raise ValueError("PixMoCap output splits must be exactly train/validation")
    complete = manifest_path.parent / COMPLETE_NAME
    if not complete.is_file() or complete.read_bytes() != (raw_sha256 + "\n").encode("ascii"):
        raise ValueError("PixMoCap validation manifest COMPLETE receipt differs")
    live_splits = _validate_live_pixmo_cap(
        dataset_path,
        builder=builder,
        output_splits=output_splits,
    )
    return {
        "manifest_path": str(manifest_path),
        "sha256": raw_sha256,
        "dataset_path": str(dataset_path),
        "builder_sha256": expected_builder_sha256,
        "output_splits": live_splits,
    }


def _write_exact(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _provenance_builder_invocation(
    *,
    repository_root: Path,
    output_dir: Path,
    provenance_output_dir: Path,
    source_spec_sha256: str,
    source_registry_sha256: str,
    implementation_inventory_sha256: str,
    finevision_manifest_path: Path,
    finevision_manifest_sha256: str,
    hf_cache_dir: Path,
) -> dict[str, Any]:
    return {
        "cwd": str(repository_root),
        "environment": {"PYTHONPATH": "src"},
        "argv": [
            "python",
            "src/scripts/data/build_vision_alignment_perception_provenance.py",
            f"--source-spec={output_dir / SOURCE_SPEC_NAME}",
            f"--expected-source-spec-sha256={source_spec_sha256}",
            f"--expected-source-registry-sha256={source_registry_sha256}",
            f"--implementation-inventory={output_dir / IMPLEMENTATION_INVENTORY_NAME}",
            f"--expected-implementation-inventory-sha256={implementation_inventory_sha256}",
            f"--finevision-materialization-manifest={finevision_manifest_path}",
            f"--expected-finevision-materialization-sha256={finevision_manifest_sha256}",
            f"--output-dir={provenance_output_dir}",
            f"--hf-cache-dir={hf_cache_dir}",
        ],
    }


def _assert_unchanged_before_publish(
    *,
    repository_root: Path,
    expected_repository_commit: str,
    git_identity: GitIdentity,
    builder_identity: Mapping[str, str],
    pipeline_scripts: Mapping[str, Mapping[str, str]],
    implementation_inventory: Mapping[str, Any],
    source_registry_sha256: str,
    pinned_files: Sequence[tuple[Path, str, str]],
    finevision_reference: FineVisionMaterialization,
    pixmo_manifest_path: Path,
    expected_pixmo_manifest_sha256: str,
    pixmo_dataset_path: Path,
    pixmo_builder_sha256: str,
    pixmo_reference: Mapping[str, Any],
) -> None:
    if _input_materializer_identity(repository_root) != builder_identity:
        raise ValueError("Input materializer changed during construction")
    if _pipeline_script_hashes(repository_root) != pipeline_scripts:
        raise ValueError("Perception pipeline scripts changed during construction")
    if vision_alignment_perception_implementation_inventory() != implementation_inventory:
        raise ValueError("Perception implementation inventory changed during construction")
    if vision_alignment_perception_source_registry_sha256() != source_registry_sha256:
        raise ValueError("Perception source registry changed during construction")
    if (
        _validate_finevision_materialization(
            finevision_reference.manifest_path,
            finevision_reference.raw_sha256,
        )
        != finevision_reference
    ):
        raise ValueError("FineVision materialization changed during construction")
    if (
        _validate_pixmo_cap_manifest(
            pixmo_manifest_path,
            expected_pixmo_manifest_sha256,
            expected_dataset_path=pixmo_dataset_path,
            expected_builder_sha256=pixmo_builder_sha256,
        )
        != pixmo_reference
    ):
        raise ValueError("PixMoCap validation identity changed during construction")
    for path, expected_sha256, name in pinned_files:
        _validate_pinned_file(path, expected_sha256, name=name)
    _validate_pinned_file(
        pixmo_manifest_path,
        expected_pixmo_manifest_sha256,
        name="PixMoCap validation manifest",
    )
    if _git_identity(repository_root, expected_repository_commit) != git_identity:
        raise ValueError("Repository identity changed during input materialization")


def materialize_perception_inputs(
    *,
    source_spec_template: str | Path,
    expected_source_spec_template_sha256: str,
    finevision_materialization_manifest: str | Path,
    expected_finevision_materialization_sha256: str,
    pixmo_cap_validation_manifest: str | Path,
    expected_pixmo_cap_validation_manifest_sha256: str,
    output_dir: str | Path,
    expected_repository_commit: str,
    hf_cache_dir: str | Path = DEFAULT_HF_CACHE_DIR,
) -> Path:
    """Validate, construct, and atomically publish a perception input bundle.

    :returns: The published ``pins.json`` path.
    :raises ValueError: If repository, input, source, or code identity differs.
    :raises FileExistsError: If the immutable output already exists.
    """
    repository_root = _repository_root()
    source_spec_template = Path(source_spec_template).expanduser().resolve()
    finevision_manifest_path = Path(finevision_materialization_manifest).expanduser().resolve()
    pixmo_manifest_path = Path(pixmo_cap_validation_manifest).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.name != INPUT_OUTPUT_NAME:
        raise ValueError(f"Input bundle output must use canonical name {INPUT_OUTPUT_NAME!r}")
    provenance_output_dir = output_dir.parent / PROVENANCE_OUTPUT_NAME
    probe_output_dir = output_dir.parent / PROBE_OUTPUT_NAME
    audit_output = output_dir.parent / AUDIT_OUTPUT_NAME
    hf_cache_dir = Path(hf_cache_dir).expanduser().resolve()
    expected_repository_commit = _lower_commit(
        expected_repository_commit,
        name="expected repository commit",
    )
    artifact_paths = (output_dir, provenance_output_dir, probe_output_dir, audit_output)
    if any(
        _paths_overlap(first, second)
        for index, first in enumerate(artifact_paths)
        for second in artifact_paths[index + 1 :]
    ):
        raise ValueError("Bundle and downstream artifact paths must be disjoint")

    expected_source_spec_template_sha256 = _lower_sha256(
        expected_source_spec_template_sha256,
        name="expected source-spec template SHA-256",
    )
    expected_finevision_materialization_sha256 = _lower_sha256(
        expected_finevision_materialization_sha256,
        name="expected FineVision materialization SHA-256",
    )
    expected_pixmo_cap_validation_manifest_sha256 = _lower_sha256(
        expected_pixmo_cap_validation_manifest_sha256,
        name="expected PixMoCap validation manifest SHA-256",
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact {output_dir}")
    lock_path = output_dir.with_name(f".{output_dir.name}.lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another perception input materializer holds {lock_path}"
            ) from error
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite immutable artifact {output_dir}")

        git_identity = _git_identity(
            repository_root,
            expected_repository_commit,
        )
        builder_identity = _input_materializer_identity(repository_root)
        pipeline_scripts = _pipeline_script_hashes(repository_root)

        source_spec_template_raw = _read_pinned_file(
            source_spec_template,
            expected_source_spec_template_sha256,
            name="source-spec template",
        )
        source_spec = _parse_source_spec(source_spec_template_raw)
        materialization: FineVisionMaterialization = _validate_finevision_materialization(
            finevision_manifest_path,
            expected_finevision_materialization_sha256,
        )
        source_spec = _bind_finevision_materialization(source_spec, materialization)
        source_spec.validate_production_contract()
        source_spec_raw = _canonical_bytes(source_spec.as_canonical_dict())
        source_spec_sha256 = hashlib.sha256(source_spec_raw).hexdigest()
        if source_spec_sha256 != source_spec.preprocessing_sha256:
            raise ValueError("Bound source-spec canonical identity differs")

        pixmo_reference = _validate_pixmo_cap_manifest(
            pixmo_manifest_path,
            expected_pixmo_cap_validation_manifest_sha256,
            expected_dataset_path=Path(source_spec.pixmo_cap_path),
            expected_builder_sha256=pipeline_scripts["pixmo_cap_builder"]["sha256"],
        )
        implementation_inventory = vision_alignment_perception_implementation_inventory()
        implementation_inventory_raw = _canonical_bytes(implementation_inventory)
        implementation_inventory_sha256 = hashlib.sha256(implementation_inventory_raw).hexdigest()
        source_registry_sha256 = vision_alignment_perception_source_registry_sha256()
        if implementation_inventory_sha256 != source_registry_sha256:
            raise ValueError("Perception implementation inventory and registry identities differ")

        planned_outputs = {
            "provenance_dir": str(provenance_output_dir),
            "provenance_manifest": str(provenance_output_dir / PERCEPTION_PROVENANCE_MANIFEST),
            "probe_dir": str(probe_output_dir),
            "source_catalog": str(probe_output_dir / SOURCE_CATALOG_NAME),
            "source_audit": str(audit_output),
        }
        pins = {
            "format": BUNDLE_FORMAT,
            "version": BUNDLE_VERSION,
            "status": BUNDLE_STATUS,
            "builder": dict(builder_identity),
            "repository": {
                "branch": git_identity.branch,
                "commit": git_identity.commit,
            },
            "source_spec_template": {
                "path": str(source_spec_template),
                "sha256": expected_source_spec_template_sha256,
            },
            "source_spec": {"path": SOURCE_SPEC_NAME, "sha256": source_spec_sha256},
            "source_registry_version": VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
            "source_registry_sha256": source_registry_sha256,
            "implementation_inventory": {
                "path": IMPLEMENTATION_INVENTORY_NAME,
                "sha256": implementation_inventory_sha256,
            },
            "finevision_materialization": {
                "manifest_path": str(materialization.manifest_path),
                "sha256": materialization.raw_sha256,
                "content_sha256": materialization.content_sha256,
                "visualweb_path": str(materialization.visualweb_path),
                "visualweb_fingerprint": materialization.visualweb_fingerprint,
                "geo170k_path": str(materialization.geo170k_path),
                "geo170k_fingerprint": materialization.geo170k_fingerprint,
            },
            "pixmo_cap": dict(pixmo_reference),
            "scripts": pipeline_scripts,
            "planned_outputs": planned_outputs,
            "provenance_builder": _provenance_builder_invocation(
                repository_root=repository_root,
                output_dir=output_dir,
                provenance_output_dir=provenance_output_dir,
                source_spec_sha256=source_spec_sha256,
                source_registry_sha256=source_registry_sha256,
                implementation_inventory_sha256=implementation_inventory_sha256,
                finevision_manifest_path=materialization.manifest_path,
                finevision_manifest_sha256=materialization.raw_sha256,
                hf_cache_dir=hf_cache_dir,
            ),
        }
        pins_raw = _canonical_bytes(pins)
        pins_sha256 = hashlib.sha256(pins_raw).hexdigest()

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.",
                suffix=".building",
                dir=output_dir.parent,
            )
        )
        try:
            _write_exact(staging / SOURCE_SPEC_NAME, source_spec_raw)
            _write_exact(
                staging / IMPLEMENTATION_INVENTORY_NAME,
                implementation_inventory_raw,
            )
            _write_exact(staging / PINS_NAME, pins_raw)
            _write_exact(staging / COMPLETE_NAME, (pins_sha256 + "\n").encode("ascii"))
            _fsync_tree(staging)
            _assert_unchanged_before_publish(
                repository_root=repository_root,
                expected_repository_commit=expected_repository_commit,
                git_identity=git_identity,
                builder_identity=builder_identity,
                pipeline_scripts=pipeline_scripts,
                implementation_inventory=implementation_inventory,
                source_registry_sha256=source_registry_sha256,
                pinned_files=(
                    (
                        source_spec_template,
                        expected_source_spec_template_sha256,
                        "source-spec template",
                    ),
                    (
                        finevision_manifest_path,
                        expected_finevision_materialization_sha256,
                        "FineVision materialization manifest",
                    ),
                ),
                finevision_reference=materialization,
                pixmo_manifest_path=pixmo_manifest_path,
                expected_pixmo_manifest_sha256=(expected_pixmo_cap_validation_manifest_sha256),
                pixmo_dataset_path=Path(source_spec.pixmo_cap_path),
                pixmo_builder_sha256=pipeline_scripts["pixmo_cap_builder"]["sha256"],
                pixmo_reference=pixmo_reference,
            )
            _publish_staging(staging, output_dir)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return output_dir / PINS_NAME


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spec-template", required=True)
    parser.add_argument("--expected-source-spec-template-sha256", required=True)
    parser.add_argument("--finevision-materialization-manifest", required=True)
    parser.add_argument("--expected-finevision-materialization-sha256", required=True)
    parser.add_argument("--pixmo-cap-validation-manifest", required=True)
    parser.add_argument("--expected-pixmo-cap-validation-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-repository-commit", required=True)
    parser.add_argument("--hf-cache-dir", default=DEFAULT_HF_CACHE_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic input-bundle materializer CLI."""
    args = _parser().parse_args(argv)
    pins_path = materialize_perception_inputs(
        source_spec_template=args.source_spec_template,
        expected_source_spec_template_sha256=args.expected_source_spec_template_sha256,
        finevision_materialization_manifest=args.finevision_materialization_manifest,
        expected_finevision_materialization_sha256=(
            args.expected_finevision_materialization_sha256
        ),
        pixmo_cap_validation_manifest=args.pixmo_cap_validation_manifest,
        expected_pixmo_cap_validation_manifest_sha256=(
            args.expected_pixmo_cap_validation_manifest_sha256
        ),
        output_dir=args.output_dir,
        expected_repository_commit=args.expected_repository_commit,
        hf_cache_dir=args.hf_cache_dir,
    )
    print(
        json.dumps(
            {
                "pins": str(pins_path),
                "pins_sha256": _sha256_file(pins_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
