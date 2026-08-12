from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from olmo_core.data.multimodal.dataset_compat import load_from_disk_compat
from olmo_core.data.multimodal.finevision import FINEVISION_ROOT
from olmo_core.data.multimodal.vision_alignment_perception_sources import (
    VisionAlignmentPerceptionSourceSpec,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    pixmo_row_path_inventory,
    runtime_dataset_fingerprint,
)


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "data"
        / "materialize_vision_alignment_perception_inputs.py"
    )
    spec = importlib.util.spec_from_file_location("perception_input_materializer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> str:
    path.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    return _sha256(path)


def _write_source_template(module, path: Path, pixmo_dataset: Path) -> None:
    source_spec = VisionAlignmentPerceptionSourceSpec(
        phase="perception",
        pixmo_cap_path=str(pixmo_dataset),
        sequence_length=2560,
        max_crops=8,
        message_format="document",
        loss_token_weighting="root_subsegments_root_tokens",
        caption_prompt="Description:",
        transcript_prompt="Transcript:",
        require_transcript=True,
    )
    path.write_bytes(module._canonical_bytes(source_spec.as_canonical_dict()))


def _write_pixmo_artifact(module, root: Path) -> tuple[Path, str]:
    dataset_path = root / "dataset"
    DatasetDict(
        {
            "train": Dataset.from_dict({"image": ["/images/a", "/images/b", "/images/a"]}),
            "validation": Dataset.from_dict({"image": ["/images/c", "/images/d"]}),
        }
    ).save_to_disk(dataset_path)
    live = load_from_disk_compat(dataset_path)
    output_splits = {}
    for split in ("train", "validation"):
        inventory = pixmo_row_path_inventory(live[split])
        output_splits[split] = {
            "dataset_fingerprint": runtime_dataset_fingerprint(live[split]),
            "examples": len(live[split]),
            "row_image_content_path": f"{split}-row-images.sha256",
            "row_image_content_sha256": ("1" if split == "train" else "2") * 64,
            "row_image_paths_sha256": inventory["sha256"],
            "unique_image_content": inventory["unique_paths"],
            "unique_image_paths": inventory["unique_paths"],
        }
    scripts = module._pipeline_script_hashes(module._repository_root())
    manifest = {
        "format": module.PIXMO_CAP_VALIDATION_FORMAT,
        "version": module.PIXMO_CAP_VALIDATION_VERSION,
        "builder": {
            "format": module.PIXMO_CAP_BUILDER_FORMAT,
            "version": 1,
            "script": module.PIXMO_CAP_BUILDER_PATH,
            "script_sha256": scripts["pixmo_cap_builder"]["sha256"],
            "filter_algorithm": module.PIXMO_CAP_FILTER_ALGORITHM,
            "image_hash_algorithm": "sha256",
            "row_image_paths_algorithm": module.VISION_ALIGNMENT_PIXMO_ROW_PATH_INVENTORY_ALGORITHM,
            "row_image_content_algorithm": module.PIXMO_CAP_ROW_CONTENT_ALGORITHM,
        },
        "source": {},
        "output": {"dataset_path": "dataset", "splits": output_splits},
        "inventories": {},
        "filtering": {},
    }
    manifest_path = root / module.PIXMO_CAP_VALIDATION_NAME
    manifest_sha256 = _write_json(manifest_path, manifest)
    (root / "COMPLETE").write_bytes((manifest_sha256 + "\n").encode("ascii"))
    return manifest_path, manifest_sha256


def _finevision_materialization(module, manifest: Path, tmp_path: Path):
    return module.FineVisionMaterialization(
        manifest_path=manifest,
        raw_sha256=_sha256(manifest),
        content_sha256="b" * 64,
        source_root=Path(FINEVISION_ROOT),
        visualweb_path=tmp_path / "finevision" / "visualwebinstruct-filtered",
        geo170k_path=tmp_path / "finevision" / "geo170k-align",
        visualweb_fingerprint="c" * 64,
        geo170k_fingerprint="d" * 64,
    )


def test_materialize_perception_inputs_is_pinned_deterministic_and_immutable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_module()
    pixmo_manifest, pixmo_sha256 = _write_pixmo_artifact(module, tmp_path / "pixmo")
    pixmo_dataset = pixmo_manifest.parent / "dataset"
    source_template = tmp_path / "source-template.json"
    _write_source_template(module, source_template, pixmo_dataset)
    finevision_manifest = tmp_path / "finevision.json"
    finevision_manifest.write_bytes(b'{"finevision":1}')
    materialization = _finevision_materialization(module, finevision_manifest, tmp_path)
    git_identity = module.GitIdentity(branch="vision-moe", commit="a" * 40)
    finevision_validation_calls = []

    def validate_finevision(*args):
        finevision_validation_calls.append(args)
        return materialization

    monkeypatch.setattr(module, "_git_identity", lambda *_args: git_identity)
    monkeypatch.setattr(
        module,
        "_validate_finevision_materialization",
        validate_finevision,
    )

    output = tmp_path / module.INPUT_OUTPUT_NAME
    provenance = tmp_path / module.PROVENANCE_OUTPUT_NAME
    probes = tmp_path / module.PROBE_OUTPUT_NAME
    audit = tmp_path / module.AUDIT_OUTPUT_NAME
    pins_path = module.materialize_perception_inputs(
        source_spec_template=source_template,
        expected_source_spec_template_sha256=_sha256(source_template),
        finevision_materialization_manifest=finevision_manifest,
        expected_finevision_materialization_sha256=_sha256(finevision_manifest),
        pixmo_cap_validation_manifest=pixmo_manifest,
        expected_pixmo_cap_validation_manifest_sha256=pixmo_sha256,
        output_dir=output,
        expected_repository_commit="a" * 40,
    )

    assert {path.name for path in output.iterdir()} == {
        "COMPLETE",
        "implementation-inventory.json",
        "pins.json",
        "source-spec.json",
    }
    pins_raw = pins_path.read_bytes()
    pins = json.loads(pins_raw)
    assert pins["format"] == module.BUNDLE_FORMAT
    assert pins["version"] == 2
    assert pins["status"] == "verified"
    assert pins["repository"] == {"branch": "vision-moe", "commit": "a" * 40}
    assert (
        pins["source_registry_version"]
        == module.VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION
    )
    assert pins["builder"] == {
        "path": module.INPUT_MATERIALIZER_PATH,
        "sha256": _sha256(module._repository_root() / module.INPUT_MATERIALIZER_PATH),
    }
    assert pins["pixmo_cap"]["sha256"] == pixmo_sha256
    assert set(pins["pixmo_cap"]["output_splits"]) == {"train", "validation"}
    assert pins["planned_outputs"] == {
        "provenance_dir": str(provenance),
        "provenance_manifest": str(provenance / module.PERCEPTION_PROVENANCE_MANIFEST),
        "probe_dir": str(probes),
        "source_catalog": str(probes / module.SOURCE_CATALOG_NAME),
        "source_audit": str(audit),
    }
    assert pins["provenance_builder"]["environment"] == {"PYTHONPATH": "src"}
    assert pins["provenance_builder"]["argv"] == [
        "python",
        "src/scripts/data/build_vision_alignment_perception_provenance.py",
        f"--source-spec={output / 'source-spec.json'}",
        f"--expected-source-spec-sha256={pins['source_spec']['sha256']}",
        f"--expected-source-registry-sha256={pins['source_registry_sha256']}",
        f"--implementation-inventory={output / 'implementation-inventory.json'}",
        "--expected-implementation-inventory-sha256="
        f"{pins['implementation_inventory']['sha256']}",
        f"--finevision-materialization-manifest={finevision_manifest}",
        f"--expected-finevision-materialization-sha256={_sha256(finevision_manifest)}",
        f"--output-dir={provenance}",
        f"--hf-cache-dir={module.DEFAULT_HF_CACHE_DIR}",
    ]
    assert _sha256(output / "source-spec.json") == pins["source_spec"]["sha256"]
    assert (
        _sha256(output / "implementation-inventory.json")
        == pins["implementation_inventory"]["sha256"]
    )
    assert (output / "COMPLETE").read_bytes() == (
        hashlib.sha256(pins_raw).hexdigest() + "\n"
    ).encode("ascii")
    assert not pins_raw.endswith(b"\n")
    assert not (output / "source-spec.json").read_bytes().endswith(b"\n")
    assert not (output / "implementation-inventory.json").read_bytes().endswith(b"\n")
    assert "created_at" not in pins
    assert finevision_validation_calls == [
        (finevision_manifest, _sha256(finevision_manifest)),
        (finevision_manifest, _sha256(finevision_manifest)),
    ]

    with pytest.raises(FileExistsError, match="overwrite"):
        module.materialize_perception_inputs(
            source_spec_template=source_template,
            expected_source_spec_template_sha256=_sha256(source_template),
            finevision_materialization_manifest=finevision_manifest,
            expected_finevision_materialization_sha256=_sha256(finevision_manifest),
            pixmo_cap_validation_manifest=pixmo_manifest,
            expected_pixmo_cap_validation_manifest_sha256=pixmo_sha256,
            output_dir=output,
            expected_repository_commit="a" * 40,
        )


def test_materialize_perception_inputs_cleans_staging_on_prepublish_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _load_module()
    pixmo_manifest, pixmo_sha = _write_pixmo_artifact(module, tmp_path / "pixmo")
    source_template = tmp_path / "source-template.json"
    _write_source_template(module, source_template, pixmo_manifest.parent / "dataset")
    finevision_manifest = tmp_path / "finevision.json"
    finevision_manifest.write_bytes(b"{}")
    materialization = _finevision_materialization(module, finevision_manifest, tmp_path)
    monkeypatch.setattr(
        module,
        "_git_identity",
        lambda *_args: module.GitIdentity("vision-moe", "a" * 40),
    )
    monkeypatch.setattr(
        module,
        "_validate_finevision_materialization",
        lambda *_args: materialization,
    )
    monkeypatch.setattr(
        module,
        "_assert_unchanged_before_publish",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("injected code drift")),
    )
    output = tmp_path / module.INPUT_OUTPUT_NAME

    with pytest.raises(ValueError, match="code drift"):
        module.materialize_perception_inputs(
            source_spec_template=source_template,
            expected_source_spec_template_sha256=_sha256(source_template),
            finevision_materialization_manifest=finevision_manifest,
            expected_finevision_materialization_sha256=_sha256(finevision_manifest),
            pixmo_cap_validation_manifest=pixmo_manifest,
            expected_pixmo_cap_validation_manifest_sha256=pixmo_sha,
            output_dir=output,
            expected_repository_commit="a" * 40,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(f".{module.INPUT_OUTPUT_NAME}.*.building"))


def test_pixmo_validation_requires_exact_complete_and_live_identity(tmp_path: Path):
    module = _load_module()
    pixmo_manifest, pixmo_sha = _write_pixmo_artifact(module, tmp_path / "pixmo")
    builder_sha = module._pipeline_script_hashes(module._repository_root())["pixmo_cap_builder"][
        "sha256"
    ]
    (pixmo_manifest.parent / "COMPLETE").write_bytes((pixmo_sha + "\n\n").encode("ascii"))
    with pytest.raises(ValueError, match="COMPLETE"):
        module._validate_pixmo_cap_manifest(
            pixmo_manifest,
            pixmo_sha,
            expected_dataset_path=pixmo_manifest.parent / "dataset",
            expected_builder_sha256=builder_sha,
        )

    manifest = json.loads(pixmo_manifest.read_bytes())
    manifest["output"]["splits"]["train"]["examples"] += 1
    pixmo_sha = _write_json(pixmo_manifest, manifest)
    (pixmo_manifest.parent / "COMPLETE").write_bytes((pixmo_sha + "\n").encode("ascii"))
    with pytest.raises(ValueError, match="Live canonical PixMoCap train split differs"):
        module._validate_pixmo_cap_manifest(
            pixmo_manifest,
            pixmo_sha,
            expected_dataset_path=pixmo_manifest.parent / "dataset",
            expected_builder_sha256=builder_sha,
        )


def test_git_identity_requires_pinned_commit_owned_branch_and_clean_tree(tmp_path: Path):
    module = _load_module()
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-b", "vision-moe"], cwd=repository, check=True)
    tracked = repository / "tracked.txt"
    tracked.write_text("clean\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "initial",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    identity = module._git_identity(repository, commit)
    assert identity == module.GitIdentity("vision-moe", commit)
    with pytest.raises(ValueError, match="requires commit"):
        module._git_identity(repository, "b" * 40)

    subprocess.run(["git", "branch", "-m", "main"], cwd=repository, check=True)
    with pytest.raises(ValueError, match="requires branch"):
        module._git_identity(repository, commit)
    subprocess.run(["git", "branch", "-m", "vision-moe"], cwd=repository, check=True)

    tracked.write_text("dirty\n")
    with pytest.raises(ValueError, match="clean repository worktree"):
        module._git_identity(repository, commit)


def test_materialize_perception_inputs_requires_canonical_output_name(tmp_path: Path):
    module = _load_module()
    with pytest.raises(ValueError, match="canonical name"):
        module.materialize_perception_inputs(
            source_spec_template=tmp_path / "template.json",
            expected_source_spec_template_sha256="a" * 64,
            finevision_materialization_manifest=tmp_path / "finevision.json",
            expected_finevision_materialization_sha256="b" * 64,
            pixmo_cap_validation_manifest=tmp_path / "pixmo.json",
            expected_pixmo_cap_validation_manifest_sha256="c" * 64,
            output_dir=tmp_path / "custom-input-bundle",
            expected_repository_commit="d" * 40,
        )
