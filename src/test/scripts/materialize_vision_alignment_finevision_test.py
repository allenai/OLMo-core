from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from olmo_core.data.multimodal.dataset_compat import load_from_disk_compat


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "data"
        / "materialize_vision_alignment_finevision.py"
    )
    spec = importlib.util.spec_from_file_location("finevision_materializer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_source(root: Path, name: str, rows: int) -> None:
    source = root / name
    source.mkdir(parents=True)
    schema = pa.schema(
        [
            pa.field(
                "images",
                pa.list_(
                    pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
                ),
            ),
            pa.field(
                "texts",
                pa.list_(
                    pa.struct([pa.field("user", pa.string()), pa.field("assistant", pa.string())])
                ),
            ),
            pa.field("formatting_min", pa.int64()),
            pa.field("visual_dependency_min", pa.int64()),
            pa.field("image_correspondence_min", pa.int64()),
            pa.field("relevance_min", pa.int64()),
        ],
        metadata={
            b"huggingface": json.dumps(
                {
                    "info": {
                        "features": {
                            "images": {"feature": {"_type": "Image"}, "_type": "List"},
                            "texts": {
                                "feature": {
                                    "user": {"dtype": "string", "_type": "Value"},
                                    "assistant": {"dtype": "string", "_type": "Value"},
                                },
                                "_type": "List",
                            },
                            **{
                                field: {"dtype": "int64", "_type": "Value"}
                                for field in (
                                    "formatting_min",
                                    "visual_dependency_min",
                                    "image_correspondence_min",
                                    "relevance_min",
                                )
                            },
                        }
                    }
                }
            ).encode()
        },
    )
    table = pa.Table.from_pylist(
        [
            {
                "images": [{"bytes": f"image-{index}".encode(), "path": None}],
                "texts": [{"user": "question", "assistant": f"answer-{index}"}],
                "formatting_min": 5,
                "visual_dependency_min": 5,
                "image_correspondence_min": 5,
                "relevance_min": 5,
            }
            for index in range(rows)
        ],
        schema=schema,
    )
    pq.write_table(table, source / "train-00000-of-00001.parquet")


def test_materialize_sources_is_compatible_atomic_and_immutable(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "source"
    _write_source(source, "alpha", 3)
    _write_source(source, "beta", 2)
    specs = (
        module.FineVisionSourceSpec("alpha", "alpha-output", 1, 3),
        module.FineVisionSourceSpec("beta", "beta-output", 1, 2),
    )
    output = tmp_path / "artifact"

    manifest_path = module.materialize_sources(source_root=source, output_dir=output, specs=specs)

    assert (output / "COMPLETE").read_text().strip() == module._sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "verified"
    assert [entry["rows"] for entry in manifest["outputs"]] == [3, 2]
    assert len(load_from_disk_compat(output / "alpha-output")) == 3
    assert load_from_disk_compat(output / "alpha-output")[1]["texts"][0]["assistant"] == "answer-1"
    assert not output.with_name(f".{output.name}.building").exists()

    with pytest.raises(FileExistsError, match="overwrite"):
        module.materialize_sources(source_root=source, output_dir=output, specs=specs)


def test_materialize_resume_rejects_source_drift(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "source"
    _write_source(source, "alpha", 1)
    spec = (module.FineVisionSourceSpec("alpha", "alpha-output", 1, 1),)
    output = tmp_path / "artifact"
    staging = output.with_name(f".{output.name}.building")
    staging.mkdir()
    (staging / "build-plan.json").write_text("{}")

    with pytest.raises(ValueError, match="resume plan"):
        module.materialize_sources(
            source_root=source,
            output_dir=output,
            specs=spec,
            resume=True,
        )


def test_materialize_resume_recovers_arrow_without_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_module()
    source = tmp_path / "source"
    _write_source(source, "alpha", 2)
    specs = (module.FineVisionSourceSpec("alpha", "alpha-output", 1, 2),)
    output = tmp_path / "artifact"
    original_write = module._write_json_atomic
    interrupted = False

    def interrupt_receipt(path, value):
        nonlocal interrupted
        if path.name.endswith(".receipt.json") and not interrupted:
            interrupted = True
            raise RuntimeError("injected receipt interruption")
        return original_write(path, value)

    monkeypatch.setattr(module, "_write_json_atomic", interrupt_receipt)
    with pytest.raises(RuntimeError, match="receipt interruption"):
        module.materialize_sources(source_root=source, output_dir=output, specs=specs)

    staging = output.with_name(f".{output.name}.building")
    arrow = staging / "alpha-output" / "data-00000-of-00001.arrow"
    assert arrow.is_file()
    assert not arrow.with_suffix(".receipt.json").exists()

    monkeypatch.setattr(module, "_write_json_atomic", original_write)
    manifest = module.materialize_sources(
        source_root=source,
        output_dir=output,
        specs=specs,
        resume=True,
    )
    assert manifest.is_file()
    assert not staging.exists()
    assert len(load_from_disk_compat(output / "alpha-output")) == 2
