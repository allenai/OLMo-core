import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/upload_olmo_ddp_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("upload_olmo_ddp_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def publication_ready_checkpoint(tmp_path: Path) -> tuple[Path, dict]:
    source = tmp_path / "source"
    checkpoint = tmp_path / "converted"
    source.mkdir()
    (checkpoint / "model_and_optim").mkdir(parents=True)
    for relative in ("README.md", "config.json", "source_config.json"):
        (checkpoint / relative).write_text("{}", encoding="utf-8")
    (checkpoint / "model_and_optim/.metadata").write_bytes(b"metadata")
    write_json(
        checkpoint / "conversion_manifest.json",
        {
            "source_checkpoint": str(source.resolve()),
            "optimizer_state_included": False,
            "trainer_state_included": False,
        },
    )
    write_json(
        checkpoint / "strict_tensor_verification.json",
        {
            "status": "STRICT_TENSOR_MATCH",
            "bitwise_equal": True,
            "target_model_only": True,
        },
    )
    write_json(
        checkpoint / "exact_logits_verification.json",
        {"status": "LOGITS_MATCH", "exact_match": True},
    )
    write_json(
        checkpoint / "legacy_config_schema_validation.json",
        {"status": "LEGACY_CONFIG_SCHEMA_MATCH"},
    )
    entry = {"id": "pretraining/baseline/275m/cx1", "source_checkpoint": str(source)}
    return checkpoint, entry


def test_validate_acceptance_requires_all_exact_checks(tmp_path: Path) -> None:
    checkpoint, entry = publication_ready_checkpoint(tmp_path)
    MODULE.validate_acceptance(checkpoint, entry)
    report = MODULE.load_json(checkpoint / "exact_logits_verification.json")
    report["exact_match"] = False
    write_json(checkpoint / "exact_logits_verification.json", report)
    try:
        MODULE.validate_acceptance(checkpoint, entry)
    except ValueError as error:
        assert "Exact logits" in str(error)
    else:
        raise AssertionError("Expected exact-logit acceptance failure")


def test_payload_excludes_large_logit_artifacts_and_success_files(tmp_path: Path) -> None:
    checkpoint, _ = publication_ready_checkpoint(tmp_path)
    (checkpoint / "verification").mkdir()
    (checkpoint / "verification/legacy_logits.pt").write_bytes(b"artifact")
    (checkpoint / "verification/export.log").write_text("log", encoding="utf-8")
    (checkpoint / MODULE.UPLOAD_REPORT).write_text("report", encoding="utf-8")
    (checkpoint / MODULE.SUCCESS_MARKER).write_text("success", encoding="utf-8")
    relative = {
        path.relative_to(checkpoint).as_posix() for path in MODULE.payload_files(checkpoint)
    }
    assert "model_and_optim/.metadata" in relative
    assert "verification/legacy_logits.pt" not in relative
    assert "verification/export.log" not in relative
    assert MODULE.UPLOAD_REPORT not in relative
    assert MODULE.SUCCESS_MARKER not in relative
