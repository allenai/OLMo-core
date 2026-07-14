import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/audit_jacobm_migration.py"
SPEC = importlib.util.spec_from_file_location("audit_jacobm_migration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_complete_audit_requires_every_exact_check_and_artifact(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    family_root = tmp_path / "families"
    artifact_root = tmp_path / "artifacts"
    write_json(
        manifest,
        {
            "models": [
                {
                    "id": "pretraining/baseline/275m/cx1",
                    "family": "baseline",
                    "optimizer_state_included": False,
                    "trainer_state_included": False,
                }
            ]
        },
    )
    write_json(
        family_root / "baseline/family_summary.json",
        {
            "status": "FAMILY_COMPLETE",
            "model_count": 1,
            "completed_at": "now",
            "models": [
                {
                    "model_id": "pretraining/baseline/275m/cx1",
                    **{key: True for key in MODULE.REQUIRED_MODEL_CHECKS},
                }
            ],
        },
    )
    write_json(
        artifact_root / "evals/_SUCCESS.json",
        {
            "status": "COMPLETE",
            "file_count": 2,
            "total_bytes": 3,
            "verification": "checksum dry run reported no changes",
        },
    )

    report = MODULE.audit(manifest, family_root, artifact_root, ["evals"])
    assert report["status"] == "COMPLETE"
    assert report["completed_checkpoints"] == 1

    summary_path = family_root / "baseline/family_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["models"][0]["exact_logits"] = False
    write_json(summary_path, summary)
    report = MODULE.audit(manifest, family_root, artifact_root, ["evals"])
    assert report["status"] == "INCOMPLETE"
    assert any("exact_logits" in problem for problem in report["problems"])
