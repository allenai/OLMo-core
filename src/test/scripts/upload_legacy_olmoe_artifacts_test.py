import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts/upload_legacy_olmoe_artifacts.py"
SPEC = importlib.util.spec_from_file_location("upload_legacy_olmoe_artifacts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_inventory_is_relative_and_exact(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_bytes(b"abc")
    (tmp_path / "nested/b.bin").write_bytes(b"12345")

    result = MODULE.inventory(tmp_path)

    assert result["file_count"] == 2
    assert result["total_bytes"] == 8
    assert [item["path"] for item in result["files"]] == ["a.txt", "nested/b.bin"]


def test_inventory_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("data", encoding="utf-8")
    (tmp_path / "link").symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"):
        MODULE.inventory(tmp_path)


@pytest.mark.parametrize("text", ["Would copy a to b", "would delete gs://extra"])
def test_dry_run_change_detection(text: str) -> None:
    assert MODULE.DRY_RUN_CHANGE.search(text)
