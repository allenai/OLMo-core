import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/merge_olmoe_ladder_wandb_cache.py"
SPEC = importlib.util.spec_from_file_location("merge_olmoe_ladder_wandb_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_cache(path: Path, *, step: int, rows: int, state: str = "finished") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "cache_version": 2,
                    "history_keys": ["_step", "train/CE loss"],
                    "state": state,
                    "summary_step": step,
                    "summary_total_tokens": step * 100,
                },
                "history": [{"_step": index} for index in range(rows)],
            }
        ),
        encoding="utf-8",
    )


def test_merge_selects_more_complete_cache_and_preserves_unique_entries(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    destination = tmp_path / "destination"
    write_cache(first / "shared.json", step=10, rows=10)
    write_cache(second / "shared.json", step=11, rows=11)
    write_cache(first / "first_only.json", step=5, rows=5)
    write_cache(second / "second_only.json", step=6, rows=6)

    report = MODULE.merge([first, second], destination)

    assert report["candidate_filenames"] == 3
    assert report["errors"] == []
    shared = json.loads((destination / "shared.json").read_text())
    assert shared["metadata"]["summary_step"] == 11
    assert (destination / "first_only.json").is_file()
    assert (destination / "second_only.json").is_file()


def test_merge_reports_invalid_candidates(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "bad.json").write_text("not json", encoding="utf-8")
    report = MODULE.merge([source], tmp_path / "destination")
    assert report["counts"]["invalid_candidates"] == 1
    assert report["counts"]["unresolved_files"] == 1
    assert len(report["errors"]) == 1
