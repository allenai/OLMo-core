from __future__ import annotations

import sys
from pathlib import Path


LADDER_DIR = Path(__file__).parents[3] / "scripts" / "train" / "jacobm_olmoe_ladder"
if str(LADDER_DIR) not in sys.path:
    sys.path.insert(0, str(LADDER_DIR))

from wandb_cache import scan_history_cached


class FakeRun:
    def __init__(self, *, run_id="run1", state="finished", step=10, tokens=10_000_000):
        self.id = run_id
        self.state = state
        self.display_name = "fake-run"
        self.url = "https://wandb.invalid/run1"
        self.summary = {"_step": step, "throughput/total tokens": tokens, "_runtime": 1}
        self.updated_at = "2026-06-12T00:00:00Z"
        self.scan_calls = 0
        self.history = [
            {"_step": 1, "train/CE loss": 3.0, "throughput/total tokens": 1_000_000},
            {"_step": step, "train/CE loss": 2.0, "throughput/total tokens": tokens},
        ]

    def scan_history(self, *, keys, page_size=1000):
        self.scan_calls += 1
        for row in self.history:
            yield {key: row.get(key) for key in keys}


def test_finished_run_history_is_cached(tmp_path: Path) -> None:
    run = FakeRun()
    keys = ["_step", "train/CE loss", "throughput/total tokens"]

    first = scan_history_cached(run, project="entity/project", keys=keys, cache_dir=tmp_path)
    second = scan_history_cached(run, project="entity/project", keys=keys, cache_dir=tmp_path)

    assert first == second
    assert run.scan_calls == 1


def test_cache_reuses_reordered_or_subset_history_keys(tmp_path: Path) -> None:
    run = FakeRun()
    keys = ["_step", "train/CE loss", "throughput/total tokens"]
    reordered_keys = ["throughput/total tokens", "_step", "train/CE loss"]
    subset_keys = ["_step", "train/CE loss"]

    first = scan_history_cached(run, project="entity/project", keys=keys, cache_dir=tmp_path)
    reordered = scan_history_cached(run, project="entity/project", keys=reordered_keys, cache_dir=tmp_path)
    subset = scan_history_cached(run, project="entity/project", keys=subset_keys, cache_dir=tmp_path)

    assert first == reordered == subset
    assert run.scan_calls == 1


def test_cache_misses_when_requested_keys_are_not_cached(tmp_path: Path) -> None:
    run = FakeRun()
    keys = ["_step", "train/CE loss"]
    missing_keys = ["_step", "train/CE loss", "throughput/total tokens"]

    scan_history_cached(run, project="entity/project", keys=keys, cache_dir=tmp_path)
    scan_history_cached(run, project="entity/project", keys=missing_keys, cache_dir=tmp_path)

    assert run.scan_calls == 2


def test_finished_cache_is_sticky_without_refresh_stale(tmp_path: Path) -> None:
    run = FakeRun(step=10, tokens=10_000_000)
    keys = ["_step", "train/CE loss", "throughput/total tokens"]

    cached = scan_history_cached(run, project="entity/project", keys=keys, cache_dir=tmp_path)
    run.summary["_step"] = 20
    run.summary["throughput/total tokens"] = 20_000_000
    run.history.append({"_step": 20, "train/CE loss": 1.9, "throughput/total tokens": 20_000_000})
    reused = scan_history_cached(run, project="entity/project", keys=keys, cache_dir=tmp_path)

    assert reused == cached
    assert run.scan_calls == 1


def test_refresh_stale_cache_redownloads_short_finished_history(tmp_path: Path) -> None:
    run = FakeRun(step=10, tokens=10_000_000)
    keys = ["_step", "train/CE loss", "throughput/total tokens"]

    scan_history_cached(run, project="entity/project", keys=keys, cache_dir=tmp_path)
    run.summary["_step"] = 20
    run.summary["throughput/total tokens"] = 20_000_000
    run.history.append({"_step": 20, "train/CE loss": 1.9, "throughput/total tokens": 20_000_000})

    refreshed = scan_history_cached(
        run,
        project="entity/project",
        keys=keys,
        cache_dir=tmp_path,
        refresh_stale_cache=True,
    )

    assert run.scan_calls == 2
    assert refreshed[-1]["_step"] == 20
