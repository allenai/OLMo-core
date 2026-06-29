from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "olmoe3_ladder" / "wandb_histories"
CACHE_VERSION = 2


def run_cache_key(project: str, run_id: str) -> str:
    return f"{project.replace('/', '__')}__{run_id}.json"


def tail_cache_key(project: str, run_id: str, window_tokens: int) -> str:
    return f"{project.replace('/', '__')}__{run_id}__tail_{window_tokens}.json"


def run_cache_metadata(run: Any, keys: list[str] | None = None) -> dict[str, Any]:
    summary = dict(run.summary)
    return {
        "cache_version": CACHE_VERSION,
        "history_keys": list(keys) if keys is not None else None,
        "run_id": run.id,
        "state": run.state,
        "summary_step": summary.get("_step"),
        "summary_total_tokens": summary.get("throughput/total tokens") or summary.get("optim/total tokens"),
        "summary_runtime": summary.get("_runtime"),
        "updated_at": str(getattr(run, "updated_at", "")),
    }


def _history_max(history: list[dict[str, Any]], key: str) -> float | None:
    vals = [row.get(key) for row in history if row.get(key) is not None]
    if not vals:
        return None
    return max(float(val) for val in vals)


def _history_min(history: list[dict[str, Any]], key: str) -> float | None:
    vals = [row.get(key) for row in history if row.get(key) is not None]
    if not vals:
        return None
    return min(float(val) for val in vals)


def _tokens_key(keys: list[str]) -> str:
    for key in keys:
        if key.endswith("total tokens"):
            return key
    return "throughput/total tokens"


def _tail_step_range(run: Any, *, window_tokens: int, min_tail_steps: int = 2000) -> tuple[int, int]:
    summary = dict(run.summary)
    summary_step = summary.get("_step") or getattr(run, "lastHistoryStep", None)
    summary_tokens = summary.get("throughput/total tokens") or summary.get("optim/total tokens")
    last_step = getattr(run, "lastHistoryStep", None) or summary_step
    if last_step is None:
        return 0, min_tail_steps

    tail_steps = min_tail_steps
    if summary_step and summary_tokens:
        tokens_per_step = float(summary_tokens) / max(float(summary_step), 1.0)
        if tokens_per_step > 0:
            tail_steps = max(min_tail_steps, math.ceil(window_tokens / tokens_per_step) + 100)

    max_step = int(last_step) + 1
    min_step = max(0, max_step - tail_steps)
    return min_step, max_step


def cached_tail_is_short(
    cached_history: list[dict[str, Any]],
    current_meta: dict[str, Any],
    *,
    window_tokens: int,
    tokens_key: str = "throughput/total tokens",
) -> bool:
    if not cached_history:
        return True

    summary_tokens = current_meta.get("summary_total_tokens")
    cached_tokens = _history_max(cached_history, tokens_key)
    if cached_tokens is None:
        return True
    if summary_tokens is not None and cached_tokens < float(summary_tokens) - 5_000_000:
        return True

    cached_min_tokens = _history_min(cached_history, tokens_key)
    if cached_min_tokens is None:
        return True

    summary_step = current_meta.get("summary_step")
    tokens_per_step = 0.0
    if summary_step and summary_tokens:
        tokens_per_step = float(summary_tokens) / max(float(summary_step), 1.0)
    # The tail cache must reach far enough left to cover the final averaging
    # window. Allow a small step-sized cushion because the window boundary can
    # fall between logged token values.
    cushion = max(tokens_per_step * 2, 1.0)
    return cached_min_tokens > cached_tokens - window_tokens + cushion


def cached_history_is_short(
    cached_history: list[dict[str, Any]],
    current_meta: dict[str, Any],
    *,
    step_key: str = "_step",
    tokens_key: str = "throughput/total tokens",
) -> bool:
    summary_step = current_meta.get("summary_step")
    if summary_step is not None:
        cached_step = _history_max(cached_history, step_key)
        if cached_step is None or cached_step + 1 < float(summary_step):
            return True

    summary_tokens = current_meta.get("summary_total_tokens")
    if summary_tokens is not None:
        cached_tokens = _history_max(cached_history, tokens_key)
        # W&B history can legitimately miss a tiny tail. Treat multi-million-token
        # gaps as stale, but do not churn cache files for harmless logging jitter.
        if cached_tokens is None or cached_tokens < float(summary_tokens) - 5_000_000:
            return True

    return False


def read_history_from_cache(
    cache_dir: Path,
    project: str,
    run: Any,
    *,
    allow_stale: bool = False,
    require_complete: bool = False,
    tokens_key: str = "throughput/total tokens",
    keys: list[str] | None = None,
) -> list[dict[str, Any]] | None:
    cache_path = cache_dir / run_cache_key(project, run.id)
    if not cache_path.exists():
        return None

    with cache_path.open("r") as f:
        cached = json.load(f)

    cached_meta = cached.get("metadata", {})
    current_meta = run_cache_metadata(run, keys)
    if cached_meta.get("cache_version") != CACHE_VERSION:
        return None
    if keys is not None:
        cached_keys = cached_meta.get("history_keys")
        if cached_keys is None or not set(keys).issubset(set(cached_keys)):
            return None
    if cached_meta.get("state") != current_meta["state"]:
        return None
    if not allow_stale and cached_meta.get("summary_step") != current_meta["summary_step"]:
        return None

    history = cached.get("history", [])
    if require_complete and cached_history_is_short(history, current_meta, tokens_key=tokens_key):
        return None

    return history


def read_tail_history_from_cache(
    cache_dir: Path,
    project: str,
    run: Any,
    *,
    window_tokens: int,
    tokens_key: str = "throughput/total tokens",
    keys: list[str] | None = None,
) -> list[dict[str, Any]] | None:
    cache_path = cache_dir / tail_cache_key(project, run.id, window_tokens)
    if not cache_path.exists():
        return None

    with cache_path.open("r") as f:
        cached = json.load(f)

    cached_meta = cached.get("metadata", {})
    current_meta = run_cache_metadata(run, keys)
    if cached_meta.get("cache_version") != CACHE_VERSION:
        return None
    if cached_meta.get("cache_kind") != "tail":
        return None
    if cached_meta.get("tail_window_tokens") != window_tokens:
        return None
    if keys is not None:
        cached_keys = cached_meta.get("history_keys")
        if cached_keys is None or not set(keys).issubset(set(cached_keys)):
            return None
    if cached_meta.get("state") != current_meta["state"]:
        return None
    if cached_meta.get("summary_step") != current_meta["summary_step"]:
        return None

    history = cached.get("history", [])
    if cached_tail_is_short(history, current_meta, window_tokens=window_tokens, tokens_key=tokens_key):
        return None
    return history


def write_tail_history_to_cache(
    cache_dir: Path,
    project: str,
    run: Any,
    history: list[dict[str, Any]],
    *,
    window_tokens: int,
    min_step: int,
    max_step: int,
    keys: list[str] | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / tail_cache_key(project, run.id, window_tokens)
    metadata = run_cache_metadata(run, keys)
    metadata.update(
        {
            "cache_kind": "tail",
            "tail_window_tokens": window_tokens,
            "min_step": min_step,
            "max_step": max_step,
        }
    )
    payload = {
        "metadata": metadata,
        "display_name": run.display_name,
        "url": run.url,
        "history": history,
    }
    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f)
    tmp_path.replace(cache_path)


def write_history_to_cache(
    cache_dir: Path,
    project: str,
    run: Any,
    history: list[dict[str, Any]],
    *,
    keys: list[str] | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / run_cache_key(project, run.id)
    payload = {
        "metadata": run_cache_metadata(run, keys),
        "display_name": run.display_name,
        "url": run.url,
        "history": history,
    }
    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f)
    tmp_path.replace(cache_path)


def scan_history_cached(
    run: Any,
    *,
    project: str,
    keys: list[str],
    cache_dir: Path,
    refresh_cache: bool = False,
    refresh_stale_cache: bool = False,
    page_size: int = 1000,
    tail_window_tokens: int | None = None,
    tail_page_size: int = 100_000,
) -> list[dict[str, Any]]:
    tokens_key = _tokens_key(keys)
    if not refresh_cache and run.state == "finished":
        cached = read_history_from_cache(
            cache_dir,
            project,
            run,
            allow_stale=not refresh_stale_cache,
            require_complete=refresh_stale_cache or tail_window_tokens is not None,
            tokens_key=tokens_key,
            keys=keys,
        )
        if cached is not None:
            return cached

        if tail_window_tokens is not None:
            tail_cached = read_tail_history_from_cache(
                cache_dir,
                project,
                run,
                window_tokens=tail_window_tokens,
                tokens_key=tokens_key,
                keys=keys,
            )
            if tail_cached is not None:
                return tail_cached

    if run.state == "finished" and tail_window_tokens is not None:
        min_step, max_step = _tail_step_range(run, window_tokens=tail_window_tokens)
        history = [
            dict(row)
            for row in run.scan_history(
                keys=keys,
                min_step=min_step,
                max_step=max_step,
                page_size=max(tail_page_size, max_step - min_step),
            )
        ]
        write_tail_history_to_cache(
            cache_dir,
            project,
            run,
            history,
            window_tokens=tail_window_tokens,
            min_step=min_step,
            max_step=max_step,
            keys=keys,
        )
        return history

    if run.state == "finished" and os.environ.get("SKIP_UNCACHED_FINISHED_HISTORY") == "1":
        return []

    history = [dict(row) for row in run.scan_history(keys=keys, page_size=page_size)]
    if run.state == "finished":
        write_history_to_cache(cache_dir, project, run, history, keys=keys)
    return history
