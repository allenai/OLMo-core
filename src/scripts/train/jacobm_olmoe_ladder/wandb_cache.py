from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "olmoe3_ladder" / "wandb_histories"
CACHE_VERSION = 2


def run_cache_key(project: str, run_id: str) -> str:
    return f"{project.replace('/', '__')}__{run_id}.json"


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
) -> list[dict[str, Any]]:
    if not refresh_cache and run.state == "finished":
        tokens_key = "throughput/total tokens"
        for key in keys:
            if key.endswith("total tokens"):
                tokens_key = key
                break
        cached = read_history_from_cache(
            cache_dir,
            project,
            run,
            allow_stale=not refresh_stale_cache,
            require_complete=refresh_stale_cache,
            tokens_key=tokens_key,
            keys=keys,
        )
        if cached is not None:
            return cached

    history = [dict(row) for row in run.scan_history(keys=keys, page_size=page_size)]
    if run.state == "finished":
        write_history_to_cache(cache_dir, project, run, history, keys=keys)
    return history
