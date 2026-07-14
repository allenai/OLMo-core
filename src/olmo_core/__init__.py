from __future__ import annotations

import os
import socket
from pathlib import Path


def _default_triton_cache_dir() -> None:
    """
    Avoid multi-rank races in Triton's on-disk compiler cache.

    Triton's default cache under the user's home directory is shared by every
    local rank. During autotune several ranks can compile the same kernel/config
    at the same time, which has shown up as missing `.cubin` files after one
    process observes another process's in-progress cache entry. Keep the default
    cache process-local by local rank, while still allowing launchers to override
    it explicitly with TRITON_CACHE_DIR.
    """
    if os.environ.get("TRITON_CACHE_DIR"):
        return
    if os.environ.get("OLMO_DISABLE_PER_RANK_TRITON_CACHE"):
        return

    local_rank = (
        os.environ.get("LOCAL_RANK")
        or os.environ.get("SLURM_LOCALID")
        or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or os.environ.get("MV2_COMM_WORLD_LOCAL_RANK")
        or "0"
    )
    job_id = (
        os.environ.get("BEAKER_EXPERIMENT_ID")
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("JOB_ID")
        or "default"
    )
    host = socket.gethostname().split(".")[0] or "host"
    base = Path(os.environ.get("OLMO_TRITON_CACHE_BASE", "/tmp/olmo-triton-cache"))
    cache_dir = base / str(job_id) / host / f"local_rank_{local_rank}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)


_default_triton_cache_dir()
