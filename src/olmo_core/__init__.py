from __future__ import annotations

import os
import socket
from pathlib import Path

from . import mxfp8_config as _mxfp8_config

_mxfp8_config.get_mxfp8_default_scale_mode()  # Resolves OLMO_MXFP8_SCALE_MODE once.

# Marks a ``TRITON_CACHE_DIR`` value that we set ourselves (vs. a user-provided override), so
# processes that inherit our env — e.g. ranks spawned after this module was first imported — can
# tell the difference and recompute their own directory instead of reusing the parent's.
_TRITON_CACHE_AUTO_ENV = "OLMO_TRITON_CACHE_DIR_AUTO"


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
    existing = os.environ.get("TRITON_CACHE_DIR")
    # Respect an explicit user-provided TRITON_CACHE_DIR, but recompute a value we set ourselves:
    # a parent that imported this module (with no rank env yet) would otherwise leak its
    # ``local_rank_0`` directory to every spawned child.
    if existing and existing != os.environ.get(_TRITON_CACHE_AUTO_ENV):
        return
    if os.environ.get("OLMO_DISABLE_PER_RANK_TRITON_CACHE"):
        return

    local_rank = (
        os.environ.get("LOCAL_RANK")
        or os.environ.get("SLURM_LOCALID")
        or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or os.environ.get("MV2_COMM_WORLD_LOCAL_RANK")
    )
    job_id = (
        os.environ.get("BEAKER_EXPERIMENT_ID")
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("JOB_ID")
        or f"pid{os.getpid()}"  # no launcher job id (bare torchrun / torch.multiprocessing)
    )
    # If the rank env isn't set yet — e.g. this module is imported before a spawned worker sets
    # LOCAL_RANK inside the child — fall back to a per-process leaf so ranks sharing a job id don't
    # all collide on local_rank_0.
    rank_component = f"local_rank_{local_rank}" if local_rank is not None else f"pid{os.getpid()}"
    host = socket.gethostname().split(".")[0] or "host"
    base = Path(os.environ.get("OLMO_TRITON_CACHE_BASE", "/tmp/olmo-triton-cache"))
    cache_dir = base / str(job_id) / host / rank_component
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    os.environ[_TRITON_CACHE_AUTO_ENV] = str(cache_dir)


_default_triton_cache_dir()
