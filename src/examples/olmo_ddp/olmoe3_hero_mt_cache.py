"""Keep MT's downloaded document sidecars off the container's ephemeral disk."""

import gzip
import json
import os
from pathlib import Path

from olmoe3_hero_mt_plan import METADATA_CACHE, MOUNT


def validate_cache():
    """Verify the actual cached_path default and real mounted storage before preparation."""
    from cached_path import get_cache_dir

    assert MOUNT.is_mount(), "Refuse an overlay-backed metadata cache"
    assert os.environ.get("CACHED_PATH_CACHE_ROOT") == str(METADATA_CACHE)
    assert Path(get_cache_dir()) == METADATA_CACHE
    assert METADATA_CACHE.resolve() == METADATA_CACHE
    METADATA_CACHE.mkdir(parents=True, exist_ok=True)
    fs = os.statvfs(METADATA_CACHE)
    assert fs.f_bavail * fs.f_frsize >= 12_000_000_000_000
    print("MT_METADATA_CACHE", json.dumps(dict(path=str(METADATA_CACHE))), flush=True)


def probe():
    """Exercise a real compressed document-sidecar download and cache hit in the image."""
    from cached_path import cached_path
    from olmoe3_hero_mt import MIX, SourceMixtureList

    from olmo_core.io import get_file_size

    validate_cache()
    for source in SourceMixtureList.from_yaml(str(MIX)).sources:
        if not source.target_ratio:
            continue
        for path in source.resolved_paths[:8]:
            url = str(path).removesuffix(".npy") + ".csv.gz"
            size = get_file_size(url)
            if not 0 < size <= 32 * 1024 * 1024:
                continue
            local = Path(cached_path(url, quiet=True))
            assert local.parent == METADATA_CACHE and local.stat().st_size == size
            with gzip.open(local, "rt") as handle:
                fields = handle.readline().split(",")
                assert 0 <= int(fields[0]) < int(fields[1])
            assert Path(cached_path(url, quiet=True)) == local
            print("MT_METADATA_CACHE_PROBE_PASSED", json.dumps(dict(bytes=size)), flush=True)
            return
    raise RuntimeError("No bounded metadata-sidecar probe candidate found")


if __name__ == "__main__":
    probe()
