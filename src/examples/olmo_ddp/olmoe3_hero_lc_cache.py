"""Require the exact existing LC packing cache; keep shared data read-only."""

import json
import os
from pathlib import Path

from olmoe3_hero_lc_plan import (
    DATA_GLOB,
    METADATA_CACHE,
    MOUNT,
    PACKING_CACHE,
    SEQUENCE_LENGTH,
    TRAINING_MOUNT,
)


def validate_cache():
    """Check all 375 cache products before dataset.prepare can attempt packing."""
    from cached_path import get_cache_dir

    from olmo_core.data import (
        InstanceFilterConfig,
        NumpyPackedFSLDatasetConfig,
        TokenizerConfig,
    )

    assert MOUNT.is_mount() and TRAINING_MOUNT.is_mount(), "Refuse overlay data/cache"
    assert os.environ.get("CACHED_PATH_CACHE_ROOT") == str(METADATA_CACHE)
    assert Path(get_cache_dir()) == METADATA_CACHE
    assert METADATA_CACHE.resolve() == METADATA_CACHE
    METADATA_CACHE.mkdir(parents=True, exist_ok=True)
    fs = os.statvfs(MOUNT)
    assert fs.f_bavail * fs.f_frsize >= 12_000_000_000_000
    config = NumpyPackedFSLDatasetConfig.glob(
        DATA_GLOB,
        tokenizer=TokenizerConfig.dolma2(),
        work_dir=str(PACKING_CACHE),
        sequence_length=SEQUENCE_LENGTH,
        source_group_size=8,
        source_permutation_seed=123,
        instance_filter_config=InstanceFilterConfig(),
    )
    dataset = config.build()
    assert len(dataset.paths) == 1000
    count = total_bytes = 0
    for paths in dataset._source_path_groups:
        for method in (
            dataset._get_document_indices_path,
            dataset._get_instance_offsets_path,
            dataset._get_docs_by_instance_path,
        ):
            path = method(*paths)
            assert path.is_file() and path.resolve() == path and path.stat().st_size > 0, path
            total_bytes += path.stat().st_size
            count += 1
    assert count == 375 and len(dataset) == 34_796_260
    sample = dataset[0]["input_ids"]
    assert sample.numel() == SEQUENCE_LENGTH and bool((sample < 100352).all())
    print(
        "LC_DATA_CACHE_VERIFIED",
        json.dumps(
            dict(
                source_files=1000,
                products=count,
                cache_bytes=total_bytes,
                packed_instances=len(dataset),
                sequence_length=SEQUENCE_LENGTH,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    validate_cache()
