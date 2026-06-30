#!/usr/bin/env python
"""Prewarm the OLMoE3 long-context packed data cache.

This builds the same packed FSL dataset used by
``src/scripts/train/OLMoE3-dev-260614-s002-long-context.py`` without building
the model or initializing distributed training.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("NO_GCE_CHECK", "true")

from olmo_core.data import (  # noqa: E402
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.utils import prepare_cli_environment  # noqa: E402


log = logging.getLogger(__name__)

DEFAULT_WORK_DIR = (
    "/weka/oe-training-default/ai2-llm/checkpoints/"
    "OLMoE3-dev-260614-s002-long-context/dataset-cache"
)
DEFAULT_MIX_BASE_DIR = "s3://ai2-llm"
DEFAULT_SEQUENCE_LENGTH = 65536
DEFAULT_GLOBAL_BATCH_SIZE = 8 * 1024 * 1024
DEFAULT_DATA_PREP_WORKERS = 8
DATA_LOADER_SEED = 34521


def _count_existing_packed_indices(dataset) -> tuple[int, int]:
    existing = 0
    missing = 0
    for source_paths in dataset._source_path_groups:
        paths = (
            dataset._get_document_indices_path(*source_paths),
            dataset._get_instance_offsets_path(*source_paths),
            dataset._get_docs_by_instance_path(*source_paths),
        )
        if all(path.is_file() for path in paths):
            existing += 1
        else:
            missing += 1
    return existing, missing


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mix-base-dir",
        default=DEFAULT_MIX_BASE_DIR,
        help="Base path for DataMix.OLMo_longmino_mix_0925, e.g. s3://ai2-llm.",
    )
    parser.add_argument(
        "--work-dir",
        default=DEFAULT_WORK_DIR,
        help="Dataset cache directory. Use the same value as training.",
    )
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--global-batch-size", type=int, default=DEFAULT_GLOBAL_BATCH_SIZE)
    parser.add_argument("--source-group-size", type=int, default=8)
    parser.add_argument("--source-permutation-seed", type=int, default=123)
    parser.add_argument(
        "--data-prep-workers",
        type=int,
        default=None,
        help=(
            "Number of subprocesses for source-group packing. "
            "Defaults to OLMO_DATA_PREP_WORKERS or 8."
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=None,
        help=(
            "Seconds between data-prep heartbeat logs. Defaults to "
            "OLMO_DATA_PREP_PROGRESS_INTERVAL or 30."
        ),
    )
    parser.add_argument(
        "--index-source",
        choices=("metadata", "array", "auto"),
        default="metadata",
        help=(
            "How to find document boundaries. 'metadata' uses sibling .csv.gz files, "
            "'array' scans local .npy files for EOS, and 'auto' keeps the library default."
        ),
    )
    parser.add_argument(
        "--skip-global-indices",
        action="store_true",
        help="Only pack document groups; do not build epoch global data-order indices.",
    )
    parser.add_argument("--epoch", type=int, default=1, help="Epoch for global data-order indices.")
    parser.add_argument(
        "--no-instance-filter",
        action="store_true",
        help="Match a config with instance filtering disabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prepare_cli_environment()

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    data_prep_workers = args.data_prep_workers
    if data_prep_workers is None:
        data_prep_workers = int(os.environ.get("OLMO_DATA_PREP_WORKERS", DEFAULT_DATA_PREP_WORKERS))
    os.environ["OLMO_DATA_PREP_WORKERS"] = str(data_prep_workers)
    if args.progress_interval is not None:
        os.environ["OLMO_DATA_PREP_PROGRESS_INTERVAL"] = str(args.progress_interval)
    if args.index_source == "metadata":
        os.environ["OLMO_DATA_PREP_USE_ARRAY_IF_LOCAL"] = "false"
    elif args.index_source == "array":
        os.environ["OLMO_DATA_PREP_USE_ARRAY_IF_LOCAL"] = "true"

    log.info("Prewarming OLMoE3 long-context packed data")
    log.info("mix=DataMix.OLMo_longmino_mix_0925")
    log.info("mix_base_dir=%s", args.mix_base_dir)
    log.info("work_dir=%s", work_dir)
    log.info("sequence_length=%s", f"{args.sequence_length:,d}")
    log.info("global_batch_size=%s", f"{args.global_batch_size:,d}")
    log.info("source_group_size=%d", args.source_group_size)
    log.info("source_permutation_seed=%d", args.source_permutation_seed)
    log.info("OLMO_DATA_PREP_WORKERS=%s", os.environ["OLMO_DATA_PREP_WORKERS"])
    log.info(
        "OLMO_DATA_PREP_USE_ARRAY_IF_LOCAL=%s",
        os.environ.get("OLMO_DATA_PREP_USE_ARRAY_IF_LOCAL", "auto"),
    )

    dataset_config = NumpyPackedFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_longmino_mix_0925,
        mix_base_dir=args.mix_base_dir,
        tokenizer=TokenizerConfig.dolma2(),
        work_dir=str(work_dir),
        sequence_length=args.sequence_length,
        generate_doc_lengths=True,
        source_group_size=args.source_group_size,
        source_permutation_seed=args.source_permutation_seed,
        instance_filter_config=None
        if args.no_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13,
            repetition_min_period=1,
            repetition_max_count=32,
        ),
    )

    started_at = time.monotonic()
    dataset = dataset_config.build()
    existing, missing = _count_existing_packed_indices(dataset)
    log.info(
        "Resolved %d data path(s) in %d source group(s): "
        "%d packed group cache(s) already exist, %d missing",
        len(dataset.paths),
        len(dataset._source_path_groups),
        existing,
        missing,
    )

    if args.skip_global_indices:
        dataset.prepare()
    else:
        data_loader_config = NumpyDataLoaderConfig(
            global_batch_size=args.global_batch_size,
            seed=DATA_LOADER_SEED,
            num_workers=0,
        )
        data_loader = data_loader_config.build(dataset)
        log.info("Warming global data-order indices for epoch %d", args.epoch)
        data_loader.reshuffle(epoch=args.epoch)

    existing, missing = _count_existing_packed_indices(dataset)
    elapsed = time.monotonic() - started_at
    log.info(
        "Prewarm complete in %.1fs: %d packed group cache(s) exist, %d missing",
        elapsed,
        existing,
        missing,
    )


if __name__ == "__main__":
    main()
