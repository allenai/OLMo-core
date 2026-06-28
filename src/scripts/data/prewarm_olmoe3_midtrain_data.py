#!/usr/bin/env python
"""Prewarm the OLMoE3 midtraining data cache.

This builds the same source-mixture dataset used by
``src/scripts/train/OLMoE3-dev-260614-s002-midtrain.py`` without building the
model or initializing distributed training.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from glob import glob
from pathlib import Path
from typing import Optional


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("NO_GCE_CHECK", "true")

from olmo_core.data import (  # noqa: E402
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.source_mixture import (  # noqa: E402
    SourceMixtureDatasetConfig,
    SourceMixtureList,
)
from olmo_core.utils import prepare_cli_environment  # noqa: E402


log = logging.getLogger(__name__)

DEFAULT_SOURCE_MIX_YAML = (
    SRC_ROOT / "olmo_core/data/source_mixtures/OLMo3-32B-midtraining-modelnamefilter.yaml"
)
DEFAULT_LOCAL_WORK_DIR = (
    "/weka/oe-training-default/ai2-llm/checkpoints/"
    "OLMoE3-dev-260614-s002-midtraining/dataset-cache"
)
DEFAULT_SOURCE_DATA_BASE_DIR = "/weka/oe-training-default/ai2-llm"
DEFAULT_REQUESTED_TOKENS = 100_000_000_000
DEFAULT_GLOBAL_BATCH_SIZE = 4 * 1024 * 1024
DEFAULT_SEQUENCE_LENGTH = 8192
DEFAULT_SEED = 2026
DEFAULT_SOURCE_MIX_PROCESSES = 16
DEFAULT_DATA_PREP_WORKERS = 8
DATA_LOADER_SEED = 34521


def _local_path_matches(path: str) -> bool:
    if "*" in path:
        return bool(glob(path, recursive=True))
    return Path(path).exists()


def _rewrite_source_paths(
    source_list: SourceMixtureList,
    source_data_base_dir: Optional[str],
    *,
    local_missing_fallback: bool,
) -> None:
    if not source_data_base_dir:
        return

    clean_base = source_data_base_dir.rstrip("/")
    if not Path(clean_base).exists():
        log.warning(
            "Source data base dir '%s' does not exist; keeping source YAML paths", clean_base
        )
        return

    rewritten = 0
    fallback = 0
    for source in source_list.sources:
        rewritten_paths = []
        for path in source.paths:
            if path.startswith("gs://ai2-llm/"):
                rewritten_path = path.replace("gs://ai2-llm", clean_base, 1)
            elif path.startswith("s3://ai2-llm/"):
                rewritten_path = path.replace("s3://ai2-llm", clean_base, 1)
            else:
                rewritten_path = path

            if rewritten_path != path:
                rewritten += 1

            if (
                rewritten_path != path
                and local_missing_fallback
                and not _local_path_matches(rewritten_path)
            ):
                fallback += 1
                log.warning(
                    "No local matches for '%s'; falling back to source YAML path '%s'",
                    rewritten_path,
                    path,
                )
                rewritten_paths.append(path)
            else:
                rewritten_paths.append(rewritten_path)
        source.paths = rewritten_paths
        source._resolved_paths = None

    log.info(
        "Path rewrite summary: %d path pattern(s) rewritten to '%s', %d fallback(s)",
        rewritten,
        clean_base,
        fallback,
    )


def _limit_source_paths(
    source_list: SourceMixtureList,
    max_paths_per_source: int,
    *,
    seed: int,
) -> None:
    if max_paths_per_source <= 0:
        return

    for source in source_list.sources:
        resolved_paths = list(source.resolved_paths)
        if len(resolved_paths) <= max_paths_per_source:
            continue

        rng = random.Random(f"{seed}:{source.source_name}")
        selected_indices = sorted(rng.sample(range(len(resolved_paths)), max_paths_per_source))
        selected_paths = [resolved_paths[idx] for idx in selected_indices]
        log.info(
            "Limiting source '%s' from %d to %d resolved paths",
            source.source_name,
            len(resolved_paths),
            len(selected_paths),
        )
        source.paths = selected_paths
        source._resolved_paths = selected_paths


def _count_existing_mixture_indices(dataset) -> tuple[int, int]:
    existing = 0
    missing = 0
    for path in dataset.paths:
        if dataset._get_instance_indices_path(path).is_file():
            existing += 1
        else:
            missing += 1
    return existing, missing


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-mix-yaml",
        default=str(DEFAULT_SOURCE_MIX_YAML),
        help="Source-mixture YAML to materialize.",
    )
    parser.add_argument(
        "--work-dir",
        default="./dataset-cache",
        help="Dataset cache directory. Use the same value as training.",
    )
    parser.add_argument(
        "--local-work-dir",
        action="store_true",
        help=f"Use the midtrain script's local Weka work dir: {DEFAULT_LOCAL_WORK_DIR}",
    )
    parser.add_argument(
        "--source-data-base-dir",
        default=DEFAULT_SOURCE_DATA_BASE_DIR,
        help="Local base used to rewrite gs://ai2-llm and s3://ai2-llm paths.",
    )
    parser.add_argument(
        "--no-local-missing-fallback",
        action="store_true",
        help="Fail through to missing local paths instead of falling back to YAML URLs.",
    )
    parser.add_argument("--requested-tokens", type=int, default=DEFAULT_REQUESTED_TOKENS)
    parser.add_argument("--global-batch-size", type=int, default=DEFAULT_GLOBAL_BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source-mix-processes", type=int, default=DEFAULT_SOURCE_MIX_PROCESSES)
    parser.add_argument("--max-paths-per-source", type=int, default=0)
    parser.add_argument(
        "--data-prep-workers",
        type=int,
        default=None,
        help=(
            "Number of subprocesses for per-file instance-index creation. "
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
        help="Only create per-file mixture instance indices; do not build epoch global indices.",
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

    if args.local_work_dir:
        args.work_dir = DEFAULT_LOCAL_WORK_DIR
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

    log.info("Prewarming OLMoE3 midtraining data")
    log.info("source_mix_yaml=%s", args.source_mix_yaml)
    log.info("work_dir=%s", work_dir)
    log.info("requested_tokens=%s", f"{args.requested_tokens:,d}")
    log.info("global_batch_size=%s", f"{args.global_batch_size:,d}")
    log.info("sequence_length=%s", f"{args.sequence_length:,d}")
    log.info("seed=%d", args.seed)
    log.info("source_mix_processes=%d", args.source_mix_processes)
    log.info("OLMO_DATA_PREP_WORKERS=%s", os.environ["OLMO_DATA_PREP_WORKERS"])
    log.info(
        "OLMO_DATA_PREP_USE_ARRAY_IF_LOCAL=%s",
        os.environ.get("OLMO_DATA_PREP_USE_ARRAY_IF_LOCAL", "auto"),
    )

    source_list = SourceMixtureList.from_yaml(args.source_mix_yaml)
    log.info("Loaded %d source mixture entries", len(source_list.sources))
    _rewrite_source_paths(
        source_list,
        args.source_data_base_dir,
        local_missing_fallback=not args.no_local_missing_fallback,
    )
    _limit_source_paths(source_list, args.max_paths_per_source, seed=args.seed)
    source_list.validate()

    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=SourceMixtureDatasetConfig(
            source_list=source_list,
            requested_tokens=args.requested_tokens,
            global_batch_size=args.global_batch_size,
            processes=args.source_mix_processes,
            seed=args.seed,
            render_tables=False,
            quiet=True,
        ),
        tokenizer=TokenizerConfig.dolma2(),
        work_dir=str(work_dir),
        sequence_length=args.sequence_length,
        max_target_sequence_length=max(args.sequence_length, DEFAULT_SEQUENCE_LENGTH),
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
    existing, missing = _count_existing_mixture_indices(dataset)
    log.info(
        "Resolved %d selected data path(s): %d mixture index file(s) already exist, %d missing",
        len(dataset.paths),
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
            ignore_fingerprint_mismatch=True,
        )
        data_loader = data_loader_config.build(dataset)
        log.info("Warming global data-order indices for epoch %d", args.epoch)
        data_loader.reshuffle(epoch=args.epoch)

    existing, missing = _count_existing_mixture_indices(dataset)
    elapsed = time.monotonic() - started_at
    log.info(
        "Prewarm complete in %.1fs: %d mixture index file(s) exist, %d missing",
        elapsed,
        existing,
        missing,
    )


if __name__ == "__main__":
    main()
