"""
Build a mixture dataset from a list of source datasets and benchmark it.
"""

import logging
import os
import time
from tempfile import TemporaryDirectory

import s3fs

from olmo_core.data import NumpyDatasetDType
from olmo_core.data.mixture_dataset import SourceMixtureDatasetConfig, SourceMixtureConfig

log = logging.getLogger(__name__)


def build_config(output_dir, processes) -> SourceMixtureDatasetConfig:
    s3 = s3fs.S3FileSystem()
    books = s3.glob("s3://ai2-llm/preprocessed/books/allenai_dolma2/*.npy")
    dclm = s3.glob(
        "s3://ai2-llm/preprocessed/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/*.npy"
    )

    print(f"Found {len(books)} books files")
    print(f"Found {len(dclm)} dclm files")

    return SourceMixtureDatasetConfig(
        max_tokens=1_000_000_000,
        source_configs=[
            SourceMixtureConfig(
                source_name="books",
                paths=[f"s3://{path}" for path in books],
                max_repetition_ratio=1.0,
                target_ratio=0.1,
            ),
            SourceMixtureConfig(
                source_name="dclm",
                paths=[f"s3://{path}" for path in dclm],
                target_ratio=0.9,
            ),
        ],
        dtype=NumpyDatasetDType.uint32,
        output_dir=output_dir,
        processes=processes,
        seed=42,
        dry_run=False,
    )


if __name__ == "__main__":
    with TemporaryDirectory() as temp_dir:
        processes = os.cpu_count()
        # TODO: ADD DRY RUN TIME
        print(f"Running with {processes} processes")
        config_a = build_config(temp_dir, processes)
        start_time = time.time()
        dataset = config_a.build()
        end_time = time.time()
        print(f"Built dataset in {end_time - start_time:.2f} seconds")
