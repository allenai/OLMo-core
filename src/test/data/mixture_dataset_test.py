from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import numpy as np
import pytest

from olmo_core.aliases import PathOrStr
from olmo_core.data import NumpyDatasetDType
from olmo_core.data.mixture_dataset import (
    SourceMixtureConfig,
    SourceMixtureDataset,
    SourceMixtureDatasetConfig,
)
from olmo_core.data.utils import load_array_slice
from olmo_core.exceptions import OLMoConfigurationError

DATA = {
    "dtype": NumpyDatasetDType.uint32,
    "tokens_per_file": 1_000_000,
}


def _make_mmaps(tmp_path: Path, prefix: str, num_files: int, size: int) -> List[PathOrStr]:
    mmaps = []
    for i in range(num_files):
        filepath = f"{tmp_path}/{prefix}_{i}.npy"
        data = np.random.randint(0, 2**32, size=size, dtype=np.uint32)
        mm = np.memmap(filepath, mode="w+", dtype=DATA["dtype"].as_np_dtype(), shape=(size,))
        mm[:] = data
        mm.flush()
        mmaps.append(Path(filepath))

    return mmaps


def test_source_mixture_config_dry_run(tmp_path: Path, capsys):
    source_paths = {
        "1": _make_mmaps(
            tmp_path=tmp_path, prefix="source1", num_files=2, size=DATA["tokens_per_file"]
        ),
        "2": _make_mmaps(
            tmp_path=tmp_path, prefix="source2", num_files=2, size=DATA["tokens_per_file"]
        ),
        "3": _make_mmaps(
            tmp_path=tmp_path, prefix="source3", num_files=2, size=DATA["tokens_per_file"]
        ),
    }

    source_configs = [
        SourceMixtureConfig(
            source_name="1",
            target_ratio=0.33,
            paths=source_paths["1"],
        ),
        SourceMixtureConfig(source_name="2", target_ratio=0.33, paths=source_paths["2"]),
        SourceMixtureConfig(
            source_name="3",
            target_ratio=0.34,
            paths=source_paths["3"],
        ),
    ]

    max_tokens = 5_000_000

    with TemporaryDirectory() as tmp_dir:
        config = SourceMixtureDatasetConfig(
            max_tokens=max_tokens,
            source_configs=source_configs,
            dtype=NumpyDatasetDType.uint32,
            output_dir=tmp_dir,
            dry_run=True,
        )

        with capsys.disabled():
            print("\n")
            mixture = config.build()
            assert isinstance(mixture, SourceMixtureDataset)


def test_source_mixture_config_validation():
    with pytest.raises(OLMoConfigurationError):
        SourceMixtureConfig(
            source_name="source1", target_ratio=1.2, paths=["/path/to/source1"]
        ).validate()

    with pytest.raises(OLMoConfigurationError):
        SourceMixtureConfig(
            source_name="source1",
            target_ratio=0.5,
            max_source_fraction=0.4,
            paths=["/path/to/source1"],
        ).validate()

    with pytest.raises(OLMoConfigurationError):
        SourceMixtureConfig(source_name="source1", target_ratio=0.5, paths=[]).validate()

    config = SourceMixtureConfig(
        source_name="source1", target_ratio=0.5, paths=["/path/to/source1"]
    )
    config.validate()


def test_dataset_mixture_config_validation():
    source_configs = [
        SourceMixtureConfig(source_name="source1", target_ratio=0.5, paths=["/path/to/source1"]),
        SourceMixtureConfig(source_name="source2", target_ratio=0.5, paths=["/path/to/source2"]),
    ]

    with TemporaryDirectory() as tmp_dir:
        config = SourceMixtureDatasetConfig(
            max_tokens=1000,
            source_configs=source_configs,
            dtype=NumpyDatasetDType.uint32,
            output_dir=tmp_dir,
        )
        config.validate()

        source_configs_invalid = [
            SourceMixtureConfig(
                source_name="source1", target_ratio=0.7, paths=["/path/to/source1"]
            ),
            SourceMixtureConfig(
                source_name="source2", target_ratio=0.5, paths=["/path/to/source2"]
            ),
        ]

        config_invalid = SourceMixtureDatasetConfig(
            max_tokens=1000,
            source_configs=source_configs_invalid,
            dtype=NumpyDatasetDType.uint32,
            output_dir=tmp_dir,
        )

        with pytest.raises(OLMoConfigurationError):
            config_invalid.validate()


def test_dataset_mixture_build(tmp_path: Path):
    source_paths = {
        "1": _make_mmaps(
            tmp_path=tmp_path, prefix="source1", num_files=2, size=DATA["tokens_per_file"]
        ),
        "2": _make_mmaps(
            tmp_path=tmp_path, prefix="source2", num_files=2, size=DATA["tokens_per_file"]
        ),
        "3": _make_mmaps(
            tmp_path=tmp_path, prefix="source3", num_files=2, size=DATA["tokens_per_file"]
        ),
    }

    source_configs = [
        SourceMixtureConfig(
            source_name="1",
            target_ratio=0.33,
            paths=source_paths["1"],
        ),
        SourceMixtureConfig(source_name="2", target_ratio=0.33, paths=source_paths["2"]),
        SourceMixtureConfig(
            source_name="3",
            target_ratio=0.34,
            paths=source_paths["3"],
        ),
    ]

    max_tokens = 5_000_000

    with TemporaryDirectory() as tmp_dir:
        config = SourceMixtureDatasetConfig(
            max_tokens=max_tokens,
            source_configs=source_configs,
            dtype=DATA["dtype"],
            output_dir=tmp_dir,
        )

        mixture = config.build()
        assert isinstance(mixture, SourceMixtureDataset)


def test_dataset_mixture_build_insufficient_source_data(tmp_path: Path):
    source_paths = {
        "1": _make_mmaps(
            tmp_path=tmp_path, prefix="source1", num_files=1, size=DATA["tokens_per_file"]
        ),
        "2": _make_mmaps(
            tmp_path=tmp_path, prefix="source2", num_files=2, size=DATA["tokens_per_file"]
        ),
        "3": _make_mmaps(
            tmp_path=tmp_path, prefix="source3", num_files=2, size=DATA["tokens_per_file"]
        ),
    }
    source_configs = [
        SourceMixtureConfig(
            source_name="1",
            target_ratio=0.5,
            paths=source_paths["1"],
        ),
        SourceMixtureConfig(source_name="2", target_ratio=0.25, paths=source_paths["2"]),
        SourceMixtureConfig(
            source_name="3",
            target_ratio=0.25,
            paths=source_paths["3"],
        ),
    ]

    max_tokens = 5_000_000

    with TemporaryDirectory() as tmp_dir:
        config = SourceMixtureDatasetConfig(
            max_tokens=max_tokens,
            source_configs=source_configs,
            dtype=DATA["dtype"],
            output_dir=tmp_dir,
        )

        # Should raise exception because the target ratio for source 1 @50% (2.5M) is infeasible without repetition (default max_repetition_ratio=1)
        with pytest.raises(OLMoConfigurationError):
            config.build()


def test_dataset_mixture_build_with_repetition(tmp_path: Path):
    """
    Test building a dataset with repetition of a source.

    Source 1 has a target ratio of 90% and a max repetition ratio of 4.0, so it should be possible to meet the target of 3600 tokens with 1 file of 1000 tokens repeated 4 times.
    """
    source_paths = {
        "1": _make_mmaps(
            tmp_path=tmp_path, prefix="source1", num_files=1, size=DATA["tokens_per_file"]
        ),
        "2": _make_mmaps(
            tmp_path=tmp_path, prefix="source2", num_files=2, size=DATA["tokens_per_file"]
        ),
        "3": _make_mmaps(
            tmp_path=tmp_path, prefix="source3", num_files=2, size=DATA["tokens_per_file"]
        ),
    }

    source_configs = [
        SourceMixtureConfig(
            source_name="1",
            target_ratio=0.5,
            max_repetion_ratio=3.0,  # Allow 3x repetition of source1 so that we can meet the target of 2.5M
            paths=source_paths["1"],
        ),
        SourceMixtureConfig(source_name="2", target_ratio=0.25, paths=source_paths["2"]),
        SourceMixtureConfig(
            source_name="3",
            target_ratio=0.25,
            paths=source_paths["3"],
        ),
    ]

    max_tokens = 5_000_000

    with TemporaryDirectory() as tmp_dir:
        config = SourceMixtureDatasetConfig(
            max_tokens=max_tokens,
            source_configs=source_configs,
            dtype=DATA["dtype"],
            output_dir=tmp_dir,
        )

        mixture = config.build()
        assert isinstance(mixture, SourceMixtureDataset)


def test_dataset_mixture_build_insufficient_source_max_fraction(tmp_path: Path):
    source_paths = {
        "1": _make_mmaps(
            tmp_path=tmp_path, prefix="source1", num_files=1, size=DATA["tokens_per_file"]
        ),
        "2": _make_mmaps(
            tmp_path=tmp_path, prefix="source2", num_files=2, size=DATA["tokens_per_file"]
        ),
        "3": _make_mmaps(
            tmp_path=tmp_path, prefix="source3", num_files=2, size=DATA["tokens_per_file"]
        ),
    }
    source_configs = [
        SourceMixtureConfig(
            source_name="1",
            target_ratio=0.25,
            paths=source_paths["1"],
            max_source_fraction=0.10,  # Allow only 10% of source1 to be used (population is 1M tokens)
        ),
        SourceMixtureConfig(
            source_name="2",
            target_ratio=0.25,
            paths=source_paths["2"],
        ),
        SourceMixtureConfig(
            source_name="3",
            target_ratio=0.5,
            paths=source_paths["3"],
        ),
    ]

    # 5 source files * 1_000_000 tokens per file
    max_tokens = len(list(chain(*source_paths.values()))) * DATA["tokens_per_file"]

    with TemporaryDirectory() as tmp_dir:
        config = SourceMixtureDatasetConfig(
            max_tokens=max_tokens,
            source_configs=source_configs,
            dtype=DATA["dtype"],
            output_dir=tmp_dir,
        )

        # Should raise exception because the target ratio for source 1 is infeasible because
        # we limit usage to 10% of the source
        with pytest.raises(OLMoConfigurationError):
            config.build()


def test_dataset_mixture_build_expected_files(tmp_path: Path):
    source_paths = {
        "1": _make_mmaps(
            tmp_path=tmp_path, prefix="source1", num_files=1, size=DATA["tokens_per_file"]
        ),
        "2": _make_mmaps(
            tmp_path=tmp_path, prefix="source2", num_files=2, size=DATA["tokens_per_file"]
        ),
        "3": _make_mmaps(
            tmp_path=tmp_path, prefix="source3", num_files=2, size=DATA["tokens_per_file"]
        ),
    }
    source_configs = [
        SourceMixtureConfig(
            source_name="1",
            target_ratio=0.10,
            paths=source_paths["1"],
        ),
        SourceMixtureConfig(
            source_name="2",
            target_ratio=0.40,
            paths=source_paths["2"],
        ),
        SourceMixtureConfig(
            source_name="3",
            target_ratio=0.5,
            paths=source_paths["3"],
        ),
    ]

    max_tokens = 10 * 1000

    with TemporaryDirectory() as tmp_dir:
        config = SourceMixtureDatasetConfig(
            max_tokens=max_tokens,
            source_configs=source_configs,
            dtype=DATA["dtype"],
            output_dir=tmp_dir,
        )

        mixture = config.build()
        assert isinstance(mixture, SourceMixtureDataset)

        out_tokens = []

        for source in mixture.sources:
            for path in source.paths:
                out_tokens.extend(
                    load_array_slice(
                        path=path,
                        start_idx=0,
                        end_idx=DATA["tokens_per_file"],
                        dtype=DATA["dtype"].as_np_dtype(),
                    )
                )

        assert len(out_tokens) == max_tokens


def test_dataset_mixture_render_table(tmp_path: Path, capsys):
    source_paths = {
        "1": _make_mmaps(
            tmp_path=tmp_path, prefix="source1", num_files=1, size=DATA["tokens_per_file"]
        ),
        "2": _make_mmaps(
            tmp_path=tmp_path, prefix="source2", num_files=2, size=DATA["tokens_per_file"]
        ),
        "3": _make_mmaps(
            tmp_path=tmp_path, prefix="source3", num_files=2, size=DATA["tokens_per_file"]
        ),
    }
    source_configs = [
        SourceMixtureConfig(
            source_name="1",
            target_ratio=0.10,
            paths=source_paths["1"],
        ),
        SourceMixtureConfig(
            source_name="2",
            target_ratio=0.40,
            paths=source_paths["2"],
        ),
        SourceMixtureConfig(
            source_name="3",
            target_ratio=0.5,
            paths=source_paths["3"],
        ),
    ]

    max_tokens = 10 * 1000

    with TemporaryDirectory() as tmp_dir:
        config = SourceMixtureDatasetConfig(
            max_tokens=max_tokens,
            source_configs=source_configs,
            dtype=DATA["dtype"],
            output_dir=tmp_dir,
        )

        with capsys.disabled():
            print("\n")
            mixture = config.build()
            assert isinstance(mixture, SourceMixtureDataset)
