import logging
import math
from itertools import chain
from pathlib import Path

import pytest

from olmo_core.data import NumpyDatasetDType
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDataset,
    SourceMixtureDatasetConfig,
)
from olmo_core.exceptions import OLMoConfigurationError

from .utils import mk_mmaps


def test_source_mixture_config(tmp_path: Path, caplog, capsys):
    source_paths = {
        "1": mk_mmaps(tmp_path=tmp_path, prefix="source1", num_files=2, size=1_000_000),
        "2": mk_mmaps(tmp_path=tmp_path, prefix="source2", num_files=2, size=1_000_000),
        "3": mk_mmaps(tmp_path=tmp_path, prefix="source3", num_files=2, size=1_000_000),
    }

    source_configs = [
        SourceMixtureConfig(
            source_name="1",
            target_ratio=0.33333,
            paths=[str(i[0]) for i in source_paths["1"]],
        ),
        SourceMixtureConfig(
            source_name="2", target_ratio=0.33333, paths=[str(i[0]) for i in source_paths["2"]]
        ),
        SourceMixtureConfig(
            source_name="3",
            target_ratio=0.33333,
            paths=[str(i[0]) for i in source_paths["3"]],
        ),
    ]

    max_tokens = 5_000_000
    sequence_length = 1024
    global_batch_size = 1024 * 32

    config = SourceMixtureDatasetConfig(
        max_tokens=max_tokens,
        source_configs=source_configs,
        dtype=NumpyDatasetDType.uint32,
        sequence_length=sequence_length,
        quiet=True,
        render_tables=True,
        global_batch_size=global_batch_size,
    )

    # NOTE: We need to disable capsys so we can override log capture as
    # we want to see the rendered tables in the case
    with capsys.disabled(), caplog.at_level(logging.DEBUG):
        config.validate()
        mixture = config.build()
        assert isinstance(mixture, SourceMixtureDataset)

        requested_instances = math.ceil(max_tokens / global_batch_size) * int(
            global_batch_size / sequence_length
        )
        assert (
            sum([tokens // sequence_length for _, tokens in mixture.to_index().items()])
            == requested_instances
        ), f"Expected {requested_instances} instances, but got {sum([tokens // sequence_length for _, tokens in mixture.to_index().items()])}"
        #  print(caplog.text)  # uncomment if you want to see the table


def test_source_mixture_config_validation():
    with pytest.raises(OLMoConfigurationError):
        SourceMixtureConfig(
            source_name="source1", target_ratio=1.2, paths=["/path/to/source1"]
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

    config = SourceMixtureDatasetConfig(
        max_tokens=1000,
        source_configs=source_configs,
        dtype=NumpyDatasetDType.uint32,
        sequence_length=1024,
        quiet=True,
        render_tables=False,
        global_batch_size=1024 * 32,
    )
    config.validate()

    source_configs_invalid = [
        SourceMixtureConfig(source_name="source1", target_ratio=0.7, paths=["/path/to/source1"]),
        SourceMixtureConfig(source_name="source2", target_ratio=0.5, paths=["/path/to/source2"]),
    ]

    config_invalid = SourceMixtureDatasetConfig(
        max_tokens=1000,
        source_configs=source_configs_invalid,
        dtype=NumpyDatasetDType.uint32,
        sequence_length=1024,
        quiet=True,
        render_tables=False,
        global_batch_size=1024 * 32,
    )

    with pytest.raises(OLMoConfigurationError):
        config_invalid.validate()


def test_dataset_mixture_build(tmp_path: Path):
    source_paths = {
        "1": mk_mmaps(tmp_path=tmp_path, prefix="source1", num_files=2, size=1_000_000),
        "2": mk_mmaps(tmp_path=tmp_path, prefix="source2", num_files=2, size=1_000_000),
        "3": mk_mmaps(tmp_path=tmp_path, prefix="source3", num_files=2, size=1_000_000),
    }

    source_configs = [
        SourceMixtureConfig(
            source_name="1",
            target_ratio=0.33,
            paths=[str(i[0]) for i in source_paths["1"]],
        ),
        SourceMixtureConfig(
            source_name="2", target_ratio=0.33, paths=[str(i[0]) for i in source_paths["2"]]
        ),
        SourceMixtureConfig(
            source_name="3",
            target_ratio=0.34,
            paths=[str(i[0]) for i in source_paths["3"]],
        ),
    ]

    max_tokens = 5_000_000
    sequence_length = 1024
    global_batch_size = sequence_length * 32

    config = SourceMixtureDatasetConfig(
        max_tokens=max_tokens,
        source_configs=source_configs,
        dtype=NumpyDatasetDType.uint32,
        sequence_length=sequence_length,
        quiet=True,
        render_tables=False,
        global_batch_size=global_batch_size,
    )

    mixture = config.build()
    assert isinstance(mixture, SourceMixtureDataset)

    requested_instances = math.ceil(max_tokens / global_batch_size) * int(
        global_batch_size / sequence_length
    )
    assert (
        sum([tokens // sequence_length for _, tokens in mixture.to_index().items()])
        == requested_instances
    ), f"Expected {requested_instances} instances, but got {sum([tokens // sequence_length for _, tokens in mixture.to_index().items()])}"


def test_dataset_mixture_build_insufficient_source_data(tmp_path: Path):
    source_paths = {
        "1": mk_mmaps(tmp_path=tmp_path, prefix="source1", num_files=1, size=1_000_000),
        "2": mk_mmaps(tmp_path=tmp_path, prefix="source2", num_files=2, size=1_000_000),
        "3": mk_mmaps(tmp_path=tmp_path, prefix="source3", num_files=2, size=1_000_000),
    }
    source_configs = [
        SourceMixtureConfig(
            source_name="1",
            target_ratio=0.5,
            paths=[str(i[0]) for i in source_paths["1"]],
        ),
        SourceMixtureConfig(
            source_name="2", target_ratio=0.25, paths=[str(i[0]) for i in source_paths["2"]]
        ),
        SourceMixtureConfig(
            source_name="3",
            target_ratio=0.25,
            paths=[str(i[0]) for i in source_paths["3"]],
        ),
    ]

    max_tokens = 5_000_000

    config = SourceMixtureDatasetConfig(
        max_tokens=max_tokens,
        source_configs=source_configs,
        dtype=NumpyDatasetDType.uint32,
        sequence_length=1024,
        quiet=True,
        render_tables=False,
        global_batch_size=1024 * 32,
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
        "1": mk_mmaps(tmp_path=tmp_path, prefix="source1", num_files=1, size=1_000_000),
        "2": mk_mmaps(tmp_path=tmp_path, prefix="source2", num_files=2, size=1_000_000),
        "3": mk_mmaps(tmp_path=tmp_path, prefix="source3", num_files=2, size=1_000_000),
    }

    source_configs = [
        SourceMixtureConfig(
            source_name="1",
            target_ratio=0.5,
            max_repetition_ratio=3.0,  # Allow 3x repetition of source1 so that we can meet the target of 2.5M
            paths=[str(i[0]) for i in source_paths["1"]],
        ),
        SourceMixtureConfig(
            source_name="2", target_ratio=0.25, paths=[str(i[0]) for i in source_paths["2"]]
        ),
        SourceMixtureConfig(
            source_name="3",
            target_ratio=0.25,
            paths=[str(i[0]) for i in source_paths["3"]],
        ),
    ]

    max_tokens = 5_000_000
    sequence_length = 1024
    global_batch_size = sequence_length * 32

    config = SourceMixtureDatasetConfig(
        max_tokens=max_tokens,
        source_configs=source_configs,
        dtype=NumpyDatasetDType.uint32,
        sequence_length=sequence_length,
        quiet=True,
        render_tables=False,
        global_batch_size=global_batch_size,
    )

    mixture = config.build()
    sources = [source for source in mixture.sources]
    all_paths = []
    for source in sources:
        all_paths.extend([item for item in source.path_tokens])

    total_tokens = sum([item.tokens for item in all_paths])
    assert isinstance(mixture, SourceMixtureDataset)

    requested_instances = math.ceil(max_tokens / global_batch_size) * int(
        global_batch_size / sequence_length
    )
    assert (
        sum([tokens // sequence_length for _, tokens in mixture.to_index().items()])
        == requested_instances
    ), f"Expected {requested_instances} instances, but got {sum([tokens // sequence_length for _, tokens in mixture.to_index().items()])}"

    assert total_tokens == 5013504


def test_dataset_mixture_build_insufficient_source_max_fraction(tmp_path: Path):
    source_paths = {
        "1": mk_mmaps(tmp_path=tmp_path, prefix="source1", num_files=1, size=1_000_000),
        "2": mk_mmaps(tmp_path=tmp_path, prefix="source2", num_files=2, size=1_000_000),
        "3": mk_mmaps(tmp_path=tmp_path, prefix="source3", num_files=2, size=1_000_000),
    }
    source_configs = [
        SourceMixtureConfig(
            source_name="1",
            target_ratio=0.25,
            paths=[str(i[0]) for i in source_paths["1"]],
            max_source_fraction=0.10,  # Allow only 10% of source1 to be used (population is 1M tokens)
        ),
        SourceMixtureConfig(
            source_name="2",
            target_ratio=0.25,
            paths=[str(i[0]) for i in source_paths["2"]],
        ),
        SourceMixtureConfig(
            source_name="3",
            target_ratio=0.5,
            paths=[str(i[0]) for i in source_paths["3"]],
        ),
    ]

    # 5 source files * 1_000_000 tokens per file
    max_tokens = len(list(chain(*source_paths.values()))) * 1_000_000

    config = SourceMixtureDatasetConfig(
        max_tokens=max_tokens,
        source_configs=source_configs,
        dtype=NumpyDatasetDType.uint32,
        sequence_length=1024,
        quiet=True,
        render_tables=False,
        global_batch_size=1024 * 32,
    )

    # Should raise exception because the target ratio for source 1 is infeasible because
    # we limit usage to 10% of the source
    with pytest.raises(OLMoConfigurationError):
        config.build()


def test_dataset_mixture_build_duplicate_paths(tmp_path: Path):
    sources = {
        "1": mk_mmaps(tmp_path=tmp_path, prefix="source1", num_files=1, size=500_000),
        "2": mk_mmaps(tmp_path=tmp_path, prefix="source2", num_files=2, size=1_000_000),
        "3": mk_mmaps(tmp_path=tmp_path, prefix="source3", num_files=2, size=1_000_000),
    }

    source_configs = [
        SourceMixtureConfig(
            source_name="1",
            target_ratio=0.33,  # 990k tokens
            max_repetition_ratio=2.0,
            paths=[
                str(sources["1"][0][0]),
                str(sources["1"][0][0]),
            ],  # Duplicate the 1 path for source 1
        ),
        SourceMixtureConfig(
            source_name="2", target_ratio=0.33, paths=[str(i[0]) for i in sources["2"]]
        ),
        SourceMixtureConfig(
            source_name="3",
            target_ratio=0.34,
            paths=[str(i[0]) for i in sources["3"]],
        ),
    ]

    max_tokens = 3_000_000
    sequence_length = 1024
    global_batch_size = sequence_length * 32

    config = SourceMixtureDatasetConfig(
        max_tokens=max_tokens,
        source_configs=source_configs,
        dtype=NumpyDatasetDType.uint32,
        sequence_length=sequence_length,
        quiet=True,
        render_tables=False,
        global_batch_size=global_batch_size,
    )

    expected = [str(sources["1"][0][0])] + [str(item[0]) for item in list(chain(*sources.values()))]
    mixture = config.build()
    index = mixture.to_index()
    paths = mixture.to_paths()
    assert paths == expected
    assert len(index) == 6, "Expected 6 unique paths in the index, but got {}".format(len(index))
    assert isinstance(mixture, SourceMixtureDataset)
    assert len(mixture.sources) == 3, "Expected 3 sources, but got {}".format(len(mixture.sources))
    requested_instances = math.ceil(max_tokens / global_batch_size) * int(
        global_batch_size / sequence_length
    )
    assert (
        sum([tokens // sequence_length for _, tokens in mixture.to_index().items()])
        == requested_instances
    ), f"Expected {requested_instances} instances, but got {sum([tokens // sequence_length for _, tokens in mixture.to_index().items()])}"
