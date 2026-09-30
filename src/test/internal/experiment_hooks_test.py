"""
Tests for the small extension points in :mod:`olmo_core.internal.experiment` that non-transformer
experiments (e.g. multimodal recipes) rely on: CLI parsing as a function, and a generic dataset
branch in the data-loader builder. The standard numpy and composable branches are unchanged.
"""

from dataclasses import dataclass
from typing import Any, Optional

import pytest

from olmo_core.config import Config
from olmo_core.internal.experiment import SubCmd, _build_data_loader, parse_cli_args


def test_parse_cli_args_reads_argv(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["train.py", "dry_run", "run01", "ai2/neptune", "--launch.num_nodes=2"]
    )
    context = parse_cli_args()
    assert context.script == "train.py"
    assert context.cmd == SubCmd.dry_run
    assert context.run_name == "run01"
    assert context.cluster == "ai2/neptune"
    assert context.overrides == ["--launch.num_nodes=2"]


def test_parse_cli_args_exits_on_unknown_subcommand(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "bogus", "run01", "ai2/neptune"])
    with pytest.raises(SystemExit):
        parse_cli_args()


@dataclass
class _SelfContainedDataset(Config):
    size: int = 3

    def build(self):
        return list(range(self.size))


@dataclass
class _Loader(Config):
    built_with: Optional[Any] = None

    def build(self, dataset, dp_process_group=None):
        return ("loader", dataset, dp_process_group)


@dataclass
class _Experiment(Config):
    dataset: Any
    data_loader: Any


def test_generic_dataset_branch_builds_self_contained_datasets(monkeypatch):
    monkeypatch.setattr("olmo_core.internal.experiment.barrier", lambda: None)
    config = _Experiment(dataset=_SelfContainedDataset(size=2), data_loader=_Loader())
    assert _build_data_loader(config) == ("loader", [0, 1], None)  # type: ignore[arg-type]


def test_generic_dataset_branch_requires_a_build_method():
    @dataclass
    class NoBuild(Config):
        pass

    config = _Experiment(dataset=NoBuild(), data_loader=_Loader())
    with pytest.raises(NotImplementedError):
        _build_data_loader(config)  # type: ignore[arg-type]
