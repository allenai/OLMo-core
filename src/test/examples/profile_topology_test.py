"""Checks specific to the full-node B300 profiling launcher."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[2] / "examples/olmo_ddp/olmoe3_profile_topology.py"
_SPEC = importlib.util.spec_from_file_location("profile_topology", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _matrix(disconnected=None):
    rows = []
    for source in range(8):
        links = [
            "X" if source == target else "SYS" if disconnected in (source, target) else "NV18"
            for target in range(8)
        ]
        rows.append(f"GPU{source}\t" + "\t".join(links) + "\tPXB\t0-31\t0")
    return "\n".join(rows)


def test_healthy_topology():
    result = _MODULE.validate_topology(_matrix())
    assert result["all_pairs_nvlink"]
    assert result["gpus"] == 8


def test_disconnected_gpu_fails():
    with pytest.raises(ValueError, match="Degraded local GPU topology"):
        _MODULE.validate_topology(_matrix(disconnected=1))


def test_missing_gpu_fails():
    with pytest.raises(ValueError, match="Expected GPU0"):
        _MODULE.validate_topology("\n".join(_matrix().splitlines()[:-1]))
