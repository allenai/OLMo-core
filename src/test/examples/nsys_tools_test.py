"""Reject empty reports and preserve distinct kernel/copy/NCCL activity counts."""

import sqlite3

import pytest

from examples.olmo_ddp.olmoe3_nsys_tools import NsysSettings, summarize_sqlite


def test_settings_preserve_historical_capture():
    settings = NsysSettings.from_env({})
    assert settings.ranks == tuple(range(64))
    assert settings.version == "installed"
    assert settings.autograd_nvtx is True
    assert settings.clean_windows(100) == [[31, 70], [81, 100]]


def test_selective_short_capture():
    settings = NsysSettings.from_env(
        {
            "OLMOE3_NSYS_VERSION": "2026.4.1",
            "OLMOE3_NSYS_RANKS": "0,8,16,24,32,40,48,56",
            "OLMOE3_NSYS_START": "36",
            "OLMOE3_NSYS_END": "37",
            "OLMOE3_NSYS_AUTOGRAD_NVTX": "0",
        }
    )
    assert settings.ranks == tuple(range(0, 64, 8))
    assert settings.clean_windows(60) == [[31, 35], [45, 60]]
    assert settings.autograd_nvtx is False
    with pytest.raises(ValueError, match="ends before capture"):
        settings.clean_windows(35)


@pytest.mark.parametrize("ranks", ["", "0,0", "-1", "64", "0,8,64"])
def test_invalid_capture_ranks(ranks):
    with pytest.raises(ValueError):
        NsysSettings.from_env({"OLMOE3_NSYS_RANKS": ranks})


def test_invalid_nvtx_setting():
    with pytest.raises(ValueError, match="must be 0 or 1"):
        NsysSettings.from_env({"OLMOE3_NSYS_AUTOGRAD_NVTX": "maybe"})


def test_empty_report(tmp_path):
    path = tmp_path / "empty.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE unrelated (value INTEGER)")
    summary = summarize_sqlite(path)
    assert summary["kernel_count"] == summary["memcpy_count"] == summary["nccl_kernel_count"] == 0


def test_cuda_report(tmp_path):
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE StringIds (id INTEGER, value TEXT);"
            "INSERT INTO StringIds VALUES (1, 'ncclDevKernel_AllReduce'), (2, 'matmul');"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (demangledName INTEGER, start INTEGER, end INTEGER);"
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 0, 2000000), (2, 1, 1000001), (2, 2, 1000002);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER);"
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (0, 1000);"
        )
    summary = summarize_sqlite(path)
    assert summary["kernel_count"] == 3
    assert summary["memcpy_count"] == 1
    assert summary["nccl_kernel_count"] == 1
    assert sum(k["duration_ms_sum"] for k in summary["top_kernels_nonadditive"]) == 4
