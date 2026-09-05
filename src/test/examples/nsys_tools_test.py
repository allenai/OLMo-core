"""Reject empty reports and preserve distinct kernel/copy/NCCL activity counts."""

import sqlite3

from examples.olmo_ddp.olmoe3_nsys_tools import summarize_sqlite


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
