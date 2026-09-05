"""Check per-device overlap accounting and memory clipping without CUDA."""

import sqlite3

from examples.olmo_ddp.olmoe3_nsys_analyze import interval_union, summarize_timeline


def test_interval_union():
    assert interval_union([]) == 0
    assert interval_union([(1, 4), (2, 3), (4, 8), (10, 11), (20, 10)]) == 8


def test_timeline_overlap(tmp_path):
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE StringIds (id INTEGER, value TEXT);"
            "INSERT INTO StringIds VALUES (1, 'ncclAllReduce'), (2, 'gemm'), (3, 'cudaEventQuery');"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INTEGER, start INTEGER, end INTEGER, demangledName INTEGER);"
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 0, 4000000, 2), (0, 2000000, 6000000, 1), (0, 8000000, 10000000, 2), (1, 0, 1000000, 2);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (deviceId INTEGER, start INTEGER, end INTEGER);"
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (0, -1000000, 1000000), (0, 5000000, 7000000), (0, 9500000, 12000000);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, nameId INTEGER);"
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0, 2000, 3), (5000, 6000, 3);"
        )
    summary = summarize_timeline(path)
    first, second = summary["devices"]
    assert first["kernel_span_ms"] == 10
    assert first["kernel_busy_union_ms"] == 8
    assert first["collective_union_ms"] == 4
    assert first["collective_without_other_kernel_ms"] == 2
    assert first["collective_overlapping_other_kernel_ms"] == 2
    assert first["kernel_and_memory_busy_union_ms"] == 9
    assert first["no_recorded_gpu_operation_in_kernel_span_ms"] == 1
    assert second["kernel_busy_union_ms"] == 1
    assert summary["cpu_api_inclusive_nonadditive"][0]["inclusive_sum_ms"] == 0.003


def test_empty_timeline(tmp_path):
    path = tmp_path / "empty.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE unrelated (value INTEGER)")
    assert summarize_timeline(path)["devices"] == []
