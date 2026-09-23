from dataclasses import dataclass

from olmo_core.train.callbacks.profiler import _summarize_distributed_events


@dataclass
class _Event:
    name: str
    cpu_time_total: float
    device_time_total: float
    input_shapes: object = None


def test_summarize_distributed_events() -> None:
    events = [
        _Event("c10d::allreduce_", 10.0, 0.0, [[64]]),
        _Event("c10d::allreduce_", 30.0, 0.0, [[64]]),
        _Event("ncclDevKernel_AllReduce_Sum_f32", 2.0, 8.0),
        _Event("cudaDeviceSynchronize", 7.0, 0.0),
        _Event("aten::linear", 100.0, 90.0),
    ]

    summary = _summarize_distributed_events(events)

    assert [event["category"] for event in summary] == [
        "collective",
        "collective",
        "synchronization",
    ]
    c10d = next(event for event in summary if event["name"] == "c10d::allreduce_")
    assert c10d["count"] == 2
    assert c10d["cpu_us"] == {
        "mean": 20.0,
        "p50": 20.0,
        "p95": 29.0,
        "total": 40.0,
    }
    assert c10d["device_us"]["total"] == 0.0


def test_summarize_distributed_events_returns_empty_for_unrelated_ops() -> None:
    assert _summarize_distributed_events([_Event("aten::linear", 10.0, 9.0)]) == []
