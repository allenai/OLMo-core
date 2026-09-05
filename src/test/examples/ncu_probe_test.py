"""Parse real counter rows without treating a report file alone as success."""

from examples.olmo_ddp.olmoe3_ncu_probe import parse_metrics


def test_parse_counter_rows():
    text = 'Profiler preamble\n"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n"0","kernel","gpu__time_duration.sum","usecond","1,200"\n"0","kernel","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","42.5"\n'
    metrics = parse_metrics(text)
    assert len(metrics) == 2
    assert metrics[0]["value"] == "1,200"
    assert metrics[1]["unit"] == "%"


def test_missing_counter_report():
    assert parse_metrics("ERR_NVGPUCTRPERM: profiling is not permitted") == []
