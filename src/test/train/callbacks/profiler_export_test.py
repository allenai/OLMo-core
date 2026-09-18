"""Tests for ``ProfilerCallback.export_chrome_trace``.

The chrome-trace export gzips the whole trace on the training process and has hung a
Molmo2 Stage-2 run outright. These pin that the escape hatch actually skips it, and --
more importantly -- that the summary tables are still produced when it does, since those
are the whole point of running the profiler.
"""

from unittest.mock import MagicMock

from olmo_core.train.callbacks import ProfilerCallback


def _callback(*, export: bool) -> ProfilerCallback:
    cb = ProfilerCallback(export_chrome_trace=export)
    prof = MagicMock()
    prof.step_num = 23
    prof.key_averages.return_value.table.return_value = "TABLE"
    cb._profiler = prof
    cb.trainer = MagicMock()
    return cb


def test_export_chrome_trace_defaults_to_true():
    assert ProfilerCallback().export_chrome_trace is True


def test_trace_is_not_written_when_disabled():
    cb = _callback(export=False)
    cb._on_trace_ready(cb._profiler)
    cb._profiler.export_chrome_trace.assert_not_called()
    # The failure mode this guards against is a *silent* skip that also drops the
    # summaries, which would make the escape hatch useless.
    assert cb._profiler.key_averages.call_count == 2


def test_trace_is_written_when_enabled():
    cb = _callback(export=True)
    cb.trainer.work_dir.__truediv__.return_value = MagicMock()
    cb._on_trace_ready(cb._profiler)
    cb._profiler.export_chrome_trace.assert_called_once()
