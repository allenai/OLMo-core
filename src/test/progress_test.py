import io

import pytest

from olmo_core.progress import Progress, format_duration


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _run(total, n_items, *, every=100, step_seconds=1.0, n=1):
    clock, out = FakeClock(), io.StringIO()
    prog = Progress(total, "job", every=every, stream=out, clock=clock)
    for _ in range(0, n_items, n):
        clock.t += step_seconds * n
        prog.update(n)
    return prog, out.getvalue().splitlines()


def _counts(lines):
    return [int(line.split()[1].split("/")[0].replace(",", "")) for line in lines]


def test_prints_early_then_every_n_then_last():
    _, lines = _run(450, 450, every=100)
    assert _counts(lines) == [1, 2, 5, 10, 25, 50, 100, 200, 300, 400, 450]


def test_first_item_is_visible_immediately():
    """A loop that stalls after one item must already have said so."""
    _, lines = _run(10_000, 1)
    assert len(lines) == 1 and lines[0].startswith("[job] 1/10,000")


def test_line_has_elapsed_rate_and_eta():
    _, lines = _run(100, 10, every=1000, step_seconds=2.0)
    last = lines[-1]
    assert "10/100 (10.0%)" in last
    assert "elapsed 20s" in last
    assert "rate 0.5/s" in last
    assert "eta 3m00s" in last


def test_unknown_total_has_no_eta():
    _, lines = _run(None, 5)
    assert lines[-1].startswith("[job] 5 ") and "eta" not in lines[-1]


def test_batched_updates_cross_marks():
    _, lines = _run(1000, 1000, every=300, n=64)
    counts = _counts(lines)
    assert counts[0] == 64  # crossed 1,2,5,10,25,50 in one update -> one line
    assert counts[-1] == 1024  # the last batch overshoots total and still prints


def test_done_prints_once():
    prog, lines = _run(7, 7)
    out = prog.stream
    prog.done("kept=7")
    assert out.getvalue().count("7/7") == 1  # the last update already printed it

    clock, out2 = FakeClock(), io.StringIO()
    p2 = Progress(None, "job", every=100, stream=out2, clock=clock)
    for _ in range(7):
        p2.update()
    p2.done("kept=7")
    assert out2.getvalue().splitlines()[-1].endswith("kept=7")


@pytest.mark.parametrize(
    "seconds, text", [(0.4, "0.4s"), (42, "42s"), (185, "3m05s"), (4800, "1h20m")]
)
def test_format_duration(seconds, text):
    assert format_duration(seconds) == text
