"""
Early, flushed progress lines for long loops (data builders, eval drivers).

A loop that prints only every N items is silent exactly when it matters: a livelock or a stall
produces zero output and reads as "slow work". :class:`Progress` prints at items 1, 2, 5, 10, 25 and
50, then every ``every`` items, each line with elapsed time, rate and ETA, always with
``flush=True``::

    prog = Progress(len(rows), "tokenize", every=500)
    for row in rows:
        ...
        prog.update()
    prog.done()

which prints ``[tokenize] 1/12,000 (0.0%) elapsed 0.4s rate 2.5/s eta 1h20m`` and so on.
"""

import sys
import time
from typing import Callable, Optional, TextIO

#: Item counts that always print, before the periodic ``every``.
EARLY_MARKS = (1, 2, 5, 10, 25, 50)


def format_duration(seconds: float) -> str:
    """
    Compact human duration: ``0.4s``, ``42s``, ``3m05s``, ``1h20m``.

    :param seconds: Duration in seconds.
    """
    if seconds < 10:
        return f"{seconds:.1f}s"
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class Progress:
    """
    Progress printer for a loop of (optionally) known length.

    :param total: Number of items, or ``None`` if unknown (no percentage / ETA then).
    :param desc: Label printed in brackets at the start of each line.
    :param every: Print every this many items after the early marks.
    :param stream: Where to print (default ``sys.stdout``).
    :param clock: Time source (tests).
    """

    def __init__(
        self,
        total: Optional[int],
        desc: str,
        *,
        every: int = 100,
        stream: Optional[TextIO] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.total = total
        self.desc = desc
        self.every = max(1, int(every))
        self.stream = stream
        self.clock = clock
        self.count = 0
        self.start = clock()
        self._last_printed = 0

    def should_print(self, count: int) -> bool:
        """Whether item number ``count`` gets a line (early marks, every N, and the last item)."""
        return (
            count in EARLY_MARKS
            or count % self.every == 0
            or (self.total is not None and count == self.total)
        )

    def line(self, extra: str = "") -> str:
        """The progress line for the current count."""
        elapsed = self.clock() - self.start
        rate = self.count / elapsed if elapsed > 0 else 0.0
        if self.total:
            pct = 100.0 * self.count / self.total
            head = f"{self.count:,}/{self.total:,} ({pct:.1f}%)"
        else:
            head = f"{self.count:,}"
        parts = [f"[{self.desc}] {head}", f"elapsed {format_duration(elapsed)}"]
        parts.append(f"rate {rate:.3g}/s")
        if self.total and rate > 0 and self.count < self.total:
            parts.append(f"eta {format_duration((self.total - self.count) / rate)}")
        if extra:
            parts.append(extra)
        return " ".join(parts)

    def update(self, n: int = 1, extra: str = "") -> None:
        """
        Advance by ``n`` items and print if a mark was crossed.

        :param n: Items completed since the last call.
        :param extra: Free text appended to a printed line (e.g. a running metric).
        """
        before = self.count
        self.count += n
        crossed = (
            any(before < m <= self.count for m in EARLY_MARKS)
            or self.count // self.every > before // self.every
            or (self.total is not None and before < self.total <= self.count)
        )
        if crossed:
            self._emit(extra)

    def done(self, extra: str = "") -> None:
        """Print a final line unless the last update already printed this count."""
        if self._last_printed != self.count or self.count == 0:
            self._emit(extra)

    def _emit(self, extra: str) -> None:
        print(self.line(extra), file=self.stream or sys.stdout, flush=True)
        self._last_printed = self.count
