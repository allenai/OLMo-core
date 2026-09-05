"""
Peak-memory attribution from a ``torch.cuda.memory._snapshot()``: replay the allocator trace to
the moment of peak allocated memory and print the live blocks at that moment grouped by their
allocation site (first frame inside ``olmo_core``, else the innermost frame).
"""

from __future__ import annotations

import collections
from typing import Any, Dict


def _site(frames) -> str:
    for f in frames or []:
        fn = f.get("filename", "")
        if "olmo_core" in fn or "memexpress" in fn:
            return f"{fn.split('/')[-1]}:{f.get('line')} {f.get('name')}"
    if frames:
        f = frames[0]
        return f"{f.get('filename', '?').split('/')[-1]}:{f.get('line')} {f.get('name')}"
    return "?"


def summarize_peak(snap: Dict[str, Any], top: int = 16) -> None:
    """Print the live set at the peak of the recorded trace, grouped by site."""
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank != 0:
        return
    live: Dict[int, tuple] = {}
    cur = peak = 0
    peak_live: Dict[int, tuple] = {}
    for trace in snap.get("device_traces", []):
        for ev in trace:
            act, addr, size = ev.get("action"), ev.get("addr"), ev.get("size", 0)
            if act == "alloc":
                live[addr] = (size, _site(ev.get("frames")))
                cur += size
                if cur > peak:
                    peak = cur
                    peak_live = dict(live)
            elif act in ("free_completed",):
                if addr in live:
                    cur -= live.pop(addr)[0]
        break  # device 0 only
    by = collections.Counter()
    for size, site in peak_live.values():
        by[site] += size
    print(f"[mem-snapshot] peak live {peak / 2**30:.2f} GB in {len(peak_live)} blocks; top sites:", flush=True)
    for site, size in by.most_common(top):
        print(f"[mem-snapshot]   {size / 2**30:6.2f} GB  {site}", flush=True)
