"""
Turn a ``run_ladder.sbatch`` log into the speed/quality frontier table.

Reads ONE ladder log (every arm ran inside it, so no comparison crosses a job boundary), pulls
each arm's per-step wall-clock and CE curve out of the console logger, and joins them to the
FLOP meter's ``flops.json`` and, if present, the eval JSON.

Step seconds come from ``throughput/device/BPS`` (SpeedMonitor's own timer, which skips the first
step) and are cross-checked against the wall-clock gaps between consecutive console-logger
timestamps; both are reported, because a large disagreement means the metric reduce fell behind
and the timestamps are no longer step boundaries.

    python research/kv_route/flexcompute_train/collect.py <ladder.log> [--warmup 5] [--json out.json]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import statistics
from typing import Any, Dict, List, Optional

ARM_RE = re.compile(r"^######## ARM=(\S+) RUN=(\S+) VARIANT=(\S+)")
# The router callbacks also write a plain console line every `log_every` steps, which is the only
# place the REALIZED operating point (as opposed to the run-average in flops.json) is recorded.
FFN_RE = re.compile(r"\[ffn-moe\] step (\d+): mean_cost=([\d.]+) .*target=([\d.]+)")
KV_RE = re.compile(r"\[kv-route\] step (\d+): keep ([\d.]+) \(target ([\d.]+)\)")
BSK_RE = re.compile(r"\[block-skip\] step (\d+): run ([\d.]+) \(target ([\d.]+)\)")
STEP_RE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\t\S+\t\S*console_logger.*\[step=(\d+)/"
)
METRIC_RE = re.compile(r"^    ([^=]+)=(.+)$")
#: Node-local on the training node. From anywhere else, point --root at
#: /net/<node>/data/prasann/flexcompute_train (read-only inspection; see CLAUDE.md on /net).
DEFAULT_ROOT = "/data/prasann/flexcompute_train"


def _num(s: str) -> Optional[float]:
    """Parse the console logger's formatted numbers ('1,093', '3.412P', '4.58E-06')."""
    s = s.strip().replace(",", "")
    mult = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15}.get(s[-1:], None)
    if mult is not None:
        s = s[:-1]
    try:
        v = float(s)
    except ValueError:
        return None
    return v * mult if mult else v


def parse_log(path: str) -> Dict[str, Dict[str, Any]]:
    """Split a ladder log into per-arm step records."""
    arms: Dict[str, Dict[str, Any]] = {}
    cur: Optional[Dict[str, Any]] = None
    pending: Optional[Dict[str, Any]] = None
    with open(path, errors="replace") as f:
        for line in f:
            m = ARM_RE.match(line)
            if m:
                cur = {
                    "arm": m.group(1),
                    "run": m.group(2),
                    "variant": m.group(3),
                    "steps": [],
                    "router": {},
                }
                arms[m.group(1)] = cur
                pending = None
                continue
            if cur is None:
                continue
            for key, rx in (("ffn", FFN_RE), ("kv", KV_RE), ("bskip", BSK_RE)):
                rm = rx.search(line)
                if rm:
                    # keep the LAST one seen = the end-of-run operating point
                    cur["router"][key] = (int(rm.group(1)), float(rm.group(2)), float(rm.group(3)))
            m = STEP_RE.match(line)
            if m:
                pending = {
                    "step": int(m.group(2)),
                    "t": dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp(),
                }
                cur["steps"].append(pending)
                continue
            if pending is not None:
                m = METRIC_RE.match(line.rstrip("\n"))
                if m:
                    v = _num(m.group(2))
                    if v is not None:
                        pending[m.group(1).strip()] = v
                elif line.strip():
                    pending = None
    return arms


def summarize(arm: Dict[str, Any], warmup: int, root: str) -> Dict[str, Any]:
    """Median/p10/p90 step seconds, CE curve, and the FLOP meter's own summary for one arm."""
    steps = arm["steps"]
    bps = [(s["step"], s["throughput/device/BPS"]) for s in steps if "throughput/device/BPS" in s]
    sec = [1.0 / b for st, b in bps if st > warmup and b > 0]
    # Independent cross-check: gaps between consecutive console timestamps.
    gaps = [
        steps[i]["t"] - steps[i - 1]["t"]
        for i in range(1, len(steps))
        if steps[i]["step"] == steps[i - 1]["step"] + 1 and steps[i]["step"] > warmup
    ]
    ce = [(s["step"], s["train/CE loss"]) for s in steps if "train/CE loss" in s]
    out: Dict[str, Any] = {
        "arm": arm["arm"],
        "run": arm["run"],
        "variant": arm["variant"],
        "job": arm.get("job", ""),
        "n_steps_logged": len(steps),
        "n_steps_timed": len(sec),
        "step_s_median": statistics.median(sec) if sec else None,
        "step_s_p10": _pct(sec, 10),
        "step_s_p90": _pct(sec, 90),
        "step_s_median_from_timestamps": statistics.median(gaps) if gaps else None,
        "ce_curve": ce,
        "ce_final": ce[-1][1] if ce else None,
        "ce_last5_mean": (statistics.mean([v for _, v in ce[-5:]]) if ce else None),
        # (last_logged_step, realized, target) per router -- the END operating point. flops.json's
        # *_frac fields are token-weighted RUN AVERAGES and are dragged up by the keep-all warmup.
        "router_end": arm["router"],
    }
    fj = os.path.join(root, "ckpts", arm["run"], "flops.json")
    if os.path.exists(fj):
        with open(fj) as f:
            out["flops"] = json.load(f)
    ej = os.path.join(root, "eval", f"{arm['run']}_contra_iid.json")
    if os.path.exists(ej):
        with open(ej) as f:
            out["eval"] = json.load(f)
    return out


def _pct(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[i]


def _eval_cell(ev: Optional[Dict[str, Any]]) -> str:
    """`contradiction_<rung>` scores from the ladder eval JSON, with their binomial SE.

    Every rung is 500 examples (`eval_size=500`), so the SE at f1 0.7 is +-0.020 and at 0.9 is
    +-0.013 -- quote differences against that, not against the third decimal.
    """
    if not ev:
        return "-"
    rungs = [(k.split("_", 1)[1], v) for k, v in ev.items() if k.startswith("contradiction_")]
    if not rungs:
        return "-"
    order = {"2k": 0, "4k": 1, "8k": 2, "16k": 3, "32k": 4}
    rungs.sort(key=lambda kv: order.get(kv[0], 99))
    return " / ".join("-" if v is None else f"{v:.3f}" for _, v in rungs)


def table(rows: List[Dict[str, Any]], control: str) -> str:
    """Markdown frontier table, everything measured relative to the dense control."""
    base = next((r for r in rows if r["arm"] == control), None)
    b_t = base and base["step_s_median"]
    b_f = base and (base.get("flops") or {}).get("actual_pflops")
    hdr = (
        "| arm | job | step s (median) | p10 | p90 | speedup | actual PF | FLOP cut | conversion "
        "| FFN cost | block keep | KV keep | CE final | contra_iid f1 (2k/8k) |"
    )
    sep = "|" + "---|" * 14
    out = [hdr, sep]
    for r in rows:
        fl = r.get("flops") or {}
        sp = (b_t / r["step_s_median"]) if (b_t and r["step_s_median"]) else None
        fc = (b_f / fl["actual_pflops"]) if (b_f and fl.get("actual_pflops")) else None
        # conversion = fraction of the FLOP saving that became wall-clock saving
        conv = None
        if sp and fc and fc > 1.0 and sp >= 1.0:
            conv = (1 - 1 / sp) / (1 - 1 / fc)
        f = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
        out.append(
            f"| {r['arm']} | {r['job'].split('_')[-1].removesuffix('.log')} | "
            f"{f(r['step_s_median'],2)} | {f(r['step_s_p10'],2)} | "
            f"{f(r['step_s_p90'],2)} | {f(sp,3)}x | {f(fl.get('actual_pflops'),1)} | "
            f"{f(fc,2)}x | {f(conv,2)} | {f(fl.get('ffn_cost_frac'))} | "
            f"{f(fl.get('block_skip_keep_frac'))} | {f(fl.get('kv_route_keep_frac'))} | "
            f"{f(r['ce_final'])} | {_eval_cell(r.get('eval'))} |"
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", nargs="+", help="one or more ladder logs; arms are merged in order")
    ap.add_argument("--warmup", type=int, default=5, help="steps discarded before timing")
    ap.add_argument("--control", default="dense")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="flexcompute_train root (ckpts/ + eval/)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    arms: Dict[str, Dict[str, Any]] = {}
    for path in args.log:
        for name, arm in parse_log(path).items():
            arm["job"] = os.path.basename(path)
            arms[name] = arm
    rows = [summarize(a, args.warmup, args.root) for a in arms.values()]
    print(table(rows, args.control))
    print()
    for r in rows:
        ts, bp = r["step_s_median_from_timestamps"], r["step_s_median"]
        drift = f"{ts / bp:.3f}x" if (ts and bp) else "-"
        rt = r.get("router_end") or {}
        rs = "  ".join(f"{k}={v[1]:.3f}(tgt {v[2]:.3f})@s{v[0]}" for k, v in sorted(rt.items()))
        print(
            f"[{r['arm']}] steps={r['n_steps_logged']} timed={r['n_steps_timed']} "
            f"timestamp/BPS={drift} ce_last5={r['ce_last5_mean']}"
            + (f"\n    realized: {rs}" if rs else "")
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
