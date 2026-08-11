"""
Figure-4-style grid plot for the CTC suite (``records/ctc-suite-scaling-plan.md`` §8) -- SKELETON.

Reads ``all_results.jsonl`` (append-only; the LAST record per (task, model, arm, rung) wins) and
renders one panel per task, panels grouped/ordered by complexity class, x = ``rung_tokens`` on a
log2 axis, y = the task metric, one line per arm (full-attention vs chunked) with +/-1 binomial-SE
bands from :func:`results_io.binomial_se`. Headless (Agg backend) by construction.

This is intentionally basic -- it will be refined once real results exist (gap-growth annotations,
per-model facets, results-hub styling). Self-test (no GPU)::

    PYTHONPATH=src python -m scripts.eval.ctc_suite.plot_ctc_suite --selftest
"""

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import
import matplotlib.pyplot as plt

try:  # package import (PYTHONPATH=src) or same-directory fallback (direct file execution)
    from scripts.eval.ctc_suite import results_io
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import results_io  # type: ignore[no-redef]

#: Panel ordering by complexity class (plan §3).
CLASS_ORDER = ("N", "NM", "N2", "N3")

#: Fixed arm styling: color follows the arm identity (never its rank), CVD-safe pair, with
#: linestyle + marker as secondary (non-color) encoding. Extra arms get the remaining fixed slots.
ARM_STYLE = {
    "full": {"color": "#2a78d6", "linestyle": "-", "marker": "o"},
    "chunked": {"color": "#eb6834", "linestyle": "--", "marker": "s"},
    "chunked-mix": {"color": "#008300", "linestyle": ":", "marker": "^"},
}
_EXTRA_COLORS = ["#4a3aa7", "#1baf7a", "#e87ba4", "#eda100", "#e34948"]


def load_results(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Load ``all_results.jsonl``, deduplicating to the LAST record per (task, model, arm, rung).

    :param jsonl_path: Path to the append-only results log.

    :returns: The deduplicated records, in file order of their last occurrence.
    """
    latest: Dict[tuple, Dict[str, Any]] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["task"], rec["model"], rec["arm"], rec["rung_tokens"])
            latest[key] = rec
    return list(latest.values())


def _arm_style(arm: str, fallback_idx: int) -> Dict[str, str]:
    """Fixed style for an arm; unknown arms take stable extra slots (never cycled per-figure)."""
    if arm in ARM_STYLE:
        return ARM_STYLE[arm]
    return {
        "color": _EXTRA_COLORS[fallback_idx % len(_EXTRA_COLORS)],
        "linestyle": "-.",
        "marker": "D",
    }


def plot_grid(records: List[Dict[str, Any]], out_png: str, ncols: int = 4) -> str:
    """
    Render the per-task grid figure and save it as a PNG.

    :param records: Deduplicated §8 records (see :func:`load_results`).
    :param out_png: Output PNG path.
    :param ncols: Panels per row.

    :returns: ``out_png``.

    :raises SystemExit: If there are no records to plot.
    """
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_task[rec["task"]].append(rec)
    if not by_task:
        raise SystemExit("no records to plot")

    def class_rank(task: str) -> int:
        cls = by_task[task][0].get("complexity_class", "?")
        return CLASS_ORDER.index(cls) if cls in CLASS_ORDER else len(CLASS_ORDER)

    tasks = sorted(by_task, key=lambda t: (class_rank(t), t))
    nrows = (len(tasks) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 2.9 * nrows), squeeze=False, sharex=False
    )

    all_arms: List[str] = []
    for task_recs in by_task.values():
        for rec in task_recs:
            if rec["arm"] not in all_arms:
                all_arms.append(rec["arm"])
    all_arms.sort(key=lambda a: (list(ARM_STYLE).index(a) if a in ARM_STYLE else 99, a))

    for i, task in enumerate(tasks):
        ax = axes[i // ncols][i % ncols]
        task_recs = by_task[task]
        cls = task_recs[0].get("complexity_class", "?")
        metric_name = task_recs[0].get("metric_name", "metric")
        for j, arm in enumerate(all_arms):
            pts = sorted((r for r in task_recs if r["arm"] == arm), key=lambda r: r["rung_tokens"])
            if not pts:
                continue
            xs = [r["rung_tokens"] for r in pts]
            ys = [r["metric_value"] for r in pts]
            style = _arm_style(arm, j)
            ax.plot(xs, ys, label=arm, markersize=4, linewidth=2, **style)
            # +/-1 binomial SE band -- only meaningful for rate metrics in [0, 1].
            if all(0.0 <= y <= 1.0 for y in ys):
                ses = [results_io.binomial_se(y, max(r["eval_size"], 1)) for y, r in zip(ys, pts)]
                ax.fill_between(
                    xs,
                    [y - s for y, s in zip(ys, ses)],
                    [y + s for y, s in zip(ys, ses)],
                    color=style["color"],
                    alpha=0.15,
                    linewidth=0,
                )
        ax.set_xscale("log", base=2)
        rungs = sorted({r["rung_tokens"] for r in task_recs})
        ax.set_xticks(rungs)
        ax.set_xticklabels([f"{t // 1024}k" for t in rungs], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(f"{task}  [{cls}]", fontsize=10)
        ax.set_ylabel(metric_name, fontsize=8, color="#555555")
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for i in range(len(tasks), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels),
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, 1.0),
        )
    fig.suptitle(
        "CTC suite: metric vs context tokens (band = +/-1 binomial SE)", fontsize=11, y=1.04
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _selftest() -> None:
    """Generate a synthetic ``all_results.jsonl`` via results_io and render a PNG in a tmp dir."""
    tmp = tempfile.mkdtemp(prefix="ctc_plot_selftest_")
    out_root = os.path.join(tmp, "results", "ctc_suite")
    synth = [
        ("retrieval", "N", "gold_id_f1", {"full": 0.02, "chunked": 0.04}),
        ("outlier", "NM", "set_f1", {"full": 0.05, "chunked": 0.12}),
        ("contradiction", "N2", "set_f1", {"full": 0.08, "chunked": 0.28}),
    ]
    rungs = [2048, 4096, 8192, 16384, 32768]
    for task, cls, metric, decay in synth:
        for arm, slope in decay.items():
            for k, rung in enumerate(rungs):
                record = {
                    "task": task,
                    "complexity_class": cls,
                    "model": "qwen3.5-0.8b",
                    "arm": arm,
                    "rung_tokens": rung,
                    "metric_name": metric,
                    "metric_value": round(max(0.05, 0.9 - slope * k), 3),
                    "eval_size": 500,
                    "cot_label": "no-cot",
                    "aux_metrics": {"eval_seconds": 100.0},
                    "provenance": {
                        "git_commit": "selftest",
                        "data_path": "synthetic",
                        "ckpt_path": "synthetic",
                        "eval_backend": "synthetic",
                        "launcher": "plot_ctc_suite --selftest",
                        "date": "2026-07-18",
                    },
                }
                results_io.write_result(out_root, record)
    jsonl = os.path.join(out_root, results_io.ALL_RESULTS_NAME)
    records = load_results(jsonl)
    assert len(records) == 3 * 2 * len(rungs), len(records)
    out_png = plot_grid(records, os.path.join(tmp, "ctc_suite_grid.png"))
    size = os.path.getsize(out_png)
    assert size > 10_000, f"suspiciously small PNG ({size} bytes)"
    print(f"[plot_ctc_suite selftest] PASS  records={len(records)}  png={out_png} ({size} bytes)")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results", default="results/ctc_suite/all_results.jsonl", help="path to all_results.jsonl"
    )
    ap.add_argument("--out", default="results/ctc_suite/ctc_suite_grid.png", help="output PNG path")
    ap.add_argument("--ncols", type=int, default=4, help="panels per row")
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="synthesize a results log and render a PNG in a tmp dir",
    )
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    records = load_results(args.results)
    out = plot_grid(records, args.out, ncols=args.ncols)
    print(f"[plot_ctc_suite] {len(records)} records -> {out}")


if __name__ == "__main__":
    main()
