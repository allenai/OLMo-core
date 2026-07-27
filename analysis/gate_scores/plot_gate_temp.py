#!/usr/bin/env python3
"""Plot the learned per-layer landmark gate temperature from ``dump_gate_temp.py``.

Reads the ``JSON [...]`` line that :mod:`dump_gate_temp` prints (pass either that raw line, a file
containing it, or a bare JSON array) and draws ``T = exp(log_gate_temp)`` against layer index.

Usage::

    python analysis/gate_scores/plot_gate_temp.py gate_temp_s8550.json --out figs/gate_temp.png
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

INIT_T = 1.0


def _load(src: str):
    """Accept a path to a file, or a literal JSON array / 'JSON [...]' log line."""
    text = open(src).read() if os.path.exists(src) else src
    start = text.index("[")
    return json.loads(text[start : text.rindex("]") + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="file (or literal string) holding dump_gate_temp's JSON array")
    ap.add_argument("--out", default="analysis/gate_scores/figs/gate_temp_by_layer.png")
    ap.add_argument("--title", default="Learned landmark gate temperature by layer")
    ap.add_argument(
        "--subtitle",
        default="q4b-comp-gate-temp-5task-dolci25-32k / step8550 (Qwen3-4B, block 64)",
    )
    args = ap.parse_args()

    rows = sorted(_load(args.source), key=lambda r: r["layer"])
    layers = [r["layer"] for r in rows]
    temps = [r["temp"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))

    ax.axhline(INIT_T, color="0.6", lw=1, ls="--", zorder=1, label="init (T = 1)")
    ax.plot(layers, temps, "-o", ms=4, lw=1.5, color="#1f77b4", zorder=3)

    # Ticks every 0.001 in T, i.e. one tick per tenth of a percent.
    ax.yaxis.set_major_locator(MultipleLocator(0.001))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.set_ylim(min(temps) - 0.0005, INIT_T + 0.0005)

    ax.set_xlabel("layer")
    ax.set_ylabel("T = exp(log_gate_temp)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle(args.title, fontsize=13, y=0.98)
    fig.text(0.5, 0.90, args.subtitle, ha="center", fontsize=9.5, color="0.4")
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")
    print(f"T range [{min(temps):.6f}, {max(temps):.6f}], mean {sum(temps) / len(temps):.6f}")


if __name__ == "__main__":
    main()
