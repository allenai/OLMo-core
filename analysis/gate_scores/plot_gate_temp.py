#!/usr/bin/env python3
"""Plot the learned per-layer landmark gate temperature from ``dump_gate_temp.py``.

Reads the ``JSON [...]`` line that :mod:`dump_gate_temp` prints (pass either that raw line, a file
containing it, or a bare JSON array) and draws ``T = exp(log_gate_temp)`` against layer index.

The whole story is the y-axis scale, so the figure makes it explicit: the left panel is drawn on the
range the parameter *could* have reached, the right panel zooms to the range it actually reached,
with the deviation from the ``T = 1`` init annotated in percent.

Usage::

    python analysis/gate_scores/plot_gate_temp.py gate_temp_s8550.json --out figs/gate_temp.png
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
    # Deviation from the T = 1 init, in percent -- the only scale on which this parameter moved.
    dev_pct = [100.0 * (t - INIT_T) for t in temps]

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: the range a temperature could plausibly occupy, to show how little this one moved.
    ax_full.axhline(INIT_T, color="0.6", lw=1, ls="--", zorder=1)
    ax_full.plot(layers, temps, "-o", ms=3.5, lw=1.4, color="#1f77b4", zorder=3)
    ax_full.set_ylim(0.5, 1.5)
    ax_full.set_xlabel("layer")
    ax_full.set_ylabel("T = exp(log_gate_temp)")
    ax_full.set_title("plausible range", fontsize=10)
    ax_full.annotate(
        "init T = 1",
        xy=(len(layers) * 0.02, INIT_T),
        xytext=(len(layers) * 0.02, 1.12),
        fontsize=9,
        color="0.35",
        arrowprops=dict(arrowstyle="->", color="0.6", lw=1),
    )
    ax_full.text(
        0.5,
        0.06,
        "every layer sits on the T = 1 line at this scale",
        transform=ax_full.transAxes,
        ha="center",
        fontsize=9,
        color="0.35",
    )

    # Right: the range actually occupied.
    ax_zoom.axhline(0.0, color="0.6", lw=1, ls="--", zorder=1, label="init (T = 1)")
    ax_zoom.plot(layers, dev_pct, "-o", ms=4, lw=1.5, color="#d62728", zorder=3)
    ax_zoom.set_xlabel("layer")
    ax_zoom.set_ylabel("deviation from init  (%)")
    ax_zoom.set_title("actual range (note the axis: tenths of a percent)", fontsize=10)
    ax_zoom.legend(frameon=False, fontsize=9, loc="lower left")

    lo = min(dev_pct, key=lambda d: d)
    lo_layer = layers[dev_pct.index(lo)]
    ax_zoom.annotate(
        f"layer {lo_layer}: {lo:.2f}%  (T = {min(temps):.4f})",
        xy=(lo_layer, lo),
        xytext=(len(layers) * 0.22, lo + 0.015),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="0.5", lw=1),
    )

    for ax in (ax_full, ax_zoom):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", lw=0.8, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle(args.title, fontsize=13, y=0.99)
    fig.text(0.5, 0.915, args.subtitle, ha="center", fontsize=9.5, color="0.4")
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")
    print(
        f"T range [{min(temps):.6f}, {max(temps):.6f}], mean {sum(temps) / len(temps):.6f}; "
        f"max deviation from init {max(abs(d) for d in dev_pct):.3f}%"
    )


if __name__ == "__main__":
    main()
