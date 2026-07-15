#!/usr/bin/env python
"""Plot a v2 pretraining intervention against the wide-integration baseline.

The registry below deliberately names every W&B run used in a comparison. This
keeps result regeneration deterministic and prevents similarly named smoke or
rerun jobs from entering a plot unnoticed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import wandb


V2_DIR = Path(__file__).resolve().parent
V1_DIR = V2_DIR.parent / "v1"
if str(V1_DIR) not in sys.path:
    sys.path.insert(0, str(V1_DIR))

from wandb_cache import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    _tail_step_range,
    read_tail_history_from_cache,
    scan_history_cached,
    write_tail_history_to_cache,
)
from experiment_summary_plots import SummaryVariant, plot_observed_best_summary  # noqa: E402


PROJECT = "ai2-llm/jacobm-olmoe-ladder"
LOSS_KEY = "train/CE loss"
TOKENS_KEY = "throughput/total tokens"
HISTORY_FIELDS = ["_step", TOKENS_KEY, LOSS_KEY]
PLOTTABLE_STATES = {"finished", "running"}


@dataclass(frozen=True)
class RegisteredRun:
    cx: int
    lr: float
    run_id: str


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    color: str
    runs: tuple[RegisteredRun, ...]


@dataclass(frozen=True)
class Wave:
    key: str
    title: str
    model: str
    intervention_label: str
    architecture_note: str
    active_parameters: int
    baseline_active_parameters: int
    baseline: Variant
    intervention: Variant


@dataclass(frozen=True)
class Point:
    model: str
    variant: str
    variant_label: str
    cx: int
    lr: float
    lr_tag: str
    loss: float
    state: str
    tokens_b: float
    run_id: str
    name: str
    url: str


WIDE_275M = Variant(
    key="wide_integration",
    label="wide integration (SWA)",
    color="#6b7280",
    runs=(
        RegisteredRun(1, 8e-4, "kfua3dcq"),
        RegisteredRun(1, 1.6e-3, "h86x1nv3"),
        RegisteredRun(1, 3.2e-3, "afxq80js"),
        RegisteredRun(2, 8e-4, "o2bdr3gw"),
        RegisteredRun(2, 1.6e-3, "6porpbo2"),
        RegisteredRun(2, 3.2e-3, "0f782vrw"),
        RegisteredRun(4, 4e-4, "n1gjknwg"),
        RegisteredRun(4, 8e-4, "9n3xk8gs"),
        RegisteredRun(4, 1.6e-3, "ttjquo05"),
        RegisteredRun(4, 3.2e-3, "5u03fshf"),
        RegisteredRun(8, 4e-4, "iv901lom"),
        RegisteredRun(8, 8e-4, "qe052lo4"),
        RegisteredRun(8, 1.6e-3, "qu2zaxr7"),
        RegisteredRun(8, 3.2e-3, "235ye5lg"),
    ),
)

HYBRID_GDN_EV1_275M = Variant(
    key="hybrid_gdn_ev1",
    label="hybrid (GDN, expand_v=1)",
    color="#2563eb",
    runs=(
        RegisteredRun(1, 4e-4, "fkm77yos"),
        RegisteredRun(1, 8e-4, "yo22u93q"),
        RegisteredRun(1, 1.6e-3, "moknw6oc"),
        RegisteredRun(1, 3.2e-3, "mettf0d3"),
        RegisteredRun(2, 4e-4, "s5qmhyb2"),
        RegisteredRun(2, 8e-4, "07qo96gy"),
        RegisteredRun(2, 1.6e-3, "j12fk559"),
        RegisteredRun(2, 3.2e-3, "mem73c7g"),
        RegisteredRun(4, 4e-4, "socvue3a"),
        RegisteredRun(4, 8e-4, "xvk92054"),
        RegisteredRun(4, 1.6e-3, "uhw9wfed"),
        RegisteredRun(4, 3.2e-3, "sr1jgmao"),
        RegisteredRun(8, 4e-4, "b0z3qfmi"),
        RegisteredRun(8, 8e-4, "rkxojd03"),
        RegisteredRun(8, 1.6e-3, "66aja50m"),
        RegisteredRun(8, 3.2e-3, "ntoo8vlo"),
    ),
)

WAVES = {
    "275m_hybrid_gdn_ev1": Wave(
        key="275m_hybrid_gdn_ev1",
        title="275M active hybrid intervention",
        model="275m active",
        intervention_label="hybrid GDN (expand_v=1)",
        architecture_note=(
            "Wide-integration architecture with SWA layers 2/4/6/8/10 replaced "
            "by GDN; full-attention layers are unchanged."
        ),
        active_parameters=288_194_512,
        baseline_active_parameters=280_207_872,
        baseline=WIDE_275M,
        intervention=HYBRID_GDN_EV1_275M,
    )
}


def _mean_tail_loss(history: list[dict[str, Any]], window_tokens: int) -> tuple[float, float] | None:
    samples: list[tuple[float, float]] = []
    for row in history:
        tokens = row.get(TOKENS_KEY)
        loss = row.get(LOSS_KEY)
        if tokens is None or loss is None:
            continue
        tokens = float(tokens)
        loss = float(loss)
        if math.isfinite(tokens) and math.isfinite(loss):
            samples.append((tokens, loss))
    if not samples:
        return None

    samples.sort()
    end_tokens = samples[-1][0]
    losses = [loss for tokens, loss in samples if tokens >= end_tokens - window_tokens]
    if not losses:
        return None
    return statistics.mean(losses), end_tokens / 1e9


def _load_tail_history(
    run: Any,
    *,
    project: str,
    cache_dir: Path,
    window_tokens: int,
    refresh_cache: bool,
    refresh_stale_cache: bool,
) -> list[dict[str, Any]]:
    if run.state == "finished":
        return scan_history_cached(
            run,
            project=project,
            keys=HISTORY_FIELDS,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache,
            refresh_stale_cache=refresh_stale_cache,
            tail_window_tokens=window_tokens,
            page_size=10_000,
        )

    # A running run changes too often for the finished-run cache path. Still
    # fetch only the final window and keep a step-keyed snapshot, avoiding a
    # full W&B history download every time plots are refreshed.
    if not refresh_cache:
        cached = read_tail_history_from_cache(
            cache_dir,
            project,
            run,
            window_tokens=window_tokens,
            tokens_key=TOKENS_KEY,
            keys=HISTORY_FIELDS,
        )
        if cached is not None:
            return cached

    min_step, max_step = _tail_step_range(run, window_tokens=window_tokens)
    history = [
        dict(row)
        for row in run.scan_history(
            keys=HISTORY_FIELDS,
            min_step=min_step,
            max_step=max_step,
            page_size=max(100_000, max_step - min_step),
        )
    ]
    write_tail_history_to_cache(
        cache_dir,
        project,
        run,
        history,
        window_tokens=window_tokens,
        min_step=min_step,
        max_step=max_step,
        keys=HISTORY_FIELDS,
    )
    return history


def load_points(
    wave: Wave,
    *,
    project: str,
    cache_dir: Path,
    window_m: int,
    include_running: bool,
    refresh_cache: bool,
    refresh_stale_cache: bool,
) -> list[Point]:
    api = wandb.Api(timeout=90)
    points: list[Point] = []
    allowed_states = PLOTTABLE_STATES if include_running else {"finished"}
    window_tokens = window_m * 1_000_000

    for variant in (wave.baseline, wave.intervention):
        for registered in variant.runs:
            run = api.run(f"{project}/{registered.run_id}")
            if run.state not in allowed_states:
                print(f"skip {registered.run_id}: state={run.state}")
                continue
            history = _load_tail_history(
                run,
                project=project,
                cache_dir=cache_dir,
                window_tokens=window_tokens,
                refresh_cache=refresh_cache,
                refresh_stale_cache=refresh_stale_cache,
            )
            loss_info = _mean_tail_loss(history, window_tokens)
            if loss_info is None:
                print(f"skip {registered.run_id}: no usable {LOSS_KEY} history")
                continue
            loss, tokens_b = loss_info
            points.append(
                Point(
                    model="275m",
                    variant=variant.key,
                    variant_label=variant.label,
                    cx=registered.cx,
                    lr=registered.lr,
                    lr_tag=f"{registered.lr:.2g}",
                    loss=loss,
                    state=run.state,
                    tokens_b=tokens_b,
                    run_id=registered.run_id,
                    name=run.display_name or run.name,
                    url=run.url,
                )
            )
            print(
                f"loaded {variant.key:>18} Cx{registered.cx} {registered.lr:.2g} "
                f"{run.state:>8} avg{window_m}M={loss:.6f} tokens={tokens_b:.3f}B"
            )
    return sorted(points, key=lambda point: (point.cx, point.variant, point.lr))


def _expected_count(variant: Variant, cx: int) -> int:
    return sum(run.cx == cx for run in variant.runs)


def _finished(points: list[Point], variant: str, cx: int) -> list[Point]:
    return sorted(
        [point for point in points if point.variant == variant and point.cx == cx and point.state == "finished"],
        key=lambda point: point.lr,
    )


def _best_finished(points: list[Point], variant: str, cx: int) -> Point | None:
    candidates = _finished(points, variant, cx)
    return min(candidates, key=lambda point: point.loss) if candidates else None


def _fit_minimum(points: list[Point]) -> tuple[float, float] | None:
    if len(points) < 3:
        return None
    best_index = min(range(len(points)), key=lambda index: points[index].loss)
    if best_index in {0, len(points) - 1}:
        return None
    local = points[best_index - 1 : best_index + 2]
    x = np.array([math.log10(point.lr) for point in local])
    y = np.array([point.loss for point in local])
    a, b, c = np.polyfit(x, y, 2)
    if a <= 0:
        return None
    optimum_x = -b / (2 * a)
    if optimum_x < min(x) or optimum_x > max(x):
        return None
    return 10**optimum_x, float(a * optimum_x**2 + b * optimum_x + c)


def plot_intervention_uplot(points: list[Point], wave: Wave, output_path: Path, window_m: int) -> Path:
    """Match the v1 all-Cx U-plot format for one intervention."""

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for cx in (1, 2, 4, 8):
        group = _finished(points, wave.intervention.key, cx)
        if not group:
            continue
        (line,) = ax.plot(
            [point.lr for point in group],
            [point.loss for point in group],
            marker="o",
            linewidth=1.8,
            label=f"Cx{cx}",
        )
        fit = _fit_minimum(group)
        if fit is not None:
            fit_lr, fit_loss = fit
            ax.axvline(fit_lr, color=line.get_color(), linestyle=":", linewidth=1.2, alpha=0.75)
            ax.scatter(
                [fit_lr],
                [fit_loss],
                marker="*",
                s=80,
                color=line.get_color(),
                edgecolor="black",
                linewidth=0.4,
                zorder=5,
            )
            ax.annotate(
                f"Cx{cx} fit3: {fit_lr:.2g}",
                (fit_lr, fit_loss),
                textcoords="offset points",
                xytext=(6, -14),
                ha="left",
                fontsize=8,
                color=line.get_color(),
                alpha=0.9,
            )
        last = group[-1]
        ax.annotate(
            f"Cx{cx}",
            (last.lr, last.loss),
            textcoords="offset points",
            xytext=(8, 0),
            ha="left",
            va="center",
            fontsize=8,
            color=line.get_color(),
        )

    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel(f"train CE avg{window_m}M")
    ax.set_title(f"{wave.model} {wave.intervention_label} LR sweeps")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_summary(points: list[Point], wave: Wave, output_path: Path, window_m: int) -> Path:
    """Use the shared v1 observed-best summary implementation."""

    eligible_cxs = {
        cx
        for cx in (1, 2, 4, 8)
        if _fit_minimum(_finished(points, wave.intervention.key, cx)) is not None
    }
    eligible_points = [point for point in points if point.cx in eligible_cxs]
    plot_observed_best_summary(
        eligible_points,
        out_path=output_path,
        title=f"{wave.title} observed best",
        variants=(
            SummaryVariant(
                "baseline",
                (wave.baseline.key,),
                wave.baseline.label,
                color="black",
                linestyle="--",
            ),
            SummaryVariant(
                wave.intervention.key,
                (wave.intervention.key,),
                wave.intervention.label,
                color=wave.intervention.color,
            ),
        ),
        window_m=window_m,
    )
    return output_path


def write_results(points: list[Point], wave: Wave, output_path: Path, window_m: int) -> tuple[Path, Path]:
    generated_at = datetime.now(UTC).isoformat()
    best_by_cx: list[dict[str, Any]] = []
    for cx in (1, 2, 4, 8):
        baseline = _best_finished(points, wave.baseline.key, cx)
        intervention = _best_finished(points, wave.intervention.key, cx)
        complete_count = len(_finished(points, wave.intervention.key, cx))
        expected_count = _expected_count(wave.intervention, cx)
        best_by_cx.append(
            {
                "cx": cx,
                "sweep_complete": complete_count == expected_count,
                "completed_intervention_runs": complete_count,
                "expected_intervention_runs": expected_count,
                "wide_best": asdict(baseline) if baseline else None,
                "intervention_best": asdict(intervention) if intervention else None,
                "delta_intervention_minus_wide": (
                    intervention.loss - baseline.loss if baseline and intervention else None
                ),
            }
        )

    payload = {
        "generated_at": generated_at,
        "project": PROJECT,
        "wave": wave.key,
        "model": wave.model,
        "architecture_note": wave.architecture_note,
        "selection_metric": f"mean {LOSS_KEY} over the final {window_m}M observed tokens",
        "selection_rule": (
            "result table uses the observed best among finished runs; summary requires the observed best "
            "to be an interior point with a valid quadratic fit; fitted minima are display-only"
        ),
        "baseline_active_parameters": wave.baseline_active_parameters,
        "intervention_active_parameters": wave.active_parameters,
        "active_parameter_delta_fraction": (
            wave.active_parameters / wave.baseline_active_parameters - 1
        ),
        "runs": [asdict(point) for point in points],
        "best_by_cx": best_by_cx,
    }
    json_path = output_path.with_suffix(".json")
    md_path = output_path.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        f"# {wave.title}",
        "",
        f"Generated: `{generated_at}`",
        "",
        f"Selection metric: final `{window_m}M`-token mean training CE. Only finished runs are eligible.",
        "The summary includes a Cx only when its observed best is bracketed and supports a valid quadratic fit.",
        "Fitted LR minima in the U-plot are visual aids and are never used to select results.",
        "",
        "## Observed best",
        "",
        "| Cx | Status | Wide loss (LR) | Intervention loss (LR) | Delta |",
        "|---:|---|---:|---:|---:|",
    ]
    for result in best_by_cx:
        wide = result["wide_best"]
        intervention = result["intervention_best"]
        status = "complete" if result["sweep_complete"] else (
            f"provisional ({result['completed_intervention_runs']}/{result['expected_intervention_runs']})"
        )
        wide_text = f"{wide['loss']:.6f} ({wide['lr']:.2g})" if wide else "—"
        intervention_text = (
            f"{intervention['loss']:.6f} ({intervention['lr']:.2g})" if intervention else "—"
        )
        delta = result["delta_intervention_minus_wide"]
        delta_text = f"{delta:+.6f}" if delta is not None else "—"
        lines.append(f"| Cx{result['cx']} | {status} | {wide_text} | {intervention_text} | {delta_text} |")

    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Variant | Cx | LR | State | Tokens (B) | Final-window CE | W&B |",
            "|---|---:|---:|---|---:|---:|---|",
        ]
    )
    for point in points:
        lines.append(
            f"| {point.variant_label} | {point.cx} | {point.lr:.2g} | {point.state} | "
            f"{point.tokens_b:.3f} | {point.loss:.6f} | [{point.run_id}]({point.url}) |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", choices=sorted(WAVES), default="275m_hybrid_gdn_ev1")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--window-m", type=int, default=250)
    parser.add_argument("--include-running", action="store_true")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force selected histories to be downloaded again.",
    )
    parser.add_argument(
        "--refresh-stale-cache",
        action="store_true",
        help="Refresh only missing, stale, or too-short finished histories.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--results-path", type=Path)
    args = parser.parse_args()

    wave = WAVES[args.wave]
    output_dir = args.output_dir or V2_DIR / "plots" / "pretraining" / wave.key
    results_path = args.results_path or V2_DIR / "results" / "pretraining" / wave.key
    output_dir.mkdir(parents=True, exist_ok=True)

    points = load_points(
        wave,
        project=args.project,
        cache_dir=args.cache_dir,
        window_m=args.window_m,
        include_running=args.include_running,
        refresh_cache=args.refresh_cache,
        refresh_stale_cache=args.refresh_stale_cache,
    )
    if not points:
        raise RuntimeError("No usable W&B histories were found")

    paths = [
        plot_intervention_uplot(
            points,
            wave,
            output_dir / f"{wave.key}_uplot.png",
            args.window_m,
        ),
        plot_summary(
            points,
            wave,
            output_dir / "summary_observed_best.png",
            args.window_m,
        ),
    ]
    result_paths = write_results(points, wave, results_path, args.window_m)
    print("\nWrote:")
    for path in (*paths, *result_paths):
        print(path)


if __name__ == "__main__":
    main()
