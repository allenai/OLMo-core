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
FINAL_WINDOW_M = 250
FINAL_WINDOW_TOKENS = FINAL_WINDOW_M * 1_000_000


@dataclass(frozen=True)
class RegisteredRun:
    model: str
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
    intervention_label: str
    architecture_note: str
    models: tuple[str, ...]
    lr_sweep_models: tuple[str, ...]
    active_parameters: dict[str, int]
    baseline_active_parameters: dict[str, int]
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


class IncompleteTailWindowError(RuntimeError):
    def __init__(self, *, available_tokens: float, required_tokens: int) -> None:
        self.available_tokens = available_tokens
        self.required_tokens = required_tokens
        super().__init__(
            f"history covers only {available_tokens / 1e6:.1f}M of the required "
            f"{required_tokens / 1e6:.0f}M final-token window"
        )


WIDE_INTEGRATION = Variant(
    key="wide_integration",
    label="wide integration (SWA)",
    color="#6b7280",
    runs=(
        RegisteredRun("275m", 1, 8e-4, "kfua3dcq"),
        RegisteredRun("275m", 1, 1.6e-3, "h86x1nv3"),
        RegisteredRun("275m", 1, 3.2e-3, "afxq80js"),
        RegisteredRun("275m", 2, 8e-4, "o2bdr3gw"),
        RegisteredRun("275m", 2, 1.6e-3, "6porpbo2"),
        RegisteredRun("275m", 2, 3.2e-3, "0f782vrw"),
        RegisteredRun("275m", 4, 4e-4, "n1gjknwg"),
        RegisteredRun("275m", 4, 8e-4, "9n3xk8gs"),
        RegisteredRun("275m", 4, 1.6e-3, "ttjquo05"),
        RegisteredRun("275m", 4, 3.2e-3, "5u03fshf"),
        RegisteredRun("275m", 8, 4e-4, "iv901lom"),
        RegisteredRun("275m", 8, 8e-4, "qe052lo4"),
        RegisteredRun("275m", 8, 1.6e-3, "qu2zaxr7"),
        RegisteredRun("275m", 8, 3.2e-3, "235ye5lg"),
        RegisteredRun("480m", 1, 1.2e-3, "z4wxvc6h"),
        RegisteredRun("480m", 2, 9e-4, "ywj13bkw"),
        RegisteredRun("810m", 1, 6e-4, "w912irkq"),
        RegisteredRun("810m", 2, 5.6e-4, "jpbqhfvc"),
        RegisteredRun("810m", 4, 4e-4, "58ftjxmw"),
        RegisteredRun("810m", 8, 4e-4, "kyti8h1y"),
        RegisteredRun("1p2b", 1, 4e-4, "hww8eksq"),
        RegisteredRun("1p2b", 2, 6e-4, "jfwntmwm"),
        RegisteredRun("1p2b", 4, 3e-4, "u7ab1tpb"),
        RegisteredRun("1p2b", 8, 4e-4, "bqjzmiqi"),
    ),
)

HYBRID_GDN_EV1 = Variant(
    key="hybrid_gdn_ev1",
    label="hybrid (GDN, expand_v=1)",
    color="#2563eb",
    runs=(
        RegisteredRun("275m", 1, 4e-4, "fkm77yos"),
        RegisteredRun("275m", 1, 8e-4, "yo22u93q"),
        RegisteredRun("275m", 1, 1.6e-3, "moknw6oc"),
        RegisteredRun("275m", 1, 3.2e-3, "mettf0d3"),
        RegisteredRun("275m", 2, 4e-4, "s5qmhyb2"),
        RegisteredRun("275m", 2, 8e-4, "07qo96gy"),
        RegisteredRun("275m", 2, 1.6e-3, "j12fk559"),
        RegisteredRun("275m", 2, 3.2e-3, "mem73c7g"),
        RegisteredRun("275m", 4, 4e-4, "socvue3a"),
        RegisteredRun("275m", 4, 8e-4, "xvk92054"),
        RegisteredRun("275m", 4, 1.6e-3, "uhw9wfed"),
        RegisteredRun("275m", 4, 3.2e-3, "sr1jgmao"),
        RegisteredRun("275m", 8, 4e-4, "b0z3qfmi"),
        RegisteredRun("275m", 8, 8e-4, "rkxojd03"),
        RegisteredRun("275m", 8, 1.6e-3, "66aja50m"),
        RegisteredRun("275m", 8, 3.2e-3, "ntoo8vlo"),
        RegisteredRun("480m", 1, 1.2e-3, "wl8ebsd8"),
        RegisteredRun("480m", 2, 9e-4, "4vzmrld1"),
        RegisteredRun("810m", 1, 6e-4, "h1rmcm2p"),
        RegisteredRun("810m", 2, 5.6e-4, "1d5gxgjv"),
        RegisteredRun("810m", 4, 4e-4, "bvlzu2c9"),
        RegisteredRun("810m", 8, 4e-4, "k1d1td9b"),
        RegisteredRun("1p2b", 1, 4e-4, "1d24xfx5"),
        RegisteredRun("1p2b", 2, 6e-4, "vr2jfn4c"),
        RegisteredRun("1p2b", 4, 3e-4, "h5ft97x1"),
        RegisteredRun("1p2b", 8, 4e-4, "zyeib8rb"),
    ),
)

WAVES = {
    "hybrid_gdn_ev1": Wave(
        key="hybrid_gdn_ev1",
        title="Active hybrid GDN intervention",
        intervention_label="hybrid GDN (expand_v=1)",
        architecture_note=(
            "Integration-wide architecture at each size with every SWA layer "
            "replaced by GDN; full-attention layers are unchanged."
        ),
        models=("275m", "480m", "810m", "1p2b"),
        lr_sweep_models=("275m",),
        active_parameters={
            "275m": 288_194_512,
            "480m": 501_228_784,
            "810m": 859_400_792,
            "1p2b": 1_288_662_592,
        },
        baseline_active_parameters={
            "275m": 280_207_872,
            "480m": 486_348_800,
            "810m": 823_569_920,
            "1p2b": 1_225_011_712,
        },
        baseline=WIDE_INTEGRATION,
        intervention=HYBRID_GDN_EV1,
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
    available_tokens = end_tokens - samples[0][0]
    if available_tokens < window_tokens:
        raise IncompleteTailWindowError(
            available_tokens=available_tokens,
            required_tokens=window_tokens,
        )
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
    if window_m != FINAL_WINDOW_M:
        raise ValueError(f"canonical pretraining summaries require exactly {FINAL_WINDOW_M}M tokens")
    api = wandb.Api(timeout=90)
    points: list[Point] = []
    allowed_states = PLOTTABLE_STATES if include_running else {"finished"}
    window_tokens = FINAL_WINDOW_TOKENS
    incomplete_tails: list[str] = []

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
            try:
                loss_info = _mean_tail_loss(history, window_tokens)
            except IncompleteTailWindowError as exc:
                incomplete_tails.append(
                    f"{registered.run_id} ({variant.key} {registered.model} Cx{registered.cx}): {exc}"
                )
                print(f"INCOMPLETE FINAL WINDOW: {incomplete_tails[-1]}")
                continue
            if loss_info is None:
                print(f"skip {registered.run_id}: no usable {LOSS_KEY} history")
                continue
            loss, tokens_b = loss_info
            points.append(
                Point(
                    model=registered.model,
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
                f"loaded {variant.key:>18} {registered.model:>4} "
                f"Cx{registered.cx} {registered.lr:.2g} "
                f"{run.state:>8} avg{window_m}M={loss:.6f} tokens={tokens_b:.3f}B"
            )
    if incomplete_tails:
        details = "\n  - ".join(incomplete_tails)
        raise RuntimeError(
            "Refusing to generate partial final-window summaries. Register and combine "
            "the predecessor W&B run history for each reset/resume segment:\n  - " + details
        )
    model_order = {model: index for index, model in enumerate(wave.models)}
    return sorted(
        points,
        key=lambda point: (model_order[point.model], point.cx, point.variant, point.lr),
    )


def _expected_count(variant: Variant, model: str, cx: int) -> int:
    return sum(run.model == model and run.cx == cx for run in variant.runs)


def _finished(points: list[Point], variant: str, model: str, cx: int) -> list[Point]:
    return sorted(
        [
            point
            for point in points
            if point.variant == variant
            and point.model == model
            and point.cx == cx
            and point.state == "finished"
        ],
        key=lambda point: point.lr,
    )


def _best_finished(points: list[Point], variant: str, model: str, cx: int) -> Point | None:
    candidates = _finished(points, variant, model, cx)
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


def plot_intervention_uplot(
    points: list[Point],
    wave: Wave,
    model: str,
    output_path: Path,
    window_m: int,
) -> Path:
    """Match the v1 all-Cx U-plot format for one intervention."""

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for cx in (1, 2, 4, 8):
        group = _finished(points, wave.intervention.key, model, cx)
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
    ax.set_title(f"{model} active {wave.intervention_label} LR sweeps")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_optimal_summary(points: list[Point], wave: Wave, output_path: Path, window_m: int) -> Path:
    """Plot only model/Cx cells backed by a bracketed LR sweep."""

    eligible_keys = {
        (model, cx)
        for model in wave.lr_sweep_models
        for cx in (1, 2, 4, 8)
        if _fit_minimum(_finished(points, wave.intervention.key, model, cx)) is not None
    }
    eligible_points = [point for point in points if (point.model, point.cx) in eligible_keys]
    plot_observed_best_summary(
        eligible_points,
        out_path=output_path,
        title=f"{wave.title}: observed optimal LRs",
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


def plot_fixed_lr_scale_comparison(
    points: list[Point],
    wave: Wave,
    output_path: Path,
    window_m: int,
) -> Path:
    """Compare all sizes while clearly marking unfinished hybrid cells."""

    models = list(wave.models)
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(max(8.0, 3.6 * len(models)), 4.6),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    for ax, model in zip(axes[0], models):
        ax.set_facecolor("white")
        intervention_cxs: list[int] = []
        for variant, color, linestyle in (
            (wave.baseline, "black", "--"),
            (wave.intervention, wave.intervention.color, "-"),
        ):
            selected = [
                point
                for cx in (1, 2, 4, 8)
                if (point := _best_finished(points, variant.key, model, cx)) is not None
            ]
            if not selected:
                continue
            if variant.key == wave.intervention.key:
                intervention_cxs = [point.cx for point in selected]
            ax.plot(
                [point.cx for point in selected],
                [point.loss for point in selected],
                marker="o",
                linewidth=2.0,
                color=color,
                linestyle=linestyle,
                label=variant.label,
            )
        for cx in intervention_cxs:
            point = _best_finished(points, wave.intervention.key, model, cx)
            assert point is not None
            ax.annotate(
                f"LR {point.lr:.2g}",
                (point.cx, point.loss),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color=wave.intervention.color,
            )
        pending_cxs = [cx for cx in (1, 2, 4, 8) if cx not in intervention_cxs]
        if pending_cxs:
            pending = ", ".join(f"Cx{cx}" for cx in pending_cxs)
            ax.text(
                0.04,
                0.04,
                f"hybrid pending: {pending}",
                transform=ax.transAxes,
                fontsize=7,
                color="#6b7280",
                ha="left",
                va="bottom",
            )
        mode = "observed-optimal LR" if model in wave.lr_sweep_models else "wide-LR transfer"
        ax.set_xscale("log", base=2)
        ax.set_xticks((1, 2, 4, 8))
        ax.set_xticklabels(("Cx1", "Cx2", "Cx4", "Cx8"))
        ax.set_xlabel("data multiple")
        ax.set_title(f"{model}\n{mode}", fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
    axes[0][0].set_ylabel(f"train CE avg{window_m}M")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
        )
    fig.suptitle("All-size hybrid comparison (finished runs only)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)
    return output_path


def write_results(points: list[Point], wave: Wave, output_path: Path, window_m: int) -> tuple[Path, Path]:
    generated_at = datetime.now(UTC).isoformat()
    results: list[dict[str, Any]] = []
    registered_cells = {
        (registered.model, registered.cx) for registered in wave.intervention.runs
    }
    for model, cx in sorted(
        registered_cells,
        key=lambda cell: (wave.models.index(cell[0]), cell[1]),
    ):
        baseline = _best_finished(points, wave.baseline.key, model, cx)
        intervention = _best_finished(points, wave.intervention.key, model, cx)
        finished = _finished(points, wave.intervention.key, model, cx)
        complete_count = len(finished)
        expected_count = _expected_count(wave.intervention, model, cx)
        mode = "lr_sweep" if model in wave.lr_sweep_models else "fixed_lr_transfer"
        results.append(
            {
                "model": model,
                "cx": cx,
                "mode": mode,
                "complete": complete_count == expected_count,
                "optimal_summary_eligible": (
                    mode == "lr_sweep" and _fit_minimum(finished) is not None
                ),
                "completed_intervention_runs": complete_count,
                "expected_intervention_runs": expected_count,
                "wide_reference": asdict(baseline) if baseline else None,
                "intervention_result": asdict(intervention) if intervention else None,
                "delta_intervention_minus_wide": (
                    intervention.loss - baseline.loss if baseline and intervention else None
                ),
            }
        )

    payload = {
        "generated_at": generated_at,
        "project": PROJECT,
        "wave": wave.key,
        "models": list(wave.models),
        "architecture_note": wave.architecture_note,
        "selection_metric": f"mean {LOSS_KEY} over the final {window_m}M observed tokens",
        "selection_rule": (
            "275M optimal-summary cells require an interior observed best and a valid quadratic fit. "
            "Larger sizes are fixed-LR transfer comparisons at the wide-optimal LR and are never "
            "reported as hybrid-optimal without an LR sweep."
        ),
        "baseline_active_parameters": wave.baseline_active_parameters,
        "intervention_active_parameters": wave.active_parameters,
        "active_parameter_delta_fraction": {
            model: wave.active_parameters[model] / wave.baseline_active_parameters[model] - 1
            for model in wave.models
        },
        "runs": [asdict(point) for point in points],
        "results": results,
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
        "The optimal-LR summary includes only bracketed 275M sweeps with a valid quadratic fit.",
        "The all-size figure uses observed-optimal points for 275M and fixed wide-LR transfer points for larger sizes; pending hybrid cells are labeled explicitly.",
        "Fitted LR minima in the 275M U-plot are visual aids and are never used to select results.",
        "",
        "## Completed results",
        "",
        "| Model | Cx | Mode | Status | Wide reference (LR) | Hybrid result (LR) | Delta |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for result in results:
        wide = result["wide_reference"]
        intervention = result["intervention_result"]
        if result["complete"]:
            status = "complete" if result["mode"] == "lr_sweep" else "finished"
        else:
            status = (
                f"provisional ({result['completed_intervention_runs']}/"
                f"{result['expected_intervention_runs']})"
                if result["mode"] == "lr_sweep"
                else "pending"
            )
        wide_text = f"{wide['loss']:.6f} ({wide['lr']:.2g})" if wide else "—"
        intervention_text = (
            f"{intervention['loss']:.6f} ({intervention['lr']:.2g})" if intervention else "—"
        )
        delta = result["delta_intervention_minus_wide"]
        delta_text = f"{delta:+.6f}" if delta is not None else "—"
        mode = "LR sweep" if result["mode"] == "lr_sweep" else "fixed-LR transfer"
        lines.append(
            f"| {result['model']} | Cx{result['cx']} | {mode} | {status} | "
            f"{wide_text} | {intervention_text} | {delta_text} |"
        )

    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Model | Variant | Cx | LR | State | Tokens (B) | Final-window CE | W&B |",
            "|---|---|---:|---:|---|---:|---:|---|",
        ]
    )
    for point in points:
        lines.append(
            f"| {point.model} | {point.variant_label} | {point.cx} | {point.lr:.2g} | {point.state} | "
            f"{point.tokens_b:.3f} | {point.loss:.6f} | [{point.run_id}]({point.url}) |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", choices=sorted(WAVES), default="hybrid_gdn_ev1")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
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
    results_path = (
        args.results_path
        or V2_DIR / "results" / "pretraining" / wave.key / "results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    points = load_points(
        wave,
        project=args.project,
        cache_dir=args.cache_dir,
        window_m=FINAL_WINDOW_M,
        include_running=args.include_running,
        refresh_cache=args.refresh_cache,
        refresh_stale_cache=args.refresh_stale_cache,
    )
    if not points:
        raise RuntimeError("No usable W&B histories were found")

    paths = [
        *(
            plot_intervention_uplot(
                points,
                wave,
                model,
                output_dir / f"{model}_uplot.png",
                FINAL_WINDOW_M,
            )
            for model in wave.lr_sweep_models
        ),
        plot_optimal_summary(
            points,
            wave,
            output_dir / "summary_observed_best.png",
            FINAL_WINDOW_M,
        ),
        plot_fixed_lr_scale_comparison(
            points,
            wave,
            output_dir / "fixed_lr_scale_comparison.png",
            FINAL_WINDOW_M,
        ),
    ]
    result_paths = write_results(points, wave, results_path, FINAL_WINDOW_M)
    print("\nWrote:")
    for path in (*paths, *result_paths):
        print(path)


if __name__ == "__main__":
    main()
