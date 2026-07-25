#!/usr/bin/env python
"""Plot the canonical 275M GDN2 and KDA LR sweeps.

Run IDs are resolved from an exact, checked-in list of W&B display names. This
lets queued jobs initialize after this file is committed while preserving the
project rule that similarly named runs must never enter plots implicitly. An
exact name that resolves to more than one W&B run is an error and must be
replaced with an explicit segment decision.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import wandb
from plot_pretraining_wave import (
    DEFAULT_CACHE_DIR,
    FINAL_WINDOW_M,
    GEOMETRY_GDN2_EV2_NOPE_GATED,
    GEOMETRY_GDN_EV2_NOPE_GATED,
    PROJECT,
    V2_DIR,
    WIDE_INTEGRATION,
    Point,
    RegisteredRun,
    SummaryVariant,
    Variant,
    Wave,
    _expected_count,
    _finished,
    _fit_minimum,
    load_points,
    plot_intervention_uplot,
    plot_observed_best_summary,
)

CXS = (1, 2, 4, 8)
LRS = (4e-4, 8e-4, 1.6e-3, 3.2e-3)
GDN2_KEY = "geometry_gdn2_ev1_noneg_nope_gated"
KDA_KEY = "geometry_kda_ev1_noneg_nope_gated"
EXPECTED_SWEEP_POINTS = 4


def _lr_name(lr: float) -> str:
    return {
        4e-4: "4e-4",
        8e-4: "8e-4",
        1.6e-3: "1p6e-3",
        3.2e-3: "3p2e-3",
    }[lr]


def _planned_display_names() -> dict[str, list[tuple[int, float, str]]]:
    planned = {GDN2_KEY: [], KDA_KEY: []}
    for cx in CXS:
        for lr in LRS:
            tag = _lr_name(lr)
            planned[GDN2_KEY].append(
                (
                    cx,
                    lr,
                    f"pt-275m-geometry-hybrid-gdn2-ev1-noneg-nope-gated-cx{cx}-lr{tag}-r1",
                )
            )
            planned[KDA_KEY].append(
                (
                    cx,
                    lr,
                    f"pt-275m-geometry-kda-ev1-noneg-nope-gated-cx{cx}-lr{tag}",
                )
            )
    return planned


def resolve_interventions(api: Any, project: str) -> tuple[Variant, Variant, dict[str, list[str]]]:
    """Resolve exact W&B names once and reject ambiguous histories."""

    planned = _planned_display_names()
    display_names = [name for rows in planned.values() for _, _, name in rows]
    matches = list(
        api.runs(
            project,
            filters={"display_name": {"$in": display_names}},
            per_page=max(50, len(display_names)),
        )
    )
    by_name: dict[str, list[Any]] = {name: [] for name in display_names}
    for run in matches:
        if run.display_name in by_name:
            by_name[run.display_name].append(run)

    unresolved: dict[str, list[str]] = {GDN2_KEY: [], KDA_KEY: []}

    def build_variant(key: str, label: str, color: str) -> Variant:
        registered: list[RegisteredRun] = []
        for cx, lr, display_name in planned[key]:
            exact = by_name[display_name]
            if not exact:
                unresolved[key].append(display_name)
                continue
            if len(exact) != 1:
                ids = ", ".join(sorted(run.id for run in exact))
                raise RuntimeError(
                    f"{display_name!r} resolved to multiple W&B runs ({ids}); "
                    "register the intended run and predecessor segments explicitly"
                )
            registered.append(RegisteredRun("275m", cx, lr, exact[0].id))
        return Variant(key=key, label=label, color=color, runs=tuple(registered))

    canonical_gdn2 = build_variant(
        GDN2_KEY,
        "canonical GDN2 (expand_v=1, nonnegative)",
        "#db2777",
    )
    canonical_kda = build_variant(
        KDA_KEY,
        "canonical KDA (expand_v=1, nonnegative)",
        "#2563eb",
    )
    return canonical_gdn2, canonical_kda, unresolved


def comparison_wave(canonical_gdn2: Variant, canonical_kda: Variant) -> Wave:
    return Wave(
        key="canonical_gdn2_kda",
        title="Canonical GDN2 and KDA recurrent-mixer comparison",
        intervention_label="canonical KDA",
        architecture_note=(
            "Both new interventions use the same 275M ten-layer gated-NoPE MoE "
            "geometry. The comparison references wide integration, matching "
            "geometry-matched gated-NoPE GDN1, and the original expand_v=2 "
            "gated-NoPE GDN2 model."
        ),
        models=("275m",),
        lr_sweep_models=("275m",),
        active_parameters={"275m": 274_470_720},
        baseline_active_parameters={"275m": 280_207_872},
        baseline=WIDE_INTEGRATION,
        additional_baselines=(
            GEOMETRY_GDN_EV2_NOPE_GATED,
            GEOMETRY_GDN2_EV2_NOPE_GATED,
            canonical_gdn2,
        ),
        intervention=canonical_kda,
        uplot_baselines=False,
    )


def _variant_expected_count(variant: Variant, cx: int) -> int:
    if variant.key in {GDN2_KEY, KDA_KEY}:
        return EXPECTED_SWEEP_POINTS
    return _expected_count(variant, "275m", cx)


def plot_shared_best_of(
    points: list[Point],
    variants: tuple[Variant, ...],
    interventions: tuple[Variant, ...],
    output_path: Path,
) -> Path:
    """Plot observed bests only where each variant has a bracketed curve."""

    eligible: dict[str, set[int]] = {}
    for variant in variants:
        eligible[variant.key] = {
            cx for cx in CXS if _fit_minimum(_finished(points, variant.key, "275m", cx)) is not None
        }
    new_eligible = set().union(*(eligible[variant.key] for variant in interventions))

    filtered = [
        point
        for point in points
        if point.cx in new_eligible and point.cx in eligible[point.variant]
    ]
    provisional = {
        ("275m", cx, variant.key)
        for variant in variants
        for cx in eligible[variant.key]
        if len(_finished(points, variant.key, "275m", cx)) < _variant_expected_count(variant, cx)
    }
    linestyles = ("--", ":", "-.", "-", "-")
    summary_variants = tuple(
        SummaryVariant(
            "baseline" if index == 0 else variant.key,
            (variant.key,),
            variant.label,
            color="black" if index == 0 else variant.color,
            linestyle=linestyles[index],
        )
        for index, variant in enumerate(variants)
    )
    plot_observed_best_summary(
        filtered,
        out_path=output_path,
        title="275M canonical GDN2 vs KDA: observed optimal LRs",
        variants=summary_variants,
        window_m=FINAL_WINDOW_M,
        provisional_points=provisional,
    )
    return output_path


def write_comparison_results(
    points: list[Point],
    variants: tuple[Variant, ...],
    unresolved: dict[str, list[str]],
    output_path: Path,
) -> tuple[Path, Path]:
    generated_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for cx in CXS:
            finished = _finished(points, variant.key, "275m", cx)
            fit = _fit_minimum(finished)
            best = min(finished, key=lambda point: point.loss) if finished else None
            expected = _variant_expected_count(variant, cx)
            rows.append(
                {
                    "variant": variant.key,
                    "variant_label": variant.label,
                    "cx": cx,
                    "finished_points": len(finished),
                    "expected_points": expected,
                    "bracketed": fit is not None,
                    "provisional": fit is not None and len(finished) < expected,
                    "observed_best": asdict(best) if best else None,
                    "predicted_fit_lr": fit[0] if fit else None,
                    "predicted_fit_loss": fit[1] if fit else None,
                }
            )

    payload = {
        "generated_at": generated_at,
        "project": PROJECT,
        "selection_metric": f"mean train/CE loss over the final {FINAL_WINDOW_M}M tokens",
        "selection_rule": (
            "The best-of plot shows the observed best LR only after that variant/Cx "
            "has an interior observed minimum and a valid local quadratic fit. "
            "Predicted minima are visual aids and are not selected results."
        ),
        "unresolved_planned_runs": unresolved,
        "runs": [asdict(point) for point in points],
        "results": rows,
    }
    json_path = output_path.with_suffix(".json")
    md_path = output_path.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Canonical GDN2 and KDA 275M comparison",
        "",
        f"Generated: `{generated_at}`",
        "",
        (
            "Only bracketed curves enter the shared best-of plot. Every label is the "
            "observed-best LR; fitted minima are never substituted."
        ),
        "",
        "| Variant | Cx | Coverage | Curve | Observed best | Predicted fit (visual only) |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        best = row["observed_best"]
        observed = f"{best['loss']:.6f} @ {best['lr']:.2g}" if best else "—"
        predicted = (
            f"{row['predicted_fit_loss']:.6f} @ {row['predicted_fit_lr']:.2g}"
            if row["bracketed"]
            else "—"
        )
        curve = "bracketed†" if row["provisional"] else "bracketed" if row["bracketed"] else "—"
        lines.append(
            f"| {row['variant_label']} | Cx{row['cx']} | "
            f"{row['finished_points']}/{row['expected_points']} | {curve} | "
            f"{observed} | {predicted} |"
        )
    pending = sum(len(names) for names in unresolved.values())
    lines.extend(["", f"Uninitialized planned W&B runs: `{pending}`."])
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--refresh-stale-cache", action="store_true")
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Validate exact-name W&B registration without loading histories or plotting.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--results-path", type=Path)
    args = parser.parse_args()

    api = wandb.Api(timeout=90)
    canonical_gdn2, canonical_kda, unresolved = resolve_interventions(api, args.project)
    discovered = len(canonical_gdn2.runs) + len(canonical_kda.runs)
    print(f"Resolved {discovered}/32 planned runs by exact W&B display name.")
    for key, names in unresolved.items():
        print(f"  {key}: {len(names)} pending")
    if args.resolve_only:
        return

    wave = comparison_wave(canonical_gdn2, canonical_kda)
    points = load_points(
        wave,
        project=args.project,
        cache_dir=args.cache_dir,
        window_m=FINAL_WINDOW_M,
        include_running=False,
        refresh_cache=args.refresh_cache,
        refresh_stale_cache=args.refresh_stale_cache,
    )
    output_dir = args.output_dir or V2_DIR / "plots" / "pretraining" / wave.key
    results_path = args.results_path or V2_DIR / "results" / "pretraining" / wave.key / "results"
    variants = (
        WIDE_INTEGRATION,
        GEOMETRY_GDN_EV2_NOPE_GATED,
        GEOMETRY_GDN2_EV2_NOPE_GATED,
        canonical_gdn2,
        canonical_kda,
    )
    paths = [
        plot_intervention_uplot(
            points,
            replace(
                wave,
                title="Canonical GDN2 LR sweep",
                intervention_label=canonical_gdn2.label,
                intervention=canonical_gdn2,
                additional_baselines=(),
                uplot_baselines=False,
            ),
            "275m",
            output_dir / "gdn2_275m_uplot.png",
            FINAL_WINDOW_M,
        ),
        plot_intervention_uplot(
            points,
            replace(
                wave,
                title="Canonical KDA LR sweep",
                intervention_label=canonical_kda.label,
                intervention=canonical_kda,
                additional_baselines=(),
                uplot_baselines=False,
            ),
            "275m",
            output_dir / "kda_275m_uplot.png",
            FINAL_WINDOW_M,
        ),
        plot_shared_best_of(
            points,
            variants,
            (canonical_gdn2, canonical_kda),
            output_dir / "summary_observed_best.png",
        ),
    ]
    result_paths = write_comparison_results(points, variants, unresolved, results_path)
    print("\nWrote:")
    for path in (*paths, *result_paths):
        print(path)


if __name__ == "__main__":
    main()
