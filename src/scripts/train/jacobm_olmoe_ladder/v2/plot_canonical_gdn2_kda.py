#!/usr/bin/env python
"""Plot the canonical GDN2/KDA sweeps and recurrent-mixer scale transfers.

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
    plot_fixed_lr_scale_comparison,
    plot_intervention_uplot,
    plot_observed_best_summary,
    write_results,
)

CXS = (1, 2, 4, 8)
LRS = (4e-4, 8e-4, 1.6e-3, 3.2e-3)
MODELS = ("275m", "480m", "810m", "1p2b")
GDN2_KEY = "geometry_gdn2_ev1_noneg_nope_gated"
KDA_KEY = "geometry_kda_ev1_noneg_nope_gated"
KDA_EV2_NEG_KEY = "geometry_kda_ev2_neg_nope_gated"
EXPECTED_SWEEP_POINTS = 4
SCALE_LRS = {
    ("480m", 1): 1.2e-3,
    ("480m", 2): 9e-4,
    ("480m", 4): 8e-4,
    ("480m", 8): 8e-4,
    ("810m", 1): 6e-4,
    ("810m", 2): 5.6e-4,
    ("810m", 4): 4e-4,
    ("810m", 8): 4e-4,
    ("1p2b", 1): 4e-4,
    ("1p2b", 2): 6e-4,
    ("1p2b", 4): 3e-4,
    ("1p2b", 8): 4e-4,
}
CANONICAL_GDN2_ACTIVE_PARAMETERS = {
    "275m": 284_915_520,
    "480m": 489_954_144,
    "810m": 823_189_952,
    "1p2b": 1_228_949_248,
}
KDA_EV2_NEG_ACTIVE_PARAMETERS = {
    "275m": 290_503_488,
    "480m": 498_741_600,
    "810m": 839_239_616,
    "1p2b": 1_251_462_912,
}
WIDE_ACTIVE_PARAMETERS = {
    "275m": 280_207_872,
    "480m": 486_348_800,
    "810m": 823_569_920,
    "1p2b": 1_225_011_712,
}
EXPLICIT_RESUME_CHAINS = {
    "pt-480m-geometry-hybrid-gdn2-ev1-noneg-nope-gated-cx2-lr9e-4-r1": (
        "2ui4npyk",
        "tjmfr7de",
        "2140i574",
        "w9d4rof7",
        "kzug2rko",
        "u1qy19e5",
        "eiuqm2ne",
    ),
    "pt-1p2b-geometry-hybrid-gdn2-ev1-noneg-nope-gated-cx1-lr4e-4-ep8-sync-r1": (
        "1odf2b6k",
        "cqvqih3h",
        "lvx2cb1m",
        "0lfktqhf",
        "aknoel8q",
        "8lmvitbp",
        "ji0e0rcg",
    ),
}
LOCAL_HISTORY_RECOVERIES = {
    "pt-1p2b-geometry-hybrid-gdn2-ev1-noneg-nope-gated-cx8-lr4e-4-ep8-sync-r1": (
        "results/pretraining/canonical_gdn2_kda/recovered_histories/dlerge4x.json"
    ),
}


def _lr_name(lr: float) -> str:
    return {
        4e-4: "4e-4",
        8e-4: "8e-4",
        1.6e-3: "1p6e-3",
        3.2e-3: "3p2e-3",
    }[lr]


def _scale_lr_name(lr: float) -> str:
    return {
        1.6e-3: "1p6e-3",
        1.2e-3: "1p2e-3",
        9e-4: "9e-4",
        8e-4: "8e-4",
        6e-4: "6e-4",
        5.6e-4: "5p6e-4",
        4e-4: "4e-4",
        3e-4: "3e-4",
    }[lr]


def _planned_display_names() -> dict[str, list[tuple[str, int, float, str]]]:
    planned = {GDN2_KEY: [], KDA_KEY: [], KDA_EV2_NEG_KEY: []}
    for cx in CXS:
        for lr in LRS:
            tag = _lr_name(lr)
            planned[GDN2_KEY].append(
                (
                    "275m",
                    cx,
                    lr,
                    f"pt-275m-geometry-hybrid-gdn2-ev1-noneg-nope-gated-cx{cx}-lr{tag}-r1",
                )
            )
            planned[KDA_KEY].append(
                (
                    "275m",
                    cx,
                    lr,
                    f"pt-275m-geometry-kda-ev1-noneg-nope-gated-cx{cx}-lr{tag}",
                )
            )
    for (model, cx), lr in SCALE_LRS.items():
        ep_suffix = "-ep8-sync" if model == "1p2b" else ""
        planned[GDN2_KEY].append(
            (
                model,
                cx,
                lr,
                (
                    f"pt-{model}-geometry-hybrid-gdn2-ev1-noneg-nope-gated-"
                    f"cx{cx}-lr{_scale_lr_name(lr)}{ep_suffix}-r1"
                ),
            )
        )
        if model == "480m":
            planned[KDA_KEY].append(
                (
                    model,
                    cx,
                    lr,
                    (
                        f"pt-{model}-geometry-hybrid-kda-ev1-noneg-nope-gated-"
                        f"cx{cx}-lr{_scale_lr_name(lr)}-r1"
                    ),
                )
            )
    gdn1_275m_lrs = {1: 8e-4, 2: 1.6e-3, 4: 8e-4, 8: 8e-4}
    for cx, lr in gdn1_275m_lrs.items():
        planned[KDA_EV2_NEG_KEY].append(
            (
                "275m",
                cx,
                lr,
                (
                    f"pt-275m-geometry-hybrid-kda-ev2-neg-nope-gated-"
                    f"cx{cx}-lr{_scale_lr_name(lr)}-r1"
                ),
            )
        )
    for model in ("480m", "810m", "1p2b"):
        for cx in CXS:
            lr = SCALE_LRS[(model, cx)]
            ep_suffix = "-ep8-sync" if model == "1p2b" else ""
            planned[KDA_EV2_NEG_KEY].append(
                (
                    model,
                    cx,
                    lr,
                    (
                        f"pt-{model}-geometry-hybrid-kda-ev2-neg-nope-gated-"
                        f"cx{cx}-lr{_scale_lr_name(lr)}{ep_suffix}-r1"
                    ),
                )
            )
    return planned


def resolve_interventions(
    api: Any,
    project: str,
) -> tuple[Variant, Variant, Variant, dict[str, list[str]]]:
    """Resolve exact W&B names once and reject ambiguous histories."""

    planned = _planned_display_names()
    display_names = [name for rows in planned.values() for _, _, _, name in rows]
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

    unresolved: dict[str, list[str]] = {
        GDN2_KEY: [],
        KDA_KEY: [],
        KDA_EV2_NEG_KEY: [],
    }

    def build_variant(key: str, label: str, color: str) -> Variant:
        registered: list[RegisteredRun] = []
        for model, cx, lr, display_name in planned[key]:
            exact = by_name[display_name]
            if not exact:
                unresolved[key].append(display_name)
                continue
            exact_ids = {run.id for run in exact}
            resume_chain = EXPLICIT_RESUME_CHAINS.get(display_name)
            if resume_chain is not None:
                if exact_ids != set(resume_chain):
                    ids = ", ".join(sorted(exact_ids))
                    raise RuntimeError(
                        f"{display_name!r} resume-chain registry is stale; "
                        f"W&B currently resolves ({ids})"
                    )
                registered.append(
                    RegisteredRun(
                        model,
                        cx,
                        lr,
                        resume_chain[-1],
                        predecessor_run_ids=resume_chain[:-1],
                        recovered_history_path=LOCAL_HISTORY_RECOVERIES.get(display_name),
                    )
                )
                continue
            if len(exact) != 1:
                ids = ", ".join(sorted(exact_ids))
                raise RuntimeError(
                    f"{display_name!r} resolved to multiple W&B runs ({ids}); "
                    "register the intended run and predecessor segments explicitly"
                )
            registered.append(
                RegisteredRun(
                    model,
                    cx,
                    lr,
                    exact[0].id,
                    recovered_history_path=LOCAL_HISTORY_RECOVERIES.get(display_name),
                )
            )
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
    kda_ev2_neg = build_variant(
        KDA_EV2_NEG_KEY,
        "KDA (expand_v=2, negative eigenvalues)",
        "#d97706",
    )
    return canonical_gdn2, canonical_kda, kda_ev2_neg, unresolved


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


def canonical_scale_wave(canonical_gdn2: Variant, canonical_kda: Variant) -> Wave:
    return Wave(
        key="canonical_gdn2_scale",
        title="Canonical GDN2 and KDA fixed-LR scale comparison",
        intervention_label="canonical GDN2 (expand_v=1, nonnegative)",
        architecture_note=(
            "Canonical expand_v=1, nonnegative GDN2 and KDA in the matching "
            "gated-NoPE geometry. The 275M panel uses each architecture's "
            "observed-optimal LR sweep; larger models use the corresponding "
            "transferred wide-integration LR. KDA is currently planned at 480M."
        ),
        models=MODELS,
        lr_sweep_models=("275m",),
        active_parameters=CANONICAL_GDN2_ACTIVE_PARAMETERS,
        baseline_active_parameters=WIDE_ACTIVE_PARAMETERS,
        baseline=WIDE_INTEGRATION,
        additional_baselines=(
            GEOMETRY_GDN_EV2_NOPE_GATED,
            GEOMETRY_GDN2_EV2_NOPE_GATED,
            canonical_kda,
        ),
        intervention=canonical_gdn2,
        uplot_baselines=False,
    )


def kda_ev2_neg_scale_wave(
    canonical_gdn2: Variant,
    canonical_kda: Variant,
    kda_ev2_neg: Variant,
) -> Wave:
    return Wave(
        key="kda_ev2_neg_scale",
        title="KDA expand_v=2 negative-eigenvalue fixed-LR transfer",
        intervention_label=kda_ev2_neg.label,
        architecture_note=(
            "KDA uses the matching gated-NoPE geometry with expand_v=2 and "
            "negative eigenvalues. The 275M cells use the observed-best LRs "
            "from matching GDN1; larger sizes use transferred wide-integration LRs."
        ),
        models=MODELS,
        lr_sweep_models=(),
        active_parameters=KDA_EV2_NEG_ACTIVE_PARAMETERS,
        baseline_active_parameters={
            model: WIDE_ACTIVE_PARAMETERS[model] for model in MODELS
        },
        baseline=WIDE_INTEGRATION,
        additional_baselines=(
            GEOMETRY_GDN_EV2_NOPE_GATED,
            GEOMETRY_GDN2_EV2_NOPE_GATED,
            canonical_gdn2,
            canonical_kda,
        ),
        intervention=kda_ev2_neg,
        uplot_baselines=False,
        model_mode_labels={
            "275m": "GDN1-LR transfer",
            "480m": "wide-LR transfer",
            "810m": "wide-LR transfer",
        },
    )


def _variant_expected_count(variant: Variant, model: str, cx: int) -> int:
    if model == "275m" and variant.key in {GDN2_KEY, KDA_KEY}:
        return EXPECTED_SWEEP_POINTS
    return _expected_count(variant, model, cx)


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
        if len(_finished(points, variant.key, "275m", cx))
        < _variant_expected_count(variant, "275m", cx)
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
        legend_columns=2,
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
            expected = _variant_expected_count(variant, "275m", cx)
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
    canonical_gdn2, canonical_kda, kda_ev2_neg, unresolved = resolve_interventions(
        api,
        args.project,
    )
    discovered = len(canonical_gdn2.runs) + len(canonical_kda.runs) + len(kda_ev2_neg.runs)
    planned = sum(len(rows) for rows in _planned_display_names().values())
    print(f"Resolved {discovered}/{planned} planned runs by exact W&B display name.")
    for key, names in unresolved.items():
        print(f"  {key}: {len(names)} pending")
    if args.resolve_only:
        return

    wave = comparison_wave(canonical_gdn2, canonical_kda)
    scale_wave = canonical_scale_wave(canonical_gdn2, canonical_kda)
    kda_ev2_neg_wave = kda_ev2_neg_scale_wave(
        canonical_gdn2,
        canonical_kda,
        kda_ev2_neg,
    )
    points = load_points(
        wave,
        project=args.project,
        cache_dir=args.cache_dir,
        window_m=FINAL_WINDOW_M,
        include_running=False,
        refresh_cache=args.refresh_cache,
        refresh_stale_cache=args.refresh_stale_cache,
    )
    scale_points = load_points(
        scale_wave,
        project=args.project,
        cache_dir=args.cache_dir,
        window_m=FINAL_WINDOW_M,
        include_running=False,
        refresh_cache=args.refresh_cache,
        refresh_stale_cache=args.refresh_stale_cache,
    )
    kda_ev2_neg_points = load_points(
        kda_ev2_neg_wave,
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
        plot_fixed_lr_scale_comparison(
            scale_points,
            scale_wave,
            output_dir / "gdn2_fixed_lr_scale_comparison.png",
            FINAL_WINDOW_M,
        ),
        plot_fixed_lr_scale_comparison(
            kda_ev2_neg_points,
            kda_ev2_neg_wave,
            output_dir / "kda_ev2_neg_fixed_lr_scale_comparison.png",
            FINAL_WINDOW_M,
        ),
    ]
    sweep_unresolved = {
        key: [name for name in names if name.startswith("pt-275m-")]
        for key, names in unresolved.items()
        if key in {GDN2_KEY, KDA_KEY}
    }
    result_paths = (
        *write_comparison_results(points, variants, sweep_unresolved, results_path),
        *write_results(
            scale_points,
            scale_wave,
            results_path.with_name("scale_results"),
            FINAL_WINDOW_M,
        ),
        *write_results(
            kda_ev2_neg_points,
            kda_ev2_neg_wave,
            results_path.with_name("kda_ev2_neg_scale_results"),
            FINAL_WINDOW_M,
        ),
    )
    print("\nWrote:")
    for path in (*paths, *result_paths):
        print(path)


if __name__ == "__main__":
    main()
