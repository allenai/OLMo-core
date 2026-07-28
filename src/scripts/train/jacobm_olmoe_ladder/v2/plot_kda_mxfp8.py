#!/usr/bin/env python
"""Plot the 275M aggressive-MXFP8 KDA sweep against matching BF16 KDA."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import wandb
from plot_pretraining_wave import (
    DEFAULT_CACHE_DIR,
    FINAL_WINDOW_M,
    PROJECT,
    V2_DIR,
    RegisteredRun,
    SummaryVariant,
    Variant,
    Wave,
    _finished,
    _fit_minimum,
    load_points,
    plot_intervention_uplot,
    plot_observed_best_summary,
    write_results,
)

CXS = (1, 2, 4, 8)
MODELS = ("275m", "480m", "810m", "1p2b")
LRS = (4e-4, 8e-4, 1.6e-3, 3.2e-3)
BF16_KEY = "geometry_kda_ev2_neg_nope_gated"
MXFP8_KEY = "geometry_kda_ev2_neg_nope_gated_mxfp8_672"
BF16_LRS = {
    ("275m", 1): 8e-4,
    ("275m", 2): 1.6e-3,
    ("275m", 4): 8e-4,
    ("275m", 8): 8e-4,
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
MXFP8_TRANSFER_LRS = {
    ("480m", 1): 1.2e-3,
    ("480m", 2): 9e-4,
    ("480m", 4): 8e-4,
    ("480m", 8): 8e-4,
}
BF16_ACTIVE_PARAMETERS = 290_503_488
MXFP8_ACTIVE_PARAMETERS = 291_885_888
LOCAL_HISTORY_RECOVERIES = {
    "pt-275m-kda-ev2-neg-nope-gated-mxfp8-672-cx1-lr8e-4-r1": (
        "results/pretraining/kda_mxfp8/recovered_histories/uzg7z0t2.json"
    ),
}


def _lr_name(lr: float) -> str:
    return {
        3e-4: "3e-4",
        4e-4: "4e-4",
        5.6e-4: "5p6e-4",
        6e-4: "6e-4",
        8e-4: "8e-4",
        9e-4: "9e-4",
        1.2e-3: "1p2e-3",
        1.6e-3: "1p6e-3",
        3.2e-3: "3p2e-3",
    }[lr]


def _planned_names() -> dict[str, list[tuple[str, int, float, str]]]:
    planned = {BF16_KEY: [], MXFP8_KEY: []}
    for (model, cx), lr in BF16_LRS.items():
        ep_suffix = "-ep8-rowwise-ext3" if model == "1p2b" else ""
        planned[BF16_KEY].append(
            (
                model,
                cx,
                lr,
                (
                    f"pt-{model}-geometry-hybrid-kda-ev2-neg-nope-gated-"
                    f"cx{cx}-lr{_lr_name(lr)}{ep_suffix}-r1"
                ),
            )
        )
    for cx in CXS:
        for lr in LRS:
            planned[MXFP8_KEY].append(
                (
                    "275m",
                    cx,
                    lr,
                    (f"pt-275m-kda-ev2-neg-nope-gated-mxfp8-672-cx{cx}-lr{_lr_name(lr)}-r1"),
                )
            )
    for (model, cx), lr in MXFP8_TRANSFER_LRS.items():
        planned[MXFP8_KEY].append(
            (
                model,
                cx,
                lr,
                (f"pt-{model}-kda-ev2-neg-nope-gated-mxfp8-832-cx{cx}-lr{_lr_name(lr)}-r1"),
            )
        )
    return planned


def resolve_variants(
    api: Any,
    project: str,
) -> tuple[Variant, Variant, dict[str, list[str]]]:
    planned = _planned_names()
    names = [name for rows in planned.values() for _, _, _, name in rows]
    matches = list(
        api.runs(
            project,
            filters={"display_name": {"$in": names}},
            per_page=max(50, len(names)),
        )
    )
    by_name: dict[str, list[Any]] = {name: [] for name in names}
    for run in matches:
        if run.display_name in by_name:
            by_name[run.display_name].append(run)

    unresolved = {BF16_KEY: [], MXFP8_KEY: []}

    def build(key: str, label: str, color: str) -> Variant:
        registered = []
        for model, cx, lr, name in planned[key]:
            exact = by_name[name]
            if not exact:
                unresolved[key].append(name)
                continue
            if len(exact) != 1:
                ids = ", ".join(sorted(run.id for run in exact))
                raise RuntimeError(
                    f"{name!r} resolved to multiple W&B runs ({ids}); "
                    "register the intended resume chain explicitly"
                )
            registered.append(
                RegisteredRun(
                    model,
                    cx,
                    lr,
                    exact[0].id,
                    recovered_history_path=LOCAL_HISTORY_RECOVERIES.get(name),
                )
            )
        return Variant(key=key, label=label, color=color, runs=tuple(registered))

    bf16 = build(
        BF16_KEY,
        "BF16 KDA (expand_v=2, negative eigenvalues; transferred LR)",
        "#d97706",
    )
    mxfp8 = build(
        MXFP8_KEY,
        "aggressive MXFP8 KDA (32-aligned experts, fused_v2/FA4)",
        "#7c3aed",
    )
    return bf16, mxfp8, unresolved


def comparison_wave(bf16: Variant, mxfp8: Variant) -> Wave:
    return Wave(
        key="kda_mxfp8_275m",
        title="275M KDA aggressive-MXFP8 comparison",
        intervention_label=mxfp8.label,
        architecture_note=(
            "Aggressive MXFP8 uses the same KDA expand_v=2/negative-eigenvalue "
            "recipe as the BF16 reference. It rounds each expert width to its "
            "audited 32-aligned value and enables fused_v2/FlashAttention-4, so "
            "the comparison does not isolate precision alone."
        ),
        models=("275m",),
        lr_sweep_models=("275m",),
        active_parameters={
            "275m": MXFP8_ACTIVE_PARAMETERS,
            "480m": 496_253_280,
        },
        baseline_active_parameters={
            "275m": BF16_ACTIVE_PARAMETERS,
            "480m": 498_741_600,
        },
        baseline=bf16,
        intervention=mxfp8,
        uplot_baselines=False,
    )


def plot_best_of(points: list[Any], output_path: Path) -> Path:
    eligible = {
        cx for cx in CXS if _fit_minimum(_finished(points, MXFP8_KEY, "275m", cx)) is not None
    }
    # Keep every finished transferred-LR BF16 KDA reference visible while the
    # MXFP8 sweep is still filling in. Only the MXFP8 series is gated on a
    # bracketed quadratic curve; filtering the whole point set here would hide
    # valid KDA scale references until their MXFP8 counterparts finish.
    filtered = [
        point
        for point in points
        if point.variant == BF16_KEY
        or (point.variant == MXFP8_KEY and (point.model != "275m" or point.cx in eligible))
    ]
    provisional = {
        ("275m", cx, MXFP8_KEY)
        for cx in eligible
        if len(_finished(points, MXFP8_KEY, "275m", cx)) < len(LRS)
    }
    plot_observed_best_summary(
        filtered,
        out_path=output_path,
        title="KDA: BF16 transfer vs aggressive MXFP8",
        variants=(
            SummaryVariant(
                BF16_KEY,
                (BF16_KEY,),
                "BF16 KDA (our settings; transferred LR)",
                color="#d97706",
                linestyle="--",
            ),
            SummaryVariant(
                MXFP8_KEY,
                (MXFP8_KEY,),
                "aggressive MXFP8 KDA (observed best)",
                color="#7c3aed",
            ),
        ),
        window_m=FINAL_WINDOW_M,
        provisional_points=provisional,
        legend_columns=1,
        models=MODELS,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--refresh-stale-cache", action="store_true")
    parser.add_argument("--resolve-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--results-path", type=Path)
    args = parser.parse_args()

    api = wandb.Api(timeout=90)
    planned = _planned_names()
    bf16, mxfp8, unresolved = resolve_variants(api, args.project)
    print(
        f"Resolved BF16={len(bf16.runs)}/{len(planned[BF16_KEY])} and "
        f"MXFP8={len(mxfp8.runs)}/{len(planned[MXFP8_KEY])} "
        "exact W&B names."
    )
    for key, names in unresolved.items():
        print(f"  {key}: {len(names)} pending")
    if args.resolve_only:
        return

    load_wave = comparison_wave(bf16, mxfp8)
    load_wave = replace(load_wave, models=MODELS)
    wave = replace(load_wave, models=("275m",))
    points = load_points(
        load_wave,
        project=args.project,
        cache_dir=args.cache_dir,
        window_m=FINAL_WINDOW_M,
        include_running=False,
        refresh_cache=args.refresh_cache,
        refresh_stale_cache=args.refresh_stale_cache,
    )
    points_275m = [point for point in points if point.model == "275m"]
    output_dir = args.output_dir or V2_DIR / "plots" / "pretraining" / "kda_mxfp8"
    results_path = args.results_path or V2_DIR / "results" / "pretraining" / "kda_mxfp8" / "results"
    paths = (
        plot_intervention_uplot(
            points_275m,
            wave,
            "275m",
            output_dir / "275m_uplot.png",
            FINAL_WINDOW_M,
        ),
        plot_best_of(points, output_dir / "best_of.png"),
        *write_results(points_275m, wave, results_path, FINAL_WINDOW_M),
    )
    print("\nWrote:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
