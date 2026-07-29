#!/usr/bin/env python
"""Plot paper-matched KDA LatentMoE sweeps against the BF16 KDA parent."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import wandb
from plot_canonical_gdn2_kda import (
    KDA_EV2_NEG_ACTIVE_PARAMETERS,
    resolve_interventions,
)
from plot_pretraining_wave import (
    DEFAULT_CACHE_DIR,
    FINAL_WINDOW_M,
    PROJECT,
    V2_DIR,
    Point,
    RegisteredRun,
    Variant,
    Wave,
    _finished,
    _fit_minimum,
    load_points,
    plot_fixed_lr_scale_comparison,
    plot_intervention_uplot,
)

CXS = (1, 2, 4, 8)
LRS = (4e-4, 8e-4, 1.6e-3, 3.2e-3)
MODELS = ("275m", "480m", "810m", "1p2b")
L2_KEY = "geometry_kda_ev2_neg_nope_gated_latentmoe_l2_paper"
L4_KEY = "geometry_kda_ev2_neg_nope_gated_latentmoe_l4_paper"
ACTIVE_PARAMETERS = {
    L2_KEY: {"275m": 295_664_448},
    L4_KEY: {"275m": 296_770_368},
}


def _lr_name(lr: float) -> str:
    return {
        4e-4: "4e-4",
        8e-4: "8e-4",
        1.6e-3: "1p6e-3",
        3.2e-3: "3p2e-3",
    }[lr]


def _planned_names() -> dict[str, list[tuple[str, int, float, str]]]:
    planned = {L2_KEY: [], L4_KEY: []}
    for key, compression in ((L2_KEY, 2), (L4_KEY, 4)):
        for cx in CXS:
            for lr in LRS:
                planned[key].append(
                    (
                        "275m",
                        cx,
                        lr,
                        (
                            "pt-275m-kda-ev2-neg-nope-gated-latentmoe-"
                            f"l{compression}-paper-cx{cx}-lr{_lr_name(lr)}-r1"
                        ),
                    )
                )
    return planned


def resolve_latent_variants(
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

    unresolved = {L2_KEY: [], L4_KEY: []}

    def build(key: str, label: str, color: str) -> Variant:
        registered: list[RegisteredRun] = []
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
            registered.append(RegisteredRun(model, cx, lr, exact[0].id))
        return Variant(key=key, label=label, color=color, runs=tuple(registered))

    return (
        build(L2_KEY, "LatentMoE L=2 (paper-matched)", "#7c3aed"),
        build(L4_KEY, "LatentMoE L=4 (paper-matched)", "#0891b2"),
        unresolved,
    )


def comparison_wave(kda: Variant, l2: Variant, l4: Variant) -> Wave:
    return Wave(
        key="kda_latent_moe",
        title="KDA LatentMoE paper-matched comparison",
        intervention_label=l4.label,
        architecture_note=(
            "The parent is BF16 KDA with expand_v=2, negative eigenvalues, NoPE, "
            "and gated attention. Paper-matched LatentMoE keeps the full-width "
            "router and scales total/active experts with compression: L=2 uses "
            "512/top-16 and L=4 uses 1024/top-32."
        ),
        models=MODELS,
        lr_sweep_models=("275m",),
        active_parameters={"275m": ACTIVE_PARAMETERS[L4_KEY]["275m"]},
        baseline_active_parameters=KDA_EV2_NEG_ACTIVE_PARAMETERS,
        baseline=kda,
        additional_baselines=(l2,),
        intervention=l4,
        uplot_baselines=False,
        model_mode_labels={
            "275m": "observed-best LR sweep",
            "480m": "pending scale config",
            "810m": "pending scale config",
            "1p2b": "pending scale config",
        },
    )


def plot_best_of(
    points: list[Point],
    wave: Wave,
    l2: Variant,
    l4: Variant,
    output_path: Path,
) -> Path:
    eligible = {
        variant.key: {
            cx for cx in CXS if _fit_minimum(_finished(points, variant.key, "275m", cx)) is not None
        }
        for variant in (l2, l4)
    }
    filtered = [
        point
        for point in points
        if point.variant == wave.baseline.key
        or (
            point.model == "275m"
            and point.variant in eligible
            and point.cx in eligible[point.variant]
        )
    ]
    return plot_fixed_lr_scale_comparison(
        filtered,
        replace(
            wave,
            title="KDA vs paper-matched LatentMoE L=2/L=4 best-of",
        ),
        output_path,
        FINAL_WINDOW_M,
    )


def write_results(
    points: list[Point],
    kda: Variant,
    l2: Variant,
    l4: Variant,
    unresolved: dict[str, list[str]],
    output_path: Path,
) -> tuple[Path, Path]:
    generated_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for variant in (l2, l4):
        for cx in CXS:
            finished = _finished(points, variant.key, "275m", cx)
            fit = _fit_minimum(finished)
            best = min(finished, key=lambda point: point.loss) if finished else None
            rows.append(
                {
                    "variant": variant.key,
                    "variant_label": variant.label,
                    "cx": cx,
                    "finished_points": len(finished),
                    "expected_points": len(LRS),
                    "bracketed": fit is not None,
                    "provisional": fit is not None and len(finished) < len(LRS),
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
            "LatentMoE enters the best-of plot only after its 275M Cx curve "
            "brackets a valid quadratic minimum. The plotted point remains the "
            "best observed finished LR; the fitted prediction is diagnostic only."
        ),
        "active_parameters": ACTIVE_PARAMETERS,
        "baseline": kda.key,
        "unresolved_planned_runs": unresolved,
        "runs": [asdict(point) for point in points],
        "results": rows,
    }
    json_path = output_path.with_suffix(".json")
    md_path = output_path.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# KDA LatentMoE paper-matched pretraining",
        "",
        f"Generated: `{generated_at}`",
        "",
        (
            f"Metric: final `{FINAL_WINDOW_M}M`-token mean training CE. The "
            "best-of plot uses observed finished points only after a curve is bracketed."
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
    parser.add_argument("--resolve-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--results-path", type=Path)
    args = parser.parse_args()

    api = wandb.Api(timeout=90)
    _, _, kda, _ = resolve_interventions(api, args.project)
    l2, l4, unresolved = resolve_latent_variants(api, args.project)
    planned = _planned_names()
    print(
        f"Resolved L=2 {len(l2.runs)}/{len(planned[L2_KEY])}, "
        f"L=4 {len(l4.runs)}/{len(planned[L4_KEY])}, and "
        f"KDA parent {len(kda.runs)}/16 exact W&B names."
    )
    for key, names in unresolved.items():
        print(f"  {key}: {len(names)} pending")
    if args.resolve_only:
        return

    wave = comparison_wave(kda, l2, l4)
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
    paths = (
        plot_intervention_uplot(
            points,
            replace(
                wave,
                title="275M paper-matched LatentMoE L=2 LR sweep",
                intervention_label=l2.label,
                intervention=l2,
                additional_baselines=(),
            ),
            "275m",
            output_dir / "l2_275m_uplot.png",
            FINAL_WINDOW_M,
        ),
        plot_intervention_uplot(
            points,
            replace(
                wave,
                title="275M paper-matched LatentMoE L=4 LR sweep",
                intervention_label=l4.label,
                intervention=l4,
                additional_baselines=(),
            ),
            "275m",
            output_dir / "l4_275m_uplot.png",
            FINAL_WINDOW_M,
        ),
        plot_best_of(points, wave, l2, l4, output_dir / "best_of.png"),
        *write_results(points, kda, l2, l4, unresolved, results_path),
    )
    print("\nWrote:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
