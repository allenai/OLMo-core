#!/usr/bin/env python3
"""Materialize Beaker specs and a launch plan for OLMoDDP conversion families."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from string import Template
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "JACOBM_DDP_PUBLICATION_MANIFEST.json"
DEFAULT_TEMPLATE = REPO_ROOT / "src/scripts/beaker/jacobm_olmo_ddp_family_template.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src/scripts/beaker/generated/olmo_ddp_conversion_families"
DEFAULT_EXCLUDED_FAMILIES = frozenset({"baseline"})
SAFE_FAMILY = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        help="Generate only this family (repeatable). By default all families except baseline are used.",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include baseline when --family is not provided.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def get_family_counts(manifest: dict[str, Any]) -> Counter[str]:
    entries = manifest.get("models")
    if not isinstance(entries, list):
        raise TypeError("Manifest 'models' must be a list")
    counts: Counter[str] = Counter()
    for entry in entries:
        family = entry.get("family")
        if not isinstance(family, str) or not SAFE_FAMILY.fullmatch(family):
            raise ValueError(f"Unsafe or invalid manifest family: {family!r}")
        counts[family] += 1
    if sum(counts.values()) != manifest.get("model_count"):
        raise ValueError("Manifest model_count does not match its model entries")
    return counts


def select_families(
    counts: Counter[str], requested: list[str] | None, *, include_baseline: bool
) -> list[str]:
    if requested:
        selected = sorted(set(requested))
    else:
        excluded = set() if include_baseline else DEFAULT_EXCLUDED_FAMILIES
        selected = sorted(set(counts) - excluded)
    unknown = sorted(set(selected) - set(counts))
    if unknown:
        raise ValueError(f"Unknown families: {', '.join(unknown)}")
    return selected


def render_spec(template: Template, *, family: str, model_count: int) -> str:
    return template.substitute(
        FAMILY=family,
        FAMILY_SLUG=family.replace("_", "-"),
        MODEL_COUNT=str(model_count),
        # Preserve shell variables in the YAML template while using string.Template.
        NEW_REPO="${NEW_REPO}",
        LEGACY_REPO="${LEGACY_REPO}",
        NEW_SOURCE="${NEW_SOURCE}",
        LEGACY_SOURCE="${LEGACY_SOURCE}",
        STATUS_ROOT="${STATUS_ROOT}",
        CREDENTIALS="${CREDENTIALS}",
    )


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    template_path = args.template.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = load_json(manifest_path)
    counts = get_family_counts(manifest)
    families = select_families(counts, args.families, include_baseline=args.include_baseline)
    if not families:
        raise ValueError("No families selected")

    template = Template(template_path.read_text(encoding="utf-8"))
    jobs: list[dict[str, Any]] = []
    for family in families:
        spec_name = f"jacobm_olmo_ddp_{family}_family.yaml"
        spec_path = output_dir / spec_name
        atomic_write(
            spec_path,
            render_spec(template, family=family, model_count=counts[family]),
        )
        jobs.append(
            {
                "family": family,
                "model_count": counts[family],
                "gpu_count": 1,
                "spec": spec_name,
                "experiment_name": (
                    f"jacobm-olmo-ddp-{family.replace('_', '-')}-family-v1-urgent"
                ),
            }
        )

    plan = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "workspace": "ai2/OLMo-3-moe-experiments",
        "cluster": "ai2/holmes",
        "priority": "urgent",
        "gpu_count_per_job": 1,
        "job_count": len(jobs),
        "total_gpu_count": sum(job["gpu_count"] for job in jobs),
        "total_model_count": sum(job["model_count"] for job in jobs),
        "excluded_running_families": sorted(set(counts) - set(families)),
        "jobs": jobs,
    }
    atomic_write(output_dir / "launch_plan.json", json.dumps(plan, indent=2) + "\n")

    print(
        f"Prepared {plan['job_count']} family jobs for {plan['total_model_count']} models "
        f"({plan['total_gpu_count']} GPUs if launched concurrently)."
    )
    for job in jobs:
        print(f"{job['family']}: {job['model_count']} models -> {output_dir / job['spec']}")
    print(f"Launch plan: {output_dir / 'launch_plan.json'}")


if __name__ == "__main__":
    main()
