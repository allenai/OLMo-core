#!/usr/bin/env python3
"""Materialize and validate legacy configs for every publication manifest model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "JACOBM_DDP_PUBLICATION_MANIFEST.json"
DEFAULT_OUTPUT = REPO_ROOT / "JACOBM_DDP_CONFIGS"
DEFAULT_LEGACY_REPO = Path("/weka/oe-adapt-default/jacobm/olmoe3/OLMo-core")
DEFAULT_HF_ROOT = Path("/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/hf-checkpoints")
BATCH_BY_CX = {1: 32, 2: 48, 4: 64, 8: 96}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--legacy-repo", type=Path, default=DEFAULT_LEGACY_REPO)
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument(
        "--stage", action="append", choices=("pretraining", "midtraining"), default=[]
    )
    parser.add_argument("--skip-checkpoint-validation", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_hash(config: dict[str, Any]) -> str:
    value = json.dumps(
        config["model"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(value).hexdigest()


def _archived_config(entry: dict[str, Any], hf_root: Path) -> Path:
    return hf_root / entry["source_run_name"] / f"step{entry['source_step']}" / "olmo_config.json"


def _trusted_config(entry: dict[str, Any], hf_root: Path) -> tuple[str, Path] | None:
    colocated = Path(entry["source_checkpoint"]) / "config.json"
    if colocated.is_file():
        return "colocated", colocated
    archived = _archived_config(entry, hf_root)
    if archived.is_file():
        return "archived_hf", archived
    return None


def _builder_for_family(legacy_repo: Path, family: str) -> tuple[Path, list[str]]:
    ladder = legacy_repo / "src/scripts/train/jacobm_olmoe_ladder"
    base = ladder / "moe_a0_ladder.py"
    mapping = {
        "baseline": (base, []),
        "expert_coarse_24e_top2": (base, ["--expert-geometry=coarse_24e_top2"]),
        "expert_fine_96e_top8": (base, ["--expert-geometry=fine_96e_top8"]),
        "sparsity_high_96e_top4": (
            base,
            ["--total-sparsity=high_total_96e_top4"],
        ),
        "sparsity_huge_192e_top4": (
            base,
            ["--total-sparsity=huge_total_192e_top4"],
        ),
        "shared_no_shared": (
            base,
            ["--shared-expert-config=no_shared_matched_active"],
        ),
        "dense0_shared": (base, ["--dense-schedule=dense0_shared"]),
        "dense2_shared": (base, ["--dense-schedule=dense2_shared"]),
        "dense4_shared": (base, ["--dense-schedule=dense4_shared"]),
        "qwen_active_4p5d": (
            ladder / "experiments/qwen3_like/qwen3_like_ladder.py",
            ["--qwen3-like=active_matched"],
        ),
        "qwen_true_3d": (
            ladder / "experiments/qwen3_like/qwen3_like_ladder.py",
            ["--qwen3-like=true_3d_depth_matched"],
        ),
        "integration_wide": (
            ladder / "experiments/integration/integration_ladder.py",
            ["--integration-config=wide_256e8k"],
        ),
        "integration_deep": (
            ladder / "experiments/integration/integration_ladder.py",
            ["--integration-config=deep_256e8k"],
        ),
    }
    try:
        return mapping[family]
    except KeyError as exc:
        raise ValueError(f"No legacy config builder mapping for family {family!r}") from exc


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import legacy builder {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _reconstruct_config(
    entry: dict[str, Any], legacy_repo: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if entry["stage"] != "pretraining":
        raise ValueError(f"No reconstruction recipe for {entry['id']}; expected exact MT config")
    legacy_src = str(legacy_repo / "src")
    if legacy_src not in sys.path:
        sys.path.insert(0, legacy_src)
    script, family_args = _builder_for_family(legacy_repo, entry["family"])
    cx = int(entry["data_multiple"])
    common = [
        f"--model-size={entry['model_size']}",
        "--save-folder=/tmp/reconstructed-config-only",
        f"--name={entry['source_run_name']}",
        "--data-root=s3://ai2-llm",
        f"--lr={entry['learning_rate']}",
        f"--chinchilla-multiple={cx}",
        f"--global-batch-size-seq={BATCH_BY_CX[cx]}",
        "--num-nodes=1",
        "--gpus-per-node=1",
        "--micro-batch-size=1",
        "--ep-dim=1",
        "--ladder-evals",
        "--eval-task-set=fast",
        "--eval-interval=2000",
        "--save-interval=999999999",
        "--ephemeral-save-interval=500",
        "--no-pre-train-checkpoint",
        f"--tag={entry['source_run_name']}",
    ]
    argv = [*common, *family_args]
    module_name = "_legacy_config_" + entry["id"].replace("/", "_")
    module = _load_module(script, module_name)
    opts, overrides = module.get_parser().parse_known_args(argv)
    config = module.build_config(opts, overrides).as_config_dict()
    return config, {"builder": str(script), "argv": argv}


def _validate_with_legacy_code(
    *,
    legacy_repo: Path,
    checkpoint: Path,
    config: Path,
    output: Path,
) -> dict[str, Any]:
    validator = REPO_ROOT / "src/scripts/validate_legacy_moe_checkpoint_config.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(legacy_repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        str(validator),
        str(checkpoint),
        "--config",
        str(config),
        "--output",
        str(output),
    ]
    process = subprocess.run(
        command,
        cwd=legacy_repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Legacy config validation failed for {checkpoint}:\n{process.stdout}")
    return _load_json(output)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    legacy_repo = args.legacy_repo.expanduser().resolve()
    hf_root = args.hf_root.expanduser().resolve()
    manifest = _load_json(manifest_path)
    entries = list(manifest["models"])
    if args.family:
        entries = [entry for entry in entries if entry["family"] in set(args.family)]
    if args.stage:
        entries = [entry for entry in entries if entry["stage"] in set(args.stage)]
    if not entries:
        raise ValueError("No manifest entries matched the requested filters")

    trusted_hashes: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for entry in manifest["models"]:
        trusted = _trusted_config(entry, hf_root)
        if trusted is not None:
            trusted_hashes[(entry["stage"], entry["family"], entry["model_size"])].add(
                _model_hash(_load_json(trusted[1]))
            )

    records: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        directory = (
            output_root
            / entry["stage"]
            / entry["family"]
            / entry["model_size"]
            / f"cx{entry['data_multiple']}"
        )
        config_path = directory / "config.json"
        provenance_path = directory / "provenance.json"
        validation_path = directory / "schema_validation.json"
        if config_path.exists() and provenance_path.exists() and not args.force:
            provenance = _load_json(provenance_path)
            if provenance.get("manifest_id") == entry["id"]:
                print(f"[{index}/{len(entries)}] reuse {entry['id']}", flush=True)
                if not args.skip_checkpoint_validation:
                    validation = _validate_with_legacy_code(
                        legacy_repo=legacy_repo,
                        checkpoint=Path(entry["source_checkpoint"]),
                        config=config_path,
                        output=validation_path,
                    )
                    provenance["schema_validation"] = validation["status"]
                    _write_json(provenance_path, provenance)
                records.append(provenance)
                continue

        trusted = _trusted_config(entry, hf_root)
        reconstruction: dict[str, Any] | None = None
        if trusted is not None:
            resolution, origin_path = trusted
            config = _load_json(origin_path)
        else:
            resolution = "reconstructed_from_legacy_builder"
            origin_path = None
            config, reconstruction = _reconstruct_config(entry, legacy_repo)
            allowed = trusted_hashes[(entry["stage"], entry["family"], entry["model_size"])]
            if allowed and _model_hash(config) not in allowed:
                raise ValueError(
                    f"Reconstructed model config for {entry['id']} differs from "
                    "trusted configs in the same stage/family/model-size group"
                )

        print(f"[{index}/{len(entries)}] {resolution} {entry['id']}", flush=True)
        _write_json(config_path, config)
        validation_status = "skipped"
        if not args.skip_checkpoint_validation:
            validation = _validate_with_legacy_code(
                legacy_repo=legacy_repo,
                checkpoint=Path(entry["source_checkpoint"]),
                config=config_path,
                output=validation_path,
            )
            validation_status = validation["status"]

        provenance = {
            "manifest_id": entry["id"],
            "source_checkpoint": entry["source_checkpoint"],
            "resolution": resolution,
            "origin_config": str(origin_path) if origin_path is not None else None,
            "origin_config_sha256": (_sha256(origin_path) if origin_path is not None else None),
            "materialized_config": str(config_path),
            "materialized_config_sha256": _sha256(config_path),
            "model_config_sha256": _model_hash(config),
            "schema_validation": validation_status,
            "reconstruction": reconstruction,
        }
        _write_json(provenance_path, provenance)
        records.append(provenance)

    counts = Counter(record["resolution"] for record in records)
    validation_counts = Counter(record["schema_validation"] for record in records)
    index = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "legacy_repo": str(legacy_repo),
        "model_count": len(records),
        "resolution_counts": dict(sorted(counts.items())),
        "validation_counts": dict(sorted(validation_counts.items())),
        "filters": {"families": args.family, "stages": args.stage},
        "models": records,
    }
    index_name = "INDEX.json" if not args.family and not args.stage else "INDEX.filtered.json"
    _write_json(output_root / index_name, index)
    print(f"Materialized {len(records)} configs in {output_root}", flush=True)


if __name__ == "__main__":
    main()
