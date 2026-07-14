#!/usr/bin/env python3
"""Run one resumable legacy-to-OLMoDDP conversion family through publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "JACOBM_DDP_PUBLICATION_MANIFEST.json"
DEFAULT_CONFIG_ROOT = REPO_ROOT / "JACOBM_DDP_CONFIGS"
DEFAULT_LEGACY_REPO = Path("/weka/oe-adapt-default/jacobm/olmoe3/OLMo-core")
DEFAULT_STATUS_ROOT = Path(
    "/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/converted-checkpoints/_family_status"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--legacy-repo", type=Path, default=DEFAULT_LEGACY_REPO)
    parser.add_argument("--new-repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--status-root", type=Path, default=DEFAULT_STATUS_ROOT)
    parser.add_argument("--results-dir", type=Path, default=Path("/results"))
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=6198)
    parser.add_argument("--load-thread-count", type=int, default=8)
    parser.add_argument("--save-thread-count", type=int, default=8)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == value:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_content_sha256(path: Path) -> str:
    value = json.dumps(load_json(path), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def same_path(left: str | Path | None, right: Path) -> bool:
    if left is None:
        return False
    return Path(left).expanduser().resolve() == right.expanduser().resolve()


def repo_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo / "src") + (os.pathsep + existing if existing else "")
    return env


def run_command(
    command: list[str],
    *,
    log_path: Path,
    cwd: Path,
    env: dict[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = shlex.join(command)
    print(f"RUN {rendered}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n$ {rendered}\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def schema_valid(report_path: Path, *, source: Path, config: Path) -> bool:
    if not report_path.is_file():
        return False
    try:
        report = load_json(report_path)
    except Exception:
        return False
    return bool(
        report.get("status") == "LEGACY_CONFIG_SCHEMA_MATCH"
        and same_path(report.get("checkpoint"), source)
        and report.get("config_sha256") == file_sha256(config)
        and report.get("keys_exact") is True
        and report.get("numel_exact") is True
        and report.get("dtypes_exact") is True
    )


def conversion_valid(target: Path, *, source: Path, config: Path) -> bool:
    required = [
        target / "model_and_optim/.metadata",
        target / "source_config.json",
        target / "config.json",
        target / "conversion_manifest.json",
    ]
    if not all(path.is_file() for path in required):
        return False
    try:
        report = load_json(target / "conversion_manifest.json")
    except Exception:
        return False
    return bool(
        same_path(report.get("source_checkpoint"), source)
        and report.get("optimizer_state_included") is False
        and report.get("trainer_state_included") is False
        and json_content_sha256(target / "source_config.json") == json_content_sha256(config)
    )


def strict_valid(report_path: Path, *, source: Path, target: Path, config: Path) -> bool:
    if not report_path.is_file():
        return False
    try:
        report = load_json(report_path)
    except Exception:
        return False
    return bool(
        report.get("status") == "STRICT_TENSOR_MATCH"
        and report.get("bitwise_equal") is True
        and report.get("target_model_only") is True
        and report.get("optimizer_state_included") is False
        and report.get("trainer_state_included") is False
        and same_path(report.get("source_checkpoint"), source)
        and same_path(report.get("target_checkpoint"), target)
        and report.get("source_config_sha256") == file_sha256(config)
        and report.get("source_numel") == report.get("target_numel")
    )


def logits_valid(
    report_path: Path,
    *,
    source: Path,
    target: Path,
    source_config: Path,
    target_config: Path,
    sequence_length: int,
) -> bool:
    if not report_path.is_file() or not source_config.is_file() or not target_config.is_file():
        return False
    try:
        report = load_json(report_path)
        reference = report["reference_metadata"]
        candidate = report["candidate_metadata"]
    except Exception:
        return False
    return bool(
        report.get("status") == "LOGITS_MATCH"
        and report.get("bitwise_equal") is True
        and report.get("intermediates_bitwise_equal") is True
        and report.get("exact_match") is True
        and report.get("atol") == 0
        and report.get("rtol") == 0
        and same_path(reference.get("checkpoint"), source)
        and same_path(candidate.get("checkpoint"), target)
        and reference.get("config_sha256") == file_sha256(source_config)
        and candidate.get("config_sha256") == file_sha256(target_config)
        and reference.get("sequence_length") == sequence_length
        and candidate.get("sequence_length") == sequence_length
        and reference.get("captured_intermediate_count", 0) > 0
        and candidate.get("captured_intermediate_count", 0) > 0
        and reference.get("grad_enabled") is True
        and candidate.get("grad_enabled") is True
    )


def select_schema_report(*, target: Path, status_dir: Path, source: Path, config: Path) -> Path:
    candidates = [
        target / "legacy_config_schema_validation.json",
        status_dir / "legacy_config_schema_validation.json",
        config.parent / "schema_validation.json",
    ]
    for candidate in candidates:
        if schema_valid(candidate, source=source, config=config):
            return candidate
    return status_dir / "legacy_config_schema_validation.json"


def render_readme(
    entry: dict[str, Any],
    *,
    config: Path,
    target: Path,
    provenance: dict[str, Any],
) -> str:
    strict = target / "strict_tensor_verification.json"
    logits = target / "exact_logits_verification.json"
    conversion = target / "conversion_manifest.json"
    schema = target / "legacy_config_schema_validation.json"
    return f"""# {entry["id"]}

Converted model-only OLMoDDP checkpoint for the Jacob OLMoE3 ladder.

| Field | Value |
| --- | --- |
| Stage | `{entry["stage"]}` |
| Family | `{entry["family"]}` |
| Model size | `{entry["model_size"]}` |
| Data multiple | `Cx{entry["data_multiple"]}` |
| Learning rate | `{entry["learning_rate"]}` |
| Source run | `{entry["source_run_name"]}` |
| Source step | `{entry["source_step"]}` |
| Source checkpoint | `{entry["source_checkpoint"]}` |
| Config resolution | `{provenance["resolution"]}` |
| Config SHA256 | `{file_sha256(config)}` |
| Local converted checkpoint | `{target}` |
| GCS destination | `{entry["gcs_uri"]}` |

## Verification

- Legacy config/checkpoint schema: `LEGACY_CONFIG_SCHEMA_MATCH`
- Converted tensor mapping: `STRICT_TENSOR_MATCH` with raw-byte equality
- Full-vocabulary logits and captured intermediates: exact raw-byte equality at
  sequence length 128, with `atol=0` and `rtol=0`
- Optimizer state included: `false`
- Trainer state included: `false`

Verification file hashes:

- `legacy_config_schema_validation.json`: `{file_sha256(schema)}`
- `conversion_manifest.json`: `{file_sha256(conversion)}`
- `strict_tensor_verification.json`: `{file_sha256(strict)}`
- `exact_logits_verification.json`: `{file_sha256(logits)}`
"""


def model_state(
    entry: dict[str, Any], *, config: Path, status_dir: Path, sequence_length: int
) -> dict[str, Any]:
    source = Path(entry["source_checkpoint"]).resolve()
    target = Path(entry["local_output"]).resolve()
    schema_report = select_schema_report(
        target=target,
        status_dir=status_dir,
        source=source,
        config=config,
    )
    return {
        "model_id": entry["id"],
        "schema": schema_valid(schema_report, source=source, config=config),
        "conversion": conversion_valid(target, source=source, config=config),
        "strict_tensors": strict_valid(
            target / "strict_tensor_verification.json",
            source=source,
            target=target,
            config=config,
        ),
        "exact_logits": logits_valid(
            target / "exact_logits_verification.json",
            source=source,
            target=target,
            source_config=config,
            target_config=target / "config.json",
            sequence_length=sequence_length,
        ),
        "local_publication_marker": (target / "_SUCCESS.json").is_file(),
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    config_root = args.config_root.expanduser().resolve()
    legacy_repo = args.legacy_repo.expanduser().resolve()
    new_repo = args.new_repo.expanduser().resolve()
    status_root = args.status_root.expanduser().resolve() / args.family
    results_dir = args.results_dir.expanduser().resolve()
    manifest = load_json(manifest_path)
    entries = [entry for entry in manifest["models"] if entry["family"] == args.family]
    if not entries:
        raise ValueError(f"No manifest entries found for family {args.family!r}")

    config_paths: dict[str, Path] = {}
    for entry in entries:
        config = config_root / entry["id"] / "config.json"
        source = Path(entry["source_checkpoint"])
        if not config.is_file():
            raise FileNotFoundError(f"Missing materialized config: {config}")
        if not (source / "model_and_optim/.metadata").is_file():
            raise FileNotFoundError(f"Missing source checkpoint: {source}")
        config_paths[entry["id"]] = config

    if args.dry_run:
        print(f"family={args.family} models={len(entries)}", flush=True)
        for entry in entries:
            status_dir = (
                status_root / entry["stage"] / entry["model_size"] / f"cx{entry['data_multiple']}"
            )
            print(
                json.dumps(
                    model_state(
                        entry,
                        config=config_paths[entry["id"]],
                        status_dir=status_dir,
                        sequence_length=args.sequence_length,
                    ),
                    sort_keys=True,
                ),
                flush=True,
            )
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = status_root / "family_summary.json"
    result_summary_path = results_dir / f"{args.family}_family_summary.json"
    summary: dict[str, Any] = {
        "protocol": "olmo_ddp_family_conversion_v1",
        "status": "RUNNING",
        "family": args.family,
        "model_count": len(entries),
        "manifest": str(manifest_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "models": [],
    }
    write_json(summary_path, summary)
    write_json(result_summary_path, summary)

    legacy_env = repo_env(legacy_repo)
    new_env = repo_env(new_repo)
    try:
        for index, entry in enumerate(entries, start=1):
            model_id = entry["id"]
            source = Path(entry["source_checkpoint"]).resolve()
            target = Path(entry["local_output"]).resolve()
            config = config_paths[model_id]
            provenance = load_json(config.parent / "provenance.json")
            status_dir = (
                status_root / entry["stage"] / entry["model_size"] / f"cx{entry['data_multiple']}"
            )
            logs_dir = status_dir / "logs"
            status_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{index}/{len(entries)}] START {model_id}", flush=True)

            target_schema = target / "legacy_config_schema_validation.json"
            status_schema = status_dir / "legacy_config_schema_validation.json"
            schema_report = select_schema_report(
                target=target,
                status_dir=status_dir,
                source=source,
                config=config,
            )
            if schema_valid(schema_report, source=source, config=config):
                print(f"[{index}/{len(entries)}] SKIP schema {model_id}", flush=True)
            else:
                run_command(
                    [
                        sys.executable,
                        str(new_repo / "src/scripts/validate_legacy_moe_checkpoint_config.py"),
                        str(source),
                        "--config",
                        str(config),
                        "--output",
                        str(status_schema),
                    ],
                    log_path=logs_dir / "schema.log",
                    cwd=legacy_repo,
                    env=legacy_env,
                )
                schema_report = status_schema
            if not schema_valid(schema_report, source=source, config=config):
                raise RuntimeError(f"Schema stage did not produce a valid report for {model_id}")

            if conversion_valid(target, source=source, config=config):
                print(f"[{index}/{len(entries)}] SKIP conversion {model_id}", flush=True)
            else:
                if target.exists():
                    raise RuntimeError(
                        f"Existing converted checkpoint is invalid; refusing automatic cleanup: {target}"
                    )
                run_command(
                    [
                        sys.executable,
                        str(new_repo / "src/scripts/convert_legacy_moe_v2_to_olmo_ddp.py"),
                        str(source),
                        str(target),
                        "--config",
                        str(config),
                        "--load-thread-count",
                        str(args.load_thread_count),
                        "--save-thread-count",
                        str(args.save_thread_count),
                    ],
                    log_path=logs_dir / "conversion.log",
                    cwd=new_repo,
                    env=new_env,
                )
            if not conversion_valid(target, source=source, config=config):
                raise RuntimeError(f"Conversion stage did not validate for {model_id}")
            if not target_schema.is_file():
                shutil.copy2(schema_report, target_schema)

            strict_report = target / "strict_tensor_verification.json"
            if strict_valid(strict_report, source=source, target=target, config=config):
                print(f"[{index}/{len(entries)}] SKIP strict tensors {model_id}", flush=True)
            else:
                run_command(
                    [
                        sys.executable,
                        str(new_repo / "src/scripts/verify_legacy_moe_v2_to_olmo_ddp_strict.py"),
                        str(source),
                        str(target),
                        "--config",
                        str(config),
                        "--load-thread-count",
                        str(args.load_thread_count),
                    ],
                    log_path=logs_dir / "strict_tensors.log",
                    cwd=new_repo,
                    env=new_env,
                )
            if not strict_valid(strict_report, source=source, target=target, config=config):
                raise RuntimeError(f"Strict tensor stage did not validate for {model_id}")

            logits_report = target / "exact_logits_verification.json"
            target_config = target / "config.json"
            if logits_valid(
                logits_report,
                source=source,
                target=target,
                source_config=config,
                target_config=target_config,
                sequence_length=args.sequence_length,
            ):
                print(f"[{index}/{len(entries)}] SKIP exact logits {model_id}", flush=True)
            else:
                verification_dir = target / "verification"
                verification_dir.mkdir(parents=True, exist_ok=True)
                legacy_artifact = verification_dir / "legacy_logits.pt"
                candidate_artifact = verification_dir / "olmo_ddp_logits.pt"
                work_root = Path("/tmp/olmo-ddp-family-work") / model_id.replace("/", "_")
                torchrun = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node=1",
                ]
                export_script = new_repo / "src/scripts/export_moe_checkpoint_logits.py"
                common = [
                    "--sequence-length",
                    str(args.sequence_length),
                    "--seed",
                    str(args.seed),
                    "--capture-intermediates",
                    "--enable-grad",
                ]
                run_command(
                    [
                        *torchrun,
                        str(export_script),
                        str(source),
                        str(legacy_artifact),
                        "--model-kind",
                        "legacy",
                        "--config",
                        str(config),
                        "--work-dir",
                        str(work_root / "legacy"),
                        *common,
                    ],
                    log_path=logs_dir / "legacy_logits.log",
                    cwd=legacy_repo,
                    env=legacy_env,
                )
                run_command(
                    [
                        *torchrun,
                        str(export_script),
                        str(target),
                        str(candidate_artifact),
                        "--model-kind",
                        "olmo-ddp",
                        "--config",
                        str(target_config),
                        "--input-artifact",
                        str(legacy_artifact),
                        "--work-dir",
                        str(work_root / "olmo_ddp"),
                        *common,
                    ],
                    log_path=logs_dir / "olmo_ddp_logits.log",
                    cwd=new_repo,
                    env=new_env,
                )
                run_command(
                    [
                        sys.executable,
                        str(new_repo / "src/scripts/compare_moe_checkpoint_logits.py"),
                        str(legacy_artifact),
                        str(candidate_artifact),
                        "--output",
                        str(logits_report),
                        "--atol",
                        "0",
                        "--rtol",
                        "0",
                        "--top-k",
                        "10",
                        "--require-exact",
                    ],
                    log_path=logs_dir / "compare_logits.log",
                    cwd=new_repo,
                    env=new_env,
                )
            if not logits_valid(
                logits_report,
                source=source,
                target=target,
                source_config=config,
                target_config=target_config,
                sequence_length=args.sequence_length,
            ):
                raise RuntimeError(f"Exact logits stage did not validate for {model_id}")

            readme = render_readme(
                entry,
                config=config,
                target=target,
                provenance=provenance,
            )
            write_text(target / "README.md", readme)

            if args.skip_upload:
                print(f"[{index}/{len(entries)}] SKIP upload by request {model_id}", flush=True)
            else:
                run_command(
                    [
                        sys.executable,
                        str(new_repo / "src/scripts/upload_olmo_ddp_checkpoint.py"),
                        str(target),
                        "--manifest",
                        str(manifest_path),
                        "--model-id",
                        model_id,
                    ],
                    log_path=logs_dir / "upload.log",
                    cwd=new_repo,
                    env=new_env,
                )

            state = model_state(
                entry,
                config=config,
                status_dir=status_dir,
                sequence_length=args.sequence_length,
            )
            state["completed_at"] = datetime.now(timezone.utc).isoformat()
            summary["models"] = [
                prior for prior in summary["models"] if prior["model_id"] != model_id
            ]
            summary["models"].append(state)
            write_json(summary_path, summary)
            write_json(result_summary_path, summary)
            print(f"[{index}/{len(entries)}] COMPLETE {model_id}", flush=True)

        summary["status"] = "FAMILY_COMPLETE"
        summary["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(summary_path, summary)
        write_json(result_summary_path, summary)
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        summary["status"] = "FAILED"
        summary["failed_at"] = datetime.now(timezone.utc).isoformat()
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        write_json(summary_path, summary)
        write_json(result_summary_path, summary)
        raise


if __name__ == "__main__":
    main()
