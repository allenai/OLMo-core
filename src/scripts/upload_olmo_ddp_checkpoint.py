#!/usr/bin/env python3
"""Idempotently upload and CRC32C-verify one converted OLMoDDP checkpoint."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import google_crc32c
from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage


UPLOAD_REPORT = "upload_verification.json"
SUCCESS_MARKER = "_SUCCESS.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_checksums(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    checksum = google_crc32c.Checksum()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            checksum.update(chunk)
    return sha256.hexdigest(), base64.b64encode(checksum.digest()).decode("ascii")


def file_crc32c(path: Path) -> str:
    return file_checksums(path)[1]


def manifest_entry(manifest: dict[str, Any], model_id: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["models"] if entry["id"] == model_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest entry for {model_id!r}, got {len(matches)}")
    return matches[0]


def validate_acceptance(checkpoint: Path, entry: dict[str, Any]) -> None:
    required = [
        "README.md",
        "config.json",
        "source_config.json",
        "conversion_manifest.json",
        "legacy_config_schema_validation.json",
        "strict_tensor_verification.json",
        "exact_logits_verification.json",
        "model_and_optim/.metadata",
    ]
    missing = [relative for relative in required if not (checkpoint / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is not publication-ready; missing={missing}")

    conversion = load_json(checkpoint / "conversion_manifest.json")
    strict = load_json(checkpoint / "strict_tensor_verification.json")
    logits = load_json(checkpoint / "exact_logits_verification.json")
    schema = load_json(checkpoint / "legacy_config_schema_validation.json")
    if conversion.get("optimizer_state_included") is not False:
        raise ValueError("Conversion manifest does not prove optimizer state is absent")
    if conversion.get("trainer_state_included") is not False:
        raise ValueError("Conversion manifest does not prove trainer state is absent")
    if strict.get("status") != "STRICT_TENSOR_MATCH" or strict.get("bitwise_equal") is not True:
        raise ValueError("Strict tensor verification has not passed")
    if strict.get("target_model_only") is not True:
        raise ValueError("Strict tensor report does not prove the target is model-only")
    if logits.get("status") != "LOGITS_MATCH" or logits.get("exact_match") is not True:
        raise ValueError("Exact logits/intermediates verification has not passed")
    if schema.get("status") != "LEGACY_CONFIG_SCHEMA_MATCH":
        raise ValueError("Legacy config/checkpoint schema validation has not passed")
    if str(Path(entry["source_checkpoint"]).resolve()) != conversion.get("source_checkpoint"):
        raise ValueError("Conversion source does not match publication manifest")


def payload_files(checkpoint: Path) -> list[Path]:
    excluded_names = {UPLOAD_REPORT, SUCCESS_MARKER}
    output: list[Path] = []
    for path in checkpoint.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(checkpoint)
        if relative.name in excluded_names or relative.name.endswith(".tmp"):
            continue
        if relative.suffix in {".pt", ".log"}:
            continue
        output.append(path)
    return sorted(output, key=lambda path: path.relative_to(checkpoint).as_posix())


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got {uri!r}")
    bucket, prefix = uri.removeprefix("gs://").split("/", 1)
    return bucket, prefix.rstrip("/") + "/"


def verify_or_upload(blob, path: Path, *, expected_size: int, expected_crc32c: str) -> str:
    try:
        blob.reload()
    except NotFound:
        try:
            blob.upload_from_filename(
                path,
                if_generation_match=0,
                checksum="crc32c",
                timeout=3600,
            )
        except PreconditionFailed:
            pass
        blob.reload()
        action = "uploaded"
    else:
        action = "already_present"
    if blob.size != expected_size or blob.crc32c != expected_crc32c:
        raise ValueError(
            f"GCS object mismatch for gs://{blob.bucket.name}/{blob.name}: "
            f"expected size={expected_size} crc32c={expected_crc32c}, "
            f"got size={blob.size} crc32c={blob.crc32c}"
        )
    return action


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_json(manifest_path)
    entry = manifest_entry(manifest, args.model_id)
    validate_acceptance(checkpoint, entry)

    bucket_name, prefix = parse_gcs_uri(entry["gcs_uri"])
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    files = payload_files(checkpoint)
    objects: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(checkpoint).as_posix()
        sha256, crc32c = file_checksums(path)
        objects.append(
            {
                "name": relative,
                "size": path.stat().st_size,
                "crc32c": crc32c,
                "sha256": sha256,
            }
        )

    expected_remote = {prefix + item["name"] for item in objects}
    expected_remote.update({prefix + UPLOAD_REPORT, prefix + SUCCESS_MARKER})
    existing_remote = {blob.name for blob in client.list_blobs(bucket, prefix=prefix)}
    unexpected_remote = sorted(existing_remote - expected_remote)
    if unexpected_remote:
        raise ValueError(
            "GCS destination contains objects outside this local checkpoint; "
            f"refusing automatic cleanup or overwrite: {unexpected_remote[:20]}"
        )

    uploaded = 0
    reused = 0
    for path, item in zip(files, objects, strict=True):
        blob = bucket.blob(prefix + item["name"])
        action = verify_or_upload(
            blob,
            path,
            expected_size=item["size"],
            expected_crc32c=item["crc32c"],
        )
        uploaded += action == "uploaded"
        reused += action == "already_present"
        print(f"{action}: gs://{bucket_name}/{blob.name}", flush=True)

    report = {
        "protocol": "olmo_ddp_gcs_crc32c_v1",
        "status": "UPLOAD_VERIFIED",
        "model_id": entry["id"],
        "source_checkpoint": entry["source_checkpoint"],
        "local_checkpoint": str(checkpoint),
        "gcs_uri": entry["gcs_uri"],
        "publication_manifest_sha256": file_sha256(manifest_path),
        "conversion_manifest_sha256": file_sha256(checkpoint / "conversion_manifest.json"),
        "strict_tensor_report_sha256": file_sha256(checkpoint / "strict_tensor_verification.json"),
        "exact_logits_report_sha256": file_sha256(checkpoint / "exact_logits_verification.json"),
        "payload_object_count": len(objects),
        "payload_bytes": sum(item["size"] for item in objects),
        "objects": objects,
    }
    report_path = checkpoint / UPLOAD_REPORT
    write_json(report_path, report)
    report_blob = bucket.blob(prefix + UPLOAD_REPORT)
    report_action = verify_or_upload(
        report_blob,
        report_path,
        expected_size=report_path.stat().st_size,
        expected_crc32c=file_crc32c(report_path),
    )

    success = {
        "protocol": report["protocol"],
        "status": "PUBLICATION_COMPLETE",
        "model_id": entry["id"],
        "gcs_uri": entry["gcs_uri"],
        "upload_verification_sha256": file_sha256(report_path),
        "payload_object_count": len(objects),
        "payload_bytes": report["payload_bytes"],
    }
    success_path = checkpoint / SUCCESS_MARKER
    write_json(success_path, success)
    success_blob = bucket.blob(prefix + SUCCESS_MARKER)
    success_action = verify_or_upload(
        success_blob,
        success_path,
        expected_size=success_path.stat().st_size,
        expected_crc32c=file_crc32c(success_path),
    )

    final_remote = {blob.name for blob in client.list_blobs(bucket, prefix=prefix)}
    if final_remote != expected_remote:
        raise ValueError(
            "Final GCS object set mismatch: "
            f"missing={sorted(expected_remote - final_remote)[:20]}, "
            f"unexpected={sorted(final_remote - expected_remote)[:20]}"
        )
    print(
        json.dumps(
            {
                "status": "PUBLICATION_COMPLETE",
                "model_id": entry["id"],
                "gcs_uri": entry["gcs_uri"],
                "uploaded_payload_objects": uploaded,
                "reused_payload_objects": reused,
                "upload_report": report_action,
                "success_marker": success_action,
                "verified_remote_object_count": len(final_remote),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
