#!/usr/bin/env python3
"""Restartably upload and verify a legacy artifact tree with gcloud rsync."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DRY_RUN_CHANGE = re.compile(r"\bwould\s+(?:copy|delete)\b", re.IGNORECASE)


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def inventory(source: Path) -> dict[str, Any]:
    files = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlinks are not supported in artifact trees: {path}")
        if path.is_file():
            stat = path.stat()
            files.append(
                {
                    "path": path.relative_to(source).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {
        "protocol": "olmoe_legacy_artifact_inventory_v1",
        "source": str(source.resolve()),
        "created_at": datetime.now(UTC).isoformat(),
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "files": files,
    }


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("RUN", " ".join(command), flush=True)
    return subprocess.run(command, check=True, text=True, capture_output=capture)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--manifest-destination", required=True)
    parser.add_argument("--status-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.source.is_dir():
        raise FileNotFoundError(f"artifact source is not a directory: {args.source}")
    if not args.destination.startswith("gs://"):
        raise ValueError("destination must be a gs:// URI")
    if not args.manifest_destination.startswith("gs://"):
        raise ValueError("manifest destination must be a gs:// URI")

    status_dir = args.status_dir / args.label
    status_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = status_dir / "source_inventory.json"
    source_inventory = inventory(args.source)
    write_json_atomically(inventory_path, source_inventory)

    # checksums-only makes reruns resume safely: matching objects are skipped,
    # while same-size but different-content objects are replaced.
    base_command = [
        "gcloud",
        "storage",
        "rsync",
        "--recursive",
        "--checksums-only",
        str(args.source),
        args.destination,
    ]
    run(base_command)

    verification = run(base_command[:3] + ["--dry-run"] + base_command[3:], capture=True)
    verification_output = verification.stdout + verification.stderr
    (status_dir / "verification_dry_run.txt").write_text(verification_output)
    if DRY_RUN_CHANGE.search(verification_output):
        raise RuntimeError("post-upload checksum dry run still reports pending copies or deletes")

    completed_at = datetime.now(UTC).isoformat()
    receipt = {
        "protocol": "olmoe_legacy_artifact_upload_v1",
        "status": "COMPLETE",
        "label": args.label,
        "source": str(args.source.resolve()),
        "destination": args.destination.rstrip("/") + "/",
        "file_count": source_inventory["file_count"],
        "total_bytes": source_inventory["total_bytes"],
        "verification": "gcloud storage rsync --checksums-only --dry-run reported no changes",
        "completed_at": completed_at,
    }
    receipt_path = status_dir / "_SUCCESS.json"
    write_json_atomically(receipt_path, receipt)

    manifest_root = args.manifest_destination.rstrip("/")
    run(["gcloud", "storage", "cp", str(inventory_path), f"{manifest_root}/{args.label}_source_inventory.json"])
    run(["gcloud", "storage", "cp", str(receipt_path), f"{manifest_root}/{args.label}_SUCCESS.json"])
    run(["gcloud", "storage", "cp", str(receipt_path), f"{args.destination.rstrip('/')}/_SUCCESS.json"])
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
