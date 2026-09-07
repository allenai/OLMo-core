"""Read-only checkpoint-capacity audit before the medium CBS smoke and training."""

import json
import os
import time
from collections import Counter
from pathlib import Path

MOUNT = Path("/weka/olmo-3p5-checkpoints")


def main():
    """Read existing metadata only; never register, upload, prune or edit a lineage."""
    if not os.path.ismount(MOUNT):
        raise RuntimeError("The real checkpoint Weka mount is required")
    fs = os.statvfs(MOUNT)
    state = MOUNT / "uploader/state"
    regs = [
        json.loads(p.read_text()) for p in (MOUNT / "uploader/control/registrations").glob("*.json")
    ]
    records = [json.loads(p.read_text()) for p in (state / "checkpoints").glob("*/step-*.json")]
    local = [r for r in records if not r["local_deleted"] and Path(r["checkpoint_path"]).is_dir()]
    pending = [r for r in local if not r["remote_verified"]]
    logical = lambda rows: sum(
        (r.get("source_inventory") or {}).get("total_bytes", 0) for r in rows
    )
    report = {
        "timestamp": time.time(),
        "mount": str(MOUNT),
        "capacity_bytes_statvfs": fs.f_blocks * fs.f_frsize,
        "free_bytes_statvfs": fs.f_bavail * fs.f_frsize,
        "free_inodes_statvfs": fs.f_favail,
        "enabled_registrations": sum(r.get("enabled", True) for r in regs),
        "registered_buckets": sorted({r["bucket_id"] for r in regs if r.get("enabled", True)}),
        "deletion_modes": dict(Counter(r["deletion_mode"] for r in regs if r.get("enabled", True))),
        "recorded_local_checkpoints": len(local),
        "recorded_local_logical_bytes": logical(local),
        "pending_upload_count": len(pending),
        "pending_upload_logical_bytes": logical(pending),
        "pending_statuses": dict(Counter(r["status"] for r in pending)),
        "latest_audit_mtime": (state / "audit.jsonl").stat().st_mtime,
        "medium_checkpoint_estimate_bytes": 42_759_806_592 * 12,
        "caveat": "Metadata snapshot; statvfs quota semantics depend on Weka. Logical counts exclude undiscovered/partial checkpoints and other files. Uploader liveness is checked separately in Beaker logs.",
        "pending_by_lineage": {
            lineage: {
                "count": len(rows := [r for r in pending if r["lineage_id"] == lineage]),
                "bytes": logical(rows),
            }
            for lineage in sorted({r["lineage_id"] for r in pending})
        },
    }
    output = Path(os.environ.get("RESULTS_DIR", "/results"))
    output.mkdir(exist_ok=True, parents=True)
    (output / "medium-storage-audit.json").write_text(json.dumps(report, indent=2))
    print(
        "MEDIUM_STORAGE_AUDIT",
        json.dumps({k: v for k, v in report.items() if k != "pending_by_lineage"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
