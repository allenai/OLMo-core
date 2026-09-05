"""Read-only Weka capacity/uploader preflight for the authorized integration pair."""

import json
import shutil
from pathlib import Path


def main():
    """Export only small metadata; never edit registrations/checkpoints or delete files."""
    mount = Path("/weka/olmo-3p5-checkpoints")
    assert mount.is_mount(), f"Missing Weka mount: {mount}"
    root = mount / "uploader"
    reference = "olmoe3-small-cbs-8mi-100b-lr1p3em3-uploader-r1"
    branch = "olmoe3-small-cbs-16mi-from-step4000-lr1p85em3-uploader-r1"
    registration = json.loads((root / "control/registrations" / f"{reference}.json").read_text())
    checkpoint = mount / "production-cbs" / branch / "step7500"
    files = [p for p in checkpoint.rglob("*") if p.is_file()]
    sizes = [p.stat().st_size for p in files]
    usage = shutil.disk_usage(mount)
    manifests = []
    for path in (root / "state/manifests").glob("*"):
        if reference in path.name or branch in path.name:
            manifests.append(
                {"path": str(path), "bytes": path.stat().st_size, "mtime": path.stat().st_mtime}
            )
    report = {
        "free_bytes": usage.free,
        "used_bytes": usage.used,
        "total_bytes": usage.total,
        "reference_registration": {
            key: registration[key]
            for key in (
                "run_id",
                "lineage_id",
                "bucket_id",
                "remote_prefix",
                "deletion_mode",
                "enabled",
            )
        },
        "checkpoint": str(checkpoint),
        "checkpoint_files": len(files),
        "checkpoint_bytes": sum(sizes),
        "integration_pair_25_checkpoints_each_bytes": 50 * sum(sizes),
        "manifests": manifests,
        "shared_data_order_files": [
            str(p) for p in (mount / "production-cbs/work" / reference).glob("global_indices*.npy")
        ],
    }
    output = Path("/results/integration-storage-audit.json")
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    assert files and (checkpoint / ".metadata.json").is_file(), "Source checkpoint missing"
    assert usage.free > 54 * sum(
        sizes
    ), "Insufficient headroom for integration pair plus smoke saves"


if __name__ == "__main__":
    main()
