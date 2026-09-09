"""Read-only HF-bucket download into the isolated September 9 conversion scratch root."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath

BUCKET = "allenai/olmo-3p5-small"
SCRATCH = Path("/weka/olmo-3p5-checkpoints/scratch/hero-hf-20260909")
BATCH = 16_777_216
# 637.534208B is exactly token-matched to dense PT step76000.
TARGETS = {100: 6000, 200: 11900, 300: 17900, 400: 23750, 637: 38000}
log = logging.getLogger(__name__)


def write_json(path: Path, data: object) -> None:
    """Atomically publish a small metadata record, never a checkpoint payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def prepare_scratch() -> None:
    """Require the real checkpoint mount and reject symlinked scratch ancestors."""
    mount = Path("/weka/olmo-3p5-checkpoints")
    if not os.path.ismount(mount):
        raise RuntimeError("The real checkpoint Weka volume is not mounted")
    for path in (mount, mount / "scratch", SCRATCH):
        if path.is_symlink():
            raise RuntimeError(f"Refusing symlinked scratch path: {path}")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    owner = SCRATCH / "_OWNER.json"
    identity = {"campaign": "hero-hf-20260909", "bucket": BUCKET, "scratch": str(SCRATCH)}
    if owner.exists():
        if json.loads(owner.read_text()) != identity:
            raise RuntimeError("Scratch directory belongs to another campaign")
    else:
        write_json(owner, identity)


def read_remote_json(api, remote: str) -> dict:
    """Read one metadata object without changing the bucket."""
    with tempfile.TemporaryDirectory(prefix="hero-hf-metadata-") as tmp:
        path = Path(tmp) / "metadata.json"
        api.download_bucket_files(BUCKET, [(remote, path)], raise_on_missing_files=True)
        return json.loads(path.read_text())


def catalog(api) -> dict:
    """Record requested milestones and which ones have verified upload receipts."""
    from huggingface_hub import BucketFile

    rows = []
    for arm in ("emo", "non-emo"):
        prefix = f"{arm}/receipts/"
        receipts = {
            item.path
            for item in api.list_bucket_tree(BUCKET, prefix=prefix, recursive=True)
            if isinstance(item, BucketFile) and item.path.startswith(prefix)
        }
        for target, step in TARGETS.items():
            rows.append(
                {
                    "arm": arm,
                    "target_billions": target,
                    "step": step,
                    "tokens": step * BATCH,
                    "remote_verified": f"{prefix}step{step}.verified.json" in receipts,
                }
            )
    result = {"bucket": BUCKET, "checkpoints": rows, "checked_at_unix": time.time()}
    write_json(SCRATCH / "catalog.json", result)
    print("HERO_HF_CATALOG " + json.dumps(result), flush=True)
    return result


def relative_path(prefix: str, remote: str) -> str:
    """Reject prefix collisions and traversal before creating any local paths."""
    if not remote.startswith(prefix + "/"):
        raise ValueError(f"Remote object is outside exact checkpoint prefix: {remote}")
    suffix = remote[len(prefix) + 1 :]
    path = PurePosixPath(suffix)
    if not suffix or path.is_absolute() or any(p in ("..", ".") for p in suffix.split("/")):
        raise ValueError(f"Unsafe remote object path: {remote}")
    if "\\" in suffix:
        raise ValueError(f"Unsafe remote object path: {remote}")
    return suffix


def download(api, arm: str, step: int) -> Path:
    """Download all files, verify inventories/critical hashes, and atomically publish."""
    from huggingface_hub import BucketFile

    root = SCRATCH / arm / f"step{step}"
    if root.resolve() != root or root.is_symlink():
        raise RuntimeError("Scratch checkpoint path contains symlinks")
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{arm}/checkpoints/step{step}"
    receipt = read_remote_json(api, f"{arm}/receipts/step{step}.verified.json")
    expected_identity = {
        "bucket_id": BUCKET,
        "step": step,
        "remote_checkpoint_prefix": prefix,
        "lineage_id": f"olmo35-small-hero-20260907-{arm}",
        "verification": "exact-path-size-xet-inventory-plus-critical-readback",
    }
    if any(receipt.get(k) != v for k, v in expected_identity.items()):
        raise RuntimeError("Unexpected checkpoint verification receipt identity")
    objects = [
        item
        for item in api.list_bucket_tree(BUCKET, prefix=prefix + "/", recursive=True)
        if isinstance(item, BucketFile) and item.path.startswith(prefix + "/")
    ]
    actual_remote = sorted(
        [{"path": item.path, "size": item.size, "xet_hash": item.xet_hash} for item in objects],
        key=lambda x: x["path"],
    )
    if actual_remote != sorted(receipt["remote_files"], key=lambda x: x["path"]):
        raise RuntimeError("Remote checkpoint no longer matches verified inventory")
    total = sum(item.size for item in objects)
    if total != receipt["source_total_bytes"] or len(objects) != receipt["source_file_count"]:
        raise RuntimeError("Remote checkpoint file count or byte total is inconsistent")
    if total > 200_000_000_000:
        raise RuntimeError("Checkpoint exceeds this small-model campaign's 200GB input limit")
    final = root / "olmo-core"
    staging = root / "olmo-core.partial"
    if final.is_symlink() or staging.is_symlink():
        raise RuntimeError("Refusing symlinked checkpoint storage")
    target = final if final.exists() else staging
    target.mkdir(exist_ok=True)
    expected = {relative_path(prefix, item.path): item.size for item in objects}
    # Existing partial data is retained on failure. Completed downloads are validated again.
    if not final.exists():
        free = shutil.disk_usage(SCRATCH).free
        log.info("HERO_HF_SPACE free_bytes=%d planned_bytes=%d", free, total)
        if free - total < 11_000_000_000_000:
            raise RuntimeError("Download would leave less than 11TB free; no payload downloaded")
        transfers = []
        for item in objects:
            path = target / relative_path(prefix, item.path)
            if path.resolve() != path:
                raise RuntimeError(f"Symlink in downloaded checkpoint: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            transfers.append((item, path))
        log.info(
            "HERO_HF_DOWNLOAD_START arm=%s step=%d files=%d bytes=%d",
            arm,
            step,
            len(objects),
            total,
        )
        api.download_bucket_files(BUCKET, transfers, raise_on_missing_files=True)
    local = {}
    for path in target.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Symlink in downloaded checkpoint: {path}")
        if path.is_file():
            local[str(path.relative_to(target))] = path.stat().st_size
    if local != expected:
        raise RuntimeError("Downloaded file inventory differs from the verified remote inventory")
    for name, digest in receipt["critical_file_sha256"].items():
        safe = relative_path(prefix, prefix + "/" + name)
        with (target / safe).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != digest:
            raise RuntimeError(f"Downloaded critical file checksum mismatch: {safe}")
    # Buckets are mutable: verify remote metadata and the receipt did not change mid-download.
    after = sorted(
        [
            {"path": x.path, "size": x.size, "xet_hash": x.xet_hash}
            for x in api.list_bucket_tree(BUCKET, prefix=prefix + "/", recursive=True)
            if isinstance(x, BucketFile) and x.path.startswith(prefix + "/")
        ],
        key=lambda x: x["path"],
    )
    if (
        after != actual_remote
        or read_remote_json(api, f"{arm}/receipts/step{step}.verified.json") != receipt
    ):
        raise RuntimeError("Remote checkpoint changed during download")
    if not final.exists():
        os.rename(staging, final)
    write_json(root / "source-receipt.json", receipt)
    record = {
        "bucket": BUCKET,
        "prefix": prefix,
        "step": step,
        "tokens": step * BATCH,
        "raw_path": str(final),
        "files": len(objects),
        "bytes": total,
        "critical_hashes_verified": True,
        "completed_at_unix": time.time(),
    }
    write_json(root / "download-success.json", record)
    print("HERO_HF_DOWNLOAD_SUCCESS " + json.dumps(record), flush=True)
    return final


def main() -> None:
    """Catalog uploads, optionally downloading exactly one requested milestone."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("emo", "non-emo"))
    parser.add_argument("--step", type=int, choices=tuple(TARGETS.values()))
    args = parser.parse_args()
    if (args.arm is None) != (args.step is None):
        parser.error("--arm and --step must be provided together")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from huggingface_hub import HfApi
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    prepare_scratch()
    api = HfApi()
    if not api.bucket_info(BUCKET).private:
        raise RuntimeError("Expected the private hero checkpoint bucket")
    catalog(api)
    if args.arm is not None:
        download(api, args.arm, args.step)


if __name__ == "__main__":
    main()
