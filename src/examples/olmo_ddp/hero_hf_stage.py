"""Stage immutable local hero checkpoint copies, with verified HF download fallback."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path

from hero_hf_download import BATCH, SCRATCH, TARGETS, download, prepare_scratch, write_json
from olmoe3_small_hero_plan import CONTROL, Run

log = logging.getLogger(__name__)


def inventory(root: Path) -> dict:
    """Reject links and record identity/size/mtime so concurrent deletion fails closed."""
    if root.is_symlink() or root.resolve() != root:
        raise RuntimeError(f"Unsafe source root: {root}")
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Symlink in checkpoint: {path}")
        if path.is_file():
            st = path.stat()
            result[str(path.relative_to(root))] = (st.st_size, st.st_mtime_ns, st.st_ino)
    for required in (".metadata.json", "model_and_optim/.metadata", "train/rank0.pt"):
        if required not in result:
            raise FileNotFoundError(f"Not a finalized checkpoint: {root / required}")
    return result


def ready_event(arm: str, step: int) -> tuple[Path, dict]:
    """Require a matching trainer-published completion event before local copying."""
    run = Run(arm == "emo")
    source = run.root / f"step{step}"
    event_path = CONTROL / "inbox" / run.run_id / f"step-{step:012d}.ready.json"
    event = json.loads(event_path.read_text())
    expected = {
        "event": "checkpoint_ready",
        "step": step,
        "run_id": run.run_id,
        "lineage_id": run.run_id,
        "checkpoint_path": str(source),
    }
    if any(event.get(k) != value for k, value in expected.items()):
        raise RuntimeError(f"Mismatched ready event: {event_path}")
    digest = hashlib.sha256((source / ".metadata.json").read_bytes()).hexdigest()
    if digest != event["checkpoint_metadata_sha256"]:
        raise RuntimeError(f"Source metadata no longer matches completion event: {source}")
    return source, event


def copy_local(arm: str, step: int) -> Path:
    """Copy bytes, never link or mutate originals; verify both hashes and source stability."""
    source, event = ready_event(arm, step)
    before = inventory(source)
    total = sum(row[0] for row in before.values())
    if not 0 < total <= 200_000_000_000:
        raise RuntimeError(f"Unexpected small checkpoint size: {total}")
    root = SCRATCH / arm / f"step{step}"
    if root.resolve() != root:
        raise RuntimeError("Symlinked scratch destination")
    root.mkdir(parents=True, exist_ok=True)
    final, staging = root / "olmo-core", root / "olmo-core.local-partial"
    if final.exists() or final.is_symlink() or staging.is_symlink():
        raise RuntimeError("Refusing to overwrite an existing raw checkpoint")
    if shutil.disk_usage(SCRATCH).free - total < 11_000_000_000_000:
        raise RuntimeError("Local copy would leave less than 11TB free")
    staging.mkdir(exist_ok=True)
    log.info(
        "HERO_LOCAL_COPY_START arm=%s step=%d files=%d bytes=%d source=%s",
        arm,
        step,
        len(before),
        total,
        source,
    )

    def one(name: str) -> tuple[str, str]:
        src, dst = source / name, staging / name
        if dst.resolve() != dst:
            raise RuntimeError(f"Symlink in destination: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with src.open("rb") as inp, dst.open("wb") as out:
            while chunk := inp.read(8 * 1024 * 1024):
                out.write(chunk)
                digest.update(chunk)
            out.flush()
            os.fsync(out.fileno())
        with dst.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != digest.hexdigest() or dst.stat().st_size != before[name][0]:
            raise RuntimeError(f"Local copy checksum/size mismatch: {name}")
        return name, actual

    hashes = {}
    copied = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one, name) for name in before]
        for future in concurrent.futures.as_completed(futures):
            name, digest = future.result()
            hashes[name] = digest
            copied += before[name][0]
            if len(hashes) % 100 == 0:
                log.info(
                    "HERO_LOCAL_COPY_PROGRESS files=%d/%d bytes=%d/%d",
                    len(hashes),
                    len(before),
                    copied,
                    total,
                )
    if inventory(source) != before or ready_event(arm, step)[1] != event:
        raise RuntimeError("Source changed or was deleted during copy; raw staging retained")
    after = inventory(staging)
    if {k: v[0] for k, v in after.items()} != {k: v[0] for k, v in before.items()}:
        raise RuntimeError("Local copy inventory mismatch")
    os.rename(staging, final)
    receipt = {
        "source_kind": "local_copy",
        "source": str(source),
        "ready_event": event,
        "source_sha256": hashes,
        "source_file_count": len(hashes),
        "source_total_bytes": total,
    }
    write_json(root / "source-receipt.json", receipt)
    record = {
        "source_kind": "local_copy",
        "source": str(source),
        "arm": arm,
        "step": step,
        "tokens": step * BATCH,
        "raw_path": str(final),
        "files": len(hashes),
        "bytes": total,
        "all_file_hashes_verified": True,
        "completed_at_unix": time.time(),
    }
    write_json(root / "download-success.json", record)
    print("HERO_LOCAL_COPY_SUCCESS " + json.dumps(record), flush=True)
    return final


def local_catalog() -> list:
    """Read only requested source directories, never scan payloads across the volume."""
    rows = []
    for arm in ("emo", "non-emo"):
        for target, step in TARGETS.items():
            source = Run(arm == "emo").root / f"step{step}"
            row = {
                "arm": arm,
                "target_billions": target,
                "step": step,
                "tokens": step * BATCH,
                "source": str(source),
                "local_available": False,
            }
            try:
                ready_event(arm, step)
                inv = inventory(source)
                row.update(
                    local_available=True, files=len(inv), bytes=sum(x[0] for x in inv.values())
                )
            except FileNotFoundError:
                pass
            rows.append(row)
    print("HERO_LOCAL_CATALOG " + json.dumps(rows), flush=True)
    write_json(SCRATCH / "local-catalog.json", {"time": time.time(), "checkpoints": rows})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("emo", "non-emo"))
    parser.add_argument("--step", type=int, choices=tuple(TARGETS.values()))
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    if (args.arm is None) != (args.step is None):
        parser.error("--arm and --step are required together")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare_scratch()
    local_catalog()
    if args.arm is None:
        return
    with (SCRATCH / "stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        root = SCRATCH / args.arm / f"step{args.step}"
        if (root / "download-success.json").is_file():
            log.info("Already staged: %s", root)
            return
        if (Run(args.arm == "emo").root / f"step{args.step}" / ".metadata.json").is_file():
            copy_local(args.arm, args.step)
        elif args.local_only:
            raise FileNotFoundError("Requested checkpoint is not locally available")
        else:
            from huggingface_hub import HfApi

            download(HfApi(), args.arm, args.step)


if __name__ == "__main__":
    main()
