"""Scoped 100B cosine MT from the 14T + 200B EMO fixed PT decay endpoint, EMO kept on."""

import concurrent.futures
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from olmoe3_lr_sweep_plan import checkpoint_complete
from olmoe3_lr_sweep_watch import atomic_json, log
from olmoe3_small_hero_plan import BATCH, BUCKET, CONTROL, MOUNT

CAMPAIGN = "olmo35-small-14t200b-mt20-20261012"
BRANCH = "tanushy/hero-14t200b-mt20-20261012"
ROOT = MOUNT / "production-hero-small-midtrain" / CAMPAIGN
AUTOMATION = MOUNT / "uploader/automation" / CAMPAIGN
EVAL_ROOT = MOUNT / "scratch" / CAMPAIGN
CORE_REF = "b1fd2c9746e88baeb20e372bdca340d788d0f7e5"
HELPER_REF = "3542d99ebeb6d156887ed1eaf2871cff5f07666d"
DATA_WORK = ROOT / "data-work"
METADATA_CACHE = DATA_WORK / "cached-path-metadata"
PARENT = "olmo35-small-14t-plus200bdecay-20261008-emo"
PT_STEP = 846387
SOURCE = MOUNT / "production-fixed-pt-decays" / PARENT / f"step{PT_STEP}"
LR = 1.1e-3 * 0.2
WARMUP = 2000
REQUESTED_TOKENS = 100_000_000_000
END = (REQUESTED_TOKENS + BATCH - 1) // BATCH
SMOKE_END = 2
SEED = 103_117_110_100_105 % (2**31 - 1)
MIX_SHA256 = "1ed52c81c3f33fb864157f8e01ed0d0aff24548149c2ca4376182578181c25ba"
DECAY_JOBS = {"emo": "01M4D782XZNKD0BFK1EKNB40W5"}


def inventory(root):
    """Reject symlinks and incomplete checkpoints, recording change-detection metadata."""
    root = Path(root)
    if root.resolve() != root or not checkpoint_complete(root):
        raise ValueError(f"Not a complete, real checkpoint: {root}")
    result = {}
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValueError(f"Symlink in checkpoint: {p}")
        if p.is_file():
            s = p.stat()
            result[str(p.relative_to(root))] = (s.st_size, s.st_mtime_ns, s.st_ino)
    return result


def validate_checkpoint(root, step):
    """Require full-state metadata and immutable per-rank save audits for all 64 ranks."""
    inventory(root)
    for rank in range(64):
        row = json.loads((root / "resume_audit" / f"rank{rank}.json").read_text())
        if (row["step"], row["tokens"], row["rank"], row["gpus"]) != (step, step * BATCH, rank, 64):
            raise ValueError(f"Wrong checkpoint audit: {root}, rank {rank}")
        if not (root / "train" / f"rank{rank}.pt").is_file():
            raise ValueError(f"Missing trainer state: rank{rank}")


def ready(run, step):
    """Validate a trainer completion event, not just the presence of a step directory."""
    source = run.root / f"step{step}"
    path = CONTROL / "inbox" / run.run_id / f"step-{step:012d}.ready.json"
    if not path.is_file() or not checkpoint_complete(source):
        return False
    row = json.loads(path.read_text())
    expected = dict(
        event="checkpoint_ready",
        step=step,
        run_id=run.run_id,
        lineage_id=run.run_id,
        checkpoint_path=str(source),
    )
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Mismatched completion event: {path}")
    if (
        row["checkpoint_metadata_sha256"]
        != hashlib.sha256((source / ".metadata.json").read_bytes()).hexdigest()
    ):
        raise ValueError(f"Completion metadata hash mismatch: {source}")
    validate_checkpoint(source, step)
    return True


def verified_copy(source, destination, step):
    """Copy every byte, fsync, read-back hash, then publish; never link/delete originals."""
    source, destination = Path(source), Path(destination)
    receipt = destination.with_name(destination.name + "-copy.json")
    if receipt.is_file():
        saved = json.loads(receipt.read_text())
        assert saved["source"] == str(source) and saved["destination"] == str(destination)
        assert saved["step"] == step
        validate_checkpoint(destination, step)
        assert {k: v[0] for k, v in inventory(destination).items()} == saved["sizes"]
        return saved
    validate_checkpoint(source, step)
    before = inventory(source)
    total = sum(v[0] for v in before.values())
    if not 0 < total < 200_000_000_000:
        raise ValueError(f"Unexpected small-model checkpoint size: {total}")
    if not MOUNT.is_mount() or shutil.disk_usage(MOUNT).free - total < 11_000_000_000_000:
        raise RuntimeError("Require real Weka and at least 11TB after staging")
    if destination.resolve() != destination:
        raise ValueError("Symlinked destination")
    # A publish-before-receipt crash can safely be reconciled by hashing the existing copy.
    staging = (
        destination
        if destination.exists()
        else destination.with_name(destination.name + ".partial")
    )
    if staging.resolve() != staging:
        raise ValueError("Symlinked staging path")
    staging.mkdir(parents=True, exist_ok=True)
    log("CHECKPOINT_COPY_STARTED", source=str(source), destination=str(destination), bytes=total)

    def one(name):
        src, dst = source / name, staging / name
        if dst.resolve() != dst:
            raise ValueError(f"Unsafe destination: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as inp:
            digest = hashlib.file_digest(inp, "sha256").hexdigest()
        if not destination.exists():
            with src.open("rb") as inp, dst.open("wb") as out:
                shutil.copyfileobj(inp, out, 8 * 1024 * 1024)
                out.flush()
                os.fsync(out.fileno())
        with dst.open("rb") as inp:
            assert hashlib.file_digest(inp, "sha256").hexdigest() == digest, name
        assert dst.stat().st_size == before[name][0]
        return name, digest

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(one, before))
    if inventory(source) != before:
        raise RuntimeError("Source changed during copy; retained staging and failed closed")
    assert {k: v[0] for k, v in inventory(staging).items()} == {k: v[0] for k, v in before.items()}
    if staging != destination:
        os.rename(staging, destination)
    record = dict(
        source=str(source),
        destination=str(destination),
        step=step,
        sizes={k: v[0] for k, v in before.items()},
        sha256=hashes,
        total_bytes=total,
        all_file_hashes_verified=True,
    )
    atomic_json(receipt, record)
    log("CHECKPOINT_COPY_VERIFIED", source=str(source), destination=str(destination), bytes=total)
    return record


@dataclass(frozen=True)
class MTRun:
    """Independent MT counters, uploader namespace, optimizer and data stream."""

    arm: str
    smoke: bool = False

    def __post_init__(self):
        assert self.arm in DECAY_JOBS and not self.smoke

    @property
    def emo(self):
        return self.arm == "emo"

    @property
    def run_id(self):
        return f"{CAMPAIGN}-{self.arm}"

    @property
    def root(self):
        return ROOT / self.run_id

    @property
    def parent(self):
        return PARENT

    @property
    def source(self):
        return SOURCE

    @property
    def bucket(self):
        return BUCKET

    @property
    def prefix(self):
        return f"mt20-after-14t200b/{self.arm}"

    def registration(self):
        """Fields for the uploader Registration of this isolated child lineage."""
        return dict(
            run_id=self.run_id,
            lineage_id=self.run_id,
            checkpoint_root=str(self.root),
            bucket_id=self.bucket,
            remote_prefix=self.prefix,
            deletion_mode="apply",
            min_local_checkpoints=2,
            delete_grace_seconds=3600,
        )

    def as_dict(self):
        return dict(
            run_id=self.run_id,
            arm=self.arm,
            checkpoint_root=str(self.root),
            parent=self.parent,
            parent_experiment=DECAY_JOBS[self.arm],
            parent_step=PT_STEP,
            parent_tokens=PT_STEP * BATCH,
            source=str(self.source),
            bucket_id=self.bucket,
            remote_prefix=self.prefix,
            batch_tokens=BATCH,
            gpus=64,
            nodes=8,
            lr=LR,
            warmup=WARMUP,
            schedule="cosine-to-zero",
            requested_mt_tokens=REQUESTED_TOKENS,
            mt_steps=END,
            mt_tokens=END * BATCH,
            total_seen_tokens=(PT_STEP + END) * BATCH,
            optimizer_reset_at_mt_start=True,
            data_seed=SEED,
            mixture_sha256=MIX_SHA256,
        )


def runs():
    return [MTRun(arm) for arm in DECAY_JOBS]


def find_run(name):
    return next(r for r in runs() if r.run_id == name)
