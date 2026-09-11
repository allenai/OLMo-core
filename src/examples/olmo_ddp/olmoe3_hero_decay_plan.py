"""Explicit, bounded ~2T hero decay identities and immutable checkpoint copying."""

import concurrent.futures
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from olmoe3_lr_sweep_plan import checkpoint_complete
from olmoe3_lr_sweep_watch import atomic_json, log
from olmoe3_small_hero_plan import BATCH, BUCKET, CONTROL, MOUNT, Run

CAMPAIGN = "olmo35-small-decay2t-20260911"
BRANCH = "codex/small-hero-2t-decays-20260911"
BASE_COMMIT = "89bf37a87d955b8ff8a76ac11df6dd3bec976d30"
START, END, LR = 108_000, 120_000, 1.1e-3
SMOKE_END = START + 2
AUTOMATION = MOUNT / "uploader/automation" / CAMPAIGN
ROOT = MOUNT / "production-hero-small-decays" / CAMPAIGN
EVAL_ROOT = MOUNT / "scratch" / CAMPAIGN
PARENTS = {"emo": "01M20YQDPJD82K0Y05RH67NNYK", "non-emo": "01M20YQHDJNMWG7B6V6382RDBN"}
UPLOADER_REF = "50069318bd7b6bcfed655a8a01d2892e56b7abff"
CORE_REF = "b1fd2c9746e88baeb20e372bdca340d788d0f7e5"
HELPER_REF = "3542d99ebeb6d156887ed1eaf2871cff5f07666d"


@dataclass(frozen=True)
class DecayRun:
    """A child lineage, never a continuation of the parent namespace."""

    arm: str
    smoke: bool = False

    def __post_init__(self):
        if self.arm not in PARENTS or self.smoke:
            raise ValueError("Only the two approved hero decay trajectories are supported")

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
    def bucket(self):
        return BUCKET

    @property
    def prefix(self):
        return f"decay10-2t/{self.arm}"

    @property
    def parent(self):
        return Run(self.emo)

    @property
    def source(self):
        return AUTOMATION / "sources" / self.arm / f"step{START}"

    def as_dict(self):
        return dict(
            run_id=self.run_id,
            arm=self.arm,
            emo=self.emo,
            root=str(self.root),
            bucket=self.bucket,
            prefix=self.prefix,
            parent=self.parent.run_id,
            parent_experiment=PARENTS[self.arm],
            fork_step=START,
            stop_step=END,
            fork_tokens=START * BATCH,
            final_tokens=END * BATCH,
            lr=LR,
            decay_steps=END - START,
            batch_tokens=BATCH,
            gpus=64,
            min_local_checkpoints=2,
            copied_source=str(self.source),
        )


def runs():
    return [DecayRun(arm) for arm in PARENTS]


def find_run(name):
    return next(r for r in runs() if r.run_id == name)


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
