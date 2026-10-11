"""Scoped 100B cosine MT from the 12T + 400B EMO fixed PT decay endpoint, EMO kept on."""

import json
from dataclasses import dataclass
from pathlib import Path

from olmoe3_lr_sweep_plan import checkpoint_complete
from olmoe3_small_hero_plan import BATCH, BUCKET, MOUNT

CAMPAIGN = "olmo35-small-12t400b-mt20-20261011"
BRANCH = "tanushy/hero-12t400b-mt20-20261011"
ROOT = MOUNT / "production-hero-small-midtrain" / CAMPAIGN
AUTOMATION = MOUNT / "uploader/automation" / CAMPAIGN
DATA_WORK = ROOT / "data-work"
METADATA_CACHE = DATA_WORK / "cached-path-metadata"
PARENT = "olmo35-small-12t-plus400bdecay-20261006-emo"
PT_STEP = 739342
SOURCE = MOUNT / "production-fixed-pt-decays" / PARENT / f"step{PT_STEP}"
LR = 1.1e-3 * 0.2
WARMUP = 2000
REQUESTED_TOKENS = 100_000_000_000
END = (REQUESTED_TOKENS + BATCH - 1) // BATCH
SMOKE_END = 2
SEED = 103_117_110_100_105 % (2**31 - 1)
MIX_SHA256 = "1ed52c81c3f33fb864157f8e01ed0d0aff24548149c2ca4376182578181c25ba"
DECAY_JOBS = {"emo": "01M4AKS3VR0DH01X2JB9A2903X"}


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
        return f"mt20-after-12t400b/{self.arm}"

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
