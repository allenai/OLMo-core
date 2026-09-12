"""Scoped 100B cosine MT pair from the two approved 2T decay endpoints."""

from dataclasses import dataclass

from olmoe3_hero_decay_plan import DecayRun
from olmoe3_small_hero_plan import BATCH, BUCKET, CONTROL, MOUNT, STATE, UPLOADER, WORKSPACE

CAMPAIGN = "olmo35-small-2t-mt20-20260912"
BRANCH = "codex/small-hero-2t-midtrain-20260912"
ROOT = MOUNT / "production-hero-small-midtrain" / CAMPAIGN
AUTOMATION = MOUNT / "uploader/automation" / CAMPAIGN
EVAL_ROOT = MOUNT / "scratch" / CAMPAIGN
DATA_WORK = ROOT / "data-work"
PT_STEP = 120000
LR = 1.1e-3 * 0.2
WARMUP = 2000
REQUESTED_TOKENS = 100_000_000_000
END = (REQUESTED_TOKENS + BATCH - 1) // BATCH
SMOKE_END = 2
SEED = 103_117_110_100_105 % (2**31 - 1)
MIX_SHA256 = "1ed52c81c3f33fb864157f8e01ed0d0aff24548149c2ca4376182578181c25ba"
DECAY_JOBS = {"emo": "01M2939WZK87EE84CY52NR4BAQ", "non-emo": "01M29945TQSYQND9RRTXW1FA8H"}


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
        return DecayRun(self.arm)

    @property
    def source(self):
        return AUTOMATION / "sources" / self.arm / f"step{PT_STEP}"

    @property
    def bucket(self):
        return BUCKET

    @property
    def prefix(self):
        return f"mt20-after-decay2t/{self.arm}"

    def as_dict(self):
        return dict(
            run_id=self.run_id,
            arm=self.arm,
            checkpoint_root=str(self.root),
            parent=self.parent.run_id,
            parent_experiment=DECAY_JOBS[self.arm],
            parent_step=PT_STEP,
            parent_tokens=PT_STEP * BATCH,
            copied_source=str(self.source),
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
