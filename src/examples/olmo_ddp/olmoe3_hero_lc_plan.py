"""100B ladder LC, from the two hero MT endpoints; independent immutable lineages."""

from dataclasses import dataclass
from pathlib import Path

from olmoe3_hero_mt_plan import MTRun
from olmoe3_small_hero_plan import (
    BATCH,
    BUCKET,
    CONTROL,
    MOUNT,
    STATE,
    UPLOADER,
    WORKSPACE,
)

CAMPAIGN = "olmo35-small-2t-lc100b-20260913"
BRANCH = "codex/small-hero-lc-20260913"
ROOT = MOUNT / "production-hero-small-lc" / CAMPAIGN
AUTOMATION = MOUNT / "uploader/automation" / CAMPAIGN
EVAL_ROOT = MOUNT / "scratch" / CAMPAIGN
DATA_WORK = ROOT / "data-work"
METADATA_CACHE = DATA_WORK / "cached-path-metadata"
TRAINING_MOUNT = Path("/weka/oe-training-default")
PACKING_CACHE = TRAINING_MOUNT / "ai2-llm/checkpoints/akshitab/dataset-cache"
DATA_GLOB = str(
    TRAINING_MOUNT
    / "ai2-llm/preprocessed/tylerr/lc-reshard-final-cleaned/v0.1/allenai/dolma2-tokenizer/*.npy"
)
SOURCE_STEP = 5961
SEQUENCE_LENGTH = 65536
LR = 1.1e-3 * 0.2 * 0.5
WARMUP = 2000
REQUESTED_TOKENS = 100_000_000_000
END = (REQUESTED_TOKENS + BATCH - 1) // BATCH
SMOKE_END = 2
GATE_END = 4
SEED = 119_105_108_108 % (2**31 - 1)
MT_JOBS = {"emo": "01M2BASTQ94J6CKAA0EG4R7J90", "non-emo": "01M2B7GXHRSW8J2T5K0637HGJM"}


@dataclass(frozen=True)
class LCRun:
    """One allowed LC stage, with no reuse of PT/MT uploader namespaces."""

    arm: str
    smoke: bool = False

    def __post_init__(self):
        assert self.arm in MT_JOBS and not self.smoke

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
        return MTRun(self.arm)

    @property
    def source(self):
        return AUTOMATION / "sources" / self.arm / f"step{SOURCE_STEP}"

    @property
    def bucket(self):
        return BUCKET

    @property
    def prefix(self):
        return f"lc100b-after-mt20-decay2t/{self.arm}"

    def as_dict(self):
        return dict(
            run_id=self.run_id,
            arm=self.arm,
            checkpoint_root=str(self.root),
            parent=self.parent.run_id,
            parent_experiment=MT_JOBS[self.arm],
            parent_step=SOURCE_STEP,
            parent_mt_tokens=SOURCE_STEP * BATCH,
            copied_source=str(self.source),
            bucket_id=self.bucket,
            remote_prefix=self.prefix,
            batch_tokens=BATCH,
            gpus=64,
            nodes=8,
            sequence_length=SEQUENCE_LENGTH,
            microbatch_sequences=1,
            gradient_accumulation=4,
            ep=1,
            pp=1,
            cp=1,
            block_recomputation=True,
            lr=LR,
            warmup=WARMUP,
            schedule="linear-to-zero",
            requested_lc_tokens=REQUESTED_TOKENS,
            lc_steps=END,
            lc_tokens=END * BATCH,
            total_seen_tokens=(120000 + SOURCE_STEP + END) * BATCH,
            optimizer_reset_at_lc_start=True,
            data_seed=SEED,
            data_glob=DATA_GLOB,
            packing_cache=str(PACKING_CACHE),
            min_local_checkpoints=2,
        )


def runs():
    return [LCRun(arm) for arm in MT_JOBS]


def find_run(name):
    return next(r for r in runs() if r.run_id == name)
