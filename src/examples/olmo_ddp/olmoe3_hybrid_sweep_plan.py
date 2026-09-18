"""Approved six-point 3:1 sweep with in-process 10% decay; no service dependencies."""

from dataclasses import dataclass

from olmoe3_small_hero_plan import (
    CONTROL,
    DATA_ROOT,
    DOLMA_MOUNT,
    MOUNT,
    STATE,
    UPLOADER,
)

CAMPAIGN = "small-hybrid3to1-lr100b-20260918"
BRANCH = "codex/small-hybrid-3to1-20260918"
WORKSPACE = "ai2/olmo3p5-training"
BUCKET = "allenai/olmo-checkpoint-uploader-pilot-20260902-jm01"
ROOT = MOUNT / "production-lr-sweeps" / CAMPAIGN
AUTOMATION = MOUNT / "uploader/automation" / CAMPAIGN
BATCH = 16_777_216
LRS = (("2p6em3", 2.6e-3), ("1p3em3", 1.3e-3), ("5p2em3", 5.2e-3))
TRAIN_TEMPLATE = "01M2SSCRMSS386QSDJW4HT332H"


@dataclass(frozen=True)
class Run:
    """Independent fresh initialization; center LR is submitted first in each arm."""

    emo: bool
    label: str = "2p6em3"
    smoke: bool = False

    @property
    def arm(self):
        return "emo" if self.emo else "non-emo"

    @property
    def run_id(self):
        return f"olmoe3-{CAMPAIGN}-{'smoke-' if self.smoke else ''}{self.arm}-lr{self.label}"

    @property
    def root(self):
        return ROOT / self.run_id

    @property
    def lr(self):
        return dict(LRS)[self.label]

    @property
    def end(self):
        return 6 if self.smoke else 6000

    @property
    def warmup(self):
        return 2 if self.smoke else 2000

    @property
    def decay(self):
        return 2 if self.smoke else 600

    @property
    def saves(self):
        return [4, 6] if self.smoke else [4200, 4800, 5400, 6000]

    @property
    def prefix(self):
        return f"sweeps/{CAMPAIGN}/{self.run_id}"

    def as_dict(self):
        return dict(
            run_id=self.run_id,
            arm=self.arm,
            emo=self.emo,
            smoke=self.smoke,
            lr=self.lr,
            end=self.end,
            warmup=self.warmup,
            decay=self.decay,
            saves=self.saves,
            batch_tokens=BATCH,
            root=str(self.root),
            bucket=BUCKET,
            remote_prefix=self.prefix,
            gpus=64,
            min_local_checkpoints=4,
            in_process_decay=True,
        )


def runs():
    """EMO first, then non-EMO, with identical LR grids."""
    return [Run(emo, label) for emo in (True, False) for label, _ in LRS]


def smoke_runs():
    """Both router policies share one isolated 64-GPU smoke allocation."""
    return [Run(True, smoke=True), Run(False, smoke=True)]


def find_run(name):
    """Fail closed on any run outside this campaign."""
    return next(r for r in runs() + smoke_runs() if r.run_id == name)
