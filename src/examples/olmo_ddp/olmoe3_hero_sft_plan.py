"""Bounded matched SFT sweep on the two final small hero LC checkpoints."""

import json
from dataclasses import dataclass
from pathlib import Path

from olmoe3_small_hero_plan import (  # noqa: F401 -- public campaign constants
    BUCKET,
    CONTROL,
    MOUNT,
    STATE,
    TEST_BUCKET,
    UPLOADER,
    WORKSPACE,
)

CAMPAIGN = "olmo35-small-gptoss-sft-20260914"
BRANCH = "codex/small-hero-sft-20260914"
ROOT = MOUNT / "production-hero-small-sft" / CAMPAIGN
AUTOMATION = MOUNT / "uploader/automation" / CAMPAIGN
DATA = (
    Path("/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/sft-data")
    / "gptoss120b-deduped-olmo-thinker-20260914"
)
DATA_PLAN = AUTOMATION / "data-plan.json"
CACHE = ROOT / "packing-cache"
BATCH = 524288
SEQUENCE = 65536
GPUS = 8
SEED = 1729
LRS = {"1em5": 1e-5, "5em5": 5e-5, "1em4": 1e-4}
LC_CAMPAIGN = "olmo35-small-2t-lc100b-20260913"
LC_JOBS = {"emo": "01M2CJ0D247DDJF8C5CV66SRZ5", "non-emo": "01M2CKNMBW4WV21HCJQV5FTJAX"}


@dataclass(frozen=True)
class SFTRun:
    """One LR/arm, with an independent optimizer, storage prefix and W&B identity."""

    arm: str
    lr_label: str
    smoke: bool = False

    def __post_init__(self):
        assert self.arm in LC_JOBS and self.lr_label in LRS
        assert not self.smoke or self.lr_label == "5em5"

    @property
    def run_id(self):
        return f"{CAMPAIGN}-{'smoke-' if self.smoke else ''}{self.arm}-lr{self.lr_label}"

    @property
    def emo(self):
        return self.arm == "emo"

    @property
    def lr(self):
        return LRS[self.lr_label]

    @property
    def root(self):
        return ROOT / self.run_id

    @property
    def source(self):
        return (
            MOUNT
            / "production-hero-small-lc"
            / LC_CAMPAIGN
            / f"{LC_CAMPAIGN}-{self.arm}"
            / "step5961"
        )

    @property
    def bucket(self):
        return TEST_BUCKET if self.smoke else BUCKET

    @property
    def prefix(self):
        return f"{CAMPAIGN}/{'smoke/' if self.smoke else ''}{self.arm}/lr{self.lr_label}"

    def as_dict(self):
        return {
            "run_id": self.run_id,
            "arm": self.arm,
            "lr": self.lr,
            "smoke": self.smoke,
            "source": str(self.source),
            "source_experiment": LC_JOBS[self.arm],
            "checkpoint_root": str(self.root),
            "bucket": self.bucket,
            "prefix": self.prefix,
            "epochs": 2,
            "batch_tokens": BATCH,
            "sequence_length": SEQUENCE,
            "gpus": GPUS,
            "data": str(DATA),
            "schedule": "linear",
            "warmup_fraction": 0.03,
            "kernel": "FLA-0.5.2 packed-document path",
            "recompute_each_block": True,
        }


def runs(smoke=False):
    """Return only the approved six trials or the two bounded qualification runs."""
    return [SFTRun(arm, label, smoke) for arm in LC_JOBS for label in (["5em5"] if smoke else LRS)]


def find_run(name):
    """Reject arbitrary source checkpoints and sweep members."""
    return next(r for r in runs() + runs(True) if r.run_id == name)


def data_plan():
    """Read the actual packed epoch length produced by the real-image CPU gate."""
    p = json.loads(DATA_PLAN.read_text())
    assert p["passed"] and p["batch_tokens"] == BATCH and p["sequence_length"] == SEQUENCE
    return p
