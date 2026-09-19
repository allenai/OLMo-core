"""4T LC descendants: GPT-OSS high-effort SFT, two epochs, selected LR, EMO off."""

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

BASELINE_CAMPAIGN = "olmo35-small-gptoss-sft-20260914"
CAMPAIGN = "olmoe3-integration810m-cx8-high-sft-20260919"
BRANCH = "codex/integration-810m-high-sft-20260919"
ROOT = MOUNT / "production-hero-small-sft" / CAMPAIGN
AUTOMATION = MOUNT / "uploader/automation" / CAMPAIGN
DATA = (
    Path("/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/sft-data")
    / "gptoss120b-high-olmo-thinker-20260917"
)
DATA_PLAN = AUTOMATION / "data-plan.json"
CACHE = ROOT / "packing-cache"
BATCH = 524288
SEQUENCE = 65536
GPUS = 8
SEED = 1729
LRS = {"5em5": 5e-5}
LC_CAMPAIGNS = {arm: "olmo35-small-4t-lc100b-noemo-20260916" for arm in ("emo", "non-emo")}
LC_JOBS = {"non-emo": "01M1D1J2ST47VDJESRPDDAP6W2"}
SOURCE = Path("/weka/oe-training-default/ai2-llm/scaling-ladders/olmoe3/jacobm") / (
    "v0.1.0-dev-olmoe3-lc-mt20-dense-rule-fa8e0c182428/810M-Cx8/long-context/step45876"
)
TEMPLATE = "01M2RF2SK6K8KS2YJKT9RD6NY8"


@dataclass(frozen=True)
class SFTRun:
    """One LR/arm, with an independent optimizer, storage prefix and W&B identity."""

    arm: str
    lr_label: str
    smoke: bool = False
    epochs: int = 2

    def __post_init__(self):
        assert self.arm in LC_JOBS and self.lr_label in LRS
        assert not self.smoke or self.lr_label == "5em5"
        assert self.epochs == 2

    @property
    def run_id(self):
        return f"{CAMPAIGN}-{'smoke-' if self.smoke else ''}{self.arm}-lr{self.lr_label}-ep{self.epochs}"

    @property
    def emo(self):
        # Arm denotes PT lineage. MT, LC and SFT all have EMO disabled.
        return False

    @property
    def lr(self):
        return LRS[self.lr_label]

    @property
    def root(self):
        return ROOT / self.run_id

    @property
    def source(self):
        return SOURCE

    @property
    def bucket(self):
        return TEST_BUCKET if self.smoke else BUCKET

    @property
    def prefix(self):
        return f"{CAMPAIGN}/{'smoke/' if self.smoke else ''}{self.arm}/lr{self.lr_label}/ep{self.epochs}"

    @property
    def total_steps(self):
        return data_plan()["steps_per_epoch"] * self.epochs

    @property
    def checkpoint_steps(self):
        # Keep per-epoch recovery points; the uploader protects the latest two.
        return [data_plan()["steps_per_epoch"] * epoch for epoch in range(1, self.epochs + 1)]

    def as_dict(self):
        return {
            "run_id": self.run_id,
            "arm": self.arm,
            "pretrain_emo": self.arm == "emo",
            "midtrain_emo": False,
            "long_context_emo": False,
            "source_emo": False,
            "sft_emo": self.emo,
            "baseline_campaign": BASELINE_CAMPAIGN,
            "lr": self.lr,
            "smoke": self.smoke,
            "source": str(self.source),
            "source_experiment": LC_JOBS[self.arm],
            "checkpoint_root": str(self.root),
            "bucket": self.bucket,
            "prefix": self.prefix,
            "epochs": self.epochs,
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
    """Return one two-epoch run (or source-load/restart smoke) per 4T LC lineage."""
    return [SFTRun(arm, "5em5", smoke, 2) for arm in LC_JOBS]


def find_run(name):
    """Reject arbitrary source checkpoints and sweep members."""
    return next(r for r in runs() + runs(True) if r.run_id == name)


def data_plan():
    """Read the actual packed epoch length produced by the real-image CPU gate."""
    p = json.loads(DATA_PLAN.read_text())
    assert p["passed"] and p["batch_tokens"] == BATCH and p["sequence_length"] == SEQUENCE
    return p
