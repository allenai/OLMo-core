"""Twelve independently initialized, corrected-tokenizer SFT controls."""

import json
from dataclasses import dataclass
from pathlib import Path

import olmoe3_qkgain_plan as base
from olmoe3_dolci_hero_plan import PARENTS

CAMPAIGN = "olmo35-fixedtok-sft-20260921"
BRANCH = "codex/corrected-sft-20260921"
SCRIPT = "src/examples/olmo_ddp/olmoe3_corrected_sft.py"
AUTO = base.MOUNT / "uploader/automation" / CAMPAIGN
ROOT = base.MOUNT / "production-corrected-sft" / CAMPAIGN
EVAL = base.MOUNT / "scratch" / CAMPAIGN
SHARED = Path("/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/sft-data")
INPUT = SHARED / "retokenize-olmo3-20260921/datasets"
DATA = SHARED / "corrected-sft-20260921/data"
TOKENIZER_REV = "6ff857587e040d6d523a3d5f3a56e918f5401d66"
WAVES = (
    ("4t", "gptoss-medium"),
    ("4t", "gptoss-high"),
    ("2t", "gptoss-medium"),
    ("2t", "gptoss-high"),
    ("4t", "dolci-think"),
    ("2t", "dolci-think"),
)
TEMPERATURES = (0.6, 0.8, 1.0)
BUNDLES = ("math500", "ifbench", "humaneval", "alpaca")


@dataclass(frozen=True)
class Run(base.Run):
    """Only data, LR, and parent identity differ from the qualified SFT recipe."""

    milestone: str = "4t"
    dataset: str = "gptoss-medium"
    lineage: str = "emo"

    @property
    def run_id(self):
        return f"{CAMPAIGN}-{self.milestone}-{self.lineage}-{self.dataset}"

    @property
    def root(self):
        return ROOT / self.run_id

    @property
    def prefix(self):
        return f"{CAMPAIGN}/{self.milestone}/{self.lineage}/{self.dataset}"

    @property
    def data(self):
        return DATA / self.dataset

    @property
    def data_plan(self):
        return self.data / "data-plan.json"

    @property
    def source(self):
        kind = "dolci" if self.milestone == "4t" else "dolci2t"
        return PARENTS[f"{kind}-{self.lineage}"]

    @property
    def batch(self):
        return 8388608

    @property
    def gpus(self):
        return 64

    @property
    def nodes(self):
        return 8

    @property
    def epochs(self):
        return 2

    @property
    def end(self):
        return 2 * json.loads(self.data_plan.read_text())["steps_per_epoch"]

    @property
    def lr(self):
        return 5e-5 if self.dataset == "dolci-think" else 2e-4

    @property
    def emo(self):
        return False

    @property
    def hf(self):
        return EVAL / self.run_id / "emo" / f"step{self.end}/hf"

    @property
    def saves(self):
        return [self.end // 2, self.end]

    def as_dict(self):
        return dict(
            super().as_dict(),
            dataset=self.dataset,
            milestone=self.milestone,
            pt_lineage=self.lineage,
            epochs=2,
            save_step0=False,
            tokenizer_revision=TOKENIZER_REV,
            data_plan=str(self.data_plan),
            temperatures=TEMPERATURES,
            eval_bundles=BUNDLES,
        )


def runs(smoke=False):
    """Ordered EMO/non-EMO pairs; each starts from LC, never another SFT run."""
    assert not smoke
    return [
        Run("7to1-split", "sft", False, milestone, dataset, lineage)
        for milestone, dataset in WAVES
        for lineage in ("emo", "non-emo")
    ]


def find_run(name):
    """Resolve only the authorized twelve identities."""
    return next(r for r in runs() if r.run_id == name)


def install():
    """Bind the existing immutable recipe helpers to this independent campaign."""
    base.CAMPAIGN, base.BRANCH, base.ROOT, base.EVAL_ROOT = CAMPAIGN, BRANCH, ROOT, EVAL
    base.AUTOMATION = AUTO
    base.runs, base.find_run = runs, find_run
