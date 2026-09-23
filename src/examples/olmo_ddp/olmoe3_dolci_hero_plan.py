"""Approved paired Dolci SFT and 128-GPU non-EMO, 3:1/shared-QK hero campaign."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import olmoe3_qkgain_plan as base

CAMPAIGN = "olmo35-dolci-hero-20260920"
BRANCH = "codex/hero2t-128gpu-dolci-20260922"
SCRIPT = "src/examples/olmo_ddp/olmoe3_dolci_hero.py"
AUTO = base.MOUNT / "uploader/automation" / CAMPAIGN
ROOT = base.MOUNT / "production-dolci-hero" / CAMPAIGN
EVAL = base.MOUNT / "scratch" / CAMPAIGN
DATA = Path("/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/sft-data/dolci-think-20260920")
INPUT = Path(
    "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache/numpy_sft/15bfc110a1-6068a350"
)
FINAL = 120_000
DECAY_START = 108_000
PIN = AUTO / "sources/hero/step6000"
CONTINUATION_PIN = AUTO / f"sources/hero/step{DECAY_START}"
KINDS = ("dolci-emo", "dolci-non-emo", "hero", "decay", "mt", "lc", "sft")
NEW_KINDS = (
    "hero2t-mt",
    "hero2t-lc",
    "hero2t-sft-high",
    "hero2t-sft-dolci",
    "dolci2t-emo",
    "dolci2t-non-emo",
)
KINDS += NEW_KINDS
PARENT_KINDS = {
    "mt": "decay",
    "lc": "mt",
    "sft": "lc",
    "hero2t-mt": "hero",
    "hero2t-lc": "hero2t-mt",
    "hero2t-sft-high": "hero2t-lc",
    "hero2t-sft-dolci": "hero2t-lc",
}
OLD_2T_SOURCES = {
    "dolci2t-emo": (
        "posttrain-noemo-20260914/lc100b/emo",
        "olmo35-small-2t-lc100b-noemo-20260914-emo",
    ),
    "dolci2t-non-emo": (
        "lc100b-after-mt20-decay2t/non-emo",
        "olmo35-small-2t-lc100b-20260913-non-emo",
    ),
}
PARENTS = {
    "dolci-emo": base.MOUNT
    / "uploader/automation/olmo35-4t-emo-lc-sft8mi-20260919/sources/posttrain-noemo-4t-20260916/lc100b/emo/step5961/olmo-core",
    "dolci-non-emo": base.MOUNT
    / "uploader/automation/olmo35-4t-lc-sft8mi-5epoch-20260920/sources/posttrain-noemo-4t-20260916/lc100b/non-emo/step5961/olmo-core",
}
PARENTS.update(
    {
        kind: AUTO / "sources/old-2t" / prefix / "step5961/olmo-core"
        for kind, (prefix, _) in OLD_2T_SOURCES.items()
    }
)


@dataclass(frozen=True)
class Run(base.Run):
    """One authorized job; warmup, schedule and data are resolved in the adapter."""

    kind: str = "hero"

    @property
    def run_id(self):
        return CAMPAIGN + "-" + self.kind

    @property
    def root(self):
        return ROOT / self.run_id

    @property
    def prefix(self):
        return CAMPAIGN + "/" + self.kind

    @property
    def emo(self):
        return False

    @property
    def split(self):
        return self.kind.startswith("dolci")

    @property
    def epochs(self):
        return 2

    @property
    def uses_dolci(self):
        return "dolci" in self.kind

    @property
    def batch(self):
        if self.uses_dolci:
            return 8388608
        if self.kind == "hero2t-lc":
            return 16777216
        return super().batch

    @property
    def gpus(self):
        if self.kind in ("hero", "hero2t-mt", "hero2t-lc"):
            return 128
        return 64 if self.uses_dolci else super().gpus

    @property
    def nodes(self):
        return self.gpus // 8

    @property
    def microbatch(self):
        return (
            int(os.environ.get("CAMPAIGN_PT_MB", "4")) * self.sequence
            if self.kind in ("hero", "decay")
            else super().microbatch
        )

    @property
    def end(self):
        if self.uses_dolci:
            return 2 * json.loads((AUTO / "dolci-data-plan.json").read_text())["steps_per_epoch"]
        if self.kind in ("hero2t-mt", "hero2t-lc"):
            return 5961
        return FINAL if self.kind == "hero" else super().end

    @property
    def lr(self):
        return 1.1e-4 if self.kind == "hero2t-lc" else super().lr

    @property
    def start(self):
        return 6000 if self.kind == "decay" else 0

    @property
    def source(self):
        if self.kind in PARENTS:
            return PARENTS[self.kind]
        if self.kind == "hero":
            return None
        if self.kind == "decay":
            return PIN
        parent = run(PARENT_KINDS[self.kind])
        return parent.root / f"step{parent.end}"

    @property
    def hf(self):
        return EVAL / self.run_id / "emo" / f"step{self.end}/hf"

    @property
    def saves(self):
        if self.uses_dolci:
            return [2, 4, self.end // 2, self.end]
        if self.kind == "hero":
            # Whole-step ceilings at the requested token thresholds; pin exactly6000.
            a = 200_000_000_000 // self.batch
            b = 500_000_000_000 // self.batch
            return sorted(
                set(
                    [2, 4, 25, 6000, DECAY_START, FINAL]
                    + list(range(100, a + 1, 100))
                    + list(range(((a // 250) + 1) * 250, b + 1, 250))
                    + list(range(((b // 500) + 1) * 500, FINAL, 500))
                )
            )
        if self.kind == "decay":
            return [6002, 6004, 6100, 6200, 6300, 6400, 6500, 6600, 6667]
        return super().saves

    def as_dict(self):
        return dict(
            super().as_dict(),
            kind=self.kind,
            epochs=2 if self.stage == "sft" else None,
            dataset=(
                "allenai/Dolci-Think-SFT"
                if self.uses_dolci
                else (
                    "jacobmorrison/length-investigation-gptoss-120b-high"
                    if self.stage == "sft"
                    else None
                )
            ),
        )


def run(kind):
    assert kind in KINDS
    stage = (
        "sft"
        if "sft" in kind or "dolci" in kind
        else ("pt" if kind in ("hero", "decay") else kind.rsplit("-", 1)[-1])
    )
    return Run("7to1-split" if kind.startswith("dolci") else "3to1-shared", stage, False, kind)


def runs(smoke=False):
    assert not smoke
    return [run(k) for k in KINDS]


def find_run(name):
    return next(r for r in runs() if r.run_id == name)


def install():
    base.CAMPAIGN, base.BRANCH, base.ROOT, base.EVAL_ROOT = CAMPAIGN, BRANCH, ROOT, EVAL
    base.AUTOMATION = AUTO
    base.runs, base.find_run = runs, find_run


def self_test():
    assert run("hero").batch // (128 * 32768) == 4
    assert all(run(kind).gpus == 64 for kind in ("decay", "mt", "lc"))
    assert run("decay").batch // (run("decay").gpus * run("decay").microbatch) == (
        8 if int(os.environ.get("CAMPAIGN_PT_MB", "4")) == 4 else 16
    )
    assert run("hero").end * 16777216 == 2_013_265_920_000
    assert FINAL - DECAY_START == FINAL // 10
    assert DECAY_START in run("hero").saves
    assert run("decay").end == 6667 and run("decay").start == 6000
    assert run("mt").end == 2125 and run("lc").end == 8498 and run("sft").end == 840
    assert run("mt").lr == 2.2e-4 and run("lc").lr == 5.5e-5 and run("sft").lr == 5e-5
    assert all(not r.emo for r in runs())
    for kind in NEW_KINDS:
        r = run(kind)
        assert r.batch % (r.gpus * r.microbatch) == 0
        assert r.end in r.saves
    assert run("hero2t-mt").end * run("hero2t-mt").batch == 100_008_984_576
    assert run("hero2t-lc").end * run("hero2t-lc").batch == 100_008_984_576
