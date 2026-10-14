"""Authorized 64-GPU 14T+600B+100B MT -> 100B LC -> full Dolci Think SFT."""

from dataclasses import dataclass

import olmoe3_qkgain_plan as base

CAMPAIGN = "olmo35-14t600b-mt100b-lc-sft-20261014"
BRANCH = "codex/hero14t-lc-sft-64gpu-20261014"
SCRIPT = "src/examples/olmo_ddp/olmoe3_14t_lc_sft.py"
ROOT = base.MOUNT / "production-lc-sft" / CAMPAIGN
AUTO = base.MOUNT / "uploader/automation" / CAMPAIGN
EVAL = base.MOUNT / "scratch" / CAMPAIGN
PROTECTED = base.MOUNT / "protected" / CAMPAIGN
ORIGINAL_PARENT = base.MOUNT / (
    "production-hero-small-midtrain/olmo35-small-14t600b-mt20-20261012/"
    "olmo35-small-14t600b-mt20-20261012-emo/step5961"
)
PARENT = PROTECTED / "mt/step5961/olmo-core"


@dataclass(frozen=True)
class Run(base.Run):
    """Two immutable stage identities and their agreed numerical settings."""

    @property
    def run_id(self):
        return CAMPAIGN + "-" + self.stage

    @property
    def root(self):
        return ROOT / self.run_id

    @property
    def prefix(self):
        return CAMPAIGN + "/" + self.stage

    @property
    def gpus(self):
        return 64

    @property
    def nodes(self):
        return 8

    @property
    def batch(self):
        return 16777216 if self.stage == "lc" else 8388608

    @property
    def end(self):
        return 5961 if self.stage == "lc" else 5402

    @property
    def lr(self):
        return 1.1e-4 if self.stage == "lc" else 5e-5

    @property
    def emo(self):
        return False

    @property
    def source(self):
        return PARENT if self.stage == "lc" else PROTECTED / "lc/step5961/olmo-core"

    @property
    def saves(self):
        return sorted(
            set(
                [10, *range(500, self.end, 500), self.end] + ([2701] if self.stage == "sft" else [])
            )
        )

    @property
    def epochs(self):
        return 2

    @property
    def dataset(self):
        return "dolci-think"

    @property
    def milestone(self):
        return "14t600b-mt100b-lc100b"

    @property
    def lineage(self):
        return "emo"

    @property
    def data(self):
        from olmoe3_corrected_sft_plan import DATA

        return DATA / self.dataset

    @property
    def data_plan(self):
        return self.data / "data-plan.json"

    @property
    def future_parent(self):
        return self.stage == "sft"

    @property
    def hf(self):
        return EVAL / self.run_id / "emo" / f"step{self.end}/hf"

    def as_dict(self):
        return dict(
            super().as_dict(),
            epochs=2 if self.stage == "sft" else None,
            dataset=(
                "Dolci-Think-SFT-32B" if self.stage == "sft" else "lc-reshard-final-cleaned/v0.1"
            ),
            save_step0=False,
            warmup=2000 if self.stage == "lc" else "3%",
            weight_decay=0.1 if self.stage == "lc" else 0,
            parent_mt_emo=True,
            lc_emo=False,
            sft_emo=False,
        )


def run(stage):
    """Resolve only the two authorized stages."""
    assert stage in ("lc", "sft")
    return Run("7to1-split", stage, False)


def runs(smoke=False):
    """Return the production chain; no independent crash/resume test jobs."""
    assert not smoke
    return [run("lc"), run("sft")]


def find_run(name):
    """Reject identities outside this campaign."""
    return next(r for r in runs() if r.run_id == name)


def install():
    """Bind the existing recipe modules before importing their runtime adapters."""
    import olmoe3_corrected_sft_plan as s

    for module in (base, s):
        module.CAMPAIGN, module.BRANCH, module.ROOT = CAMPAIGN, BRANCH, ROOT
        module.EVAL_ROOT = module.EVAL = EVAL
        module.AUTOMATION = module.AUTO = AUTO
        module.SCRIPT = SCRIPT
        module.runs, module.find_run = runs, find_run
