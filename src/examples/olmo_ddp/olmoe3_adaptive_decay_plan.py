"""Two authorized 32-GPU, native-denominator expert-cooldown branches."""

from dataclasses import dataclass
from pathlib import Path

import olmoe3_qkgain_plan as base

CAMPAIGN = "adaptive-small-decay-20261003-32g"
BRANCH = "jacobm/adaptive-compute-2026-10-03-expert-decay"
SCRIPT = "src/examples/olmo_ddp/olmoe3_adaptive_decay.py"
WORKSPACE = "ai2/olmo3p5-training"
SOURCE_GPUS = 128
GPUS = 32
GPUS_PER_NODE = 8
NODES = GPUS // GPUS_PER_NODE
ROOT = base.MOUNT / "adaptive-compute-redux" / CAMPAIGN
EVAL = ROOT / "evals"
AUTO = Path(
    "/weka/oe-adapt-default/jacobm/adaptive-compute-redux/results/data/olmo_small_decay_2026-10-03/training32g"
)
SOURCE = base.MOUNT / "uploader/automation/olmo35-dolci-hero-20260920/sources/hero/step108000"
START = 108000
END = 120000
BATCH = 16777216
SCHEDULES = ("fixed8", "gradual8")


def expert_count(schedule: str, update: int) -> int:
    """Return K for a one-based optimizer update, including resumed boundaries."""
    if schedule not in SCHEDULES or not START <= update <= END:
        raise ValueError((schedule, update))
    if schedule == "fixed8":
        return 8
    return max(8, 16 - 2 * (max(0, update - START - 1) // 1500))


@dataclass(frozen=True)
class Run(base.Run):
    """Immutable identities, checkpoint roots and matched global training recipe."""

    schedule: str = "fixed8"

    @property
    def gpus(self):
        return GPUS

    @property
    def nodes(self):
        return NODES

    @property
    def run_id(self):
        return f"{CAMPAIGN}-{self.schedule}"

    @property
    def root(self):
        return ROOT / self.run_id

    @property
    def prefix(self):
        return f"adaptive-compute-redux/{CAMPAIGN}/{self.schedule}"

    @property
    def start(self):
        return START

    @property
    def end(self):
        return END

    @property
    def source(self):
        return SOURCE

    @property
    def emo(self):
        return False

    @property
    def hf(self):
        return EVAL / self.run_id / f"step{END}" / "hf"

    @property
    def saves(self):
        return [START + 2, START + 4] + list(range(START + 500, END + 1, 500))

    def as_dict(self):
        return dict(
            super().as_dict(),
            schedule=self.schedule,
            start=START,
            added_tokens=(END - START) * BATCH,
            reference_top_k=16,
            normalization="native-top16-denominator-times16",
            initial_gpus=SOURCE_GPUS,
            accumulation=self.batch // (self.gpus * self.microbatch),
            world_size_change="Optimizer reshard; per-rank RNG reinitialized; microbatch LB group is32",
        )


def run(schedule):
    """Resolve one explicitly authorized arm."""
    if schedule not in SCHEDULES:
        raise ValueError(schedule)
    return Run("3to1-shared", "pt", False, schedule)


def runs(smoke=False):
    """Return the two full runs; their startup checks are inside their allocations."""
    if smoke:
        return []
    return [run(s) for s in SCHEDULES]


def find_run(name):
    """Resolve an exact owned run identifier."""
    return next(r for r in runs() if r.run_id == name)


def install():
    """Bind the historical recipe adapter to this campaign before importing it."""
    base.CAMPAIGN, base.BRANCH, base.ROOT, base.EVAL_ROOT = CAMPAIGN, BRANCH, ROOT, EVAL
    base.AUTOMATION, base.WORKSPACE = AUTO, WORKSPACE
    base.runs, base.find_run = runs, find_run


def self_test():
    """Check schedule boundaries, data budget and physical GPU allocation."""
    for r in runs():
        assert r.gpus == 32 and r.nodes == 4 and r.batch == BATCH
        assert r.batch // (r.gpus * r.microbatch) == 16
        assert r.lr == 1.1e-3 and r.sequence == 8192 and not r.emo and not r.split
        assert (r.end - r.start) * r.batch == 201326592000
        assert len(set(r.saves)) == len(r.saves) and END in r.saves
    assert [
        expert_count("gradual8", s)
        for s in (
            108001,
            109500,
            109501,
            111000,
            111001,
            112501,
            114000,
            114001,
            120000,
        )
    ] == [16, 16, 14, 14, 12, 10, 10, 8, 8]
    assert sum(expert_count("gradual8", s) for s in range(START + 1, END + 1)) == 126000


if __name__ == "__main__":
    self_test()
    print("ADAPTIVE_DECAY_PLAN_PASSED")
