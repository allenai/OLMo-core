"""Token-accounted medium CBS plan; this module never submits experiments."""

import math
from dataclasses import dataclass
from pathlib import Path

CAMPAIGN = "olmoe3-medium-cbs-20260907-r1"
MOUNT = Path("/weka/olmo-3p5-checkpoints")
ROOT = MOUNT / "production-cbs-medium"
CONTROL = MOUNT / "uploader/control"
AUTOMATION = MOUNT / "uploader/automation" / CAMPAIGN
WORKSPACE = "ai2/olmo3p5-training"
B16 = 16_777_216
B32 = 33_554_432
BASELINE_LR = 9.2e-4
BRANCH_LR = BASELINE_LR * math.sqrt(2)
TARGET_TOKENS = 6000 * B16
FORK_STEP = 4000
FORK_TOKENS = FORK_STEP * B16


@dataclass(frozen=True)
class Run:
    """A lineage with explicit absolute token accounting after a batch-size fork."""

    run_id: str
    batch: int
    lr: float
    start: int
    end: int
    interval: int
    keep: int
    parent: str | None = None
    parent_step: int = 0

    @property
    def root(self):
        """Checkpoint root, disjoint from all profiling evidence."""
        return ROOT / self.run_id

    @property
    def load_path(self):
        """Initial parent restore; the trainer prefers this run's own newer saves."""
        return ROOT / self.parent / f"step{self.parent_step}" if self.parent else None

    def tokens_at(self, step):
        """Count restored parent tokens plus newly processed child tokens."""
        if step < self.start or step > self.end:
            raise ValueError((self.run_id, step))
        return self.parent_step * B16 + (step - self.start) * self.batch


BASELINE = Run(f"{CAMPAIGN}-16mi", B16, BASELINE_LR, 0, 6000, 500, 5)
BRANCH = Run(
    f"{CAMPAIGN}-32mi", B32, BRANCH_LR, FORK_STEP, 5000, 250, 2, BASELINE.run_id, FORK_STEP
)
SMOKE_BASELINE = Run(f"{CAMPAIGN}-smoke16mi", B16, BASELINE_LR, 0, 4, 2, 2)
SMOKE_BRANCH = Run(f"{CAMPAIGN}-smoke32mi", B32, BRANCH_LR, 2, 4, 1, 2, SMOKE_BASELINE.run_id, 2)
RUNS = (BASELINE, BRANCH, SMOKE_BASELINE, SMOKE_BRANCH)


def find_run(name):
    """Reject arbitrary run names, paths, token budgets, and branches."""
    return next(r for r in RUNS if r.run_id == name)


def validate():
    """Validate branch alignment, token horizons and persistent fork retention."""
    assert BASELINE.tokens_at(BASELINE.end) == BRANCH.tokens_at(BRANCH.end) == TARGET_TOKENS
    assert BRANCH.tokens_at(BRANCH.start) == FORK_TOKENS
    assert FORK_TOKENS % B32 == 0
    assert BASELINE.interval * B16 == BRANCH.interval * B32
    latest_parent_steps = list(range(0, BASELINE.end + 1, BASELINE.interval))[-BASELINE.keep :]
    assert FORK_STEP in latest_parent_steps
    assert math.isclose(BRANCH.lr / BASELINE.lr, math.sqrt(2), rel_tol=1e-15)
    assert SMOKE_BRANCH.tokens_at(2) == SMOKE_BASELINE.tokens_at(2)
    assert all(r.keep >= 2 and r.end % r.interval == 0 for r in RUNS)


if __name__ == "__main__":
    validate()
    print("MEDIUM_CBS_PLAN_VALIDATED", TARGET_TOKENS, FORK_TOKENS, BASELINE_LR, BRANCH_LR)
