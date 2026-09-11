"""Exact, bounded September10 medium resume/speed/CBS campaign; no submissions."""

import math
from dataclasses import dataclass
from pathlib import Path

CAMPAIGN = "olmoe3-medium-cbs-20260910"
BRANCH_NAME = "codex/medium-cbs64-20260910"
MOUNT = Path("/weka/olmo-3p5-checkpoints")
ROOT = MOUNT / "production-cbs-medium" / CAMPAIGN
CONTROL = MOUNT / "uploader/control"
AUTOMATION = MOUNT / "uploader/automation" / CAMPAIGN
WORKSPACE = "ai2/olmo3p5-training"
# CPU-only, Weka-enabled, and batch-capable; consumes no GPU allocation.
CPU_CLUSTER = "ai2/phobos"
BUCKET = "allenai/olmo-checkpoint-uploader-pilot-20260902-jm01"
UPLOADER = "01M1YTRJV16A5YC300SEZAW6D1"
PARENT_RUN = "olmoe3-medium-cbs-20260907-r3-16mi"
PARENT_ROOT = MOUNT / "production-cbs-medium" / PARENT_RUN
PARENT = PARENT_ROOT / "step4000"
B16 = 16_777_216
FORK_STEP = 4000
FORK_TOKENS = FORK_STEP * B16
TARGET_TOKENS = 6000 * B16
VARIANT = "optimized-metrics5"
# Explicit user-authorized retry; preserve the original submission ledger.
SUBMISSION_SUFFIX = "-r2"


@dataclass(frozen=True)
class Phase:
    """Each timing pass and actual branch has a separate immutable output root."""

    name: str
    gpus: int
    batch_mi: int
    start: int = FORK_STEP
    end: int = FORK_STEP + 60
    source_phase: str | None = None
    save: bool = False
    cbs: bool = False

    @property
    def run_id(self):
        return f"{CAMPAIGN}-{self.name}"

    @property
    def root(self):
        return ROOT / self.name

    @property
    def batch(self):
        return self.batch_mi * 1024**2

    @property
    def lr(self):
        return 9.2e-4 * math.sqrt(self.batch_mi / 16)

    @property
    def load_path(self):
        return ROOT / self.source_phase / f"step{self.start}" if self.source_phase else PARENT

    @property
    def save_interval(self):
        return 250 if self.batch_mi == 32 else 125

    @property
    def eval_interval(self):
        return 500 if self.batch_mi == 32 else 250

    def tokens_at(self, step):
        return FORK_TOKENS + (step - FORK_STEP) * self.batch


PHASES = {
    p.name: p
    for p in (
        Phase("64g-32mi-speed", 64, 32, save=True),
        Phase("64g-32mi-restore", 64, 32, 4060, 4061, "64g-32mi-speed"),
        Phase("64g-64mi-speed", 64, 64, save=True),
        Phase("64g-64mi-restore", 64, 64, 4060, 4061, "64g-64mi-speed"),
        Phase("128g-32mi-a", 128, 32),
        Phase("128g-64mi", 128, 64),
        Phase("128g-32mi-b", 128, 32),
        Phase("64g-32mi-cbs", 64, 32, end=5000, save=True, cbs=True),
        Phase("64g-64mi-cbs", 64, 64, end=4500, save=True, cbs=True),
    )
}
WAVES = {
    "resume-speed64": tuple(list(PHASES)[:4]),
    "speed128": tuple(list(PHASES)[4:7]),
    "cbs32": ("64g-32mi-cbs",),
    "cbs64": ("64g-64mi-cbs",),
}


def phase_environment(environ, phase):
    """Pin timing-only phase settings before imports, without inherited Nsight ranks."""
    env = {k: v for k, v in environ.items() if not k.startswith("OLMOE3_NSYS_")}
    env.update(
        OLMOE3_MEDIUM_CBS64_PHASE=phase.name,
        OLMOE3_MEDIUM_GPUS=str(phase.gpus),
        OLMOE3_MEDIUM_MB="2",
        OLMOE3_MEDIUM_BATCH=str(phase.batch),
        OLMOE3_DEEP_PROFILE_TEST=VARIANT,
        OLMOE3_DEEP_PROFILE_PASS="timing",
        OLMOE3_MEDIUM_CAPTURE="0",
    )
    return env


def old_rank_for_half(rank, half, group):
    """Contiguous optimizer halves: dense DP, or node-local EP8's DP group."""
    if not 0 <= rank < 64 or half not in (0, 1) or group not in ("dp", "ep_dp"):
        raise ValueError((rank, half, group))
    return 2 * rank + half if group == "dp" else (2 * (rank // 8) + half) * 8 + rank % 8


def validate():
    """Reject accidental warmup, horizon, topology, or branch-token changes."""
    for phase in PHASES.values():
        assert phase.gpus in (64, 128) and phase.batch_mi in (32, 64)
        assert phase.batch % (phase.gpus * 2 * 8192) == 0
        assert FORK_STEP <= phase.start < phase.end <= 5000
        assert phase.tokens_at(phase.start) < phase.tokens_at(phase.end) <= TARGET_TOKENS
        if phase.cbs:
            assert phase.tokens_at(phase.end) == TARGET_TOKENS
            assert phase.end % phase.save_interval == 0
        if phase.source_phase:
            source = PHASES[phase.source_phase]
            assert source.save and source.end == phase.start and source.batch == phase.batch
    for group in ("dp", "ep_dp"):
        assert sorted(old_rank_for_half(r, h, group) for r in range(64) for h in (0, 1)) == list(
            range(128)
        )


if __name__ == "__main__":
    validate()
    print("MEDIUM_CBS64_PLAN_VALID", FORK_TOKENS, TARGET_TOKENS)
