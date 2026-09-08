"""Approved small hero pair; dependency-free schedule and storage policy."""

from dataclasses import asdict, dataclass
from pathlib import Path

CAMPAIGN = "olmo35-small-hero-20260907"
BRANCH = "codex/small-hero-20260907"
WORKSPACE = "ai2/olmo3p5-training"
MOUNT = Path("/weka/olmo-3p5-checkpoints")
DOLMA_MOUNT = Path("/weka/dolma-3p5")
DATA_ROOT = DOLMA_MOUNT / "ai2-llm"
COMPLETE = DOLMA_MOUNT / ".transfer/dolma3p5-14t-e6c5a51bd6b9/COMPLETE.json"
MIX_SHA256 = "992ea0c56506fe0e03140f7094b5f52022885f5accfafd390c50a8bf193b0c1b"
ROOT = MOUNT / "production-hero-small" / CAMPAIGN
AUTOMATION = MOUNT / "uploader/automation" / CAMPAIGN
CONTROL = MOUNT / "uploader/control"
STATE = MOUNT / "uploader/state"
BUCKET = "allenai/olmo-3p5-small"
TEST_BUCKET = "allenai/olmo-checkpoint-uploader-pilot-20260902-jm01"
UPLOADER = "01M1YTRJV16A5YC300SEZAW6D1"
UPLOADER_COMMIT = "2ca46e2f1354bd1c8b558bbd90d6178c36053810"
QUALIFIED_EXPERIMENT = "01M1RN3NHFH32P2Z952BCR03YD"
# Observed failures in this campaign on 2026-09-07/08. Keep the runtime topology
# guard enabled; these exclusions affect only our smoke and future hero jobs.
EXCLUDED_HOSTNAMES = {
    "holmes-cs-aus-503.reviz.ai2.in",  # GPU1 disconnected; both initial heroes failed preflight.
    "holmes-cs-aus-534.reviz.ai2.in",  # GPU6 disconnected from every local NVLink peer.
    "holmes-cs-aus-550.reviz.ai2.in",  # Beaker interconnect healthcheck ALLREDUCE timeout.
}
BATCH = 16_777_216
LR = 1.1e-3
WARMUP = 2000
SWITCH_STEP = 60_000
INITIAL_STOP = 179_000
FINAL_STEPS = (14_000_000_000_000 + BATCH - 1) // BATCH
WARN_BYTES = 10_000_000_000_000
STOP_BYTES = 5_000_000_000_000
START_BYTES = 12_000_000_000_000


@dataclass(frozen=True)
class Run:
    """An independent, immutable initialization/optimizer/data lineage."""

    emo: bool
    smoke: bool = False

    @property
    def arm(self):
        return "emo" if self.emo else "non-emo"

    @property
    def run_id(self):
        return f"{CAMPAIGN}-{'smoke-' if self.smoke else ''}{self.arm}"

    @property
    def root(self):
        return ROOT / self.run_id

    @property
    def bucket(self):
        return TEST_BUCKET if self.smoke else BUCKET

    @property
    def prefix(self):
        return f"smoke/{self.run_id}" if self.smoke else self.arm

    def as_dict(self):
        return {
            **asdict(self),
            "run_id": self.run_id,
            "checkpoint_root": str(self.root),
            "bucket_id": self.bucket,
            "remote_prefix": self.prefix,
            "batch_tokens": BATCH,
            "lr": LR,
            "warmup": WARMUP,
            "initial_stop": INITIAL_STOP,
            "full_horizon": FINAL_STEPS,
        }


def runs(smoke=False):
    """Return the matched pair, with test payloads isolated from the hero bucket."""
    return [Run(True, smoke), Run(False, smoke)]


def find_run(name):
    """Reject arbitrary names/lineages."""
    return next(r for r in runs() + runs(True) if r.run_id == name)


def scheduled_save(step):
    """Describe native fixed-step plus periodic checkpoint scheduling."""
    return step == 0 or (0 < step <= SWITCH_STEP and step % 100 == 0) or step % 500 == 0


def disk_action(free_bytes):
    """Stop with ample room for two final checkpoints; never delete here."""
    return "stop" if free_bytes < STOP_BYTES else "warn" if free_bytes < WARN_BYTES else "ok"


def validate_plan():
    """Check the approved token budgets and independent remote/local namespaces."""
    assert INITIAL_STOP * BATCH == 3_003_121_664_000
    assert SWITCH_STEP * BATCH == 1_006_632_960_000
    assert FINAL_STEPS == 834_466
    assert FINAL_STEPS * BATCH >= 14_000_000_000_000 > (FINAL_STEPS - 1) * BATCH
    assert scheduled_save(INITIAL_STOP)
    assert BUCKET != TEST_BUCKET
    assert len({r.root for r in runs() + runs(True)}) == 4
    assert len({(r.bucket, r.prefix) for r in runs() + runs(True)}) == 4
    return [r.as_dict() for r in runs() + runs(True)]
