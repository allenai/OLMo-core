"""Immutable small-production LR sweep plan; no Torch or service dependencies."""

from dataclasses import asdict, dataclass
from pathlib import Path

SWEEP = "small-lr100b-20260906-r1"
DEPLOYMENT = "launch2"
MOUNT = Path("/weka/olmo-3p5-checkpoints")
ROOT = MOUNT / "production-lr-sweeps" / SWEEP
CONTROL = MOUNT / "uploader/control"
STATE = MOUNT / "uploader/state"
AUTOMATION = MOUNT / "uploader/automation" / SWEEP / DEPLOYMENT
WORKSPACE = "ai2/olmo3p5-training"
BUCKET = "allenai/olmo-checkpoint-uploader-pilot-20260902-jm01"
BATCH = 16_777_216
STEPS = 6000
WARMUP = 2000
SAVE = 300
QUALIFIED_EXPERIMENT = "01M1RN3NHFH32P2Z952BCR03YD"
UPLOADER_EXPERIMENT = "01M1SDYA8S877VN54CJRN4RJZA"
UPLOADER_COMMIT = "3b3a102b106956ae8ca1ea8e92a0ad7ec8fbacf0"
LRS = (
    ("3p25em4", 3.25e-4),
    ("6p5em4", 6.5e-4),
    ("1p3em3", 1.3e-3),
    ("2p6em3", 2.6e-3),
    ("5p2em3", 5.2e-3),
)
DECAYS = (300, 600, 1200, 1800)


@dataclass(frozen=True)
class Run:
    """A distinct trajectory with a distinct uploader lineage."""

    run_id: str
    lr: float
    decay: int = 0
    parent: str | None = None
    end: int = STEPS
    warmup: int = WARMUP
    save: int = SAVE

    @property
    def start(self):
        return self.end - self.decay if self.parent else 0

    @property
    def root(self):
        return ROOT / self.run_id

    @property
    def keep(self):
        return 1 if self.parent else 7

    @property
    def parent_path(self):
        return ROOT / self.parent / f"step{self.start}" if self.parent else None

    def as_dict(self):
        return {
            **asdict(self),
            "start": self.start,
            "root": str(self.root),
            "min_local_checkpoints": self.keep,
            "batch_tokens": BATCH,
        }


def runs():
    """Return five trunks and twenty decay children in deterministic launch order."""
    trunks = [Run(f"olmoe3-{SWEEP}-lr{label}-trunk", lr) for label, lr in LRS]
    return trunks + [
        Run(t.run_id.replace("-trunk", f"-decay{d:04d}"), t.lr, d, t.run_id)
        for t in trunks
        for d in DECAYS
    ]


def smoke_runs():
    """The same model/topology with three short fresh-process save/restore passes."""
    parent = Run(f"olmoe3-{SWEEP}-smoke-trunk", 1.3e-3, end=6, warmup=2, save=2)
    child = Run(f"olmoe3-{SWEEP}-smoke-decay", 1.3e-3, 4, parent.run_id, end=6, warmup=2, save=2)
    return [parent, child]


def find_run(run_id):
    """Reject arbitrary names instead of silently falling back to sweep defaults."""
    return next(r for r in runs() + smoke_runs() if r.run_id == run_id)


def checkpoint_complete(path):
    """Check full-state completion, never accepting an in-progress directory."""
    import json

    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        return False
    if not all(
        (path / f).is_file()
        for f in (".metadata.json", "model_and_optim/.metadata", "train/rank0.pt")
    ):
        return False
    return json.loads((path / ".metadata.json").read_text()).get("ephemeral") is not True


def validate_plan():
    """Assert schedule, token budget, identity and retention invariants."""
    items = runs()
    assert len(items) == len({r.run_id for r in items}) == 25
    assert BATCH * STEPS == 100_663_296_000
    protected = set(range(STEPS - 6 * SAVE, STEPS + 1, SAVE))
    for r in items:
        assert len(r.run_id) <= 128
        assert r.end == STEPS and r.warmup == WARMUP and r.save == SAVE
        if r.parent:
            assert r.start in protected and r.decay % SAVE == 0
            assert r.start > WARMUP and r.start + r.decay == STEPS and r.keep == 1
        else:
            assert r.keep == 7
    return [r.as_dict() for r in items]
