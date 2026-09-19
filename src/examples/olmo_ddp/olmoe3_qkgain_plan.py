"""Approved matched 111.854B PT -> ladder MT/LC -> high-reasoning SFT campaign."""

from dataclasses import dataclass
from pathlib import Path

CAMPAIGN = "olmo35-qkgain-20260919"
BRANCH = "codex/hybrid-qknorm-ladder-20260919"
MOUNT = Path("/weka/olmo-3p5-checkpoints")
ROOT = MOUNT / "production-qkgain" / CAMPAIGN
AUTOMATION = MOUNT / "uploader/automation" / CAMPAIGN
EVAL_ROOT = MOUNT / "scratch" / CAMPAIGN
DOWNLOAD_ROOT = AUTOMATION / "sources"
REFERENCE = DOWNLOAD_ROOT / "emo/step6000/olmo-core"
CONTROL = MOUNT / "uploader/control"
STATE = MOUNT / "uploader/state"
BUCKET = "allenai/olmo-3p5-small"
WORKSPACE = "ai2/olmo3p5-training"
EVAL_WORKSPACE = "ai2/OLMo-3-moe-experiments"
UPLOADER = "01M1YTRJV16A5YC300SEZAW6D1"
ARMS = ("3to1-split", "3to1-shared", "7to1-split")
STAGES = ("pt", "mt", "lc", "sft")
SFT_DATA = Path("/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/sft-data/gptoss120b-high-olmo-thinker-20260917")
SFT_CACHE = MOUNT / "production-hero-small-sft/olmo35-small-gptoss-high-sft-20260917/packing-cache"
SFT_DATA_PLAN = MOUNT / "uploader/automation/olmo35-small-gptoss-high-sft-20260917/data-plan.json"


@dataclass(frozen=True)
class Run:
    """Names and invariants for one independent stage, never an arbitrary checkpoint."""

    arm: str
    stage: str
    smoke: bool = False

    def __post_init__(self):
        assert self.arm in ARMS and self.stage in STAGES

    @property
    def run_id(self):
        return f"{CAMPAIGN}-{self.arm}-{self.stage}" + ("-smoke" if self.smoke else "")

    @property
    def root(self):
        return ROOT / self.run_id

    @property
    def prefix(self):
        return f"{CAMPAIGN}/{self.arm}/{self.stage}" + ("/smoke" if self.smoke else "")

    @property
    def bucket(self):
        return BUCKET

    @property
    def batch(self):
        return dict(pt=16777216, mt=786432, lc=786432, sft=524288)[self.stage]

    @property
    def sequence(self):
        return 8192 if self.stage in ("pt", "mt") else 65536

    @property
    def gpus(self):
        return dict(pt=64, mt=16, lc=12, sft=8)[self.stage]

    @property
    def nodes(self):
        return dict(pt=8, mt=2, lc=2, sft=1)[self.stage]

    @property
    def microbatch(self):
        return dict(pt=4, mt=3, lc=1, sft=1)[self.stage] * self.sequence

    @property
    def end(self):
        return dict(pt=6667, mt=45321, lc=45321, sft=3360)[self.stage]

    @property
    def start(self):
        return 6000 if self.stage == "pt" and self.arm == "7to1-split" else 0

    @property
    def lr(self):
        return dict(pt=1.1e-3, mt=8e-5, lc=4e-5, sft=5e-5)[self.stage]

    @property
    def emo(self):
        return self.stage == "pt"

    @property
    def split(self):
        return self.arm != "3to1-shared"

    @property
    def source(self):
        if self.stage == "pt":
            return REFERENCE if self.start else None
        previous = Run(self.arm, STAGES[STAGES.index(self.stage) - 1])
        return previous.root / f"step{previous.end}"

    @property
    def hf(self):
        # Frozen export CLI calls this slot 'emo'; full lineage lives in run_id.
        return EVAL_ROOT / self.run_id / "emo" / f"step{self.end}/hf"

    @property
    def saves(self):
        if self.smoke:
            return [self.start + 2, self.start + 4]
        if self.stage == "pt":
            # Entire short run is below 200B; keep the undecayed step6000 too.
            return list(range(100, self.end, 100)) + [self.end]
        if self.stage == "sft":
            return [1680, 3360]
        return list(range(5000, self.end, 5000)) + [self.end]

    def as_dict(self):
        return dict(run_id=self.run_id, arm=self.arm, stage=self.stage, smoke=self.smoke,
                    root=str(self.root), source=str(self.source) if self.source else None,
                    bucket=self.bucket, prefix=self.prefix, batch=self.batch, sequence=self.sequence,
                    gpus=self.gpus, nodes=self.nodes, microbatch=self.microbatch, lr=self.lr,
                    emo=self.emo, split_qk_gains=self.split, end=self.end,
                    tokens=self.end*self.batch, fixed_saves=self.saves, save_step0=self.start==0)


def runs(smoke=False):
    return [Run(arm, stage, smoke) for arm in ARMS for stage in STAGES]


def find_run(name):
    return next(r for r in runs() + runs(True) if r.run_id == name)


def validate_checkpoint(path, step=None, batch=None, gpus=None):
    """Require completed full-state checkpoints and all per-rank save audits."""
    import json
    path = Path(path)
    assert path.resolve() == path and not path.is_symlink()
    assert (path / ".metadata.json").is_file() and (path / "model_and_optim/.metadata").is_file()
    first = json.loads((path / "resume_audit/rank0.json").read_text())
    step = first["step"] if step is None else step
    gpus = first["gpus"] if gpus is None else gpus
    for rank in range(gpus):
        row = json.loads((path / "resume_audit" / f"rank{rank}.json").read_text())
        assert (row["step"], row["rank"], row["gpus"]) == (step, rank, gpus)
        assert batch is None or row["tokens"] == step * batch
        assert (path / "train" / f"rank{rank}.pt").is_file()


def self_test():
    import math
    pt = 6667*16777216
    span=math.log(1e12/4e9); pos=math.log(pt/4e9)/span
    budget=round(pt*(.5+(-1.2+.1*span)*pos**2+(.8-.1*span)*pos**3))
    assert budget == 35641421562 and math.ceil(budget/786432) == 45321
    assert pt == 111853699072
    assert len({r.run_id for r in runs()+runs(True)}) == 24
    for r in runs():
        assert r.batch % (r.gpus*r.microbatch) == 0
        assert r.gpus % r.nodes == 0 and r.end in r.saves
        assert len(r.saves)+1 == dict(pt=68,mt=11,lc=11,sft=3)[r.stage]
    assert sum(1+len(r.saves) for r in runs() if r.stage!='pt') == 75


if __name__ == "__main__":
    self_test()
    print("QKGAIN_PLAN_VALIDATED")
