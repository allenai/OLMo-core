"""Minimal adapters around the qualified training recipe; no new numerical kernels."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from olmoe3_dolci_hero_plan import AUTO, DATA, PIN, ROOT, find_run, install, run

install()
import olmoe3_qkgain_train as adapter
from olmoe3_lr_sweep_watch import atomic_json
from olmo_core.distributed.utils import get_rank
from olmo_core.optim.scheduler import ConstantWithWarmup
from olmo_core.train.callbacks import Callback


@dataclass
class BranchPin(Callback):
    """Preserve the immutable branch initializer against ordinary trunk retention."""

    priority: ClassVar[int] = -15

    def post_checkpoint_saved(self, path):
        if get_rank() != 0 or Path(path).name != "step6000":
            return
        p = Path(path)
        assert p == run("hero").root / "step6000"
        PIN.parent.mkdir(parents=True, exist_ok=True)
        temporary = PIN.with_name("step6000.partial")
        if not PIN.exists():
            temporary.mkdir(exist_ok=True)
            for f in sorted(p.rglob("*")):
                assert not f.is_symlink()
                dest = temporary / f.relative_to(p)
                if f.is_dir():
                    dest.mkdir(exist_ok=True)
                elif f.is_file():
                    if dest.exists():
                        assert dest.stat().st_ino == f.stat().st_ino
                    else:
                        os.link(f, dest)
            temporary.rename(PIN)
        assert (PIN / "model_and_optim/.metadata").is_file()
        atomic_json(
            AUTO / "branch-source.json",
            dict(
                passed=True,
                source=str(PIN),
                step=6000,
                gpus=128,
                batch=16777216,
                metadata_sha256=hashlib.sha256((PIN / ".metadata.json").read_bytes()).hexdigest(),
                protection="owned immutable hardlink copy; never deleted automatically",
            ),
        )


def install_adapters(r):
    """Bind campaign-specific data/schedule before building import-qualified callbacks."""
    original_scheduler = adapter.scheduler
    adapter.scheduler = lambda x: (
        ConstantWithWarmup(warmup=2000) if x.kind == "hero" else original_scheduler(x)
    )
    adapter.hero.disk_action = lambda free: (
        "stop" if free < 10_000_000_000_000 else ("warn" if free < 12_000_000_000_000 else "ok")
    )
    original_trainer = adapter.trainer_config

    def trainer(common):
        c = original_trainer(common)
        if r.kind == "hero":
            c.callbacks["branch_pin"] = BranchPin()
        if (
            r.kind in ("hero", "decay")
            and int(os.environ.get("QKGAIN_STOP", r.end)) <= r.start + 25
        ):
            c.metrics_collect_interval = 1
            c.no_evals = True
        return c

    adapter.trainer_config = trainer
    if r.kind.startswith("dolci"):
        adapter.SFT_DATA = DATA
        adapter.SFT_CACHE = DATA / "packing-cache"
        adapter.SFT_DATA_PLAN = AUTO / "dolci-data-plan.json"


def train():
    r = adapter.current()
    install_adapters(r)
    adapter.hero.qualified.apply_policy()
    adapter.main(config_builder=adapter.builder(r))
