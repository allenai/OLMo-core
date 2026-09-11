"""Exact live hero model/runtime with a separate, full-state linear WSD decay lineage."""

import json
import math
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import olmoe3_small_hero as hero
from olmoe3_hero_decay_plan import BATCH, END, LR, SMOKE_END, START, find_run, runs
from olmoe3_lr_sweep_watch import atomic_json

from olmo_core.distributed.utils import get_rank
from olmo_core.internal.experiment import CliContext, SubCmd, build_config, main
from olmo_core.optim.scheduler import WSD
from olmo_core.train import Duration
from olmo_core.train.common import LoadStrategy

# Entrypoint-local identity adapter only. The original hero file/core kernels remain unchanged.
hero.find_run = find_run


@dataclass
class DecayAudit(hero.HeroAudit):
    """Retain hero save/load audits and validate decay LR, data, and completion per rank."""

    def post_checkpoint_loaded(self, path):
        super().post_checkpoint_loaded(path)
        import torch

        from olmo_core.train.utils import EnvRngStates

        saved = torch.load(
            Path(path) / "train" / f"rank{get_rank()}.pt", map_location="cpu", weights_only=False
        )

        def equal(a, b):
            if isinstance(a, dict):
                return (
                    isinstance(b, dict)
                    and a.keys() == b.keys()
                    and all(equal(a[k], b[k]) for k in a)
                )
            if isinstance(a, (list, tuple)):
                return (
                    isinstance(b, (list, tuple))
                    and len(a) == len(b)
                    and all(equal(x, y) for x, y in zip(a, b))
                )
            if torch.is_tensor(a):
                return torch.equal(a.cpu(), b.cpu())
            if hasattr(a, "shape"):
                return bool((a == b).all())
            return a == b

        assert equal(saved["rng"], EnvRngStates.current_state().as_dict()), "RNG restore mismatch"
        assert equal(
            saved["data_loader"], self.trainer.data_loader.state_dict()
        ), "Data position mismatch"
        assert START <= self.step <= END
        for group in self.trainer.train_module.optim.param_groups:
            assert math.isclose(float(group.get("initial_lr", group["lr"])), LR, abs_tol=1e-10)
        atomic_json(
            Path(self.output_dir) / f"full-restore-step{self.step}-rank{get_rank()}.json",
            dict(
                source=str(path),
                step=self.step,
                sampled_state_exact=True,
                rng_exact=True,
                data_state_exact=True,
            ),
        )

    def log_metrics(self, step, metrics):
        super().log_metrics(step, metrics)
        for key, value in metrics.items():
            if key.startswith("optim/LR ("):
                expected = LR * (END - step) / (END - START)
                assert math.isclose(float(value), expected, rel_tol=1e-6, abs_tol=1e-10), (
                    step,
                    key,
                    value,
                    expected,
                )

    def post_train(self):
        stop = int(os.environ["OLMO35_HERO_STOP"])
        if not (find_run(self.run_id).root / "STORAGE_PAUSED.json").exists():
            assert self.step == stop and self.trainer.global_train_tokens_seen == stop * BATCH
        atomic_json(
            Path(self.output_dir) / f"decay-pass-step{self.step}-rank{get_rank()}.json",
            dict(step=self.step, requested_stop=stop, tokens=self.trainer.global_train_tokens_seen),
        )


def train_module_config(common):
    """Change only the schedule; restored optimizer state and base LR stay intact."""
    config = hero.train_module_config(common)
    config.scheduler = WSD(warmup=2000, decay=END - START, decay_fraction=None, decay_min_lr=0.0)
    return config


def trainer_config(common):
    """Independent local/upload/W&B lineage; same checkpoints, metrics, evals and precision."""
    r = find_run(common.run_name)
    config = hero.trainer_config(common)
    stop = int(os.environ["OLMO35_HERO_STOP"])
    assert stop in (SMOKE_END, END)
    config.max_duration = Duration.steps(END)
    config.hard_stop = Duration.steps(stop)
    config.load_strategy = LoadStrategy.always
    source = Path(os.environ.get("OLMO35_DECAY_LOAD", str(r.source)))
    allowed = {r.source} | {
        r.root / f"step{s}" for s in [SMOKE_END, *range(START + 500, END + 1, 500)]
    }
    # Interrupted clean saves may occur off-cadence; only accept a complete child path.
    if source not in allowed:
        assert (
            source.parent == r.root and source.name.startswith("step") and source.name[4:].isdigit()
        )
        assert START <= int(source.name[4:]) <= END
    config.load_path = str(source)
    config.callbacks["checkpointer"].fixed_steps = [SMOKE_END]
    config.callbacks["checkpointer"].save_interval = 500
    config.callbacks["hero_audit"] = DecayAudit(output_dir=str(r.root / "audit"), run_id=r.run_id)
    config.callbacks["wandb"].group = "olmo35-small-2t-decays"
    config.callbacks["wandb"].tags = [
        r.arm,
        "decay10-2t",
        "64g",
        "16mi",
        "mb4",
        "linear-wsd",
        "full-state-resume",
    ]
    if stop == SMOKE_END:
        config.no_evals = True
    return config


def config_builder():
    return partial(
        build_config,
        global_batch_size=BATCH,
        max_sequence_length=8192,
        num_nodes=8,
        common_config_builder=hero.common_components,
        data_config_builder=hero.data_components,
        model_config_builder=hero.model_config,
        train_module_config_builder=train_module_config,
        trainer_config_builder=trainer_config,
        beaker_image=hero.qualified.base.BEAKER_IMAGE,
        beaker_workspace=hero.WORKSPACE,
        include_default_evals=False,
        num_execution_units=1,
    )


def validate():
    """Build both actual configs, checking architecture, LR endpoints and full-state restore flags."""
    for run in runs():
        config = config_builder()(
            CliContext(__file__, SubCmd.dry_run, run.run_id, "ai2/holmes", [])
        )
        config.as_dict(json_safe=True)
        tm, tr = config.train_module, config.trainer
        assert (
            config.model.num_active_params == 794_233_472
            and config.model.num_params == 12_496_341_632
        )
        assert config.data_loader.global_batch_size == BATCH
        assert (
            tm.rank_microbatch_size == 32768 and tm.ac_config is None and tm.float8_config is None
        )
        assert tm.ep_config is None and tm.pp_config is None and tm.dp_config.use_reduce_scatter
        assert tr.load_optim_state and tr.load_trainer_state and not tr.save_overwrite
        assert tr.load_strategy == LoadStrategy.always and tr.load_path == str(run.source)
        assert not tr.callbacks["checkpointer"].save_async
        assert tr.callbacks["checkpointer"].max_checkpoints is None
        assert tm.scheduler.get_lr(LR, START, END) == LR
        assert math.isclose(tm.scheduler.get_lr(LR, START + 1, END), LR * 11999 / 12000)
        assert tm.scheduler.get_lr(LR, END, END) == 0
        print("HERO_DECAY_CONFIG_VALIDATED", json.dumps(run.as_dict()), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OLMO35_HERO_EXPECTED_START", str(START))
    os.environ.setdefault("OLMO35_HERO_STOP", str(END))
    os.environ["OLMO35_HERO_ALLOW_CONTINUATION"] = "1"
    hero.qualified.apply_policy()
    if sys.argv[1:] == ["--validate-only"]:
        validate()
    else:
        main(config_builder=config_builder())
