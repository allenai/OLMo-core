"""Production worker for the prepared continuous PT/MT decay; never submits jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import olmoe3_combined_decay_plan as p


def equal(a, b):
    """Compare nested trainer/RNG state without approximate tensor equality."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(map(equal, a, b))
    if hasattr(a, "shape"):
        return hasattr(b, "shape") and a.shape == b.shape and bool((a == b).all())
    return a == b


def mt_data(common, plan):
    """Use Tanush's frozen 100B mixture, tokenizer, seed and packing settings."""
    import olmo_core
    from olmo_core.data import InstanceFilterConfig, NumpyDataLoaderConfig, NumpyFSLDatasetConfig
    from olmo_core.data.source_mixture import SourceMixtureDatasetConfig, SourceMixtureList
    from olmo_core.internal.experiment import DataComponents

    mix = (
        Path(olmo_core.__file__).parent
        / "data/source_mixtures/OLMo3-32B-midtraining-modelnamefilter.yaml"
    )
    assert hashlib.sha256(mix.read_bytes()).hexdigest() == plan["mt_mixture_sha256"]
    sources = SourceMixtureList.from_yaml(str(mix))
    sources.validate()
    return DataComponents(
        dataset=NumpyFSLDatasetConfig.from_src_mix(
            src_mix=SourceMixtureDatasetConfig(
                source_list=sources,
                requested_tokens=p.MT_TOKENS,
                global_batch_size=p.BATCH,
                processes=16,
                seed=p.MT_SEED,
            ),
            tokenizer=common.tokenizer,
            work_dir=str(Path(plan["save_folder"]) / "mt-data-work"),
            sequence_length=8192,
            instance_filter_config=InstanceFilterConfig(),
        ),
        data_loader=NumpyDataLoaderConfig(
            global_batch_size=p.BATCH,
            seed=p.MT_SEED,
            num_workers=8,
            prefetch_factor=8,
            num_threads=4,
        ),
    )


def train_phase(args):
    """Run one data segment with a common absolute-step schedule and optimizer."""
    plan = p.fixed.read_json(args.plan)
    os.environ.update(
        OLMO35_HERO_GPUS=str(plan["gpus"]),
        OLMO35_HERO_ALLOW_CONTINUATION="1",
        OLMO35_HERO_SMOKE="0",
        WANDB_RESUME="allow",
        WANDB_RUN_ID=hashlib.sha256(plan["run_name"].encode()).hexdigest()[:16],
    )
    import olmoe3_small_hero as hero
    import torch
    import torch.distributed as dist
    from olmoe3_small_hero_runtime import verify_runtime

    from olmo_core.internal.experiment import CliContext, SubCmd, train
    from olmo_core.optim.scheduler import WSD
    from olmo_core.train import (
        Duration,
        prepare_training_environment,
        teardown_training_environment,
    )
    from olmo_core.train.common import LoadStrategy
    from olmo_core.train.utils import EnvRngStates

    verify_runtime()
    prepare_training_environment(backend="cpu:gloo,cuda:nccl", shared_filesystem=True)
    assert dist.get_world_size() == plan["gpus"] == 64
    selection = [None]
    if dist.get_rank() == 0:
        try:
            p.validate_plan(plan)
            root = Path(plan["save_folder"])
            root.mkdir(parents=True, exist_ok=True)
            identity = root / "combined-decay-plan.json"
            if identity.exists():
                assert p.fixed.read_json(identity) == plan, "Existing output has another plan"
            else:
                assert not any(root.iterdir()), "Refuse an unowned output directory"
                p.write_json(identity, plan)
            for flag in ("STOP.json", "STORAGE_PAUSED.json"):
                assert not (root / flag).exists(), f"Operator attention required: {flag}"
            selection[0] = dict(selection=p.select_checkpoint(plan))
        except Exception as exc:
            selection[0] = dict(error=repr(exc))
    dist.broadcast_object_list(selection, src=0)
    if "error" in selection[0]:
        raise ValueError(selection[0]["error"])
    start, load_path = selection[0]["selection"]
    stop = plan["switch_step"] if args.phase == "pt" else plan["end_step"]
    if start >= stop:
        # Reconcile endpoint protection after a failure just after saving the boundary.
        if dist.get_rank() == 0:
            p.protect_checkpoint(plan, stop)
        dist.barrier()
        teardown_training_environment()
        return
    assert (args.phase == "pt" and start < plan["switch_step"]) or (
        args.phase == "mt" and plan["switch_step"] <= start < plan["end_step"]
    )
    os.environ.update(OLMO35_HERO_EXPECTED_START=str(start), OLMO35_HERO_STOP=str(stop))
    child = p.fixed.ChildRun(plan)
    registration = p.fixed.read_json(hero.CONTROL / "registrations" / f"{child.run_id}.json")
    assert all(registration.get(k) == v for k, v in plan["registration"].items())
    assert registration["enabled"]

    def find_run(name):
        if name != child.run_id:
            raise ValueError(name)
        return child

    hero.find_run = find_run
    hero.FINAL_STEPS = plan["end_step"]
    hero.qualified.apply_policy()
    transition = args.phase == "mt" and start == plan["switch_step"]
    if args.phase == "mt":
        hero.data_components = lambda common: mt_data(common, plan)

    @dataclass
    class CombinedAudit(hero.HeroAudit):
        """Audit resharding, exact ordinary restores, the data switch and continuous LR."""

        def post_attach(self):
            if transition:
                original_load = self.trainer.load_state_dict
                self._transition_seen = False

                def load(state):
                    assert not self._transition_seen
                    self._transition_seen = True
                    fresh = self.trainer.data_loader.state_dict()
                    adjusted = p.transition_state(plan, state, fresh)
                    self._expected_data = adjusted["data_loader"]
                    return original_load(adjusted)

                self.trainer.load_state_dict = load

        def post_checkpoint_loaded(self, path):
            rank = dist.get_rank()
            saved = torch.load(
                Path(path) / "train" / f"rank{rank}.pt", map_location="cpu", weights_only=False
            )
            initial_reshard = Path(path) == Path(plan["source"])
            if initial_reshard:
                from olmoe3_combined_decay_reshard import verify_halved_dp

                proof = verify_halved_dp(self.trainer, path, hero.state_sample(self.trainer))
            else:
                super().post_checkpoint_loaded(path)
                assert equal(saved["rng"], EnvRngStates.current_state().as_dict())
                proof = dict(sampled_state_exact=True, rng_exact=True)
            expected = self._expected_data if transition else saved["data_loader"]
            assert equal(expected, self.trainer.data_loader.state_dict())
            assert self.step == start and self.trainer.global_train_tokens_seen == start * p.BATCH
            if transition:
                assert self._transition_seen and self.trainer.epoch == 1
            p.write_json(
                Path(self.output_dir) / f"{args.phase}-restore-{start}-rank{rank}.json",
                dict(
                    source=str(path),
                    step=start,
                    data_state_exact=True,
                    data_reset=transition,
                    optimizer_reset=False,
                    **proof,
                ),
            )

        def pre_train(self):
            assert hero.MOUNT.is_mount() and hero.DOLMA_MOUNT.is_mount()
            assert Path(self.trainer.save_folder) == child.root
            assert self.trainer.global_train_tokens_seen == self.step * p.BATCH
            expected_tokens = (
                self.step - plan["switch_step"] if args.phase == "mt" else self.step
            ) * p.BATCH
            assert self.trainer.data_loader.tokens_processed == expected_tokens
            assert not (child.root / "STORAGE_PAUSED.json").exists()
            output = Path(self.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            self._first_batch = True
            tm = self.trainer.train_module
            original_save = tm.save_state_dict_direct

            def guarded_save(directory, **kwargs):
                before = hero.state_sample(self.trainer)
                original_save(directory, **kwargs)
                after = hero.state_sample(self.trainer)
                assert before == after, "Saving mutated model/optimizer state"
                target = Path(directory).parent
                p.write_json(target / "resume_audit" / f"rank{dist.get_rank()}.json", after)
                if dist.get_rank() == 0:
                    p.write_json(
                        target / "combined-decay.json",
                        dict(plan_sha256=p.digest(plan), phase=args.phase),
                    )

            tm.save_state_dict_direct = guarded_save
            if dist.get_rank() == 0:
                p.write_json(
                    output / f'session-{os.environ.get("BEAKER_JOB_ID")}-{start}.json',
                    dict(
                        plan=plan,
                        phase=args.phase,
                        start=start,
                        stop=stop,
                        source_commit=os.environ.get("GIT_REF"),
                        settings=hero.qualified.SETTINGS,
                    ),
                )

        def log_metrics(self, step, metrics):
            super().log_metrics(step, metrics)
            for name, value in metrics.items():
                if name.startswith("optim/LR ("):
                    assert math.isclose(
                        float(value), p.learning_rate(plan, step), rel_tol=1e-6, abs_tol=1e-10
                    )

    config = hero.config_builder()(
        CliContext(__file__, SubCmd.train, child.run_id, args.cluster, [])
    )
    source_config = p.fixed.read_json(Path(plan["source"]) / "config.json")
    assert json.loads(json.dumps(config.model.as_config_dict())) == source_config["model"]
    assert config.data_loader.global_batch_size == p.BATCH
    assert config.dataset.sequence_length == 8192
    if args.phase == "pt":
        assert p.digest(config.dataset.as_config_dict()) == plan["source_dataset_sha256"]
    config.launch = None
    tm, tr = config.train_module, config.trainer
    tm.scheduler = WSD(warmup=2000, decay=plan["decay_steps"], decay_fraction=None, decay_min_lr=0)
    tm.reset_optimizer_states_on_load = tm.reset_optimizer_states_on_resume = False
    assert tm.optim.lr == p.LR
    tr.load_path = load_path
    tr.load_strategy = LoadStrategy.always
    tr.load_optim_state = tr.load_trainer_state = True
    tr.max_duration = Duration.steps(plan["end_step"])
    tr.hard_stop = Duration.steps(stop)
    tr.metrics_collect_interval = 1
    cp = tr.callbacks["checkpointer"]
    cp.pre_train_checkpoint = False
    cp.save_interval = 500
    cp.fixed_steps = sorted({start + 10, stop})
    cp.max_checkpoints = None
    tr.callbacks["hero_audit"] = CombinedAudit(
        output_dir=str(child.root / "audit"), run_id=child.run_id
    )
    wb = tr.callbacks["wandb"]
    wb.group = "hero-combined-pt-mt-decay"
    wb.tags = ["7to1", "emo", "64g", "continuous-decay", "600b-pt", "100b-mt", args.phase]
    try:
        train(config)
        assert not (
            child.root / "STORAGE_PAUSED.json"
        ).exists(), "Storage pause requires inspection"
        if dist.get_rank() == 0:
            info = p.fixed.checkpoint_info(child.root / f"step{stop}")
            assert info["start_step"] == stop and info["gpus"] == plan["gpus"]
            p.protect_checkpoint(plan, stop)
            p.write_json(
                child.root / "audit" / f"{args.phase}-success.json",
                dict(passed=True, step=stop, plan_sha256=p.digest(plan)),
            )
        dist.barrier()
    except (AssertionError, ValueError, FloatingPointError) as exc:
        p.write_json(
            child.root / "STOP.json", dict(error=repr(exc), phase=args.phase, rank=dist.get_rank())
        )
        raise
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--phase", choices=["pt", "mt"], required=True)
    parser.add_argument("--cluster", default="ai2/holmes")
    train_phase(parser.parse_args())
