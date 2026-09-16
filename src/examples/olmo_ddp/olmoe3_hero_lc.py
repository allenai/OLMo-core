"""Exact hero architecture/runtime; fresh LC data/optimizer with the ladder's 10% PT LR."""

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import olmoe3_small_hero as hero
from olmoe3_hero_lc_plan import (
    BATCH,
    CAMPAIGN,
    DATA_GLOB,
    DATA_WORK,
    END,
    GATE_END,
    LR,
    PACKING_CACHE,
    REQUESTED_TOKENS,
    SEED,
    SEQUENCE_LENGTH,
    SMOKE_END,
    SOURCE_STEP,
    WARMUP,
    find_run,
    runs,
)
from olmoe3_lr_sweep_watch import atomic_json

from olmo_core.data import (
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
)
from olmo_core.distributed.utils import get_rank
from olmo_core.internal.experiment import (
    CliContext,
    DataComponents,
    SubCmd,
    build_config,
    main,
)
from olmo_core.optim.scheduler import LinearWithWarmup
from olmo_core.train import Duration
from olmo_core.train.common import LoadStrategy

hero.find_run = find_run


def common_components(context, **kwargs):
    common = hero.common_components(context, **kwargs)
    common.work_dir = str(DATA_WORK)
    return common


def data_components(common):
    """Reuse the ladder's frozen 100B LC mixture and data-loader settings."""
    return DataComponents(
        dataset=NumpyPackedFSLDatasetConfig.glob(
            DATA_GLOB,
            tokenizer=common.tokenizer,
            work_dir=str(PACKING_CACHE),
            sequence_length=SEQUENCE_LENGTH,
            source_group_size=8,
            source_permutation_seed=123,
            instance_filter_config=InstanceFilterConfig(),
        ),
        data_loader=NumpyDataLoaderConfig(
            global_batch_size=BATCH,
            work_dir=str(DATA_WORK / common.run_name / "loader"),
            seed=SEED,
            num_workers=16,
            prefetch_factor=8,
            num_threads=4,
        ),
    )


@dataclass
class LCAudit(hero.HeroAudit):
    """Validate weights-only transfer, then full-state LC resumes and finite linear LRs."""

    def post_checkpoint_loaded(self, path):
        import torch

        from olmo_core.train.utils import EnvRngStates

        r = find_run(self.run_id)
        if Path(path) == r.source:
            assert self.step == self.trainer.global_train_tokens_seen == 0
            assert self.trainer.data_loader.tokens_processed == 0
            saved = json.loads((r.source / "resume_audit" / f"rank{get_rank()}.json").read_text())
            current = hero.state_sample(self.trainer)
            keys = {
                k for k in saved["tensors"] if k.startswith("model_param/") or k.endswith(".main")
            }
            keys.update(self.trainer.train_module._persistent_model_buffer_state_dict())
            assert keys and all(current["tensors"][k] == saved["tensors"][k] for k in keys)
            optim = self.trainer.train_module.optim
            assert not optim._losses and not optim._grad_norms
            moments = 0
            for name, tensor in optim.states.items():
                if name.endswith((".exp_avg", ".exp_avg_sq", ".step")):
                    tensor = tensor.to_local() if hasattr(tensor, "to_local") else tensor
                    assert not torch.count_nonzero(tensor).item(), name
                    moments += 1
            assert moments > 0
            proof = dict(
                weights_and_buffers_sampled_exact=True,
                optimizer_reset=True,
                data_reset=True,
                source_step=SOURCE_STEP,
            )
            atomic_json(Path(self.output_dir) / f"initial-lc-transfer-rank{get_rank()}.json", proof)
        else:
            assert Path(path).parent == r.root
            super().post_checkpoint_loaded(path)
            saved = torch.load(
                Path(path) / "train" / f"rank{get_rank()}.pt",
                map_location="cpu",
                weights_only=False,
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

            assert equal(saved["rng"], EnvRngStates.current_state().as_dict())
            assert equal(saved["data_loader"], self.trainer.data_loader.state_dict())
            proof = dict(sampled_state_exact=True, rng_exact=True, data_state_exact=True)
        atomic_json(
            Path(self.output_dir) / f"lc-restore-step{self.step}-rank{get_rank()}.json",
            dict(source=str(path), step=self.step, **proof),
        )

    def log_metrics(self, step, metrics):
        super().log_metrics(step, metrics)
        for key, value in metrics.items():
            if key.startswith("optim/LR ("):
                expected = LinearWithWarmup(warmup=WARMUP, alpha_f=0.0).get_lr(LR, step, END)
                assert math.isclose(float(value), expected, rel_tol=1e-6, abs_tol=1e-10)

    def post_train(self):
        stop = int(os.environ["OLMO35_HERO_STOP"])
        if not (find_run(self.run_id).root / "STORAGE_PAUSED.json").exists():
            assert self.step == stop and self.trainer.global_train_tokens_seen == stop * BATCH


def train_module_config(common):
    config = hero.train_module_config(common)
    config.rank_microbatch_size = SEQUENCE_LENGTH
    config.optim.lr = LR
    config.scheduler = LinearWithWarmup(warmup=WARMUP, alpha_f=0.0)
    r = find_run(common.run_name)
    config.reset_optimizer_states_on_load = (
        Path(os.environ.get("OLMO35_LC_LOAD", str(r.source))) == r.source
    )
    return config


def trainer_config(common):
    r = find_run(common.run_name)
    config = hero.trainer_config(common)
    start = int(os.environ.get("OLMO35_HERO_EXPECTED_START", "0"))
    stop = int(os.environ.get("OLMO35_HERO_STOP", str(END)))
    assert 0 <= start <= stop <= END
    config.max_duration = Duration.steps(END)
    config.hard_stop = Duration.steps(stop)
    if stop <= GATE_END:
        config.metrics_collect_interval = 1
    config.load_strategy = LoadStrategy.always
    source = Path(os.environ.get("OLMO35_LC_LOAD", str(r.source)))
    fresh = source == r.source
    assert (fresh and start == 0) or source == r.root / f"step{start}"
    config.load_path = str(source)
    config.load_optim_state = not fresh
    config.load_trainer_state = not fresh
    cp = config.callbacks["checkpointer"]
    cp.fixed_steps = [SMOKE_END, GATE_END]
    cp.save_interval = 500
    config.callbacks["hero_audit"] = LCAudit(output_dir=str(r.root / "audit"), run_id=r.run_id)
    wb = config.callbacks["wandb"]
    wb.group = CAMPAIGN
    wb.tags = [
        r.arm,
        "long-context",
        "100b-mt-after-decayed-4t-source",
        "lc-half-mt-lr",
        "linear",
        "64g",
        "16mi",
        "mb1-64k",
        "block-recomputation",
        "sync-checkpoints",
    ]
    wb.notes = json.dumps(r.as_dict())
    if stop <= GATE_END:
        config.no_evals = True
    return config


def model_config(common):
    """Preserve hero weights/architecture, with the successful ladder's block recompute."""
    from olmoe3_hero_mt_plan import assert_no_emo

    config = hero.model_config(common)
    assert_no_emo(config)
    config.recompute_each_block = True
    config.recompute_all_blocks_by_chunk = False
    assert not config.two_batch_overlap
    assert os.environ.get("OLMO_PROFILE_LB_COUNT_BATCHED", "0") == "0"
    return config


def config_builder():
    return partial(
        build_config,
        global_batch_size=BATCH,
        max_sequence_length=SEQUENCE_LENGTH,
        num_nodes=8,
        common_config_builder=common_components,
        data_config_builder=data_components,
        model_config_builder=model_config,
        train_module_config_builder=train_module_config,
        trainer_config_builder=trainer_config,
        beaker_image=hero.qualified.base.BEAKER_IMAGE,
        beaker_workspace=hero.WORKSPACE,
        include_default_evals=False,
        num_execution_units=1,
    )


def validate():
    """Validate both fresh starts and LC resumes without materializing model/data."""
    assert END == 5961 and END * BATCH == 100_008_984_576
    for start in (0, 2):
        os.environ["OLMO35_HERO_EXPECTED_START"] = str(start)
        for r in runs():
            os.environ["OLMO35_LC_LOAD"] = str(r.source if start == 0 else r.root / f"step{start}")
            c = config_builder()(CliContext(__file__, SubCmd.dry_run, r.run_id, "ai2/holmes", []))
            c.as_dict(json_safe=True)
            tm, tr = c.train_module, c.trainer
            assert (c.model.num_active_params, c.model.num_params) == (794_233_472, 12_496_341_632)
            assert (
                c.data_loader.global_batch_size == BATCH
                and tm.rank_microbatch_size == SEQUENCE_LENGTH
            )
            assert c.dataset.sequence_length == SEQUENCE_LENGTH
            assert c.dataset.source_group_size == 8 and c.dataset.source_permutation_seed == 123
            assert c.model.recompute_each_block and not c.model.recompute_all_blocks_by_chunk
            assert Path(c.data_loader.work_dir).is_relative_to(DATA_WORK)
            assert c.data_loader.seed == SEED
            assert tm.ep_config is None and tm.pp_config is None and tm.dp_config.use_reduce_scatter
            assert tm.ac_config is None and tm.float8_config is None
            assert tr.load_optim_state == tr.load_trainer_state == (start > 0)
            assert tm.reset_optimizer_states_on_load == (start == 0)
            assert not tr.save_overwrite and not tr.callbacks["checkpointer"].save_async
            assert tr.callbacks["checkpointer"].max_checkpoints is None
            assert tm.scheduler.get_lr(LR, WARMUP, END) == LR
            assert tm.scheduler.get_lr(LR, END, END) == 0.0
            print(
                "HERO_LC_CONFIG_VALIDATED", json.dumps(dict(start=start, **r.as_dict())), flush=True
            )


def data_probe():
    """Fail closed on missing cache products; never repack or write shared training data."""
    from olmoe3_hero_lc_cache import validate_cache

    validate_cache()


if __name__ == "__main__":
    os.environ.setdefault("OLMO35_HERO_EXPECTED_START", "0")
    os.environ.setdefault("OLMO35_HERO_STOP", str(END))
    os.environ["OLMO35_HERO_ALLOW_CONTINUATION"] = "1"
    hero.qualified.apply_policy()
    if sys.argv[1:] == ["--validate-only"]:
        validate()
    elif sys.argv[1:] == ["--data-probe"]:
        data_probe()
    else:
        main(config_builder=config_builder())
