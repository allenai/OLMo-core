"""Compare real 64/128-GPU configs in the unchanged production runtime."""

import json
from dataclasses import replace

import olmoe3_small_hero as hero
from olmoe3_small_hero_runtime import verify_runtime

from olmo_core.internal.experiment import CliContext, SubCmd

verify_runtime()
configs = []
for gpus in (64, 128):
    hero.GPUS = gpus
    hero.qualified.SYSTEM = replace(hero.qualified.SYSTEM, num_nodes=gpus // 8)
    config = hero.config_builder()(
        CliContext(__file__, SubCmd.dry_run, "olmo35-small-hero-20260907-emo", "ai2/holmes", [])
    )
    config.as_dict(json_safe=True)
    assert config.data_loader.global_batch_size == 16777216
    assert config.train_module.rank_microbatch_size == 4 * 8192
    assert config.train_module.optim.lr == 1.1e-3
    assert config.trainer.load_optim_state and config.trainer.load_trainer_state
    assert config.trainer.max_duration.value == 834466
    configs.append(config)
for attribute in ("model", "train_module", "dataset", "data_loader"):
    assert getattr(configs[0], attribute).as_dict(json_safe=True) == getattr(
        configs[1], attribute
    ).as_dict(json_safe=True), attribute
before, after = [c.trainer.as_dict(json_safe=True) for c in configs]
old_tags = before["callbacks"]["wandb"].pop("tags")
new_tags = after["callbacks"]["wandb"].pop("tags")
assert set(new_tags) - set(old_tags) == {"small-128g", "ga4"}
assert set(old_tags) - set(new_tags) == {"small-64g", "ga8"}
assert before == after
assert configs[0].init_seed == configs[1].init_seed == 12536
print(
    "HERO_128_PREFLIGHT_SUCCESS",
    json.dumps(
        {
            "batch": 16777216,
            "microbatch_sequences": 4,
            "gradient_accumulation": 4,
            "model_optimizer_data_and_schedule_unchanged": True,
            "kernel_environment_unchanged": True,
        }
    ),
    flush=True,
)
