"""Bind corrected data to the qualified packed, eager SFT recipe at each run's size."""

import json
import os

import olmoe3_corrected_sft_plan as p

p.install()
import olmoe3_qkgain_train as adapter
from olmoe3_corrected_sft_data import tokenizer_check, tokenizer_config
from olmoe3_lr_sweep_watch import atomic_json


def install(r):
    """No numerical changes: fresh optimizer, EMO off, and new packing/data provenance."""
    adapter.SFT_DATA = r.data
    adapter.SFT_CACHE = r.data / "packing-cache"
    adapter.SFT_DATA_PLAN = r.data_plan
    original_common, original_trainer = adapter.common_components, adapter.trainer_config

    def common(context, **kwargs):
        value = original_common(context, **kwargs)
        value.tokenizer = tokenizer_config(r.data / "train/tokenizer")
        return value

    def trainer(common):
        value = original_trainer(common)
        value.callbacks["checkpointer"].pre_train_checkpoint = False
        value.metrics_collect_interval = 1
        value.callbacks["wandb"].tags += [r.milestone, r.lineage, r.dataset, "fixed-tokenizer"]
        return value

    adapter.common_components, adapter.trainer_config = common, trainer
    # Keep checkpoint pressure bounded, using the already agreed 5TB stop threshold.
    adapter.hero.disk_action = lambda free: (
        "stop" if free < 5_000_000_000_000 else ("warn" if free < 8_000_000_000_000 else "ok")
    )


def validate(r):
    """Fail before scheduling training if any source/config/tokenizer invariant drifts."""
    from olmo_core.internal.experiment import CliContext, SubCmd

    os.environ.update(QKGAIN_RUN=r.run_id, QKGAIN_START="0", QKGAIN_STOP=str(r.end))
    install(r)
    adapter.hero.qualified.apply_policy()
    c = adapter.builder(r)(CliContext(__file__, SubCmd.dry_run, r.run_id, "ai2/holmes", []))
    c.as_dict(json_safe=True)
    expected = 787359872 if r.arm == "3to1-shared" else 794233472
    assert c.model.num_active_params == expected
    if r.future_parent:
        assert r.arm == "3to1-shared" and not r.split and r.parent_ready()
        assert all(layer in c.model.block_overrides for layer in (3, 7, 11, 15))
    assert c.data_loader.global_batch_size == 8388608
    assert c.train_module.rank_microbatch_size == 65536
    assert r.batch % (r.gpus * r.microbatch) == 0
    assert c.train_module.ep_config is None and c.train_module.pp_config is None
    assert c.model.recompute_each_block and not c.model.two_batch_overlap
    assert c.train_module.optim.lr == r.lr and c.train_module.optim.weight_decay == 0
    assert not c.train_module.compile_model and c.train_module.reset_optimizer_states_on_load
    assert not c.trainer.load_optim_state and not c.trainer.load_trainer_state
    assert c.trainer.load_path == str(r.source)
    assert c.trainer.callbacks["checkpointer"].fixed_steps == r.saves
    assert not c.trainer.callbacks["checkpointer"].pre_train_checkpoint
    assert not c.trainer.callbacks["checkpointer"].save_async
    for block in [c.model.block, *c.model.block_overrides.values()]:
        router = getattr(block, "routed_experts_router", None)
        assert router is None or router.emo is None
        mixer = block.sequence_mixer
        assert not getattr(mixer, "use_cute_kernel", False)
        if hasattr(mixer, "qk_norm_per_head_gains"):
            assert mixer.qk_norm_per_head_gains == r.split
    tok = tokenizer_check(r.data / "train/tokenizer")
    from olmoe3_hero_sft_metadata import inference_template

    tok.chat_template = inference_template(tok.chat_template)
    assert tok.apply_chat_template(
        [{"role": "user", "content": "2+2?"}], tokenize=False, add_generation_prompt=True
    ).endswith("assistant\n<think>")
    p.base.validate_checkpoint(r.source, 5961, 16777216, r.source_gpus)
    atomic_json(
        p.AUTO / "config-proofs" / f"{r.run_id}.json",
        dict(passed=True, commit=os.environ["GIT_REF"], run=r.as_dict()),
    )
    print("CORRECTED_SFT_CONFIG_VERIFIED", json.dumps(r.as_dict()), flush=True)


def train(r):
    """Run the established SFT loop with actual first-batch and optimizer-reset audits."""
    install(r)
    adapter.hero.qualified.apply_policy()
    adapter.main(config_builder=adapter.builder(r))
