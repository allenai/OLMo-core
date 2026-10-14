"""Reuse the established LC and full-Dolci runtime with explicit stage handoffs."""

import json
import os
import sys

import olmoe3_14t_lc_sft_plan as p

p.install()


def adapter(r):
    """Install stage-specific settings without changing the trained architecture."""
    import olmoe3_qkgain_train as a
    from olmo_core.optim.scheduler import LinearWithWarmup

    if r.stage == "sft":
        import olmoe3_corrected_sft_train as s

        s.install(r)
    else:
        original_scheduler = a.scheduler
        a.scheduler = lambda value: (
            LinearWithWarmup(warmup=2000, alpha_f=0)
            if value.stage == "lc"
            else original_scheduler(value)
        )
        original_trainer = a.trainer_config

        def trainer(common):
            cfg = original_trainer(common)
            cfg.callbacks["checkpointer"].pre_train_checkpoint = False
            cfg.metrics_collect_interval = 1
            return cfg

        a.trainer_config = trainer
    a.hero.disk_action = lambda free: (
        "stop" if free < 5_000_000_000_000 else ("warn" if free < 8_000_000_000_000 else "ok")
    )
    return a


def validate(r, *, future=False):
    """Check the built config and native source before allocating the training group."""
    from olmo_core.internal.experiment import CliContext, SubCmd
    from olmoe3_lr_sweep_watch import atomic_json

    os.environ.update(QKGAIN_RUN=r.run_id, QKGAIN_START="0", QKGAIN_STOP=str(r.end))
    a = adapter(r)
    a.hero.qualified.apply_policy()
    cfg = a.builder(r)(CliContext(__file__, SubCmd.dry_run, r.run_id, "ai2/holmes", []))
    assert cfg.model.num_active_params == 794233472
    assert cfg.data_loader.global_batch_size == r.batch
    assert cfg.train_module.rank_microbatch_size == 65536
    assert cfg.train_module.max_sequence_length == 65536
    assert cfg.train_module.optim.lr == r.lr
    assert cfg.train_module.optim.weight_decay == (0.1 if r.stage == "lc" else 0)
    assert not cfg.trainer.load_optim_state and not cfg.trainer.load_trainer_state
    assert cfg.train_module.reset_optimizer_states_on_load
    assert cfg.trainer.load_path == str(r.source)
    assert cfg.trainer.max_duration.value == r.end
    assert cfg.trainer.callbacks["checkpointer"].fixed_steps == r.saves
    assert not cfg.trainer.callbacks["checkpointer"].pre_train_checkpoint
    assert cfg.model.recompute_each_block and not cfg.model.two_batch_overlap
    assert cfg.train_module.ep_config is None and cfg.train_module.pp_config is None
    assert cfg.train_module.scheduler.get_lr(r.lr, r.end, r.end) == 0
    for block in [cfg.model.block, *cfg.model.block_overrides.values()]:
        router = getattr(block, "routed_experts_router", None)
        assert router is None or router.emo is None
        mixer = block.sequence_mixer
        if hasattr(mixer, "qk_norm_per_head_gains"):
            assert mixer.qk_norm_per_head_gains
    if r.stage == "lc":
        assert cfg.train_module.scheduler.warmup == 2000
    else:
        from olmoe3_corrected_sft_data import tokenizer_check

        tokenizer_check(r.data / "train/tokenizer")
        data = json.loads(r.data_plan.read_text())
        assert data["passed"] and data["total_steps"] == r.end
        assert data["sequence_length"] == 65536 and data["batch_tokens"] == r.batch
        assert not cfg.train_module.compile_model
    if not future:
        p.base.validate_checkpoint(r.source, 5961, 16777216, 64)
    atomic_json(
        p.AUTO / "config-proofs" / (r.stage + ("-future" if future else "") + ".json"),
        dict(
            passed=True,
            future_source=future,
            run=r.as_dict(),
            commit=os.environ["GIT_REF"],
            config=cfg.as_dict(json_safe=True),
        ),
    )
    print("LC_SFT_CONFIG_VERIFIED", r.stage, "future" if future else "source-present", flush=True)


def main():
    """Dispatch training, persistent control, or frozen evaluation workers."""
    mode = sys.argv[1]
    if mode == "watch":
        from olmoe3_14t_lc_sft_control import watch

        return watch()
    r = p.find_run(sys.argv[2])
    if mode == "validate":
        return validate(r, future="--future" in sys.argv[3:])
    if mode == "node":
        from olmoe3_corrected_sft import node

        return node(r)
    if mode == "train":
        try:
            a = adapter(r)
            a.hero.qualified.apply_policy()
            return a.main(config_builder=a.builder(r))
        except (AssertionError, FloatingPointError, ValueError) as exc:
            from olmoe3_lr_sweep_watch import atomic_json

            atomic_json(
                r.root / "STOP.json",
                dict(
                    error=f"{type(exc).__name__}: {exc}",
                    rank=os.environ.get("RANK"),
                    reason="Numerical or configuration failure requires inspection before retry",
                ),
            )
            raise
    if mode == "convert" and r.stage == "sft":
        from olmoe3_corrected_sft_eval import convert

        return convert(r)
    if mode == "eval" and r.stage == "sft":
        from olmoe3_corrected_sft_eval import evaluate

        return evaluate(r, sys.argv[3], float(sys.argv[4]))
    if mode == "base-eval":
        if sys.argv[3] == "ruler":
            import olmoe3_hero_lc_ruler as rr

            rr.CAMPAIGN = p.CAMPAIGN + "-ruler"
            rr.model_path = lambda milestone, arm: r.hf
            source = sys.argv[sys.argv.index("--source") + 1]
            sys.argv = [
                rr.__file__,
                "--milestone",
                "lc100b",
                "--arm",
                "emo",
                "--helper",
                source,
                "--instances",
                "4",
            ]
            return rr.main()
        import olmoe3_qkgain_eval as e

        sys.argv = [e.__file__, *sys.argv[3:], "--run", r.run_id]
        return e.main()
    raise ValueError(mode)


if __name__ == "__main__":
    main()
