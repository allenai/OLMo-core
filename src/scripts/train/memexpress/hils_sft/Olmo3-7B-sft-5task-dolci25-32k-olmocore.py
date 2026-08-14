"""
BRIDGE ARM: Olmo-3-1025-7B SFT'd in **olmo_core**, on the identical data as the veomni arms.

**The contrast.** Same model, same data, same window as `olmo3-7b` in `run_sft_beaker.py` — only
the *trainer* differs (olmo_core here, veomni there). That makes the trainer's contribution
measurable instead of assumed away, and ties these rows to our existing Qwen3.5 SFT ladder, which
was produced by this stack.

**The readout.** The v2 ladder via `eval_lc_native.py` (`../hils_eval/`): contra/nq/rerank/outlier/
oolong at 2k–32k plus the four OOD ladders. Read against the veomni `olmo3-7b` arm first — a gap
there is trainer, not architecture — and only then against the `hils-7b` arm.

**Deliberately not matched** vs. the Qwen3.5 ladder: model family, vocabulary and base checkpoint
all differ, so those rows are context rather than a control.

**Same data, literally.** This reads the *materialized pack* that the veomni arms read — not the
per-task shards, and not the same recipe re-run. `ConcatAndChunkInstanceSource` at the pack's own
`sequence_length` recovers exactly the windows `sft_shard_dataset.materialize()` wrote, because
every shard length is an exact multiple of it. Pointing this at the per-task dirs with
`PackingInstanceSource` instead would re-mix and re-pack, and the arms would silently train on
different data while every config still looked identical.

Because the pack is fixed, token-matched and data-matched coincide: a step is a window is a
document multiset.

    PYTHONPATH=src python src/scripts/train/memexpress/hils_sft/Olmo3-7B-sft-5task-dolci25-32k-olmocore.py \\
        dry_run olmo3-7b-sft-5task-dolci25-32k-olmocore ai2/jupiter-cirrascale-2
    PYTHONPATH=src python src/scripts/train/memexpress/hils_sft/Olmo3-7B-sft-5task-dolci25-32k-olmocore.py \\
        launch  olmo3-7b-sft-5task-dolci25-32k-olmocore ai2/jupiter-cirrascale-2 \\
        --launch.follow=false --launch.step_soft_timeout=null
"""

from dataclasses import replace
from datetime import datetime

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode, TransformerConfig
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    SlackNotifierCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 32768  # must equal the pack's max_seq_len, or the windows are re-cut

#: The materialized pack written by sft_shard_dataset.py --out. The veomni arms read this same dir.
PACK_DIR = "/weka/oe-training-default/amandab/sft_olmo3/packed_32k"

#: olmo_core distcp conversion of allenai/Olmo-3-1025-7B, weights only. Produced by the ctc_suite
#: converter (convert_olmo3_base.sbatch) pointed at weka. MUST be the same weights the veomni arm
#: loads from HF, or this stops being a trainer comparison.
BASE_CHECKPOINT = "/weka/oe-training-default/amandab/olmo3-7b-base-olmocore/model_and_optim"

DP_DEGREE = 8
GLOBAL_BATCH_SIZE = DP_DEGREE * SEQUENCE_LENGTH  # tokens per optimizer step
#: From the pack manifest's `windows` count. Built 2026-08-14, Beaker 01M00TBDC1CB43P5FT9HZJAF8X:
#: 24,849 windows x 32768 = 814,252,032 window-token slots at 86.0% packing efficiency = 700M
#: content tokens, the requested budget. Realized shares matched targets to 3 decimals on all six
#: sources. None here refuses to launch rather than guess, which would rescale this arm against the
#: veomni arms it is controlled against.
PACK_WINDOWS: int | None = 24849


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    if PACK_WINDOWS is None:
        raise RuntimeError(
            "PACK_WINDOWS is None. Read `windows` from "
            f"{PACK_DIR}/pack_manifest.json and hardcode it here with the date it was built. "
            "Deriving the step budget from a guess would rescale this arm against the veomni "
            "arms, which train on exactly this many windows."
        )

    root_dir = get_root_dir(cli_context.cluster)
    tokenizer_config = TokenizerConfig.dolma2()  # OLMo-3 vocabulary, 100278

    max_steps = round(PACK_WINDOWS / DP_DEGREE)
    # content tokens = PACK_WINDOWS * SEQUENCE_LENGTH, by construction identical to the veomni arms

    beaker_launch_config = build_launch_config(
        name=cli_context.run_name,
        cmd=[
            "src/scripts/train/memexpress/hils_sft/Olmo3-7B-sft-5task-dolci25-32k-olmocore.py",
            "train",
            cli_context.run_name,
            cli_context.cluster,
        ],
        cluster=cli_context.cluster,
        root_dir=root_dir,
        task_name="train",
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=1,
        num_gpus=DP_DEGREE,
    )
    beaker_launch_config.priority = "urgent"

    model_config = TransformerConfig.olmo3_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        dtype=DType.bfloat16,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=1e-5,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            group_overrides=[OptimGroupOverride(name="embeddings", opts=dict(weight_decay=0.0))],
        ),
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        cp_config=TransformerContextParallelConfig(degree=DP_DEGREE),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full
        ),
        scheduler=LinearWithWarmup(warmup_steps=round(0.03 * max_steps)),
        max_grad_norm=1.0,
        z_loss_multiplier=None,
        attention_backend=AttentionBackendName.flash_2,
    )

    # ---- data: the SHARED pack, chunked at its own window length ----------------------------
    # ConcatAndChunk, not Packing: these windows are already packed. Re-packing here would produce
    # different windows from the veomni arms -- the one thing this arm must not do.
    #
    # bos_token_id=None mirrors the other SFT launchers: doc-boundary detection splits at an EOS
    # *followed by* a BOS, which never happens in EOS-separated SFT data. It is moot for a
    # pre-packed source (the windows are fixed) but kept so the loader's varlen masking behaves as
    # it does everywhere else in this repo.
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)
    instance_source_config = ConcatAndChunkInstanceSourceConfig.from_npy(
        f"{PACK_DIR}/token_ids_part_*.npy",
        tokenizer=doc_tokenizer_config,
        sequence_length=SEQUENCE_LENGTH,
        label_mask_paths=[f"{PACK_DIR}/labels_mask_*.npy"],
        expand_glob=True,
        label="olmo3-sft-pack-32k",
    )
    data_loader_config = ComposableDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,  # same seed as the veomni arms
        num_workers=4,
    )

    run_name_with_ts = f"{cli_context.run_name}-{datetime.now().strftime('%m%d%H%M')}"
    trainer_config = (
        TrainerConfig(
            # Fresh save folder per run: Trainer.fit() tries save_folder FIRST and only falls back
            # to load_path, so a reused name silently resumes the previous run's weights, optimizer
            # and dataloader position.
            save_folder=f"{root_dir}/checkpoints/amandab/{cli_context.run_name}",
            save_overwrite=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.steps(max_steps),
            load_path=BASE_CHECKPOINT,
            load_strategy=LoadStrategy.always,
            # Weights-only, strict: state_dict_load_opts={"strict": False} prunes EVERY missing key
            # and only warns, so a renamed weight would load at init and this arm would start from
            # a partially uninitialized model while its control started from the real one.
            load_optim_state=False,
            load_trainer_state=False,
            work_dir=get_work_dir(root_dir),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=100000, ephemeral_save_interval=max_steps, max_checkpoints=2,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=cli_context.run_name,
                entity="ai2-llm",
                project="memory-networks",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback("slack_notifier", SlackNotifierCallback(name=run_name_with_ts, enabled=False))
        .with_callback("config_saver", ConfigSaverCallback())
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=[instance_source_config],
        data_loader=data_loader_config,
    )
    return experiment_config.merge(cli_context.overrides)


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
