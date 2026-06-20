from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.rope import YaRNRoPEScalingConfig
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
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

# ---------------------------------------------------------------------------
# Unified task-suite SFT for Qwen3-4B DENSE baseline (full flash-attn 2 + YaRN), with packed,
# intra-document-masked sequences.
#
# The dense counterpart of Qwen3-4B-fast-landmark-unified-SFT.py. Data is the combined suite SFT
# mix from convert_unified_to_sft.py. Dense flash attention supports per-document masking via
# `cu_doc_lens` (unlike landmark), so we pack with ConcatAndChunk + generate_doc_lengths=True and
# keep context parallelism (CP=8). YaRN (factor 2) extends native 32k -> 64k. Eval is the oe-eval
# cr_* suite (same build_prompt), so train/eval inputs are byte-identical.
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH = 65536  # packed window length; YaRN extends Qwen3's native 32k to this.

DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/suite_it_sft_qwen/combined"


def resolve_dataset_path(run_name: str) -> str:
    """Single combined suite mix; a '-debug' run name uses the small debug shards."""
    if "debug" in run_name:
        return DATA_ROOT.rstrip("/") + "_debug"
    return DATA_ROOT


# Dense CPT checkpoint to initialize from (parallels the landmark variants' CPT init).
BASE_CHECKPOINT = "/weka/oe-training-default/ai2-llm/checkpoints/q4b-dense-dolma3longmino/step2385/model_and_optim"

NUM_NODES = 4

# Global batch in *tokens*. With CP=8 on 32 GPUs (4 DP replicas) and rank_microbatch=SEQUENCE_LENGTH,
# SEQUENCE_LENGTH * 32 ~ 2M tokens (matches the landmark variants for a fair comparison).
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 32

LR = 5e-5
NUM_EPOCHS = 1  # large multitask mix; raise if undertrained


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/prasanns/{cli_context.run_name}"

    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        root_dir=root_dir,
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=NUM_NODES,
    )
    if beaker_launch_config is not None:
        beaker_launch_config.priority = "urgent"

    tokenizer_config = TokenizerConfig.qwen3()
    # SFT shards are separated by single EOS tokens; qwen3 sets bos==eos, so drop bos for doc splitting.
    from dataclasses import replace

    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(factor=2, beta_fast=32, beta_slow=1, old_context_len=32768)
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=1,
        ),
        cp_config=TransformerContextParallelConfig.ulysses(degree=8),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.7,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # Packed dense SFT pipeline: ConcatAndChunk over the EOS-delimited shards, with per-document
    # masking derived from EOS (generate_doc_lengths=True -> cu_doc_lens for flash-attn varlen).
    clean_path = resolve_dataset_path(cli_context.run_name).rstrip("/")
    instance_source_config = ConcatAndChunkInstanceSourceConfig.from_npy(
        f"{clean_path}/token_ids_part_*.npy",
        tokenizer=doc_tokenizer_config,
        sequence_length=SEQUENCE_LENGTH,
        label_mask_paths=[f"{clean_path}/labels_mask_*.npy"],
        expand_glob=True,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
        generate_doc_lengths=True,  # block-diagonal masking at EOS doc boundaries (dense flash varlen)
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            load_path=BASE_CHECKPOINT,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.epochs(NUM_EPOCHS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=500,
                max_checkpoints=3,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=cli_context.run_name,
                entity="prasanns-allen-institute-for-ai",
                project="memory-networks",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "slack_notifier",
            SlackNotifierCallback(name=run_name_with_ts, enabled=False),
        )
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
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


if __name__ == "__main__":
    """
    Single-epoch packed unified-suite SFT of Qwen3-4B dense (flash-attn 2 + YaRN).

        python src/scripts/train/sft/Qwen3-4B-dense-unified-SFT.py \\
            launch q4b-dense-unified-sft ai2/jupiter-cirrascale-2
    """
    main(config_builder=build_experiment_config)
