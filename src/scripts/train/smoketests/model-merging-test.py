"""
Smoke test for ModelMergeCallback.

Trains for 25 steps with a 5-step merge window at step 20. Verifies that:
  - Weight accumulation runs over steps 16-20
  - A merged checkpoint is saved to step20-merged/
  - The LM evaluator runs on the merged weights (logged under eval/merged)
  - Original weights are restored and training continues to step 25

Usage:
    # Dry run:
    python src/scripts/train/smoketests/model-merging-test.py dry_run test-model-merging ai2/jupiter

    # Launch on Beaker:
    python src/scripts/train/smoketests/model-merging-test.py launch test-model-merging ai2/jupiter \
        --launch.priority=normal \
        --launch.follow=false
"""

from typing import Optional

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import build_launch_config, get_gpu_type, get_root_dir, get_work_dir
from olmo_core.internal.cookbook import configure_required_callbacks
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelWrappingStrategy
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    LMEvaluatorCallbackConfig,
    ModelMergeCallback,
    WandBCallback,
)
from olmo_core.train.callbacks.model_merger import compute_merge_window_starts
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

SEQ_LENGTH = 8192
MERGE_STEP = 20
MERGE_LAST_N_STEPS = 5


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    gpu_type = get_gpu_type(cli_context.cluster)
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/{cli_context.run_name}"

    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        root_dir=root_dir,
        workspace="ai2/OLMo_3",
        num_nodes=1,
        nccl_debug=False,
    )

    tokenizer_config = TokenizerConfig.dolma2()

    attn_backend = (
        AttentionBackendName.flash_2 if "B200" in gpu_type else AttentionBackendName.flash_3
    )
    model_config = TransformerConfig.olmo3_190M(
        vocab_size=tokenizer_config.padded_vocab_size(), attn_backend=attn_backend
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQ_LENGTH * 2,
        max_sequence_length=SEQ_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000),
    )

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0925,
        mix_base_dir="gs://ai2-llm",
        work_dir=work_dir,
        tokenizer=tokenizer_config,
        sequence_length=SEQ_LENGTH,
        max_target_sequence_length=SEQ_LENGTH,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=SEQ_LENGTH * 16, seed=34521, num_workers=4
    )

    merge_steps = [MERGE_STEP]
    window_starts = compute_merge_window_starts(merge_steps, MERGE_LAST_N_STEPS)

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(25),
        )
        .with_callbacks(configure_required_callbacks(cli_context.run_name))
        .with_callback(
            "model_merger",
            ModelMergeCallback(
                merge_step=merge_steps,
                merge_last_n_steps=MERGE_LAST_N_STEPS,
                enabled=True,
            ),
        )
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir="gs://ai2-llm",
                    sequence_length=SEQ_LENGTH,
                    tokenizer=tokenizer_config,
                    work_dir=work_dir,
                ),
                eval_interval=None,
                fixed_steps=window_starts + merge_steps,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=cli_context.run_name,
                group=cli_context.run_name,
                entity="ai2-llm",
                project="olmo3",
                enabled=False,
            ),
        )
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
    )
    return experiment_config.merge(cli_context.overrides)


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
