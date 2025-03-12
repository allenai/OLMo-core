"""
Train a 1B OLMo model. Run this script without any arguments to see usage info.
"""

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDatasetConfig,
    NumpyDatasetType,
    VSLCurriculumConfig,
    VSLCurriculumType,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import get_root_dir, get_work_dir
from olmo_core.internal.experiment import CommonComponents, main
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.common import Duration


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.starcoder2_3b(
        vocab_size=common.tokenizer.padded_vocab_size(),
        compile=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
    )


def build_optim_config(common: CommonComponents) -> AdamWConfig:
    del common
    return AdamWConfig(
        lr=7.0e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
        fused=True,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    # print("COMMON LAUNCH", common.launch, '\n\n', dir(common.launch), '\n\n', common.launch.clusters[0])
    root_dir = get_root_dir(common.launch.clusters[0])
    # MJ: Change the CommonComponents dataset
    dataset_config = NumpyDatasetConfig.from_data_mix(
        # DataMix.OLMoE_mix_0824,
        DataMix.OLMo2_subsample_4pct,
        tokenizer=common.tokenizer,
        mix_base_dir=root_dir,
        sequence_length=4096,
        max_target_sequence_length=8192,
        min_sequence_length=256,
        max_sequence_length=8192,
        vsl_curriculum=VSLCurriculumConfig(
            name=VSLCurriculumType.grow_p2, num_cycles=8, balanced=False
        ),
        work_dir=get_work_dir(root_dir),
    )

    lm_evaluator = LMEvaluatorCallbackConfig(
        eval_dataset=NumpyDatasetConfig.from_data_mix(
            DataMix.v3_small_ppl_validation,
            name=NumpyDatasetType.padded_fsl,
            mix_base_dir=root_dir,
            sequence_length=dataset_config.effective_sequence_length,
            tokenizer=common.tokenizer,
            work_dir=get_work_dir(root_dir),
        ),
        eval_interval=1000,
    )

    common.dataset = dataset_config
    common.callbacks["lm_evaluator"] = lm_evaluator

    # /MJ
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            rank_microbatch_size=4 * 4096,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=1,
            z_loss_multiplier=1e-5,
            compile_loss=True,
            max_duration=Duration.tokens(318_651_801_600),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=20,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=common.run_name,
                workspace="ai2",
                project="learn2code",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="learn2code",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
    )


if __name__ == "__main__":
    main(
        global_batch_size=512 * 4096,
        model_config_builder=build_model_config,
        optim_config_builder=build_optim_config,
        trainer_config_builder=build_trainer_config,
    )
