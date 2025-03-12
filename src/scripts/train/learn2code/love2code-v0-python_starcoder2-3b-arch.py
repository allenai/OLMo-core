"""
Train a large OLMoE model. Run this script without any arguments to see usage info.
"""

import logging

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDatasetConfig,
    NumpyDatasetType,
    VSLCurriculumConfig,
    VSLCurriculumType,
)
from olmo_core.distributed.parallel import DataParallelType, PipelineScheduleType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import get_root_dir, get_work_dir
from olmo_core.internal.experiment import CommonComponents, main
from olmo_core.launch.beaker import OLMoCoreBeakerImage
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerExpertParallelConfig,
    TransformerPipelineParallelConfig,
    TransformerPipelineTrainModuleConfig,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)


PIPELINE_PARALLEL = True
DEFAULT_NUM_NODES = 2 if PIPELINE_PARALLEL else 2  # 2 nodes for now


def build_model_config(common: CommonComponents) -> TransformerConfig:
    d_model = 4096

    return TransformerConfig.starcoder2_3b(
        vocab_size=common.tokenizer.padded_vocab_size(),
    )


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    config_cls = (
        TransformerPipelineTrainModuleConfig if PIPELINE_PARALLEL else TransformerTrainModuleConfig
    )
    kwargs = {}
    if PIPELINE_PARALLEL:
        kwargs["pp_config"] = TransformerPipelineParallelConfig(
            degree=DEFAULT_NUM_NODES,
            schedule=PipelineScheduleType.interleaved_1F1B,
        )
    return config_cls(
        rank_microbatch_size=(4 if PIPELINE_PARALLEL else 2) * 4096,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=AdamWConfig(
            lr=7.0e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.bfloat16,
            num_replicas=1,  # to enable full-way expert parallel
            prefetch_factor=1,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        ep_config=TransformerExpertParallelConfig(degree=-1),
        float8_config=Float8Config(enabled=True),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000),
        **kwargs,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    root_dir = get_root_dir(common.launch.clusters[0])
    # MJ: Change the CommonComponents dataset
    dataset_config = NumpyDatasetConfig.from_data_mix(
        DataMix.love2code_v0_python,
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

    common.dataset = dataset_config

    return (
        TrainerConfig(
            save_folder=common.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=1,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1_000,
                ephemeral_save_interval=20,  # 20 for now to check leakiness
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
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
        # nightly needed right now for FP8 to work with PP
        # https://github.com/pytorch/pytorch/issues/143194
        beaker_image=OLMoCoreBeakerImage.nightly,
        num_nodes=DEFAULT_NUM_NODES,
    )
