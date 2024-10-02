"""
Train a 1B Peteish model. Run this script without any arguments to see usage info.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, cast

from olmo_core.config import Config, DType, StrEnum
from olmo_core.data import NumpyDatasetConfig, NumpyDatasetType, TokenizerConfig
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType
from olmo_core.distributed.utils import init_hybrid_shard_mesh
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    ProfilerCallback,
    SchedulerCallback,
    SequenceLengthSchedulerCallback,
    WandBCallback,
)
from olmo_core.utils import get_default_device, seed_all


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    optim: AdamWConfig
    dataset: NumpyDatasetConfig
    trainer: TrainerConfig
    init_seed: int = 6198


class Platform(StrEnum):
    """
    An enumeration of supported platforms.
    """

    lumi = "lumi"
    beaker = "beaker"


def build_config(run_name: str, platform: Platform, overrides: List[str]) -> ExperimentConfig:
    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.olmo_1B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        norm_eps=1e-6,
        compile=True,
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
    )

    optim_config = AdamWConfig(
        lr=4e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        group_overrides=[OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))],
        fused=True,
    )

    if platform == Platform.lumi:
        data_base_path = os.environ["DATA_BASE_PATH"]
        save_base_path = os.environ["SCRATCH_DIR"]
        tmp_base_path = os.path.join(os.environ["SCRATCH_DIR"], "tmp")
    elif platform == Platform.beaker:
        raise NotImplementedError
    else:
        raise NotImplementedError

    dataset_config = NumpyDatasetConfig.glob(
        os.path.join(
            data_base_path,
            "ai2-llm/preprocessed/proof-pile-2/v0_decontaminated/algebraic-stack/train/allenai/dolma2-tokenizer/part-*.npy",
        ),
        os.path.join(
            data_base_path,
            "ai2-llm/preprocessed/proof-pile-2/v0_decontaminated/arxiv/train/allenai/dolma2-tokenizer/part-*.npy",
        ),
        os.path.join(
            data_base_path,
            "ai2-llm/preprocessed/proof-pile-2/v0_decontaminated/open-web-math/train/allenai/dolma2-tokenizer/part-*.npy",
        ),
        os.path.join(data_base_path, "ai2-llm/preprocessed/pes2o/allenai/dolma2-tokenizer/part-*.npy"),
        os.path.join(
            data_base_path,
            "ai2-llm/preprocessed/starcoder/v1-decon-100_to_20k-2star-top_token_030/allenai/dolma2-tokenizer/part-*.npy",
        ),
        os.path.join(
            data_base_path,
            "ai2-llm/preprocessed/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/part-*.npy",
        ),
        os.path.join(
            data_base_path,
            "ai2-llm/preprocessed/olmo-mix/danyh-compiled-v1_7/documents/wiki/allenai/dolma2-tokenizer/part-*.npy",
        ),
        name=NumpyDatasetType.fsl,
        sequence_length=4096,
        max_target_sequence_length=4096,
        #  name=NumpyDatasetType.vsl,
        #  max_sequence_length=2048,
        #  min_sequence_length=256,
        #  vsl_curriculum=VSLCurriculumConfig(name=VSLCurriculumType.grow_p2, num_cycles=4),
        tokenizer=tokenizer_config,
        work_dir=os.path.join(tmp_base_path, "dataset-cache"),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=os.path.join(save_base_path, run_name),
            global_batch_size=1024 * 4096,
            rank_microbatch_size=4 * 4096,
            save_overwrite=True,
            data_seed=6198,
            data_loader_workers=4,
            metrics_collect_interval=10,
            cancel_check_interval=1,
            z_loss_multiplier=1e-5,
        )
        .with_callback("lr_scheduler", SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=100)))
        .with_callback(
            "seq_len_scheduler",
            SequenceLengthSchedulerCallback(min_sequence_length=128, warmup_steps=100, enabled=False),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("grad_clipper", GradClipperCallback(max_grad_norm=1.0))
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=100,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,  # change to true to enable
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        # .with_callback(
        #     "evaluator",
        #     LMEvaluatorCallbackConfig(
        #         eval_dataset=NumpyDatasetConfig(
        #             paths=["/net/nfs/allennlp/llm-data/c4/en/c4-validation.00000-00008.npy"],
        #             metadata=[{"label": "c4-validation"}],
        #             name=NumpyDatasetType.padded_fsl,
        #             sequence_length=1024,
        #             tokenizer=tokenizer_config,
        #             work_dir="/tmp/dataset-cache",
        #         ),
        #         eval_interval=250,
        #         eval_duration=Duration.steps(10),
        #     ),
        # )
    )

    return ExperimentConfig(
        model=model_config, optim=optim_config, dataset=dataset_config, trainer=trainer_config
    ).merge(overrides)


def main(run_name: str, platform: Platform, overrides: List[str]):
    config = build_config(run_name, platform, overrides)

    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(
        init_device="meta",
        device=get_default_device(),
        dp_mesh=init_hybrid_shard_mesh(),
    )
    optim = config.optim.build(model)
    dataset = config.dataset.build()
    trainer = config.trainer.build(model, optim, dataset)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train.
    trainer.fit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name platform [OVERRIDES...]")
        sys.exit(1)

    run_name, platform, *overrides = sys.argv[1:]

    prepare_training_environment()
    try:
        main(run_name, Platform[platform], overrides=overrides)
    finally:
        teardown_training_environment()
