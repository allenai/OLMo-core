"""
Example of how to train a transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/train_with_mixture.py run_name [OVERRIDES...]
"""

import os
import sys
from dataclasses import dataclass
from typing import List, cast, Union

import s3fs
import boto3

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyFSLDatasetMixtureConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.io import _get_s3_client
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.data.source_mixture import SourceMixtureConfig, SourceMixtureDatasetConfig
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType
from olmo_core.distributed.utils import init_hybrid_shard_mesh
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    LMEvaluatorCallbackConfig,
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
    dataset: Union[NumpyDatasetConfig, NumpyFSLDatasetMixtureConfig]
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def build_config(run_name: str, overrides: List[str]) -> ExperimentConfig:
    tokenizer_config = TokenizerConfig.gpt2()

    model_config = TransformerConfig.llama2_271M(
        vocab_size=tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
        compile=True,
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
    )

    optim_config = AdamWConfig(
        lr=1e-3,
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
    )

    s3 = s3fs.S3FileSystem()

    # DCLM docs + rewrites
    baseline = s3.glob(
        "s3://ai2-llm/preprocessed/dclm/samples/src-100b/**/allenai/dolma2-tokenizer/*.npy"
    )
    rewrites = s3.glob(
        "s3://ai2-llm/preprocessed/dclm/samples/rewrite-100b/**/allenai/dolma2-tokenizer/*.npy"
    )

    sequence_length = 1024
    source_config = SourceMixtureDatasetConfig(
        max_tokens=20_000_000,
        sequence_length=sequence_length,
        source_configs=[
            SourceMixtureConfig(
                paths=[f"s3://{path}" for path in baseline],
                source_name="baseline",
                max_repetition_ratio=1.0,
                target_ratio=0.8,
            ),
            SourceMixtureConfig(
                source_name="rewrites",
                paths=[f"s3://{path}" for path in rewrites],
                target_ratio=0.2,
            ),
        ],
        dtype=NumpyDatasetDType.uint32,
        seed=42,
    )

    dataset_config = NumpyFSLDatasetMixtureConfig(
        source_mixture_config=source_config,
        sequence_length=sequence_length,
        max_target_sequence_length=8192,
        tokenizer=TokenizerConfig.dolma2(),
        work_dir="/tmp/dataset-cache",
        bust_index_cache=True,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=256 * 1024,
        seed=0,
        num_workers=4,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"/tmp/{run_name}",
            rank_microbatch_size=16 * 1024,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
        )
        .with_callback("lr_scheduler", SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=100)))
        .with_callback(
            "seq_len_scheduler",
            SequenceLengthSchedulerCallback(
                min_sequence_length=128, warmup_steps=100, enabled=False
            ),
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
            "comet",
            CometCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,  # change to true to enable
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
        .with_callback(
            "evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyDatasetConfig(
                    paths=[
                        "s3://ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/c4_en/val/part-0-00000.npy"
                    ],
                    metadata=[{"label": "c4-validation"}],
                    name=NumpyDatasetType.padded_fsl,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir="/tmp/dataset-cache",
                ),
                eval_interval=250,
                eval_duration=Duration.steps(10),
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        optim=optim_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        trainer=trainer_config,
    ).merge(overrides)


def main(run_name: str, overrides: List[str]):
    config = build_config(run_name, overrides)

    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(
        init_device="meta",
        device=get_default_device(),
        dp_mesh=init_hybrid_shard_mesh(num_replicas=2),
    )
    optim = config.optim.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset)
    trainer = config.trainer.build(model, optim, data_loader)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train.
    trainer.fit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]

    prepare_training_environment()
    try:
        main(run_name, overrides=overrides)
    finally:
        teardown_training_environment()
