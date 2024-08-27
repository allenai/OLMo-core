"""
Train a 7B OLMo model. See below for usage.
"""

import json
import sys
from dataclasses import dataclass
from typing import List, Optional

from beaker import Beaker

from olmo_core.config import Config, DType, StrEnum
from olmo_core.data import MemMapDatasetConfig, TokenizerConfig
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType
from olmo_core.distributed.utils import get_num_nodes, get_rank, init_hybrid_shard_mesh
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerLaunchConfig,
    BeakerWekaBucket,
    OLMoCoreBeakerImage,
)
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    SchedulerCallback,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.utils import generate_uuid, get_default_device, prepare_cli_environment


class SubCmd(StrEnum):
    launch = "launch"
    train = "train"


@dataclass
class ExperimentConfig(Config):
    run_name: str
    launch: BeakerLaunchConfig
    model: TransformerConfig
    optim: AdamWConfig
    dataset: MemMapDatasetConfig
    trainer: TrainerConfig
    load_path: Optional[str] = None
    seed: int = 3423


def build_config(run_name: str, overrides: List[str]) -> ExperimentConfig:
    beaker_user = (Beaker.from_env().account.whoami().name).upper()

    launch_config = BeakerLaunchConfig(
        name=f"{run_name}-{generate_uuid()[:8]}",
        budget="ai2/oe-training",
        cmd=["src/scripts/train/OLMo-7B.py", SubCmd.train, run_name, *overrides],
        task_name="train",
        workspace="ai2/OLMo-core",
        description="Testing OLMo-core launch utilities",
        clusters=["ai2/jupiter-cirrascale-2"],
        weka_buckets=[BeakerWekaBucket("oe-training-default", "/weka/oe-training-default")],
        beaker_image=OLMoCoreBeakerImage.nightly,  # some features require nightly at the moment
        num_nodes=1,
        num_gpus=8,
        shared_filesystem=True,
        allow_dirty=False,
        env_secrets=[
            BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
            BeakerEnvSecret(name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"),
            BeakerEnvSecret(name="AWS_CONFIG", secret=f"{beaker_user}_AWS_CONFIG"),
            BeakerEnvSecret(name="AWS_CREDENTIALS", secret=f"{beaker_user}_AWS_CREDENTIALS"),
            BeakerEnvSecret(name="R2_ENDPOINT_URL", secret="R2_ENDPOINT_URL"),
            BeakerEnvSecret(name="WEKA_ENDPOINT_URL", secret="WEKA_ENDPOINT_URL"),
            BeakerEnvSecret(name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"),
        ],
        setup_steps=[
            # Setup python environment.
            "conda shell.bash activate base",
            "pip install -e '.[all]'",
            # In case you want to try a different version of PyTorch:
            #  "pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121",
            "pip freeze",
            # Move AWS credentials from env to relevant files
            "mkdir -p ~/.aws",
            "printenv AWS_CONFIG > ~/.aws/config",
            "printenv AWS_CREDENTIALS > ~/.aws/credentials",
        ],
    )

    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.llama2_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        compile=True,
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
    )

    optim_config = AdamWConfig(
        lr=3e-4,
        weight_decay=0.1,
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
    )

    dataset_config = MemMapDatasetConfig.glob(
        # Wikipedia
        "/weka/oe-training-default/ai2-llm/preprocessed/olmo-mix/danyh-compiled-v1_7/documents/wiki/allenai/dolma2-tokenizer/*.npy",
        sequence_length=4096,
        tokenizer=tokenizer_config,
    )

    save_folder = f"/weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/{beaker_user.lower()}/{run_name}"

    trainer_config = (
        TrainerConfig(
            work_dir=save_folder,
            save_folder=save_folder,
            global_batch_size=1024,
            microbatch_size=2,
            autocast_precision=DType.bfloat16,
            save_overwrite=True,
            data_seed=34521,
            data_loader_workers=4,
            metrics_collect_interval=10,
        )
        .with_callback(SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=200)))
        .with_callback(GPUMemoryMonitorCallback())
        .with_callback(GradClipperCallback(max_grad_norm=1.0))
        .with_callback(
            SpeedMonitorCallback(
                num_flops_per_token=model_config.num_flops_per_token(dataset_config.sequence_length)
            )
        )
    )

    experiment_config = ExperimentConfig(
        run_name=run_name,
        launch=launch_config,
        model=model_config,
        optim=optim_config,
        dataset=dataset_config,
        trainer=trainer_config,
    ).merge(overrides)

    experiment_config.trainer.with_callback(
        CheckpointerCallback(
            save_interval=10_000,
            ephemeral_save_interval=1024,
            save_async=True,
            pre_train_checkpoint=experiment_config.load_path is None,
        )
    )

    return experiment_config


def launch(config: ExperimentConfig):
    config.launch.launch(follow=True)


def train(config: ExperimentConfig):
    config_dict = config.as_config_dict()

    # Add W&B callback.
    config.trainer.with_callback(
        WandBCallback(
            name=config.run_name,
            config=config_dict,
            entity="ai2-llm",
            project="OLMo-core-testing",
            enabled=True,
        )
    )

    # Build components.
    model = config.model.build(
        init_device="meta",
        device=get_default_device(),
        max_seq_len=config.dataset.sequence_length,
        dp_mesh=None if get_num_nodes() == 1 else init_hybrid_shard_mesh(),
    )
    optim = config.optim.build(model)
    dataset = config.dataset.build()
    trainer = config.trainer.build(model, optim, dataset)

    # Save config to file.
    if get_rank() == 0:
        trainer.write_file("config.json", json.dumps(config_dict, indent=2))

    # Maybe load a checkpoint.
    if config.load_path is not None:
        trainer.load_checkpoint(config.load_path)

    # Train.
    trainer.fit()


if __name__ == "__main__":
    usage = f"Usage: python {sys.argv[0]} {SubCmd.launch}|{SubCmd.train} run_name [OVERRIDES...]"

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    cmd = sys.argv[1]
    run_name = sys.argv[2]
    overrides = sys.argv[3:]

    if sys.argv[1] == SubCmd.launch:
        prepare_cli_environment()
        config = build_config(run_name, overrides)
        launch(config)
    else:
        prepare_training_environment()
        config = build_config(run_name, overrides)
        try:
            train(config)
        finally:
            teardown_training_environment()
