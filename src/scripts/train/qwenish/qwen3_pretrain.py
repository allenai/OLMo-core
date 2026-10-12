"""
Pretrain a Qwen3-shaped model (untied embeddings) from scratch on a Qwen3-tokenized slice of Dolma 3,
for 1x Chinchilla tokens (20 tokens/param).

Usage:
    python src/scripts/train/qwenish/qwen3_pretrain.py {dry_run|launch|train} SIZE RUN_NAME [OVERRIDES...]

where SIZE is one of: 0.6B, 1.7B. E.g.
    python src/scripts/train/qwenish/qwen3_pretrain.py launch 1.7B qwen3-1.7B-untied-dolma3-1xC --launch.min_runtime=30h
"""

import logging
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, cast

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.numpy_dataset import NumpyDatasetConfig, NumpyDatasetDType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import build_launch_config, get_root_dir
from olmo_core.launch.beaker import BeakerLaunchConfig, BeakerWekaBucket
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_cli_environment,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

#######################
#### CONFIGURATION ####
#######################

# Qwen3 base tokenizer: EOS = <|endoftext|> (151643), no BOS; pad falls back to EOS.
TOKENIZER_CONFIG = TokenizerConfig.from_hf("Qwen/Qwen3-1.7B-Base")
SEQUENCE_LENGTH = 4096
# Tokenized with `dolma tokens` (see burglar/pretrain_data); the glob is expanded on the cluster.
DATA_GLOB = "/weka/oe-adapt-default/victoriag/burglar/pretrain_data/tokenized/dolma3_150B_subset/Qwen3-1.7B-Base/*.npy"
SAVE_ROOT = "/weka/oe-adapt-default/victoriag/burglar/pretrain_ckpts"

# Per-size settings. Untied embeddings; tokens = 20 x total params (incl. untied lm_head).
# Micro-batches are small because Qwen3's ~152k vocab makes the logits large (8 x 4096 tokens of
# bf16 logits alone is 9.3 GiB per GPU, which OOMed on H100s).
SIZES: Dict[str, Dict] = {
    "0.6B": dict(
        model=TransformerConfig.qwen3_0_6B,
        total_params=0.752e9,
        global_batch_seqs=256,  # ~1M tokens/step
        rank_microbatch_seqs=4,
        lr=6e-4,
    ),
    "1.7B": dict(
        model=TransformerConfig.qwen3_1_7B,
        total_params=2.032e9,
        global_batch_seqs=512,  # ~2M tokens/step
        rank_microbatch_seqs=4,
        lr=4e-4,
    ),
}
WARMUP_FRACTION = 0.05

# Beaker.
BEAKER_CLUSTER = "ai2/jupiter"
BEAKER_WORKSPACE = "ai2/olmo-instruct"
BEAKER_BUDGET = "ai2/oe-other"
NUM_NODES = 2
WANDB_ENTITY = "ai2-llm"
WANDB_PROJECT = "qwenish-pretrain"

###########################
#### END CONFIGURATION ####
###########################


@dataclass
class ExperimentConfig(Config):
    launch: BeakerLaunchConfig
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def build_config(script: str, size: str, run_name: str, overrides: List[str]) -> ExperimentConfig:
    if size not in SIZES:
        raise ValueError(f"SIZE must be one of {list(SIZES)}, got {size!r}")
    s = SIZES[size]
    model_fn = cast(Callable[..., TransformerConfig], s["model"])
    model_config = model_fn(
        vocab_size=TOKENIZER_CONFIG.padded_vocab_size(), tie_word_embeddings=False
    )
    total_tokens = int(20 * s["total_params"])
    save_folder = f"{SAVE_ROOT}/{run_name}"

    dataset_config = NumpyFSLDatasetConfig(
        paths=[DATA_GLOB],
        expand_glob=True,
        dtype=NumpyDatasetDType.uint32,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=TOKENIZER_CONFIG,
        work_dir=f"{SAVE_ROOT}/dataset-work-dir",
    )
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=s["global_batch_seqs"] * SEQUENCE_LENGTH,
        seed=34521,
        num_workers=4,
    )
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=s["rank_microbatch_seqs"] * SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=s["lr"],
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_fraction=WARMUP_FRACTION),
    )
    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(total_tokens),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(save_interval=1000, ephemeral_save_interval=250, save_async=True),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name, entity=WANDB_ENTITY, project=WANDB_PROJECT, cancel_check_interval=10
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback("beaker", BeakerCallback())
    )

    launch_config = build_launch_config(
        name=run_name,
        root_dir=get_root_dir(BEAKER_CLUSTER),
        cmd=[script, "train", size, run_name, *overrides],
        cluster=BEAKER_CLUSTER,
        workspace=BEAKER_WORKSPACE,
        budget=BEAKER_BUDGET,
        num_nodes=NUM_NODES,
    )
    # Our tokenized data and checkpoints live on the oe-adapt-default bucket.
    launch_config.weka_buckets.append(
        BeakerWekaBucket("oe-adapt-default", "/weka/oe-adapt-default")
    )
    launch_config.priority = "urgent"

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        launch=launch_config,
    ).merge(overrides)


def train(config: ExperimentConfig):
    seed_all(config.init_seed)
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config.as_config_dict()
    # Resumes automatically from the latest checkpoint in `save_folder`, if any.
    trainer.fit()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    script, cmd, size, run_name, *overrides = sys.argv

    if cmd == "train":
        prepare_training_environment()
    else:
        prepare_cli_environment()

    config = build_config(script, size, run_name, overrides)
    log.info(config)

    if cmd == "train":
        train(config)
        teardown_training_environment()
    elif cmd == "launch":
        config.launch.launch(follow=False)
    elif cmd == "dry_run":
        pass
    else:
        print(__doc__)
        sys.exit(1)
