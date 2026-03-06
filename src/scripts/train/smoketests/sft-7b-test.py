"""
Smoke test for the 7B SFT pipeline.

Trains for 20 steps on a pre-tokenized SFT dataset to verify the pipeline
runs end-to-end without errors. Derived from Olmo-3-7B-SFT.py.

Usage:
    # First, tokenize your dataset (requires open-instruct; see README at
    # src/scripts/train/sft/README.md for full tokenization instructions).

    # Dry run to check the config:
    python src/scripts/train/smoketests/sft-7b-test.py \\
        dry_run smoke-test-sft /weka/oe-adapt-default/jacobm/olmo-core-checkpoints/<ckpt>/model_and_optim \\
        ai2/jupiter-cirrascale-2 \\
        --dataset_path /weka/oe-adapt-default/<user>/datasets/<dataset_name>

    # Launch on Beaker:
    python src/scripts/train/smoketests/sft-7b-test.py \\
        launch smoke-test-sft /weka/oe-adapt-default/jacobm/olmo-core-checkpoints/<ckpt>/model_and_optim \\
        ai2/jupiter-cirrascale-2 \\
        --dataset_path /weka/oe-adapt-default/<user>/datasets/<dataset_name> \\
        --budget ai2/oe-adapt \\
        --workspace ai2/olmo-instruct \\
        --launch.priority=normal \\
        --launch.follow=false
"""

import argparse
import fnmatch
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, cast
from urllib.parse import urlparse

from rich import print

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_local_rank, get_rank
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.common import (
    CLUSTER_TO_GPU_TYPE,
    build_launch_config,
    get_beaker_username,
    get_root_dir,
    get_work_dir,
)
from olmo_core.io import copy_dir, dir_is_empty, get_parent, join_path, list_directory
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.callbacks.wandb import WandBCallback
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.config import (
    TransformerContextParallelConfig,
)
from olmo_core.utils import prepare_cli_environment, seed_all

log = logging.getLogger(__name__)

DEFAULT_SEQUENCE_LENGTH = 32_768
DEFAULT_NUM_NODES = 1
GPUS_PER_NODE = 8
MAX_RANK_MICROBATCH_SIZE_TOKENS = 16_384  # max tokens this config can handle on an H100


@dataclass
class BatchSizeConfig:
    global_batch_size_tokens: int
    sequence_length: int
    world_size: int
    gpu_type: str
    rank_microbatch_size_tokens: int = field(init=False)
    rank_microbatch_size_sequences: int = field(init=False)
    grad_accum_steps: int = field(init=False)
    cp_degree: Optional[int] = None

    def __post_init__(self):
        assert self.global_batch_size_tokens > 0
        assert self.sequence_length > 0
        assert (self.sequence_length & (self.sequence_length - 1)) == 0
        assert self.world_size > 0
        assert (self.world_size & (self.world_size - 1)) == 0

        max_tokens_per_rank = MAX_RANK_MICROBATCH_SIZE_TOKENS
        if "B200" in self.gpu_type:
            max_tokens_per_rank *= 2

        if self.sequence_length > max_tokens_per_rank:
            min_cp_degree = 2
            while (self.sequence_length // min_cp_degree) > max_tokens_per_rank:
                min_cp_degree *= 2
            self.cp_degree = min_cp_degree

        cp_factor = self.cp_degree if self.cp_degree is not None else 1
        dp_world_size = self.world_size // cp_factor
        rank_batch_size_tokens = self.global_batch_size_tokens // dp_world_size

        if rank_batch_size_tokens > max_tokens_per_rank * cp_factor:
            self.grad_accum_steps = 1
            while rank_batch_size_tokens // self.grad_accum_steps > (
                max_tokens_per_rank * cp_factor
            ):
                self.grad_accum_steps *= 2
            self.rank_microbatch_size_tokens = rank_batch_size_tokens // self.grad_accum_steps
        else:
            self.rank_microbatch_size_tokens = rank_batch_size_tokens
            self.grad_accum_steps = 1

        assert self.rank_microbatch_size_tokens % self.sequence_length == 0
        self.rank_microbatch_size_sequences = (
            self.rank_microbatch_size_tokens // self.sequence_length
        )

        total_tokens = self.rank_microbatch_size_tokens * dp_world_size * self.grad_accum_steps
        assert self.global_batch_size_tokens == total_tokens


def _separate_prefix_and_glob(prefix: str) -> Tuple[str, str]:
    if any(char in prefix for char in ["*", "?", "[", "]"]):
        parts = prefix.split("/")
        base_parts = []
        for part in parts:
            if any(char in part for char in ["*", "?", "[", "]"]):
                break
            base_parts.append(part)
    else:
        base_parts = prefix.split("/")
    if not base_parts:
        return ".", prefix
    new_prefix = "/".join(base_parts)
    glob_str = prefix[len(new_prefix) :]
    return new_prefix, glob_str.lstrip("/")


def glob_remote_dataset(prefix: str) -> List[str]:
    parsed_path = urlparse(prefix)
    scheme, bucket, parsed_prefix = (
        parsed_path.scheme,
        parsed_path.netloc,
        parsed_path.path.lstrip("/"),
    )
    parsed_prefix_pre_glob, glob_str = _separate_prefix_and_glob(parsed_prefix)
    base_prefix_without_scheme = Path(f"{bucket}/{parsed_prefix_pre_glob}")

    paths: List[str] = []
    for path in list_directory(f"{scheme}://{base_prefix_without_scheme}"):
        parsed_path = urlparse(path)
        path_without_scheme = Path(parsed_path.netloc) / parsed_path.path.lstrip("/")
        relative_to_base_prefix = path_without_scheme.relative_to(base_prefix_without_scheme)
        if glob_str and not fnmatch.fnmatch(str(relative_to_base_prefix), glob_str):
            continue
        paths.append(path)
    return paths


def build_sft_dataset(
    root_dir: str,
    tokenizer_config: TokenizerConfig,
    sequence_length: int,
    dataset_path: str,
) -> NumpyPackedFSLDatasetConfig:
    clean_path = dataset_path.rstrip("/")
    return NumpyPackedFSLDatasetConfig(
        tokenizer=tokenizer_config,
        work_dir=get_work_dir(root_dir),
        paths=[f"{clean_path}/token_ids_part_*.npy"],
        expand_glob=True,
        label_mask_paths=[f"{clean_path}/labels_mask_*.npy"],
        generate_doc_lengths=True,
        long_doc_strategy=LongDocStrategy.truncate,
        sequence_length=sequence_length,
    )


@dataclass
class SFTConfig(Config):
    """
    Config for the 7B SFT smoke test.
    """

    run_name: str

    launch: BeakerLaunchConfig
    model: TransformerConfig
    dataset: Optional[NumpyPackedFSLDatasetConfig]
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int

    @classmethod
    def build(
        cls,
        *,
        script: str,
        cmd: str,
        run_name: str,
        seq_len: int,
        num_nodes: int,
        global_batch_size: int,
        checkpoint: str,
        cluster: str,
        overrides: List[str],
        workspace: str,
        budget: str,
        init_seed: int = 33333,
        dataset_path: str,
    ) -> "SFTConfig":
        root_dir = get_root_dir(cluster)
        user_name = get_beaker_username()

        tokenizer_config = TokenizerConfig.from_hf("allenai/olmo-3-tokenizer-instruct-dev")
        dataset_config = build_sft_dataset(
            root_dir=root_dir,
            tokenizer_config=tokenizer_config,
            sequence_length=seq_len,
            dataset_path=dataset_path,
        )
        gpu_type = CLUSTER_TO_GPU_TYPE[cluster]

        bs_config = BatchSizeConfig(
            sequence_length=seq_len,
            world_size=num_nodes * GPUS_PER_NODE,
            global_batch_size_tokens=global_batch_size,
            gpu_type=gpu_type,
        )
        if get_local_rank() == 0:
            print("Batch size config (before overrides):")
            print(bs_config)

        dp_shard_degree = GPUS_PER_NODE // (bs_config.cp_degree or 1)
        if not dp_shard_degree > 0:
            raise OLMoConfigurationError(f"dp_shard_degree ({dp_shard_degree}) must be positive.")

        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        )

        cp_config = (
            (
                TransformerContextParallelConfig.llama3(degree=bs_config.cp_degree)
                if dataset_config.generate_doc_lengths
                else TransformerContextParallelConfig.zig_zag(degree=bs_config.cp_degree)
            )
            if bs_config.cp_degree
            else None
        )

        dp_config = TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            shard_degree=GPUS_PER_NODE // (bs_config.cp_degree or 1),
        )

        model = TransformerConfig.olmo3_7B(
            vocab_size=tokenizer_config.padded_vocab_size(),
        ).with_rope_scaling(
            YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)
        )

        config = SFTConfig(
            run_name=run_name,
            launch=build_launch_config(
                name=run_name,
                root_dir=root_dir,
                cmd=[
                    script,
                    cmd,
                    run_name,
                    checkpoint,
                    cluster,
                    f"--seq_len={seq_len}",
                    f"--num_nodes={num_nodes}",
                    f"--global_batch_size={global_batch_size}",
                    f"--budget={budget}",
                    f"--workspace={workspace}",
                    f"--dataset_path={dataset_path}",
                    *overrides,
                ],
                cluster=cluster,
                num_nodes=num_nodes,
                budget=budget,
                workspace=workspace,
            ),
            model=model,
            dataset=None,
            data_loader=NumpyDataLoaderConfig(
                global_batch_size=bs_config.global_batch_size_tokens, seed=34521, num_workers=4
            ),
            train_module=TransformerTrainModuleConfig(
                rank_microbatch_size=bs_config.rank_microbatch_size_tokens,
                max_sequence_length=bs_config.sequence_length,
                z_loss_multiplier=None,
                compile_model=True,
                optim=SkipStepAdamWConfig(
                    lr=8e-05,
                    weight_decay=0.0,
                    betas=(0.9, 0.95),
                    compile=False,
                ),
                dp_config=dp_config,
                cp_config=cp_config,
                ac_config=ac_config,
                scheduler=LinearWithWarmup(
                    warmup_fraction=0.03,
                    alpha_f=0.0,
                ),
                max_grad_norm=1.0,
            ),
            trainer=TrainerConfig(
                save_folder=f"{root_dir}/checkpoints/{user_name}/olmo-sft-smoketest/{run_name}",
                load_strategy=LoadStrategy.never,
                checkpointer=CheckpointerConfig(
                    save_thread_count=1, load_thread_count=32, throttle_uploads=True
                ),
                save_overwrite=True,
                metrics_collect_interval=5,
                cancel_check_interval=5,
                # Smoke test: just 20 steps
                max_duration=Duration.steps(20),
            )
            .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
            .with_callback("config_saver", ConfigSaverCallback())
            .with_callback("garbage_collector", GarbageCollectorCallback())
            .with_callback(
                "checkpointer",
                CheckpointerCallback(
                    save_interval=20,  # save only at the end of the smoke test
                    save_async=True,
                ),
            )
            .with_callback(
                "wandb",
                WandBCallback(
                    name=run_name,
                    entity="ai2-llm",
                    project=f"{user_name}-7B-sft-smoketest",
                    enabled=False,
                    cancel_check_interval=10,
                ),
            ),
            init_seed=init_seed,
        ).merge(overrides)

        config.dataset = dataset_config

        print(config)

        return config


def train(checkpoint: str, config: SFTConfig, no_save_tokenizer: bool):
    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    if config.dataset is not None:
        dataset = config.dataset.build()
        data_loader = config.data_loader.build(
            dataset, dp_process_group=train_module.dp_process_group
        )
        trainer = config.trainer.build(train_module, data_loader)

        if not no_save_tokenizer and get_rank() == 0:
            tokenizer_path = join_path(get_parent(dataset.paths[0]), "tokenizer")
            if not dir_is_empty(tokenizer_path):
                log.info("Saving tokenizer...")
                destination_path = join_path(trainer.save_folder, "tokenizer")
                if not dir_is_empty(destination_path):
                    log.info(f"Tokenizer already exists: {destination_path}")
                else:
                    log.info(f"Saving tokenizer to {destination_path}")
                    copy_dir(tokenizer_path, destination_path)

        config_dict = config.as_config_dict()
        cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
        cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

        log.info("Loading checkpoint...")
        if not trainer.maybe_load_checkpoint(trainer.save_folder):
            log.info(
                f"No checkpoint found in save folder '{trainer.save_folder}', "
                f"loading from pretrain checkpoint '{checkpoint}'"
            )
            trainer.load_checkpoint(checkpoint, load_trainer_state=False)
        else:
            log.info(f"Loaded checkpoint from save folder '{trainer.save_folder}'")

        trainer.fit()
    else:
        log.error(f"Config dataset is None: {config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SFT smoke test for the 7B model (20 steps).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dolci baseline:
  python %(prog)s launch smoke-dolci /weka/oe-adapt-default/jacobm/olmo-core-checkpoints/<ckpt>/model_and_optim \\
      ai2/jupiter-cirrascale-2 \\
      --dataset_path /weka/oe-adapt-default/<user>/datasets/dolci-smoke \\
      --budget ai2/oe-adapt --workspace ai2/olmo-instruct \\
      --launch.priority=normal --launch.follow=false

  # Dolci + Dr Tulu:
  python %(prog)s launch smoke-dolci-drtulu /weka/oe-adapt-default/jacobm/olmo-core-checkpoints/<ckpt>/model_and_optim \\
      ai2/jupiter-cirrascale-2 \\
      --dataset_path /weka/oe-adapt-default/<user>/datasets/dolci-drtulu-smoke \\
      --budget ai2/oe-adapt --workspace ai2/olmo-instruct \\
      --launch.priority=normal --launch.follow=false
""",
    )

    parser.add_argument(
        "cmd",
        choices=["launch", "train", "dry_run"],
        help="Subcommand to run",
    )
    parser.add_argument("run_name", help="Name for this run (used in checkpoint paths and W&B).")
    parser.add_argument("pretrain_checkpoint", help="Path to the pretraining checkpoint to load.")
    parser.add_argument("cluster", help="The Beaker cluster (e.g., 'ai2/jupiter-cirrascale-2').")
    parser.add_argument(
        "--seq_len",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Max sequence length (should match the tokenization seq_len).",
    )
    parser.add_argument(
        "--num_nodes", type=int, default=DEFAULT_NUM_NODES, help="Number of nodes."
    )
    parser.add_argument(
        "--no_save_tokenizer",
        action="store_true",
        help="Disable saving the dataset's tokenizer in the model directory.",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=8 * DEFAULT_SEQUENCE_LENGTH,  # smaller than production for quick smoke test
        help="Global batch size in tokens.",
    )
    parser.add_argument("--budget", help="Beaker budget.")
    parser.add_argument("--workspace", help="Beaker workspace.")
    parser.add_argument("--dataset_path", required=True, help="Path to the pre-tokenized SFT dataset.")

    args, overrides = parser.parse_known_args()

    if args.cmd in ("launch", "dry_run"):
        prepare_cli_environment()
    elif args.cmd == "train":
        prepare_training_environment()
    else:
        raise NotImplementedError(args.cmd)

    config = SFTConfig.build(
        script=sys.argv[0],
        cmd="train",
        run_name=args.run_name,
        checkpoint=args.pretrain_checkpoint,
        cluster=args.cluster,
        seq_len=args.seq_len,
        num_nodes=args.num_nodes,
        global_batch_size=args.global_batch_size,
        overrides=overrides,
        budget=args.budget,
        workspace=args.workspace,
        dataset_path=args.dataset_path,
    )

    if get_local_rank() == 0:
        print(config)

    if args.cmd == "dry_run":
        pass
    elif args.cmd == "launch":
        config.launch.launch()
    elif args.cmd == "train":
        try:
            train(args.pretrain_checkpoint, config, args.no_save_tokenizer)
        finally:
            teardown_training_environment()
    else:
        raise NotImplementedError(args.cmd)
