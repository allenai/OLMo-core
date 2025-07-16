"""
This script can be used to launch an SFT run for the 7B model on Beaker.
Run the script without any arguments to see usage info. See the README for more details.
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, cast

import yaml
from rich import print

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyDatasetConfig, TokenizerConfig
from olmo_core.data.types import LongDocStrategy, NumpyDatasetType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_local_rank
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.common import (
    CLUSTER_TO_GPU_TYPE,
    build_launch_config,
    get_beaker_username,
    get_root_dir,
    get_work_dir,
)
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.attention import SlidingWindowAttentionConfig
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

DEFAULT_SEQUENCE_LENGTH = 16_384
DEFAULT_NUM_NODES = 1
GPUS_PER_NODE = 8
MAX_RANK_MICROBATCH_SIZE_TOKENS = 16_384  # max tokens this config can handle on an H100


@dataclass
class BatchSizeConfig:
    global_batch_size_tokens: int
    sequence_length: int
    dp_world_size: int
    gpu_type: str
    rank_microbatch_size_tokens: int = field(init=False)
    rank_microbatch_size_sequences: int = field(init=False)
    grad_accum_steps: int = field(init=False)
    cp_degree: Optional[int] = None

    def __post_init__(self):
        assert self.global_batch_size_tokens > 0, "global_batch_size_tokens must be positive"
        assert self.sequence_length > 0, "sequence_length must be positive"
        assert (self.sequence_length & (self.sequence_length - 1)) == 0, (
            "sequence_length must be a power of 2"
        )
        assert self.dp_world_size > 0, "dp_world_size must be positive"
        assert (self.dp_world_size & (self.dp_world_size - 1)) == 0, (
            "dp_world_size must be a power of 2"
        )

        # Determine max tokens per rank based on GPU type
        max_tokens_per_rank = MAX_RANK_MICROBATCH_SIZE_TOKENS
        if "B200" in self.gpu_type:
            max_tokens_per_rank *= 2

        # Check if we need context parallelism based on sequence length
        if self.sequence_length > max_tokens_per_rank:
            # Calculate minimum CP degree needed to fit sequence length
            min_cp_degree = 2
            while (self.sequence_length // min_cp_degree) > max_tokens_per_rank:
                min_cp_degree *= 2

            self.cp_degree = min_cp_degree
            log.info(
                f"Sequence length ({self.sequence_length} tokens) exceeds "
                f"max tokens per rank ({max_tokens_per_rank} tokens). Setting cp_degree={self.cp_degree}"
            )

        # Calculate rank batch size and grad accum steps
        cp_factor = self.cp_degree if self.cp_degree is not None else 1
        rank_batch_size_tokens = self.global_batch_size_tokens // (self.dp_world_size * cp_factor)

        # Ensure rank_batch_size_tokens doesn't exceed max_tokens_per_rank
        if rank_batch_size_tokens > max_tokens_per_rank:
            # Need gradient accumulation
            self.grad_accum_steps = 1
            while rank_batch_size_tokens // self.grad_accum_steps > max_tokens_per_rank:
                self.grad_accum_steps *= 2

            self.rank_microbatch_size_tokens = rank_batch_size_tokens // self.grad_accum_steps
            log.info(
                f"Rank batch size ({rank_batch_size_tokens} tokens) exceeds "
                f"max tokens per rank ({max_tokens_per_rank} tokens). "
                f"Using grad_accum_steps={self.grad_accum_steps}"
            )
        else:
            self.rank_microbatch_size_tokens = rank_batch_size_tokens
            self.grad_accum_steps = 1

        # Validate that rank_microbatch_size_tokens is divisible by sequence_length
        assert self.rank_microbatch_size_tokens % self.sequence_length == 0, (
            "rank_microbatch_size_tokens must be divisible by sequence_length (got "
            f"{self.rank_microbatch_size_tokens} and {self.sequence_length})"
        )
        self.rank_microbatch_size_sequences = (
            self.rank_microbatch_size_tokens // self.sequence_length
        )

        # Final validation
        total_tokens = (
            self.rank_microbatch_size_tokens
            * self.dp_world_size
            * cp_factor
            * self.grad_accum_steps
        )
        assert self.global_batch_size_tokens == total_tokens, (
            "global_batch_size_tokens must equal "
            "(rank_microbatch_size_tokens * dp_world_size * cp_degree * grad_accum_steps) (got "
            f"{self.global_batch_size_tokens} and {total_tokens})"
        )


def build_sft_dataset(
    dataset_name: str,
    root_dir: str,
    tokenizer_config: TokenizerConfig,
    sequence_length: int,
    dataset_path: Optional[str],
) -> NumpyDatasetConfig:
    if dataset_path is not None:
        print("globbing dataset paths")
        dataset_path_object = Path(dataset_path)
        token_id_paths = dataset_path_object.glob("token_ids_part_*.npy")
        label_mask_paths = dataset_path_object.glob("labels_mask_*.npy")
        expand_glob = True
    else:
        # NOTE: dataset path can be configured relative to root_dir
        # root_dir is /weka/oe-training-default/ai2-llm or gs://ai2-llm depending on the cluster
        sft_datasets_file = Path(__file__).parent / "sft_datasets.yaml"
        with open(sft_datasets_file, "r") as f:
            sft_datasets_config = yaml.safe_load(f)

        if dataset_name not in sft_datasets_config["datasets"]:
            raise OLMoConfigurationError(f"Dataset '{dataset_name}' not found in sft_datasets.yaml")

        dataset_config = sft_datasets_config["datasets"][dataset_name]
        dataset_dir = Path(dataset_config["base_dir"])

        token_id_paths, label_mask_paths = [], []
        for token_file in dataset_config["token_ids"]:
            token_path = dataset_dir / token_file
            token_id_paths.append(root_dir + "/" + str(token_path))
        for label_mask_file in dataset_config["label_mask"]:
            label_mask_path = dataset_dir / label_mask_file
            label_mask_paths.append(root_dir + "/" + str(label_mask_path))
        expand_glob = False

    dataset = NumpyDatasetConfig(
        # general config
        tokenizer=tokenizer_config,
        mix_base_dir=root_dir,
        work_dir=get_work_dir(root_dir),
        paths=token_id_paths,
        expand_glob=expand_glob,
        label_mask_paths=label_mask_paths,
        name=NumpyDatasetType.packed_fsl,  # concatenated short docs into a single sequence... (see also "padded_fsl")
        generate_doc_lengths=True,  # ...and mask attention so that they don't attend to each other
        long_doc_strategy=LongDocStrategy.truncate,  # truncate docs...
        sequence_length=sequence_length,  # ...that are over this length
    )
    return dataset


@dataclass
class SFTConfig(Config):
    """
    Custom config class for the sft run.

    Making config classes isn't strictly necessary for OLMo-core, but it gives us a nice way to
    capture all of the hyperparameters for a run and an easy way to override those options from
    the command line without configuring a complicated command line parser.
    """

    run_name: str

    launch: BeakerLaunchConfig
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 53184

    @classmethod
    def build(
        cls,
        *,
        script: str,
        cmd: str,
        run_name: str,
        dataset_name: str,
        seq_len: int,
        num_nodes: int,
        global_batch_size: int,
        checkpoint: str,
        cluster: str,
        overrides: List[str],
        workspace: str,
        budget: str,
        model_name: str,
        dataset_path: Optional[str],
    ) -> "SFTConfig":
        root_dir = get_root_dir(cluster)
        user_name = get_beaker_username()

        tokenizer_config = TokenizerConfig.dolma2()
        dataset_config = build_sft_dataset(dataset_name, root_dir, tokenizer_config, seq_len, dataset_path)
        gpu_type = CLUSTER_TO_GPU_TYPE[cluster]

        bs_config = BatchSizeConfig(
            sequence_length=seq_len,
            dp_world_size=num_nodes * GPUS_PER_NODE,  # every rank is data parallel
            global_batch_size_tokens=global_batch_size,
            gpu_type=gpu_type,  # used to double microbatch size for B200s
        )
        if get_local_rank() == 0:
            print("Batch size config (before overrides):")
            print(bs_config)

        dp_shard_degree = GPUS_PER_NODE // (bs_config.cp_degree or 1)
        if not dp_shard_degree > 0:
            raise OLMoConfigurationError(f"dp_shard_degree ({dp_shard_degree}) must be positive.")

        if model_name == "olmo2-7b":
            model = TransformerConfig.olmo2_7B(  # Based on https://github.com/allenai/OLMo-core/blob/dustins/anneal-repro/src/scripts/train/lc_cont_train/OLMo2-7B-lc_anneal_tp4.py
                vocab_size=tokenizer_config.padded_vocab_size(),
                use_flash=True,
                rope_theta=8 * 10**6,
            )
        elif model_name == "olmo3-7b":
            model = TransformerConfig.olmo2_7B(  # Based on https://github.com/allenai/OLMo-core/pull/310/files#diff-03f6a1f5db18fc4be7a243d8168698ae674cd50b2866253bcdadba5d48590b3dR48
                vocab_size=tokenizer_config.padded_vocab_size(),
                n_kv_heads=8,
                hidden_size_multiplier=1.2,
                hidden_size_multiple_of=1024,
            )
            model.block.attention.sliding_window = SlidingWindowAttentionConfig(
                force_full_attention_on_first_layer=False,
                force_full_attention_on_last_layer=True,
                pattern=[4096, 4096, 4096, -1],
            )
            model.block.attention.use_flash = True
            model.block.attention.use_head_qk_norm = True
        else:
            raise OLMoConfigurationError(f"Must set a valid model_name: {model_name}")

        print("overrides here:")
        print(overrides)

        config = SFTConfig(
            run_name=run_name,
            launch=build_launch_config(
                name=run_name,
                root_dir=root_dir,
                cmd=[
                    script,
                    cmd,
                    run_name,
                    dataset_name,
                    checkpoint,
                    cluster,
                    f"--seq_len={seq_len}",
                    f"--num_nodes={num_nodes}",
                    f"--global_batch_size={global_batch_size}",
                    f"--workspace={workspace}",
                    f"--model_name={model_name}",
                    f"--dataset_path={dataset_path}",
                    *overrides,
                ],
                cluster=cluster,
                num_nodes=num_nodes,
                budget=budget,
                workspace=workspace,
            ),
            model=model,
            dataset=dataset_config,
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
                    weight_decay=0.0,  # NOTE: different from pretraining
                    betas=(0.9, 0.95),
                    compile=False,
                ),
                dp_config=TransformerDataParallelConfig(
                    name=DataParallelType.hsdp,
                    param_dtype=DType.bfloat16,
                    reduce_dtype=DType.float32,
                    shard_degree=GPUS_PER_NODE  # try to keep communication w/in a node
                    // (bs_config.cp_degree or 1),
                ),
                cp_config=(
                    (
                        TransformerContextParallelConfig.llama3(degree=bs_config.cp_degree)
                        if dataset_config.generate_doc_lengths  # only use llama3 if we're masking docs
                        else TransformerContextParallelConfig.zig_zag(degree=bs_config.cp_degree)
                    )
                    if bs_config.cp_degree
                    else None
                ),
                ac_config=TransformerActivationCheckpointingConfig(
                    mode=TransformerActivationCheckpointingMode.selected_modules,
                    modules=["blocks.*.feed_forward"],
                ),
                scheduler=LinearWithWarmup(
                    warmup_fraction=0.03,
                    alpha_f=0.0,  # lr drops all the way to 0.0 at the end
                ),
                max_grad_norm=1.0,
            ),
            trainer=TrainerConfig(
                save_folder=f"{root_dir}/checkpoints/{user_name}/olmo2-7B-sft/{run_name}",
                load_strategy=LoadStrategy.never,  # we manually load the checkpoint below
                checkpointer=CheckpointerConfig(
                    save_thread_count=1, load_thread_count=32, throttle_uploads=True
                ),
                save_overwrite=True,
                metrics_collect_interval=10,
                cancel_check_interval=10,
                max_duration=Duration.epochs(3),
            )
            .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
            .with_callback("config_saver", ConfigSaverCallback())
            .with_callback("garbage_collector", GarbageCollectorCallback())
            .with_callback(
                "checkpointer",
                CheckpointerCallback(
                    save_interval=1000, ephemeral_save_interval=500, save_async=True
                ),
            )
            .with_callback(
                "wandb",
                WandBCallback(
                    name=run_name,
                    entity="ai2-llm",
                    project=f"{user_name}-7B-sft",
                    enabled=False,
                    cancel_check_interval=10,
                ),
            ),
        )
        print(config)
        quit()
        #.merge(overrides)

        return config


def train(checkpoint: str, config: SFTConfig):
    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    # Record the config to W&B/Comet and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Try loading a checkpoint from the save folder, otherwise start from the pretraining checkpoint.
    log.info("Loading checkpoint...")
    if not trainer.maybe_load_checkpoint(trainer.save_folder):
        log.info(
            f"No checkpoint found in save folder '{trainer.save_folder}', attempting to load from pretraining checkpoint '{checkpoint}'"
        )
        trainer.load_checkpoint(checkpoint, load_trainer_state=False)
    else:
        log.info(f"Loaded checkpoint from save folder '{trainer.save_folder}'")

    # Train.
    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SFT the 7B model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s dry_run test my-dataset-name /path/to/ckpt ai2/cluster
  python %(prog)s launch run01 OpenThoughts3-1.2M /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_final_anneal/step11921 ai2/jupiter-cirrascale-2 --seq_len=4096 --num_nodes=2 --launch.priority=high
""",
    )

    # Subcommand
    parser.add_argument(
        "cmd",
        choices=["launch", "train", "dry_run"],
        help="Subcommand to run",
    )

    # Positional arguments
    parser.add_argument(
        "run_name",
        help="The name of the run. Used for the run name in W&B/Comet and the checkpoint dir.",
    )
    parser.add_argument(
        "dataset_name", help="The name of the dataset to use. Must be defined in sft_datasets.yaml."
    )
    parser.add_argument("pretrain_checkpoint", help="Path to the pretraining checkpoint to load.")
    parser.add_argument(
        "cluster", help="The Beaker cluster to use (e.g., 'ai2/jupiter-cirrascale-2')."
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        help="The maximum sequence length to use.",
        default=DEFAULT_SEQUENCE_LENGTH,
    )
    parser.add_argument(
        "--num_nodes", type=int, help="The number of nodes to use.", default=DEFAULT_NUM_NODES
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        help="The global batch size in tokens.",
        default=64 * DEFAULT_SEQUENCE_LENGTH,
    )
    parser.add_argument(
        "--model_name", help="The name of the model architecture to use."
    )
    parser.add_argument(
        "--budget", help="The beaker budget to use."
    )
    parser.add_argument(
        "--workspace", help="The workspace to run in."
    )
    parser.add_argument(
        "--dataset_path", help="The path to the pre-tokenized SFT dataset."
    )

    # Parse known args to get positional arguments and cmd
    args, overrides = parser.parse_known_args()

    # Prepare the environment for the given command.
    if args.cmd in ("launch", "dry_run"):
        prepare_cli_environment()
    elif args.cmd == "train":
        prepare_training_environment()
    else:
        raise NotImplementedError(args.cmd)

    # Build the config, applying any overrides.
    config = SFTConfig.build(
        script=sys.argv[0],
        cmd="train",
        run_name=args.run_name,
        dataset_name=args.dataset_name,
        checkpoint=args.pretrain_checkpoint,
        cluster=args.cluster,
        seq_len=args.seq_len,
        num_nodes=args.num_nodes,
        global_batch_size=args.global_batch_size,
        overrides=overrides,
        budget=args.budget,
        workspace=args.workspace,
        model_name=args.model_name,
        dataset_path=args.dataset_path,
    )

    # Print the config for debugging and then execute the command.
    if get_local_rank() == 0:
        print(config)

    if args.cmd == "dry_run":
        pass
    elif args.cmd == "launch":
        config.launch.launch(follow=True)
    elif args.cmd == "train":
        try:
            train(args.pretrain_checkpoint, config)
        finally:
            teardown_training_environment()
    else:
        raise NotImplementedError(args.cmd)
