"""
This script can be used to launch an SFT run for the 7B model on Beaker.
Run the script without any arguments to see usage info. See the README for more details.
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Optional, cast, Tuple

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
from olmo_core.io import list_directory
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig, TransformerBlockConfig
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
from pathlib import Path
import shutil
from urllib.parse import urlparse
import fnmatch


log = logging.getLogger(__name__)

DEFAULT_SEQUENCE_LENGTH = 16_384
DEFAULT_NUM_NODES = 1
GPUS_PER_NODE = 8
MAX_RANK_MICROBATCH_SIZE_TOKENS = 16_384  # max tokens this config can handle on an H100


@dataclass
class BatchSizeConfig:
    global_batch_size_tokens: int
    sequence_length: int
    world_size: int  # assumes all ranks are either data parallel or context parallel
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
        assert self.world_size > 0, "world_size must be positive"
        assert (self.world_size & (self.world_size - 1)) == 0, "world_size must be a power of 2"

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
        dp_world_size = self.world_size // cp_factor
        rank_batch_size_tokens = self.global_batch_size_tokens // dp_world_size

        # Ensure rank_batch_size_tokens doesn't exceed max_tokens_per_rank (adjusted by the cp_factor)
        if rank_batch_size_tokens > max_tokens_per_rank * cp_factor:
            # Need gradient accumulation
            self.grad_accum_steps = 1
            while rank_batch_size_tokens // self.grad_accum_steps > (
                max_tokens_per_rank * cp_factor
            ):
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
        total_tokens = self.rank_microbatch_size_tokens * dp_world_size * self.grad_accum_steps
        assert self.global_batch_size_tokens == total_tokens, (
            "global_batch_size_tokens must equal "
            "(rank_microbatch_size_tokens * dp_world_size * grad_accum_steps) (got "
            f"{self.global_batch_size_tokens} and {total_tokens})"
        )



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
    scheme, bucket, parsed_prefix = parsed_path.scheme, parsed_path.netloc, parsed_path.path.lstrip("/")
    parsed_prefix_pre_glob, glob_str = _separate_prefix_and_glob(parsed_prefix)
    base_prefix_without_scheme = Path(f"{bucket}/{parsed_prefix_pre_glob}")

    paths: List[str] = []

    for path in list_directory(f"{scheme}://{base_prefix_without_scheme}"):
        parsed_path = urlparse(path)
        path_without_scheme = Path(parsed_path.netloc) / parsed_path.path.lstrip('/')
        relative_to_base_prefix = path_without_scheme.relative_to(base_prefix_without_scheme)

        if glob_str and not fnmatch.fnmatch(str(relative_to_base_prefix), glob_str):
            # must match the glob if a glob was provided
            continue

        # path was valid!
        paths.append(path)

    return paths



def build_sft_dataset(
    root_dir: str,
    tokenizer_config: TokenizerConfig,
    sequence_length: int,
    dataset_path: Optional[str],
) -> NumpyDatasetConfig:
    clean_path = dataset_path.rstrip("/")
    if dataset_path.startswith("gs://") or dataset_path.startswith("s3://"):
        token_id_paths = glob_remote_dataset(f"{clean_path}/token_ids_part_*.npy")
        label_mask_paths = glob_remote_dataset(f"{clean_path}/token_ids_part_*.npy")
        expand_glob = False
    else:
        token_id_paths = [f"{clean_path}/token_ids_part_*.npy"]
        label_mask_paths = [f"{clean_path}/labels_mask_*.npy"]
        expand_glob = True

    dataset = NumpyDatasetConfig(
        # general config
        tokenizer=tokenizer_config,
        mix_base_dir=root_dir,
        work_dir=get_work_dir(root_dir),
        paths=token_id_paths,
        expand_glob=expand_glob,
        label_mask_paths=label_mask_paths,
        # name=NumpyDatasetType.padded_fsl,  # concatenated short docs into a single sequence... (see also "padded_fsl")
        # generate_doc_lengths=False,  # ...and mask attention so that they don't attend to each other
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
    dataset: Optional[NumpyDatasetConfig]
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
        elif model_name == "olmo2.5-7b":
            model = TransformerConfig.olmo2_7B(vocab_size=tokenizer_config.padded_vocab_size())
            model.block.attention.sliding_window = SlidingWindowAttentionConfig(
                force_full_attention_on_first_layer=False,
                force_full_attention_on_last_layer=True,
                pattern=[4096, 4096, 4096, -1],
            )
            model.block.attention.use_flash = True
        elif model_name == "olmo2.5-7b-yarn":
            # @soldni: copied from https://github.com/allenai/olmo-cookbook/blob/fe0d0ef5bbef7ad2caa2c91047d88062cf565df9/src/cookbook/model/config.py#L200-L206
            model = TransformerConfig.olmo2_7B(vocab_size=tokenizer_config.padded_vocab_size())
            model.block.attention.sliding_window = SlidingWindowAttentionConfig(
                force_full_attention_on_first_layer=False,
                force_full_attention_on_last_layer=True,
                pattern=[4096, 4096, 4096, -1],
            )
            model.block.attention.use_flash = True
            model.block.attention.rope.scaling = YaRNRoPEScalingConfig(
                factor=8, beta_fast=32, beta_slow=1, old_context_len=8192
            )
        elif model_name == "olmo2.5-7b-yarn-fullonly":
            # @soldni: copied from https://github.com/allenai/olmo-cookbook/blob/fe0d0ef5bbef7ad2caa2c91047d88062cf565df9/src/cookbook/model/config.py#L209-L224
            model = TransformerConfig.olmo2_7B(vocab_size=tokenizer_config.padded_vocab_size())
            model.block.attention.sliding_window = SlidingWindowAttentionConfig(
                force_full_attention_on_first_layer=False,
                force_full_attention_on_last_layer=True,
                pattern=[4096, 4096, 4096, -1],
            )
            model.block.attention.use_flash = True
            model.block.attention.rope.scaling = YaRNRoPEScalingConfig(
                factor=8, beta_fast=32, beta_slow=1, old_context_len=8192
            )
            def no_rope_scaling(block: TransformerBlockConfig) -> TransformerBlockConfig:
                rope_config = block.attention.rope
                if rope_config is not None:
                    rope_config.scaling = None
                    block.attention.rope = rope_config
                return block
            model.block_overrides = {
                i: no_rope_scaling(model.block.copy())
                for i in range(model.n_layers)
                if model.block.attention.sliding_window.should_use_swa(i, model.n_layers)
            }
        elif model_name == "olmo2.9-7b":
            model = TransformerConfig.olmo2_7B(
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
                    checkpoint,
                    cluster,
                    f"--seq_len={seq_len}",
                    f"--num_nodes={num_nodes}",
                    f"--global_batch_size={global_batch_size}",
                    f"--budget={budget}",
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
        ).merge(overrides)

        config.dataset = dataset_config

        print(config)

        return config


def train(checkpoint: str, config: SFTConfig, save_tokenizer: bool):
    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    if save_tokenizer and get_local_rank() == 0:
        tokenizer_path = Path(dataset.paths[0]).parent / "tokenizer"
        if tokenizer_path.exists() and tokenizer_path.is_dir():
            log.info("saving tokenizer...")
            destination_path = Path(trainer.save_folder) / "tokenizer"
            if destination_path.exists():
                log.info(f"Tokenizer already exists: {destination_path}")
            else:
                log.info(f"Saving tokenizer to {destination_path}")
                shutil.copytree(tokenizer_path, destination_path, dirs_exist_ok=True)


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
        "--follow", type=bool, help="Whether to follow the experiment in the terminal.", default=False
    )
    parser.add_argument(
        "--save_tokenizer", type=bool, help="Whether to save the dataset's tokenizer in the model directory.", default=True
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        help="The global batch size in tokens.",
        default=64 * DEFAULT_SEQUENCE_LENGTH,
    )
    parser.add_argument("--model_name", help="The name of the model architecture to use.")
    parser.add_argument("--budget", help="The beaker budget to use.")
    parser.add_argument("--workspace", help="The workspace to run in.")
    parser.add_argument("--dataset_path", help="The path to the pre-tokenized SFT dataset.")

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
        config.launch.launch(follow=args.follow)
    elif args.cmd == "train":
        try:
            train(args.pretrain_checkpoint, config, args.save_tokenizer)
        finally:
            teardown_training_environment()
    else:
        raise NotImplementedError(args.cmd)
