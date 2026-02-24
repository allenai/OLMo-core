import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import List

from rich import print

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_local_rank, get_world_size
from olmo_core.internal.common import (
    CLUSTER_TO_GPU_TYPE,
    build_launch_config,
    get_beaker_username,
    get_root_dir,
)
from olmo_core.launch.beaker import BeakerWekaBucket
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
from olmo_core.train.train_module.transformer.config import TransformerContextParallelConfig
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)

# Import shared utilities from the 32B SFT script (hyphenated filename requires importlib)
_spec = importlib.util.spec_from_file_location(
    "olmo3_32b_sft", Path(__file__).parent / "Olmo-3-32B-SFT.py"
)
assert _spec is not None and _spec.loader is not None
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)

SFTConfig = _base.SFTConfig
build_sft_dataset = _base.build_sft_dataset
train = _base.train

GPUS_PER_NODE = 8
DEFAULT_SEQUENCE_LENGTH = 16_384
DEFAULT_NUM_NODES = 1

MODEL_SIZES = ["60M", "100M", "600M", "760M", "1B", "3B"]

YARN_ROPE_SCALING = YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)

MODEL_BUILDERS = {
    "60M": TransformerConfig.olmo3_60M,
    "100M": TransformerConfig.olmo3_100M,
    "600M": TransformerConfig.olmo3_600M,
    "760M": TransformerConfig.olmo3_760M,
    "1B": TransformerConfig.olmo3_1B,
    "3B": TransformerConfig.olmo3_3B,
}

MAX_RANK_MICROBATCH_SIZE_TOKENS = {
    "60M": 131_072,
    "100M": 131_072,
    "600M": 131_072,
    "760M": 131_072,
    "1B": 131_072,
    "3B": 32_768,
}


def build_config(
    *,
    model_size: str,
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
    tokenizer_config = TokenizerConfig.dolma2()
    gpu_type = CLUSTER_TO_GPU_TYPE[cluster]

    dataset_config = build_sft_dataset(
        root_dir=root_dir,
        tokenizer_config=tokenizer_config,
        sequence_length=seq_len,
        dataset_path=dataset_path,
    )

    # BatchSizeConfig reads MAX_RANK_MICROBATCH_SIZE_TOKENS from the base module's globals
    _base.MAX_RANK_MICROBATCH_SIZE_TOKENS = MAX_RANK_MICROBATCH_SIZE_TOKENS[model_size]
    bs_config = _base.BatchSizeConfig(
        sequence_length=seq_len,
        world_size=num_nodes * GPUS_PER_NODE,
        global_batch_size_tokens=global_batch_size,
        gpu_type=gpu_type,
    )
    if get_local_rank() == 0:
        print("Batch size config (before overrides):")
        print(bs_config)

    # --- Model ---
    model = MODEL_BUILDERS[model_size](
        vocab_size=tokenizer_config.padded_vocab_size(),
    ).with_rope_scaling(YARN_ROPE_SCALING)

    # --- Activation checkpointing (only needed for 3B) ---
    if model_size == "3B":
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        )
    else:
        ac_config = None

    # --- Context parallelism ---
    if bs_config.cp_degree:
        cp_config = (
            TransformerContextParallelConfig.llama3(degree=bs_config.cp_degree)
            if dataset_config.generate_doc_lengths
            else TransformerContextParallelConfig.zig_zag(degree=bs_config.cp_degree)
        )
    else:
        cp_config = None

    # --- Data parallelism ---
    dp_shard_degree = min(
        GPUS_PER_NODE // (bs_config.cp_degree or 1),
        get_world_size(),
    )
    dp_config = TransformerDataParallelConfig(
        name=(
            DataParallelType.fsdp
            if dp_shard_degree == get_world_size()
            else DataParallelType.hsdp
        ),
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
        shard_degree=dp_shard_degree,
    )

    launch = build_launch_config(
        name=run_name,
        root_dir=root_dir,
        cmd=[
            script, cmd, run_name, checkpoint, cluster,
            f"--model_size={model_size}",
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
    )

    mounted_paths = {b.mount for b in launch.weka_buckets}
    if dataset_path.startswith("/weka/"):
        weka_root = "/".join(dataset_path.split("/")[:3])
        if weka_root not in mounted_paths:
            bucket_name = dataset_path.split("/")[2]
            launch.weka_buckets.append(BeakerWekaBucket(bucket_name, weka_root))

    config = SFTConfig(
        run_name=run_name,
        launch=launch,
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
            save_folder=f"{root_dir}/checkpoints/{user_name}/olmo-sft/{run_name}",
            load_strategy=LoadStrategy.never,
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
                project=f"{user_name}-{model_size}-sft",
                enabled=False,
                cancel_check_interval=10,
            ),
        ),
        init_seed=init_seed,
    ).merge(overrides)

    config.dataset = dataset_config
    print(config)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ladder SFT for OLMo-3 models (60M, 100M, 600M, 760M, 1B, 3B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s dry_run test /path/to/ckpt ai2/cluster --model_size=600M --dataset_path=/path/to/dataset
  python %(prog)s launch my-run /path/to/ckpt ai2/jupiter --model_size=1B --dataset_path=/path/to/dataset
""",
    )
    parser.add_argument("cmd", choices=["launch", "train", "dry_run"])
    parser.add_argument("run_name")
    parser.add_argument("pretrain_checkpoint")
    parser.add_argument("cluster")
    parser.add_argument("--model_size", required=True, choices=MODEL_SIZES)
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--num_nodes", type=int, default=DEFAULT_NUM_NODES)
    parser.add_argument("--no_save_tokenizer", action="store_true")
    parser.add_argument("--global_batch_size", type=int, default=64 * DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--budget")
    parser.add_argument("--workspace")
    parser.add_argument("--dataset_path", required=True)

    args, overrides = parser.parse_known_args()

    if args.cmd in ("launch", "dry_run"):
        prepare_cli_environment()
    elif args.cmd == "train":
        prepare_training_environment()
    else:
        raise NotImplementedError(args.cmd)

    config = build_config(
        model_size=args.model_size,
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
