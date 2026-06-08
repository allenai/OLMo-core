import argparse
import sys
from datetime import datetime
from functools import partial

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyPackedFSLDatasetConfig
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import (
    build_launch_config,
    get_beaker_username,
    get_root_dir,
    get_work_dir,
)
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    DataComponents,
    SubCmd,
    build_config,
)
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.nn.transformer.config import TransformerBlockConfig
from olmo_core.optim import LinearWithWarmup, SkipStepAdamWConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 32_768
GLOBAL_BATCH_SIZE = 64 * SEQUENCE_LENGTH
DEFAULT_NUM_NODES = 8
DATASET_PATH = (
    "/weka/oe-training-default/ai2-llm/jacobm/data/sft/rl-sft-32k/olmo-hybrid-sft-triple-tools"
)
# Default checkpoint to initialize from when ``--pretrain_checkpoint`` is not given.
DEFAULT_LOAD_PATH = "/weka/oe-training-default/ai2-llm/checkpoints/willm/linear-rnns/OLMo3.1-7B-6T-30h-long-context-drope/step23842"

# Remove heads to match params/TPS of OLMo3 7B transformer. This is to enable a
# fair comparison with OLMo3 7B. If training from scratch, we recommend setting the
# number of attention heads to 32 (or some power of 2 that makes sense for your model size).
REMOVE_HEADS = 2


def build_model_config(common: CommonComponents) -> TransformerConfig:
    config = TransformerConfig.olmo3_7B(
        vocab_size=common.tokenizer.padded_vocab_size(),
    )
    assert isinstance(config.block, TransformerBlockConfig)
    assert isinstance(config.block.sequence_mixer, AttentionConfig)

    # Remove heads (and scale down d_model) to compensate for extra params.
    config.d_model -= REMOVE_HEADS * 128
    num_heads = config.block.sequence_mixer.n_heads - REMOVE_HEADS
    config.block.sequence_mixer.n_heads = num_heads
    assert config.d_model / num_heads == 128

    attn_block = config.block

    # RoPE embeddings were disabled at the start of LC extension and so
    # they are disabled here as well.
    attn_block = attn_block.replace(
        sequence_mixer=attn_block.sequence_mixer.replace(rope=None),
    )

    gdn_block = attn_block.replace(
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=num_heads,
            head_dim=int(0.75 * config.d_model / num_heads),
            allow_neg_eigval=True,
        ),
    )

    # 3 GDN layers followed by 1 attention layer, repeating.
    config.block = {"gdn": gdn_block, "attn": attn_block}
    config.block_pattern = ["gdn", "gdn", "gdn", "attn"]
    assert config.n_layers % len(config.block_pattern) == 0

    # Save memory by using fused linear loss implementation.
    config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    return config


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=common.max_sequence_length,
        max_sequence_length=common.max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=2.5e-5,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            compile=False,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        cp_config=TransformerContextParallelConfig.ulysses(degree=2),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.1,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
        scheduler=LinearWithWarmup(
            warmup_fraction=0.03,
            alpha_f=0.0,
        ),
    )


def build_data_components(
    common: CommonComponents,
    dataset_path: str,
) -> DataComponents:
    clean_path = dataset_path.rstrip("/")
    dataset_config = NumpyPackedFSLDatasetConfig(
        tokenizer=common.tokenizer,
        work_dir=common.work_dir,
        paths=[f"{clean_path}/token_ids_part_*.npy"],
        expand_glob=True,
        label_mask_paths=[f"{clean_path}/labels_mask_*.npy"],
        generate_doc_lengths=True,
        long_doc_strategy=LongDocStrategy.truncate,
        sequence_length=common.max_sequence_length,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size, seed=34521, num_workers=4
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def build_trainer_config(
    common: CommonComponents, load_path: str = DEFAULT_LOAD_PATH
) -> TrainerConfig:
    cancel_check_interval = 10

    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    return (
        TrainerConfig(
            load_strategy=LoadStrategy.always,
            load_path=load_path,
            load_trainer_state=False,
            load_optim_state=False,  # Double check
            save_folder=f"{common.save_folder}/",
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.epochs(2),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=500,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                group=common.run_name,
                entity="ai2-llm",
                project=f"{get_beaker_username()}-Hybrid-7B-sft",
                enabled=False,
                cancel_check_interval=cancel_check_interval,
            ),
        )
    )


def build_common_with_relaunch(
    cli_context: CliContext,
    *,
    tokenizer,
    global_batch_size: int,
    max_sequence_length: int,
    beaker_image: str,
    num_nodes: int,
    beaker_workspace: str,
    flight_recorder: bool,
    num_execution_units,
    pretrain_checkpoint: str,
    budget: str,
    dataset_path: str,
) -> CommonComponents:
    """
    Like :func:`olmo_core.internal.experiment.build_common_components`, but builds the remote
    Beaker command so that the CLI args that are *not* config-overrides (``--seq_len``,
    ``--num_nodes``, ``--budget``, ``--workspace``, ``--dataset_path`` and the positional
    ``pretrain_checkpoint``) are re-emitted, ensuring the relaunched ``train`` job rebuilds the
    exact same config.
    """
    root_dir = get_root_dir(cli_context.cluster)
    beaker_user = get_beaker_username()

    launch_config = None
    if beaker_user is not None:
        post_cmd = cli_context.cmd.post_launch_subcmd()
        remote_cmd = [
            cli_context.script,
            str(post_cmd),
            cli_context.run_name,
            pretrain_checkpoint,
            cli_context.cluster,
            f"--seq_len={max_sequence_length}",
            f"--num_nodes={num_nodes}",
            f"--global_batch_size={global_batch_size}",
            f"--budget={budget}",
            f"--workspace={beaker_workspace}",
            f"--dataset_path={dataset_path}",
            *cli_context.overrides,
        ]
        launch_config = build_launch_config(
            name=f"{cli_context.run_name}-{post_cmd}",
            root_dir=root_dir,
            cmd=remote_cmd,
            cluster=cli_context.cluster,
            nccl_debug=True,
            flight_recorder=flight_recorder,
            beaker_image=beaker_image,
            num_nodes=num_nodes,
            workspace=beaker_workspace,
            budget=budget,
            num_execution_units=num_execution_units,
        )
        launch_config.launch_timeout = 5 * 60

    if beaker_user is not None:
        save_folder = f"{root_dir}/checkpoints/{beaker_user.lower()}/{cli_context.run_name}"
    else:
        save_folder = f"{root_dir}/checkpoints/{cli_context.run_name}"

    return CommonComponents(
        run_name=cli_context.run_name,
        root_dir=root_dir,
        work_dir=get_work_dir(root_dir),
        save_folder=save_folder,
        launch=launch_config,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        global_batch_size=global_batch_size,
    )


def parse_args() -> "tuple[argparse.Namespace, list]":
    parser = argparse.ArgumentParser(
        description="SFT the OLMo hybrid (GatedDeltaNet) 7B model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s dry_run test-run /path/to/base/ckpt ai2/jupiter --dataset_path /path/to/sft-data
  python %(prog)s launch hybrid-sft-run /path/to/base/ckpt ai2/jupiter \\
      --dataset_path /path/to/sft-data --seq_len 32768 --num_nodes 4 \\
      --budget ai2/oe-omai --workspace ai2/oe-science \\
      --train_module.optim.lr=1e-4 --launch.priority=urgent
""",
    )
    parser.add_argument("cmd", choices=[str(s) for s in SubCmd], help="Subcommand to run.")
    parser.add_argument("run_name", help="Run name for W&B and the checkpoint dir.")
    parser.add_argument(
        "pretrain_checkpoint",
        nargs="?",
        default=DEFAULT_LOAD_PATH,
        help="OLMo-core checkpoint to initialize from (defaults to the hybrid LC checkpoint).",
    )
    parser.add_argument("cluster", help="Beaker cluster (e.g. 'ai2/jupiter').")
    parser.add_argument("--seq_len", type=int, default=SEQUENCE_LENGTH, help="Max sequence length.")
    parser.add_argument("--num_nodes", type=int, default=DEFAULT_NUM_NODES, help="Number of nodes.")
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=None,
        help="Global batch size in tokens (defaults to 64 * seq_len).",
    )
    parser.add_argument("--budget", default="ai2/oe-other", help="Beaker budget.")
    parser.add_argument("--workspace", default="ai2/olmo-instruct", help="Beaker workspace.")
    parser.add_argument(
        "--dataset_path", default=DATASET_PATH, help="Path to the pre-tokenized SFT dataset."
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args, overrides = parse_args()
    global_batch_size = args.global_batch_size or (64 * args.seq_len)

    cmd = SubCmd(args.cmd)
    cli_context = CliContext(
        script=sys.argv[0],
        cmd=cmd,
        run_name=args.run_name,
        cluster=args.cluster,
        overrides=overrides,
    )

    config = build_config(
        cli_context,
        common_config_builder=partial(
            build_common_with_relaunch,
            pretrain_checkpoint=args.pretrain_checkpoint,
            budget=args.budget,
            dataset_path=args.dataset_path,
        ),
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=partial(build_trainer_config, load_path=args.pretrain_checkpoint),
        global_batch_size=global_batch_size,
        max_sequence_length=args.seq_len,
        num_nodes=args.num_nodes,
        beaker_workspace=args.workspace,
        num_execution_units=1,
        include_default_evals=False,
        dataset_path=args.dataset_path,
    )

    cmd.prepare_environment(config)
    cmd.run(config)
