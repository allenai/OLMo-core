import os
import sys
from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import ComposableDataLoaderConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import ConfigSaverCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))

from longmino_512k_mix import build_longmino_512k_mix  # noqa: E402

# Single-node smoke test for the Qwen3.5-4B 512k configs. Derived from
# Qwen3.5-4B-dense-longmino512k.py rather than the 190M base script, because the whole point is to
# exercise Qwen3.5-4B's hybrid GatedDeltaNet + full-attention stack -- a 190M dense model would test
# none of it.
#
# What this verifies, in order of how much rides on it:
#   1. Ulysses CP works at runtime with GatedDeltaNet. The existing Qwen3.5 scripts assert it does
#      not ("incompatible with the GDN recurrence") and fall back to shard_degree=8 with no CP, but
#      GatedDeltaNet.apply_cp has implemented Ulysses since it was added and has a parity test
#      (test_context_parallel_gdn_ulysses). The likely cause of that belief is cp_degree=8, copied
#      from the Qwen3-4B scripts, which raises here because Qwen3.5-4B has n_kv_heads=4.
#   2. flash_3 is usable on this image (it is marked beta in AttentionBackendName).
#   3. The longmino-512k mix loads and produces batches.
#   4. The Qwen3.5-4B olmo-core checkpoint loads under CP + FSDP sharding.
#
# It does NOT verify the 512k memory estimate (~45GB of 80): that depends on shard_degree=16 across
# 16 DP ranks, which only exists in the 8-node topology.
#
# Scaled down: 128k instead of 512k, 1 node instead of 8, cp=4 (the value under test) with dp=2 and
# shard_degree=2.
SEQUENCE_LENGTH = 131072  # 128k (divisible by CP_DEGREE)
CP_DEGREE = 4  # the value under test; must divide n_kv_heads=4
NUM_NODES = 1  # 8 GPUs -> DP = 8 / 4 = 2

GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 2  # 1 sequence per DP rank
MAX_STEPS = 20
LR = 3.2e-4

DATA_ROOT = "/weka/oe-training-default/amandab/longmino_512k"
CHECKPOINT_PATH = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/Qwen3.5-4B-olmocore/model_and_optim"
)


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/{cli_context.run_name}"

    beaker_launch_config: Optional[BeakerLaunchConfig] = build_launch_config(
        name=cli_context.run_name,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        root_dir=root_dir,
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=NUM_NODES,
    )
    if beaker_launch_config is not None:
        beaker_launch_config.priority = "urgent"

    tokenizer_config = TokenizerConfig.qwen3_5()

    model_config = TransformerConfig.qwen3_5_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_3,
    )
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup=10, alpha_f=0.0),
        compile_model=False,  # GatedDeltaNet custom kernels
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=2,  # shard across both DP ranks
        ),
        cp_config=TransformerContextParallelConfig.ulysses(degree=CP_DEGREE),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    instance_source_config = build_longmino_512k_mix(
        tokenizer=tokenizer_config,
        sequence_length=SEQUENCE_LENGTH,
        tree="qwen35",
        root=DATA_ROOT,
        seed=1234,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=2,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            load_path=CHECKPOINT_PATH,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            metrics_collect_interval=1,  # per-step metrics; this run is only 20 steps
            cancel_check_interval=1,
            max_duration=Duration.steps(MAX_STEPS),
            no_checkpoints=True,
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=cli_context.run_name,
                entity="ai2-llm",
                project="memory-networks",
                enabled=False,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=[instance_source_config],
        data_loader=data_loader_config,
    )
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


if __name__ == "__main__":
    """
    Single-node smoke test: Qwen3.5-4B at 128k with Ulysses cp=4, dp=2, shard_degree=2, flash_3.

        python src/scripts/train/smoketests/qwen35-cp4-flash3-test.py \\
            launch q35-cp4-smoke ai2/jupiter-cirrascale-2 \\
            --launch.follow=false --launch.step_soft_timeout=null
    """
    main(config_builder=build_experiment_config)
