import os
import sys
from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LandmarkInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionBackendName, AttentionType
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

# Landmark counterpart to qwen35-cp4-flash3-test.py. Deliberately identical in every respect that
# affects throughput -- same 128k model sequence length, same cp=4/dp=2/shard_degree=2, same batch,
# same full AC, same 20 steps -- so the two runs' MFU is directly comparable and isolates the cost
# of the landmark mixer against the dense flash_3 baseline (which measured 40.9% MFU,
# 5,553 tokens/s/device).
#
# What this is expected to show: the landmark model should be SLOWER. Top-k retrieval is an
# eval/decode-time feature; the fused kernel's training forward does full per-head soft gating over
# every block, so it pays the same O(T^2) attention as dense plus landmark scoring (~T^2/64), and it
# cannot use flash_3 because it never touches self.backend.
#
# Note the token accounting differs slightly: 1 position in every 64 is a landmark token, so the
# same 131,072-token instance carries 129,024 content tokens. Compare MFU (FLOPs/s), not raw
# tokens/s, when reading the two runs against each other.
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 131072  # 128k model tokens (divisible by BLOCK_SIZE and by CP_DEGREE)
CONTENT_SEQUENCE_LENGTH = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 129024

LANDMARK_TOKEN_ID = 248200  # Qwen3.5 unused embedding row (vocab 248320)

CP_DEGREE = 4
NUM_NODES = 1  # 8 GPUs -> DP = 2

GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 2
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

    # Backend is inert here: the full-attention blocks become landmark blocks (own Triton kernel)
    # and the GDN blocks never use a flash backend.
    model_config = TransformerConfig.qwen3_5_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )
    attn_mixer = model_config.block["attn"].sequence_mixer  # type: ignore[index]
    attn_mixer.name = AttentionType.fast_compressive_landmark
    attn_mixer.mem_freq = MEM_FREQ
    # Keep the elementwise gate from qwen3_5_4B; landmark attention applies it and w_g loads from
    # the checkpoint.

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
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=2,
        ),
        cp_config=TransformerContextParallelConfig.ulysses(degree=CP_DEGREE),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    instance_source_config = LandmarkInstanceSourceConfig(
        source=build_longmino_512k_mix(
            tokenizer=tokenizer_config,
            sequence_length=CONTENT_SEQUENCE_LENGTH,
            tree="qwen35",
            root=DATA_ROOT,
            seed=1234,
        ),
        mem_freq=MEM_FREQ,
        mem_id=LANDMARK_TOKEN_ID,
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
            metrics_collect_interval=1,
            cancel_check_interval=1,
            max_duration=Duration.steps(MAX_STEPS),
            # No no_checkpoints=True: it gates loading as well as saving, which would silently
            # start from random weights. No CheckpointerCallback is registered, so nothing is saved.
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
    Single-node smoke test: Qwen3.5-4B + fast compressive landmark attention at 128k, Ulysses cp=4,
    dp=2, shard_degree=2. Matched to qwen35-cp4-flash3-test.py for MFU comparison.
    """
    main(config_builder=build_experiment_config)
