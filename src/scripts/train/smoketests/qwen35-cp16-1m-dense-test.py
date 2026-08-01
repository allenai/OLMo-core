import os
import sys
from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import ComposableDataLoaderConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import (
    BeakerEnvVar,
    BeakerLaunchConfig,
    OLMoCoreBeakerImage,
)
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))

from longmino_512k_mix import build_longmino_512k_mix  # noqa: E402

# 10-step smoke test: Qwen3.5-4B DENSE (hybrid GDN + full attention, 3:1) at 1M context, Ulysses
# cp=16. Paired with qwen35-cp16-1m-landmark-test.py.
#
# THIS IS A PLUMBING CHECK, NOT AN EXPERIMENT. It exists to prove that (a) CP=16 works now that the
# Ulysses all-to-all replicates KV heads, and (b) 1M actually fits on an 80GB H100. The loss curve
# means nothing: the mix has no documents above 512k and RoPE is unchanged from the native 262,144
# ceiling, so every 1M instance is concatenated text at positions the model was never trained on.
# Both of those are deliberately out of scope here.
#
# Why cp=16 specifically. The per-rank footprint is driven by T_local = T / cp; fitting the two
# measured anchors on this model (T_local=65,536 -> ~40GB, runs; T_local=131,072 -> 74.7GB, OOMs)
# gives peak(GB) ~= 5.5 + 0.528MB * T_local. At T=1M that is:
#     cp=4  -> T_local=262,144 -> ~144GB   (no)
#     cp=8  -> T_local=131,072 -> ~74.7GB  (this is the exact point that already OOMed)
#     cp=16 -> T_local=65,536  -> ~40GB    (same footprint as the 256k runs that work today)
# cp=16 used to be impossible: Ulysses required cp to divide n_kv_heads, and qwen3_5_4B has
# n_kv_heads=4. all_to_all_qkv_cp2hp now replicates the KV heads when it doesn't, so the only
# remaining requirement is that cp divides n_heads (=16). The architecture is unchanged.
#
# Parallelism is otherwise still architecture-forced: TP=1 (GatedDeltaNet.apply_tp raises), and
# Ulysses only (ring/zigzag CP is rejected by both GatedDeltaNet and landmark attention).
SEQUENCE_LENGTH = 1_048_576  # 1M
CP_DEGREE = 16  # 1048576 / 16 = 65536 tokens per rank
NUM_NODES = 2  # 2 x 8 = 16 GPUs -> DP = 16 / 16 = 1

# One sequence per step. DP=1, so this is the smallest batch reachable at 1M and the cheapest
# configuration that still exercises a CP group spanning two nodes -- which is the genuinely new
# thing here, since at cp<=8 the group is NVLink-local within one node.
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH
MAX_STEPS = 10
LR = 3.2e-4

DATA_ROOT = "/weka/oe-training-default/amandab/longmino_512k"

# The Qwen3.5-4B olmo-core base checkpoint. Override with --trainer.load_path=
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
        # Long-context runs fragment the allocator badly; at 512k this cost ~20% of the card before
        # expandable segments. Both spellings: torch 2.9 renamed this to PYTORCH_ALLOC_CONF and
        # warns on the old name, but the old one is what older images honour.
        for _var in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            beaker_launch_config.env_vars.append(
                BeakerEnvVar(name=_var, value="expandable_segments:True")
            )

    tokenizer_config = TokenizerConfig.qwen3_5()

    # flash_3 (Hopper FA3): at 1M, attention is ~91% of total FLOPs for this model even though only
    # 8 of 32 layers are full attention, so the attention kernel dominates wall-clock. Fall back with
    # --model.block.attn.sequence_mixer.backend=flash_2 if FA3 misbehaves (costs ~a third of
    # throughput).
    model_config = TransformerConfig.qwen3_5_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_3,
    )

    # Mandatory, not an optimization: with CP=16 each rank holds 65,536 tokens, and dense logits
    # would be 65536 x 248320 x 2B = ~32GB in bf16 alone (double that once cross-entropy upcasts to
    # fp32, plus an equal-sized grad). Liger's fused linear cross-entropy never materializes them.
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # one full sequence per DP rank, split across CP
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
        # GatedDeltaNet layers use custom kernels; compile stays off. This also rules out 'budget'
        # activation checkpointing, which is a no-op without compile.
        compile_model=False,
        # fsdp, not hsdp: DP=1 here, so there is no replica dimension to trade against. FSDP shards
        # params/grads/optim over the flattened dp_cp model mesh (CP is folded into DP for
        # parameter-synchronization purposes), i.e. across all 16 ranks.
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        cp_config=TransformerContextParallelConfig.ulysses(degree=CP_DEGREE),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
        float8_config=Float8Config(enabled=False),
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
            metrics_collect_interval=1,
            cancel_check_interval=1,
            max_duration=Duration.steps(MAX_STEPS),
            # Not no_checkpoints=True: that gates loading as well as saving (trainer.py:679), which
            # would silently start from random weights.
        )
        # Register a disabled checkpointer explicitly. Simply omitting one does NOT mean nothing is
        # saved -- the trainer auto-adds a default CheckpointerCallback (trainer.py:345), which
        # writes a full ~57GB checkpoint when the run ends. Useless for a 10-step smoke test.
        .with_callback("checkpointer", CheckpointerCallback(enabled=False))
        # The number this smoke test exists to produce: per-rank peak memory, to check the ~40GB
        # prediction for 1M at cp=16.
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
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
    Two-node smoke test: Qwen3.5-4B dense at 1M context, Ulysses cp=16, dp=1, 10 steps.

    Success means the run completes 10 steps with a finite loss and a reported peak near ~40GB.
    A peak far above that invalidates the memory model behind the 8-node projection.

        python src/scripts/train/smoketests/qwen35-cp16-1m-dense-test.py \\
            dry_run test-1m-dense ai2/jupiter-cirrascale-2

        python src/scripts/train/smoketests/qwen35-cp16-1m-dense-test.py \\
            launch q35-4b-dense-1m-smoke ai2/jupiter-cirrascale-2
    """
    main(config_builder=build_experiment_config)
