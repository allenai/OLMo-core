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
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import (
    BeakerEnvVar,
    BeakerLaunchConfig,
    OLMoCoreBeakerImage,
)
from olmo_core.nn.attention import AttentionBackendName, AttentionType
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
    SlackNotifierCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")
)

from longmino_512k_mix import build_longmino_512k_mix  # noqa: E402

# Qwen3.5-4B (hybrid Gated DeltaNet + full attention, 3:1) with FAST COMPRESSIVE LANDMARK attention
# replacing the full-attention layers, at 512k context on the custom longmino-512k mix, 10B tokens,
# 8 H100 nodes. Paired with Qwen3.5-4B-dense-longmino512k.py.
#
# Parallelism matches the dense script and is architecture-forced: TP=1 (GatedDeltaNet has no TP),
# Ulysses only (ring CP is rejected by both GDN and landmark attention), and cp_degree must divide
# n_kv_heads=4. So cp=4, tp=1, dp=16 on 64 GPUs.
#
# Expect this to be SLOWER per token than the dense baseline, not faster. Top-k landmark retrieval
# is an eval/decode-time feature only -- the fused kernel's training/prefill forward does full
# per-head soft gating over every block, so training pays the same O(T^2) attention as dense, plus
# landmark scoring (~T^2/64), plus the gated-softmax bookkeeping. It also cannot use flash_3: the
# landmark path never touches self.backend, running its own Triton kernel instead.
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 262144  # 256k (divisible by BLOCK_SIZE: 262144 / 64 = 4096)
CONTENT_SEQUENCE_LENGTH = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 258048

LANDMARK_TOKEN_ID = 248200  # Qwen3.5 unused embedding row (vocab 248320)

CP_DEGREE = 4  # max allowed: n_kv_heads=4
NUM_NODES = 8  # 8 x 8 = 64 GPUs -> DP = 64 / 4 = 16

# Batch is counted in *model* tokens (landmarks included), matching the dense script's step size.
# Content per step is 63/64 of this: ~8.26M tokens.
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 16  # ~4.2M tokens
MAX_TOKENS = 10_000_000_000  # 10B -> ~2384 steps
LR = 3.2e-4  # matches the 64k runs, whose ~4M batch this now also matches

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
        # At 512k the allocator fragments badly: the first attempt OOMed with 58.2 GiB actually
        # allocated but 15.5 GiB reserved-and-unusable, i.e. ~20% of the card lost to fragmentation.
        # Expandable segments lets the allocator grow existing segments instead of stranding them.
        # Both spellings: torch 2.9 renamed this to PYTORCH_ALLOC_CONF and warns on the old
        # name, but the old one is what older images honour.
        for _var in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            beaker_launch_config.env_vars.append(
                BeakerEnvVar(name=_var, value="expandable_segments:True")
            )

    tokenizer_config = TokenizerConfig.qwen3_5()

    # The backend below is inert for this model: the full-attention blocks are swapped to the
    # landmark mixer, which uses its own fused Triton kernel, and the GDN blocks never use a
    # flash backend either. Left at flash_2 to avoid implying an FA3 dependency that does not exist.
    model_config = TransformerConfig.qwen3_5_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )
    attn_mixer = model_config.block["attn"].sequence_mixer  # type: ignore[index]
    attn_mixer.name = AttentionType.fast_compressive_landmark
    attn_mixer.mem_freq = MEM_FREQ
    # Keep attn_mixer.gate (the elementwise gate from qwen3_5_4B): landmark attention applies it, so
    # gated attention is preserved and w_g loads straight from the converted checkpoint.

    # Mandatory at 512k (see the dense script): with CP=4 each rank holds 131,072 tokens and dense
    # logits would be ~65GB in bf16 alone.
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
        # ~2384 steps at this batch size, matching the 64k runs' warmup fraction.
        scheduler=LinearWithWarmup(warmup=400, alpha_f=0.0),
        # GatedDeltaNet custom kernels; compile off, which also rules out 'budget' AC.
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=16,  # shard across all 16 DP ranks -> ~4GB/rank of params+optim
        ),
        # Landmark attention performs its own cp2hp/hp2cp all-to-all inside forward(); it requires
        # Ulysses and rejects ring CP outright.
        cp_config=TransformerContextParallelConfig.ulysses(degree=CP_DEGREE),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    # Per-stratum longmino-512k mix (qwen35 tree) at the *content* length, then landmark insertion
    # brings each instance up to SEQUENCE_LENGTH:
    #   MixingInstanceSource (seq_len=CONTENT_SEQUENCE_LENGTH=516096)
    #     -> LandmarkInstanceSource (one landmark every MEM_FREQ tokens -> 524288)
    # The landmark-attention layers consume them positionally; the GDN layers see them as ordinary
    # tokens, and the label mask keeps them out of the loss.
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
        num_workers=4,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            load_path=CHECKPOINT_PATH,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(MAX_TOKENS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=250,  # ~2384 steps total
                ephemeral_save_interval=None,
                max_checkpoints=3,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=cli_context.run_name,
                entity="ai2-llm",
                project="memory-networks",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "slack_notifier",
            SlackNotifierCallback(name=run_name_with_ts, enabled=False),
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
    Qwen3.5-4B + fast compressive landmark attention at 512k on the longmino-512k mix, 10B tokens,
    8 nodes.

        python src/scripts/train/Qwen3/Qwen3.5-4B-fast-compressive-landmark-longmino512k.py \\
            dry_run test-512k-lm ai2/jupiter-cirrascale-2

        python src/scripts/train/Qwen3/Qwen3.5-4B-fast-compressive-landmark-longmino512k.py \\
            launch q35-4b-fastcomplm-512k ai2/jupiter-cirrascale-2
    """
    main(config_builder=build_experiment_config)
