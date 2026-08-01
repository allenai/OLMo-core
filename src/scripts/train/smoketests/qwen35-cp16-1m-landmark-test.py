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

# 10-step smoke test: Qwen3.5-4B with FAST COMPRESSIVE LANDMARK attention replacing the
# full-attention layers, at 1M context, Ulysses cp=16. Paired with qwen35-cp16-1m-dense-test.py.
#
# THIS IS A PLUMBING CHECK, NOT AN EXPERIMENT. See the dense script's header for the memory math
# behind cp=16 and for why the loss curve is meaningless here (no documents above 512k in the mix,
# RoPE unchanged from the native 262,144 ceiling -- both deliberately out of scope).
#
# What this specifically checks beyond the dense arm: FastLandmarkAttention does its own
# cp2hp/hp2cp all-to-all inside forward() rather than going through a backend, so it needs the
# KV-replication path independently. It also asserted cp | n_kv_heads directly, which is now relaxed
# to cp | n_heads. FastCompressiveLandmarkAttention overrides neither forward nor apply_cp, so the
# landmark_fast.py path is what is under test here.
#
# Expect this to be SLOWER per token than the dense arm, not faster. Top-k landmark retrieval is an
# eval/decode-time feature only -- the fused kernel's training/prefill forward does full per-head
# soft gating over every block, so training pays the same O(T^2) attention as dense, plus landmark
# scoring (~T^2/64), plus gated-softmax bookkeeping. It also cannot use flash_3: the landmark path
# never touches self.backend.
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 1_048_576  # 1M model tokens; 1048576 / 64 = 16384 blocks, / 16 = 65536 per rank
CONTENT_SEQUENCE_LENGTH = SEQUENCE_LENGTH // BLOCK_SIZE * MEM_FREQ  # 1_032_192

LANDMARK_TOKEN_ID = 248200  # Qwen3.5 unused embedding row (vocab 248320)

CP_DEGREE = 16  # 1048576 / 16 = 65536 tokens per rank
NUM_NODES = 2  # 2 x 8 = 16 GPUs -> DP = 16 / 16 = 1

# Batch is counted in *model* tokens (landmarks included), matching the dense script's step size.
# Content per step is 63/64 of this.
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
        for _var in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            beaker_launch_config.env_vars.append(
                BeakerEnvVar(name=_var, value="expandable_segments:True")
            )

    tokenizer_config = TokenizerConfig.qwen3_5()

    # The backend below is inert for this model: the full-attention blocks are swapped to the
    # landmark mixer, which uses its own fused Triton kernel, and the GDN blocks never use a flash
    # backend either. Left at flash_2 to avoid implying an FA3 dependency that does not exist.
    model_config = TransformerConfig.qwen3_5_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )
    attn_mixer = model_config.block["attn"].sequence_mixer  # type: ignore[index]
    attn_mixer.name = AttentionType.fast_compressive_landmark
    attn_mixer.mem_freq = MEM_FREQ
    # Keep attn_mixer.gate (the elementwise gate from qwen3_5_4B): landmark attention applies it, so
    # gated attention is preserved and w_g loads straight from the converted checkpoint.

    # Mandatory (see the dense script): with CP=16 each rank holds 65,536 tokens and dense logits
    # would be ~32GB in bf16 alone.
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
        # GatedDeltaNet custom kernels; compile off, which also rules out 'budget' AC.
        compile_model=False,
        # fsdp, not hsdp: DP=1, so there is no replica dimension. FSDP shards over the flattened
        # dp_cp model mesh, i.e. across all 16 ranks.
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
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

    # Mix is built at the *content* length; landmark insertion then brings each instance up to
    # SEQUENCE_LENGTH:
    #   MixingInstanceSource (seq_len=CONTENT_SEQUENCE_LENGTH=1_032_192)
    #     -> LandmarkInstanceSource (one landmark every MEM_FREQ tokens -> 1_048_576)
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
    Two-node smoke test: Qwen3.5-4B + fast compressive landmark attention at 1M context,
    Ulysses cp=16, dp=1, 10 steps.

    Success means the run completes 10 steps with a finite loss and a reported peak near ~40GB.

        python src/scripts/train/smoketests/qwen35-cp16-1m-landmark-test.py \\
            dry_run test-1m-lm ai2/jupiter-cirrascale-2

        python src/scripts/train/smoketests/qwen35-cp16-1m-landmark-test.py \\
            launch q35-4b-fastcomplm-1m-smoke ai2/jupiter-cirrascale-2
    """
    main(config_builder=build_experiment_config)
