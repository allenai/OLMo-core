"""
Train an OLMoE model. Run this script without any arguments to see usage info.
"""

import logging
import math
from dataclasses import replace

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.parallel.pipeline_parallel import PipelineScheduleType
from olmo_core.float8 import AOFloat8LinearConfig, Float8Config
from olmo_core.internal.experiment import CommonComponents, ExperimentConfig, main
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.moe import (
    MoEConfig,
    MoELoadBalancingLossGranularity,
    MoERouterConfig,
    MoERouterGatingFunction,
    MoEType,
)
from olmo_core.nn.transformer import (
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.optim import (
    WSD,
    AdamWConfig,
    OptimGroupOverride,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    BatchSizeSchedulerCallback,
    CheckpointerCallback,
    CometCallback,
    NvidiaProfilerCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerExpertParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer import TransformerPipelineParallelConfig

log = logging.getLogger(__name__)


SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE_SEQ = 256
GLOBAL_BATCH_SIZE = (GLOBAL_BATCH_SIZE_SEQ) * SEQUENCE_LENGTH
MAX_DURATION = int(1000e9)  # int(6e12), don't forget to adjust the LR when you increase this
EVAL_INTERVAL = 1000
LR = 1e-4

NUM_EXPERTS = 16
TOP_K = 6
D_MODEL = 2048
MOE_HIDDEN_SIZE = 1024
SHARED_MLP_HIDDEN_SIZE = (
    2048  # Hidden size for shared MLP (or dense branch MLP in arctic) in MoE blocks
)
HEAD_DIM = 64
NUM_HEAD = D_MODEL // HEAD_DIM
MICRO_BSZ = 4
NUM_LAYERS = 8
DP_DIM = 8
EP_DIM = 1
PP_DIM = 1
SPLIT_POINTS = None

TAG = f"U-fsdp-old-dbg"


def build_model_config(common: CommonComponents) -> TransformerConfig:
    d_model = D_MODEL

    config = TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=NUM_LAYERS,
        n_heads=NUM_HEAD,
        n_kv_heads=4,
        name=TransformerType.moe,
        block_name=TransformerBlockType.moe_reordered_norm,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
        # dropless
        feed_forward_moe=MoEConfig(
            name=MoEType.dropless,
            num_experts=NUM_EXPERTS,
            hidden_size=MOE_HIDDEN_SIZE,
            # capacity_factor=1.0,
            router=MoERouterConfig(
                top_k=TOP_K,
                gating_function=MoERouterGatingFunction.sigmoid,
                uniform_expert_assignment=True,
            ),
            shared_mlp=FeedForwardConfig(hidden_size=SHARED_MLP_HIDDEN_SIZE, bias=False),
            lb_loss_weight=0.005,
            z_loss_weight=None,
            lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
            scale_loss_by_num_layers=False,
        ),
        # feed_forward=FeedForwardConfig(hidden_size=SHARED_MLP_HIDDEN_SIZE, bias=False),
        init_std=0.01,
    )

    config.lm_head.loss_implementation = LMLossImplementation.fused_linear
    WINDOW_SIZE = 4095
    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE, -1],
    )
    config.block.attention.use_flash = True
    config.block.attention.use_head_qk_norm = True

    # First block will be a regular transformer block (no MoE component).
    config.block_overrides = {
        0: replace(
            config.block,
            name=TransformerBlockType.reordered_norm,
            feed_forward_moe=None,
            feed_forward=FeedForwardConfig(
                hidden_size=(TOP_K * MOE_HIDDEN_SIZE + SHARED_MLP_HIDDEN_SIZE) * 2, bias=False
            ),
        ),
    }

    return config


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=MICRO_BSZ * SEQUENCE_LENGTH,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            # fused=True,
            compile=False,
            foreach=True,
        ),
        compile_model=True,
        # FSDP
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            #  num_replicas=1,  # to enable full-way expert parallel
            shard_degree=DP_DIM,
            prefetch_factor=1,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        ep_config=TransformerExpertParallelConfig(degree=EP_DIM)
        if EP_DIM != 1
        else None,  # EP=1 means no expert parallel
        pp_config=TransformerPipelineParallelConfig(
            degree=PP_DIM,
            schedule=PipelineScheduleType.custom_1F1B,
            use_custom_stage_implementation=True,  # use custom stage implementation that re-uses receive buffers across micro-batches
            split_points=SPLIT_POINTS,
        )
        if PP_DIM > 1
        else None,
        # ac_config=TransformerActivationCheckpointingConfig(
        #     mode=TransformerActivationCheckpointingMode.full,
        #     # mode=TransformerActivationCheckpointingMode.selected_modules,
        #     # modules=["*norm*", "*mlp*"],
        # ),
        float8_config=Float8Config(
            ao=AOFloat8LinearConfig(
                enable_fsdp_float8_all_gather=True,
                force_recompute_fp8_weight_in_bwd=True,
                round_scales_to_power_of_2=True,
            ),
            enabled=False,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=WSD(
            units=SchedulerUnits.steps,
            warmup=500,
            # NOTE: be aware of when decay will happen relative to batch_wup schedule
            decay=(int(50e9 / GLOBAL_BATCH_SIZE)),
            decay_fraction=None,
        ),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10

    cluster = "ai2/augusta-google-1"

    return (
        TrainerConfig(
            save_folder=f"/workspace/tmp/{common.run_name}_{D_MODEL}d_{NUM_LAYERS}L{MOE_HIDDEN_SIZE}M{SHARED_MLP_HIDDEN_SIZE}S_{NUM_EXPERTS}E{TOP_K}K_{TAG}",
            # load_path='gs://ai2-llm/checkpoints/OLMo3-moe-integrationtest-5-32L/OLMo3-moe-integrationtest-5-32L_2048d_32L2560M2560S_64E4K_dev/step19000',
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(MAX_DURATION),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=200,
                save_async=True,
                pre_train_checkpoint=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="tianhuat-ai2",
                project="olmo-core-moe",
                # project="olmo3",
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            "batchwup",
            BatchSizeSchedulerCallback(
                batch_sizes=[
                    GLOBAL_BATCH_SIZE,
                    GLOBAL_BATCH_SIZE * 2,
                    GLOBAL_BATCH_SIZE * 4,
                    GLOBAL_BATCH_SIZE * 8,
                ],
                schedule=[
                    Duration.tokens(0),
                    Duration.tokens(167_772_160_000),
                    Duration.tokens(503_316_480_000),
                    Duration.tokens(838_860_800_000),
                ],
            ),
        )
        .with_callback(
            "profiler",
            NvidiaProfilerCallback(
                enabled=True, profile_ranks=[0, 8, 16, 24], start=30, end=33  # NOTE: change this
            ),
        )
        # TODO: might not be able to run in-loop evals depending on parallel strategies
        # .with_recommended_evals(
        #     common.tokenizer, SEQUENCE_LENGTH, cluster, task_set="fast", eval_interval=EVAL_INTERVAL
        # )
    )


# WORK_DIR = "/jfs/tianhua-tao/ws-olmoe"
WORK_DIR = "/weka/oe-training-default/tianhua/ws-megatron"


def finalize_config(config: ExperimentConfig):
    # config.dataset.mix = 'OLMo-mix-0625' # new dataset mix
    # config.dataset.mix_base_dir = "gs://ai2-llm" # only avail on Google Cloud
    # config.dataset.mix = "OLMoE-mix-0824-dev"
    # config.data_loader.num_workers = 1

    # DATA_ROOT = "/jfs/tianhua-tao/ws-olmoe/data"
    DATA_ROOT = "/weka/oe-training-default/ai2-llm"

    DATA_WORK_DIR = "/tmp/dataset-cache"
    # SAVE_ROOT = "/tmp/olmo-core/runs"  # NOTE: change this to what you want

    from olmo_core.data import (
        DataMix,
        NumpyDataLoaderConfig,
        NumpyDatasetConfig,
        NumpyDatasetType,
        TokenizerConfig,
    )

    dataset_config = NumpyDatasetConfig.from_data_mix(
        DataMix.OLMoE_mix_0824,
        tokenizer=TokenizerConfig.dolma2(),
        mix_base_dir=DATA_ROOT,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=max(8192, SEQUENCE_LENGTH),
        work_dir=f"{WORK_DIR}/{DATA_WORK_DIR}",
    )

    config.dataset = dataset_config
    config.dataset.mix = "OLMoE-mix-0824-dev"

    config.data_loader.num_workers = 1

    # add active & total params to the wandb name
    total_params_in_B = config.model.num_params / 1000 / 1000 / 1000
    active_params_in_B = config.model.num_active_params / 1000 / 1000 / 1000
    config.trainer.callbacks[
        "wandb"
    ].name += f"_{active_params_in_B:.2f}@{total_params_in_B:.2f}B"  # print to 2 decimal places
    config.trainer.callbacks[
        "wandb"
    ].name += f"_{TOP_K}K{NUM_EXPERTS}N_{TAG}"  # print to 2 decimal places


if __name__ == "__main__":
    main(
        global_batch_size=GLOBAL_BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_instance_filter=False,  # We use SkipStepOptimizer for this problem.
        include_default_evals=False,
        # intra_document_masking=True,
        finalize_config=finalize_config,
    )
