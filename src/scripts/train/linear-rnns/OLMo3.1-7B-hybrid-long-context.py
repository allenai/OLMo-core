import logging
from datetime import datetime
from functools import partial

from olmo_core.config import DType
from olmo_core.data import (
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import (
    CommonComponents,
    DataComponents,
    build_config,
    main,
)
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.fla.layer import FLAConfig
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode, TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockType
from olmo_core.optim import (
    LinearWithWarmup,
    OptimGroupOverride,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)

SEQUENCE_LENGTH = 65536
GLOBAL_BATCH_SIZE = 4 * 1024 * 1024  # ~4M tokens
MAX_TOKENS = 100_000_000_000  # 100B
LR = 0.00020712352850360292


# Remove heads to match params/TPS of transformer.
REMOVE_HEADS = 2

### OLMo "3.1" 7B Settings (from OLMo 3 32B)
HARD_STOP = None
INSTANCE_FILTER = True

# TODO: does GDN support intra-document masking?


def build_model_config(common: CommonComponents) -> TransformerConfig:
    config = TransformerConfig.olmo3_7B(
        vocab_size=common.tokenizer.padded_vocab_size(),
        # See README for how to override with flash_3 using CLI.
        attn_backend=AttentionBackendName.flash_2,
    ).with_rope_scaling(
        # Yarn scaling for full attention layers only
        YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)
    )

    # Remove heads (and scale down d_model) to compensate for extra params.
    config.d_model -= (  # Lets not do this anymore, 32 heads is easier to work with
        REMOVE_HEADS * 128
    )
    config.block.attention.n_heads -= REMOVE_HEADS
    assert config.d_model / config.block.attention.n_heads == 128

    ### Copied below from hybrid/gated_deltanet_0_25_rnn_first.py ###

    # Update the config to use an FLA block.
    config.block.name = TransformerBlockType.fla_hybrid
    assert config.n_layers % 4 == 0, "Current logic assumes n_layers is multiple of 4"
    config.block.fla_hybrid_attention_indices = [i for i in range(config.n_layers) if i % 4 == 3]

    # Configure the non-attention part of the block to be a DeltaNet.
    config.block.fla = FLAConfig(
        name="GatedDeltaNet",
        dtype=config.dtype,
        fla_layer_kwargs={
            # FLA repo says num_heads * head_dim = 0.75 * hidden_size
            "head_dim": int(0.75 * config.d_model / config.block.attention.n_heads),
            "use_gate": True,
            "allow_neg_eigval": True,
        },
    )

    # Save memory by using fused linear loss implementation.
    config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    return config


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    # if common.launch is not None:
    #     gpus = {CLUSTER_TO_GPU_TYPE.get(c, "unknown") for c in common.launch.clusters}
    #     if all("B200" in g for g in gpus):
    #         rank_microbatch_size *= 2

    # Added because FLA models seem to use more memory than transformers.
    # rank_microbatch_size = int(rank_microbatch_size // MICROBATCH_DISCOUNT)

    return TransformerTrainModuleConfig(
        rank_microbatch_size=common.max_sequence_length,
        max_sequence_length=common.max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        cp_config=TransformerContextParallelConfig.ulysses(degree=6),
        # tp_config=TransformerTensorParallelConfig(degree=8),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.1,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=LinearWithWarmup(units=SchedulerUnits.steps, warmup=200, alpha_f=0.0),
    )


def build_data_components(
    common: CommonComponents,
    intra_document_masking: bool = True,
    include_instance_filter: bool = False,
) -> DataComponents:
    dataset_config = NumpyPackedFSLDatasetConfig.glob(
        "gs://ai2-llm/preprocessed/tylerr/lc-reshard-final-cleaned/v0.1/allenai/dolma2-tokenizer/*.npy",
        tokenizer=common.tokenizer,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        generate_doc_lengths=intra_document_masking,  # enables intra-document masking
        source_group_size=8,
        source_permutation_seed=123,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size, seed=34521, num_workers=8
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10

    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    return (
        TrainerConfig(
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=True,
            # !!! TODO: update this to the final midtrain checkpoint
            load_path=f"{common.root_dir}/checkpoints/willm/linear-rnns/OLMo3.1-7B-6T-30h-midtrain-deux/step23842/",
            save_folder=f"{common.root_dir}/checkpoints/willm/linear-rnns/{common.run_name}/",
            save_overwrite=True,
            metrics_collect_interval=50,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(MAX_TOKENS),
            hard_stop=HARD_STOP,
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
                project="linear-rnns",
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        # .with_recommended_evals(common.tokenizer, SEQUENCE_LENGTH, cluster, task_set="fast")
    )


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
        include_instance_filter=INSTANCE_FILTER,
        beaker_workspace="ai2/linear-rnns",
        use_hostname_constraints=True,
        num_execution_units=1,
    )
    main(config_builder=config_builder)
