"""
Official SFT think script for OLMo-hybrid-7B.

Fine-tunes the OLMo-hybrid-7B model on think SFT data. The hybrid architecture
combines Gated Delta Net (GDN) recurrent layers with standard attention layers
in a 3:1 ratio. RoPE embeddings are disabled (as in the long-context extension).
"""

import argparse
from typing import List

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.nn.transformer.config import TransformerBlockConfig
from olmo_core.optim import LinearWithWarmup, SkipStepAdamWConfig
from olmo_core.script_utils import ExperimentConfig, main
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

DEFAULT_SEQUENCE_LENGTH = 32_768
GLOBAL_BATCH_SIZE = 64 * DEFAULT_SEQUENCE_LENGTH
LR = 2.5e-5

# TODO: update Olmo-core versions of SFT checkpoints to olmo-checkpoints.org
LOAD_PATH = "/weka/oe-training-default/ai2-llm/checkpoints/willm/linear-rnns/OLMo3.1-7B-6T-30h-long-context-drope/step23842"

# TODO: upload tokenized SFT datasets to olmo-data.org
DATASET_PATH = (
    "/weka/oe-training-default/ai2-llm/jacobm/data/sft/rl-sft-32k/olmo-hybrid-sft-triple-tools"
)

# Remove heads to match params/TPS of OLMo3 7B transformer. This is to enable a
# fair comparison with OLMo3 7B. If training from scratch, we recommend setting the
# number of attention heads to 32 (or some power of 2 that makes sense for your model size).
REMOVE_HEADS = 2


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()

    # Build model config.
    model_config = TransformerConfig.olmo3_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )
    assert isinstance(model_config.block, TransformerBlockConfig)
    assert isinstance(model_config.block.sequence_mixer, AttentionConfig)

    # Remove heads (and scale down d_model) to compensate for extra params.
    model_config.d_model -= REMOVE_HEADS * 128
    num_heads = model_config.block.sequence_mixer.n_heads - REMOVE_HEADS
    model_config.block.sequence_mixer.n_heads = num_heads
    assert model_config.d_model / num_heads == 128

    attn_block = model_config.block

    # RoPE embeddings were disabled at the start of LC extension and so
    # they are disabled here as well.
    attn_block = attn_block.replace(
        sequence_mixer=attn_block.sequence_mixer.replace(rope=None),
    )

    gdn_block = attn_block.replace(
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=num_heads,
            head_dim=int(0.75 * model_config.d_model / num_heads),
            allow_neg_eigval=True,
        ),
    )

    # 3 GDN layers followed by 1 attention layer, repeating.
    model_config.block = {"gdn": gdn_block, "attn": attn_block}
    model_config.block_pattern = ["gdn", "gdn", "gdn", "attn"]
    assert model_config.n_layers % len(model_config.block_pattern) == 0

    # Save memory by using fused linear loss implementation.
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    # Build dataset config.
    clean_path = DATASET_PATH.rstrip("/")
    dataset_config = NumpyPackedFSLDatasetConfig(
        tokenizer=tokenizer_config,
        paths=[f"{clean_path}/token_ids_part_*.npy"],
        expand_glob=True,
        label_mask_paths=[f"{clean_path}/labels_mask_*.npy"],
        generate_doc_lengths=True,
        long_doc_strategy=LongDocStrategy.truncate,
        sequence_length=sequence_length,
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
    )

    # Build train module config.
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=sequence_length,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            compile=False,
        ),
        scheduler=LinearWithWarmup(
            warmup_fraction=0.03,
            alpha_f=0.0,
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
    )

    # Build trainer config.
    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.epochs(2),
            load_path=LOAD_PATH,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=False,
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
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,  # NOTE: change to true to enable
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


if __name__ == "__main__":
    main(build_config)
