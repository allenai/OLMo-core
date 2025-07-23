"""
Example of how to train a Llama transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/llama/train.py run_name [OVERRIDES...]
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import List, cast
import glob
import traceback
from pathlib import Path

from torch.distributed.fsdp import register_fsdp_forward_method

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    NumpyByteFSLDataset,
    ByteTokenizerConfig,
    TokenizerConfig,
    ByteDataCollator,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.transformer import TransformerConfig, TransformerType
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.blt import LocalEncoderConfig, LocalDecoderConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.common import LoadStrategy
from olmo_core.nn.blt.config import BLTConfig
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

NUM_WORKERS = 16
SEQUENCE_LENGTH = 1024
QUICK_DEBUG = False
GLOBAL_BATCH_SIZE = 64
LOCAL_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 16
TRAIN_MODE = os.environ.get("TRAIN_MODE", "local_encoder_only")
DATA_SOURCE = os.environ.get("DATA_SOURCE", "dclm")

if DATA_SOURCE == "dclm":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources.txt").read().strip().splitlines()
elif DATA_SOURCE == "dolmino":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources_dolmino.txt").read().strip().splitlines()
else:
    raise ValueError(f"Unknown DATA_SOURCE: {DATA_SOURCE}. Must be one of 'dclm', 'dolmino'.")

if os.environ.get("HAS_WEKA"):
    OLMO_1B_CKPT_PATH = os.environ.get(
        "OLMO_CKPT_PATH",
        "/weka/oe-training-default/benjaminm/checkpoints/olmo2_1b/model_and_optim",
    )
    DATA_PATHS = ["/weka/oe-training-default/" + x for x in _DATA_SOURCES]
    EMBEDDING_INIT_PATH = os.environ.get(
        "EMBEDDING_INIT_PATH",
        "/weka/oe-training-default/benjaminm/olmo_1b_blt_hash_embedding_init",
    )
else:
    OLMO_1B_CKPT_PATH = "gs://allennlp-benjaminm/checkpoints/olmo2_1b/model_and_optim"
    DATA_PATHS = ["gs://" + x for x in _DATA_SOURCES]
    raise NotImplementedError()

DATA_WORK_DIR = "/tmp/dataset-cache"

log = logging.getLogger(__name__)

@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def build_config(run_name: str, overrides: List[str]) -> ExperimentConfig:
    BYTE_EXPANSION_FACTOR = 8  # default (max) expansion factor

    byte_tokenizer_config = ByteTokenizerConfig.blt()
    subword_tokenizer_config = TokenizerConfig.dolma2()

    if QUICK_DEBUG:
        teacher_model_config = TransformerConfig.olmo2_190M(
            vocab_size=subword_tokenizer_config.padded_vocab_size()
        )
        local_d_model = 384
    else:
        teacher_model_config = TransformerConfig.olmo2_1B_v2(
            vocab_size=subword_tokenizer_config.padded_vocab_size()
        )
        local_d_model = 1024

    teacher_model_config = teacher_model_config.replace(
        freeze_params=["*"] # don't train teacher
    )

    local_encoder_n_layers = 1
    local_decoder_n_layers = 9
    local_attn_n_heads = 16
    local_cross_attn_n_heads = 16
    local_block = teacher_model_config.block.replace(
        attention=teacher_model_config.block.attention.replace(n_heads=local_attn_n_heads),
        feed_forward=FeedForwardConfig(hidden_size=local_d_model, bias=False),
    )

    local_encoder = LocalEncoderConfig(
        hash_byte_group_size=[3, 4, 5, 6, 7, 8],
        hash_byte_group_vocab=100_002,
        hash_byte_group_nb_functions=1,
        sliding_window_size=512,
        d_model=local_d_model,
        n_layers=local_encoder_n_layers,
        cross_attn_n_heads=local_cross_attn_n_heads,
        block_config=local_block,
        add_out_projection=False,
    )
    local_decoder = LocalDecoderConfig(
        sliding_window_size=512,
        d_model=local_d_model,
        n_layers=local_decoder_n_layers,
        cross_attn_n_heads=local_cross_attn_n_heads,
        block_config=local_block,
    )
    model_config = teacher_model_config.replace(
        name=TransformerType.blt,
        vocab_size=byte_tokenizer_config.padded_vocab_size(),
        local_encoder=local_encoder,
        local_decoder=local_decoder,
        teacher_config=teacher_model_config,
        share_blocks_between_teacher_and_student=True,
        add_boundary_predictor=True,
        freeze_params=[
            "blocks*" # freeze inner transformer layers
        ]
    )

    if QUICK_DEBUG:
        # save on hash embeddings to reduce gpu memory
        model_config = model_config.replace(
            local_encoder=model_config.local_encoder.replace(  # type: ignore
                hash_byte_group_size=[3],
                hash_byte_group_nb_functions=1,
            )
        )

    dataset_config = NumpyDatasetConfig(
        paths=DATA_PATHS,
        name=NumpyDatasetType.byte_fsl,
        sequence_length=SEQUENCE_LENGTH, # subword sequence length
        max_sequence_length=SEQUENCE_LENGTH * BYTE_EXPANSION_FACTOR, # max. length of the byte sequence
        tokenizer=byte_tokenizer_config,
        work_dir=DATA_WORK_DIR,
    )

    optim = AdamWConfig(
        lr=1e-3,
        group_overrides=[
            OptimGroupOverride(
                params=[
                    "local_encoder.embedding.weight",
                    "local_encoder.hash_embeddings.*.weight"
                ],
                opts=dict(weight_decay=0.0)
            )
        ],
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE * SEQUENCE_LENGTH * BYTE_EXPANSION_FACTOR,
        seed=0,
        num_workers=NUM_WORKERS if not QUICK_DEBUG else 0,
    )

    if TRAIN_MODE == "local_encoder_only":
        losses = ["local_encoder"]
        loss_weights = [1.0]
    elif TRAIN_MODE == "local_decoder_only":
        losses = ["local_decoder"]
        loss_weights = [1.0]
    else:
        raise ValueError(f"Unknown TRAIN_MODE: {TRAIN_MODE}. Must be one of 'local_encoder_only', 'local_decoder_only'.")

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=LOCAL_BATCH_SIZE * SEQUENCE_LENGTH * BYTE_EXPANSION_FACTOR,
        max_sequence_length=dataset_config.effective_sequence_length,
        optim=optim,
        compile_model=True,
        blt_config=BLTConfig(
            tokenizer=byte_tokenizer_config,
            losses=losses,
            loss_weights=loss_weights,
            skip_blocks=TRAIN_MODE == "local_encoder_only",
            skip_teacher=False,
            use_oracle_patch_reps=TRAIN_MODE == "local_decoder_only",
        ),  
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup=1000),
    )

    if QUICK_DEBUG:
        eval_tasks = [
            "arc_challenge_test_rc_5shot",
        ]
    else:
        eval_tasks = [
            "arc_challenge_test_rc_5shot",
            "arc_easy_test_rc_5shot",
            "hellaswag_rc_5shot",  # 1K subset of HellaSwag
            "winogrande_val_rc_5shot",  # Helpful after 750M-5xC scale
            "csqa_val_rc_5shot",
            "piqa_val_rc_5shot",
            "mmlu_stem_test_rc_5shot",
            "mmlu_humanities_test_rc_5shot",
            "mmlu_social_sciences_test_rc_5shot",
            "mmlu_other_test_rc_5shot",
        ]
    if TRAIN_MODE == "local_encoder_only":
        eval_names = [f"downstream_orig_head" for _ in eval_tasks]
        eval_batch_kwargs = [{"eval_mode": "orig_head"} for _ in eval_tasks]
    elif TRAIN_MODE == "local_decoder_only":
        eval_names = [f"downstream_orig_trunk" for _ in eval_tasks]
        eval_batch_kwargs = [{"eval_mode": "orig_trunk"} for _ in eval_tasks]
    else:
        eval_names = [f"downstream" for _ in eval_tasks]
        eval_batch_kwargs = [{} for _ in eval_tasks]

    trainer_config = (
        TrainerConfig(
            save_folder=f"/tmp/{run_name}",
            save_overwrite=True,
            load_strategy=LoadStrategy.never,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(10000),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                pre_train_checkpoint=False,
                save_interval=10000,
                ephemeral_save_interval=1000,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                project="benjaminm-tok",
                entity="ai2-llm",
                cancel_check_interval=10,
                enabled=True,  # change to true to enable
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        #  FIXME: make byte tokenizer work for eval
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=eval_tasks,
                names=eval_names,
                batch_kwargs=eval_batch_kwargs,
                tokenizer=byte_tokenizer_config,
                eval_interval=2500,
                eval_on_startup=False,
                batch_size=EVAL_BATCH_SIZE * SEQUENCE_LENGTH, # these are subword tokens, so no expansion factor
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,  # type: ignore
        dataset=dataset_config,  # type: ignore
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


def main(run_name: str, overrides: List[str]):
    config = build_config(run_name, overrides)

    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)

    dataset = config.dataset.build()
    data_loader = config.data_loader.build(
        dataset,
        collator=ByteDataCollator(pad_token_id=dataset.pad_token_id) if isinstance(dataset, NumpyByteFSLDataset) else None,
        dp_process_group=train_module.dp_process_group
    )
    trainer = config.trainer.build(train_module, data_loader)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    if not QUICK_DEBUG:
        # Load OLMo 1B checkpoint.
        # assume share_blocks=True, so we don't have to map/duplicate block weights
        random_init_keys = {"local_encoder", "boundary_predictor", "local_decoder"}
        key_mapping = {
            key: None for key in model.state_dict().keys() if any(key.startswith(x) for x in random_init_keys)
        } | {
            f"teacher.{key}": key for key in model.teacher.state_dict().keys()  # type: ignore
        }
        incompatible_keys = load_model_and_optim_state(OLMO_1B_CKPT_PATH, model, key_mapping=key_mapping, strict=False)

        if len(incompatible_keys.unexpected_keys) > 0:
            raise ValueError(f"Unexpected keys when loading checkpoint: {incompatible_keys.unexpected_keys} (assume we use all teacher weights)")

        for missing_key in incompatible_keys.missing_keys:
            log.info(f"Key {missing_key} was not found in checkpoint, is randomly initialized (this is expected for local encoder/decoder and student lm head).")

        # init embeddings + scale appropriately
        model.fix_init(EMBEDDING_INIT_PATH)  # type: ignore

    # TODO(benjaminm): this is not a nice place?
    register_fsdp_forward_method(model, "original_head_forward")
    register_fsdp_forward_method(model, "original_trunk_forward")

    # Train.
    trainer.fit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]

    prepare_training_environment()
    try:
        main(run_name, overrides=overrides)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        traceback.print_exc()
        if get_rank() == 0:
            import ipdb; ipdb.post_mortem()
    finally:
        teardown_training_environment()
