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
from olmo_core.float8 import Float8Config
from olmo_core.nn.transformer import (
    TransformerConfig,
    TransformerType,
    TransformerBlockConfig,
    TransformerBlockType,
)
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.mamba import MambaConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.blt.config import LocalEncoderConfig, LocalDecoderConfig
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
TEACHER_MODE = os.environ.get("TEACHER_MODE", None)
GLOBAL_MODEL_LEARNING_RATE = os.environ.get("GLOBAL_MODEL_LEARNING_RATE", "")
LOCAL_MODEL_STYLE = os.environ.get("LOCAL_MODEL_STYLE", "hnet")
DATA_SOURCE = os.environ.get("DATA_SOURCE", "dclm")
ADD_HASH_EMBEDDINGS = os.environ.get("ADD_HASH_EMBEDDINGS", "1").lower() in {"1", "true", "yes"}
OLMO_ARCH = os.environ.get("OLMO_ARCH", "olmo2_1B_v2")

if DATA_SOURCE == "dclm":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources.txt").read().strip().splitlines()
elif DATA_SOURCE == "dolmino":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources_dolmino.txt").read().strip().splitlines()
else:
    raise ValueError(f"Unknown DATA_SOURCE: {DATA_SOURCE}. Must be one of 'dclm', 'dolmino'.")

STAGE1_CKPT_PATH = os.environ.get(
    "STAGE1_CKPT_PATH",
    "/weka/oe-training-default/benjaminm/runs_persist/hnet_v3_d2048_emb_v2_smaller_hash_embed/step50000/model_and_optim",
)
DATA_PATHS = ["/weka/oe-training-default/" + x for x in _DATA_SOURCES]

if not os.environ.get("HAS_WEKA"):
    STAGE1_CKPT_PATH = STAGE1_CKPT_PATH.replace("/weka/oe-training-default/", "gs://ai2-llm/")
    DATA_PATHS = [x.replace("/weka/oe-training-default/", "gs://") for x in DATA_PATHS] # slight inconsistency

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
    global NUM_WORKERS, GLOBAL_BATCH_SIZE, LOCAL_BATCH_SIZE

    BYTE_EXPANSION_FACTOR = int(os.environ.get("BYTE_EXPANSION_FACTOR", "6"))  # default (max) expansion factor
    SAVE_FOLDER = os.environ.get("SAVE_FOLDER", f"/tmp/{run_name}")

    if not os.environ.get("HAS_WEKA"):
        SAVE_FOLDER = SAVE_FOLDER.replace("/weka/oe-training-default/", "gs://ai2-llm/")

    byte_tokenizer_config = ByteTokenizerConfig.blt()
    subword_tokenizer_config = TokenizerConfig.dolma2()

    if QUICK_DEBUG:
        NUM_WORKERS = 0
        GLOBAL_BATCH_SIZE = 4
        LOCAL_BATCH_SIZE = 4

    teacher_model_config = getattr(TransformerConfig, OLMO_ARCH)(
        vocab_size=subword_tokenizer_config.padded_vocab_size()
    )
    teacher_model_config = teacher_model_config.replace(
        freeze_params=["*"] # don't train teacher
    )

    if LOCAL_MODEL_STYLE == "blt":
        if OLMO_ARCH == "olmo2_1B_v2":
            local_d_model = 1024
        elif OLMO_ARCH == "olmo2_7B":
            local_d_model = 2048
        else:
            raise ValueError(f"Unknown OLMO_ARCH: {OLMO_ARCH}. Must be one of 'olmo2_1B_v2', 'olmo2_7B'.")

        local_encoder_n_layers = 1
        local_decoder_n_layers = 9
        local_attn_n_heads = 16
        local_cross_attn_n_heads = 16
        local_block = teacher_model_config.block.replace(
            attention=teacher_model_config.block.attention.replace(n_heads=local_attn_n_heads),
            feed_forward=FeedForwardConfig(hidden_size=local_d_model, bias=False),
        )

        local_encoder = LocalEncoderConfig(
            add_hash_embeddings=ADD_HASH_EMBEDDINGS,
            hash_byte_group_size=[3, 4, 5, 6, 7, 8],
            hash_byte_group_vocab=[1536, 3072, 6144, 12288, 24576, 49152],
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
    elif LOCAL_MODEL_STYLE == "hnet":
        if OLMO_ARCH == "olmo2_1B_v2":
            local_d_model = 2048
        elif OLMO_ARCH == "olmo2_7B":
            local_d_model = 4096
        else:
            raise ValueError(f"Unknown OLMO_ARCH: {OLMO_ARCH}. Must be one of 'olmo2_1B_v2', 'olmo2_7B'.")

        local_encoder_n_layers = 4
        local_decoder_n_layers = 4
        local_block = TransformerBlockConfig(
            name=TransformerBlockType.mamba,
            attention=AttentionConfig(), # not used
            mamba=MambaConfig(
                chunk_size=256,
                d_conv=4,
                d_state=128,
                expand=2,
            ),
            feed_forward=None,
            layer_norm=teacher_model_config.block.layer_norm,
        )

        local_encoder = LocalEncoderConfig(
            add_hash_embeddings=ADD_HASH_EMBEDDINGS,
            hash_byte_group_size=[3, 4, 5, 6, 7, 8],
            hash_byte_group_vocab=[1536, 3072, 6144, 12288, 24576, 49152],
            hash_byte_group_nb_functions=1,
            d_model=local_d_model,
            n_layers=local_encoder_n_layers,
            sliding_window_size=0,
            cross_attn_n_heads=0,
            block_config=local_block,
            add_norm_after_last_block=True,
            add_out_projection=True,
            pooling="hnet",
        )
        local_decoder = LocalDecoderConfig(
            d_model=local_d_model,
            n_layers=local_decoder_n_layers,
            sliding_window_size=0,
            cross_attn_n_heads=0,
            block_config=local_block,
            add_norm_before_first_block=True,
            add_norm_onto_residual=False,
            add_in_projection=True,
            depooling="hnet",
        )
    else:
        raise ValueError(f"Unknown LOCAL_MODEL_STYLE: {LOCAL_MODEL_STYLE}. Must be one of 'blt', 'hnet'.")

    if TEACHER_MODE == "stage1":
        # use the stage1 checkpoint as the teacher instead of the original model (stage0)
        teacher_model_config = teacher_model_config.replace(
            name=TransformerType.blt,
            vocab_size=byte_tokenizer_config.padded_vocab_size(),
            local_encoder=local_encoder,
            local_decoder=local_decoder,
            add_boundary_predictor=True,
        )

    model_config = teacher_model_config.replace(
        name=TransformerType.blt,
        vocab_size=byte_tokenizer_config.padded_vocab_size(),
        local_encoder=local_encoder,
        local_decoder=local_decoder,
        teacher_config=teacher_model_config,
        share_blocks_between_teacher_and_student=False,
        use_teacher_embs_with_vocab_size=subword_tokenizer_config.padded_vocab_size() if TEACHER_MODE == "stage1" else None,
        add_boundary_predictor=True,
        freeze_params=["boundary_predictor.*"] # temporary
    )

    dataset_config = NumpyDatasetConfig(
        paths=DATA_PATHS,
        name=NumpyDatasetType.byte_fsl,
        sequence_length=SEQUENCE_LENGTH, # subword sequence length
        max_sequence_length=SEQUENCE_LENGTH * BYTE_EXPANSION_FACTOR, # max. length of the byte sequence
        tokenizer=byte_tokenizer_config,
        work_dir=os.path.join(SAVE_FOLDER, "data"),
    )

    group_overrides = [
        OptimGroupOverride(
            params=[
                "local_encoder.embedding.weight",
            ] + [
                "local_encoder.hash_embeddings.*.weight"
            ] if ADD_HASH_EMBEDDINGS else [],
            opts=dict(weight_decay=0.0)
        )
    ]

    if GLOBAL_MODEL_LEARNING_RATE:
        group_overrides.append(
            OptimGroupOverride(
                params=[
                    "blocks.*"
                ],
                opts=dict(lr=float(GLOBAL_MODEL_LEARNING_RATE))
            )
        )

    optim = AdamWConfig(lr=1e-3, group_overrides=group_overrides)

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE * SEQUENCE_LENGTH * BYTE_EXPANSION_FACTOR,
        seed=0,
        num_workers=NUM_WORKERS if not QUICK_DEBUG else 0,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=LOCAL_BATCH_SIZE * SEQUENCE_LENGTH * BYTE_EXPANSION_FACTOR,
        max_sequence_length=dataset_config.effective_sequence_length,
        optim=optim,
        compile_model=True,
        float8_config=Float8Config(enabled=False),
        blt_config=BLTConfig(
            tokenizer=byte_tokenizer_config,
            losses=["ce"],
            loss_weights=[1.0],
            skip_blocks=False,
            skip_teacher=TEACHER_MODE is None,
            use_oracle_patch_reps=False,
        ),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup=10000),
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
            "basic_skills_string_operations_rc_5shot",
            "basic_skills_pattern_rc_5shot",
            "basic_skills_logical_reasoning_rc_5shot",
            "basic_skills_common_knowledge_rc_5shot",
            "basic_skills_coding_rc_5shot",
            "basic_skills_arithmetic_rc_5shot",
        ]

    all_eval_tasks = eval_tasks
    all_eval_names = ["downstream" for _ in eval_tasks]
    all_eval_batch_kwargs = [{} for _ in eval_tasks]

    trainer_config = (
        TrainerConfig(
            save_folder=SAVE_FOLDER,
            save_overwrite=True,
            load_strategy=LoadStrategy.never if QUICK_DEBUG else LoadStrategy.if_available,
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
                ephemeral_save_interval=500,
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
                tasks=all_eval_tasks,
                names=all_eval_names,
                batch_kwargs=all_eval_batch_kwargs,
                tokenizer=byte_tokenizer_config,
                eval_interval=5000,
                eval_on_startup=False,
                save_results=True,
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

    # Load Stage 1 checkpoint.
    # since we can"t share blocks (we might train them), we need to duplicate the blocks to the teacher
    prefixes_to_duplicate = ["blocks"]
    # If we are using the stage1 checkpoint as the teacher, we also need to duplicate
    # the local encoder / decoder / boundary predictor / lm head to teacher
    if TEACHER_MODE == "stage1":
        prefixes_to_duplicate += ["local_encoder", "local_decoder", "boundary_predictor", "lm_head"]

    key_mapping = {}
    extend_key_mapping = {}

    for prefix in prefixes_to_duplicate:
        extend_key_mapping.update({
            key: key.replace(f"{prefix}", f"teacher.{prefix}")
            for key in model.state_dict().keys() if key.startswith(prefix)
        })

    if config.model.use_teacher_embs_with_vocab_size is not None:
        key_mapping["teacher_embeddings.weight"] = "teacher.embeddings.weight"

    incompatible_keys = load_model_and_optim_state(STAGE1_CKPT_PATH, model, key_mapping=key_mapping, extend_key_mapping=extend_key_mapping)

    if len(incompatible_keys.unexpected_keys) > 0:
        raise ValueError(f"Unexpected keys when loading checkpoint: {incompatible_keys.unexpected_keys} (assume we use all teacher weights)")

    for missing_key in incompatible_keys.missing_keys:
        log.info(f"Key {missing_key} was not found in checkpoint, is randomly initialized (this is expected for local encoder/decoder and student lm head).")

    # TODO(benjaminm): this is not a nice place?
    register_fsdp_forward_method(model, "student_forward")
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
