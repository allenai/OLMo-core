"""
Example of how to train a Llama transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/llama/train.py run_name [OVERRIDES...]
"""

import os
import sys
from dataclasses import dataclass
from typing import List, cast
import traceback
import logging
from pathlib import Path

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.optim.scheduler import WSD, LinearWithWarmup, ConstantScheduler
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
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

NUM_WORKERS = 16
SEQUENCE_LENGTH = int(os.environ.get("SEQUENCE_LENGTH", 1024))
QUICK_DEBUG = False
GLOBAL_BATCH_SIZE = 64
LOCAL_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 16
DATA_SOURCE = os.environ.get("DATA_SOURCE", "dclm")
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "linear_with_warmup")
OLMO_ARCH = os.environ.get("OLMO_ARCH", "olmo2_1B_v2")
TOKENIZER = os.environ.get("TOKENIZER", "dolma2")

if DATA_SOURCE == "dclm":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources.txt").read().strip().splitlines()
elif DATA_SOURCE == "dolmino":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources_dolmino.txt").read().strip().splitlines()
elif DATA_SOURCE == "dolma2_code_string":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources_dolma2_code_string.txt").read().strip().splitlines()
elif DATA_SOURCE == "dolma2_150b_code_string":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources_dolma2_150b_code_string.txt").read().strip().splitlines()
elif DATA_SOURCE == "dolmino_code_string":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources_dolmino_code_string.txt").read().strip().splitlines()
elif DATA_SOURCE == "tulu3":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources_tulu3.txt").read().strip().splitlines()
elif DATA_SOURCE == "fineweb2_thai_sample":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources_fineweb2_thai_sample.txt").read().strip().splitlines()
elif DATA_SOURCE == "fineweb2_thai_sample_typhoon_tokenized":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources_fineweb2_thai_sample_typhoon_tokenized.txt").read().strip().splitlines()
else:
    raise ValueError(f"Unknown DATA_SOURCE: {DATA_SOURCE}. Must be one of 'dclm', 'dolmino', 'dolma2_code_string', 'dolmino_code_string'.")

OLMO_ARCH = os.environ.get("OLMO_ARCH", "olmo2_1B_v2")

OLMO_CKPT_PATH = os.environ.get(
    "OLMO_CKPT_PATH",
    "/weka/oe-training-default/benjaminm/checkpoints/olmo2_1b/model_and_optim",
)
DATA_PATHS = ["/weka/oe-training-default/" + x for x in _DATA_SOURCES]

if not os.environ.get("HAS_WEKA"):
    OLMO_CKPT_PATH = OLMO_CKPT_PATH.replace("/weka/oe-training-default/ai2-llm/", "gs://ai2-llm/").replace("/weka/oe-training-default/", "gs://ai2-llm/")
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

    SAVE_FOLDER = os.environ.get("SAVE_FOLDER", f"/tmp/{run_name}")

    if not os.environ.get("HAS_WEKA"):
        SAVE_FOLDER = SAVE_FOLDER.replace("/weka/oe-training-default/", "gs://ai2-llm/")

    if QUICK_DEBUG:
        NUM_WORKERS = 0
        GLOBAL_BATCH_SIZE = 4
        LOCAL_BATCH_SIZE = 4

    tokenizer_config = getattr(TokenizerConfig, TOKENIZER)()
    model_config = getattr(TransformerConfig, OLMO_ARCH)(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    dataset_config = NumpyFSLDatasetConfig(
        paths=DATA_PATHS,
        sequence_length=SEQUENCE_LENGTH, # subword sequence length
        tokenizer=tokenizer_config,
        work_dir=os.path.join(SAVE_FOLDER, "data"),
    )

    optim = AdamWConfig(
        lr=1e-3,
        group_overrides=[
            OptimGroupOverride(
                params=["embeddings.weight"],
                opts=dict(weight_decay=0.0)
            )
        ],
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE * SEQUENCE_LENGTH,
        seed=0,
        num_workers=NUM_WORKERS,
    )

    if LR_SCHEDULE == "linear_with_warmup":
        scheduler = LinearWithWarmup(warmup_fraction=0.1, alpha_f=0.0)
    elif LR_SCHEDULE == "wsd":
        scheduler = WSD(
            warmup_fraction=0.1,
            decay_fraction=0.2,
            decay_kind="inv_sqrt",
            cosine_decay_alpha=10,
            decay_min_lr=None,
            decay_min_lr_ratio=0.1,
        )
    elif LR_SCHEDULE == "constant":
        scheduler = ConstantScheduler()
    else:
        raise ValueError(f"Unknown LR_SCHEDULE: {LR_SCHEDULE}. Must be one of 'linear_with_warmup', 'wsd', 'constant'.")

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=LOCAL_BATCH_SIZE * SEQUENCE_LENGTH,
        max_sequence_length=dataset_config.sequence_length,
        optim=optim,
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        scheduler=scheduler,
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
                save_interval=1000,
                ephemeral_save_interval=100,
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
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=all_eval_tasks,
                names=all_eval_names,
                batch_kwargs=all_eval_batch_kwargs,
                tokenizer=tokenizer_config,
                eval_interval=5000,
                eval_on_startup=False,
                save_results=True,
                batch_size=EVAL_BATCH_SIZE * SEQUENCE_LENGTH, # these are subword tokens, so no expansion factor
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
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
        dp_process_group=train_module.dp_process_group
    )
    trainer = config.trainer.build(train_module, data_loader)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    load_model_and_optim_state(OLMO_CKPT_PATH, model)

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
        import ipdb; ipdb.post_mortem()
    finally:
        teardown_training_environment()
