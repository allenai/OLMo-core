"""
32k, context-parallel (Ulysses 8) Beaker/gantry **SINGLE-TASK** length-ladder SFT of the Qwen3-4B
FAST-COMPRESSIVE-LANDMARK CPT model -- ONE task's whole ladder (4k/8k/16k/32k), **NO CPT mix**.

Single-task counterpart of ``Qwen3-4B-compressive-cptmix-5task-32k-SFT.py``. Differences:
  * NO CPT source and NO 5-task mix: exactly one SFT source at ratio 1.0 (CPT_FRAC=0).
  * The task is parsed from the run name (so no extra CLI plumbing through the internal framework):
    the run name must contain one of ``contra|nq|oolong|rerank|outlier``.
  * Budget is EPOCHS over the (~2000-example) task ladder, not a fixed 1465-step token budget.

Data (weka): per-task ladder shards under
``prasanns/single_task_ladders/<task>/{token_ids_part_*.npy,labels_mask_*.npy}`` -- the single-task
tokenizer output uploaded from /scratch. Compressive CPT base + LandmarkPacking pipeline are unchanged
from the cptmix launcher (the kernel supports cu_doc_lens via DOC_MASK, so packing is efficient).

    PYTHONPATH=src python src/scripts/train/sft/singletask_ladder/Qwen3-4B-compressive-singletask-ladder-32k-SFT.py \\
        dry_run q4b-comp-contra-ladder32k ai2/neptune
    PYTHONPATH=src python src/scripts/train/sft/singletask_ladder/Qwen3-4B-compressive-singletask-ladder-32k-SFT.py \\
        launch  q4b-comp-contra-ladder32k ai2/neptune
"""

from dataclasses import replace
from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LandmarkPackingInstanceSourceConfig,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode, TransformerConfig
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

MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1
SEQUENCE_LENGTH = 40960
LANDMARK_TOKEN_ID = 151860
NONSELECTED_LANDMARK_MASS = 0.1
CP_DEGREE = 8
NUM_NODES = 1
EPOCHS = 2

# weka per-task ladder data + compressive CPT base.
DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders"
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/"
    "q4b-base-fast-compressive-landmark-8node/step2385/model_and_optim"
)

_TASK_DIR = {"contra": "contradiction", "nq": "nq", "oolong": "oolong",
             "rerank": "rerank", "outlier": "outlier"}
_TASK_LABEL = {"contra": "contradiction", "nq": "nq_retrieval", "oolong": "oolong",
               "rerank": "rerank", "outlier": "outlier"}

LR = 1e-5
GLOBAL_BATCH_SIZE = NUM_NODES * SEQUENCE_LENGTH


def _task_from_run_name(run_name: str) -> str:
    for key in _TASK_DIR:
        if key in run_name:
            return key
    raise SystemExit(
        f"run name {run_name!r} must contain one of {sorted(_TASK_DIR)} to pick the task."
    )


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    task = _task_from_run_name(cli_context.run_name)
    task_root = f"{DATA_ROOT}/{_TASK_DIR[task]}"
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{root_dir}/checkpoints/prasanns/{cli_context.run_name}"

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
        # Ship the (uncommitted) single-task launcher via an ephemeral ref -- no permanent commit.
        beaker_launch_config.allow_dirty = True

    tokenizer_config = TokenizerConfig.qwen3()
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        fast_compressive_landmark=True,
        nonselected_landmark_mass=NONSELECTED_LANDMARK_MASS,
        mem_freq=MEM_FREQ,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR, weight_decay=0.0, betas=(0.9, 0.95),
            group_overrides=[OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full, shard_degree=1,
        ),
        cp_config=TransformerContextParallelConfig.ulysses(degree=CP_DEGREE),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget, activation_memory_budget=0.7,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # ---- Single SFT source, ratio 1.0, NO CPT ----
    task_source = NumpyDocumentSourceConfig(
        source_paths=[f"{task_root}/token_ids_part_*.npy"],
        tokenizer=doc_tokenizer_config,
        label_mask_paths=[f"{task_root}/labels_mask_*.npy"],
        expand_glob=True,
    )
    instance_source_config = LandmarkPackingInstanceSourceConfig(
        source=MixingDocumentSourceConfig(source_specs=[
            MixingDocumentSourceSpecConfig(
                source=task_source, ratio=1.0, max_repetition_factor=8.0, label=_TASK_LABEL[task],
            )
        ]),
        sequence_length=SEQUENCE_LENGTH,
        mem_freq=MEM_FREQ,
        mem_id=LANDMARK_TOKEN_ID,
        pad_id=tokenizer_config.pad_token_id,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config, work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE, seed=34521, num_workers=4,
        generate_doc_lengths=False,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir, save_overwrite=True, load_path=BASE_CHECKPOINT,
            load_strategy=LoadStrategy.always, load_trainer_state=False, load_optim_state=False,
            metrics_collect_interval=10, cancel_check_interval=10,
            max_duration=Duration.epochs(EPOCHS),
        )
        .with_callback("checkpointer", CheckpointerCallback(
            save_interval=100000, ephemeral_save_interval=500, max_checkpoints=2, save_async=True))
        .with_callback("wandb", WandBCallback(
            name=run_name_with_ts, group=cli_context.run_name,
            entity="prasanns-allen-institute-for-ai", project="memory-networks",
            enabled=True, cancel_check_interval=10))
        .with_callback("slack_notifier", SlackNotifierCallback(name=run_name_with_ts, enabled=False))
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
    main(config_builder=build_experiment_config)
