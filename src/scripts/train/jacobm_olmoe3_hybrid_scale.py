#!/usr/bin/env python3
"""Train active-matched integration-wide GDN hybrids at any ladder size."""

# ruff: noqa: E402

from __future__ import annotations

import logging
import os
import socket
from functools import partial
from pathlib import Path
from typing import Any, cast


def configure_rank_local_compile_cache() -> None:
    local_rank = os.environ.get("LOCAL_RANK", "0")
    job_id = os.environ.get("BEAKER_EXPERIMENT_ID", "local")
    host = socket.gethostname().split(".")[0]
    cache_dir = Path("/tmp/olmo-compile-cache") / job_id / host / f"rank{local_rank}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_dir / "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir / "inductor"))


configure_rank_local_compile_cache()

import torch

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    DataComponents,
    ExperimentConfig,
    build_config,
    main,
)
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from olmo_core.optim import OLMoDDPOptimizerConfig, OptimGroupOverride, SchedulerUnits
from olmo_core.optim.scheduler import (
    ComposableScheduler,
    ComposableSchedulerStage,
    ComposableSchedulerStageType,
)
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    CheckpointRemovalStrategy,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    LMEvaluatorCallbackConfig,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)
from scripts.train.jacobm_olmoe_ladder.v2.models.hybrid_wide import (
    MAX_ACTIVE_PARAMETER_DELTA_FRACTION,
    MODEL_SIZES,
    build_hybrid_model_config,
    load_wide_model_config,
)

log = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")

# The audited 1.2B geometry + attention-gating profiles are 6.1155% larger in
# active parameters than the wide baseline. Keep the ordinary near-match guard
# at 6%, but allow these explicitly audited profiles up to 6.2% rather than
# rejecting them before training starts.
GATED_MAX_ACTIVE_PARAMETER_DELTA_FRACTION = 0.062
# GDN2 adds mixer parameters without changing the MoE, attention, or model
# geometry. These size-specific ceilings admit the exact audited configs while
# retaining a narrow guard against accidentally changing another dimension.
GDN2_MAX_ACTIVE_PARAMETER_DELTA_FRACTIONS = {
    "275m": 0.093,
    "480m": 0.084,
    "810m": 0.111,
    "1p2b": 0.125,
}

GDN2_275M_NOPE_SETTINGS = {
    "geometry_275m_gdn2_ev2_nope_gated": (2.0, True),
    "geometry_275m_gdn2_ev1_neg_nope_gated": (1.0, True),
    "geometry_275m_gdn2_ev2_noneg_nope_gated": (2.0, False),
    "geometry_275m_gdn2_ev1_noneg_nope_gated": (1.0, False),
}
GDN2_SCALE_NOPE_SETTINGS = {
    "geometry_matched_gdn2_ev2_nope_gated": (2.0, True),
    "geometry_matched_gdn2_ev1_noneg_nope_gated": (1.0, False),
}
KDA_275M_VARIANT = "geometry_275m_kda_ev1_noneg_nope_gated"
KDA_SCALE_VARIANT = "geometry_matched_kda_ev1_noneg_nope_gated"


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


MODEL_SIZE = os.environ.get("OLMOE3_HYBRID_MODEL_SIZE", "275m")
MODEL_VARIANT = os.environ.get("OLMOE3_HYBRID_MODEL_VARIANT", "integration_wide_gdn_ev1")
SEQUENCE_LENGTH = int(os.environ.get("OLMOE3_HYBRID_SEQUENCE_LENGTH", "8192"))
GLOBAL_BATCH_SIZE = int(os.environ.get("OLMOE3_HYBRID_GLOBAL_BATCH_SIZE", "262144"))
WORLD_SIZE = int(os.environ.get("OLMOE3_HYBRID_WORLD_SIZE", "2"))
NUM_NODES = int(os.environ.get("OLMOE3_HYBRID_NUM_NODES", "1"))
EP_SIZE = int(os.environ.get("OLMOE3_HYBRID_EP_SIZE", "1"))
EP_PATH = ExpertParallelPath(
    os.environ.get("OLMOE3_HYBRID_EP_PATH", ExpertParallelPath.rowwise_nvshmem.value)
)
EP_USE_CODE_DEFAULTS = env_bool("OLMOE3_HYBRID_EP_USE_CODE_DEFAULTS", False)
EP_ROWWISE_GET_NBLOCKS = os.environ.get("OLMOE3_HYBRID_EP_ROWWISE_GET_NBLOCKS")
EP_ROWWISE_PUT_NBLOCKS = os.environ.get("OLMOE3_HYBRID_EP_ROWWISE_PUT_NBLOCKS")
EP_ROWWISE_WEIGHTED_PUT_NBLOCKS = os.environ.get(
    "OLMOE3_HYBRID_EP_ROWWISE_WEIGHTED_PUT_NBLOCKS"
)
RANK_MICROBATCH_SEQUENCES = int(os.environ.get("OLMOE3_HYBRID_RANK_MICROBATCH_SEQUENCES", "16"))
LEARNING_RATE = float(os.environ.get("OLMOE3_HYBRID_LR", "1.6e-3"))
CHINCHILLA_MULTIPLE = float(os.environ.get("OLMOE3_HYBRID_CHINCHILLA_MULTIPLE", "1"))
MAX_TOKENS_OVERRIDE = os.environ.get("OLMOE3_HYBRID_MAX_TOKENS")
HARD_STOP_STEPS = int(os.environ.get("OLMOE3_HYBRID_HARD_STOP_STEPS", "0"))
USE_COMPILE = env_bool("OLMOE3_HYBRID_USE_COMPILE", True)
WANDB_ENABLED = env_bool("OLMOE3_HYBRID_WANDB", True)
CHECKPOINTS_ENABLED = env_bool("OLMOE3_HYBRID_CHECKPOINTS", True)
CHECKPOINT_WRITES_ENABLED = env_bool(
    "OLMOE3_HYBRID_CHECKPOINT_WRITES", CHECKPOINTS_ENABLED
)
EVALS_ENABLED = env_bool("OLMOE3_HYBRID_EVALS", False)
DP_USE_REDUCE_SCATTER = env_bool("OLMOE3_HYBRID_DP_USE_REDUCE_SCATTER", False)
DP_BUCKET_CAP_MB = os.environ.get("OLMOE3_HYBRID_DP_BUCKET_CAP_MB")
GDN2_DISABLE_RECOMPUTE = env_bool("OLMOE3_HYBRID_GDN2_DISABLE_RECOMPUTE", False)
GDN2_LOCALIZE_NONFINITE = env_bool("OLMOE3_HYBRID_GDN2_LOCALIZE_NONFINITE", False)
GDN2_LOCALIZE_START_STEP = int(
    os.environ.get("OLMOE3_HYBRID_GDN2_LOCALIZE_START_STEP", "0")
)
GDN2_LOCALIZE_END_STEP = int(
    os.environ.get("OLMOE3_HYBRID_GDN2_LOCALIZE_END_STEP", "0")
)
GDN2_LOCALIZE_DUMP_ROOT = os.environ.get(
    "OLMOE3_HYBRID_GDN2_LOCALIZE_DUMP_ROOT",
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/debug/gdn2-localizer",
)
GDN2_LOCALIZE_RUN_ID = os.environ.get(
    "OLMOE3_HYBRID_GDN2_LOCALIZE_RUN_ID", "gdn2-localizer"
)
EVAL_INTERVAL = int(os.environ.get("OLMOE3_HYBRID_EVAL_INTERVAL", "1000"))
EVAL_STEPS = int(os.environ.get("OLMOE3_HYBRID_EVAL_STEPS", "0"))
EVAL_TASK_SET = os.environ.get("OLMOE3_HYBRID_EVAL_TASK_SET", "hellaswag")
EVAL_ON_FINISH = env_bool("OLMOE3_HYBRID_EVAL_ON_FINISH", False)
EVAL_CHECKPOINT = os.environ.get("OLMOE3_HYBRID_EVAL_CHECKPOINT")
EVAL_BACKFILL = EVAL_CHECKPOINT is not None
SAVE_INTERVAL = int(os.environ.get("OLMOE3_HYBRID_SAVE_INTERVAL", "1000"))
EPHEMERAL_SAVE_INTERVAL = int(os.environ.get("OLMOE3_HYBRID_EPHEMERAL_SAVE_INTERVAL", "500"))
CHECKPOINT_REMOVAL = CheckpointRemovalStrategy(
    os.environ.get("OLMOE3_HYBRID_CHECKPOINT_REMOVAL", CheckpointRemovalStrategy.never.value)
)
SAVE_ROOT = os.environ.get(
    "OLMOE3_HYBRID_SAVE_ROOT",
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/pretraining",
)
WORK_DIR = os.environ.get(
    "OLMOE3_HYBRID_WORK_DIR",
    "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/dataset-cache",
)
DATA_ROOT = os.environ.get("OLMOE3_HYBRID_DATA_ROOT", "s3://ai2-llm")
EVAL_DATA_ROOT = os.environ.get("OLMOE3_HYBRID_EVAL_DATA_ROOT", "/weka/oe-training-default/ai2-llm")

if GDN2_DISABLE_RECOMPUTE and MODEL_VARIANT not in {
    *GDN2_275M_NOPE_SETTINGS,
    *GDN2_SCALE_NOPE_SETTINGS,
    "geometry_275m_gdn2_ev2_rope_gated",
    "geometry_275m_gdn2_ev2_rope_gated_1to1",
}:
    raise ValueError(
        "OLMOE3_HYBRID_GDN2_DISABLE_RECOMPUTE is only valid for the GDN2 variant"
    )


def model_config():
    if MODEL_VARIANT == "integration_wide_gdn_ev1":
        model = build_hybrid_model_config(MODEL_SIZE)
    elif MODEL_VARIANT == "geometry_275m_gdn_ev2":
        from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_275m import (
            build_geometry_matched_model_config,
        )

        if MODEL_SIZE != "275m":
            raise ValueError("The geometry_275m_gdn_ev2 variant only supports MODEL_SIZE=275m")
        model = build_geometry_matched_model_config("geometry_only")
    elif MODEL_VARIANT == "geometry_275m_gdn_ev2_nope":
        from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_275m import (
            build_geometry_matched_model_config,
        )

        if MODEL_SIZE != "275m":
            raise ValueError("The geometry_275m_gdn_ev2_nope variant only supports MODEL_SIZE=275m")
        model = build_geometry_matched_model_config("geometry_nope")
    elif MODEL_VARIANT == "geometry_275m_gdn_ev2_nope_gated":
        from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_275m import (
            build_geometry_matched_model_config,
        )

        if MODEL_SIZE != "275m":
            raise ValueError(
                "The geometry_275m_gdn_ev2_nope_gated variant only supports MODEL_SIZE=275m"
            )
        model = build_geometry_matched_model_config("geometry_nope_gated")
    elif MODEL_VARIANT == "geometry_275m_gdn_ev2_rope_gated":
        from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_275m import (
            build_geometry_matched_model_config,
        )

        if MODEL_SIZE != "275m":
            raise ValueError(
                "The geometry_275m_gdn_ev2_rope_gated variant only supports MODEL_SIZE=275m"
            )
        model = build_geometry_matched_model_config("geometry_rope_gated")
    elif MODEL_VARIANT == "geometry_275m_gdn2_ev2_rope_gated":
        from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_275m import (
            build_geometry_matched_gdn2_model_config,
        )

        if MODEL_SIZE != "275m":
            raise ValueError(
                "The geometry_275m_gdn2_ev2_rope_gated variant only supports MODEL_SIZE=275m"
            )
        model = build_geometry_matched_gdn2_model_config(
            disable_recompute=GDN2_DISABLE_RECOMPUTE
        )
    elif MODEL_VARIANT in GDN2_275M_NOPE_SETTINGS:
        from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_275m import (
            build_geometry_matched_gdn2_model_config,
        )

        if MODEL_SIZE != "275m":
            raise ValueError(
                f"The {MODEL_VARIANT} variant only supports MODEL_SIZE=275m"
            )
        expand_v, allow_neg_eigval = GDN2_275M_NOPE_SETTINGS[MODEL_VARIANT]
        model = build_geometry_matched_gdn2_model_config(
            rope=False,
            expand_v=expand_v,
            allow_neg_eigval=allow_neg_eigval,
            disable_recompute=GDN2_DISABLE_RECOMPUTE,
        )
    elif MODEL_VARIANT == KDA_275M_VARIANT:
        from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_275m import (
            build_geometry_matched_kda_model_config,
        )

        if MODEL_SIZE != "275m":
            raise ValueError(f"The {KDA_275M_VARIANT} variant only supports MODEL_SIZE=275m")
        model = build_geometry_matched_kda_model_config()
    elif MODEL_VARIANT == "geometry_275m_swa_rope_gated":
        from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_275m import (
            build_geometry_matched_swa_model_config,
        )

        if MODEL_SIZE != "275m":
            raise ValueError(
                "The geometry_275m_swa_rope_gated variant only supports MODEL_SIZE=275m"
            )
        model = build_geometry_matched_swa_model_config()
    elif MODEL_VARIANT in {
        "geometry_275m_gdn_ev2_rope_gated_1to1",
        "geometry_275m_gdn2_ev2_rope_gated_1to1",
        "geometry_275m_swa_rope_gated_1to1",
        "geometry_275m_swa_rope_gated_1to1_10l",
    }:
        from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_275m import (
            build_geometry_matched_one_to_one_model_config,
        )

        if MODEL_SIZE != "275m":
            raise ValueError(f"The {MODEL_VARIANT} variant only supports MODEL_SIZE=275m")
        mixer = {
            "geometry_275m_gdn_ev2_rope_gated_1to1": "gdn1",
            "geometry_275m_gdn2_ev2_rope_gated_1to1": "gdn2",
            "geometry_275m_swa_rope_gated_1to1": "swa",
            "geometry_275m_swa_rope_gated_1to1_10l": "swa",
        }[MODEL_VARIANT]
        model = build_geometry_matched_one_to_one_model_config(
            mixer,
            gdn2_disable_recompute=GDN2_DISABLE_RECOMPUTE,
            swa_n_layers=(
                10 if MODEL_VARIANT == "geometry_275m_swa_rope_gated_1to1_10l" else None
            ),
        )
    elif MODEL_VARIANT == "geometry_matched_gdn_ev2":
        from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_scale import (
            build_geometry_matched_scale_model_config,
        )

        model = build_geometry_matched_scale_model_config(MODEL_SIZE)
    elif MODEL_VARIANT == "geometry_matched_gdn_ev2_nope":
        from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_scale import (
            build_geometry_matched_scale_model_config,
        )

        model = build_geometry_matched_scale_model_config(MODEL_SIZE, rope=False)
    elif MODEL_VARIANT == "geometry_matched_gdn_ev2_nope_gated":
        from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_scale import (
            build_geometry_matched_scale_model_config,
        )

        model = build_geometry_matched_scale_model_config(
            MODEL_SIZE,
            rope=False,
            attention_gate=True,
        )
    elif MODEL_VARIANT == "geometry_matched_gdn_ev2_rope_gated":
        from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_scale import (
            build_geometry_matched_scale_model_config,
        )

        model = build_geometry_matched_scale_model_config(
            MODEL_SIZE,
            rope=True,
            attention_gate=True,
        )
    elif MODEL_VARIANT in GDN2_SCALE_NOPE_SETTINGS:
        from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_scale import (
            build_geometry_matched_scale_gdn2_model_config,
        )

        expand_v, allow_neg_eigval = GDN2_SCALE_NOPE_SETTINGS[MODEL_VARIANT]
        model = build_geometry_matched_scale_gdn2_model_config(
            MODEL_SIZE,
            rope=False,
            attention_gate=True,
            expand_v=expand_v,
            allow_neg_eigval=allow_neg_eigval,
            disable_recompute=GDN2_DISABLE_RECOMPUTE,
        )
    elif MODEL_VARIANT == KDA_SCALE_VARIANT:
        from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_scale import (
            build_geometry_matched_scale_kda_model_config,
        )

        model = build_geometry_matched_scale_kda_model_config(MODEL_SIZE)
    else:
        raise ValueError(f"Unknown model variant {MODEL_VARIANT!r}")
    if EP_SIZE > 1:
        if EP_USE_CODE_DEFAULTS and any(
            value is not None
            for value in (
                EP_ROWWISE_GET_NBLOCKS,
                EP_ROWWISE_PUT_NBLOCKS,
                EP_ROWWISE_WEIGHTED_PUT_NBLOCKS,
            )
        ):
            raise ValueError(
                "OLMOE3_HYBRID_EP_USE_CODE_DEFAULTS cannot be combined with rowwise "
                "block-count overrides"
            )
        for block in (model.block, *(model.block_overrides or {}).values()):
            if block.ep is None:
                continue
            if EP_USE_CODE_DEFAULTS:
                block.ep = ExpertParallelConfig()
                continue
            block.ep.path = EP_PATH
            if EP_ROWWISE_GET_NBLOCKS is not None:
                block.ep.rowwise_get_nblocks = int(EP_ROWWISE_GET_NBLOCKS)
            if EP_ROWWISE_PUT_NBLOCKS is not None:
                block.ep.rowwise_put_nblocks = int(EP_ROWWISE_PUT_NBLOCKS)
            if EP_ROWWISE_WEIGHTED_PUT_NBLOCKS is not None:
                block.ep.rowwise_weighted_put_nblocks = int(
                    EP_ROWWISE_WEIGHTED_PUT_NBLOCKS
                )
        model.validate()
    return model


def build_model_config(_common: CommonComponents):
    return model_config()


def max_tokens() -> int:
    if MAX_TOKENS_OVERRIDE is not None:
        return int(MAX_TOKENS_OVERRIDE)
    model = model_config()
    return int(
        Duration.chinchilla_tokens(
            CHINCHILLA_MULTIPLE,
            model_params=model.num_active_non_embedding_params,
        ).value
    )


def build_train_module_config(common: CommonComponents) -> OLMoDDPTrainModuleConfig:
    duration = max_tokens()
    warmup_tokens = max(
        GLOBAL_BATCH_SIZE,
        int((duration * 0.1 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE),
    )
    return OLMoDDPTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SEQUENCES * common.max_sequence_length,
        max_sequence_length=common.max_sequence_length,
        optim=OLMoDDPOptimizerConfig(
            lr=LEARNING_RATE,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=[
                        "*embedding_norm.weight",
                        "*q_norm.weight",
                        "*k_norm.weight",
                        "*o_norm.weight",
                        "*input_norm.weight",
                        "*lm_head.norm.weight",
                        "*attention_norm.weight",
                        "*feed_forward_norm.weight",
                    ],
                    opts={"weight_decay": 0.0},
                ),
                OptimGroupOverride(
                    params=["*routed_experts.w_up_gate", "*routed_experts.w_down"],
                    opts={"lr": LEARNING_RATE},
                ),
            ],
            compile=USE_COMPILE,
            dtype=DType.float32,
            sigma_factor=12,
            max_grad_norm=1.0,
            use_distributed=True,
        ),
        scheduler=ComposableScheduler(
            units=SchedulerUnits.tokens,
            stages=[
                ComposableSchedulerStage(
                    duration=warmup_tokens,
                    shape=ComposableSchedulerStageType.linear,
                    start_lr_fraction=0.0,
                    end_lr_fraction=1.0,
                ),
                ComposableSchedulerStage(
                    duration=max(duration - warmup_tokens, GLOBAL_BATCH_SIZE),
                    shape=ComposableSchedulerStageType.cosine,
                    end_lr_fraction=0.1,
                ),
            ],
        ),
        compile_model=USE_COMPILE,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
            bucket_cap_mb=None if DP_BUCKET_CAP_MB is None else int(DP_BUCKET_CAP_MB),
            use_reduce_scatter=DP_USE_REDUCE_SCATTER,
        ),
        ep_config=TransformerExpertParallelConfig(degree=EP_SIZE) if EP_SIZE > 1 else None,
        pp_config=None,
        tp_config=None,
        cp_config=None,
        ac_config=None,
        float8_config=None,
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )


def build_data_components(common: CommonComponents) -> DataComponents:
    dataset = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0925,
        tokenizer=common.tokenizer,
        mix_base_dir=DATA_ROOT,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        max_target_sequence_length=max(common.max_sequence_length, 8192),
        generate_doc_lengths=False,
        instance_filter_config=InstanceFilterConfig(
            repetition_max_period=13,
            repetition_min_period=1,
            repetition_max_count=32,
        ),
    )
    return DataComponents(
        dataset=dataset,
        data_loader=NumpyDataLoaderConfig(
            global_batch_size=common.global_batch_size,
            seed=34521,
            num_workers=8,
            # Eval-only backfills restore trainer state to recover the source
            # step/token counters. The training loader itself is never used,
            # so do not let an innocuous loader fingerprint difference block
            # final-checkpoint validation.
            ignore_fingerprint_mismatch=EVAL_BACKFILL,
        ),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    evals_enabled = EVALS_ENABLED or EVAL_BACKFILL
    if EVAL_TASK_SET == "hellaswag":
        downstream_tasks = ["hellaswag"]
    else:
        from olmo_core.eval.task_groups import TASK_GROUPS

        try:
            downstream_tasks = sorted(TASK_GROUPS[EVAL_TASK_SET])
        except KeyError as e:
            raise ValueError(f"Task set not recognized: {EVAL_TASK_SET}") from e
    eval_duration = Duration.steps(EVAL_STEPS) if EVAL_STEPS > 0 else Duration.epochs(1)
    trainer = TrainerConfig(
        save_folder=common.save_folder,
        save_overwrite=False,
        no_checkpoints=not CHECKPOINTS_ENABLED,
        checkpoints_to_eval=[EVAL_CHECKPOINT] if EVAL_CHECKPOINT is not None else None,
        checkpointer=CheckpointerConfig(
            save_thread_count=3,
            load_thread_count=8,
            throttle_uploads=True,
        ),
        metrics_collect_interval=1 if HARD_STOP_STEPS else 10,
        cancel_check_interval=10,
        async_bookkeeping=False,
        max_duration=Duration.tokens(max_tokens()),
        hard_stop=Duration.steps(HARD_STOP_STEPS) if HARD_STOP_STEPS else None,
    )
    if CHECKPOINTS_ENABLED:
        trainer = trainer.with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL,
                ephemeral_save_interval=EPHEMERAL_SAVE_INTERVAL,
                save_async=False,
                pre_train_checkpoint=False,
                remove=CHECKPOINT_REMOVAL,
                enabled=CHECKPOINT_WRITES_ENABLED,
            ),
        )
    geometry_variant = MODEL_VARIANT in {
        "geometry_275m_gdn_ev2",
        "geometry_275m_gdn_ev2_nope",
        "geometry_275m_gdn_ev2_nope_gated",
        "geometry_275m_gdn_ev2_rope_gated",
        *GDN2_275M_NOPE_SETTINGS,
        KDA_275M_VARIANT,
        "geometry_275m_gdn2_ev2_rope_gated",
        "geometry_275m_swa_rope_gated",
        "geometry_275m_gdn_ev2_rope_gated_1to1",
        "geometry_275m_gdn2_ev2_rope_gated_1to1",
        "geometry_275m_swa_rope_gated_1to1",
        "geometry_275m_swa_rope_gated_1to1_10l",
        "geometry_matched_gdn_ev2",
        "geometry_matched_gdn_ev2_nope",
        "geometry_matched_gdn_ev2_nope_gated",
        "geometry_matched_gdn_ev2_rope_gated",
        *GDN2_SCALE_NOPE_SETTINGS,
        KDA_SCALE_VARIANT,
    }
    if MODEL_VARIANT == "geometry_275m_gdn_ev2_rope_gated":
        variant_group = "olmoe3-275m-geometry-gdn-ev2-rope-gated"
    elif MODEL_VARIANT == "geometry_275m_gdn2_ev2_rope_gated":
        variant_group = "olmoe3-275m-geometry-gdn2-ev2-rope-gated"
    elif MODEL_VARIANT in GDN2_275M_NOPE_SETTINGS:
        expand_v, allow_neg_eigval = GDN2_275M_NOPE_SETTINGS[MODEL_VARIANT]
        eigval_tag = "neg" if allow_neg_eigval else "noneg"
        variant_group = (
            f"olmoe3-275m-geometry-gdn2-ev{expand_v:g}-{eigval_tag}-nope-gated"
        )
    elif MODEL_VARIANT == KDA_275M_VARIANT:
        variant_group = "olmoe3-275m-geometry-kda-ev1-noneg-nope-gated"
    elif MODEL_VARIANT == "geometry_275m_swa_rope_gated":
        variant_group = "olmoe3-275m-geometry-swa-rope-gated-throughput"
    elif MODEL_VARIANT == "geometry_275m_gdn_ev2_rope_gated_1to1":
        variant_group = "olmoe3-275m-geometry-gdn-ev2-rope-gated-1to1-throughput"
    elif MODEL_VARIANT == "geometry_275m_gdn2_ev2_rope_gated_1to1":
        variant_group = "olmoe3-275m-geometry-gdn2-ev2-rope-gated-1to1-throughput"
    elif MODEL_VARIANT in {
        "geometry_275m_swa_rope_gated_1to1",
        "geometry_275m_swa_rope_gated_1to1_10l",
    }:
        variant_group = "olmoe3-275m-geometry-swa-rope-gated-1to1-throughput"
    elif MODEL_VARIANT == "geometry_275m_gdn_ev2_nope_gated":
        variant_group = "olmoe3-275m-geometry-gdn-ev2-nope-gated"
    elif MODEL_VARIANT == "geometry_275m_gdn_ev2_nope":
        variant_group = "olmoe3-275m-geometry-gdn-ev2-nope"
    elif MODEL_VARIANT == "geometry_275m_gdn_ev2":
        variant_group = "olmoe3-275m-geometry-gdn-ev2"
    elif MODEL_VARIANT == "geometry_matched_gdn_ev2":
        variant_group = "olmoe3-geometry-matched-gdn-ev2-scale"
    elif MODEL_VARIANT == "geometry_matched_gdn_ev2_nope_gated":
        variant_group = "olmoe3-geometry-matched-gdn-ev2-nope-gated-scale"
    elif MODEL_VARIANT == "geometry_matched_gdn_ev2_rope_gated":
        variant_group = "olmoe3-geometry-matched-gdn-ev2-rope-gated-scale"
    elif MODEL_VARIANT == "geometry_matched_gdn_ev2_nope":
        variant_group = "olmoe3-geometry-matched-gdn-ev2-nope-scale"
    elif MODEL_VARIANT in GDN2_SCALE_NOPE_SETTINGS:
        expand_v, allow_neg_eigval = GDN2_SCALE_NOPE_SETTINGS[MODEL_VARIANT]
        eigval_tag = "neg" if allow_neg_eigval else "noneg"
        variant_group = (
            f"olmoe3-geometry-matched-gdn2-ev{expand_v:g}-{eigval_tag}-nope-gated-scale"
        )
    elif MODEL_VARIANT == KDA_SCALE_VARIANT:
        variant_group = "olmoe3-geometry-matched-kda-ev1-noneg-nope-gated-scale"
    else:
        variant_group = "olmoe3-integration-wide-hybrid-scale"
    if MODEL_VARIANT in {
        "geometry_275m_gdn_ev2_nope_gated",
        "geometry_275m_gdn_ev2_nope",
        *GDN2_275M_NOPE_SETTINGS,
        KDA_275M_VARIANT,
        KDA_SCALE_VARIANT,
        "geometry_matched_gdn_ev2_nope",
        "geometry_matched_gdn_ev2_nope_gated",
        *GDN2_SCALE_NOPE_SETTINGS,
    }:
        expand_v = (
            1.0
            if MODEL_VARIANT in {KDA_275M_VARIANT, KDA_SCALE_VARIANT}
            else {
                **GDN2_275M_NOPE_SETTINGS,
                **GDN2_SCALE_NOPE_SETTINGS,
            }.get(MODEL_VARIANT, (2.0, True))[0]
        )
        variant_tags = ["geometry-matched", f"expand-v-{expand_v:g}", "nope"]
        if MODEL_VARIANT in {
            "geometry_275m_gdn_ev2_nope_gated",
            *GDN2_275M_NOPE_SETTINGS,
            KDA_275M_VARIANT,
            KDA_SCALE_VARIANT,
            "geometry_matched_gdn_ev2_nope_gated",
            *GDN2_SCALE_NOPE_SETTINGS,
        }:
            variant_tags.append("attention-gate")
        if MODEL_VARIANT in {
            *GDN2_275M_NOPE_SETTINGS,
            *GDN2_SCALE_NOPE_SETTINGS,
        }:
            variant_tags.append("gdn2")
            gdn2_settings = {
                **GDN2_275M_NOPE_SETTINGS,
                **GDN2_SCALE_NOPE_SETTINGS,
            }
            if MODEL_VARIANT in gdn2_settings:
                allow_neg_eigval = gdn2_settings[MODEL_VARIANT][1]
                variant_tags.append(
                    "negative-eigenvalues" if allow_neg_eigval else "nonnegative-eigenvalues"
                )
            if GDN2_DISABLE_RECOMPUTE:
                variant_tags.append("gdn2-no-recompute")
        if MODEL_VARIANT in {KDA_275M_VARIANT, KDA_SCALE_VARIANT}:
            variant_tags.extend(["kda", "nonnegative-eigenvalues"])
    elif geometry_variant:
        variant_tags = ["geometry-matched", "expand-v-2", "rope"]
        if MODEL_VARIANT in {
            "geometry_275m_gdn_ev2_rope_gated",
            "geometry_275m_gdn2_ev2_rope_gated",
            "geometry_matched_gdn_ev2_rope_gated",
        }:
            variant_tags.append("attention-gate")
        if MODEL_VARIANT == "geometry_275m_gdn2_ev2_rope_gated":
            variant_tags.append("gdn2")
            if GDN2_DISABLE_RECOMPUTE:
                variant_tags.append("gdn2-no-recompute")
        if MODEL_VARIANT == "geometry_275m_swa_rope_gated":
            variant_tags = ["geometry-matched", "swa", "rope", "attention-gate"]
        if MODEL_VARIANT in {
            "geometry_275m_gdn_ev2_rope_gated_1to1",
            "geometry_275m_gdn2_ev2_rope_gated_1to1",
            "geometry_275m_swa_rope_gated_1to1",
            "geometry_275m_swa_rope_gated_1to1_10l",
        }:
            variant_tags.append("attention-1to1")
        if MODEL_VARIANT == "geometry_275m_gdn_ev2_rope_gated_1to1":
            variant_tags.append("attention-gate")
        if MODEL_VARIANT == "geometry_275m_gdn2_ev2_rope_gated_1to1":
            variant_tags.extend(["attention-gate", "gdn2"])
        if MODEL_VARIANT in {
            "geometry_275m_swa_rope_gated_1to1",
            "geometry_275m_swa_rope_gated_1to1_10l",
        }:
            variant_tags = [
                "geometry-matched",
                "swa",
                "rope",
                "attention-gate",
                "attention-1to1",
            ]
    else:
        variant_tags = ["integration-wide", "expand-v-1", "rope"]
    trainer = (
        trainer.with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("beaker", BeakerCallback())
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=EVAL_DATA_ROOT,
                    sequence_length=common.max_sequence_length,
                    tokenizer=common.tokenizer,
                    work_dir=common.work_dir,
                ),
                eval_interval=EVAL_INTERVAL,
                eval_duration=eval_duration,
                eval_on_finish=EVAL_ON_FINISH,
                enabled=evals_enabled,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=downstream_tasks,
                tokenizer=common.tokenizer,
                eval_interval=EVAL_INTERVAL,
                eval_duration=eval_duration,
                eval_on_finish=EVAL_ON_FINISH,
                enabled=evals_enabled,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                group=(f"{variant_group}-validation-backfills" if EVAL_BACKFILL else variant_group),
                project="jacobm-olmoe-ladder",
                entity="ai2-llm",
                enabled=WANDB_ENABLED,
                cancel_check_interval=10,
                tags=[
                    "pretraining",
                    MODEL_SIZE,
                    *variant_tags,
                    (
                        "hybrid"
                        if "gdn" in MODEL_VARIANT or "kda" in MODEL_VARIANT
                        else "swa-control"
                    ),
                    (
                        "kda"
                        if "kda" in MODEL_VARIANT
                        else ("gdn" if "gdn" in MODEL_VARIANT else "swa")
                    ),
                    "olmo-ddp",
                    f"ep{EP_SIZE}",
                    (
                        "validation-backfill"
                        if EVAL_BACKFILL
                        else ("smoke" if HARD_STOP_STEPS else "full-run")
                    ),
                ],
            ),
        )
    )
    if GDN2_LOCALIZE_NONFINITE:
        from scripts.train.jacobm_olmoe_ladder.v2.diagnostics.gdn2_nonfinite_localizer import (
            GDN2NonfiniteLocalizerCallback,
        )

        trainer = trainer.with_callback(
            "gdn2_nonfinite_localizer",
            GDN2NonfiniteLocalizerCallback(
                start_step=GDN2_LOCALIZE_START_STEP,
                end_step=GDN2_LOCALIZE_END_STEP,
                dump_root=GDN2_LOCALIZE_DUMP_ROOT,
                run_id=GDN2_LOCALIZE_RUN_ID,
            ),
        )
    return trainer


def build_local_common_components(
    cli_context: CliContext,
    *,
    tokenizer: TokenizerConfig,
    global_batch_size: int,
    max_sequence_length: int,
    **_kwargs: Any,
) -> CommonComponents:
    if cli_context.cluster not in {"local", "localhost"}:
        raise ValueError("Launch this script inside Beaker and pass cluster='local'")
    return CommonComponents(
        run_name=cli_context.run_name,
        root_dir=DATA_ROOT,
        work_dir=WORK_DIR,
        save_folder=os.path.join(SAVE_ROOT, cli_context.run_name),
        launch=None,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        global_batch_size=global_batch_size,
    )


def finalize_config(config: ExperimentConfig) -> None:
    if MODEL_SIZE not in MODEL_SIZES:
        raise ValueError(f"Unknown model size {MODEL_SIZE!r}; choose one of {MODEL_SIZES}")
    if HARD_STOP_STEPS and not CHECKPOINTS_ENABLED:
        log.info("Smoke checkpointing is disabled; no final hard-stop checkpoint will be written")
    if CHECKPOINT_WRITES_ENABLED and not CHECKPOINTS_ENABLED:
        raise ValueError(
            "OLMOE3_HYBRID_CHECKPOINT_WRITES=1 requires OLMOE3_HYBRID_CHECKPOINTS=1"
        )
    if CHECKPOINTS_ENABLED and not CHECKPOINT_WRITES_ENABLED:
        log.info("Checkpoint loading is enabled, but all checkpoint writes are disabled")
    if GDN2_LOCALIZE_NONFINITE:
        if MODEL_VARIANT not in {*GDN2_275M_NOPE_SETTINGS, *GDN2_SCALE_NOPE_SETTINGS}:
            raise ValueError("GDN2 non-finite localization requires a GDN2 model variant")
        if not (0 < GDN2_LOCALIZE_START_STEP <= GDN2_LOCALIZE_END_STEP):
            raise ValueError(
                "GDN2 localization requires a positive, ordered start/end step window"
            )
        if CHECKPOINT_WRITES_ENABLED or WANDB_ENABLED:
            raise ValueError(
                "GDN2 localization must disable checkpoint writes and W&B"
            )
    if EVAL_BACKFILL:
        if CHECKPOINTS_ENABLED:
            raise ValueError("Eval-only backfills must set OLMOE3_HYBRID_CHECKPOINTS=0")
        if not EVALS_ENABLED:
            log.info("Enabling evaluator callbacks because an eval checkpoint was supplied")
        assert EVAL_CHECKPOINT is not None
        if not Path(EVAL_CHECKPOINT).is_dir():
            raise ValueError(f"Eval checkpoint does not exist: {EVAL_CHECKPOINT}")
    if WORLD_SIZE < 1 or EP_SIZE < 1 or WORLD_SIZE % EP_SIZE:
        raise ValueError(f"EP size {EP_SIZE} must divide world size {WORLD_SIZE}")
    if NUM_NODES < 1:
        raise ValueError(f"NUM_NODES must be positive, got {NUM_NODES}")
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) != WORLD_SIZE:
        raise ValueError(
            f"Configured world size {WORLD_SIZE} does not match torchrun world size "
            f"{os.environ['WORLD_SIZE']}"
        )
    if GLOBAL_BATCH_SIZE % SEQUENCE_LENGTH:
        raise ValueError("Global batch size must contain a whole number of sequences")
    global_sequences = GLOBAL_BATCH_SIZE // SEQUENCE_LENGTH
    data_dp_degree = WORLD_SIZE
    if global_sequences % data_dp_degree:
        raise ValueError(
            f"Global sequence batch {global_sequences} is not divisible by "
            f"the data-parallel world size {data_dp_degree}"
        )
    rank_sequences = global_sequences // data_dp_degree
    effective_rank_microbatch = min(rank_sequences, RANK_MICROBATCH_SEQUENCES)
    if rank_sequences % effective_rank_microbatch:
        raise ValueError(
            f"Rank sequence batch {rank_sequences} is not divisible by the effective "
            f"rank microbatch {effective_rank_microbatch}"
        )
    base = load_wide_model_config(MODEL_SIZE)
    delta_fraction = (
        config.model.num_active_params - base.num_active_params
    ) / base.num_active_params
    if MODEL_VARIANT in {
        *GDN2_275M_NOPE_SETTINGS,
        *GDN2_SCALE_NOPE_SETTINGS,
        "geometry_275m_gdn2_ev2_rope_gated",
        "geometry_275m_gdn2_ev2_rope_gated_1to1",
    }:
        max_active_parameter_delta_fraction = GDN2_MAX_ACTIVE_PARAMETER_DELTA_FRACTIONS[
            MODEL_SIZE
        ]
    elif MODEL_VARIANT in {
        "geometry_matched_gdn_ev2_nope_gated",
        "geometry_matched_gdn_ev2_rope_gated",
    }:
        max_active_parameter_delta_fraction = GATED_MAX_ACTIVE_PARAMETER_DELTA_FRACTION
    else:
        max_active_parameter_delta_fraction = MAX_ACTIVE_PARAMETER_DELTA_FRACTION
    if abs(delta_fraction) > max_active_parameter_delta_fraction:
        raise ValueError(
            "Hybrid active-parameter delta is too large: "
            f"{delta_fraction:.4%} > {max_active_parameter_delta_fraction:.4%}"
        )
    log.info(
        "Hybrid config: variant=%s size=%s active=%s active_non_embedding=%s total=%s "
        "active_delta=%+.4f%% tokens=%s "
        "global_sequences=%s world=%s EP=%s data_DP=%s EP_DP=%s rank_sequences=%s "
        "rank_microbatch_cap=%s effective_rank_microbatch=%s grad_accum=%s lr=%s "
        "hard_stop_steps=%s",
        MODEL_VARIANT,
        MODEL_SIZE,
        f"{config.model.num_active_params:,}",
        f"{config.model.num_active_non_embedding_params:,}",
        f"{config.model.num_params:,}",
        100 * delta_fraction,
        f"{max_tokens():,}",
        global_sequences,
        WORLD_SIZE,
        EP_SIZE,
        data_dp_degree,
        WORLD_SIZE // EP_SIZE,
        rank_sequences,
        RANK_MICROBATCH_SEQUENCES,
        effective_rank_microbatch,
        rank_sequences // effective_rank_microbatch,
        LEARNING_RATE,
        HARD_STOP_STEPS or "off",
    )


def make_config(cli_context: CliContext) -> ExperimentConfig:
    builder = partial(
        build_config,
        common_config_builder=build_local_common_components,
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        tokenizer=TokenizerConfig.dolma2(),
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        num_nodes=NUM_NODES,
        include_default_evals=False,
        finalize_config=finalize_config,
    )
    return cast(ExperimentConfig, builder(cli_context))


if __name__ == "__main__":
    main(config_builder=make_config)
