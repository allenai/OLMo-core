"""
Qwen3.5/3.6 MoE random-init pretraining recipe.

Examples:
  python src/scripts/train/Qwen3-moe-dev-260612.py dry_run qwen-debug ai2/jupiter
  torchrun --nproc-per-node=1 src/scripts/train/Qwen3-moe-dev-260612.py train qwen-debug ai2/jupiter
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Any, Mapping, cast

# Keep this before any olmo_core imports. Several modules import nvtx at import
# time, and NVTX_DISABLE only works if it is set before nvtx is imported.
if os.getenv("QWEN_NV_PROFILE", "0").strip().lower() not in {"1", "true", "yes", "on"}:
    os.environ["NVTX_DISABLE"] = "1"

import torch
from transformers import AutoConfig

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.experiment import (
    CommonComponents,
    DataComponents,
    ExperimentConfig,
    build_config,
    main,
)
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.layer_norm import LayerNormType
from olmo_core.nn.moe.v2.qwen import (
    build_debug_qwen3_moe_config,
    build_qwen3_moe_config,
    build_qwen3_moe_config_from_hf_config,
)
from olmo_core.optim import CosWithWarmup, OLMoDDPOptimizerConfig, OptimGroupOverride
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, WandBCallback
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)

log = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else int(raw)


def _env_int_optional(name: str) -> int | None:
    raw = os.getenv(name)
    return None if raw is None or raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else float(raw)


def _env_list(name: str) -> list[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _text_config(hf_config: Mapping[str, Any]) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], hf_config.get("text_config", hf_config))


def _is_qwen35_text_config(text_config: Mapping[str, Any]) -> bool:
    return "linear_num_key_heads" in text_config


def _layer_overrides_for_limit(text_config: Mapping[str, Any], max_layers: int | None) -> dict[str, Any]:
    if max_layers is None:
        return {}
    total_layers = int(text_config["num_hidden_layers"])
    if max_layers < 1 or max_layers > total_layers:
        raise ValueError(f"QWEN_MAX_LAYERS must be in [1, {total_layers}], got {max_layers}")
    overrides: dict[str, Any] = {"n_layers": max_layers}
    if "layer_types" in text_config:
        overrides["layer_types"] = tuple(text_config["layer_types"][:max_layers])
    else:
        overrides["layer_types"] = ("full_attention",) * max_layers
    return overrides


def _qwen_tokenizer_config(identifier: str) -> TokenizerConfig:
    hf_config = AutoConfig.from_pretrained(identifier, trust_remote_code=False).to_dict()
    text_config = _text_config(hf_config)
    eos_token_id = text_config.get("eos_token_id")
    if eos_token_id is None:
        raise ValueError(f"{identifier} text config is missing eos_token_id")
    pad_token_id = text_config.get("pad_token_id")
    if pad_token_id is None:
        pad_token_id = eos_token_id
    return TokenizerConfig(
        vocab_size=int(text_config["vocab_size"]),
        eos_token_id=int(eos_token_id),
        pad_token_id=int(pad_token_id),
        bos_token_id=text_config.get("bos_token_id"),
        identifier=identifier,
    )


def _tokenizer_config() -> TokenizerConfig:
    identifier = os.getenv("QWEN_TOKENIZER_ID", MODEL_ID).strip()
    if identifier in {"dolma2", "allenai/dolma2-tokenizer"}:
        return TokenizerConfig.dolma2()
    return _qwen_tokenizer_config(identifier)


MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3.6-35B-A3B")
MODEL_SCALE = os.getenv("QWEN_MODEL_SCALE", "debug").strip().lower()
TOKENIZER_CONFIG = _tokenizer_config()

SEQUENCE_LENGTH = _env_int("QWEN_SEQUENCE_LENGTH", 512 if MODEL_SCALE == "debug" else 8192)
MAX_STEPS = _env_int("QWEN_MAX_STEPS", 10)
GLOBAL_BATCH_SEQS = _env_int("QWEN_GLOBAL_BATCH_SEQS", 8 if MODEL_SCALE == "debug" else 64)
MICRO_BATCH_SEQS = _env_int("QWEN_MICRO_BATCH_SEQS", 1)
GLOBAL_BATCH_SIZE = GLOBAL_BATCH_SEQS * SEQUENCE_LENGTH

LR = _env_float("QWEN_LR", 3e-4)
WARMUP_STEPS = _env_int("QWEN_WARMUP_STEPS", min(100, max(1, MAX_STEPS // 10)))
SAVE_INTERVAL = _env_int("QWEN_SAVE_INTERVAL", max(100, MAX_STEPS))
SEED = _env_int("QWEN_SEED", 2026)

DTYPE = DType(os.getenv("QWEN_DTYPE", DType.bfloat16.value))
ATTENTION_BACKEND = AttentionBackendName(
    os.getenv("QWEN_ATTENTION_BACKEND", AttentionBackendName.torch.value)
)
USE_COMPILE = _env_bool("QWEN_USE_COMPILE", False)
PER_LAYER_RECOMPUTE = _env_bool("QWEN_PER_LAYER_RECOMPUTE", False)
MAX_LAYERS = _env_int_optional("QWEN_MAX_LAYERS")
EP_DIM = _env_int("QWEN_EP_DIM", 1)
USE_EP_NO_SYNC = EP_DIM > 1 and _env_bool("QWEN_USE_EP_NO_SYNC", True)
USE_ROWWISE_A2A = EP_DIM > 1 and _env_bool("QWEN_USE_ROWWISE_A2A", True)

DATA_MIX = os.getenv("QWEN_DATA_MIX", DataMix.OLMo_mix_0925.value)
DATA_MIX_BASE_DIR = os.getenv("QWEN_MIX_BASE_DIR", "s3://ai2-llm")
DEFAULT_DATA_PATH = (
    "/workspace/tasks/june12/scratch/qwen3_6_loss/"
    "qwen36_retokenized_olmo_mix_0925_education_jobs_first1m.uint32.npy"
)
DATA_PATHS = (
    _env_list("QWEN_DATA_PATHS")
    if os.getenv("QWEN_DATA_PATHS") is not None
    else [DEFAULT_DATA_PATH]
)
DATA_NUM_WORKERS = _env_int("QWEN_DATA_NUM_WORKERS", 0)
LOAD_PATH = os.getenv("QWEN_LOAD_PATH") or None

torch.set_float32_matmul_precision("high")


def build_model_config(common: CommonComponents):
    hf_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=False).to_dict()
    text_config = _text_config(hf_config)
    if MODEL_SCALE == "full":
        return build_qwen3_moe_config_from_hf_config(
            hf_config,
            vocab_size=common.tokenizer.padded_vocab_size(),
            dtype=DTYPE,
            init_seed=SEED,
            attention_backend=ATTENTION_BACKEND,
            compile_friendly_recompute=PER_LAYER_RECOMPUTE,
            use_ep_no_sync=USE_EP_NO_SYNC,
            use_rowwise_all_to_all=USE_ROWWISE_A2A,
            **_layer_overrides_for_limit(text_config, MAX_LAYERS),
        )

    if MODEL_SCALE != "debug":
        raise ValueError("QWEN_MODEL_SCALE must be 'debug' or 'full'")

    debug_layers = _env_int("QWEN_DEBUG_LAYERS", 4)
    debug_d_model = _env_int("QWEN_DEBUG_D_MODEL", 256)
    debug_num_experts = _env_int("QWEN_DEBUG_NUM_EXPERTS", 8)
    debug_top_k = _env_int("QWEN_DEBUG_TOP_K", 2)
    debug_moe_hidden = _env_int("QWEN_DEBUG_MOE_HIDDEN", 128)

    if not _is_qwen35_text_config(text_config):
        debug_num_heads = _env_int("QWEN_DEBUG_NUM_HEADS", 4)
        debug_num_kv_heads = _env_int("QWEN_DEBUG_NUM_KV_HEADS", 1)
        debug_head_dim = _env_int("QWEN_DEBUG_HEAD_DIM", max(1, debug_d_model // 2))
        return build_qwen3_moe_config(
            vocab_size=common.tokenizer.padded_vocab_size(),
            d_model=debug_d_model,
            n_layers=debug_layers,
            num_attention_heads=debug_num_heads,
            num_key_value_heads=debug_num_kv_heads,
            attention_head_dim=debug_head_dim,
            attention_rotary_dim=debug_head_dim,
            rope_theta=int(text_config.get("rope_theta", 1_000_000)),
            linear_num_key_heads=None,
            linear_num_value_heads=None,
            linear_key_head_dim=None,
            linear_value_head_dim=None,
            linear_conv_kernel_dim=None,
            num_experts=debug_num_experts,
            num_experts_per_tok=debug_top_k,
            moe_intermediate_size=debug_moe_hidden,
            shared_expert_intermediate_size=None,
            rms_norm_eps=float(text_config.get("rms_norm_eps", 1e-6)),
            layer_types=("full_attention",) * debug_layers,
            attention_output_gate=False,
            norm_type=LayerNormType.rms,
            dtype=DTYPE,
            init_seed=SEED,
            attention_backend=ATTENTION_BACKEND,
            compile_friendly_recompute=PER_LAYER_RECOMPUTE,
            use_ep_no_sync=USE_EP_NO_SYNC,
            use_rowwise_all_to_all=USE_ROWWISE_A2A,
        )

    return build_debug_qwen3_moe_config(
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=debug_layers,
        d_model=debug_d_model,
        num_experts=debug_num_experts,
        num_experts_per_tok=debug_top_k,
        moe_intermediate_size=debug_moe_hidden,
        shared_expert_intermediate_size=_env_int("QWEN_DEBUG_SHARED_HIDDEN", 128),
        dtype=DTYPE,
        init_seed=SEED,
        attention_backend=ATTENTION_BACKEND,
        compile_friendly_recompute=PER_LAYER_RECOMPUTE,
        use_ep_no_sync=USE_EP_NO_SYNC,
        use_rowwise_all_to_all=USE_ROWWISE_A2A,
    )


def build_train_module_config(common: CommonComponents) -> OLMoDDPTrainModuleConfig:
    return OLMoDDPTrainModuleConfig(
        rank_microbatch_size=MICRO_BATCH_SEQS * SEQUENCE_LENGTH,
        max_sequence_length=common.max_sequence_length,
        optim=OLMoDDPOptimizerConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=[
                        "*embeddings.weight",
                        "*lm_head.norm.weight",
                        "*attention_norm.weight",
                        "*feed_forward_norm.weight",
                        "*q_norm.weight",
                        "*k_norm.weight",
                        "*o_norm.weight",
                    ],
                    opts=dict(weight_decay=0.0),
                )
            ],
            compile=USE_COMPILE,
            dtype=DType.float32,
            sigma_factor=12,
            use_distributed=True,
        ),
        compile_model=USE_COMPILE,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
        ),
        ep_config=TransformerExpertParallelConfig(degree=EP_DIM) if EP_DIM > 1 else None,
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup=WARMUP_STEPS, t_max=MAX_STEPS, alpha_f=0.1),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            save_overwrite=_env_bool("QWEN_SAVE_OVERWRITE", True),
            load_path=LOAD_PATH,
            load_trainer_state=False if LOAD_PATH is not None else None,
            load_optim_state=False if LOAD_PATH is not None else None,
            checkpointer=CheckpointerConfig(
                save_thread_count=1,
                load_thread_count=1,
                throttle_uploads=False,
            ),
            metrics_collect_interval=1,
            cancel_check_interval=10,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL,
                ephemeral_save_interval=None,
                save_async=False,
                pre_train_checkpoint=False,
                enabled=_env_bool("QWEN_CHECKPOINTER", False),
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity=os.getenv("QWEN_WANDB_ENTITY", "ai2-llm"),
                project=os.getenv("QWEN_WANDB_PROJECT", "qwen-moe-dev"),
                enabled=_env_bool("QWEN_WANDB", False),
                cancel_check_interval=10,
            ),
        )
    )


def build_data_components(
    common: CommonComponents,
    intra_document_masking: bool = False,
    include_instance_filter: bool = True,
) -> DataComponents:
    dataset_kwargs = dict(
        tokenizer=common.tokenizer,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        max_target_sequence_length=max(common.max_sequence_length, 8192),
        generate_doc_lengths=intra_document_masking,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13,
            repetition_min_period=1,
            repetition_max_count=32,
        ),
    )
    if DATA_PATHS:
        dataset_config = NumpyFSLDatasetConfig(paths=DATA_PATHS, **dataset_kwargs)
    else:
        dataset_config = NumpyFSLDatasetConfig.from_data_mix(
            DataMix(DATA_MIX),
            mix_base_dir=DATA_MIX_BASE_DIR,
            **dataset_kwargs,
        )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size,
        seed=34521,
        num_workers=DATA_NUM_WORKERS,
    )
    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def finalize_config(config: ExperimentConfig) -> None:
    total_params_in_b = config.model.num_params / 1_000_000_000
    active_params_in_b = config.model.num_active_params / 1_000_000_000
    log.info("Total params: %.3fB, active params: %.3fB", total_params_in_b, active_params_in_b)

    wandb_cb = cast(WandBCallback, config.trainer.callbacks["wandb"])
    if wandb_cb.enabled:
        wandb_cb.name = f"{wandb_cb.name}_{active_params_in_b:.3f}@{total_params_in_b:.3f}B"


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        tokenizer=TOKENIZER_CONFIG,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
        include_instance_filter=True,
        finalize_config=finalize_config,
    )

    main(config_builder=config_builder)
