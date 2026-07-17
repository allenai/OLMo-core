"""Upstream-main FSDP baseline for the MoE infrastructure tech report.

This intentionally uses the generic MoE and FSDP2 implementation available on
the upstream ``main`` branch. Its macro architecture mirrors the development
benchmark, while its sparse kernels and distributed implementation remain
native to main.
"""

from __future__ import annotations

import math
import os
import socket
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, cast

# This checkout lives beside another editable OLMo-core checkout. Ensure this
# entry point imports the upstream-main source tree it belongs to, regardless
# of which checkout was installed most recently in the environment.
MAIN_SRC_DIR = Path(__file__).resolve().parents[3]
MAIN_REPO_DIR = MAIN_SRC_DIR.parent
if str(MAIN_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_SRC_DIR))
# Upstream main constructs Beaker/Gantry metadata even for a direct ``train``
# invocation. Gantry resolves the repository from the current directory, so
# make this entry point independent of the caller's working directory.
os.chdir(MAIN_REPO_DIR)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _set_default_triton_cache_dir() -> None:
    if os.environ.get("TRITON_CACHE_DIR"):
        return
    local_rank = os.environ.get("LOCAL_RANK", "0")
    host = socket.gethostname().split(".")[0]
    cache_dir = Path("/tmp/olmo-triton-cache") / "tech-report-main" / host / local_rank
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)


PROFILE = _env_flag("TECH_REPORT_PROFILE", False)
if not PROFILE:
    os.environ.setdefault("NVTX_DISABLE", "1")
_set_default_triton_cache_dir()

import torch  # noqa: E402

from olmo_core.config import DType  # noqa: E402
from olmo_core.data import DataMix, NumpyDataLoaderConfig, NumpyFSLDatasetConfig  # noqa: E402
from olmo_core.distributed.parallel import DataParallelType  # noqa: E402
from olmo_core.distributed.utils import get_rank  # noqa: E402
from olmo_core.internal.experiment import (  # noqa: E402
    CommonComponents,
    DataComponents,
    ExperimentConfig,
    build_config,
    main,
)
from olmo_core.nn.attention.backend import AttentionBackendName  # noqa: E402
from olmo_core.nn.feed_forward import FeedForwardConfig  # noqa: E402
from olmo_core.nn.moe import (  # noqa: E402
    MoEConfig,
    MoELoadBalancingLossGranularity,
    MoERouterConfig,
    MoERouterGatingFunction,
    MoEType,
)
from olmo_core.nn.transformer import (  # noqa: E402
    TransformerBlockType,
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerType,
)
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig  # noqa: E402
from olmo_core.train import Duration, TrainerConfig  # noqa: E402
from olmo_core.train.callbacks import Callback, SpeedMonitorCallback, WandBCallback  # noqa: E402
from olmo_core.train.train_module import (  # noqa: E402
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

torch.set_float32_matmul_precision("high")

SEED = 2026
SEQUENCE_LENGTH = int(os.environ.get("TECH_REPORT_SEQUENCE_LENGTH", 8192))
GLOBAL_BATCH_SIZE = int(os.environ.get("TECH_REPORT_GLOBAL_BATCH_SIZE", 4 * 1024 * 1024))
RANK_MICROBATCH_SEQUENCES = int(os.environ.get("TECH_REPORT_RANK_MICROBATCH_SEQUENCES", 2))
MAX_STEPS = int(os.environ.get("TECH_REPORT_MAX_STEPS", 100))

SUPPORTED_NUM_EXPERTS = (8, 32, 48, 64, 128)
SUPPORTED_GLOBAL_BATCH_SIZES = tuple(
    kib_tokens * 1024
    for kib_tokens in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
)
if GLOBAL_BATCH_SIZE not in SUPPORTED_GLOBAL_BATCH_SIZES:
    raise ValueError(
        "TECH_REPORT_GLOBAL_BATCH_SIZE must be one of "
        f"{SUPPORTED_GLOBAL_BATCH_SIZES}; got {GLOBAL_BATCH_SIZE}"
    )

D_MODEL = int(os.environ.get("TECH_REPORT_D_MODEL", 4096))
HEAD_DIM = 128
NUM_HEADS = D_MODEL // HEAD_DIM
NUM_KV_HEADS = NUM_HEADS // 4
NUM_LAYERS = int(os.environ.get("TECH_REPORT_NUM_LAYERS", 8))
NUM_EXPERTS = int(os.environ.get("TECH_REPORT_NUM_EXPERTS", 64))
if NUM_EXPERTS not in SUPPORTED_NUM_EXPERTS:
    raise ValueError(
        f"TECH_REPORT_NUM_EXPERTS must be one of {SUPPORTED_NUM_EXPERTS}; got {NUM_EXPERTS}"
    )
TOP_K = int(os.environ.get("TECH_REPORT_TOP_K", 4))
EXPERT_HIDDEN_SIZE = int(os.environ.get("TECH_REPORT_EXPERT_HIDDEN_SIZE", 4096))
SHARED_HIDDEN_SIZE = int(os.environ.get("TECH_REPORT_SHARED_HIDDEN_SIZE", 4096))
DENSE_FIRST_HIDDEN_SIZE = TOP_K * EXPERT_HIDDEN_SIZE + SHARED_HIDDEN_SIZE
CAPACITY_FACTOR = float(os.environ.get("TECH_REPORT_CAPACITY_FACTOR", 1.25))


def benchmark_num_flops_per_token(vocab_size: int, seq_len: int = SEQUENCE_LENGTH) -> int:
    """Canonical idealized FLOPs/token shared with the development benchmark."""
    projection_flops = 6 * (
        D_MODEL * D_MODEL
        + 2 * D_MODEL * NUM_KV_HEADS * HEAD_DIM
        + D_MODEL * D_MODEL
    )

    attention_positions = seq_len * (seq_len + 1) // 2
    attention_flops = projection_flops + (
        12 * NUM_HEADS * HEAD_DIM * attention_positions // seq_len
    )

    flops = attention_flops + 18 * D_MODEL * DENSE_FIRST_HIDDEN_SIZE
    for _ in range(1, NUM_LAYERS):
        flops += attention_flops
        flops += 6 * D_MODEL * NUM_EXPERTS
        flops += 18 * D_MODEL * EXPERT_HIDDEN_SIZE * TOP_K
        flops += 18 * D_MODEL * SHARED_HIDDEN_SIZE
    flops += 6 * D_MODEL * vocab_size
    return flops


def _make_first_block_dense(block):
    block.name = TransformerBlockType.reordered_norm
    block.feed_forward_moe = None
    block.feed_forward = FeedForwardConfig(
        hidden_size=DENSE_FIRST_HIDDEN_SIZE,
        bias=False,
        dtype=DType.float32,
    )
    return block


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.llama_like(
        init_seed=SEED,
        d_model=D_MODEL,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=NUM_LAYERS,
        n_heads=NUM_HEADS,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        name=TransformerType.moe,
        block_name=TransformerBlockType.moe_hybrid_reordered_norm,
        block_mods={0: _make_first_block_dense},
        qk_norm=True,
        use_head_qk_norm=True,
        rope_theta=500_000,
        rope_full_precision=True,
        layer_norm_eps=1e-6,
        attn_backend=AttentionBackendName.flash_4,
        feed_forward_moe=MoEConfig(
            # Preserve upstream main's capacity-limited ParallelMLP path.
            name=MoEType.default,
            num_experts=NUM_EXPERTS,
            hidden_size=EXPERT_HIDDEN_SIZE,
            capacity_factor=CAPACITY_FACTOR,
            router=MoERouterConfig(
                top_k=TOP_K,
                gating_function=MoERouterGatingFunction.softmax,
                normalize_expert_weights=1.0,
                uniform_expert_assignment=False,
                random_expert_assignment=True,
                dtype=DType.float32,
            ),
            lb_loss_weight=0.015,
            z_loss_weight=1e-4,
            lb_loss_granularity=MoELoadBalancingLossGranularity.local_batch,
            scale_loss_by_num_layers=False,
            dtype=DType.float32,
        ),
        feed_forward=FeedForwardConfig(
            hidden_size=SHARED_HIDDEN_SIZE,
            bias=False,
            dtype=DType.float32,
        ),
        embed_scale=math.sqrt(D_MODEL),
        init_std=0.01,
        dtype=DType.float32,
    )


def build_data_components(
    common: CommonComponents,
    intra_document_masking: bool = False,
    include_instance_filter: bool = False,
) -> DataComponents:
    del include_instance_filter
    dataset = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0925,
        tokenizer=common.tokenizer,
        mix_base_dir=os.environ.get("TECH_REPORT_DATA_ROOT", "s3://ai2-llm"),
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        max_target_sequence_length=max(common.max_sequence_length, 8192),
        generate_doc_lengths=intra_document_masking,
        instance_filter_config=None,
    )
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size,
        seed=34521,
        num_workers=8,
        prefetch_factor=8,
    )
    return DataComponents(dataset=dataset, data_loader=data_loader)


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SEQUENCES * SEQUENCE_LENGTH,
        max_sequence_length=common.max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=3e-4,
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
            prefetch_factor=0,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        ep_config=None,
        pp_config=None,
        ac_config=None,
        float8_config=None,
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=10),
    )


@dataclass
class NvidiaProfilerCallback(Callback):
    start: int = 20
    end: int = 25
    enabled: bool = False
    profile_ranks: tuple[int, ...] = (0,)
    _nvtx_context: Optional[object] = field(default=None, init=False, repr=False)

    def pre_load_batch(self):
        if self.enabled and get_rank() in self.profile_ranks and self.step == self.start:
            torch.cuda.cudart().cudaProfilerStart()
            self._nvtx_context = torch.autograd.profiler.emit_nvtx(record_shapes=True)
            self._nvtx_context.__enter__()

    def post_train_batch(self):
        if self.enabled and get_rank() in self.profile_ranks and self.step == self.end:
            self._stop()

    def close(self):
        self._stop()

    def _stop(self):
        if self._nvtx_context is not None:
            self._nvtx_context.__exit__(None, None, None)
            self._nvtx_context = None
            torch.cuda.cudart().cudaProfilerStop()


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    wandb_enabled = _env_flag("TECH_REPORT_WANDB", bool(os.environ.get("WANDB_API_KEY")))
    return (
        TrainerConfig(
            save_folder=f"/workspace/checkpoint/tech_report-main/{common.run_name}",
            save_overwrite=True,
            no_checkpoints=True,
            no_evals=True,
            metrics_collect_interval=5,
            cancel_check_interval=10,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                group=os.environ.get("TECH_REPORT_WANDB_GROUP", "moe-fsdp-main-vs-ddp-dev"),
                entity="ai2-llm",
                project=os.environ.get("TECH_REPORT_WANDB_PROJECT", "olmoe-tech-report"),
                enabled=wandb_enabled,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "nvidia_profiler",
            NvidiaProfilerCallback(enabled=PROFILE,
                                   start=21,
                                   end=24,
                                   ),
        )
        .with_callback(
            "speed_monitor",
            SpeedMonitorCallback(
                num_flops_per_token=benchmark_num_flops_per_token(
                    common.tokenizer.padded_vocab_size(), common.max_sequence_length
                )
            ),
        )
    )


def finalize_config(config: ExperimentConfig) -> None:
    active_b = config.model.num_active_params / 1e9
    total_b = config.model.num_params / 1e9
    wandb = cast(WandBCallback, config.trainer.callbacks["wandb"])
    wandb.name = f"{wandb.name}_{active_b:.3f}A-{total_b:.3f}T"


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        finalize_config=finalize_config,
        include_instance_filter=False,
        include_default_evals=False,
        flight_recorder=True,
    )
    main(config_builder=config_builder)
