"""External torchtitan config registry for the contradiction-1M SFT run.

Invocation:
    cd /scratch/users/prasann/torchtitan && \\
    NGPU=4 MODULE=scripts.lib.titan_ext \\
    CONFIG=qwen3_4b_contra_1m_smoke ./run_train.sh \\
        --... CLI overrides ...

The `--module scripts.lib.titan_ext` form imports
`scripts.lib.titan_ext.config_registry` (per torchtitan/config/manager.py).
Requires PYTHONPATH to include /scratch/users/prasann/corpus-reasoning.
"""

from dataclasses import field

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.models.common import FlexAttention
from torchtitan.models.qwen3 import model_registry as qwen3_model_registry
from torchtitan.tools.utils import has_cuda_capability
from torchtitan.trainer import Trainer
from copy import deepcopy

from corpus_reasoning.lib.titan_ext.sft_pretokenized import PretokenizedSftDataLoader


# Path to the pre-tokenized data file (10 examples × ~1M tokens each).
DATA_PATH = (
    "/scratch/users/prasann/corpus-reasoning/"
    "data/contradiction_n23000_first10_qwen3_tok.pt"
)

# Smaller smoke dataset: 10 examples × ~5K tokens each (n=100 docs/example).
SMOKE_DATA_PATH = (
    "/scratch/users/prasann/corpus-reasoning/"
    "data/contradiction_n100_first10_qwen3_tok.pt"
)

# 512K-token variant: 10 examples × ~500K tokens each (n=11500 docs/example).
DATA_PATH_512K = (
    "/scratch/users/prasann/corpus-reasoning/"
    "data/contradiction_n11500_first10_qwen3_tok.pt"
)

# Path to local HF assets directory containing Qwen3-4B-Base tokenizer + config.
# Downloaded ahead of time via scripts/download_hf_assets.py.
HF_ASSETS_PATH = (
    "/scratch/users/prasann/huggingface-cache/torchtitan_assets/Qwen3-4B-Base"
)


def _short_smoke_training(seq_len: int, steps: int) -> TrainingConfig:
    return TrainingConfig(
        local_batch_size=1,
        seq_len=seq_len,
        steps=steps,
        # Tight no-validation, no-eval, no-cpu-offload defaults.
    )


def _common_optimizer() -> OptimizersContainer.Config:
    return OptimizersContainer.Config(lr=1e-5)


def _common_lr_scheduler(steps: int) -> LRSchedulersContainer.Config:
    return LRSchedulersContainer.Config(
        warmup_steps=max(1, steps // 10),
        decay_ratio=0.8,
        decay_type="linear",
        min_lr_factor=0.0,
    )


def _common_metrics() -> MetricsProcessor.Config:
    return MetricsProcessor.Config(log_freq=1)


def _common_checkpoint() -> CheckpointManager.Config:
    # No intermediate saves — only the last step, for the smoke run.
    return CheckpointManager.Config(
        interval=99999,
        last_save_model_only=True,
    )


def _qwen3_4b_contra(
    *,
    seq_len: int,
    steps: int,
    cp_degree: int,
    dp_shard_degree: int,
    ac_mode: str,
    data_path: str = DATA_PATH,
) -> Trainer.Config:
    # Nightly torchtitan model_registry takes (flavor) only — only the
    # debugmodel flavor has _flex / _flex_flash variants. For 4B we inject
    # FlexAttention with the FLASH kernel ourselves (the same recipe used
    # by debugmodel_flex_flash) — required at long context, since the
    # default SDPA backend silently falls back to the math kernel and
    # materializes the full attention matrix at >~64K tokens (148 GB OOM
    # at 524K context, 256K per rank with cp=2).
    spec = qwen3_model_registry("4B")
    spec.model.rope.max_seq_len = max(seq_len, spec.model.rope.max_seq_len)
    if has_cuda_capability(10, 0):
        block_size = (256, 128)  # Blackwell
    elif has_cuda_capability(9, 0):
        block_size = (128, 128)  # Hopper (H200)
    else:
        block_size = None
    flex_cfg = (
        FlexAttention.Config(block_size=block_size, kernel_options={"BACKEND": "FLASH"})
        if block_size is not None
        else FlexAttention.Config()
    )
    new_layers = []
    for layer_cfg in spec.model.layers:
        layer_cfg = deepcopy(layer_cfg)
        layer_cfg.attention.inner_attention = flex_cfg
        layer_cfg.attention.mask_type = "block_causal"
        new_layers.append(layer_cfg)
    spec.model.layers = new_layers
    # Loss is wired via model_spec.build_loss_fn in the nightly torchtitan;
    # no separate loss: field on Trainer.Config.
    return Trainer.Config(
        hf_assets_path=HF_ASSETS_PATH,
        metrics=_common_metrics(),
        model_spec=spec,
        dataloader=PretokenizedSftDataLoader.Config(
            data_path=data_path,
            pad_token_id=151643,  # Qwen3 eos
            infinite=True,
        ),
        optimizer=_common_optimizer(),
        lr_scheduler=_common_lr_scheduler(steps),
        training=_short_smoke_training(seq_len=seq_len, steps=steps),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=dp_shard_degree,
            data_parallel_replicate_degree=1,
            tensor_parallel_degree=1,
            context_parallel_degree=cp_degree,
            context_parallel_load_balancer="headtail",  # SDPA backend
        ),
        checkpoint=_common_checkpoint(),
        activation_checkpoint=ActivationCheckpointConfig(mode=ac_mode),
    )


# ── Smoke configs at increasing context lengths ────────────────────────────


def qwen3_4b_contra_8k_smoke() -> Trainer.Config:
    """Phase B4: small-context smoke on the n=100 (~5K-token) dataset.

    FSDP=4 CP=1 — the simplest fits-in-memory config. Validates the full-FT +
    AC + dataloader + HF-weight-load + optimizer pipeline before scaling
    context length. seq_len=8192 leaves room to pad the ~5K-token examples;
    pad tokens carry label=-100 so loss is only on the ~40-token answer.

    ac_mode="none" + attn_backend="flex": working around a torch-nightly
    backward bug seen with selective AC + SDPA (the meta-tensor "data not
    allocated yet" error during loss.backward()).
    """
    return _qwen3_4b_contra(
        seq_len=8192,
        steps=5,
        cp_degree=1,
        dp_shard_degree=4,
        ac_mode="none",
        data_path=SMOKE_DATA_PATH,
    )


def qwen3_4b_contra_32k() -> Trainer.Config:
    """Phase B4 smoke test: 4xH200 FSDP only, no CP, 32K context.

    Validates the full-FT + activation-ckpt + dataloader stack before adding
    context parallel. Examples get truncated to 32K (mostly the start of the
    long prompt) — loss won't be meaningful, but no NaNs / no OOM is the goal.
    """
    return _qwen3_4b_contra(
        seq_len=32768,
        steps=5,
        cp_degree=1,
        dp_shard_degree=4,
        ac_mode="full",
    )


def qwen3_4b_contra_256k() -> Trainer.Config:
    """Phase B5 first CP step: CP=4, no DP, 256K context."""
    return _qwen3_4b_contra(
        seq_len=262144,
        steps=5,
        cp_degree=4,
        dp_shard_degree=1,
        ac_mode="full",
    )


def qwen3_4b_contra_512k() -> Trainer.Config:
    """Scaled-down phase B5: FSDP=2 + CP=2 at 512K context.

    Per-rank: model ~32 GB sharded /2, 256K tokens after cp=2 split.
    Checkpoints at AC=full: 36 × 256K × 2560 × 2 = ~46 GB.
    Total per rank ~80 GB on H200-141GB — plenty of headroom.
    """
    return _qwen3_4b_contra(
        seq_len=524288,
        steps=10,
        cp_degree=2,
        dp_shard_degree=2,
        ac_mode="full",
        data_path=DATA_PATH_512K,
    )


def qwen3_4b_contra_1m() -> Trainer.Config:
    """Phase B5 target: FSDP=2 + CP=2 at ~1M context.

    Pure CP=4 (dp_shard=1) OOM'd because dp_shard=1 means NO FSDP sharding —
    each rank held the full 64 GB of model+grad+opt state, leaving only ~75 GB
    for activations which a 1M-context forward exceeded.

    With dp_shard=2 cp=2: model state sharded 2-way (~32 GB / rank), sequence
    sharded 2-way (500K per CP rank). AC=full keeps the activation working set
    bounded to ~one-layer's worth (5 GB residual + ~20 GB MLP intermediate at
    500K → ~25 GB peak working set). Total per-rank: ~32 + ~25 = ~57 GB +
    temporaries (15-25 GB) = ~80-100 GB — fits 141 GB H200 with headroom.
    """
    return _qwen3_4b_contra(
        seq_len=1_004_288,
        steps=20,
        cp_degree=2,
        dp_shard_degree=2,
        ac_mode="full",
    )
