"""
Infra smoke test: Qwen3-4B full fine-tune under context parallelism at long context,
using olmo-core. Random-init weights + a local random-token dataset — the goal is to
find the seq-length ceiling on 4xH200 (forward+backward without OOM), not to learn.

Launch with torchrun (see jobs/smoke_olmo_qwen3_cp_*.sh):

    torchrun --nproc-per-node=4 scripts/train/olmo_qwen3_cp_smoke.py smoke \
        --sequence-length 1048576 --cp-degree 4 --steps 3
"""

import argparse
import logging
import os
import sys

import numpy as np
import rich

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, LionConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import GPUMemoryMonitorCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.config import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerContextParallelConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

QWEN3_VOCAB = 151936
QWEN3_BASE_CTX = 40960  # Qwen3 max_position_embeddings; YaRN beyond this.


def make_random_dataset(path: str, n_tokens: int, vocab_cap: int = 50000) -> None:
    """A flat stream of random uint16 token ids (default olmo-core dtype). Ids stay
    < vocab_cap < model vocab, so they're valid embedding indices."""
    if get_rank() == 0 and not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rng = np.random.default_rng(0)
        arr = rng.integers(0, vocab_cap, size=n_tokens, dtype=np.uint16)
        np.save(path, arr)
        log.info(f"wrote random dataset: {path} ({n_tokens} tokens)")


def build_config(opts):
    seq = opts.sequence_length

    model_config = TransformerConfig.qwen3_4B(vocab_size=QWEN3_VOCAB)
    # Extend context past Qwen3's native window with YaRN (same knob the OLMo3
    # long-context recipe uses). factor = target / base, rounded up.
    if seq > QWEN3_BASE_CTX:
        factor = float(np.ceil(seq / QWEN3_BASE_CTX))
        model_config = model_config.with_rope_scaling(
            YaRNRoPEScalingConfig(
                factor=factor, beta_fast=32, beta_slow=1, old_context_len=QWEN3_BASE_CTX
            )
        )

    # Qwen3's 151936-token vocab makes the full [seq_local, vocab] logits tensor enormous at
    # long context (~80GB bf16 at 256k tokens) and is what OOMs first. Liger's fused linear
    # cross-entropy fuses the projection with the loss and never materializes full logits.
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    # Ring context parallelism is only supported by the flash_2 backend (torch SDPA and
    # flash_3 both reject ring CP), and it additionally needs the ring-flash-attn package.
    def _use_flash2(attn_cfg):
        attn_cfg.backend = AttentionBackendName.flash_2
        attn_cfg.use_flash = True

    _use_flash2(model_config.block.sequence_mixer)
    if model_config.block_overrides:
        for b in model_config.block_overrides.values():
            _use_flash2(b.sequence_mixer)

    tokenizer_config = TokenizerConfig.gpt2()  # only used for data plumbing here
    dataset_config = NumpyFSLDatasetConfig(
        paths=[opts.data_path],
        sequence_length=seq,
        tokenizer=tokenizer_config,
        work_dir=opts.work_dir,
    )
    # One full sequence per optimizer step; CP splits it across cp-degree ranks.
    data_loader_config = NumpyDataLoaderConfig(global_batch_size=seq, seed=0, num_workers=2)

    if opts.ac == "full":
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full
        )
    elif opts.ac == "budget":
        # SAC: recompute only enough to hit the memory budget (0=recompute all, 1=none).
        # Less recompute than full -> faster. Requires compile.
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=opts.ac_budget,
        )
    elif opts.ac == "selected":
        # Checkpoint every Nth block, store the rest -> recompute less than full (faster),
        # use more memory than full. No compile needed.
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_blocks,
            block_interval=opts.ac_block_interval,
        )
    else:
        ac_config = None

    # Lion keeps one state tensor (momentum) vs AdamW's two (m, v) -> ~half the optimizer
    # memory. On 2 GPUs (no DP shard) the unsharded optimizer is the fixed-memory wall, so
    # Lion frees room for budget-mode activation retention. (Random-init smoke: optimizer
    # choice doesn't affect the throughput/memory we're measuring.)
    optim = LionConfig(lr=1e-4) if opts.optim == "lion" else AdamWConfig(lr=1e-4)

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=seq,  # tokens; CP shards the sequence dim under the hood
        max_sequence_length=seq,
        optim=optim,
        compile_model=opts.compile,  # fusion speedup; required for budget-mode AC
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        # zig_zag balances a single plain causal sequence across CP ranks. (The llama3
        # balancer the OLMo3 recipe uses requires document-packed/intra-doc masking.)
        cp_config=TransformerContextParallelConfig.zig_zag(degree=opts.cp_degree),
        ac_config=ac_config,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=10),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            metrics_collect_interval=1,
            cancel_check_interval=1,
            max_duration=Duration.steps(opts.steps),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
    )

    return model_config, dataset_config, data_loader_config, train_module_config, trainer_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str)
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--cp-degree", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--ac", choices=["full", "none", "budget", "selected"], default="full")
    parser.add_argument("--ac-budget", type=float, default=0.5,
                        help="For --ac budget: activation memory budget in [0,1]; higher = less recompute = faster.")
    parser.add_argument("--ac-block-interval", type=int, default=2,
                        help="For --ac selected: checkpoint every Nth block (2 = half recomputed).")
    parser.add_argument("--compile", dest="compile", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--optim", choices=["adamw", "lion"], default="adamw")
    parser.add_argument("--data-path", type=str, default="/tmp/olmo_smoke/random_tokens.npy")
    parser.add_argument("--data-tokens", type=int, default=8_000_000)
    parser.add_argument("--work-dir", type=str, default="/tmp/olmo_smoke/cache")
    parser.add_argument("--save-folder", type=str, default="/tmp/olmo_smoke/run")
    parser.add_argument("--dry-run", action="store_true")
    opts = parser.parse_args()

    # Need enough tokens for a few sequences of the requested length.
    opts.data_tokens = max(opts.data_tokens, opts.sequence_length * (opts.steps + 2))
    make_random_dataset(opts.data_path, opts.data_tokens)

    mc, dc, dlc, tmc, tc = build_config(opts)
    if get_rank() == 0:
        rich.print({"seq": opts.sequence_length, "cp": opts.cp_degree, "ac": opts.ac})

    if opts.dry_run:
        if get_rank() == 0:
            rich.print(mc)
        return

    prepare_training_environment()
    try:
        seed_all(12536)
        model = mc.build(init_device="meta")
        train_module = tmc.build(model)
        dataset = dc.build()
        data_loader = dlc.build(dataset, dp_process_group=train_module.dp_process_group)
        trainer = tc.build(train_module, data_loader)
        trainer.fit()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
