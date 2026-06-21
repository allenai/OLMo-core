from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    SlackNotifierCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

# ---------------------------------------------------------------------------
# CPT-MIX variant of Qwen3-4B-dense-noruler-4k-SFT.py.
#
# Question this answers: does mixing a small fraction of raw continued-pretraining text
# (dolma3longmino, the corpus the CPT base came from) into the no-ruler SFT MITIGATE the RULER
# forgetting we measured (baseline 0.93/0.91/0.89 -> no-ruler-SFT 0.88/0.84/0.78 at 4k/8k/16k)?
#
# Identical to the no-ruler 4k run (same SFT data, 120 steps, FSDP, no YaRN/AC/CP) EXCEPT the
# instance source is now an 85/15 token-fraction MIX (plan doc Option B):
#   * SFT docs  -> NumpyDocumentSource with completion-only label masks (ratio 0.85)
#   * CPT docs  -> NumpyDocumentSource with all-True masks => full-sequence loss (ratio 0.15)
#   mixed by MixingDocumentSource, then ConcatAndChunk into 4k windows.
# The CPT shards are Qwen3-tokenized (eos 151643, matches this model) by tokenize_dolma3_longmino_
# sample.py --tokenizer Qwen/Qwen3-0.6B, with explicit all-True mask-*.npy sidecars.
#
# Differences vs the full dense unified SFT recipe (kept otherwise identical so it's a faithful
# small-scale version):
#   * SEQUENCE_LENGTH = 4096 (fast); native Qwen3 ctx (32k) covers this. YaRN kept to match the
#     dense CPT base's rope config (factor 2, old_context_len 32768) -- negligible at <=4k but keeps
#     train/base rope consistent.
#   * NO context parallelism (CP is pointless at 4k); single node, 8 GPUs.
#   * Data = combined_debug_noruler shards (1k suite examples, ruler + held-out excluded), built by
#     build_combined_suite_jsonl.py --tasks <all in-train minus ruler> + convert_unified_to_sft.py.
#   * normal priority (this is a cheap probe, not a headline run).
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH = 4096  # short, fast packed window (well within Qwen3's native 32k)

# 1k SFT shards that EXCLUDE ruler (and held-out tasks), capped at 4096 tokens/instance.
SFT_DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/suite_it_sft_qwen/combined_debug_noruler_4k"
# Qwen3-tokenized dolma3longmino CPT sample (part-*.npy) + all-True masks (mask-*.npy).
CPT_DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/dolma3_longmino_qwen3_sample"
# Token fraction of CPT in the mix (plan default 0.15).
CPT_FRAC = 0.15

# Dense CPT checkpoint to initialize from (same base as the full dense unified SFT).
BASE_CHECKPOINT = "/weka/oe-training-default/ai2-llm/checkpoints/q4b-dense-dolma3longmino/step2385/model_and_optim"

NUM_NODES = 1

# Global batch in *tokens*. 1 node = 8 GPUs, rank_microbatch=SEQUENCE_LENGTH, no CP (8 DP replicas)
# -> SEQUENCE_LENGTH * 8 ~ 32k tokens/step. Small, appropriate for a 1k-example probe.
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 8

LR = 5e-5
# 425 instances pack into ~204 windows of 4096 tokens; at 8 windows/step (8 GPUs, the floor) that's
# only ~25 steps/epoch. Train a fixed 120 steps (~4.7 passes) so the SFT actually moves the model
# (>=100 steps) -- the point is to give it a real chance to forget RULER.
NUM_STEPS = 120


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
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
        beaker_launch_config.priority = "normal"

    tokenizer_config = TokenizerConfig.qwen3()
    # SFT shards are separated by single EOS tokens; qwen3 sets bos==eos, so drop bos for doc splitting.
    from dataclasses import replace

    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # No YaRN: the base is natively 32k and we train at 4k, so native rope is correct (and keeps the
    # baseline CPT-base rope == SFT-ckpt rope, so the RULER before/after comparison is apples-to-apples).
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        compile_model=False,  # tiny probe; skip compile startup cost
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            # Full FSDP across the node's 8 GPUs: a 4B model + full AdamW state does NOT fit on one
            # 80GB GPU replicated (shard_degree=1 OOMs at optim.step). Sharding params+optim 8 ways
            # (~0.5B params/GPU) fits comfortably; the 8 ranks are still the data-parallel dim.
            shard_degree=8,
        ),
        # No CP and no activation checkpointing: at 4k with FSDP sharding the per-GPU model (~0.5B
        # params) and activations are small enough to fit on 80GB without either. (AC 'budget' mode
        # also requires compile, which we disable for fast startup on this tiny probe.)
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # 85/15 token-fraction mix of SFT (completion-masked) + CPT (full-loss) documents, then
    # ConcatAndChunk into 4k windows. Each document carries its own mask so a chunk straddling an
    # SFT and a CPT doc is masked correctly (SFT: completion-only; CPT: all tokens).
    sft = SFT_DATA_ROOT.rstrip("/")
    cpt = CPT_DATA_ROOT.rstrip("/")
    sft_doc_source = NumpyDocumentSourceConfig(
        source_paths=[f"{sft}/token_ids_part_*.npy"],
        tokenizer=doc_tokenizer_config,
        label_mask_paths=[f"{sft}/labels_mask_*.npy"],
        expand_glob=True,
    )
    cpt_doc_source = NumpyDocumentSourceConfig(
        source_paths=[f"{cpt}/part-*.npy"],
        tokenizer=doc_tokenizer_config,
        label_mask_paths=[f"{cpt}/mask-*.npy"],  # explicit all-True => full-sequence CPT loss
        expand_glob=True,
    )
    mixed_source = MixingDocumentSourceConfig(
        source_specs=[
            # SFT is small (~835k tokens); allow repetition within the mix so the 0.85 ratio holds.
            MixingDocumentSourceSpecConfig(
                source=sft_doc_source, ratio=1.0 - CPT_FRAC, max_repetition_factor=8.0, label="sft_noruler"
            ),
            # CPT sample is ~30M tokens; no repetition needed at 0.15.
            MixingDocumentSourceSpecConfig(
                source=cpt_doc_source, ratio=CPT_FRAC, max_repetition_factor=1.0, label="cpt_longmino"
            ),
        ],
    )
    instance_source_config = ConcatAndChunkInstanceSourceConfig(
        sources=[mixed_source],
        sequence_length=SEQUENCE_LENGTH,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
        generate_doc_lengths=True,  # block-diagonal masking at EOS doc boundaries (dense flash varlen)
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            load_path=BASE_CHECKPOINT,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.steps(NUM_STEPS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=50,
                ephemeral_save_interval=25,
                max_checkpoints=3,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=cli_context.run_name,
                entity="prasanns-allen-institute-for-ai",
                project="memory-networks",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "slack_notifier",
            SlackNotifierCallback(name=run_name_with_ts, enabled=False),
        )
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
    """
    Small-scale (1k, ruler-excluded) dense SFT of Qwen3-4B at 4k seq + 15% CPT mix, to test whether
    mixing continued-pretraining text mitigates RULER forgetting.

        python src/scripts/train/sft/Qwen3-4B-dense-noruler-4k-CPTmix-SFT.py \\
            launch q4b-dense-noruler-4k-cptmix-sft ai2/jupiter
    """
    main(config_builder=build_experiment_config)
