"""Proposed 3:1 small hero architecture; configuration only, not a job launcher.

Based on the qualified small hero at b34d2972fcdc19a4bc0425cde43b88b5c9310cf5.
Only sequence mixers at zero-based blocks 3 and 11 change from KDA to the
existing full-attention configuration. Preserve dimensions, experts, initialization,
per-head QK gains and the selectable EMO policy. Training/LR budgets are proposals.
"""

from copy import deepcopy

from olmoe3_small_medium_models import VOCAB_SIZE
from olmoe3_small_medium_models import build_model_config as build_family_model

from olmo_core.nn.transformer import OLMoDDPModelConfig

FULL_ATTENTION_LAYERS = (3, 7, 11, 15)
BATCH_TOKENS = 16_777_216
WARMUP_STEPS = 2_000
TOTAL_STEPS = 120_000
DECAY_STEPS = 12_000
DECAY_START = TOTAL_STEPS - DECAY_STEPS
# Tune before assigning the production peak LR; do not silently reuse 1.1e-3.
PROPOSED_PEAK_LR = None


def build_model_config(
    *, eos_token_id: int, emo: bool, vocab_size: int = VOCAB_SIZE
) -> OLMoDDPModelConfig:
    """Build an independent EMO/non-EMO config without modifying the 7:1 family."""
    model = build_family_model("small", eos_token_id=eos_token_id, vocab_size=vocab_size)
    attention_block = model.block_overrides[7]
    attention_block.sequence_mixer.qk_norm_per_head_gains = True
    for index in FULL_ATTENTION_LAYERS:
        model.block_overrides[index] = deepcopy(attention_block)
    if not emo:
        for block in [model.block, *model.block_overrides.values()]:
            router = getattr(block, "routed_experts_router", None)
            if router is not None:
                router.emo = None
    model.validate()
    assert model.n_layers == 16 and model.d_model == 1024
    assert sorted(model.block_overrides) == [0, *FULL_ATTENTION_LAYERS]
    if vocab_size == VOCAB_SIZE:
        assert model.num_active_params == 787_364_992
        assert model.num_active_non_embedding_params == 684_604_544
        assert model.num_params == 12_489_473_152
    return model


def checkpoint_steps() -> tuple[int, ...]:
    """Logical full-state saves, including step zero, fork point and final endpoint.

    Uses the revised hero cadence: every100 through18k, every250 through60k,
    every500 afterward. A separately registered decay should reuse the parent's
    fork checkpoint, not make an unnecessary second copy at the fork.
    """
    steps = (
        [0]
        + list(range(100, 18_001, 100))
        + list(range(18_250, 60_001, 250))
        + list(range(60_500, TOTAL_STEPS + 1, 500))
    )
    assert len(steps) == len(set(steps)) == 469
    assert DECAY_START in steps and TOTAL_STEPS in steps
    return tuple(steps)


def training_plan() -> dict:
    """Describe proposed settings; no remote writes, registrations or submissions."""
    return dict(
        gpus_per_run=64,
        sequence_length=8192,
        microbatch_sequences_per_gpu=4,
        gradient_accumulation=8,
        global_batch_tokens=BATCH_TOKENS,
        peak_lr=PROPOSED_PEAK_LR,
        warmup_steps=WARMUP_STEPS,
        stable_through_step=DECAY_START,
        linear_decay_steps=DECAY_STEPS,
        end_step=TOTAL_STEPS,
        end_tokens=TOTAL_STEPS * BATCH_TOKENS,
        checkpoint_count=len(checkpoint_steps()),
        precision="BF16 model / FP32 optimizer",
        pipeline_parallel=1,
        expert_parallel=1,
        activation_checkpointing=False,
        checkpoint_async=False,
        dataset="same local Dolma 3.5 manifest/order as qualified hero",
        optimizer="same qualified OLMoDDP AdamW, weight decay 0.1, betas (0.9, 0.95)",
        initialization_seed=12536,
        data_seed=928543231,
        optimization_policy="core-docpool-top16-wgrad-rs / optimized100b / communication=none",
    )
