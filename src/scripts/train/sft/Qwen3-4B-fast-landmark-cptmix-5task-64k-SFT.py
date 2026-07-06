"""
32k-context, context-parallel (Ulysses degree 8) Beaker/gantry SFT of the Qwen3-4B **FAST-LANDMARK**
CPT model on a MIX of 5 long-context tasks (contradiction, nq, oolong, rerank, outlier) + raw
continued-pretraining (CPT) text.

This is the LANDMARK counterpart of ``Qwen3-4B-dense-cptmix-5task-64k-SFT.py`` (which scaled the
local 8k dense recipe to 32k on weka/Beaker). It keeps that script's mixing weights / gantry / CP /
budget skeleton and swaps in landmark attention exactly as the LOCAL landmark launcher
(``Qwen3-4B-fast-landmark-cptmix-5task-local.py``, validated at 8k -> RULER 0.749 / contra 0.896 /
nq 0.98 / rerank 0.556 / oolong 0.533 / outlier 0.267) and the landmark base32k script do:

  1. Model: ``fast_landmark=True, mem_freq=63`` (block 64). NO YaRN -- the landmark memory mechanism
     extends usable context, so RoPE is left native (matches the fast-landmark CPT + the base32k /
     packed landmark SFT scripts).
  2. Init from the FAST-LANDMARK CPT base on weka (q4b-fast-landmark-dolma3longmino/step2385), NOT
     the dense base -- the landmark-token (151860) embedding + grouped-softmax attention were trained
     during landmark CPT. Loaded weights-only (load_optim_state=False).
  3. Data: ``MixingDocumentSource -> LandmarkPackingInstanceSource`` (NOT ConcatAndChunk). Landmark
     packing inserts a landmark token per document every ``mem_freq`` tokens, packs whole documents
     into 32k (landmark-space) windows, and emits block-aligned ``doc_lens`` -> block-diagonal
     masking (the landmark analogue of the dense recipe's ``generate_doc_lengths=True``). So SFT
     examples are isolated (no cross-document attention), matching eval-time single-example windows.
     ``generate_doc_lengths=False`` (boundaries come from the packing source; EOS-derived ones are
     not block-aligned and the landmark attention rejects them).

NO data re-tokenization is needed: LandmarkPacking consumes the SAME EOS-delimited weka shards the
dense 32k run used (``prasanns/cptmix_data/``) and packs them into 32k windows at load time. The
5 SFT tasks (docs <= 8k) are never dropped; only CPT docs longer than one 32k window are dropped
(fewer than at 8k, so the realised CPT fraction is closer to the nominal 0.85 than the local 8k run).

CP=8 + landmark packing + fast_landmark is the combo validated by ``Qwen3-4B-fast-landmark-packed-SFT.py``
(per-document RoPE applied per CP shard before the all-to-all gather; doc mask built on the gathered
sequence). 1 NODE x 8 GPUs -> exactly one CP=8 replica (no DP).

    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-fast-landmark-cptmix-5task-64k-SFT.py \\
        dry_run q4b-lm-cptmix-5task-64k ai2/jupiter-cirrascale-2
    PYTHONPATH=src python src/scripts/train/sft/Qwen3-4B-fast-landmark-cptmix-5task-64k-SFT.py \\
        launch  q4b-lm-cptmix-5task-64k ai2/jupiter-cirrascale-2 --launch.allow_dirty=true --launch.num_nodes=1
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

# ---------------------------------------------------------------------------
# Landmark geometry
# ---------------------------------------------------------------------------
MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
SEQUENCE_LENGTH = 65536  # 64k landmark-token-space window (full ladder length); divisible by BLOCK_SIZE
LANDMARK_TOKEN_ID = 151860  # Qwen3 reserved token used as the landmark (memory) token

# Context parallel (Ulysses) degree. Qwen3-4B: n_heads=32, n_kv_heads=8 -> CP=8 splits both cleanly.
CP_DEGREE = 8
# 1 node x 8 GPUs = exactly one CP=8 replica (no DP). Saturated clusters can't schedule 2 nodes, so
# default 1 (override with --launch.num_nodes for DP replicas if capacity frees up).
NUM_NODES = 1

# ---------------------------------------------------------------------------
# Data (weka) -- the SAME shards the dense 32k run used (no re-tokenization for landmark packing).
# ---------------------------------------------------------------------------
# LENGTH-LADDER (to-64k) SFT data: real long-context examples (ctx up to 64k) replacing the old
# <=8k packed shards. Tokenized with max_seq_len 65536, EOS 151643. CPT text is unchanged and stays
# in the original cptmix_data dir (reused, not re-uploaded).
DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/cptmix_data_ladder64k"
CONTRA_DATA_ROOT = f"{DATA_ROOT}/contradiction"
NQ_DATA_ROOT = f"{DATA_ROOT}/nq"
OOLONG_DATA_ROOT = f"{DATA_ROOT}/oolong"
RERANK_DATA_ROOT = f"{DATA_ROOT}/rerank"
OUTLIER_DATA_ROOT = f"{DATA_ROOT}/outlier"
CPT_DATA_ROOT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/amandab/"
    "dolma3_longmino_mix_sample15B_qwen"
)

# Fast-landmark CPT base (model+optim) on weka -- the dolma3longmino CPT of Qwen3-4B-Base with
# fast-landmark attention. Loaded weights-only (load_optim_state=False).
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/"
    "q4b-fast-landmark-dolma3longmino/step2385/model_and_optim"
)

# ---------------------------------------------------------------------------
# Mixing fractions (identical to the dense 32k / local 8k recipe). CPT = 85%; the 15% SFT budget is
# split contradiction 2x / rerank 1.5x / outlier 1.5x / nq 1x / oolong 1x (sum 7).
# ---------------------------------------------------------------------------
CPT_FRAC = 0.85
SFT_BUDGET = 1.0 - CPT_FRAC  # 0.15
_W = {"contra": 2.0, "rerank": 1.5, "outlier": 1.5, "nq": 1.0, "oolong": 1.0}
_WSUM = sum(_W.values())  # 7.0
NQ_FRAC = SFT_BUDGET * _W["nq"] / _WSUM
OOLONG_FRAC = SFT_BUDGET * _W["oolong"] / _WSUM
RERANK_FRAC = SFT_BUDGET * _W["rerank"] / _WSUM
OUTLIER_FRAC = SFT_BUDGET * _W["outlier"] / _WSUM
CONTRA_FRAC = max(0.0, 1.0 - CPT_FRAC - (NQ_FRAC + OOLONG_FRAC + RERANK_FRAC + OUTLIER_FRAC))

# ---------------------------------------------------------------------------
# Optimization / budget
# ---------------------------------------------------------------------------
LR = 1e-5
# Match the validated 8k recipe's optimizer dynamics (~1465 steps at lr1e-5): smallest valid global
# batch = ONE sequence/optimizer step (= SEQUENCE_LENGTH tokens), token budget sized to TARGET_STEPS.
# The earlier 16x-larger batch gave only ~61 steps at 64k and the SFT tasks never learned.
TARGET_STEPS = 1465
GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH  # one packed window/optimizer step (grad-accum 1 on the CP=8 replica)
TARGET_TOKENS = GLOBAL_BATCH_SIZE * TARGET_STEPS  # 64k -> 96.0M tokens
MAX_STEPS = max(1, round(TARGET_TOKENS / GLOBAL_BATCH_SIZE))  # = TARGET_STEPS = 1465


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
        beaker_launch_config.priority = "urgent"

    tokenizer_config = TokenizerConfig.qwen3()
    # SFT/CPT shards are separated by SINGLE EOS tokens; qwen3 ties bos==eos, so drop BOS so the
    # document splitter treats each EOS as a doc boundary (not a doubled eos+bos).
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # Qwen3-4B with FAST LANDMARK attention (no YaRN: the landmark memory mechanism extends context).
    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        fast_landmark=True,
        mem_freq=MEM_FREQ,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,  # one packed 32k window per CP replica per micro-step
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
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=1,
        ),
        # Ulysses CP: the fast-landmark mixer gathers the full sequence per rank before the grouped
        # softmax. Packing is supported under CP (per-document RoPE per shard, mask on the gathered
        # sequence). See Qwen3-4B-fast-landmark-packed-SFT.py.
        cp_config=TransformerContextParallelConfig.ulysses(degree=CP_DEGREE),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.7,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # ---- N-way mixed document source: 5 SFT tasks + CPT (identical weights to the dense 32k run) ----
    def _sft_source(root: str) -> NumpyDocumentSourceConfig:
        """Completion-only masked SFT shards (loss only on assistant tokens)."""
        r = root.rstrip("/")
        return NumpyDocumentSourceConfig(
            source_paths=[f"{r}/token_ids_part_*.npy"],
            tokenizer=doc_tokenizer_config,
            label_mask_paths=[f"{r}/labels_mask_*.npy"],
            expand_glob=True,
        )

    cpt = CPT_DATA_ROOT.rstrip("/")
    cpt_doc_source = NumpyDocumentSourceConfig(
        source_paths=[f"{cpt}/part-*.npy"],
        tokenizer=doc_tokenizer_config,
        expand_glob=True,
    )

    specs = [
        MixingDocumentSourceSpecConfig(
            source=_sft_source(CONTRA_DATA_ROOT), ratio=CONTRA_FRAC,
            max_repetition_factor=8.0, label="contradiction",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(NQ_DATA_ROOT), ratio=NQ_FRAC,
            max_repetition_factor=8.0, label="nq_retrieval",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(OOLONG_DATA_ROOT), ratio=OOLONG_FRAC,
            max_repetition_factor=8.0, label="oolong",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(RERANK_DATA_ROOT), ratio=RERANK_FRAC,
            max_repetition_factor=8.0, label="rerank",
        ),
        MixingDocumentSourceSpecConfig(
            source=_sft_source(OUTLIER_DATA_ROOT), ratio=OUTLIER_FRAC,
            max_repetition_factor=8.0, label="outlier",
        ),
        MixingDocumentSourceSpecConfig(
            source=cpt_doc_source, ratio=CPT_FRAC,
            max_repetition_factor=3.0, label="cpt_longmino",
        ),
    ]

    # Per-document landmark insertion + greedy packing into 32k windows + block-aligned doc_lens.
    instance_source_config = LandmarkPackingInstanceSourceConfig(
        source=MixingDocumentSourceConfig(source_specs=specs),
        sequence_length=SEQUENCE_LENGTH,
        mem_freq=MEM_FREQ,
        mem_id=LANDMARK_TOKEN_ID,
        pad_id=tokenizer_config.pad_token_id,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
        # Document boundaries come from LandmarkPackingInstanceSource (block-aligned doc_lens);
        # EOS-derived boundaries would NOT be block-aligned and the landmark attention rejects them.
        generate_doc_lengths=False,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            load_path=BASE_CHECKPOINT,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=100000,  # only at the end (ephemeral handles mid-run)
                ephemeral_save_interval=MAX_STEPS,
                max_checkpoints=2,
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
    cpt0.85 + weighted 5-task SFT (contradiction/nq/oolong/rerank/outlier) of the Qwen3-4B
    FAST-LANDMARK CPT model at 32k context with Ulysses CP degree 8.

        python src/scripts/train/sft/Qwen3-4B-fast-landmark-cptmix-5task-64k-SFT.py \\
            dry_run q4b-lm-cptmix-5task-64k ai2/jupiter-cirrascale-2
        python src/scripts/train/sft/Qwen3-4B-fast-landmark-cptmix-5task-64k-SFT.py \\
            launch  q4b-lm-cptmix-5task-64k ai2/jupiter-cirrascale-2 --launch.allow_dirty=true --launch.num_nodes=1
    """
    main(config_builder=build_experiment_config)
