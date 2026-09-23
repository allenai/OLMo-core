"""
Dense Qwen3.5-4B SFT on the CTC setA mix (11 tasks, 2k -> 256k, IID with the olmo-eval CTC suite),
as a controlled pair on maximum training context:

  * ``256k`` -- every example (window 262,144). 1 node x 8 GPUs, Ulysses CP=8 -> DP=1, grad-accum 4.
  * ``32k``  -- only examples that fit in 32,768 tokens; longer ones are DROPPED, not truncated
    (truncation would cut off the question/answer). 1 node x 8 GPUs, no CP, DP=8, grad-accum 4.
  * ``256k-2node`` -- the 256k arm at amandab's exact validated geometry (2 nodes, CP=4, DP=4).
    Same data, batch and LR; a fallback if 1-node CP=8 misbehaves.

One node is enough at 256k because FSDP shards params + optimizer over the flattened DP x CP mesh
(``_get_model_mesh``): CP=8 on one node still shards the 4B state 8 ways, and each rank holds
262,144 / 8 = 32,768 tokens of activations -- half of what amandab's CP=4 run held. CP=8 is legal:
Ulysses needs cp | n_heads (16) since the KV-replication fix, and GatedDeltaNet.apply_cp needs
cp | n_v_heads (32). Cost: ~2x the wall-clock of the 2-node run, same tokens per step.

Everything except the window (and the parallelism it forces) is shared: base checkpoint, data
files, tokens per step (1,048,576), LR 4e-5, warmup 3%, one epoch, seed. The 32k arm therefore sees
fewer tokens and steps -- it trains on a strict subset of the 256k arm's data.

Training recipe = amandab's validated 256k dense SFT (``sft_xlong256k/``, run
``q35-dense-contra-3ep-256k-min1h-20260914``): this file calls her
:func:`build_qwen35_xlong5_experiment` for the model / optimizer / FSDP / CP / AC settings and
replaces only the data, the budget and (for 32k) the parallelism. See her
``_qwen35_xlong5_dolci25_256k_common.py`` for why each of those settings is what it is (Ulysses
not ring CP for GDN, sqrt-scaled LR, fused-linear loss, flash_3, expandable_segments).

DATA
  ``/weka/.../prasanns/ctc_sft_sets/setA_max20_evaliid/shards_qwen35_256k_nomarkers/<task>/``,
  tokenized by ``src/scripts/data/ctc_sft/build_ctc_sft.py --no-doc-markers`` on
  ``prasann/landmark`` (``launch_setA_sft.py tokenize``). No ``<|box_start|>``/``<|box_end|>``
  markers: the olmo-eval CTC prompts carry none, and Qwen's marker embeddings are untrained
  (CLAUDE.md), so a dense run must not see them. Chat template, query position ``both``, no CoT --
  the rendering olmo-eval uses under ``CTC_SUITE_PROMPT_FORMAT=chat``.
"""

from dataclasses import replace
from typing import Any, Dict

from _qwen35_xlong5_dolci25_256k_common import build_qwen35_xlong5_experiment

from olmo_core.data.composable import (
    LongDocStrategy,
    NumpyDocumentSourceConfig,
    PackingInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.experiment import CliContext, ExperimentConfig
from olmo_core.train import Duration
from olmo_core.train.callbacks import CheckpointerCallback, WandBCallback
from olmo_core.train.train_module import TransformerDataParallelConfig

SET_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_sft_sets/setA_max20_evaliid"
SHARDS_DIR = "shards_qwen35_256k_nomarkers"
DATA_ROOT = f"{SET_ROOT}/{SHARDS_DIR}"

#: The 11 setA tasks (textgroups dropped: shortest-document shortcut, see build_ctc_sft.py).
TASKS = [
    "nq",
    "hotpotqa",
    "qdmatch_nq",
    "outlier",
    "oolong",
    "contradiction",
    "xabsence",
    "reorder",
    "rerank",
    "strmatch",
    "grouping",
]

#: Dense 256k CPT base, loaded weights-only -- the base of amandab's dense 256k SFT runs.
BASE_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/q35-4b-dense-256k-fix/step2385/model_and_optim"
)

GPUS_PER_NODE = 8
TOKENS_PER_STEP = 1_048_576  # amandab's 256k batch: 4 DP x 262,144
LR = 4e-5  # amandab's 256k LR at this batch (sqrt-scaled from the 32k family's 1e-5 @ 65,536)
WANDB_ENTITY = "prasanns-allen-institute-for-ai"
WANDB_PROJECT = "memory-networks"

ARMS: Dict[str, Dict[str, Any]] = {
    # One node: CP=8 -> 32,768 tokens/rank (amandab's CP=4 run held 65,536), 4 microbatches/step.
    "256k": dict(sequence_length=262_144, num_nodes=1, cp_degree=8),
    # 32,768 tokens fit on one GPU with full AC, so no CP: DP=8, 4 microbatches per step.
    "32k": dict(sequence_length=32_768, num_nodes=1, cp_degree=1),
    # amandab's validated geometry exactly (2 nodes, CP=4, DP=4, no accumulation).
    "256k-2node": dict(sequence_length=262_144, num_nodes=2, cp_degree=4),
}


def arm_geometry(arm: str) -> Dict[str, int]:
    """
    Derive one arm's parallelism and batch from :data:`ARMS`.

    :param arm: A key of :data:`ARMS`.

    :returns: sequence_length, num_nodes, cp_degree, dp_degree, windows_per_step, grad_accum.
    """
    spec = dict(ARMS[arm])
    world = spec["num_nodes"] * GPUS_PER_NODE
    dp = world // spec["cp_degree"]
    windows = TOKENS_PER_STEP // spec["sequence_length"]
    assert windows * spec["sequence_length"] == TOKENS_PER_STEP
    assert windows % dp == 0, f"{windows} windows/step do not split over DP={dp}"
    spec.update(dp_degree=dp, windows_per_step=windows, grad_accum=windows // dp)
    return spec


def build_setA_experiment(cli_context: CliContext, *, arm: str) -> ExperimentConfig:
    """
    Build one arm of the setA dense SFT pair.

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.
    :param arm: A key of :data:`ARMS`.

    :returns: The experiment config.
    """
    g = arm_geometry(arm)
    seq = g["sequence_length"]
    # amandab's validated 256k model / optimizer / FSDP / CP / AC settings; overrides applied last.
    config = build_qwen35_xlong5_experiment(replace(cli_context, overrides=[]), arm="qboth")
    tokenizer = config.data_loader.tokenizer  # Qwen3.5 with bos=None: every EOS ends a document

    sources = [
        NumpyDocumentSourceConfig(
            source_paths=[f"{DATA_ROOT}/{task}/token_ids_part_*.npy"],
            label_mask_paths=[f"{DATA_ROOT}/{task}/labels_mask_*.npy"],
            tokenizer=tokenizer,
            expand_glob=True,
        )
        for task in TASKS
    ]
    config.dataset = [
        PackingInstanceSourceConfig(
            sources=sources,
            sequence_length=seq,
            tokenizer=tokenizer,
            # Examples longer than the window are DROPPED (the 32k arm's cap), never truncated.
            long_doc_strategy=LongDocStrategy.exclude,
            source_group_size=1_000_000,  # BFD-pack all tasks' files together
            label="ctc_setA",
        )
    ]
    config.data_loader.global_batch_size = TOKENS_PER_STEP
    config.data_loader.seed = 34521

    tm = config.train_module
    tm.rank_microbatch_size = seq  # one window per forward; grad-accum fills the step
    tm.max_sequence_length = seq
    tm.optim.lr = LR
    if g["cp_degree"] > 1:
        tm.cp_config.degree = g["cp_degree"]
        assert seq % g["cp_degree"] == 0
    else:
        tm.cp_config = None
    # Pure FSDP over DP (x CP, flattened in): no replicas in any arm, so nothing to hybrid-shard.
    tm.dp_config = TransformerDataParallelConfig(
        name=DataParallelType.fsdp,
        param_dtype=tm.dp_config.param_dtype,
        reduce_dtype=tm.dp_config.reduce_dtype,
        wrapping_strategy=tm.dp_config.wrapping_strategy,
    )

    if config.launch is not None:  # unused by launch_setA_sft.py (gantry --replicas), kept honest
        config.launch.num_nodes = g["num_nodes"]

    tr = config.trainer
    tr.save_folder = (
        f"/weka/oe-training-default/ai2-llm/checkpoints/prasanns/{cli_context.run_name}"
    )
    tr.save_overwrite = False
    tr.load_path = BASE_CHECKPOINT
    tr.load_optim_state = False
    tr.load_trainer_state = False
    tr.max_duration = Duration.epochs(1)
    tr.callbacks["checkpointer"] = CheckpointerCallback(
        save_interval=500,
        ephemeral_save_interval=100,  # preemption resume point
        max_checkpoints=2,
        save_async=True,
    )  # post_train writes a permanent final checkpoint as well
    wandb = tr.callbacks["wandb"]
    assert isinstance(wandb, WandBCallback)
    wandb.entity = WANDB_ENTITY
    wandb.project = WANDB_PROJECT
    wandb.group = cli_context.run_name
    return config.merge(cli_context.overrides)
