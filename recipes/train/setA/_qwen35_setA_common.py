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

Training recipe = amandab's validated 256k dense SFT (formerly ``sft_xlong256k/``, run
``q35-dense-contra-3ep-256k-min1h-20260914``): this file calls her
:func:`build_qwen35_xlong5_experiment` for the model / optimizer / FSDP / CP / AC settings and
replaces only the data, the budget and (for 32k) the parallelism. See her
``_qwen35_xlong5_dolci25_256k_common.py`` for why each of those settings is what it is (Ulysses
not ring CP for GDN, sqrt-scaled LR, fused-linear loss, flash_3, expandable_segments).

DATA
  ``/weka/.../prasanns/ctc_sft_sets/setA_max20_evaliid/shards_qwen35_256k/<task>/``, the setA
  shards as built (``recipes/data/ctc_sft/build_ctc_sft.py``): chat
  template, query position ``both``, no CoT, each document wrapped in ``<|box_start|>``/
  ``<|box_end|>``. Evaluate with the olmo-eval launcher's ``--doc-markers``
  (``CTC_SUITE_DOC_MARKERS=1``), which renders those markers token-identically (checked on real
  eval rows of all 11 tasks: ``recipes/eval/setA_audit/check_doc_marker_parity.py``).

QWEN3-4B ARM (``SETA_MODEL=qwen3-4b``, ``launch_setA_sft.py --model qwen3-4b``)
  The same recipe on the full-attention Qwen3-4B-Base (length-generalization experiment). Only what
  the architecture / tokenizer forces changes; :func:`_apply_model` lists it. Model
  ``TransformerConfig.qwen3_4B`` (flash_3, fused-linear loss, as the Qwen3.5 recipe), Qwen3 tokenizer
  (EOS 151643, ``bos=None`` as above), base ``qwen3-4b-base-trainedmark`` (marker rows repaired),
  shards ``shards_qwen3_q35le32k`` (the 32k arm's examples, re-tokenized), and the ``32k`` arm's
  window 65,536 (Qwen3 tokenizes the same text longer) with grad-accum 2 instead of 4, so a step is
  still 1,048,576 tokens. Only the ``32k`` arm and the dense variant exist for it.

SMOKE RUNS
  ``SETA_SMOKE_STEPS=N`` stops after N steps (``hard_stop``: the LR schedule is the real run's),
  logs every step and adds a GPU-memory monitor. Unset, the config is unchanged.

RESEARCH VARIANTS
  ``SETA_VARIANT`` / ``SETA_VARIANT_OPTS`` (``launch_setA_sft.py --variant/--variant-opts``) apply a
  research arm from ``recipes/train/variants.py`` on top of this dense recipe -- the same hooks
  ``research_trainer/train_ctc_suite.py`` uses: model-config changes and router LR groups /
  callbacks / data overrides here, ``enable_*`` calls and the tolerant base load through
  :func:`setA_train_hooks`. Unset (``dense``) leaves the config byte-identical.
"""

import glob
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # recipes/train

import run_guards  # noqa: E402  (recipes/train/run_guards.py)
import variants as V  # noqa: E402  (recipes/train/variants.py)
from _qwen35_xlong5_dolci25_256k_common import build_qwen35_xlong5_experiment  # noqa: E402
from _setA_defaults import DEFAULT_MODEL, DEFAULT_VARIANT, MODELS  # noqa: E402
from _setA_defaults import (  # noqa: E402
    base_checkpoint_from_env,
    model_from_env,
    set_root_from_env,
    smoke_steps_from_env,
    tasks_from_env,
    variant_from_env,
    variant_opts_from_env,
)

from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    LongDocStrategy,
    NumpyDocumentSourceConfig,
    PackingInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.experiment import CliContext, ExperimentConfig, TrainHooks
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train import Duration, LoadStrategy
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    FlopMeterCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import TransformerDataParallelConfig

# Defaults live in the import-free _setA_defaults.py (shared with launch_setA_sft.py); each is
# overridable by env var -- SETA_DATA_ROOT, SETA_TASKS, SETA_BASE_CKPT -- which launch_setA_sft.py's
# --data-root / --tasks / --base-checkpoint export to the Beaker job.
#: The model (``SETA_MODEL``): ``qwen35-4b`` (the dense Qwen3.5-4B recipe) or ``qwen3-4b``.
MODEL = model_from_env()
SET_ROOT = set_root_from_env()
SHARDS_DIR = MODELS[MODEL]["shards_dir"]
DATA_ROOT = f"{SET_ROOT}/{SHARDS_DIR}"

#: The tasks trained on: the 11 setA tasks unless ``SETA_TASKS`` names others (e.g. ``nq`` alone
#: for single-task debugging).
TASKS = tasks_from_env()

#: Dense 256k CPT base, loaded weights-only -- the base of amandab's dense 256k SFT runs
#: (``SETA_BASE_CKPT`` overrides).
BASE_CHECKPOINT = base_checkpoint_from_env()

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
if MODEL != DEFAULT_MODEL:
    # The model's arms only, at its own windows (no CP: every window here fits one GPU).
    ARMS = {
        arm: dict(ARMS[arm], sequence_length=seq) for arm, seq in MODELS[MODEL]["windows"].items()
    }
    assert all(spec["cp_degree"] == 1 for spec in ARMS.values())


def arm_geometry(arm: str) -> Dict[str, int]:
    """
    Derive one arm's parallelism and batch from :data:`ARMS`.

    :param arm: A key of :data:`ARMS`.

    :returns: sequence_length, num_nodes, cp_degree, dp_degree, windows_per_step, grad_accum.
    """
    if arm not in ARMS:
        raise ValueError(f"model {MODEL!r} has no {arm!r} arm; it has {sorted(ARMS)}")
    spec = dict(ARMS[arm])
    world = spec["num_nodes"] * GPUS_PER_NODE
    dp = world // spec["cp_degree"]
    windows = TOKENS_PER_STEP // spec["sequence_length"]
    assert windows * spec["sequence_length"] == TOKENS_PER_STEP
    assert windows % dp == 0, f"{windows} windows/step do not split over DP={dp}"
    spec.update(dp_degree=dp, windows_per_step=windows, grad_accum=windows // dp)
    return spec


def build_setA_experiment(
    cli_context: CliContext,
    *,
    arm: str,
    variant: Optional[str] = None,
    variant_opts: Optional[str] = None,
) -> ExperimentConfig:
    """
    Build one arm of the setA SFT pair (dense, or a research variant on top of it).

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.
    :param arm: A key of :data:`ARMS`.
    :param variant: A ``recipes/train/variants.py`` name (default: ``SETA_VARIANT``, else dense).
    :param variant_opts: Its ``k=v,...`` options (default: ``SETA_VARIANT_OPTS``).

    :returns: The experiment config.
    """
    g = arm_geometry(arm)
    seq = g["sequence_length"]
    # amandab's validated 256k model / optimizer / FSDP / CP / AC settings; overrides applied last.
    config = build_qwen35_xlong5_experiment(replace(cli_context, overrides=[]), arm="qboth")
    if MODEL != DEFAULT_MODEL:
        _apply_model(config, MODEL)
    tokenizer = config.data_loader.tokenizer  # bos=None: every EOS ends a document

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

    smoke = smoke_steps_from_env()
    if smoke:
        tr.hard_stop = Duration.steps(smoke)  # the full run's LR schedule, cut short
        tr.metrics_collect_interval = 1
        tr.callbacks["gpu_memory_monitor"] = GPUMemoryMonitorCallback()

    name = variant if variant is not None else variant_from_env()
    spec = variant_opts if variant_opts is not None else variant_opts_from_env()
    global _PLAN
    _PLAN = None
    if name not in (DEFAULT_VARIANT, "full"):
        if MODEL != DEFAULT_MODEL:
            raise ValueError(f"variant {name!r}: the {MODEL} arm runs the dense recipe only")
        _PLAN = apply_setA_variant(config, g, name, spec)
    elif spec.strip():
        raise ValueError(f"variant options {spec!r} given for the dense recipe")
    return config.merge(cli_context.overrides)


def _apply_model(config: ExperimentConfig, model: str) -> None:
    """
    Swap amandab's Qwen3.5-4B model and tokenizer for another model's, in place. Everything else she
    sets (optimizer, schedule, AC, FSDP, flash_3, fused-linear loss, compile off) is kept.

    ``qwen3-4b`` -- what the architecture forces, and nothing more:

    * ``TransformerConfig.qwen3_4B``: 36 full-attention layers (no GDN), 32 query / 8 KV heads of
      128, RoPE theta 1e6, qk-norm, untied ``lm_head`` (HF Qwen3-4B ties it; the converted base
      carries the tied matrix in both places and they train apart, as for every olmo-core Qwen3).
      ``attn_backend=flash_3`` and ``fused_linear`` loss as in the Qwen3.5 recipe.
    * ``TokenizerConfig.qwen3()`` (vocab 151,936; EOS = ``<|endoftext|>`` 151643) with
      ``bos_token_id=None``, for the same reason as Qwen3.5: every EOS is a document boundary.
    """
    if model != "qwen3-4b":
        raise ValueError(f"unknown setA model {model!r}")
    tokenizer_config = TokenizerConfig.qwen3()
    config.model = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_3,
    )
    config.model.lm_head.loss_implementation = LMLossImplementation.fused_linear
    config.data_loader.tokenizer = replace(tokenizer_config, bos_token_id=None)


# ----------------------------------------------------------------------------------------------
# Research variants (recipes/train/variants.py) on the canonical recipe
# ----------------------------------------------------------------------------------------------


@dataclass
class SetAVariantPlan:
    """A resolved variant for one build: what :func:`setA_train_hooks` applies at train time."""

    variant: Any  # V.Variant
    options: Any  # V.VariantOptions
    ctx: Any  # V.VariantContext
    base_checkpoint: str


#: Set by :func:`build_setA_experiment`; ``None`` for dense.
_PLAN: Optional[SetAVariantPlan] = None


def estimate_total_steps(tasks: List[str], data_root: str) -> Optional[int]:
    """
    Upper-bound the optimizer steps of one epoch from the shards' on-disk token counts.

    It counts every token, including examples the 32k arm drops and BFD padding it never adds, so
    it is exact only up to packing; pass ``total_steps=<prep's packed steps>`` for the real value.

    :returns: ``ceil(tokens / TOKENS_PER_STEP)``, or ``None`` if a task has no readable shards.
    """
    total = 0
    for task in tasks:
        files = glob.glob(f"{data_root}/{task}/token_ids_part_*.npy")
        if not files:
            return None
        itemsize = 4  # uint32: the Qwen3.5 vocab does not fit 16 bits
        meta_path = f"{data_root}/{task}/metadata.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                dtype = json.load(f).get("dtype") or "uint32"
            itemsize = {"uint16": 2, "uint32": 4, "uint64": 8, "int32": 4, "int64": 8}[dtype]
        total += sum(os.path.getsize(p) for p in files) // itemsize
    return max(1, math.ceil(total / TOKENS_PER_STEP))


def apply_setA_variant(
    config: ExperimentConfig, g: Dict[str, int], name: str, spec: str
) -> SetAVariantPlan:
    """
    Apply a research variant's config-time hooks to a built dense setA config, in place.

    Config-time: model config (``apply_model_config``), router LR groups
    (``optim_group_overrides``), train-module fields, the instance source (``data_overrides``,
    BFD-matched), callbacks (``extra_callbacks`` + the method-aware flop meter), and for variants
    that add parameters, a non-strict base load (the trainer's own load is turned off). The
    ``enable_*`` calls happen at train time through :func:`setA_train_hooks`.

    :param config: The dense config from :func:`build_setA_experiment`.
    :param g: The arm geometry (:func:`arm_geometry`).
    :param name: Variant name.
    :param spec: ``k=v,...`` options; ``total_steps`` / ``seed`` are recipe keys.

    :returns: The plan the train hooks apply.

    :raises ValueError: An arm this recipe cannot run (see ``Variant.setA_blocker`` /
        ``cp_blocker``) or an unknown / ignored option.
    """
    variant, opts, recipe = V.parse_variant_spec(name, spec)
    if variant.setA_blocker:
        raise ValueError(
            f"variant {name!r} cannot run on the setA recipe: {variant.setA_blocker}. "
            "Use recipes/train/research_trainer/train_ctc_suite.py."
        )
    if g["cp_degree"] > 1 and variant.cp_blocker:
        raise ValueError(
            f"variant {name!r} cannot run on a context-parallel arm (cp={g['cp_degree']}): "
            f"{variant.cp_blocker}. Use the 32k arm."
        )
    seq = g["sequence_length"]
    accum = g["grad_accum"]
    total_steps = recipe.get("total_steps")
    if total_steps is None and variant.needs_horizon:
        total_steps = estimate_total_steps(TASKS, DATA_ROOT)
        if total_steps is None:
            raise ValueError(
                f"variant {name!r} anneals over the run horizon, and the shards under {DATA_ROOT} "
                "are not readable here: pass --variant-opts total_steps=<prep's packed steps>"
            )
        print(
            f"[setA] {name}: horizon ESTIMATED at {total_steps} steps from on-disk token counts "
            "(an upper bound; pass total_steps=<prep's packed steps> for the exact value)",
            flush=True,
        )
    total_calls = max(1, (total_steps or 1) * accum)
    ctx = V.VariantContext(
        model_family="qwen3_5",
        seed=recipe.get("seed", 0),
        total_calls=total_calls,
        calls_per_step=accum,
        seq_len=seq,
        mix_total_forwards=total_calls,  # chunked-mix: forwards per rank = steps x accum
        vocab_size=config.model.vocab_size,
        data_dir=None,
    )

    variant.apply_model_config(config.model, opts, ctx)
    tm = config.train_module
    tm.optim.group_overrides = list(tm.optim.group_overrides or []) + list(
        variant.optim_group_overrides(opts)
    )
    for k, v in variant.train_module_overrides(opts, ctx).items():
        setattr(tm, k, v)
    override = variant.data_overrides(
        opts,
        ctx,
        source_paths=[f"{DATA_ROOT}/{task}/token_ids_part_*.npy" for task in TASKS],
        label_mask_paths=[f"{DATA_ROOT}/{task}/labels_mask_*.npy" for task in TASKS],
        tokenizer=config.data_loader.tokenizer,
        best_fit=True,
    )
    if override is not None:
        override.label = "ctc_setA"
        config.dataset = [override]

    tr = config.trainer
    for cb_name, cb in variant.extra_callbacks(opts, ctx).items():
        tr.callbacks[cb_name] = cb
    # Method-aware training FLOPs, as on every research_trainer arm (EOS == the pad id).
    tr.callbacks["flop_meter"] = FlopMeterCallback(seq_len=seq, pad_id=ctx.ids.eos)
    if variant.new_params:
        # New router / gain / projector keys are absent from the base: the trainer's load is
        # strict, so it only resumes from the save folder and the base is loaded non-strictly by
        # the post_trainer hook (V.tolerant_base_load refuses a missing BACKBONE key).
        tr.load_path = None
        tr.load_strategy = LoadStrategy.if_available
    if config.train_module.compile_model:
        raise ValueError(f"variant {name!r}: the setA recipe is expected to run with compile off")
    print(
        f"[setA] variant={name} opts={V.options_to_kv(opts) or '(defaults)'} "
        f"horizon={total_steps} steps x {accum} = {total_calls} calls",
        flush=True,
    )
    return SetAVariantPlan(variant, opts, ctx, BASE_CHECKPOINT)


def setA_train_hooks() -> TrainHooks:
    """
    The :class:`TrainHooks` that finish a variant at train time (no-ops for dense): the variant's
    ``enable_*`` calls on the meta-device model, its train-module hooks, and the tolerant base load
    for variants that add parameters. Also makes ``kill -USR1 <pid>`` dump every thread's stack
    into the job log (hung-job diagnosis without py-spy).
    """
    run_guards.install_stack_dump_handler()

    def post_model(config: ExperimentConfig, model) -> None:
        if _PLAN is None:
            return
        from olmo_core.distributed.utils import get_rank

        _PLAN.ctx.rank = get_rank()
        _PLAN.variant.post_build_model(model, _PLAN.options, _PLAN.ctx)

    def post_train_module(config: ExperimentConfig, train_module) -> None:
        if _PLAN is not None:
            _PLAN.variant.post_build_train_module(train_module, _PLAN.options, _PLAN.ctx)

    def post_trainer(config: ExperimentConfig, trainer) -> None:
        if _PLAN is not None and _PLAN.variant.new_params:
            V.tolerant_base_load(_PLAN.base_checkpoint, trainer.train_module.model, "")

    return TrainHooks(
        post_model=post_model, post_train_module=post_train_module, post_trainer=post_trainer
    )
