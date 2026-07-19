"""Task-agnostic LOCAL (torchrun, no Beaker/weka) SFT trainer for the CTC suite experiment
(``records/ctc-suite-scaling-plan.md`` §4/§11): one joint run per (task, arm) on the hybrid
Qwen3.5 models (Gated DeltaNet + full attention, 3:1), full vs document-chunked arms.

Generalized from ``attn_explore/Qwen3.5-0.8B-docchunk-mask-mix-contradiction-SFT-local.py`` +
the curriculum/hard-fail logic of its Qwen3-0.6B sibling. What it keeps:

  * distcp base loading (``load_path`` -> the shared per-scale converted base; model-only), with a
    ``.metadata`` guard so a mis-staged base cannot silently train from scratch;
  * the mask-mix curriculum derivation with BOTH divisor fixes (world_size AND
    micro-batch-instances) and the hard-fail check that the anneal actually lands on
    ``mix_end_p`` (see the mask-mix-ngpu-anneal bug);
  * ``metadata.json`` as a hard requirement of the shard dir (``num_instances`` drives the
    curriculum; ``max_example_len`` guards ``--seq-len``; ``marker_set`` is cross-checked).

Arms (``--variant``):

  * ``full``        -> plain causal attention (no ``document_chunk_attention``); the box markers in
                       the shard are just ordinary tokens. Upper-bound / O(N^2) arm.
  * ``chunked``     -> document-chunked mask on the full-attention blocks (GDN blocks are linear
                       attention and ignore ``chunk_ids``), no mixing. Eval-matched arm.
  * ``chunked-mix`` -> chunked + curriculum mask mixing 0.8 -> 0.0 (the project default recipe;
                       eval still uses the pure chunked mask).

Marker ids come from ``RESERVED_IDS["qwen3_5"]`` in :mod:`olmo_core.data.document_chunk_landmark`
-- never retyped. The shard must be built with ``--marker-set qwen3_5``
(``convert_unified_to_document_landmark.py``); this script hard-fails on a mismatch.

CPU-checkable dry run (no CUDA, nothing written)::

    PYTHONPATH=src python src/scripts/train/memexpress/ctc_suite/train_ctc_suite.py \\
      --task contradiction --data <shard-dir> --variant chunked-mix --model-scale 0.8b \\
      --dry-run --dry-run-world-size 8 --dry-run-n-examples 20000

Real run (via ``run_ctc_local.sbatch``; ``--variant chunked-mix`` forces ``--no-compile``)::

    torchrun --nproc_per_node=8 src/scripts/train/memexpress/ctc_suite/train_ctc_suite.py \\
      --task contradiction --data <shard-dir> --variant chunked-mix --model-scale 0.8b \\
      --run-name ctc-contra-chunkedmix-08b --save-checkpoint --no-compile
"""

import argparse
import datetime
import json
import os
import subprocess
from typing import Any, Dict, Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    PadToLengthInstanceSourceConfig,
)
from olmo_core.data.document_chunk_landmark import RESERVED_IDS  # canonical ids -- never retype
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.attention.chunked_mask import mask_mix_standard_prob
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

MARKER_FAMILY = "qwen3_5"
IDS = RESERVED_IDS[MARKER_FAMILY]  # doc_start=248049, doc_end=248050, eos=248044

#: Qwen3.5 embedding-matrix size (rows), shared across the 0.8B/4B/9B bases.
VOCAB_SIZE = 248320

MODEL_FACTORIES = {
    "0.8b": TransformerConfig.qwen3_5_0_8B,
    "4b": TransformerConfig.qwen3_5_4B,
    "9b": TransformerConfig.qwen3_5_9B,
}

# Per-scale converted olmo distcp bases (model-only). Only 0.8B exists today; 4B/9B must be passed
# explicitly (--base-checkpoint / BASE_SRC) once converted -- repeat the marker audit first (§4).
DEFAULT_BASE_CHECKPOINT = {
    "0.8b": "/scratch/users/prasann/cpt_mix_ckpts/q35-08b-base-modelonly/model_and_optim",
}
SAVE_ROOT = "/scratch/users/prasann/olmo_ckpts"
WORK_DIR = "/scratch/users/prasann/longctx_sft_qwen/dataset-cache-ctc-suite"

# Hyperparams inherited from the source attn_explore scripts (NOT contradiction-specific).
LR = 5e-5
NUM_EPOCHS = 3
WANDB_PROJECT = "memory-networks"


def read_shard_metadata(data_dir: str, seq_len: int) -> Dict[str, Any]:
    """Read and validate the shard's ``metadata.json``.

    :param data_dir: Shard dir from ``convert_unified_to_document_landmark.py``.
    :param seq_len: The training sequence length, checked against ``max_example_len``.

    :returns: The parsed metadata dict.

    :raises SystemExit: If the metadata is missing, the marker set is not ``qwen3_5``, or
        ``seq_len`` is too short for the shard.
    """
    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise SystemExit(
            f"{meta_path} missing. The docchunk trainers REQUIRE the converter's metadata.json "
            "(num_instances drives the mask-mix anneal; max_example_len guards --seq-len)."
        )
    meta = json.load(open(meta_path))
    shard_marker_set = meta.get("marker_set")
    if shard_marker_set is not None and shard_marker_set != MARKER_FAMILY:
        raise SystemExit(
            f"shard {data_dir} was built with --marker-set {shard_marker_set!r}, but this trainer "
            f"is Qwen3.5-only ({MARKER_FAMILY!r}). A wrong-tokenizer shard produces plausible "
            "numbers, not a crash -- rebuild the shard, do not override."
        )
    for key, want in [("doc_start_id", IDS.doc_start), ("doc_end_id", IDS.doc_end)]:
        got = meta.get(key)
        if got is not None and int(got) != want:
            raise SystemExit(f"shard metadata {key}={got} != canonical {want} ({MARKER_FAMILY})")
    if meta.get("max_example_len", 0) > seq_len:
        raise SystemExit(
            f"--seq-len={seq_len} < max example length {meta['max_example_len']} "
            f"(PadToLength would SKIP the long examples). Raise --seq-len."
        )
    return meta


def derive_batch_geometry(global_batch: int, world_size: int, micro_batch_instances: int) -> int:
    """Validate the batch geometry and return instances-per-rank-per-step.

    :param global_batch: Instances per optimizer step (across all ranks).
    :param world_size: Number of DP ranks.
    :param micro_batch_instances: Instances per rank per FORWARD.

    :returns: ``global_batch // world_size`` (each rank's share of a step).

    :raises SystemExit: If a rank's share is < 1 or ``micro_batch_instances`` exceeds it --
        olmo-core would otherwise silently reinterpret the batch (quiet optimization drift).
    """
    per_rank = global_batch // world_size
    if per_rank < 1 or micro_batch_instances > per_rank:
        raise SystemExit(
            f"--micro-batch-instances={micro_batch_instances} invalid: with --global-batch="
            f"{global_batch} and world_size={world_size} each rank holds {per_rank} instance(s). "
            f"Use --global-batch >= world_size and --micro-batch-instances in [1, {per_rank}]."
        )
    return per_rank


def derive_mask_mix_curriculum(
    *,
    n_examples: int,
    epochs: int,
    global_batch: int,
    world_size: int,
    micro_batch_instances: int,
    mix_start_p: float,
    mix_end_p: float,
) -> Dict[str, Any]:
    """Derive the curriculum's ``mix_total_forwards`` and hard-check the anneal lands.

    ``mix_total_forwards`` must be the number of FORWARDS a rank actually performs, because that
    is what the curriculum's counter counts. Two divisors, both learned the hard way:

      * ``world_size`` -- data is sharded across DP ranks (the original mask-mix-ngpu-anneal bug:
        p stalled at ``mix_start_p * (1 - 1/world_size)``, worse the more GPUs).
      * ``micro_batch_instances`` -- a forward carries this many instances, so raising it CUTS the
        forward count by the same factor (observed 2026-07-16: p ended at 0.601 instead of 0.0).

    The invariant: forwards_per_rank == optimizer steps * (per-rank instances / micro-batch).

    :param n_examples: Instances in the shard (``metadata.json: num_instances``).
    :param epochs: Training epochs.
    :param global_batch: Instances per optimizer step.
    :param world_size: Number of DP ranks.
    :param micro_batch_instances: Instances per rank per forward.
    :param mix_start_p: Curriculum start collapse probability.
    :param mix_end_p: Curriculum end collapse probability.

    :returns: Dict with ``mix_total_forwards``, ``steps``, ``forwards_per_rank``, ``final_p``.

    :raises SystemExit: If the predicted final ``p_standard`` misses ``mix_end_p`` by > 0.01 --
        a silent shortfall does not crash and does not look wrong in the loss, it just means the
        model never trained predominantly on its target mask, voiding the run.
    """
    per_rank = derive_batch_geometry(global_batch, world_size, micro_batch_instances)
    mix_total_forwards = max(1, (n_examples * epochs) // (world_size * micro_batch_instances))
    # Independently re-derive the forward count from the STEP count and assert the curriculum
    # will actually land on mix_end_p, instead of trusting one formula to agree with itself.
    steps = max(1, (n_examples * epochs) // global_batch)
    forwards_per_rank = steps * (per_rank // micro_batch_instances)
    final_p = mask_mix_standard_prob(
        forwards_per_rank - 1,
        mix_start_p=mix_start_p,
        mix_end_p=mix_end_p,
        mix_total_forwards=mix_total_forwards,
    )
    # Tolerance 0.01, not 1e-3: integer-division rounding legitimately leaves the last forward a
    # hair off. We are catching gross shortfalls like the observed 0.601, not rounding dust.
    if abs(final_p - mix_end_p) > 0.01:
        raise SystemExit(
            f"mask-mix curriculum would end at p_standard={final_p:.4f}, not mix_end_p="
            f"{mix_end_p}: mix_total_forwards={mix_total_forwards} disagrees with the "
            f"{forwards_per_rank} forwards this rank will actually run. The model would keep "
            f"training on PLAIN CAUSAL for {final_p:.0%} of forwards forever, silently voiding "
            "the full-vs-chunked comparison. Check the world_size / micro_batch_instances "
            "divisors."
        )
    return dict(
        mix_total_forwards=mix_total_forwards,
        steps=steps,
        forwards_per_rank=forwards_per_rank,
        final_p=final_p,
    )


def build_model_config(opts: argparse.Namespace) -> TransformerConfig:
    """Build the per-scale Qwen3.5 :class:`TransformerConfig` for the requested variant.

    :param opts: Parsed CLI options (``model_scale``, ``variant``, mix knobs).

    :returns: The model config, with ``document_chunk_attention`` set for the chunked arms.
    """
    factory = MODEL_FACTORIES[opts.model_scale]
    # Pin flash_2 so the saved config.json supports KV-cached generation at eval (the qwen3_5
    # factories default attn_backend=None otherwise; matches the source attn_explore scripts).
    qwen_kwargs: Dict[str, Any] = dict(
        vocab_size=opts.vocab_size, attn_backend=AttentionBackendName.flash_2
    )
    if opts.variant != "full":
        # Chunked mask on the full-attention blocks only; GDN blocks ignore chunk_ids.
        qwen_kwargs["document_chunked"] = True
        qwen_kwargs["cross_doc_mode"] = "chunked"
    model_config = factory(**qwen_kwargs)
    # Fused-linear CE: never materialize float logits over the full 248k vocab. At seq-len
    # 40960 the unfused path needs ~38 GiB per rank just for logits.float() and OOMs H200s
    # (same setting as the proven 40k-seq sft_docchunk Beaker scripts).
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear
    if opts.variant != "full":
        mix_keys: Dict[str, Any] = {}
        if opts.variant == "chunked-mix":
            mix_keys = dict(
                mix_start_p=opts.mix_start_p,
                mix_end_p=opts.mix_end_p,
                mix_total_forwards=opts._mix_total_forwards,  # filled by the curriculum deriv
                mix_seed=opts.mix_seed,
                mix_log_interval=opts.mix_log_interval,
            )
        model_config.document_chunk_attention = {
            "doc_start_id": IDS.doc_start,
            "doc_end_id": IDS.doc_end,
            "eos_id": IDS.eos,
            "mode": "chunked",
            **mix_keys,
        }
    return model_config


def build_provenance(opts: argparse.Namespace, world_size: int) -> Dict[str, Any]:
    """Assemble the run's ``provenance.json`` payload (results-hub-ingestible; never fabricated).

    :param opts: Parsed CLI options.
    :param world_size: DP world size of the run.

    :returns: The provenance dict.
    """
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        git_commit = "unknown"
    return {
        "experiment": "ctc_suite",
        "script": os.path.abspath(__file__),
        "git_commit": git_commit or "unknown",
        "task": opts.task,
        "variant": opts.variant,
        "model_scale": opts.model_scale,
        "data": os.path.abspath(opts.data),
        "marker_set": {"family": MARKER_FAMILY, **IDS._asdict()},
        "world_size": world_size,
        "start_time": os.environ.get("CTC_LAUNCH_TS")
        or datetime.datetime.now().isoformat(timespec="seconds"),
        "args": {k: v for k, v in vars(opts).items() if not k.startswith("_")},
    }


def announce_wandb(opts: argparse.Namespace) -> None:
    """Print the wandb group URL at launch (project directive: always surface it)."""
    if not opts.wandb:
        return
    group = opts.wandb_group or opts.run_name
    entity = opts.wandb_entity or os.environ.get("WANDB_ENTITY")
    if entity:
        print(
            f"[ctc-suite] wandb group: https://wandb.ai/{entity}/{WANDB_PROJECT}/groups/{group}",
            flush=True,
        )
    else:
        print(
            f"[ctc-suite] wandb project={WANDB_PROJECT} group={group} "
            "(set --wandb-entity / WANDB_ENTITY for a direct group URL)",
            flush=True,
        )


def resolve_plan(opts: argparse.Namespace, world_size: int) -> Dict[str, Any]:
    """Resolve everything derivable without CUDA: metadata, batch geometry, curriculum, configs.

    :param opts: Parsed CLI options. ``opts._mix_total_forwards`` is filled as a side effect for
        the chunked-mix variant.
    :param world_size: DP world size (real, or hypothetical under ``--dry-run``).

    :returns: Dict with ``meta``, ``n_examples``, ``per_rank``, ``curriculum`` (or None),
        ``model_config``, ``instance_source_config``, ``data_loader_config``.
    """
    meta = read_shard_metadata(opts.data, opts.seq_len)
    n_examples = int(opts.dry_run_n_examples or meta["num_instances"])
    if meta.get("task") and meta["task"] != opts.task:
        print(
            f"[ctc-suite] WARNING: --task {opts.task} != shard metadata task {meta['task']!r}",
            flush=True,
        )
    per_rank = derive_batch_geometry(opts.global_batch, world_size, opts.micro_batch_instances)
    curriculum: Optional[Dict[str, Any]] = None
    if opts.variant == "chunked-mix":
        curriculum = derive_mask_mix_curriculum(
            n_examples=n_examples,
            epochs=opts.epochs,
            global_batch=opts.global_batch,
            world_size=world_size,
            micro_batch_instances=opts.micro_batch_instances,
            mix_start_p=opts.mix_start_p,
            mix_end_p=opts.mix_end_p,
        )
        opts._mix_total_forwards = curriculum["mix_total_forwards"]

    # Qwen3.5 tokenizer (its OWN vocab; NOT TokenizerConfig.qwen3()); pad == eos.
    tokenizer_config = TokenizerConfig(
        vocab_size=opts.vocab_size,
        eos_token_id=IDS.eos,
        pad_token_id=IDS.eos,
        bos_token_id=None,
        identifier="Qwen/Qwen3.5-0.8B-Base",
    )
    instance_source_config = PadToLengthInstanceSourceConfig.from_npy(
        f"{opts.data}/token_ids_part_*.npy",
        tokenizer=tokenizer_config,
        sequence_length=opts.seq_len,
        label_mask_paths=[f"{opts.data}/labels_mask_*.npy"],
        expand_glob=True,
    )
    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=opts.work_dir or WORK_DIR,
        global_batch_size=opts.global_batch * opts.seq_len,
        seed=34521 + opts.seed,
        num_workers=opts.num_workers,
    )
    return dict(
        meta=meta,
        n_examples=n_examples,
        per_rank=per_rank,
        curriculum=curriculum,
        model_config=build_model_config(opts),
        tokenizer_config=tokenizer_config,
        instance_source_config=instance_source_config,
        data_loader_config=data_loader_config,
    )


def dry_run(opts: argparse.Namespace) -> None:
    """Build all configs without CUDA / distributed init and print the resolved plan."""
    world_size = opts.dry_run_world_size
    plan = resolve_plan(opts, world_size)
    model_config = plan["model_config"]
    print("[ctc-suite] DRY RUN -- configs only, no CUDA, nothing trained, nothing written")
    print(
        f"  task={opts.task} variant={opts.variant} scale={opts.model_scale} "
        f"run_name={opts.run_name}"
    )
    print(
        f"  data={os.path.abspath(opts.data)} n_examples={plan['n_examples']}"
        + (
            f" (OVERRIDE; shard has {plan['meta']['num_instances']})"
            if opts.dry_run_n_examples
            else " (metadata)"
        )
    )
    print(
        f"  marker_set={MARKER_FAMILY} doc_start={IDS.doc_start} doc_end={IDS.doc_end} "
        f"eos={IDS.eos}"
    )
    print(
        f"  model: qwen3_5_{opts.model_scale} n_layers={model_config.n_layers} "
        f"params={model_config.num_params:,} "
        f"document_chunk_attention={model_config.document_chunk_attention}"
    )
    print(
        f"  seq_len={opts.seq_len} epochs={opts.epochs} lr={opts.lr} "
        f"global_batch={opts.global_batch} instances "
        f"({opts.global_batch * opts.seq_len:,} tokens) "
        f"micro_batch_instances={opts.micro_batch_instances}"
    )
    print(f"  world_size={world_size} (hypothetical) -> per-rank instances/step={plan['per_rank']}")
    if plan["curriculum"] is not None:
        c = plan["curriculum"]
        print(
            f"  mask-mix curriculum: p {opts.mix_start_p} -> {opts.mix_end_p} over "
            f"mix_total_forwards={c['mix_total_forwards']} | steps={c['steps']} "
            f"forwards_per_rank={c['forwards_per_rank']} "
            f"predicted final p_standard={c['final_p']:.4f}  [HARD-CHECK PASSED]"
        )
    else:
        print(f"  mask-mix: none (variant={opts.variant})")
    print(f"  data_loader: work_dir={plan['data_loader_config'].work_dir}")
    base = opts.base_checkpoint or DEFAULT_BASE_CHECKPOINT.get(opts.model_scale)
    print(f"  base_checkpoint={base or 'MISSING (required for a real run at this scale)'}")
    announce_wandb(opts)
    prov = build_provenance(opts, world_size)
    print(f"  provenance preview: {json.dumps(prov, indent=2, default=str)}")


def build_and_fit(opts: argparse.Namespace) -> None:
    """Build the model / data loader / trainer and run the fit (real training entrypoint).

    :param opts: Parsed CLI options.
    """
    save_folder = opts.save_folder or f"{SAVE_ROOT}/{opts.run_name}"
    base_checkpoint = opts.base_checkpoint or DEFAULT_BASE_CHECKPOINT.get(opts.model_scale)
    if not base_checkpoint:
        raise SystemExit(
            f"--base-checkpoint required for --model-scale {opts.model_scale} (no default base "
            "converted yet; run the marker audit on it first, see the plan §4)."
        )
    # Guard the distcp marker: staging a base via cp can drop the hidden .metadata, after which
    # olmo-core silently trains FROM SCRATCH (CE ~ ln(vocab)) instead of from the base.
    if not os.path.exists(os.path.join(base_checkpoint, ".metadata")):
        raise SystemExit(
            f"base checkpoint {base_checkpoint} missing .metadata -> would silently train from "
            "scratch. Point at the model_and_optim SUBDIR of the converted base."
        )

    world_size = max(1, get_world_size())
    plan = resolve_plan(opts, world_size)
    print(
        f"[ctc-suite] task={opts.task} variant={opts.variant} scale={opts.model_scale} "
        f"n_examples={plan['n_examples']} epochs={opts.epochs} seq_len={opts.seq_len} "
        f"global_batch={opts.global_batch} world_size={world_size} "
        f"curriculum={plan['curriculum']}",
        flush=True,
    )
    announce_wandb(opts)

    if get_rank() == 0:
        os.makedirs(save_folder, exist_ok=True)
        with open(os.path.join(save_folder, "provenance.json"), "w") as f:
            json.dump(build_provenance(opts, world_size), f, indent=2, default=str)

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.micro_batch_instances * opts.seq_len,
        max_sequence_length=opts.seq_len,
        optim=SkipStepAdamWConfig(
            lr=opts.lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        compile_model=opts.compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=1,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            load_path=base_checkpoint,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=1,
            cancel_check_interval=1,
            max_duration=Duration.epochs(opts.epochs),
            hard_stop=Duration.steps(opts.max_steps) if opts.max_steps else None,
            # no_checkpoints=True would ALSO skip the base-checkpoint LOAD block (trainer.fit
            # gates loading on `not no_checkpoints`) -> silent train-from-scratch. Keep False;
            # nothing is auto-saved mid-run (no CheckpointerCallback) -- the explicit
            # --save-checkpoint below writes the single model-only checkpoint for eval.
            no_checkpoints=False,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
    )
    if opts.wandb:
        trainer_config = trainer_config.with_callback(
            "wandb",
            WandBCallback(
                name=opts.run_name,
                group=opts.wandb_group or opts.run_name,
                entity=opts.wandb_entity,
                project=WANDB_PROJECT,
                enabled=True,
                cancel_check_interval=10,
                config={
                    "experiment": "ctc_suite",
                    "task": opts.task,
                    "variant": opts.variant,
                    "model_scale": opts.model_scale,
                    "seq_len": opts.seq_len,
                    "epochs": opts.epochs,
                    "lr": opts.lr,
                    "global_batch": opts.global_batch,
                    "seed": opts.seed,
                    "n_examples": plan["n_examples"],
                    "mix_start_p": opts.mix_start_p if opts.variant == "chunked-mix" else None,
                    "mix_end_p": opts.mix_end_p if opts.variant == "chunked-mix" else None,
                    "base_checkpoint": base_checkpoint,
                    "data": opts.data,
                },
            ),
        )

    seed_all(12536 + opts.seed)
    model = plan["model_config"].build(init_device="meta")
    train_module = train_module_config.build(model)
    source = plan["instance_source_config"].build(plan["data_loader_config"].work_dir)
    data_loader = plan["data_loader_config"].build(
        source, dp_process_group=train_module.dp_process_group
    )
    trainer = trainer_config.build(train_module, data_loader)
    trainer.fit()

    # Save a model-only checkpoint in the eval loader's expected layout (config.json +
    # model_and_optim/) so the docchunk eval scripts load it directly.
    if opts.save_checkpoint:
        from olmo_core.distributed.checkpoint import save_model_and_optim_state

        save_model_and_optim_state(
            f"{save_folder}/model_and_optim", train_module.model, save_overwrite=True
        )
        if get_rank() == 0:
            experiment = {
                "model": plan["model_config"].as_config_dict(),
                "dataset": {"tokenizer": plan["tokenizer_config"].as_config_dict()},
            }
            with open(f"{save_folder}/config.json", "w") as f:
                json.dump(experiment, f)
            print(f"[ctc-suite] saved model-only checkpoint -> {save_folder}", flush=True)


def parse_args() -> argparse.Namespace:
    """Build the CLI parser and parse arguments.

    :returns: Parsed options (with the private ``_mix_total_forwards`` slot pre-initialized).
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", required=True, help="suite task name (provenance + wandb; see plan)")
    ap.add_argument(
        "--data",
        required=True,
        help="shard dir from convert_unified_to_document_landmark.py --marker-set qwen3_5 "
        "(token_ids_part_*.npy + labels_mask_*.npy + metadata.json)",
    )
    ap.add_argument("--model-scale", choices=sorted(MODEL_FACTORIES), default="0.8b")
    ap.add_argument(
        "--variant",
        choices=["full", "chunked", "chunked-mix"],
        required=True,
        help="full = plain causal (no document_chunk_attention); chunked = pure document-chunked "
        "mask; chunked-mix = chunked + curriculum mask mixing (mix_start_p -> mix_end_p)",
    )
    ap.add_argument(
        "--seq-len", type=int, default=40960, help="fits the 32k rung + prompt/CoT overhead"
    )
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument(
        "--global-batch",
        type=int,
        default=8,
        help="instances per optimizer step across all ranks (global batch tokens = this * seq-len)",
    )
    ap.add_argument("--save-folder", default=None, help=f"default {SAVE_ROOT}/<run-name>")
    ap.add_argument("--run-name", default="ctc-suite-smoke")
    ap.add_argument("--wandb-group", default=None, help="default: the run name")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="disable wandb (ON by default; needs WANDB_API_KEY)",
    )
    ap.add_argument(
        "--base-checkpoint",
        default=None,
        help="model_and_optim distcp subdir of the converted per-scale base (default only for "
        f"0.8b: {DEFAULT_BASE_CHECKPOINT['0.8b']})",
    )
    ap.add_argument(
        "--work-dir",
        default=None,
        help=f"data-loader cache dir (default {WORK_DIR}); use node-local /data for speed",
    )
    ap.add_argument(
        "--micro-batch-instances",
        type=int,
        default=1,
        help="instances per rank per FORWARD (rank_microbatch_size = this * seq-len). Pure "
        "throughput knob (grad accumulation is exact); cannot exceed --global-batch / world_size",
    )
    ap.add_argument("--num-workers", type=int, default=2, help="dataloader workers per rank")
    ap.add_argument("--max-steps", type=int, default=0, help="hard-stop after N steps (0 = full)")
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="run-to-run seed offset (shifts data order + torch seed) for noise-floor runs",
    )
    ap.add_argument("--mix-start-p", type=float, default=0.80)
    ap.add_argument("--mix-end-p", type=float, default=0.0)
    ap.add_argument("--mix-seed", type=int, default=42)
    ap.add_argument("--mix-log-interval", type=int, default=5)
    ap.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="after fit, save model-only checkpoint (config.json + model_and_optim) for eval",
    )
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build configs (model, data loader, curriculum math) WITHOUT CUDA/distributed and "
        "print the resolved plan; trains nothing, writes nothing",
    )
    ap.add_argument(
        "--dry-run-world-size",
        type=int,
        default=8,
        help="hypothetical DP world size for --dry-run curriculum math",
    )
    ap.add_argument(
        "--dry-run-n-examples",
        type=int,
        default=0,
        help="--dry-run only: pretend the shard has this many examples (0 = use metadata)",
    )
    opts = ap.parse_args()
    opts._mix_total_forwards = 0
    if not opts.dry_run and opts.dry_run_n_examples:
        ap.error("--dry-run-n-examples is only valid with --dry-run")
    if opts.variant == "chunked-mix" and opts.compile:
        # The python-seeded mix coin + counter are not torch.compile-capturable.
        print("[ctc-suite] chunked-mix: forcing --no-compile (mix coin is not compilable)")
        opts.compile = False
    return opts


def main() -> None:
    """CLI entrypoint: dry-run without distributed init, otherwise train under torchrun."""
    opts = parse_args()
    if opts.dry_run:
        dry_run(opts)
        return
    prepare_training_environment()
    try:
        build_and_fit(opts)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
