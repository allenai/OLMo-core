"""
LOCAL (torchrun, Berkeley H200) nested-FFN-MoE experiment on contradiction, Qwen3-4B.

The question: **how little FFN compute can a token get away with?** A learned per-token router
picks one of several nested FFN widths -- full, 1/4, 1/16, 1/64, or a zero-cost null rung -- and a
budget hinge loss pushes the mean per-token FFN cost down while cross-entropy holds it up. See
:mod:`olmo_core.nn.nested_ffn_moe` for the mechanism and how it relates to AdaMoE / MatFormer /
MoNE.

This is a STANDALONE study of the FFN axis: plain causal attention, no KV compaction, no soft
tokens, no pooled-doc-KV. Composing it with the pooled-doc-KV compression is a later step, and
deliberately not wired here -- if both move at once, neither result is interpretable.

Arms (``--ffn-moe-start-layer``):
  * ``-1``  -> dense reference run (no router at all). The accuracy + wall-clock baseline.
  * ``>=0`` -> routed from that layer on. Layers below it keep the full FFN for every token, since
    both role-gated experiments (records/pooled-doc-kv-attention.md) showed that removing FFN
    compute in early layers is what destroys trainability.

The router is initialized to select the full rung with probability ~1, so step 0 is exactly the
base model and any degradation is attributable to compute actually given up.

Base checkpoint: the router/gain keys must exist in the checkpoint being loaded, so train from a
base written by ``bake_ffn_moe_into_base.py`` (``--base-ckpt``), not the raw CPT base.

Run (2x H200)::

    PYTHONPATH=<repo>/src $PY -m torch.distributed.run --nproc_per_node=2 \\
      src/scripts/train/memexpress/ffnmoe/Qwen3-4B-ffnmoe-contra-local.py \\
      --run-name q4b-ffnmoe-v1 --ffn-moe-start-layer 4 --ffn-moe-target 0.05
"""

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    NumpyDocumentSourceConfig,
    PadToLengthInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    NestedFFNMoECallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

# query-position BOTH: matches the layout eval_lc_native_docchunk_contra.py hardcodes.
DATA_DIR = "/data/prasann/ffnmoe_exp/contra_n100_qboth_train"
BASE_CKPT = "/data/prasann/ffnmoe_exp/q4b-dense-cpt-fixmark-ffnmoe/model_and_optim"
SAVE_ROOT = "/data/prasann/ffnmoe_exp/runs"
WORK_DIR = "/data/prasann/ffnmoe_exp/dataset-cache"


def build_and_fit(opts: argparse.Namespace) -> None:
    seq_len = opts.seq_len
    run_name = opts.run_name
    run_name_with_ts = f"{run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    save_folder = opts.save_folder or f"{SAVE_ROOT}/{run_name}"
    base_checkpoint = opts.base_ckpt or BASE_CKPT
    routed = opts.ffn_moe_start_layer >= 0

    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    global_batch_size = opts.batch_tokens or (seq_len * world_size)
    print(
        f"[cfg] ffn_moe_start_layer={opts.ffn_moe_start_layer} seq_len={seq_len} ws={world_size} "
        f"gbs={global_batch_size} lr={opts.lr} epochs={opts.epochs}\n"
        f"[cfg] base={base_checkpoint}\n[cfg] data={opts.data}",
        flush=True,
    )

    tokenizer_config = TokenizerConfig.qwen3()
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # ---- Model: plain causal Qwen3-4B. The router is the ONLY modification. ----
    from olmo_core.nn.attention import AttentionBackendName

    model_config = TransformerConfig.qwen3_4B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName(opts.attn_backend),
    )
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_tokens or seq_len,
        max_sequence_length=seq_len,
        optim=SkipStepAdamWConfig(
            lr=opts.lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            group_overrides=(
                [
                    OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0)),
                    # The router decides discrete routing from a cold start and wants a livelier LR
                    # than the pretrained backbone; the gains are 1-D and must not be decayed.
                    OptimGroupOverride(
                        params=[
                            "blocks.*.feed_forward._nffn_router.*",
                            "blocks.*.feed_forward._nffn_gain",
                        ],
                        opts=dict(lr=opts.router_lr, weight_decay=0.0),
                    ),
                ]
                if routed
                else [OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))]
            ),
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=opts.shard_degree or world_size,
        ),
        # Default: NO activation checkpointing. A 4B model at 1x6144 tokens fits an H200 without
        # it, and full AC recomputes every block's forward in backward -- the first wave ran
        # with it on and its throughput numbers are not "normal training" numbers.
        ac_config=(
            None
            if opts.ac_mode == "none"
            else TransformerActivationCheckpointingConfig(
                mode=TransformerActivationCheckpointingMode.full
            )
            if opts.ac_mode == "full"
            else TransformerActivationCheckpointingConfig(
                mode=TransformerActivationCheckpointingMode.selected_modules,
                modules=["blocks.*.feed_forward"],
            )
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    # ---- Data ----
    instance_source_config = PadToLengthInstanceSourceConfig(
        sources=[
            NumpyDocumentSourceConfig(
                source_paths=[f"{opts.data}/token_ids_part_*.npy"],
                tokenizer=doc_tokenizer_config,
                label_mask_paths=[f"{opts.data}/labels_mask_*.npy"],
                expand_glob=True,
            )
        ],
        sequence_length=seq_len,
        tokenizer=doc_tokenizer_config,
    )
    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=WORK_DIR,
        global_batch_size=global_batch_size,
        seed=34521 + opts.seed,
        num_workers=4,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            load_path=base_checkpoint,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=(
                Duration.steps(opts.max_steps)
                if opts.max_steps > 0
                else Duration.epochs(opts.epochs)
            ),
            async_bookkeeping=False,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=opts.save_interval,
                ephemeral_save_interval=None,
                max_checkpoints=1,
                save_async=False,
            ),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
    )
    if opts.profile:
        # torch.profiler over a few steps once the run is warm: the kernel tables land in the job
        # log and the chrome trace under the work dir. This is how the REAL 2-GPU FSDP step gets
        # attributed (attention / FFN / LM head / comm / optimizer), which no single-GPU
        # microbenchmark can do.
        from olmo_core.train.callbacks import ProfilerCallback

        trainer_config = trainer_config.with_callback(
            "profiler", ProfilerCallback(skip_first=15, wait=1, warmup=2, active=2, repeat=1)
        )
    if routed:
        # Without this you cannot distinguish a real compute saving from a collapsed router.
        trainer_config = trainer_config.with_callback(
            "ffn_moe",
            NestedFFNMoECallback(
                # Pins the router's schedule clock to the global step so a crash-resume does not
                # restart the target/exploration anneals (it did, for every arm of the first wave).
                calls_per_step=max(
                    1, global_batch_size // (world_size * (opts.rank_microbatch_tokens or seq_len))
                ),
            ),
        )
    if opts.wandb:
        trainer_config = trainer_config.with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=opts.wandb_group or run_name,
                entity=opts.wandb_entity,
                project="memory-networks",
                enabled=True,
                cancel_check_interval=10,
            ),
        )

    seed_all(12536 + opts.seed)
    print("[stage] building model...", flush=True)
    model = model_config.build(init_device="meta")
    if routed:
        # Forwards per rank = steps * grad-accum: the horizon the router schedules anneal over.
        n_examples = opts.n_examples
        gbs_examples = max(1, global_batch_size // seq_len)
        steps = -(-n_examples * opts.epochs // gbs_examples)
        micro = opts.rank_microbatch_tokens or seq_len
        accum = max(1, global_batch_size // (world_size * micro))
        total_calls = max(1, steps * accum)
        model.enable_nested_ffn_moe(
            start_layer=opts.ffn_moe_start_layer,
            divisors=[float(x) for x in opts.ffn_moe_divisors.split(",")],
            include_null=not opts.ffn_moe_no_null,
            target_cost=opts.ffn_moe_target,
            budget_weight=opts.ffn_moe_budget_weight,
            hinge_power=opts.ffn_moe_hinge_power,
            target_anneal_calls=int(total_calls * opts.ffn_moe_target_anneal_frac),
            explore_prob=opts.ffn_moe_explore,
            explore_anneal_calls=int(total_calls * opts.ffn_moe_explore_anneal_frac),
            recon_frac=opts.ffn_moe_recon_frac,
            recon_weight=opts.ffn_moe_recon_weight,
            entropy_weight=opts.ffn_moe_entropy_weight,
            seed=opts.seed,
            layer_curriculum_calls=int(total_calls * opts.ffn_moe_layer_curriculum_frac),
            width_multiple=opts.ffn_moe_width_multiple,
        )
        print(
            f"[ffn-moe] routed from layer {opts.ffn_moe_start_layer}; "
            f"target {opts.ffn_moe_target} annealed over "
            f"{int(total_calls * opts.ffn_moe_target_anneal_frac)}/{total_calls} calls",
            flush=True,
        )
    train_module = train_module_config.build(model)

    source = instance_source_config.build(data_loader_config.work_dir)
    data_loader = data_loader_config.build(source, dp_process_group=train_module.dp_process_group)
    trainer = trainer_config.build(train_module, data_loader)
    trainer.callbacks["config_saver"].config = {
        "model": model_config.as_config_dict(),
        "dataset": {"tokenizer": tokenizer_config.as_config_dict()},
    }
    print(f"[stage] loading base from {base_checkpoint} and starting fit()...", flush=True)
    trainer.fit()

    # Model-only export in the eval loader's layout. The config.json records the PLAIN
    # architecture; the extra _nffn_* keys ride along and are picked up by an eval that calls
    # enable_nested_ffn_moe with the SAME flags (they are ignored by a plain full-FFN load).
    from olmo_core.distributed.checkpoint import save_model_and_optim_state
    from olmo_core.distributed.utils import get_rank

    save_model_and_optim_state(
        f"{save_folder}/model_and_optim", train_module.model, save_overwrite=True
    )
    if get_rank() == 0:
        export_config = TransformerConfig.qwen3_4B(vocab_size=tokenizer_config.padded_vocab_size())
        export_config.lm_head.loss_implementation = LMLossImplementation.fused_linear
        with open(f"{save_folder}/config.json", "w") as f:
            json.dump(
                {
                    "model": export_config.as_config_dict(),
                    "dataset": {"tokenizer": tokenizer_config.as_config_dict()},
                    # Recorded so the eval cannot silently score with the wrong routing -- the
                    # FFN_GATE lesson: score with what you trained.
                    "ffn_moe": (
                        {
                            "start_layer": opts.ffn_moe_start_layer,
                            "divisors": opts.ffn_moe_divisors,
                            "include_null": not opts.ffn_moe_no_null,
                            "width_multiple": opts.ffn_moe_width_multiple,
                        }
                        if routed
                        else None
                    ),
                },
                f,
            )
        print(f"[stage] saved model-only checkpoint -> {save_folder}", flush=True)


def main() -> None:
    import faulthandler

    faulthandler.dump_traceback_later(900, repeat=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--seq-len", type=int, default=6144)
    ap.add_argument("--save-folder", default=None)
    ap.add_argument("--base-ckpt", default=None, help="MUST have the baked _nffn_* keys")
    ap.add_argument(
        "--n-examples", type=int, default=2000, help="rows in the shard (schedule horizon only)"
    )

    ap.add_argument(
        "--ffn-moe-start-layer",
        type=int,
        default=4,
        help="route from this layer on; -1 = dense reference run (no router)",
    )
    ap.add_argument(
        "--ffn-moe-divisors",
        default="1,4,16,64",
        help="rung cost divisors: '1,4,16,64' = full, 1/4, 1/16, 1/64 (+ null unless --no-null)",
    )
    ap.add_argument("--ffn-moe-no-null", action="store_true", help="drop the zero-compute rung")
    ap.add_argument(
        "--ffn-moe-target",
        type=float,
        default=0.05,
        help="mean per-token FFN cost the hinge allows for free (0.05 = 20x FFN reduction)",
    )
    ap.add_argument("--ffn-moe-budget-weight", type=float, default=1.0)
    ap.add_argument("--ffn-moe-hinge-power", type=int, default=1, choices=[1, 2])
    ap.add_argument(
        "--ffn-moe-target-anneal-frac",
        type=float,
        default=0.3,
        help="fraction of training over which the target falls 1.0 -> --ffn-moe-target",
    )
    ap.add_argument(
        "--ffn-moe-explore",
        type=float,
        default=0.1,
        help="initial probability a training token takes a random rung (gives the narrow rungs "
        "gradient before the router ever prefers them)",
    )
    ap.add_argument("--ffn-moe-explore-anneal-frac", type=float, default=0.3)
    ap.add_argument(
        "--ffn-moe-recon-frac",
        type=float,
        default=0.02,
        help="fraction of tokens carrying a local full-FFN reconstruction target (0 disables). "
        "FFN-only: no attention, no full-context forward.",
    )
    ap.add_argument("--ffn-moe-recon-weight", type=float, default=0.0)
    ap.add_argument("--ffn-moe-entropy-weight", type=float, default=0.0)
    ap.add_argument(
        "--ffn-moe-width-multiple",
        type=int,
        default=8,
        help="rung widths are floored to a multiple of this (and at least this). 1 allows a "
        "single-hidden-unit rung: divisor 9728 -> width 1 on Qwen3-4B",
    )
    ap.add_argument(
        "--ffn-moe-layer-curriculum-frac",
        type=float,
        default=0.0,
        help="fraction of training over which routing opens from the LAST layer down to "
        "--ffn-moe-start-layer (0 = all routed layers from step 0)",
    )
    ap.add_argument("--router-lr", type=float, default=1e-3)

    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 overrides --epochs (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-interval", type=int, default=250)
    ap.add_argument("--batch-tokens", type=int, default=0)
    ap.add_argument("--rank-microbatch-tokens", type=int, default=0)
    ap.add_argument("--shard-degree", type=int, default=0)
    ap.add_argument("--ac-mode", default="none", choices=["none", "full", "selected"])
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--profile", action="store_true", help="torch.profiler window at steps ~18-19")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument(
        "--wandb-entity",
        default=None,
        help="None -> the logged-in account's default entity (the local .netrc key is the "
        "Berkeley account, NOT the AI2 one -- a mismatched entity fails with CommError).",
    )
    opts = ap.parse_args()
    prepare_training_environment()
    try:
        build_and_fit(opts)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
