"""
CTC-suite SFT for hybrid (gated-delta-net + periodic full-attention) checkpoints.

Takes a **base checkpoint** and a **dataset** and finetunes one against the other, so this script is
not specific to the 4:1 / 7:1 comparison it was written for -- point it at any `mainline_ladder`-style
hybrid checkpoint and any olmo-core SFT shard dir.

Runs on this repo's standard SFT machinery (``sft_common.run_sft``) -- same optimizer, schedule,
FSDP, activation checkpointing, packing and Beaker plumbing as ``sft_think.py``. The additions are
(a) an arbitrary attention period rather than a fixed 4:1, and (b) Scalable-Softmax, which the
released hybrid checkpoints are trained with.

Usage::

    W=/weka/oe-training-default/ai2-llm

    # a published arm
    python src/scripts/train/hybrid-small-suite/sft_ctc.py launch my-run ai2/jupiter-cirrascale-2 \
        --preset 1.4b_7to1 \
        --dataset $W/checkpoints/prasanns/ctc_hybridish_sft/shards_long32k

    # any other checkpoint -- nothing else to specify
    python src/scripts/train/hybrid-small-suite/sft_ctc.py launch my-run ai2/jupiter-cirrascale-2 \
        --model $W/path/to/your/checkpoint/step44124/ \
        --dataset $W/checkpoints/prasanns/ctc_hybridish_sft/shards_long32k

Architecture -- depth, width, where the full-attention layers sit, whether they use Scalable-Softmax
-- is read from the checkpoint's own ``config.json``. There are no geometry flags, so there is
nothing to get wrong.

Data
----
``--dataset`` holds pre-tokenized olmo-core SFT shards, so nothing about prompt construction is
needed here -- training reads ``token_ids_part_*.npy`` + ``labels_mask_*.npy`` only.
"""

import json
import math
import os
import sys
from functools import partial
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arch import MODEL_CONFIGS  # noqa: E402
from sft_common import (  # noqa: E402
    SEQUENCE_LENGTH,
    SEED,
    build_trainer_config,
)

from olmo_core.config import DType  # noqa: E402
from olmo_core.data import NumpyDataLoaderConfig, NumpyPackedFSLDatasetConfig  # noqa: E402
from olmo_core.data.types import LongDocStrategy  # noqa: E402
from olmo_core.internal.experiment import DataComponents  # noqa: E402
from olmo_core.train import Duration  # noqa: E402
from olmo_core.distributed.parallel import DataParallelType  # noqa: E402
from olmo_core.float8 import Float8Config  # noqa: E402
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig  # noqa: E402
from olmo_core.train.train_module import (  # noqa: E402
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode  # noqa: E402
from olmo_core.internal.experiment import CommonComponents, build_config, main  # noqa: E402
from olmo_core.nn.attention import (  # noqa: E402
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GateConfig,
    GatedDeltaNetConfig,
    GateGranularity,
)
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig  # noqa: E402
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType  # noqa: E402
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation  # noqa: E402
from olmo_core.nn.transformer import (  # noqa: E402
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)

#: Default --dataset: the shard set the published checkpoints were trained on (15,183 instances).
DEFAULT_DATASET = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_hybridish_sft/shards_long32k"
)

MAINLINE = "/weka/oe-training-default/ai2-llm/scaling-ladders/mainline"

#: Shorthand for the two published base checkpoints. Geometry is NOT listed: it is read from each
#: checkpoint's own config.json, so this is only here to save typing a weka path.
#:
#: ⚠ The two arms are NOT parameter-matched (1.42B vs 2.15B, +51%): they hold the number of
#: full-attention layers fixed at 4 and differ only by 12 extra gated-delta-net layers. Quote that
#: with any result -- a 7:1 win is not an attention-ratio effect.
PRESETS: Dict[str, str] = {
    "1.4b_4to1": f"{MAINLINE}/yashasbls/v0.0.1-ssmax-a04f0e8e7236/1.4B-Cx8/long-context/step34156/",
    "1.4b_7to1": f"{MAINLINE}/tanushy/v0.0.1-seven_to_one_hybrid_ratio-c6e480e336d5/1.4B-Cx8/long-context/step44124/",
}

# ── optimization: matched to amandab's validated SFT runs ────────────────────────────────────────
# Read off `amandab/q35-dense-contra-3ep-256k-min1h-20260914/step1005/config.json`. Those runs are
# known-good on this stack, so where they disagree with `sft_common`'s defaults we follow them:
# compile_model off (compile has been a recurring source of flakiness), activation checkpointing on
# `full` rather than `budget`, and a weight-decay override pinning the embeddings to 0.0.
#
# NOT copied: her context-parallel config (Ulysses degree 4). That exists because she trains at
# 262144; at 32768 CP only adds communication for no memory benefit.
LR = 4e-5

# Packed (OBFD) with block-diagonal masking, on 4 GPUs.
#
#   a packed 32k window holds 3.05 examples on average (p50 2, p90 5)
#   4 GPUs x 1 window/rank x accum 1  ->  4 windows  ->  ~12.2 examples/step
#   1 epoch = 4,974 windows / 4       ->  ~1,244 optimizer steps
#
# Why not 8 examples/step exactly: each rank must consume a whole window, so on N GPUs the floor is
# N * 3.05 examples. 8 GPUs bottoms out at ~24/step; 4 GPUs at ~12; only an UNPACKED (padded) run
# can hit 8 exactly, and that spends ~67% of every forward on padding.
#
# Packing is safe here because `generate_doc_lengths=True` gives block-diagonal masking -- examples
# in one window cannot attend to each other, so gradients match one-example-per-forward.
WORLD_SIZE = 4
GRAD_ACCUM = 1
RANK_MICROBATCH = SEQUENCE_LENGTH                                # one window per rank
GLOBAL_BATCH_SIZE = WORLD_SIZE * GRAD_ACCUM * SEQUENCE_LENGTH    # 131,072 tok = 4 windows
EPOCHS = 1

# `sft_common` logs to ai2-llm/hybrid-small-suite, which this account's key cannot reach (404 ->
# wandb.init raises -> trainer dies after checkpoint load, before step 1).
WANDB_ENTITY = "prasanns-allen-institute-for-ai"
WANDB_PROJECT = "memory-networks"                                  # ~466 optimizer steps total


def build_ctc_train_module_config(common, **_) -> TransformerTrainModuleConfig:
    """Train-module config mirroring amandab's validated SFT setup."""
    return TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH,
        max_sequence_length=common.max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            compile=False,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
    )


def build_ctc_model_config(
    common: CommonComponents,
    model_path: str,
    attn_backend: AttentionBackendName = AttentionBackendName.flash_3,
) -> TransformerConfig:
    """
    Rebuild the architecture from the checkpoint that is about to be loaded into it.

    olmo-core checkpoints carry their own ``config.json``, so depth, width, head count, where the
    full-attention layers sit and whether they use Scalable-Softmax all come from the checkpoint
    itself. Restating any of that by hand is how you train a different model against the right
    weights -- it either fails at load or, worse, does not.

    The only thing overridden is the attention backend, which is a property of the cluster you are
    running on, not of the checkpoint.

    :param model_path: Checkpoint step directory holding ``config.json``.
    :param attn_backend: Backend for the full-attention layers.

    :returns: The checkpoint's own :class:`TransformerConfig`.

    :raises SystemExit: If there is no ``config.json`` -- there is no safe geometry to guess.
    """
    cfg_path = os.path.join(model_path.rstrip("/"), "config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit(
            f"no config.json at {cfg_path}.\n"
            "Expected an olmo-core checkpoint step directory (the one holding model_and_optim/). "
            "The architecture is read from it; there is no safe default to fall back to."
        )
    with open(cfg_path) as f:
        raw = json.load(f)
    model_cfg = TransformerConfig.from_dict(raw["model"] if "model" in raw else raw)

    n_attn = 0
    for block in list((model_cfg.block_overrides or {}).values()) + [model_cfg.block]:
        mixer = getattr(block, "sequence_mixer", None)
        if isinstance(mixer, AttentionConfig):
            mixer.backend = attn_backend
            n_attn += 1
    print(
        f"[sft_ctc] architecture from checkpoint: n_layers={model_cfg.n_layers} "
        f"d_model={model_cfg.d_model} "
        f"attention layers at {sorted(model_cfg.block_overrides or {})} "
        f"(ssmax={getattr(next(iter((model_cfg.block_overrides or {}).values()), None), 'sequence_mixer', None) and getattr(next(iter((model_cfg.block_overrides or {}).values())).sequence_mixer, 'scalable_softmax', None)})"
    )
    if n_attn == 0:
        print("[sft_ctc] WARNING: no attention layers found in the checkpoint config")
    return model_cfg


def _assert_no_overlong(dataset_path: str, seq_len: int) -> None:
    """
    Fail loudly if any instance is longer than the training window.

    ``truncate`` would silently behead such an instance -- prompt kept, answer discarded -- and the
    only visible symptom is a slightly lower loss. Checked from the shard metadata written at
    tokenisation, so it costs nothing.
    """
    meta_path = os.path.join(dataset_path.rstrip("/"), "metadata.json")
    if not os.path.exists(meta_path):
        print(f"[sft_ctc] WARNING: no metadata.json at {meta_path}; cannot verify instance lengths")
        return
    with open(meta_path) as f:
        meta = json.load(f)
    longest = int((meta.get("token_len") or {}).get("max", 0))
    if longest > seq_len:
        raise SystemExit(
            f"{dataset_path} contains an instance of {longest:,} tokens > sequence_length "
            f"{seq_len:,}. `truncate` would drop its answer and train on a zero-loss window. "
            "Re-tokenize with --max-seq-len <= sequence_length."
        )
    print(f"[sft_ctc] instance-length check OK: longest {longest:,} <= seq_len {seq_len:,}")


def build_ctc_data_components(common, dataset_path: str):
    """Data config matching prasann's SFT scripts, not ``sft_common``'s default.

    Packed via OBFD; ``generate_doc_lengths`` gives block-diagonal (varlen) masking, so examples
    sharing a window cannot attend to one another.

    ⚠ Every CTC task puts its answer at the END, so an over-length example must never be cut: that
    keeps the prompt and drops the answer, leaving a zero-loss instance. The shards are capped at
    tokenisation, and :func:`_assert_no_overlong` enforces it rather than assuming it.
    """
    _assert_no_overlong(dataset_path, common.max_sequence_length)
    clean = dataset_path.rstrip("/")
    return DataComponents(
        dataset=NumpyPackedFSLDatasetConfig(
            tokenizer=common.tokenizer,
            work_dir=common.work_dir,
            paths=[f"{clean}/token_ids_part_*.npy"],
            expand_glob=True,
            label_mask_paths=[f"{clean}/labels_mask_*.npy"],
            generate_doc_lengths=True,     # block-diagonal masking -> packed == example-level
            long_doc_strategy=LongDocStrategy.truncate,   # can never fire; see _assert_no_overlong
            sequence_length=common.max_sequence_length,
        ),
        data_loader=NumpyDataLoaderConfig(
            global_batch_size=common.global_batch_size, seed=SEED, num_workers=4
        ),
    )


def _trainer_with_epochs(common, size: str, sft_cfg: dict, tag: str, epochs: int):
    """Trainer config with this experiment's duration and a reachable W&B project.

    Two overrides on ``sft_common.build_trainer_config``:

    * ``max_duration`` -- it hardcodes 2 epochs.
    * W&B ``entity``/``project`` -- it targets ``ai2-llm/hybrid-small-suite``, which 404s for this
      account's API key. That is not a warning: ``wandb.init()`` raises and the trainer dies before
      step 1, after loading the checkpoint. Retarget rather than disable, so the loss curve exists.
    """
    cfg = build_trainer_config(common, model_size=size, sft_configs=sft_cfg, tags=["ctc-sft", tag])
    cfg.max_duration = Duration.epochs(epochs)
    wandb_cb = (cfg.callbacks or {}).get("wandb")
    if wandb_cb is not None:
        wandb_cb.entity = WANDB_ENTITY
        wandb_cb.project = WANDB_PROJECT
    return cfg


def _parse_cli(argv):
    """Pull this script's own flags out of argv, leaving olmo-core's dotlist overrides behind.

    ``main()`` consumes ``sys.argv`` itself and rejects anything it does not recognise, so these
    have to be removed rather than merely read.

    :param argv: Full ``sys.argv``.

    :returns: ``(args, remaining_argv, model_path)``.

    :raises SystemExit: If no base checkpoint was given.
    """
    import argparse

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--model", help="base checkpoint to finetune (olmo-core step dir, weka path)")
    ap.add_argument("--preset", choices=sorted(PRESETS), help=f"shorthand for a published checkpoint")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="olmo-core SFT shard dir (weka path)")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args, rest = ap.parse_known_args(argv[4:])

    model_path = args.model or (PRESETS[args.preset] if args.preset else None)
    if not model_path:
        raise SystemExit(
            "no base checkpoint. Pass --model <olmo-core step dir> (or --preset "
            f"{'|'.join(sorted(PRESETS))}).\n"
            "Architecture is read from the checkpoint's own config.json -- nothing to specify."
        )
    return args, argv[:4] + rest, model_path


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <dry_run|launch|train> <run_name> <cluster> "
            "--model PATH [--dataset DIR] [--lr F] [--epochs N] [overrides...]\n"
            f"       --preset {'|'.join(sorted(PRESETS))} substitutes a published checkpoint.\n"
            "Architecture comes from the checkpoint's config.json; there are no geometry flags."
        )

    args, argv, MODEL_PATH = _parse_cli(sys.argv)
    sys.argv = argv

    size = args.preset or "custom"
    sft_cfg = {size: dict(lr=args.lr, global_batch_size=GLOBAL_BATCH_SIZE, load_path=MODEL_PATH)}
    print(f"[sft_ctc] base checkpoint : {MODEL_PATH}")
    print(f"[sft_ctc] dataset         : {args.dataset}")
    print(f"[sft_ctc] lr {args.lr}  epochs {args.epochs}  seq_len {SEQUENCE_LENGTH}")

    CLUSTER_ATTN_BACKENDS = {
        "saturn": AttentionBackendName.flash_2,
        "jupiter": AttentionBackendName.flash_3,
        "titan": AttentionBackendName.flash_4,
    }
    cluster_arg = " ".join(sys.argv[2:4]).lower()
    attn_backend = AttentionBackendName.flash_3
    for cluster, backend in CLUSTER_ATTN_BACKENDS.items():
        if cluster in cluster_arg:
            attn_backend = backend
            break

    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        num_nodes=1,
        data_config_builder=partial(build_ctc_data_components, dataset_path=args.dataset),
        model_config_builder=partial(
            build_ctc_model_config, model_path=MODEL_PATH, attn_backend=attn_backend
        ),
        train_module_config_builder=build_ctc_train_module_config,
        trainer_config_builder=partial(
            _trainer_with_epochs, size=size, sft_cfg=sft_cfg, tag=size, epochs=args.epochs,
        ),
        include_default_evals=False,
        beaker_workspace="ai2/flex2",
        num_execution_units=1,
    )
    main(config_builder=config_builder)
