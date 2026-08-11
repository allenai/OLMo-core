"""
Molmo2 "stage 1" caption-pretraining (reproduction of ``mm_olmo``'s captioner).

Trains the connector + LM on PixMoCap captions/transcripts with the vision encoder
**frozen**, using the float ``root_subsegments``-weighted loss and per-component
learning rates / warmups. In-loop evaluation is intentionally omitted.

Weights start from **base** checkpoints, matching mm_olmo's ``reset_with_pretrained_weights``:
the LM from ``Qwen/Qwen3-4B``, the ViT from ``google/siglip2-so400m-patch14-384``, and the
connector / extra-token embeddings randomly initialised (``INIT_FROM = "scratch"``). This is
stage-1 *pretraining*. Passing ``--init_from=molmo2`` instead continues the released,
already-post-trained ``allenai/Molmo2-4B`` on the stage-1 objective, which is a fine-tune —
useful for parity tests and continuation experiments, but not a stage-1 reproduction.

Run without arguments for usage. Quick local smoke test on synthetic data::

    torchrun --nproc-per-node=1 src/scripts/train/Molmo2-Stage1.py train smoke \\
        --dataset.dataset_path=synthetic --trainer.max_duration.value=5 \\
        --trainer.max_duration.unit=steps

.. note::
    Single-GPU, DDP, and FSDP/HSDP data parallelism are supported. TP/CP/PP/EP of the
    multimodal model are out of scope.
"""

import logging
import sys
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, cast

from olmo_core.config import Config, DType
from olmo_core.data.multimodal import (
    CoSynPointDatasetConfig,
    MixtureDataLoader,
    MultimodalCollatorConfig,
    MultimodalDataLoader,
    PixMoCapDatasetConfig,
    PixMoCountDatasetConfig,
    PixMoPointsDatasetConfig,
    Tulu4DatasetConfig,
)
from olmo_core.data.multimodal.paths import PIXMO_DATASETS
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.common import (
    build_launch_config,
    get_beaker_username,
    get_root_dir,
)
from olmo_core.launch.beaker import BeakerEnvVar, BeakerLaunchConfig
from olmo_core.nn.vision import MultimodalLM, MultimodalLMConfig
from olmo_core.optim import (
    AdamWConfig,
    CosWithWarmup,
    OptimGroupOverride,
    PerGroupScheduler,
)
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_cli_environment,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    MultimodalTransformerTrainModuleConfig,
    TransformerDataParallelConfig,
)
from olmo_core.utils import get_default_device, seed_all

log = logging.getLogger(__name__)

#######################
#### CONFIGURATION ####
#######################

# Architecture + tokenizer source. Only the HF *config* and tokenizer are read from this
# repo id; stage-1 weights come from the base checkpoints below (see INIT_FROM).
MODEL_ID = "allenai/Molmo2-4B"

# Weight init. "scratch" is mm_olmo's true stage-1 start and the default: base Qwen3-4B LM
# + base SigLIP2 ViT + randomly-initialised connector / new-token embeddings. "molmo2"
# instead *continues* the released (already post-trained) Molmo2-4B on the stage-1
# objective — that is a fine-tune, not stage-1 pretraining, so it is opt-in only
# (`--init_from=molmo2`), kept for parity tests and continuation experiments.
INIT_FROM = "scratch"
INIT_FROM_CHOICES = ("scratch", "molmo2")
SCRATCH_LM_ID = "Qwen/Qwen3-4B"
SCRATCH_VIT_ID = "google/siglip2-so400m-patch14-384"
NEW_EMBEDDING_INIT_STD = 0.02  # mm_olmo `new_embedding_init_range`
# RoPE base per weight source: base Qwen3-4B was trained with 1e6 (as is mm_olmo's
# stage-1 QWEN3_4B); the released Molmo2-4B checkpoint uses 5e6.
SCRATCH_ROPE_THETA = 1_000_000
MOLMO2_ROPE_THETA = 5_000_000
SEQUENCE_LENGTH = 4096  # fixed pad length; mm_olmo's captioner default is 2536
USE_FLEX_ATTN = True  # fused FlexAttention backend for the multimodal masks (~+8% MFU)
PACK_SEQUENCES = True  # pack several examples per sequence (most are ~1.4k of 4096 tokens)
COMPILE_MODEL = True  # torch.compile the LM (fuses pointwise ops; one-time compile warmup)
DATA_PREFETCH_WORKERS = 4  # background threads preprocessing examples (0 = synchronous)
MAX_CROPS = 8

# KNOWN DELTA vs mm_olmo stage-1 captioner: `response_residual_dropout=0.1`.
# mm_olmo applies 0.1 dropout to the residual stream of RESPONSE tokens only (input/image
# tokens get 0.0), via a per-token drop mask in its LM block (olmo/nn/llm.py: `Dropout`
# with `mask_p`). OLMo-core's `TransformerBlock` has a single uniform `nn.Dropout` (default
# 0.0) with no per-token/response path, so this regularizer is intentionally NOT applied
# here — adding it would require threading a response drop-mask through the core transformer
# block. Low impact for the short benchmark runs; revisit for a full-fidelity stage-1
# reproduction. (The other mm_olmo delta, the `style_and_length_v2` length-conditioning
# system prompt, IS implemented — see PixMoCapDataset.style_length_conditioning.)

# Instance-based batching (mm_olmo: global 8, device microbatch 1), expressed in tokens.
GLOBAL_BATCH_INSTANCES = 8
RANK_MICROBATCH_INSTANCES = 1
GLOBAL_BATCH_SIZE = GLOBAL_BATCH_INSTANCES * SEQUENCE_LENGTH
RANK_MICROBATCH_SIZE = RANK_MICROBATCH_INSTANCES * SEQUENCE_LENGTH

# Per-component LRs / warmups (mm_olmo train_captioner.py).
CONNECTOR_LR = 2e-4
LLM_LR = 2e-5
CONNECTOR_WARMUP = 200
LLM_WARMUP = 2000
ALPHA_F = 0.1

# Data: the canonical PixMoCap "cap" dataset (HF DatasetDict, load_from_disk). Override as needed.
DATASET_PATH = f"{PIXMO_DATASETS}/cap"
MAX_STEPS = 32000

# Stage-1 mixture rates (mm_olmo train_captioner --pointing/--nlp). Caption gets the
# remainder (1 - POINTING_RATE - NLP_RATE). Set both to 0.0 for a caption-only run.
POINTING_RATE = 0.30
NLP_RATE = 0.10

# Beaker.
BEAKER_CLUSTER = "ai2/jupiter"
NUM_NODES = 1
BEAKER_WORKSPACE = "ai2/OLMo-core"
BEAKER_BUDGET = "ai2/oe-other"

# Logging. Set WANDB_PROJECT to None to disable W&B (requires the WANDB_API_KEY secret
# in the Beaker workspace). Metrics always go to the console regardless.
# WANDB_ENTITY=None uses the API key's default entity (personal account), avoiding 403s
# from writing to a team the key lacks access to; set it to a team you can write to.
WANDB_PROJECT: Optional[str] = "molmo2-stage1"
WANDB_ENTITY: Optional[str] = None

###########################
#### END CONFIGURATION ####
###########################


@dataclass
class ExperimentConfig(Config):
    launch: BeakerLaunchConfig
    model: MultimodalLMConfig
    dataset: PixMoCapDatasetConfig
    collator: MultimodalCollatorConfig
    train_module: MultimodalTransformerTrainModuleConfig
    trainer: TrainerConfig
    model_id: str = MODEL_ID
    data_seed: int = 34521
    init_seed: int = 12536
    global_batch_size: int = GLOBAL_BATCH_SIZE
    """Global batch in *tokens* (= global instances × seq len). Override to scale the batch;
    pair with ``--train_module.rank_microbatch_size`` to set sequences/forward (GEMM size)."""
    pointing_rate: float = POINTING_RATE
    """Fraction of mixture samples from pointing/counting sources (mm_olmo ``--pointing``)."""
    nlp_rate: float = NLP_RATE
    """Fraction of mixture samples from Tulu4 NLP SFT (mm_olmo ``--nlp``)."""
    pack_sequences: bool = PACK_SEQUENCES
    """Pack several short examples per sequence (mm_olmo dynamic packer)."""

    init_from: str = INIT_FROM
    """``"molmo2"`` (released Molmo2-4B weights) or ``"scratch"`` (mm_olmo stage-1 init:
    base Qwen3-4B + base SigLIP2 + random connector/new embeddings)."""


def _build_model_config(init_from: str) -> MultimodalLMConfig:
    """Build the Molmo2-4B :class:`MultimodalLMConfig` natively (no weights, no HF read).

    ``MultimodalLMConfig.molmo2_4B`` is byte-identical to
    ``molmo2_config_from_hf_config(AutoConfig.from_pretrained(MODEL_ID))``, so a
    from-scratch run needs nothing from the released Molmo2 repo.

    The one architecture field that depends on the weight source is the RoPE base: the
    released Molmo2-4B uses ``5e6``, but base Qwen3-4B — whose weights ``"scratch"`` loads
    — was trained with ``1e6`` (mm_olmo's stage-1 ``QWEN3_4B`` also uses ``1e6``). Using
    the released value with base weights would put them on the wrong rotary base.
    """
    rope_theta = SCRATCH_ROPE_THETA if init_from == "scratch" else MOLMO2_ROPE_THETA
    return MultimodalLMConfig.molmo2_4B(rope_theta=rope_theta)


def _resolve_init_from(overrides: List[str]) -> str:
    """Read ``init_from`` out of the raw overrides.

    The model config depends on it (RoPE base), and it has to be built before
    :meth:`Config.merge` runs, so the override cannot be read off the merged config.
    Later occurrences win, matching ``merge``.
    """
    init_from = INIT_FROM
    for override in overrides:
        key, _, value = override.lstrip("-").partition("=")
        if key == "init_from" and value:
            init_from = value
    if init_from not in INIT_FROM_CHOICES:
        raise OLMoConfigurationError(f"init_from={init_from!r} is not one of {INIT_FROM_CHOICES}")
    return init_from


def build_config(script: str, run_name: str, overrides: List[str]) -> ExperimentConfig:
    root_dir = get_root_dir(BEAKER_CLUSTER)
    beaker_user = get_beaker_username()
    assert beaker_user is not None

    model_config = _build_model_config(_resolve_init_from(overrides))

    dataset_config = PixMoCapDatasetConfig(
        dataset_path=DATASET_PATH,
        mode="transcript_and_caption",
        max_crops=MAX_CROPS,
        max_sequence_length=SEQUENCE_LENGTH,
        loss_token_weighting="root_subsegments",
        seed=34521,
    )

    # Pad token: Molmo2/Qwen2.5 EOS (151643). Fixed-length padding so every batch has a
    # constant token count for the token-based Trainer.
    collator_config = MultimodalCollatorConfig(
        pad_token_id=151643,
        label_ignore_index=-100,
        pad_sequence_length=SEQUENCE_LENGTH,
    )

    train_module_config = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=LLM_LR,
            betas=(0.9, 0.95),
            eps=1e-6,
            weight_decay=0.0,
            group_overrides=[
                OptimGroupOverride(
                    params=["connector.*"],
                    opts=dict(lr=CONNECTOR_LR, weight_decay=0.0, scheduler_name="connector"),
                ),
            ],
        ),
        freeze_params=["vision.*"],
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        compile_model=COMPILE_MODEL,
        autocast_precision=DType.bfloat16,
        scheduler=PerGroupScheduler(
            schedulers={"connector": CosWithWarmup(warmup=CONNECTOR_WARMUP, alpha_f=ALPHA_F)},
            default=CosWithWarmup(warmup=LLM_WARMUP, alpha_f=ALPHA_F),
        ),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"{root_dir}/checkpoints/{beaker_user.lower()}/{run_name}",
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            # Synchronous checkpointing: avoids the async checkpoint thread pool whose
            # teardown raced/failed on this cluster ("cannot schedule new futures after
            # interpreter shutdown"). Saves block briefly but complete reliably.
            CheckpointerCallback(save_interval=2000, ephemeral_save_interval=500, save_async=False),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                enabled=WANDB_PROJECT is not None,
                cancel_check_interval=10,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback("beaker", BeakerCallback())
    )  # NOTE: no in-loop eval callbacks (out of scope for stage 1).

    launch_config = build_launch_config(
        name=run_name,
        root_dir=root_dir,
        cmd=[script, "train", run_name, *overrides],
        cluster=BEAKER_CLUSTER,
        workspace=BEAKER_WORKSPACE,
        budget=BEAKER_BUDGET,
        num_nodes=NUM_NODES,
    )
    # Stage-1 reads data and writes checkpoints on weka, so no S3 / GCS secrets are required.
    launch_config.aws_config_secret = None
    launch_config.aws_credentials_secret = None
    launch_config.google_credentials_secret = None
    # Only request env secrets that exist in the (debug) workspace; drop optional ones
    # (COMET / R2 / WEKA / SLACK) that aren't provisioned there.
    launch_config.env_secrets = [
        s for s in launch_config.env_secrets if s.name in ("BEAKER_TOKEN", "WANDB_API_KEY")
    ]
    # The pointing/counting Arrow datasets on weka were saved with `datasets >= 4`, whose
    # `List` feature type the image's older `datasets` can't deserialize. Upgrade after the
    # package install (olmo-core does not pin `datasets`, so this is not clobbered).
    launch_config.post_setup = "pip install -U 'datasets>=4,<6'"
    # Optionally use the fused FlexAttention backend for the multimodal masks (~+8% MFU on
    # the stage-1 mixture vs the dense `torch` backend; see USE_FLEX_ATTN).
    if USE_FLEX_ATTN:
        launch_config.env_vars = list(launch_config.env_vars) + [
            BeakerEnvVar(name="OLMO2_FLEX_ATTN", value="1")
        ]

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        collator=collator_config,
        train_module=train_module_config,
        trainer=trainer_config,
        launch=launch_config,
    ).merge(overrides)

    # Fail here rather than after a Beaker job has queued, started and downloaded weights.
    if config.init_from not in INIT_FROM_CHOICES:
        raise OLMoConfigurationError(
            f"init_from={config.init_from!r} is not one of {INIT_FROM_CHOICES}"
        )

    return config


def _load_tokenizer(init_from: str):
    """Load the tokenizer for this init mode.

    A ``"scratch"`` run uses base ``Qwen/Qwen3-4B``'s tokenizer — what mm_olmo's stage 1
    uses — so nothing is read from the released Molmo2 repo. It is a drop-in for training:
    identical BPE (``encode`` matches exactly), identical ``eos_token_id`` (151645), and
    identical output for the only chat-template calls the data pipeline makes (user-turn +
    generation prompt, see ``data/multimodal/qwen3_layout.py``). Qwen3's template differs
    only for *assistant* messages, which that pipeline never passes. The image-special token
    IDs come from ``nn/vision/molmo2_tokens.py`` constants, not from the tokenizer, so their
    absence from the base vocab does not affect training (they are unmapped when *decoding*,
    which only matters for debugging).
    """
    from transformers import AutoTokenizer

    if init_from == "scratch":
        return AutoTokenizer.from_pretrained(SCRATCH_LM_ID)
    return AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def _init_weights_from_hf(model: MultimodalLM, model_cfg: MultimodalLMConfig) -> None:
    """Load converted HF Molmo2 weights into the (meta-initialised) model."""
    from transformers import AutoModelForImageTextToText

    from olmo_core.nn.vision.molmo2_loader import (
        ensure_default_rope_registered,
        molmo2_hf_state_dict_to_multimodal_lm,
        reinit_rope_buffers,
        retie_word_embeddings,
    )

    ensure_default_rope_registered()
    log.info(f"Loading HF weights from {MODEL_ID} ...")
    hf = AutoModelForImageTextToText.from_pretrained(MODEL_ID, trust_remote_code=True)
    reinit_rope_buffers(hf)
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf.state_dict(), model_cfg)
    del hf
    model.to_empty(device=get_default_device())
    model.load_state_dict(converted, strict=False)
    # `to_empty` silently un-ties tied word embeddings (Molmo2-4B); restore the share so
    # training updates the head and the embedding table as one parameter, like mm_olmo.
    retie_word_embeddings(model)
    del converted


def _init_weights_from_scratch(model: MultimodalLM, model_cfg: MultimodalLMConfig) -> None:
    """mm_olmo's true stage-1 init (``reset_with_pretrained_weights``):

    * LM  <- base ``Qwen/Qwen3-4B`` (weight-tied), via the generic HF->olmo-core converter;
      the 128 extra-token embedding rows are ``N(0, 0.02)`` (mm_olmo
      ``new_embedding_init_range``) and the LM head is tied to the embeddings.
    * vision <- base ``google/siglip2-so400m-patch14-384`` (blocks 0..24).
    * connector <- random (``reset_parameters``: ``N(0, 0.02)`` weights, zero biases —
      matches mm_olmo's pooling + projector init).
    """
    import torch
    from transformers import AutoModelForCausalLM, SiglipVisionModel

    from olmo_core.nn.hf import convert_state_from_hf
    from olmo_core.nn.vision import siglip_state_dict_to_vision_encoder
    from olmo_core.nn.vision.molmo2_loader import retie_word_embeddings

    log.info(f"[scratch init] Loading base LM weights from {SCRATCH_LM_ID} ...")
    hf_lm = AutoModelForCausalLM.from_pretrained(SCRATCH_LM_ID, dtype=torch.float32)
    lm_state = convert_state_from_hf(hf_lm.config, hf_lm.state_dict(), model_type="qwen3")
    del hf_lm

    emb = lm_state["embeddings.weight"]
    extra = model_cfg.lm.vocab_size - emb.shape[0]
    if extra < 0:
        raise RuntimeError(f"Base LM vocab {emb.shape[0]} exceeds target {model_cfg.lm.vocab_size}")
    gen = torch.Generator().manual_seed(0)  # same rows on every rank
    new_rows = torch.empty(extra, emb.shape[1], dtype=emb.dtype)
    new_rows.normal_(std=NEW_EMBEDDING_INIT_STD, generator=gen)
    full_emb = torch.cat([emb, new_rows], dim=0)

    converted = {f"lm.{k}": v for k, v in lm_state.items()}
    converted["lm.embeddings.weight"] = full_emb
    # Tied head (mm_olmo `weight_tying=True`): identical values regardless of whether the
    # target params share storage.
    converted["lm.lm_head.w_out.weight"] = full_emb.clone()
    del lm_state

    log.info(f"[scratch init] Loading base ViT weights from {SCRATCH_VIT_ID} ...")
    hf_vit = SiglipVisionModel.from_pretrained(SCRATCH_VIT_ID, dtype=torch.float32)
    converted.update(
        siglip_state_dict_to_vision_encoder(
            hf_vit.state_dict(), n_blocks=len(model.vision.blocks), prefix="vision."
        )
    )
    del hf_vit

    model.to_empty(device=get_default_device())
    missing, unexpected = model.load_state_dict(converted, strict=False)
    del converted
    # Everything except the connector must have been covered by the two base checkpoints.
    non_connector_missing = [k for k in missing if not k.startswith("connector.")]
    if non_connector_missing or unexpected:
        raise RuntimeError(
            f"[scratch init] unexpected state-dict coverage: missing={non_connector_missing[:8]} "
            f"unexpected={list(unexpected)[:8]}"
        )
    # `to_empty` above un-ties tied word embeddings, so the head and the embedding table
    # would otherwise train as two independent parameters. Restore the share so they move
    # together, matching mm_olmo's `weight_tying=True` (no-op for untied configs).
    retie_word_embeddings(model)
    log.info("[scratch init] Randomly initialising the connector ...")
    model.connector.reset_parameters()


def _build_mixture_sources(tokenizer, config: ExperimentConfig):
    """Build the caption + pointing + NLP sources and their sampling weights (mm_olmo
    SubMixture): caption gets ``1 - pointing_rate - nlp_rate``; the pointing group shares
    ``pointing_rate`` split by sqrt(size); NLP gets ``nlp_rate``."""
    import numpy as np

    p, n = config.pointing_rate, config.nlp_rate
    datasets: List = [config.dataset.build(tokenizer)]  # caption
    weights: List[float] = [max(1.0 - p - n, 0.0)]

    if p > 0:
        pointing = [
            PixMoPointsDatasetConfig(kind="basic", max_crops=MAX_CROPS).build(tokenizer),
            PixMoCountDatasetConfig(max_crops=MAX_CROPS).build(tokenizer),
            PixMoPointsDatasetConfig(kind="high_frequency", max_crops=MAX_CROPS).build(tokenizer),
            CoSynPointDatasetConfig(max_crops=MAX_CROPS).build(tokenizer),
        ]
        frac = np.sqrt(np.array([len(d) for d in pointing], dtype=np.float64))
        frac = frac / frac.sum()
        datasets += pointing
        weights += [p * float(f) for f in frac]

    if n > 0:
        datasets.append(Tulu4DatasetConfig().build(tokenizer))
        weights.append(n)

    log.info(
        "Mixture sources / weights: %s",
        [(type(d).__name__, round(w, 3)) for d, w in zip(datasets, weights)],
    )
    return datasets, weights


def train(config: ExperimentConfig):
    seed_all(config.init_seed)

    tokenizer = _load_tokenizer(config.init_from)

    model = config.model.build(init_device="meta")
    if config.init_from == "scratch":
        _init_weights_from_scratch(model, config.model)
    elif config.init_from == "molmo2":
        _init_weights_from_hf(model, config.model)
    else:  # unreachable via build_config, which validates; guards direct train() calls
        raise OLMoConfigurationError(
            f"init_from={config.init_from!r} is not one of {INIT_FROM_CHOICES}"
        )

    train_module = config.train_module.build(model)

    collator = config.collator.build()
    # Derive the data-parallel world size / rank from the train module's DP process
    # group so each rank reads its own shard (must match the trainer's DP degree).
    dp_pg = train_module.dp_process_group
    dp_world_size, dp_rank = get_world_size(dp_pg), get_rank(dp_pg)

    if config.pointing_rate > 0 or config.nlp_rate > 0:
        datasets, weights = _build_mixture_sources(tokenizer, config)
        data_loader = MixtureDataLoader(
            datasets,
            weights,
            collator,
            work_dir=config.trainer.save_folder,
            global_batch_size=config.global_batch_size,
            seed=config.data_seed,
            pack=config.pack_sequences,
            prefetch_workers=DATA_PREFETCH_WORKERS,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )
    else:
        data_loader = MultimodalDataLoader(
            config.dataset.build(tokenizer),
            collator,
            work_dir=config.trainer.save_folder,
            global_batch_size=config.global_batch_size,
            seed=config.data_seed,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )

    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    trainer.fit()


def launch(config: ExperimentConfig):
    config.launch.launch(follow=True)


if __name__ == "__main__":
    usage = f"""
Usage
=====

› python {sys.argv[0]} [dry_run|launch|train] RUN_NAME [OVERRIDES...]

  * dry_run: Print out the final config after applying overrides and exit.
  * launch:  Launch the script on Beaker as a batch job for training.
  * train:   Run training locally (usually under torchrun).

Examples
========

Print the config:
› python {sys.argv[0]} dry_run molmo2-stage1

Local synthetic smoke test:
› torchrun --nproc-per-node=1 {sys.argv[0]} train smoke \\
      --dataset.dataset_path=synthetic --trainer.max_duration.value=5

Launch on Beaker:
› python {sys.argv[0]} launch molmo2-stage1 --launch.num_nodes=1
    """.strip()

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    script, cmd, run_name, *overrides = sys.argv

    if cmd == "train":
        # Use a generous process-group timeout (gloo + NCCL). The default 15 min was the
        # exact watchdog timeout that aborted runs when a rank lagged on a collective
        # during checkpointing / bookkeeping (and W&B network stalls can add latency).
        prepare_training_environment(timeout=timedelta(minutes=60))
    else:
        prepare_cli_environment()

    config = build_config(script, run_name, overrides)
    log.info(config)

    if cmd == "train":
        train(config)
        teardown_training_environment()
    elif cmd == "launch":
        launch(config)
    elif cmd == "dry_run":
        pass
    else:
        print(usage)
        sys.exit(1)
