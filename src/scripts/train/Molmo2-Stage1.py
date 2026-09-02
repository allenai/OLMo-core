"""
Molmo2 "stage 1" caption-pretraining (reproduction of ``mm_olmo``'s captioner).

Trains the connector, vision encoder and LM on PixMoCap captions/transcripts as three
separate optimizer groups, matching mm_olmo's captioner (``ft_connector``/``ft_vit``/
``ft_llm`` all true): connector lr 2e-4 warmup 200, ViT lr 6e-6 warmup 2000, LM lr 2e-5
warmup 2000, all cosine to ``alpha_f=0.1``. Uses the float ``root_subsegments``-weighted
loss. ``--train_vit=false`` freezes the encoder and trains two groups instead. In-loop
evaluation is intentionally omitted.

Weights start from **base** checkpoints, matching mm_olmo's ``reset_with_pretrained_weights``:
the LM from the base Qwen3 backbone, the ViT from ``google/siglip2-so400m-patch14-384``, and
the connector / extra-token embeddings randomly initialised (``INIT_FROM = "scratch"``). This
is stage-1 *pretraining*. Passing ``--init_from=molmo2`` instead continues the released,
already-post-trained Molmo2 checkpoint on the stage-1 objective, which is a fine-tune —
useful for parity tests and continuation experiments, but not a stage-1 reproduction.

``--model_size`` selects the variant: ``4b`` (Qwen3-4B, the default) or ``8b`` (Qwen3-8B).

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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Tuple, cast

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
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode
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
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
)
from olmo_core.utils import get_default_device, seed_all

log = logging.getLogger(__name__)

#######################
#### CONFIGURATION ####
#######################

# Model size. Selects the architecture factory, the base LM to initialise from, and the
# released checkpoint used by `--init_from=molmo2`. Override with `--model_size=8b`.
MODEL_SIZE = "4b"


@dataclass(frozen=True)
class ModelSizeSpec:
    """Weight sources and RoPE bases for one Molmo2 variant."""

    factory: Callable[..., MultimodalLMConfig]
    """Native architecture factory, e.g. :meth:`MultimodalLMConfig.molmo2_4B`."""

    molmo2_id: str
    """Released checkpoint, used by ``--init_from=molmo2``."""

    scratch_lm_id: str
    """Base Qwen3 backbone, used by ``--init_from=scratch``."""

    scratch_rope_theta: int
    """RoPE base of ``scratch_lm_id``'s weights."""

    molmo2_rope_theta: int
    """RoPE base of ``molmo2_id``'s weights."""

    def rope_theta(self, init_from: str) -> int:
        return self.scratch_rope_theta if init_from == "scratch" else self.molmo2_rope_theta


# Per-variant weight sources, taken from the released pretrain configs at
# /weka/oe-training-default/mm-olmo/released-models-molmo2-1225/Molmo2-{4B,8B}-Pretrain/.
# `rope_theta` follows the weight source, and the two variants used different backbones.
MODEL_SIZES = {
    "4b": ModelSizeSpec(
        factory=MultimodalLMConfig.molmo2_4B,
        molmo2_id="allenai/Molmo2-4B",
        # The released 4B pretrain ran `train_captioner.py qwen3_4b_instruct`, i.e. the
        # *instruct* backbone (`Qwen/Qwen3-4B-Instruct-2507`), whose RoPE base is 5e6 — which
        # is where the released Molmo2-4B's 5e6 comes from. Its tokenizer is identical to
        # Molmo2's for everything the data pipeline uses (BPE, eos_token_id 151645, and the
        # user-turn chat template).
        scratch_lm_id="Qwen/Qwen3-4B-Instruct-2507",
        scratch_rope_theta=5_000_000,
        molmo2_rope_theta=5_000_000,
    ),
    "8b": ModelSizeSpec(
        factory=MultimodalLMConfig.molmo2_8B,
        molmo2_id="allenai/Molmo2-8B",
        # The released 8B pretrain ran `train_captioner.py qwen3_8b` — the *base* backbone,
        # RoPE base 1e6, untied. Hence the 4B/8B asymmetry: different backbones, not a
        # Molmo2-side change.
        scratch_lm_id="Qwen/Qwen3-8B",
        scratch_rope_theta=1_000_000,
        molmo2_rope_theta=1_000_000,
    ),
}

# Weight init. "scratch" is mm_olmo's true stage-1 start and the default: base Qwen3 LM
# + base SigLIP2 ViT + randomly-initialised connector / new-token embeddings. "molmo2"
# instead *continues* the released (already post-trained) Molmo2 checkpoint on the stage-1
# objective — that is a fine-tune, not stage-1 pretraining, so it is opt-in only
# (`--init_from=molmo2`), kept for parity tests and continuation experiments.
INIT_FROM = "scratch"
INIT_FROM_CHOICES = ("scratch", "molmo2")
_BOOLS = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
SCRATCH_VIT_ID = "google/siglip2-so400m-patch14-384"  # shared by every Molmo2 variant
NEW_EMBEDDING_INIT_STD = 0.02  # mm_olmo `new_embedding_init_range`
SEQUENCE_LENGTH = 2560  # fixed pad length; the released runs passed `--seq_len=2560`
USE_FLEX_ATTN = True  # fused FlexAttention backend for the multimodal masks (~+8% MFU)
PACK_SEQUENCES = True  # pack several examples per sequence
# mm_olmo `packing: {mode: dynamic_solver, buffer_size: 48, text_weight: 1.0,
# image_weight: 1.0}` — a 2D knapsack over (text tokens, image crops) rather than the
# token-only next-fit we used before. The crop budget is the second knapsack dimension: an
# 8-crop example costs at most 9 crops (1 global + 8 high-res) and ~1348 image tokens, so at
# seq 2560 the token budget binds first and this mainly bounds worst-case ViT/collator
# memory. Set below a single example's 9 crops it would force that example into its own
# mostly-padding pack.
PACK_MAX_CROPS = 3 * (1 + 8)
PACK_BUFFER_SIZE = 48
PACK_IMAGE_WEIGHT = 1.0
COMPILE_MODEL = True  # torch.compile the LM (fuses pointwise ops; one-time compile warmup)
DATA_PREFETCH_WORKERS = 4  # background threads preprocessing examples (0 = synchronous)
DL_NUM_WORKERS = 0
"""Process workers for the packed mixture DataLoader (0 = sync pack+collate, the
historical stage-1 behaviour). >0 moves packing into worker processes, which is what
took stage-2 from ~12-14k to ~20k TPS; requires packing to be enabled."""
DL_PREFETCH_FACTOR = 2
DL_PERSISTENT_WORKERS = True
MAX_CROPS = 8

# mm_olmo applies 0.1 dropout to the residual stream of RESPONSE tokens only (prompt and
# image tokens get 0.0), via a per-token drop mask in its LM block (`Dropout(mask_p=...)`).
# Implemented as `ResidualStream(masked_dropout=...)`; the mask is derived from `loss_masks`.
RESPONSE_RESIDUAL_DROPOUT = 0.1

# Instance-based batching, expressed in tokens for the token-based Trainer.
#
# `GLOBAL_BATCH_INSTANCES` matches mm_olmo's `global_train_batch_size=128` — the
# optimization-relevant quantity. 128 also divides every DP world size we run (8/16/32),
# which the previous default of 8 did not: at 16 GPUs it failed the
# `global % (microbatch x dp_world_size) == 0` check.
#
# mm_olmo uses `device_train_microbatch_size=4`; we keep 1. That is purely a
# memory/throughput knob — gradient accumulation makes the two mathematically identical for
# the same global batch — and microbatch >1 at this sequence length OOM'd without activation
# checkpointing (see the `ac_config` follow-up).
GLOBAL_BATCH_INSTANCES = 128
# mm_olmo's `device_train_microbatch_size: 4`, which its `activation_checkpointing: true`
# pays for. NOTE 128 % (4 x dp_world_size) must be 0, so this supports world sizes up to 32
# (the released 4B ran 16 GPUs, the 8B 32); at 64 use microbatch 2.
RANK_MICROBATCH_INSTANCES = 4
GLOBAL_BATCH_SIZE = GLOBAL_BATCH_INSTANCES * SEQUENCE_LENGTH
RANK_MICROBATCH_SIZE = RANK_MICROBATCH_INSTANCES * SEQUENCE_LENGTH

# Per-component LRs / warmups (mm_olmo train_captioner.py).
CONNECTOR_LR = 2e-4
LLM_LR = 2e-5
CONNECTOR_WARMUP = 200
LLM_WARMUP = 2000
# Train the ViT as well (mm_olmo captioner `ft_vit=True`), giving it its own optimizer group.
# mm_olmo's vit_betas/vit_eps/vit_weight_decay equal the LLM's, so the group only overrides
# the learning rate and inherits betas (0.9, 0.95) / eps 1e-6 / wd 0.0 from `AdamWConfig`.
# Set False to freeze the encoder and train connector + LM only.
TRAIN_VIT = True
VIT_LR = 6e-6
VIT_WARMUP = 2000
# Hold the pretrained token embeddings fixed, matching mm_olmo's `ft_embedding="lm_head"`
# default: only the 128 image-special rows (`embeddings.extra_weight`) are learned. Because
# the table is split, this is a plain whole-tensor freeze. For a *tied* backbone (Molmo2-4B)
# the base block is also the LM head, so both roles are frozen; for an untied one
# (Molmo2-8B) only the input table is, and the head keeps training — exactly mm_olmo.
FREEZE_BASE_EMBEDDINGS = True
ALPHA_F = 0.1

# Data: the canonical PixMoCap "cap" dataset (HF DatasetDict, load_from_disk). Override as needed.
DATASET_PATH = f"{PIXMO_DATASETS}/cap"
MAX_STEPS = 32000

# Stage-1 mixture rates (mm_olmo train_captioner --pointing/--nlp). Caption gets the
# Prompt family for the pointing/counting data, from the released Molmo2-4B-Pretrain
# `data_formatter` (`prompt_templates: none`, `system_prompt: style_and_length_v2`): the question
# is the bare lowercased label behind a `"<style>:"` prefix, not a natural-language template.
# The formatter defaults to the stage-2 family, which trains a model that is then out of
# distribution for pointing evals -- worth ~11 f1 on pixmo_point_eval_v3_mp, most of it
# abstention, because the "Please say 'There are none.'" instruction only exists in the
# stage-2 template.
POINTING_DATASET_KWARGS = {
    "prompt_templates": "none",
    "system_prompt": "style_and_length_v2",
    # The pointing/counting dataset classes default to `root_subsegments`, which scales an
    # example's loss by 1/sqrt(n_labels). The released run leaves this unset for every dataset
    # (`mm_preprocessor.loss_token_weighting: None`, `message_weight: None` on all four pointing
    # entries), so it weights every response token equally. Our pointing rows carry ~12.5 labels
    # per example and counting ~2.8, so the default silently gave that data ~3.5x / ~1.7x less
    # gradient weight per token than captions -- for exactly the reason the caption dataset's
    # own comment gives: the factor does not cancel out of the global sum(CE*w)/sum(w) divisor
    # when branch counts differ across examples.
    "loss_token_weighting": "none",
}

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
    model_size: str = MODEL_SIZE
    """``"4b"`` or ``"8b"`` — selects the architecture, the base LM to initialise from, and
    the released checkpoint used by ``--init_from=molmo2``."""
    data_seed: int = 95818  # mm_olmo `data.seed`
    init_seed: int = 6198  # mm_olmo `seed`
    global_batch_size: int = GLOBAL_BATCH_SIZE
    """Global batch in *tokens* (= global instances × seq len). Override to scale the batch;
    pair with ``--train_module.rank_microbatch_size`` to set sequences/forward (GEMM size)."""
    pointing_rate: float = POINTING_RATE
    """Fraction of mixture samples from pointing/counting sources (mm_olmo ``--pointing``)."""
    nlp_rate: float = NLP_RATE
    """Fraction of mixture samples from Tulu4 NLP SFT (mm_olmo ``--nlp``)."""
    train_vit: bool = TRAIN_VIT
    """Train the vision encoder in its own optimizer group (mm_olmo ``ft_vit``). When False
    the encoder is frozen and kept in eval mode."""

    freeze_base_embeddings: bool = FREEZE_BASE_EMBEDDINGS
    """Freeze the pretrained token-embedding block, training only the extra image-token rows
    (mm_olmo ``ft_embedding="lm_head"``)."""

    pack_sequences: bool = PACK_SEQUENCES
    """Pack several short examples per sequence (mm_olmo dynamic packer)."""

    init_from: str = INIT_FROM
    """``"molmo2"`` (released Molmo2-4B weights) or ``"scratch"`` (mm_olmo stage-1 init:
    base Qwen3-4B + base SigLIP2 + random connector/new embeddings)."""

    # Data-loader knobs. Exposed as fields (rather than left as module constants) so they can
    # be swept from the CLI: data loading was measured at ~11% of step time, and the packer
    # parameters are the other lever on examples/sec.
    data_prefetch_workers: int = DATA_PREFETCH_WORKERS
    """Background threads preprocessing examples (0 = synchronous)."""

    dl_num_workers: int = DL_NUM_WORKERS
    """Process workers for the packed mixture DataLoader (0 = sync pack+collate)."""

    dl_prefetch_factor: int = DL_PREFETCH_FACTOR
    """Batches prefetched per DataLoader worker (only used when ``dl_num_workers > 0``)."""

    dl_persistent_workers: bool = DL_PERSISTENT_WORKERS
    """Keep DataLoader workers alive across epochs (only used when ``dl_num_workers > 0``)."""

    pack_max_crops: int = PACK_MAX_CROPS
    """Image-crop budget per pack — the knapsack's second dimension."""

    pack_buffer_size: int = PACK_BUFFER_SIZE
    """Examples buffered before the 2D knapsack picks a pack (mm_olmo ``buffer_size``)."""

    pack_image_weight: float = PACK_IMAGE_WEIGHT
    """Objective weight per image crop in the knapsack (mm_olmo ``image_weight``)."""


def _build_model_config(model_size: str, init_from: str) -> MultimodalLMConfig:
    """Build the Molmo2-4B :class:`MultimodalLMConfig` natively (no weights, no HF read).

    ``MultimodalLMConfig.molmo2_4B`` is byte-identical to
    ``molmo2_config_from_hf_config(AutoConfig.from_pretrained(MODEL_ID))``, so a
    from-scratch run needs nothing from the released Molmo2 repo.

    The one architecture field that depends on the weight source is the RoPE base: the
    released Molmo2-4B uses ``5e6``, but base Qwen3-4B — whose weights ``"scratch"`` loads
    — was trained with ``1e6`` (mm_olmo's stage-1 ``QWEN3_4B`` also uses ``1e6``). Using
    the released value with base weights would put them on the wrong rotary base.
    """
    spec = MODEL_SIZES[model_size]
    return spec.factory(
        rope_theta=spec.rope_theta(init_from),
        response_residual_dropout=RESPONSE_RESIDUAL_DROPOUT,
    )


def _read_override(overrides: List[str], key: str, default: str) -> str:
    """Read a top-level scalar out of the raw overrides.

    The model config depends on ``model_size`` and ``init_from`` (architecture and RoPE
    base), and it is built before :meth:`Config.merge` runs, so those overrides cannot be
    read off the merged config. Later occurrences win, matching ``merge``.
    """
    value = default
    for override in overrides:
        name, _, raw = override.lstrip("-").partition("=")
        if name == key and raw:
            value = raw
    return value


def _read_bool_override(overrides: List[str], key: str, default: bool) -> bool:
    """Read a boolean top-level scalar out of the raw overrides."""
    raw = _read_override(overrides, key, str(default)).strip().lower()
    if raw not in _BOOLS:
        raise OLMoConfigurationError(f"{key}={raw!r} is not a boolean")
    return _BOOLS[raw]


def _resolve_model_spec(overrides: List[str]) -> Tuple[str, str]:
    """Resolve ``(model_size, init_from)`` from the raw overrides, validating both."""
    model_size = _read_override(overrides, "model_size", MODEL_SIZE).lower()
    init_from = _read_override(overrides, "init_from", INIT_FROM)
    if model_size not in MODEL_SIZES:
        raise OLMoConfigurationError(
            f"model_size={model_size!r} is not one of {tuple(MODEL_SIZES)}"
        )
    if init_from not in INIT_FROM_CHOICES:
        raise OLMoConfigurationError(f"init_from={init_from!r} is not one of {INIT_FROM_CHOICES}")
    return model_size, init_from


def build_config(script: str, run_name: str, overrides: List[str]) -> ExperimentConfig:
    root_dir = get_root_dir(BEAKER_CLUSTER)
    beaker_user = get_beaker_username()
    assert beaker_user is not None

    model_size, init_from = _resolve_model_spec(overrides)
    # Resolved pre-merge because it shapes `freeze_params`, the optimizer groups and the
    # per-group scheduler, all of which are built before `Config.merge` runs.
    train_vit = _read_bool_override(overrides, "train_vit", TRAIN_VIT)
    freeze_base_embeddings = _read_bool_override(
        overrides, "freeze_base_embeddings", FREEZE_BASE_EMBEDDINGS
    )
    model_config = _build_model_config(model_size, init_from)

    dataset_config = PixMoCapDatasetConfig(
        dataset_path=DATASET_PATH,
        mode="transcript_and_caption",
        max_crops=MAX_CROPS,
        max_sequence_length=SEQUENCE_LENGTH,
        # mm_olmo's captioner leaves `loss_token_weighting` at its "none" default, so every
        # response token is weighted equally. `root_subsegments` (1/sqrt(n_branches)) does not
        # cancel out of the global `sum(CE*w)/sum(w)` divisor when branch counts differ across
        # examples, so it would re-weight caption vs pointing vs NLP relative to mm_olmo.
        loss_token_weighting="none",
        seed=95818,
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
                    params=["vision_backbone.connector.*"],
                    opts=dict(lr=CONNECTOR_LR, weight_decay=0.0, scheduler_name="connector"),
                ),
                # mm_olmo's third component group (`ft_vit=True`, vit_learning_rate=6e-6).
                *(
                    [
                        OptimGroupOverride(
                            params=["vision_backbone.vision.*"],
                            opts=dict(lr=VIT_LR, weight_decay=0.0, scheduler_name="vision"),
                        )
                    ]
                    if train_vit
                    else []
                ),
            ],
        ),
        # An empty list keeps the encoder trainable *and* in train mode — the train module
        # only forces `vision.eval()` when the encoder's params are frozen.
        # mm_olmo: `activation_checkpointing: true` with `llm.activation_checkpoint:
        # whole_layer` — every block checkpointed. This is what makes microbatch 4 fit.
        # (vision/connector AC are already on by default in the train module, matching
        # the released config's vit `activation_checkpointing: true` and
        # `connector_activation_checkpointing: true`.)
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full
        ),
        freeze_params=(
            ([] if train_vit else ["vision_backbone.vision.*"])
            # The extra image-token rows live in `embeddings.extra_weight`, so freezing
            # `embeddings.weight` leaves them trainable.
            + (["lm.embeddings.weight"] if freeze_base_embeddings else [])
        ),
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        compile_model=COMPILE_MODEL,
        autocast_precision=DType.bfloat16,
        scheduler=PerGroupScheduler(
            schedulers={
                "connector": CosWithWarmup(warmup=CONNECTOR_WARMUP, alpha_f=ALPHA_F),
                **(
                    {"vision": CosWithWarmup(warmup=VIT_WARMUP, alpha_f=ALPHA_F)}
                    if train_vit
                    else {}
                ),
            },
            default=CosWithWarmup(warmup=LLM_WARMUP, alpha_f=ALPHA_F),
        ),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            # mm_olmo's released pretrain used `fsdp.precision: float` — fp32 params,
            # gradients and buffers, with bf16 only from the `amp_bf16` autocast below.
            param_dtype=DType.float32,
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

    # `_resolve_model_spec` already validated the pre-merge values; re-check the merged
    # config so a stray `--model_size`/`--init_from`/`--train_vit` cannot slip through, and
    # fail here rather than after a Beaker job has queued, started and downloaded weights.
    resolved = (model_size, init_from, train_vit, freeze_base_embeddings)
    merged = (
        config.model_size,
        config.init_from,
        config.train_vit,
        config.freeze_base_embeddings,
    )
    if merged != resolved:
        raise OLMoConfigurationError(
            f"model_size / init_from / train_vit / freeze_base_embeddings changed during merge: "
            f"{resolved} -> {merged}"
        )

    return config


def _load_tokenizer(model_size: str, init_from: str):
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

    spec = MODEL_SIZES[model_size]
    if init_from == "scratch":
        return AutoTokenizer.from_pretrained(spec.scratch_lm_id)
    return AutoTokenizer.from_pretrained(spec.molmo2_id, trust_remote_code=True)


def _init_weights_from_hf(
    model: MultimodalLM, model_cfg: MultimodalLMConfig, molmo2_id: str
) -> None:
    """Load converted HF Molmo2 weights into the (meta-initialised) model.

    :param molmo2_id: The released checkpoint to continue from, e.g. ``allenai/Molmo2-4B``.
    """
    from transformers import AutoModelForImageTextToText

    from olmo_core.nn.vision.molmo2_loader import (
        ensure_default_rope_registered,
        molmo2_hf_state_dict_to_multimodal_lm,
        reinit_rope_buffers,
        retie_word_embeddings,
    )

    ensure_default_rope_registered()
    log.info(f"Loading HF weights from {molmo2_id} ...")
    hf = AutoModelForImageTextToText.from_pretrained(molmo2_id, trust_remote_code=True)
    reinit_rope_buffers(hf)
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf.state_dict(), model_cfg)
    del hf
    model.to_empty(device=get_default_device())
    model.load_state_dict(converted, strict=False)
    # `to_empty` silently un-ties tied word embeddings (Molmo2-4B is tied; the 8B is not);
    # restore the share so training updates the head and the embedding table as one
    # parameter, like mm_olmo. A no-op for untied configs.
    retie_word_embeddings(model)
    del converted


def _init_weights_from_scratch(
    model: MultimodalLM, model_cfg: MultimodalLMConfig, scratch_lm_id: str
) -> None:
    """mm_olmo's true stage-1 init (``reset_with_pretrained_weights``):

    * LM  <- the base Qwen3 backbone, via the generic HF->olmo-core converter;
      the 128 extra-token embedding rows are ``N(0, 0.02)`` (mm_olmo
      ``new_embedding_init_range``) and the LM head is tied to the embeddings when the
      config is tied (Molmo2-4B; the 8B and Qwen3-8B are untied).
    * vision <- base ``google/siglip2-so400m-patch14-384`` (blocks 0..24), shared by every
      released Molmo2 variant.
    * connector <- random (``reset_parameters``: ``N(0, 0.02)`` weights, zero biases —
      matches mm_olmo's pooling + projector init).

    :param scratch_lm_id: The base LM to initialise from, e.g. ``Qwen/Qwen3-4B``.
    """
    import torch
    from transformers import AutoModelForCausalLM, SiglipVisionModel

    from olmo_core.nn.hf import convert_state_from_hf
    from olmo_core.nn.vision import siglip_state_dict_to_vision_encoder
    from olmo_core.nn.vision.molmo2_loader import retie_word_embeddings

    log.info(f"[scratch init] Loading base LM weights from {scratch_lm_id} ...")
    hf_lm = AutoModelForCausalLM.from_pretrained(scratch_lm_id, dtype=torch.float32)
    lm_state = convert_state_from_hf(hf_lm.config, hf_lm.state_dict(), model_type="qwen3")
    del hf_lm

    # The base checkpoint's vocab must match our base block exactly; the extra image-token
    # rows are a separate parameter, initialised here from mm_olmo's `new_embedding_init_range`.
    emb = lm_state["embeddings.weight"]
    if emb.shape[0] != model_cfg.lm.vocab_size:
        raise RuntimeError(
            f"Base LM vocab {emb.shape[0]} != target base vocab {model_cfg.lm.vocab_size}"
        )
    n_extra = model_cfg.lm.n_extra_vocab
    gen = torch.Generator().manual_seed(0)  # same rows on every rank
    extra_rows = torch.empty(n_extra, emb.shape[1], dtype=emb.dtype)
    extra_rows.normal_(std=NEW_EMBEDDING_INIT_STD, generator=gen)

    converted = {f"lm.{k}": v for k, v in lm_state.items()}
    converted["lm.embeddings.extra_weight"] = extra_rows
    # Both our LM head and the base checkpoint's span the base vocab, so it maps straight
    # across: no padding, and for a tied config the head simply *is* the base block
    # (`retie_word_embeddings` below restores the share that `to_empty` broke).
    if model_cfg.lm.tie_word_embeddings:
        converted["lm.lm_head.w_out.weight"] = emb
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
    # `load_state_dict` reports missing keys under the model's *registered* names, which the
    # legacy-key remap deliberately leaves alone (it only rewrites the dicts it is handed).
    # So derive the connector's prefix from the module tree rather than assuming a layout.
    connector_prefix = next(
        (f"{name}." for name, mod in model.named_modules() if mod is model.connector),
        "connector.",
    )
    non_connector_missing = [k for k in missing if not k.startswith(connector_prefix)]
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
            PixMoPointsDatasetConfig(
                kind="basic", max_crops=MAX_CROPS, **POINTING_DATASET_KWARGS
            ).build(tokenizer),
            PixMoCountDatasetConfig(max_crops=MAX_CROPS, **POINTING_DATASET_KWARGS).build(
                tokenizer
            ),
            PixMoPointsDatasetConfig(
                kind="high_frequency", max_crops=MAX_CROPS, **POINTING_DATASET_KWARGS
            ).build(tokenizer),
            CoSynPointDatasetConfig(max_crops=MAX_CROPS, **POINTING_DATASET_KWARGS).build(
                tokenizer
            ),
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

    tokenizer = _load_tokenizer(config.model_size, config.init_from)

    model = config.model.build(init_device="meta")
    spec = MODEL_SIZES[config.model_size]
    if config.init_from == "scratch":
        _init_weights_from_scratch(model, config.model, spec.scratch_lm_id)
    elif config.init_from == "molmo2":
        _init_weights_from_hf(model, config.model, spec.molmo2_id)
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
            pack_max_crops=config.pack_max_crops if config.pack_sequences else None,
            pack_buffer_size=config.pack_buffer_size,
            pack_image_weight=config.pack_image_weight,
            prefetch_workers=config.data_prefetch_workers,
            dl_num_workers=config.dl_num_workers,
            dl_prefetch_factor=config.dl_prefetch_factor,
            dl_persistent_workers=config.dl_persistent_workers,
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

8B from-scratch run:
› python {sys.argv[0]} launch molmo2-stage1-8b --model_size=8b

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
