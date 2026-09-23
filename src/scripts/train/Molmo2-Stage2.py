"""
Molmo2 "stage 2" SFT (reproduction of ``mm_olmo``'s ``image-only-v9`` mixture).

Fine-tunes connector + ViT + LLM on an image-only-v9/v10/v11 mixture with 16k
sequence packing. Defaults to a 3-dataset debug subset (``tulu4``, ``text_vqa``,
``chart_qa_weighted``) for smoke tests; set ``--mixture=image-only-v9`` for the full
v9 mixture, ``--mixture=image-only-v10`` for v9 + hub FineVision + DynaMath, or
``--mixture=image-only-v11`` for v10 + ChartVerse + figure captions + web reasoning
(the CharXiv / MMMU-Pro targeted mixture).

Quick local smoke test (1 GPU, debug mixture, 5 steps)::

    torchrun --nproc-per-node=1 src/scripts/train/Molmo2-Stage2.py train smoke \\
        --trainer.max_duration.value=5 --trainer.max_duration.unit=steps \\
        --global_batch_size=16384 --train_module.rank_microbatch_size=16384 \\
        --train_module.compile_model=false

Resume weights from an OLMo-core stage-1 checkpoint (model only, fresh optimizer)::

    --trainer.load_path=/path/to/stage1/run

Set ``--trainer.load_path=null`` to initialise from HF ``allenai/Molmo2-4B`` instead.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence, cast

from olmo_core.config import Config, DType
from olmo_core.data.multimodal import MixtureDataLoader, MultimodalCollatorConfig
from olmo_core.data.multimodal.chartverse import CHARTVERSE_DEFAULT_SUBSET
from olmo_core.data.multimodal.mixture_data_loader import (
    MixtureLoaderStrategy,
    resolve_loader_strategy,
)
from olmo_core.data.multimodal.mixtures.image_only_v9 import (
    build_image_only_v9_mixture,
    build_single_image_only_v9_mixture,
)
from olmo_core.data.multimodal.mixtures.image_only_v10 import (
    build_image_only_v10_mixture,
    build_single_image_only_v10_mixture,
)
from olmo_core.data.multimodal.mixtures.image_only_v11 import (
    VALIDATION_MIXTURES_V11,
    build_image_only_v11_mixture,
    build_single_image_only_v11_mixture,
)
from olmo_core.data.multimodal.mixtures.image_only_v10 import (
    build_image_only_v10_mixture,
    build_single_image_only_v10_mixture,
)
from olmo_core.data.multimodal.mixtures.mixture_pack_profiles import (
    MULTI_IMAGE_PACK_MAX_CROPS,
    SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
    get_mixture_pack_profile,
)
from olmo_core.data.multimodal.mixtures.tiers import (
    all_validation_mixtures,
    is_v10_mixture,
)
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

MODEL_ID = "allenai/Molmo2-4B"
SEQUENCE_LENGTH = 16384  # mm_olmo image-only-v9: --seq_len 16384
USE_FLEX_ATTN = True
PACK_SEQUENCES = True
COMPILE_MODEL = True
RESPONSE_LOGITS_ONLY = True
DATA_PREFETCH_WORKERS = 0
DL_NUM_WORKERS = 8
"""Process workers for packed mixture DataLoader (0 = sync pack+collate on iterator thread).

Load-bearing at the tuned crop budget, not a nicety. The packer runs inside these workers,
and a fuller pack is more packing work per step: at the previous default of 2 the 8xB300
`single-image-only-v10` sweep measured the run **43% data-loader-bound**. 8 fixes it; 16 is
indistinguishable from 8 (12,146 vs 12,140 useful TPS), so there is nothing above 8 to buy.
"""
DL_PREFETCH_FACTOR = 2
DL_PERSISTENT_WORKERS = True
MAX_CROPS = 8
# Per-pack crop capacity for the 2D-knapsack packer. Defaults below are overridden per
# mixture tier in ``mixture_pack_profiles`` (see ``get_mixture_pack_profile``).
PACK_MAX_CROPS = MULTI_IMAGE_PACK_MAX_CROPS
PACK_SHORTCUT_MAX_LEN_IMAGES = False
EST_TOKENS_PER_EXAMPLE = 1500  # packed 16k sequences; tune if batch counts look off

# mm_olmo train_image_video_sft.py (image-only-v9): global 128, microbatch 2 per GPU.
#
# GLOBAL_BATCH_INSTANCES counts *packs*, not examples, so it does not mean the same thing
# at every crop budget. At the single-image tier's tuned budget a pack holds ~13.1 examples
# instead of ~4.1, so 128 packs is ~1,677 examples/step rather than ~524 against an
# unchanged LR schedule. **Choose the pack count deliberately** -- the validated 8-GPU runs
# used 32 packs (419 examples/step) and 48 packs (629); no one has run 128 packs at the
# tuned budget. Constraint: global packs must divide `dp_world_size x
# RANK_MICROBATCH_INSTANCES` (16 at 8 GPUs, 64 at 32 GPUs), else the trainer aborts with
# "global batch size must be divisible by micro-batch size x DP world size".
# See STAGE2_FAST_CONFIG.md.
#
# 32 packs = ~419 examples/step at the tuned crop budget, slightly *below* the 524 the old
# `crops=25` default produced, so adopting this config does not smuggle in a batch-size
# increase. `crops=80` was measured at 32 packs and still gave +85.6% (vs +87.3% at 48), so
# the speedup does not depend on a large batch.
#
# NOTE this value is only valid up to 16 GPUs: 32 % (2 x 32) != 0, so a 32-GPU run must
# raise it (64 packs is the smallest valid choice there, 838 examples/step). `launch()`
# checks this against the requested GPU count and refuses rather than letting the trainer
# abort after the job has been scheduled.
GLOBAL_BATCH_INSTANCES = 32
# Keep at 2. `mb=1` is not a safe fallback, it is a ~19% throughput loss at identical
# occupancy and examples/step (10,204 vs 12,140 useful TPS at crops=50), and mb=2 at the
# tuned crop budget does *not* OOM -- an untested assumption that it would once cost half
# the measured win. `mb=3` does OOM on B300 (~261/268 GiB).
RANK_MICROBATCH_INSTANCES = 2
GLOBAL_BATCH_SIZE = GLOBAL_BATCH_INSTANCES * SEQUENCE_LENGTH
RANK_MICROBATCH_SIZE = RANK_MICROBATCH_INSTANCES * SEQUENCE_LENGTH

# Per-component LRs / warmups (mm_olmo SFT).
CONNECTOR_LR = 5e-6
VISION_LR = 5e-6
LLM_LR = 1e-5
COMPONENT_WARMUP = 200
ALPHA_F = 0.1

MAX_STEPS = 300_000

# Extra image-SFT sources beyond mm_olmo's image-only-v9 mixture, OFF by default.
# Rates are fractions of the total mixture: the 43 official sources are scaled by
# (1 - sum(extra rates)) and the extras are appended, so enabling them dilutes the
# official recipe proportionally. All read parquet shards straight from weka.
#
# MMFineReason-SFT: multimodal reasoning; supervision is the `<answer>` content of
# `original_answer` (the `<think>` trace is dropped) — see MMFineReasonDataset.
MMFINEREASON_RATE = 0.0
#
# FineVision configs -> sampling rate. Any config downloaded under FINEVISION_ROOT works;
# these five are verified. Rows are single-turn, one image each.
#   visualwebinstruct(filtered)  263,581  web visual instruction
#   mavis_math_rule_geo           99,986  synthetic geometry with CoT answers
#   mavis_math_metagen            87,348  synthetic math with CoT answers
#   geo170k(align)                35,297  geometry caption/alignment
#   geo170k(qa)                   12,101  geometry multiple-choice
# NOTE: ~13% of MMFineReason rows are re-annotations of visualwebinstruct(filtered) images,
# so enabling both double-samples those images (with different answers).
FINEVISION_RATES: dict = {
    "visualwebinstruct(filtered)": 0.0,
    "mavis_math_rule_geo": 0.0,
    "mavis_math_metagen": 0.0,
    "geo170k(align)": 0.0,
    "geo170k(qa)": 0.0,
}
# Optional quality floor applied to every enabled FineVision config (1-5 per-row minimums;
# None = keep all). `min_visual_dependency` is the most useful: it keeps answers that
# actually need the image. Do NOT use `min_image_correspondence` here — it is 1 for 75% of
# geo170k(qa) and 63% of mavis_math_metagen rows, which would discard most of them.
FINEVISION_MIN_VISUAL_DEPENDENCY: Optional[int] = None

# Default init: latest step under this OLMo-core stage-1 run (model weights only).
# HARDCODED personal checkpoint (donovanc's stage-1 run on weka). Point this at your own
# stage-1 run via --trainer.load_path=/path/to/run, or --trainer.load_path=null to
# initialise from the released HF Molmo2-4B weights instead.
DEFAULT_LOAD_PATH = (
    "/weka/oe-training-default/donovanc/molmofication/checkpoints/"
    "molmo2-pretraining-olmo-core/8-gpu-holmes/8-gpu-holmes-olmo-core-stable"
)

# Phase-p0 ship-stack env vars, validated in the 8/16-GPU A/B sweeps (see
# launch_scripts/donovan/beaker/sft/, gitignored on this branch). Applied only to
# launch_config.env_vars below, not the in-code defaults in distributed/utils.py or
# nn/vision/multimodal.py: those are shared with Stage 1 and have a much wider blast
# radius. A local `train` run under torchrun therefore still gets the in-code defaults,
# not these ship-stack values - that's intended.
SHIP_STACK_ENV: Dict[str, str] = {
    # NOT a tuning knob -- a correctness requirement at the single-image tier's crop
    # budget. Without it the run OOMs at around step 1,307 AFTER PASSING three separate
    # 100-step smokes, reporting ~39 GiB reserved-but-unallocated. A 100-step smoke cannot
    # validate memory stability here; the only evidence that matters is a long run.
    # The raw-YAML launch path always set this; the Gantry path never did until it was
    # added here, which is exactly how the ~3-hour tail failure got shipped once already.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "VIT_CROP_MICROBATCH": "32",  # +3.0% TPS (20,185 vs 19,594, 8-GPU mb2)
    "MM_FSDP_RESHARD_AFTER_FORWARD": "0",  # +2.4% TPS (13,281 vs 12,973, 8-GPU)
    # MM_FSDP_IMAGE_ALIGN_HACK is deliberately NOT set here (i.e. the align tie stays on).
    #
    # Review flagged that disabling it could deadlock: `tulu4` is ~14% of
    # image-only-v9/v10 and text-only, so a rank can draw an all-text pack, and FSDP2
    # skips a param group's reduce-scatter when none of its parameters has a gradient.
    # An asymmetry there hangs the DP group — reproduced on 2 gloo processes, so it needs
    # neither multiple nodes nor GPUs, only two DP ranks.
    #
    # For *this* forward it turns out not to fire: the splice runs unconditionally when
    # `images is not None` (the collator always supplies a dummy zero crop) and a masked
    # index_put keeps the autograd edge, so the vision params get present-but-zero
    # gradients and the collective is issued on every rank. See
    # `src/test/nn/vision/align_tie_test.py`.
    #
    # Turning it off is *safe* — confirmed on 2 GPUs under real FSDP2 + torch.compile
    # (holmes experiment 01M2622FH6WTMHZAD6A9WT5D7F): tie off with asymmetric batches gave
    # [10, 10] symmetric collectives and no None gradients.
    #
    # It is just not worth anything here. The 8-GPU A/Bs: baseline 12,973 TPS; no-reshard
    # alone 13,281 (+2.4%); align-hack-off alone 13,299 (+2.5%); **both together 13,106**.
    # The two do not stack, so on top of MM_FSDP_RESHARD_AFTER_FORWARD=0 (which is shipped)
    # disabling the tie measured slightly *worse*, not better. Single runs with no repeats,
    # so read that as "no measured benefit" rather than a real regression — either way
    # there is no throughput case for turning it off.
}

# Beaker.
BEAKER_CLUSTER = "ai2/jupiter"
NUM_NODES = 1
BEAKER_WORKSPACE = "ai2/OLMo-core"
BEAKER_BUDGET = "ai2/oe-other"

WANDB_PROJECT: Optional[str] = "molmo2-stage2"
WANDB_ENTITY: Optional[str] = None

###########################
#### END CONFIGURATION ####
###########################


@dataclass
class ExperimentConfig(Config):
    launch: BeakerLaunchConfig
    model: MultimodalLMConfig
    collator: MultimodalCollatorConfig
    train_module: MultimodalTransformerTrainModuleConfig
    trainer: TrainerConfig
    model_id: str = MODEL_ID
    data_seed: int = 50189
    init_seed: int = 6198
    global_batch_size: int = GLOBAL_BATCH_SIZE
    mixture: str = "debug"
    """Mixture tier — see ``VALIDATION_MIXTURES`` / ``VALIDATION_MIXTURES_V10``."""
    pack_sequences: bool = PACK_SEQUENCES
    pack_max_crops: int = PACK_MAX_CROPS
    pack_shortcut_max_len_images: bool = PACK_SHORTCUT_MAX_LEN_IMAGES
    prefetch_workers: int = DATA_PREFETCH_WORKERS
    """Background threads for example preprocessing (0 = synchronous). Ignored when ``dl_num_workers > 0``."""
    dl_num_workers: int = DL_NUM_WORKERS
    """PyTorch DataLoader process workers for packed stage-2 mixtures (mm_olmo parity)."""
    dl_prefetch_factor: int = DL_PREFETCH_FACTOR
    dl_persistent_workers: bool = DL_PERSISTENT_WORKERS
    mmfinereason_rate: float = MMFINEREASON_RATE
    """Mixture fraction for MMFineReason-SFT (0 disables). The official image-only-v9
    sources are scaled by ``1 - (mmfinereason_rate + finevision_rate)``."""
    mmfinereason_supervise_cot: bool = False
    """Supervise MMFineReason's ``<think>`` derivation instead of only its ``<answer>``.

    Needs no staging -- the trace is already in the column the loader reads -- so this is
    the cheap half of the process-supervision question, with the answer-only mmfinereason
    10k diet curve as an exact control."""
    mmfinereason_cot_scratchpad: bool = False
    """With ``mmfinereason_supervise_cot``, keep the derivation inside ``<think>...</think>``
    and supervise a bare answer after it, rather than training the derivation as the graded
    prose. Requires olmo-eval's ``strip_reasoning_trace`` on the eval side."""
    finevision_rate: float = 0.0
    """Total mixture fraction for the five verified FineVision configs, split evenly
    across them via ``FINEVISION_RATES`` keys (0 disables)."""
    caption_subsets: List[str] = field(default_factory=list)
    """Caption sources to append, by ``CaptionDatasetConfig`` subset directory name --
    e.g. ``[omniscience]`` or ``[omniscience-full]``. ``caption_rate`` is split evenly
    across them. The subset is a directory under ``$MOLMO_EXPERIMENT_DATA_DIR/captions/``,
    so staging a bigger corpus needs no code change, only a new directory."""
    caption_rate: float = 0.0
    """Total mixture fraction for ``caption_subsets`` (0 disables)."""
    chartverse_rate: float = 0.0
    """Mixture fraction for ChartVerse (0 disables)."""
    chartverse_subset: str = CHARTVERSE_DEFAULT_SUBSET
    """ChartVerse subset directory under ``$MOLMO_EXPERIMENT_DATA_DIR/chartverse/``.
    The loader default pins the 250k copy, so the larger staged copies
    (``sft_600k-full``, ``sft_1800k``) are unreachable without this knob."""
    chartverse_supervise_cot: bool = False
    """Supervise ChartVerse's ``cot_solution`` derivation instead of the bare ``answer``.

    Needs a row-aligned derivation sidecar (``<subset>-cot``, built by
    ``launch_scripts/donovan/dev/stage_chartverse_cot.py``). The supervised target goes
    from ~8 characters to ~4k tokens, so ChartVerse rows per step fall roughly 3x at fixed
    GPU-hours: compare against an answer-only arm at matched rows consumed, not steps."""
    chartverse_cot_sidecar: Optional[str] = None
    """Explicit derivation-sidecar directory (defaults to ``<subset path>-cot``)."""
    chartgym_rate: float = 0.0
    """Mixture fraction for ChartGym, the synthetic chart-capability corpus (0 disables).

    ChartGym trains four visual capabilities with exact, code-derived ground truth, and
    deliberately withholds one whole visual primitive (panel layout) so that CharXiv
    templates 18/19 act as a transfer readout rather than a fit statistic. It carries
    inapplicable questions at CharXiv's measured per-template rates -- 25.0% of the
    benchmark's descriptive gold answers are "Not Applicable", and the checkpoint is worst
    exactly there -- phrased naturally, never as the benchmark's literal answer token."""
    chartgym_subset: str = "train-v1"
    """ChartGym corpus directory under ``$MOLMO_EXPERIMENT_DATA_DIR/chartgym/``."""
    chartgym_max_rows: Optional[int] = None
    """Optional row cap, for exposure-matched ablations against a smaller corpus."""
    ignore_shuffle_algo_version_mismatch: bool = False
    """Resume a checkpoint whose mixture shuffle algorithm predates
    ``MixtureDataLoader.SHUFFLE_ALGO_VERSION``. Off by default: such a resume regenerates a
    different epoch and skips into it at the old batch offset, silently repeating some
    examples and omitting others. Set it to accept that for a checkpoint written before
    the version field existed (the alternative is restarting the run)."""

    def __post_init__(self):
        # Validate only — never write back into the declared fields. `Config.merge` is
        # `as_dict()` -> apply overrides -> `from_dict()`, so a `__post_init__` that
        # mutated its inputs would feed the *already-normalized* values into the next
        # merge: `--dl_num_workers=0` on a config with `prefetch_workers=4` would see a
        # prefetch count this hook had already zeroed, and silently drop thread prefetch.
        # The effective values are derived on demand instead.
        resolve_loader_strategy(
            pack=self.pack_sequences,
            pack_max_crops=self.pack_max_crops if self.pack_sequences else None,
            prefetch_workers=self.prefetch_workers,
            dl_num_workers=self.effective_dl_num_workers,
        )

    @property
    def effective_dl_num_workers(self) -> int:
        """``dl_num_workers``, forced to 0 when sequence packing is off.

        The multiprocess loader path runs the packer inside its workers, so
        ``dl_num_workers > 0`` requires ``pack=True``. Since ``DL_NUM_WORKERS`` defaults
        to a non-zero value, ``--pack_sequences=false`` on its own would otherwise be an
        unconditional startup error; fall back to the synchronous loader rather than
        making callers pass ``--dl_num_workers=0`` too.
        """
        if not self.pack_sequences and self.dl_num_workers > 0:
            log.warning(
                "pack_sequences=false is incompatible with dl_num_workers=%d "
                "(multiprocess workers run the packer); using dl_num_workers=0.",
                self.dl_num_workers,
            )
            return 0
        return self.dl_num_workers

    @property
    def loader_strategy(self) -> MixtureLoaderStrategy:
        """Which of the two prefetch mechanisms this config resolves to."""
        strategy, _, _ = resolve_loader_strategy(
            pack=self.pack_sequences,
            pack_max_crops=self.pack_max_crops if self.pack_sequences else None,
            prefetch_workers=self.prefetch_workers,
            dl_num_workers=self.effective_dl_num_workers,
        )
        return strategy


def _build_model_config() -> MultimodalLMConfig:
    from transformers import AutoConfig

    from olmo_core.nn.vision.molmo2_loader import (
        ensure_default_rope_registered,
        molmo2_config_from_hf_config,
    )

    ensure_default_rope_registered()
    hf_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config = molmo2_config_from_hf_config(hf_config)
    # mm_olmo SFT fine-tuning setting: llm.residual_dropout = 0.1 (with
    # response_residual_dropout = 0.0, so a single residual-stream dropout matches).
    config.lm.block.dropout = 0.1
    return config


def _all_validation_mixtures():
    # v11 tiers live only on this branch; the shared resolver knows v9/v10.
    return {**all_validation_mixtures(), **VALIDATION_MIXTURES_V11}


def _mixture_dataset_names(mixture: str) -> Optional[Sequence[str]]:
    all_mixtures = _all_validation_mixtures()
    if mixture not in all_mixtures:
        known = ", ".join(sorted(all_mixtures))
        raise ValueError(f"Unknown mixture {mixture!r}; use one of: {known}")
    return all_mixtures[mixture]


def _build_mixture(tokenizer, config: ExperimentConfig):
    """Build the tier's sources and weights.

    Which registry a tier draws from (v10 vs v9, full vs single-image-only) is decided by
    ``mixtures.tiers``, not here — the pack profile needs the same answer, and when the
    two derived it independently they drifted (see ``mixture_pack_profiles``).
    """
    names_filter = _mixture_dataset_names(config.mixture)
    single_image_only = config.mixture in (
        "single-image-only-v9",
        "single-image-only-v10",
        "single-image-only-v11",
    )

    # `Any`: mypy 1.3 rejects assigning a plain function to a `Callable[...]`
    # variable. The four builders share a signature; the dispatch is the point.
    build: Any
    # v11 tiers are branch-local; test them before the shared v10/v9 resolver, or a
    # v11 tier falls through and silently drops every v11-only source.
    if config.mixture == "single-image-only-v11" or config.mixture in VALIDATION_MIXTURES_V11:
        build = (
            build_single_image_only_v11_mixture
            if single_image_only
            else build_image_only_v11_mixture
        )
    elif is_v10_mixture(config.mixture):
        build = (
            build_single_image_only_v10_mixture
            if single_image_only
            else build_image_only_v10_mixture
        )
    else:
        build = (
            build_single_image_only_v9_mixture if single_image_only else build_image_only_v9_mixture
        )

    datasets, weights, names = build(
        tokenizer,
        seed=config.data_seed,
        dataset_names=names_filter,
        max_sequence_length=SEQUENCE_LENGTH,
    )
    # Applies to every mixture tier, not just v9: these rates are documented as live
    # knobs, and wiring them into only one branch meant
    # `--mixture=single-image-only-v10 --mmfinereason_rate=0.05` started up, logged a
    # mixture with no MMFineReason in it, and trained the wrong distribution silently.
    datasets, weights, names = _append_extra_sft_sources(
        config, tokenizer, datasets, weights, names
    )
    log.info(
        "Mixture %s sources / weights: %s",
        config.mixture,
        list(zip(names, [round(w, 4) for w in weights])),
    )
    return datasets, weights, names


def _override_sets(overrides: List[str], field: str) -> bool:
    prefix = f"--{field}="
    return any(item.startswith(prefix) for item in overrides)


def _apply_mixture_pack_profile(config: ExperimentConfig, overrides: List[str]) -> ExperimentConfig:
    profile = get_mixture_pack_profile(config.mixture)
    if not _override_sets(overrides, "pack_max_crops"):
        config.pack_max_crops = profile.pack_max_crops
    if not _override_sets(overrides, "pack_shortcut_max_len_images"):
        config.pack_shortcut_max_len_images = profile.pack_shortcut_max_len_images
    if profile.description:
        log.info(
            "Mixture %s pack profile: pack_max_crops=%d shortcut_max_len_images=%s (%s)",
            config.mixture,
            config.pack_max_crops,
            config.pack_shortcut_max_len_images,
            profile.description,
        )
    return config


def build_config(script: str, run_name: str, overrides: List[str]) -> ExperimentConfig:
    root_dir = get_root_dir(BEAKER_CLUSTER)
    beaker_user = get_beaker_username()
    assert beaker_user is not None

    model_config = _build_model_config()

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
                OptimGroupOverride(
                    params=["vision_backbone.vision.*"],
                    opts=dict(lr=VISION_LR, weight_decay=0.0, scheduler_name="vision"),
                ),
            ],
        ),
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        compile_model=COMPILE_MODEL,
        autocast_precision=DType.bfloat16,
        scheduler=PerGroupScheduler(
            schedulers={
                "connector": CosWithWarmup(warmup=COMPONENT_WARMUP, alpha_f=ALPHA_F),
                "vision": CosWithWarmup(warmup=COMPONENT_WARMUP, alpha_f=ALPHA_F),
            },
            default=CosWithWarmup(warmup=COMPONENT_WARMUP, alpha_f=ALPHA_F),
        ),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_blocks,
            block_interval=2,
        ),
        response_logits_only=RESPONSE_LOGITS_ONLY,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"{root_dir}/checkpoints/{beaker_user.lower()}/{run_name}",
            save_overwrite=True,
            load_path=DEFAULT_LOAD_PATH,
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
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
    )

    launch_config = build_launch_config(
        name=run_name,
        root_dir=root_dir,
        cmd=[script, "train", run_name, *overrides],
        cluster=BEAKER_CLUSTER,
        workspace=BEAKER_WORKSPACE,
        budget=BEAKER_BUDGET,
        num_nodes=NUM_NODES,
    )
    launch_config.aws_config_secret = None
    launch_config.aws_credentials_secret = None
    launch_config.google_credentials_secret = None
    launch_config.env_secrets = [
        s for s in launch_config.env_secrets if s.name in ("BEAKER_TOKEN", "WANDB_API_KEY")
    ]
    launch_config.post_setup = "pip install -U 'datasets>=4,<6'"
    if USE_FLEX_ATTN:
        launch_config.env_vars = list(launch_config.env_vars) + [
            BeakerEnvVar(name="OLMO2_FLEX_ATTN", value="1")
        ]
    launch_config.env_vars = list(launch_config.env_vars) + [
        BeakerEnvVar(name=name, value=value) for name, value in SHIP_STACK_ENV.items()
    ]

    # Data roots the container cannot infer. MOLMO_EXPERIMENT_DATA_DIR deliberately has no
    # default (paths.py: a library default must not point at one person's scratch space),
    # so without forwarding it a v10/v11 mixture dies building its DynaMath / FineVision
    # sources *inside* the job, after it has queued for a node and started. Forward
    # whatever the submitting shell has.
    for _var in (
        "MOLMO_DATA_DIR",
        "MOLMO_EXPERIMENT_DATA_DIR",
        "MOLMO_CACHE_DIR",
        "FINEVISION_ROOT",
    ):
        _value = os.environ.get(_var)
        if _value:
            launch_config.env_vars = list(launch_config.env_vars) + [
                BeakerEnvVar(name=_var, value=_value)
            ]

    return _apply_mixture_pack_profile(
        ExperimentConfig(
            model=model_config,
            collator=collator_config,
            train_module=train_module_config,
            trainer=trainer_config,
            launch=launch_config,
        ).merge(overrides),
        overrides,
    )


def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def _init_weights_from_hf(model: MultimodalLM, model_cfg: MultimodalLMConfig) -> None:
    from transformers import AutoModelForImageTextToText

    from olmo_core.nn.vision.molmo2_loader import (
        ensure_default_rope_registered,
        molmo2_hf_state_dict_to_multimodal_lm,
        reinit_rope_buffers,
        retie_word_embeddings,
    )

    ensure_default_rope_registered()
    log.info("Loading HF weights from %s ...", MODEL_ID)
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


def _append_extra_sft_sources(config: "ExperimentConfig", tokenizer, datasets, weights, names):
    """Append MMFineReason / FineVision / caption / ChartVerse sources at their rates.

    No-op when every rate is 0, which is the default, so an unmodified launch of any
    mixture tier is unchanged.

    ``config.finevision_rate`` is split evenly across the configs named in
    ``FINEVISION_RATES``; a per-config rate in that dict adds on top (module-level
    fine-tuning knob for uneven splits). ``config.caption_rate`` is likewise split evenly
    across ``config.caption_subsets``.

    The base mixture's weights are rescaled by ``1 - extra_total``, so the appended rates
    are absolute fractions of the final mixture and the remainder is the base tier -- i.e.
    ``--mixture=single-image-only-v9 --caption_rate=0.65`` is "65% captions, 35% v9 replay".
    """
    from olmo_core.data.multimodal import (
        CaptionDatasetConfig,
        ChartVerseDatasetConfig,
        FineVisionDatasetConfig,
        MMFineReasonDatasetConfig,
    )
    from olmo_core.data.multimodal.paths import require_experiment_data_dir

    per_config = config.finevision_rate / max(len(FINEVISION_RATES), 1)
    fv = {
        name: rate + per_config for name, rate in FINEVISION_RATES.items() if rate + per_config > 0
    }
    caption_subsets = list(config.caption_subsets or [])
    per_caption = config.caption_rate / len(caption_subsets) if caption_subsets else 0.0
    captions = {name: per_caption for name in caption_subsets if per_caption > 0}
    mmfr_rate = config.mmfinereason_rate
    cv_rate = config.chartverse_rate
    cg_rate = config.chartgym_rate
    extra_total = (
        mmfr_rate + cv_rate + cg_rate + sum(fv.values()) + sum(captions.values())
    )
    if extra_total <= 0:
        return datasets, weights, names
    if extra_total >= 1:
        raise ValueError(f"Extra SFT rates sum to {extra_total}; must be < 1")

    # Appending a source the base tier already contains would silently double-count it at
    # two different weights (v11 already holds chartverse, mmfinereason and the caption
    # sources), which reads as a mixture-weight bug much later. Fail at build time instead.
    appended = (
        (["mmfinereason"] if mmfr_rate > 0 else [])
        + (["chartverse"] if cv_rate > 0 else [])
        + (["chartgym"] if cg_rate > 0 else [])
        + list(captions)
        + [f"finevision[{name}]" for name in fv]
    )
    clashes = sorted(set(appended) & set(names))
    if clashes:
        raise ValueError(
            f"Sources {clashes} are already in mixture {config.mixture!r}; appending them "
            "again would double-count them. Use a base tier that excludes them "
            "(e.g. single-image-only-v9) or drop the corresponding rate flag."
        )

    datasets = list(datasets)
    weights = [w * (1.0 - extra_total) for w in weights]
    names = list(names)
    if cg_rate > 0:
        # ChartGym is staged in FineVision's own schema (`texts` list-of-struct + `images`),
        # so it needs no loader module of its own: FineVisionDataset._build emits a flat
        # turn list, which encode_sft_example splits into independent branches sharing ONE
        # image prefix. ~16 questions per chart therefore cost one image encode, and
        # root_subsegments_root_tokens weights each branch 1/sqrt(16) so a 16-question
        # figure does not carry 16x the gradient of a 1-question row.
        datasets.append(
            FineVisionDatasetConfig(
                dataset_path=os.path.join(
                    require_experiment_data_dir("the staged ChartGym corpus"),
                    "chartgym",
                    config.chartgym_subset,
                ),
                # The row-filter index cache is keyed by row count, so a regenerated corpus
                # with the same count would silently reuse a stale index.
                index_cache_dir="",
                max_rows=config.chartgym_max_rows,
                max_crops=MAX_CROPS,
                max_sequence_length=SEQUENCE_LENGTH,
            ).build(tokenizer)
        )
        weights.append(cg_rate)
        names.append("chartgym")
    if cv_rate > 0:
        datasets.append(
            ChartVerseDatasetConfig(
                subset=config.chartverse_subset,
                max_crops=MAX_CROPS,
                max_sequence_length=SEQUENCE_LENGTH,
                supervise_cot=config.chartverse_supervise_cot,
                cot_sidecar=config.chartverse_cot_sidecar,
            ).build(tokenizer)
        )
        weights.append(cv_rate)
        names.append("chartverse")
    for subset, rate in captions.items():
        datasets.append(
            CaptionDatasetConfig(
                subset=subset,
                max_crops=MAX_CROPS,
                max_sequence_length=SEQUENCE_LENGTH,
            ).build(tokenizer)
        )
        weights.append(rate)
        names.append(subset)
    if mmfr_rate > 0:
        datasets.append(
            MMFineReasonDatasetConfig(
                max_crops=MAX_CROPS,
                max_sequence_length=SEQUENCE_LENGTH,
                supervise_cot=config.mmfinereason_supervise_cot,
                cot_scratchpad=config.mmfinereason_cot_scratchpad,
            ).build(tokenizer)
        )
        weights.append(mmfr_rate)
        names.append("mmfinereason")
    for cfg_name, rate in fv.items():
        datasets.append(
            FineVisionDatasetConfig(
                config_name=cfg_name,
                max_crops=MAX_CROPS,
                max_sequence_length=SEQUENCE_LENGTH,
                min_visual_dependency=FINEVISION_MIN_VISUAL_DEPENDENCY,
                # These five configs are documented above as one image per row, and the
                # v10 registry builder asserts the same for its own subsets. Enforce it
                # rather than relying on it: sources appended here are invisible to
                # `get_mixture_pack_profile`, which derives the crop budget from the
                # tier's *registry* sources — so a multi-image row sneaking in via
                # `--finevision_rate` would exceed a single-image tier's budget exactly
                # the way the old hand-maintained profile table did.
                require_single_image=True,
            ).build(tokenizer)
        )
        weights.append(rate)
        names.append(f"finevision[{cfg_name}]")
    return datasets, weights, names


def _warn_if_allocator_unconfigured(config: ExperimentConfig) -> None:
    """Warn when running a raised crop budget without ``expandable_segments``.

    ``SHIP_STACK_ENV`` only reaches Beaker launches. A ``torchrun`` invocation of this
    script gets the ambient environment, and without this allocator setting a run at the
    tuned crop budget OOMs at around step 1,307 -- long after every smoke test has passed.
    Fail loudly at step 0 instead of silently three hours in.
    """
    if config.pack_max_crops <= SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS:
        return
    if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        return
    log.warning(
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is NOT set, and pack_max_crops=%d "
        "is above the one-example floor of %d. This configuration has been observed to OOM "
        "at ~step 1,307 after passing three separate 100-step smokes. Export it before "
        "training (Beaker launches get it from SHIP_STACK_ENV automatically).",
        config.pack_max_crops,
        SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
    )


def train(config: ExperimentConfig):
    seed_all(config.init_seed)

    tokenizer = _load_tokenizer()

    model = config.model.build(init_device="meta")
    if config.trainer.load_path:
        log.info("Deferring weight init to checkpoint load_path=%s", config.trainer.load_path)
        model.to_empty(device=get_default_device())
        # `to_empty` breaks weight tying (Molmo2-4B). Restore the share *before* FSDP
        # wrapping and the checkpoint load so both state-dict keys fill one parameter.
        # (Requires the stage-1 checkpoint itself to hold consistent tied weights.)
        from olmo_core.nn.vision.molmo2_loader import retie_word_embeddings

        retie_word_embeddings(model)
    else:
        _init_weights_from_hf(model, config.model)

    train_module = config.train_module.build(model)
    collator = config.collator.build()

    dp_pg = train_module.dp_process_group
    dp_world_size, dp_rank = get_world_size(dp_pg), get_rank(dp_pg)

    datasets, weights, dataset_names = _build_mixture(tokenizer, config)
    log.info(
        "Stage 2 packing: pack=%s pack_max_crops=%d shortcut_max_len_images=%s vit_crop_microbatch=%s dl_num_workers=%d",
        config.pack_sequences,
        config.pack_max_crops,
        config.pack_shortcut_max_len_images,
        config.model.vit_crop_microbatch,
        config.effective_dl_num_workers,
    )
    _warn_if_allocator_unconfigured(config)
    prefetch_workers = config.prefetch_workers
    data_loader = MixtureDataLoader(
        datasets,
        weights,
        collator,
        work_dir=config.trainer.save_folder,
        global_batch_size=config.global_batch_size,
        seed=config.data_seed,
        ignore_shuffle_algo_version_mismatch=config.ignore_shuffle_algo_version_mismatch,
        pack=config.pack_sequences,
        pack_max_crops=config.pack_max_crops if config.pack_sequences else None,
        pack_shortcut_max_len_images=config.pack_shortcut_max_len_images,
        est_tokens_per_example=EST_TOKENS_PER_EXAMPLE,
        prefetch_workers=prefetch_workers,
        dl_num_workers=config.effective_dl_num_workers,
        dl_prefetch_factor=config.dl_prefetch_factor,
        dl_persistent_workers=config.dl_persistent_workers,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        dataset_names=dataset_names,
    )

    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    trainer.fit()


def launch(config: ExperimentConfig):
    # Fail here rather than on an allocated node. MOLMO_EXPERIMENT_DATA_DIR has no default
    # (paths.py), and the v10/v11 mixtures build DynaMath / FineVision sources from it, so
    # without it the job queues for a node, starts, installs its dependencies, builds the
    # model, and only then dies in dataset build. build_config forwards the variable into
    # the container when it is set.
    if is_v10_mixture(config.mixture) and not os.environ.get("MOLMO_EXPERIMENT_DATA_DIR"):
        raise OLMoConfigurationError(
            f"Mixture {config.mixture!r} includes sources under the experimental-data "
            "staging root, but MOLMO_EXPERIMENT_DATA_DIR is not set in this shell, so the "
            "job would fail at dataset build only after being scheduled. Export it before "
            "launching, e.g. /weka/oe-training-default/donovanc/molmo-experimental-data"
        )

    # The trainer asserts `global_batch_size % (rank_microbatch_size * dp_world_size) == 0`,
    # but only once every rank is up -- so a bad pack count costs a scheduling round trip.
    # Reduce it to packs and check it here, where the fix is free.
    world = max(1, config.launch.num_nodes) * max(1, config.launch.num_gpus)
    packs = config.global_batch_size // SEQUENCE_LENGTH
    per_step = config.train_module.rank_microbatch_size // SEQUENCE_LENGTH
    stride = per_step * world
    if stride and packs % stride:
        valid = [n for n in range(stride, 4 * stride + 1, stride)]
        raise OLMoConfigurationError(
            f"global batch of {packs} packs is not divisible by "
            f"rank_microbatch_instances ({per_step}) x world size ({world}) = {stride}, so "
            f"the trainer would abort after this job is scheduled. Set --global_batch_size "
            f"to one of {[v * SEQUENCE_LENGTH for v in valid]} ({valid} packs)."
        )

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
› python {sys.argv[0]} dry_run molmo2-stage2-debug

1-GPU debug smoke (5 steps, batch=1×16k):
› torchrun --nproc-per-node=1 {sys.argv[0]} train smoke \\
      --trainer.max_duration.value=5 --global_batch_size=16384 \\
      --train_module.rank_microbatch_size=16384 --train_module.compile_model=false

Full image-only-v9 mixture:
› torchrun --nproc-per-node=8 {sys.argv[0]} train my-sft-run --mixture=image-only-v9

Single-image-only-v9 (multi-image sources removed; mm_olmo-like pack settings):
› torchrun --nproc-per-node=8 {sys.argv[0]} train my-sft-run --mixture=single-image-only-v9

Full image-only-v10 mixture (richer v9 + hub FineVision + DynaMath):
› torchrun --nproc-per-node=8 {sys.argv[0]} train my-sft-run --mixture=image-only-v10

Single-image-only-v10 (v10 without multi-image v9 sources):
› torchrun --nproc-per-node=8 {sys.argv[0]} train my-sft-run --mixture=single-image-only-v10

Full image-only-v11 mixture (v10 + ChartVerse + figure captions + web reasoning):
› torchrun --nproc-per-node=8 {sys.argv[0]} train my-sft-run --mixture=image-only-v11

Single-image-only-v11 (v11 without multi-image v9 sources; the production tier):
› torchrun --nproc-per-node=8 {sys.argv[0]} train my-sft-run --mixture=single-image-only-v11

Just the v11-only sources (data-staging check):
› torchrun --nproc-per-node=1 {sys.argv[0]} train smoke --mixture=v11-new

Init from HF instead of stage-1 checkpoint:
› torchrun --nproc-per-node=1 {sys.argv[0]} train smoke --trainer.load_path=null

Launch on Beaker:
› python {sys.argv[0]} launch molmo2-stage2 --launch.num_nodes=1
    """.strip()

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    script, cmd, run_name, *overrides = sys.argv

    if cmd == "train":
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
