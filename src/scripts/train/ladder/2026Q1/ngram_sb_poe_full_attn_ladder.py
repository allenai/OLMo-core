"""Full-attention ladder with **stupid-backoff** n-gram product-of-experts
(PoE) logit bias.

Parallel of :mod:`ngram_poe_full_attn_ladder` (which uses the KN-smoothed
top-K forward index). The differences:

- Data wrapper is :class:`NgramStupidBackoffInstanceSource` instead of
  :class:`NgramSoftTargetInstanceSource`. The instance source emits
  per-position ragged ``sb_override_*`` arrays — variable-K overrides
  per position, not the fixed-K dense slots the KN-smoothed path uses.
- ``TransformerTrainModuleConfig`` is configured with ``poe_sb_table_dir``
  / ``poe_sb_alpha`` / ``poe_sb_N_max`` (mutually exclusive with the
  KN-smoothed ``poe_ngram_table_dir``). λ is shared via ``poe_lambda``.

Train-time loss::

    log p_final(w | h) = log p_lm(w | h) + λ · sb_score(w | h_t) − log Z(h)
    L = − log p_final(label | h)

where ``sb_score(w | h_t)`` is the highest-order stupid-backoff score for
``w`` given history ``h_t`` (with universal length-V unigram floor as the
fallback when ``w`` isn't observed at any higher order). The training
step adds the bias on top of the LM's logits via
:func:`olmo_core.data.sb_bias.apply_sb_bias_inplace`, then standard
cross-entropy at the gold label.

Eval-time consistency: in-loop evaluators don't go through the
InstanceSource wrapper, so :meth:`TransformerTrainModule._apply_poe_eval_bias_sb`
synchronously recomputes the per-instance overrides from the same SB
reader and applies the matching bias before computing CE — the inference
distribution and the training-time joint are therefore equivalent.

Everything else (z_loss_multiplier=1e-5, fsdp, grad clip, compile, etc.)
matches the baseline / KN-smoothed PoE ladders exactly.
"""

import argparse
import dataclasses
import logging

import olmo_core.io as io
from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import *  # noqa: F401,F403
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import (
    DeviceMeshSpec,
    ModelLadder,
    Olmo3ModelConfigurator,
    TransformerSize,
    WSDSChinchillaRunConfigurator,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)

DOLMA2_BASELINE_PATHS = [
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2-fixed/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/arxiv/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/finemath-3plus/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.2/allenai/dolma2-tokenizer/s2pdf/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/stack-edu/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/wikipedia/**/*.npy",
]

# Default SB counts_index pilot directory on Weka. Must contain
# ``counts_index/`` built at format_version=2 by
# ``data_gen/build_counts_index.py``.
DEFAULT_SB_TABLE_DIR = (
    "/weka/oe-training-default/ai2-llm/ngram-tables/pilots/"
    "pilot-2026-04-22-fraction1e-3-n5-counts"
)

DEFAULT_SB_N_MAX = 5
DEFAULT_SB_ALPHA = 0.4         # Brants et al 2007.
DEFAULT_POE_LAMBDA = 0.1       # Same default as the KN-smoothed PoE 190M sweep winner.
DEFAULT_DOLMA2_VOCAB_SIZE = 100278
DEFAULT_SB_MAX_ORDER2_CONTINUATIONS = None


@dataclasses.dataclass(kw_only=True)
class _WSDSChinchillaSmoke(WSDSChinchillaRunConfigurator):
    """Smoke variant of WSDS Chinchilla, copied from the KN-smoothed
    ladder so a 1-GPU smoke completes in minutes rather than hours."""

    def __post_init__(self):
        from olmo_core.exceptions import OLMoConfigurationError

        if self.chinchilla_multiple <= 0:
            raise OLMoConfigurationError("'chinchilla_multiple' must be positive")
        if not (0 < self.decay_fraction < 0.5):
            raise OLMoConfigurationError(
                "'decay_fraction' must be greater than 0.0 and less than 0.5"
            )

    def configure_chinchilla_periods(self, num_params: int) -> tuple[int, list[float]]:
        return 16384, [self.chinchilla_multiple]

    def configure_target_batch_size(self, num_params: int) -> int:
        return 16384


@dataclasses.dataclass(kw_only=True, eq=True)
class NgramSBPoEConfigurator(Olmo3ModelConfigurator):
    """Olmo3 configurator that wires SB-PoE knobs into the
    :class:`TransformerTrainModuleConfig`."""

    poe_lambda: float = DEFAULT_POE_LAMBDA
    sb_table_dir: str = DEFAULT_SB_TABLE_DIR
    sb_alpha: float = DEFAULT_SB_ALPHA
    sb_n_max: int = DEFAULT_SB_N_MAX
    sb_max_order2_continuations: int | None = DEFAULT_SB_MAX_ORDER2_CONTINUATIONS
    dolma2_vocab_size: int = DEFAULT_DOLMA2_VOCAB_SIZE
    smoke_1gpu: bool = False

    def configure_minimal_device_mesh_spec(
        self,
        *,
        size_spec,
        sequence_length,
        device_type,
    ) -> DeviceMeshSpec:
        if self.smoke_1gpu:
            return DeviceMeshSpec(world_size=1, dp_world_size=1)
        return super().configure_minimal_device_mesh_spec(
            size_spec=size_spec,
            sequence_length=sequence_length,
            device_type=device_type,
        )

    def build_train_module(
        self,
        *,
        size_spec,
        sequence_length,
        rank_microbatch_size,
        model_config,
        optim_config,
        scheduler,
        device_type,
    ) -> TransformerTrainModule:
        device_type = device_type.lower()
        assert "h100" in device_type or "b200" in device_type or "a100" in device_type
        assert sequence_length in {2048, 4096, 8192}
        size_spec = TransformerSize(size_spec)

        dp_config = TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        )

        train_module_config = TransformerTrainModuleConfig(
            rank_microbatch_size=rank_microbatch_size,
            max_sequence_length=sequence_length,
            optim=optim_config,
            compile_model=True,
            dp_config=dp_config,
            z_loss_multiplier=1e-5,
            poe_lambda=self.poe_lambda,
            poe_sb_table_dir=self.sb_table_dir,
            poe_sb_alpha=self.sb_alpha,
            poe_sb_N_max=self.sb_n_max,
            poe_sb_dolma2_vocab_size=self.dolma2_vocab_size,
            poe_sb_max_order2_continuations=self.sb_max_order2_continuations,
            max_grad_norm=1.0,
            scheduler=scheduler,
        )

        model = model_config.build(init_device="meta")
        train_module = train_module_config.build(model)
        assert isinstance(train_module, TransformerTrainModule)
        return train_module


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()

    base_source = ConcatAndChunkInstanceSourceConfig(  # noqa: F405
        sources=[
            NumpyDocumentSourceConfig(  # noqa: F405
                source_paths=DOLMA2_BASELINE_PATHS,
                tokenizer=tokenizer,
            ),
        ],
        sequence_length=args.sequence_length,
    )
    wrapped_source = NgramStupidBackoffInstanceSourceConfig(  # noqa: F405
        source=base_source,
        table_dir=getattr(args, "sb_table_dir", DEFAULT_SB_TABLE_DIR),
        dolma2_vocab_size=int(tokenizer.vocab_size),
        N_max=getattr(args, "sb_n_max", DEFAULT_SB_N_MAX),
        alpha=getattr(args, "sb_alpha", DEFAULT_SB_ALPHA),
        max_order2_continuations=getattr(
            args, "sb_max_order2_continuations", DEFAULT_SB_MAX_ORDER2_CONTINUATIONS
        ),
    )

    instance_sources: list[InstanceSourceConfig] = [wrapped_source]  # noqa: F405

    smoke_1gpu = getattr(args, "smoke_1gpu", False)
    max_devices = 1 if smoke_1gpu else args.max_gpus
    ladder = ModelLadder(
        name=args.name,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=[s for s in TransformerSize if s.approx_num_params <= 1e9],
        max_devices=max_devices,
        device_type=get_gpu_type(args.cluster),
        model_configurator=NgramSBPoEConfigurator(
            model_construction_kwargs={"sliding_window": None},
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
            poe_lambda=getattr(args, "poe_lambda", DEFAULT_POE_LAMBDA),
            sb_table_dir=getattr(args, "sb_table_dir", DEFAULT_SB_TABLE_DIR),
            sb_alpha=getattr(args, "sb_alpha", DEFAULT_SB_ALPHA),
            sb_n_max=getattr(args, "sb_n_max", DEFAULT_SB_N_MAX),
            sb_max_order2_continuations=getattr(
                args,
                "sb_max_order2_continuations",
                DEFAULT_SB_MAX_ORDER2_CONTINUATIONS,
            ),
            dolma2_vocab_size=int(tokenizer.vocab_size),
            smoke_1gpu=smoke_1gpu,
        ),
        run_configurator=(
            _WSDSChinchillaSmoke(chinchilla_multiple=args.chinchilla_multiple)
            if smoke_1gpu
            else WSDSChinchillaRunConfigurator(
                chinchilla_multiple=args.chinchilla_multiple
            )
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(  # noqa: F405
            # Fewer workers than the KN-smoothed PoE ladder (which uses 16):
            # SB index is 25 small files vs KN's one ~10 GB file, so each
            # worker holds 25× more mmap views into /dev/shm. With 8
            # ranks × 16 workers = 128 worker processes per node, the
            # 1e-4 SB-PoE v3 run OOM'd the host. Vectorized
            # compute_overrides_for_sequence is ~5-50ms per instance, so
            # 4 workers / rank (32 total per node) easily keeps the GPUs fed.
            num_workers=4, instance_filter_config=InstanceFilterConfig()  # noqa: F405
        ),
    )
    return ladder


def add_additional_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    """Hook called by ``internal.ladder.main`` to register custom CLI args."""
    parser.add_argument(
        "--sb-table-dir",
        type=str,
        default=DEFAULT_SB_TABLE_DIR,
        help=(
            "Pilot directory containing counts_index/ produced by "
            "data_gen/build_counts_index.py at format_version=2. "
            "Defaults to the 1e-3 pilot."
        ),
    )
    parser.add_argument(
        "--poe-lambda",
        type=float,
        default=DEFAULT_POE_LAMBDA,
        help="Constant ngram mixing weight in the PoE joint log-prob.",
    )
    parser.add_argument(
        "--sb-alpha",
        type=float,
        default=DEFAULT_SB_ALPHA,
        help="Stupid-backoff discount α per Brants et al 2007.",
    )
    parser.add_argument(
        "--sb-n-max",
        type=int,
        default=DEFAULT_SB_N_MAX,
        help="Highest ngram order to consult. Must equal the order the counts_index was built with.",
    )
    parser.add_argument(
        "--sb-max-order2-continuations",
        type=int,
        default=DEFAULT_SB_MAX_ORDER2_CONTINUATIONS,
        help=(
            "Optional hard cap on order-2 continuations per one-token history. "
            "Keeps the highest-count continuations and lets omitted tokens fall "
            "through to the unigram floor. Orders 3+ remain exact."
        ),
    )
    parser.add_argument(
        "--smoke-1gpu",
        action="store_true",
        help=(
            "Run on a single GPU for fast end-to-end smoke testing. "
            "Pair with --chinchilla-multiple ~0.001."
        ),
    )


if __name__ == "__main__":
    main(configure_ladder, add_additional_args=add_additional_args)
