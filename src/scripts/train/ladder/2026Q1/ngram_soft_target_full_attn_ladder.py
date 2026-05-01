"""
Full-attention ladder with n-gram soft-target auxiliary loss.

Data pipeline mirrors ``baseline_full_attn_ladder.py`` (same six Dolma2 source
paths, same tokenizer, same ConcatAndChunk chunking). The only data change is
that we wrap the base instance source with NgramSoftTargetInstanceSource, which
adds per-position top-K soft-target distributions drawn from the KenLM-built
ngram tables.

The train module gets two extra knobs:
  * ``soft_ce_alpha_start`` — mixing weight for soft-CE at step 0 (soft dominant).
  * ``soft_ce_alpha_ramp_fraction`` — fraction of total steps over which
    alpha linearly decays to 0. After the ramp, training is pure hard CE.
Everything else (z_loss_multiplier=1e-5, fsdp, grad clip, compile, etc.)
matches the baseline exactly.
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

# Matches ``baseline_v02_full_attn_ladder.py`` exactly so the soft-target
# arm is an apples-to-apples comparison against the v02 baseline isoparam.
# Only the s2pdf subsource moves v0.1→v0.2 (redacted → document-denylisted);
# the other five stay at v0.1 because they're unchanged between versions.
DOLMA2_BASELINE_PATHS = [
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2-fixed/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/arxiv/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/finemath-3plus/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.2/allenai/dolma2-tokenizer/s2pdf/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/stack-edu/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/wikipedia/**/*.npy",
]

# Built from the pilot-v4 ARPA (1e-3 fraction, n=5, prune 0 1 1 1 1). Must
# contain ``forward_index_topk.bin`` — the precomputed top-K forward index
# (FXTK v=1) produced by ``data_gen/build_topk_forward_index.py``. Runtime
# is pure-Python over this single mmap'd file; no kenlm at training time.
# Override via ``--ngram-table-dir`` for smoke tests on a different pilot.
DEFAULT_NGRAM_TABLE_DIR = (
    "/weka/oe-training-default/ai2-llm/ngram-tables/pilots/"
    "pilot-2026-04-22-fraction1e-3-n5"
)

# Soft-target hyperparameters. K=16, N_max=5 must match the built index.
DEFAULT_SOFT_TARGET_K = 16
DEFAULT_SOFT_TARGET_N_MAX = 5

# Soft-CE schedule defaults: start soft-dominant (but always include some hard
# CE — never α=1.0) and linearly ramp down to 0 (pure hard CE) over the first
# half of training. The 0.9 starting value is deliberate: with α=1.0 the soft
# loss puts all probability mass on the K=16 ngram-top tokens and zero on
# everything else, including (often) the gold label. Forward-KL training then
# actively drives the model away from the gold during the soft-dominant phase,
# leaving it with very low logits at gold positions; the second half of
# training is then spent recovering from that bad initialization. Setting
# α_start < 1 ensures hard CE always contributes — gold logits are always
# being pushed up — even when the soft target disagrees with the gold.
DEFAULT_SOFT_CE_ALPHA_START = 0.9
DEFAULT_SOFT_CE_ALPHA_RAMP_FRACTION = 0.5
# How to extend the truncated top-K ngram into a target distribution:
#   "renormalize"      — top-K probs sum to 1, non-top-K target = 0 (default;
#                        forces model to put zero mass on out-of-top-K gold)
#   "uniform_residual" — top-K = raw KN probs, non-top-K = uniform residual,
#                        so the target is a proper full-vocab distribution
DEFAULT_SOFT_CE_TRUNCATION = "renormalize"


@dataclasses.dataclass(kw_only=True, eq=True)
class NgramSoftTargetConfigurator(Olmo3ModelConfigurator):
    """
    Olmo3 configurator that also plumbs the soft-CE auxiliary-loss fields into
    :class:`TransformerTrainModuleConfig`. Identical to the parent's
    ``build_train_module`` except for the two extra fields.
    """

    soft_ce_alpha_start: float = DEFAULT_SOFT_CE_ALPHA_START
    soft_ce_alpha_ramp_fraction: float = DEFAULT_SOFT_CE_ALPHA_RAMP_FRACTION
    soft_ce_truncation: str = DEFAULT_SOFT_CE_TRUNCATION

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
            soft_ce_alpha_start=self.soft_ce_alpha_start,
            soft_ce_alpha_ramp_fraction=self.soft_ce_alpha_ramp_fraction,
            soft_ce_truncation=self.soft_ce_truncation,
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
    soft_ce_truncation = getattr(
        args, "soft_ce_truncation", DEFAULT_SOFT_CE_TRUNCATION
    )
    # In "uniform_residual" mode the loss function needs the raw KN log-probs
    # so it can compute the per-position residual at training time.
    output_log_probs = soft_ce_truncation == "uniform_residual"
    wrapped_source = NgramSoftTargetInstanceSourceConfig(  # noqa: F405
        source=base_source,
        table_dir=getattr(args, "ngram_table_dir", DEFAULT_NGRAM_TABLE_DIR),
        K=getattr(args, "soft_target_k", DEFAULT_SOFT_TARGET_K),
        N_max=getattr(args, "soft_target_n_max", DEFAULT_SOFT_TARGET_N_MAX),
        output_log_probs=output_log_probs,
    )

    instance_sources: list[InstanceSourceConfig] = [wrapped_source]  # noqa: F405

    ladder = ModelLadder(
        name=args.name,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=[s for s in TransformerSize if s.approx_num_params <= 1e9],
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=NgramSoftTargetConfigurator(
            model_construction_kwargs={"sliding_window": None},
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
            soft_ce_alpha_start=getattr(
                args, "soft_ce_alpha_start", DEFAULT_SOFT_CE_ALPHA_START
            ),
            soft_ce_alpha_ramp_fraction=getattr(
                args, "soft_ce_alpha_ramp_fraction", DEFAULT_SOFT_CE_ALPHA_RAMP_FRACTION
            ),
            soft_ce_truncation=soft_ce_truncation,
        ),
        run_configurator=WSDSChinchillaRunConfigurator(
            chinchilla_multiple=args.chinchilla_multiple
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(  # noqa: F405
            num_workers=16, instance_filter_config=InstanceFilterConfig()  # noqa: F405
        ),
    )
    return ladder


def add_additional_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    """Hook called by ``internal.ladder.main`` to register custom CLI args.

    Lets us point the soft-target ladder at a different table directory
    (e.g. for smoke testing against synthetic tables) without editing the
    default constant.
    """
    parser.add_argument(
        "--ngram-table-dir",
        type=str,
        default=DEFAULT_NGRAM_TABLE_DIR,
        help=(
            "Directory containing forward_index_topk.bin (FXTK v=1). Defaults "
            "to the pilot 1e-3 n=5 tables on Weka."
        ),
    )
    parser.add_argument(
        "--soft-ce-alpha-start",
        type=float,
        default=DEFAULT_SOFT_CE_ALPHA_START,
        help="Soft-CE mixing weight at step 0 (ramps linearly to 0).",
    )
    parser.add_argument(
        "--soft-ce-alpha-ramp-fraction",
        type=float,
        default=DEFAULT_SOFT_CE_ALPHA_RAMP_FRACTION,
        help="Fraction of max_steps over which alpha ramps from start to 0.",
    )
    parser.add_argument(
        "--soft-target-k",
        type=int,
        default=DEFAULT_SOFT_TARGET_K,
        help="Top-K size for the soft-target distributions.",
    )
    parser.add_argument(
        "--soft-ce-truncation",
        type=str,
        choices=["renormalize", "uniform_residual"],
        default=DEFAULT_SOFT_CE_TRUNCATION,
        help=(
            "How to extend the truncated top-K ngram into a target distribution. "
            "'renormalize' (default): top-K probs sum to 1, non-top-K target = 0. "
            "'uniform_residual': top-K = raw KN probs, non-top-K = uniform "
            "spread of (1 − Σ topK p_ngram), so the target is a proper full-vocab "
            "distribution and the model isn't penalized for keeping mass on "
            "out-of-top-K gold tokens. The 'uniform_residual' choice also "
            "switches the wrapper to output_log_probs=True so the residual "
            "can be computed at training time."
        ),
    )


if __name__ == "__main__":
    main(configure_ladder, add_additional_args=add_additional_args)
