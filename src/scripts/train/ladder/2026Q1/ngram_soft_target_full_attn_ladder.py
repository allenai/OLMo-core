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

DOLMA2_BASELINE_PATHS = [
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2-fixed/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/arxiv/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/finemath-3plus/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/s2pdf_redacted/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/stack-edu/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/wikipedia/**/*.npy",
]

# Built by data_gen/arpa_to_table.py from the pilot-v4 ARPA (1e-3 fraction,
# n=5, prune 0 1 1 1 1). Must live on a filesystem reachable from the training
# cluster — Weka for AI2 clusters. Override via CLI if needed.
DEFAULT_NGRAM_TABLE_DIR = (
    "/weka/oe-training-default/ai2-llm/ngram-tables/pilots/"
    "pilot-2026-04-22-fraction1e-3-n5/tables"
)

# Soft-target hyperparameters. K=16 and unigram_shortlist=100 align with our
# pilot-phase sizing decisions; N_max=5 matches the built tables.
DEFAULT_SOFT_TARGET_K = 16
DEFAULT_SOFT_TARGET_N_MAX = 5
DEFAULT_SOFT_TARGET_UNIGRAM_SHORTLIST = 100

# Soft-CE schedule defaults: start fully soft and linearly ramp down to 0
# (pure hard CE) over the first half of training.
DEFAULT_SOFT_CE_ALPHA_START = 1.0
DEFAULT_SOFT_CE_ALPHA_RAMP_FRACTION = 0.5


@dataclasses.dataclass(kw_only=True, eq=True)
class NgramSoftTargetConfigurator(Olmo3ModelConfigurator):
    """
    Olmo3 configurator that also plumbs the soft-CE auxiliary-loss fields into
    :class:`TransformerTrainModuleConfig`. Identical to the parent's
    ``build_train_module`` except for the two extra fields.
    """

    soft_ce_alpha_start: float = DEFAULT_SOFT_CE_ALPHA_START
    soft_ce_alpha_ramp_fraction: float = DEFAULT_SOFT_CE_ALPHA_RAMP_FRACTION

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
    wrapped_source = NgramSoftTargetInstanceSourceConfig(  # noqa: F405
        source=base_source,
        table_dir=getattr(args, "ngram_table_dir", DEFAULT_NGRAM_TABLE_DIR),
        K=getattr(args, "soft_target_k", DEFAULT_SOFT_TARGET_K),
        N_max=getattr(args, "soft_target_n_max", DEFAULT_SOFT_TARGET_N_MAX),
        unigram_shortlist=getattr(
            args, "soft_target_unigram_shortlist", DEFAULT_SOFT_TARGET_UNIGRAM_SHORTLIST
        ),
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
        ),
        run_configurator=WSDSChinchillaRunConfigurator(
            chinchilla_multiple=args.chinchilla_multiple
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(  # noqa: F405
            num_workers=8, instance_filter_config=InstanceFilterConfig()  # noqa: F405
        ),
    )
    return ladder


if __name__ == "__main__":
    main(configure_ladder)
