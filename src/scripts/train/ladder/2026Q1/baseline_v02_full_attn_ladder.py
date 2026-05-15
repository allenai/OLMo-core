"""Baseline full-attention ladder on the v02 baseline mix.

Same as `baseline_full_attn_ladder.py` except the s2pdf subsource is swapped
from v0.1/`s2pdf_redacted` (pad-masked) to v0.2/`s2pdf` (document-level
denylisted, no `<|pad|>` runs). The other 5 subsources stay at v0.1 because
they are unchanged between v01 and v02.

Motivation: isolate the method effect of ICL overlap (and other SA methods)
from the differential-cleanup confound identified in
`tex/instance_filter_analysis/`. v02 baseline is the clean control that the
v02 SA methods should be compared against.
"""
import argparse
import dataclasses
import logging

import olmo_core.io as io
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import *
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import (
    DeviceMeshSpec,
    ModelLadder,
    Olmo3ModelConfigurator,
    TransformerSize,
    WSDSChinchillaRunConfigurator,
)

log = logging.getLogger(__name__)

@dataclasses.dataclass(kw_only=True)
class _WSDSChinchillaSmoke(WSDSChinchillaRunConfigurator):
    """Smoke variant of WSDS Chinchilla. Mirrors the version in the
    ngram-soft-target ladder. Three changes vs production: relaxes the
    chinchilla_multiple ≥ 0.5 floor and the power-of-2 check; uses a
    single anneal at the requested multiple; tiny fixed batch + minimal
    warmup so a 1-GPU smoke runs in minutes."""

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
class _BaselineSmokeConfigurator(Olmo3ModelConfigurator):
    """Olmo3 configurator subclass that overrides the 8-GPU minimum mesh
    spec when ``smoke_1gpu`` is true."""

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


DOLMA2_BASELINE_PATHS = [
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2-fixed/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/arxiv/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/finemath-3plus/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.2/allenai/dolma2-tokenizer/s2pdf/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/stack-edu/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/wikipedia/**/*.npy",
]


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceConfig(
                    source_paths=DOLMA2_BASELINE_PATHS,
                    tokenizer=tokenizer,
                ),
            ],
            sequence_length=args.sequence_length,
        ),
    ]

    smoke_1gpu = getattr(args, "smoke_1gpu", False)
    max_devices = 1 if smoke_1gpu else args.max_gpus
    ladder_kwargs = {}
    if getattr(args, "seed", None) is not None:
        ladder_kwargs["seed"] = args.seed
    ladder = ModelLadder(
        name=args.name,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=[s for s in TransformerSize if s.approx_num_params <= 1e9],
        max_devices=max_devices,
        **ladder_kwargs,
        device_type=get_gpu_type(args.cluster),
        model_configurator=_BaselineSmokeConfigurator(
            model_construction_kwargs={"sliding_window": None},
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
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
        data_loader=ComposableDataLoaderConfig(
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
    )
    return ladder


def add_additional_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    """Hook called by ``internal.ladder.main`` to register custom CLI args."""
    parser.add_argument(
        "--smoke-1gpu",
        action="store_true",
        help=(
            "Run on a single GPU for fast end-to-end smoke testing. "
            "Same semantics as the ngram ladders' flag — overrides the "
            "configurator's 8-GPU minimum, forces max_devices=1, and swaps "
            "in a smoke run-configurator that allows tiny chinchilla_multiple "
            "values with minimal warmup. Pair with --chinchilla-multiple "
            "~0.001 for a ~5 minute reference run."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override the ModelLadder.seed (default 42). Used to spread "
            "a multi-seed baseline cohort (baseline-v02-seed1/2/3 "
            "convention). Pass a different int per launch."
        ),
    )


if __name__ == "__main__":
    main(configure_ladder, add_additional_args=add_additional_args)
