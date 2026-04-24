"""
Baseline ladder using the old NumpyFSL data pipeline, on v02 data.

Variant of ``old_pipeline_baseline_ladder.py`` that points at the v02 Dolma
baseline paths (``v0.2/s2pdf`` in place of ``v0.1/s2pdf_redacted``), matching
what ``baseline_v02_full_attn_ladder.py`` trains on. Used to isolate the
effect of the NumpyFSL code path (vs the new Composable data API) on
identical data, on the current olmo-core commit.

Paired with ``old_pipeline_icl_overlap_on_v02_ladder.py`` for the ICL
side of the comparison.

See ``tex/instance_filter_analysis/`` for the mechanism-hunting context
(old-vs-new pipeline sign flip on C4 CE; ~0.02 nats TE delta we have
not yet isolated the cause of).
"""

import argparse
import logging

import olmo_core.io as io
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import (
    ModelLadder,
    Olmo3ModelConfigurator,
    TransformerSize,
    WSDSChinchillaRunConfigurator,
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


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    work_dir = str(
        io.join_path(get_root_dir(args.cluster), "model-ladders", args.name, "cache")
    )

    dataset = NumpyFSLDatasetConfig.glob(
        *DOLMA2_BASELINE_PATHS,
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        work_dir=work_dir,
    )

    ladder = ModelLadder(
        name=args.name,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=[s for s in TransformerSize if s.approx_num_params <= 370e6],
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=Olmo3ModelConfigurator(
            model_construction_kwargs={"sliding_window": None},
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
        ),
        run_configurator=WSDSChinchillaRunConfigurator(
            chinchilla_multiple=args.chinchilla_multiple
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        numpy_dataset=dataset,
        numpy_data_loader=NumpyDataLoaderConfig(
            global_batch_size=0,  # set by ModelLadder at runtime
            seed=42,
            num_workers=4,
            work_dir=work_dir,
        ),
    )
    return ladder


if __name__ == "__main__":
    main(configure_ladder)
