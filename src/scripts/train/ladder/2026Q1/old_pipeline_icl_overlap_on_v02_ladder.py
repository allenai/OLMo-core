"""
ICL overlap ladder using the old NumpyFSL data pipeline, on v02 data.

Variant of ``old_pipeline_icl_overlap_ladder.py`` that uses v02 data paths
matching ``icl_overlap_v02_full_attn_ladder.py`` (v02 ICL overlap data +
v0.2/s2pdf-swapped Dolma baseline). Isolates the NumpyFSL +
``chunk_based_mixture=True`` code path as the variable, with data held
constant to the v02 new-pipeline runs, on the current olmo-core commit.

Paired with ``old_pipeline_baseline_on_v02_ladder.py``.

See ``tex/instance_filter_analysis/`` §sec:update-chunkbased for the
secondary first-K-per-file mechanism candidate this is intended to test.
"""

import argparse
import logging

import olmo_core.io as io
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
    SourceMixtureList,
)
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import (
    ModelLadder,
    Olmo3ModelConfigurator,
    TransformerSize,
    WSDSChinchillaRunConfigurator,
)

log = logging.getLogger(__name__)

ICL_OVERLAP_PATHS = [
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v02/icl-overlap-v02-8192-wo-replace/allenai/dolma2-tokenizer/*.npy",
]

DOLMA2_BASELINE_PATHS = [
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2-fixed/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/arxiv/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/finemath-3plus/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.2/allenai/dolma2-tokenizer/s2pdf/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/stack-edu/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/wikipedia/**/*.npy",
]

SEED = 42


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    work_dir = str(
        io.join_path(get_root_dir(args.cluster), "model-ladders", args.name, "cache")
    )

    src_mix = SourceMixtureDatasetConfig(
        source_list=SourceMixtureList(
            sources=[
                SourceMixtureConfig(
                    source_name="icl-overlap",
                    paths=ICL_OVERLAP_PATHS,
                    target_ratio=0.5,
                ),
                SourceMixtureConfig(
                    source_name="baseline",
                    paths=DOLMA2_BASELINE_PATHS,
                    target_ratio=0.5,
                ),
            ]
        ),
        requested_tokens=1,  # placeholder, set by ModelLadder per model size
        global_batch_size=1,  # placeholder, set by ModelLadder per model size
        seed=SEED,
    )
    dataset = NumpyFSLDatasetConfig.from_src_mix(
        src_mix,
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        work_dir=work_dir,
        chunk_based_mixture=True,
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
            seed=SEED,
            num_workers=4,
            work_dir=work_dir,
        ),
    )
    return ladder


if __name__ == "__main__":
    main(configure_ladder)
