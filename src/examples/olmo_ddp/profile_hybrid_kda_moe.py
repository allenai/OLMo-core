"""Run a short, random-init Nsight profile of an Olmo hybrid KDA/MoE model."""

import argparse
from pathlib import Path

from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig
from olmo_core.internal.experiment import ExperimentConfig, train
from olmo_core.train import (
    Duration,
    LoadStrategy,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import NvidiaProfilerCallback
from olmo_core.train.train_module.transformer.config import OLMoDDPTrainModuleConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Native OLMo-core experiment config."
    )
    parser.add_argument("--data", type=Path, required=True, help="Local uint32 NumPy token array.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--profile-step-start", type=int, default=10)
    parser.add_argument("--profile-step-end", type=int, default=13)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--micro-batch-size", type=int, default=3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    return parser


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.from_json(args.config)
    config.run_name = "profile-hybrid-kda-moe-random-init"
    config.launch = None

    if not isinstance(config.dataset, NumpyFSLDatasetConfig):
        raise TypeError(f"Expected NumpyFSLDatasetConfig, got {type(config.dataset).__name__}")
    config.dataset.paths = [str(args.data)]
    config.dataset.mix = None
    config.dataset.mix_base_dir = None
    config.dataset.source_mixture_config = None
    config.dataset.label_mask_paths = None
    config.dataset.metadata = None
    config.dataset.expand_glob = False
    config.dataset.include_instance_metadata = False
    config.dataset.instance_filter_config = None
    config.dataset.sequence_length = args.sequence_length
    config.dataset.max_target_sequence_length = args.sequence_length
    config.dataset.generate_doc_lengths = False
    config.dataset.work_dir = str(args.output_dir / "dataset-cache")
    config.dataset.ignore_fingerprint_mismatch = True

    if not isinstance(config.data_loader, NumpyDataLoaderConfig):
        raise TypeError(f"Expected NumpyDataLoaderConfig, got {type(config.data_loader).__name__}")
    rank_microbatch_size = args.micro_batch_size * args.sequence_length
    config.data_loader.global_batch_size = rank_microbatch_size * args.gradient_accumulation_steps
    config.data_loader.num_workers = 2
    config.data_loader.prefetch_factor = 2
    config.data_loader.ignore_fingerprint_mismatch = True

    if not isinstance(config.train_module, OLMoDDPTrainModuleConfig):
        raise TypeError(
            f"Expected OLMoDDPTrainModuleConfig, got {type(config.train_module).__name__}"
        )
    config.train_module.rank_microbatch_size = rank_microbatch_size
    config.train_module.max_sequence_length = args.sequence_length
    config.train_module.pp_config = None
    config.train_module.tp_config = None
    config.train_module.cp_config = None
    config.train_module.ac_config = None
    if config.train_module.ep_config is None or config.train_module.ep_config.degree != 8:
        raise ValueError("This profile contract requires expert parallel degree 8")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.trainer.save_folder = str(args.output_dir)
    config.trainer.work_dir = None
    config.trainer.load_path = None
    config.trainer.load_strategy = LoadStrategy.never
    config.trainer.save_overwrite = True
    config.trainer.max_duration = Duration.steps(args.steps)
    config.trainer.no_checkpoints = True
    config.trainer.no_evals = True
    config.trainer.metrics_collect_interval = 1
    config.trainer.callbacks = {
        name: callback
        for name, callback in config.trainer.callbacks.items()
        if name in {"config_saver", "gpu_monitor", "speed_monitor"}
    }
    config.trainer.callbacks["profiler"] = NvidiaProfilerCallback(
        start=args.profile_step_start,
        end=args.profile_step_end,
        profile_ranks=[0],
    )
    return config


def main() -> None:
    args = _build_parser().parse_args()
    config = _build_config(args)
    prepare_training_environment(backend=config.backend)
    try:
        train(config)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
