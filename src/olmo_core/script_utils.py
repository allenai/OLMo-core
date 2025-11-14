import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import rich

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config
from olmo_core.data import NumpyDataLoaderConfig, NumpyDatasetConfig
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_state_dict
from olmo_core.io import is_url, join_path, normalize_path
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import ConfigSaverCallback
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.utils import prepare_cli_environment, seed_all

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536
    load_path: Optional[str] = None


def get_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} [OPTIONS...] [CONFIG_OVERRIDES...]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name",
        type=str,
        help="""A name to assign the run for logging.""",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="""The sequence length to train and eval on. Different scripts have different default
        sequence-length values. If a value is not specified here, the default value is used.""",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="http://olmo-data.org",
        help="""The root directory/URL of the data source files.
        The default 'http://olmo-data.org' is public, but potentially very slow.
        Ai2 employees should prefer '/weka/oe-training-default/ai2-llm' when using a cluster with weka access,
        otherwise 'gs://ai2-llm' or 's3://ai2-llm'.""",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        required=True,
        help="""A local or remote directory to save checkpoints to.
        All ranks should have access to this directory, so when training in a multi-node setup
        this could either be a path to a folder on a shared filesystem (such as NFS) or a URL
        to cloud storage, like 's3://...' or 'gs://...'.""",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        help="""A local directory to use as a working directory for dataset preprocessing.
        If not set this will be inferred from the save folder.""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="""Print the config and exit.""",
    )
    return parser


def _parse_args(
    parser: Optional[argparse.ArgumentParser] = None,
) -> Tuple[argparse.Namespace, List[str]]:
    parser = parser if parser is not None else get_cli_parser()
    opts, overrides = parser.parse_known_args()
    if opts.work_dir is None:
        if is_url(opts.save_folder):
            opts.work_dir = "/tmp/olmo-core/dataset-cache"
        else:
            opts.work_dir = opts.save_folder
    return opts, overrides


def main(
    config_builder: Callable[[argparse.Namespace, List[str]], ExperimentConfig],
    parser: Optional[argparse.ArgumentParser] = None,
) -> None:
    opts, overrides = _parse_args(parser)
    if opts.dry_run:
        prepare_cli_environment()

    config = config_builder(opts, overrides)

    if opts.dry_run:
        rich.print(config)
        return

    prepare_training_environment(shared_filesystem=not is_url(opts.save_folder))

    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    # Save config to W&B and each checkpoint dir.
    for callback in trainer.callbacks.values():
        if isinstance(callback, ConfigSaverCallback):
            callback.config = config.as_config_dict()
            break

    # If we have a load path set and there is no checkpoint in the save folder, load the
    # checkpoint from the load path.
    if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and config.load_path:
        log.info(
            f"Loading checkpoint from {config.load_path} since no checkpoints were found in the save folder..."
        )
        trainer.load_checkpoint(config.load_path, load_trainer_state=False)

    # Train.
    trainer.fit()

    # Tear-down distributed backend.
    teardown_training_environment()


def get_lr_from_checkpoint(
    path: PathOrStr, param: Optional[str] = None, param_group: Optional[int] = None
) -> float:
    path = normalize_path(path)
    if not path.endswith("/model_and_optim"):
        path = join_path(path, "model_and_optim")

    metadata = get_checkpoint_metadata(path)
    if "optim.param_groups.0.params" in metadata.state_dict_metadata:
        # unflattened optimizer state
        if param is not None:
            log.warning(
                "'param' will be ignored since the optimizer state in the checkpoint to load is in unflattened format"
            )
        if param_group is None:
            param_group = 0
        key = f"optim.param_groups.{param_group}.lr"
    else:
        if param_group is not None:
            raise RuntimeError(
                "'param_group' is required since the optimizer state in the checkpoint to load is in flattened format"
            )
        if param is None:
            param = "embeddings.weight"
        key = f"optim.param_groups.{param}.lr"

    state_dict = {key: None}
    load_state_dict(path, state_dict)
    assert state_dict[key] is not None
    return float(state_dict[key])  # type: ignore
