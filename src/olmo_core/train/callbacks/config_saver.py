import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from olmo_core.aliases import PathOrStr
from olmo_core.data import NumpyDataLoaderBase
from olmo_core.distributed.utils import broadcast_object, get_rank
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import resource_path

from .beaker import BeakerCallback
from .callback import Callback
from .comet import CometCallback
from .wandb import WandBCallback

log = logging.getLogger(__name__)

DEFAULT_DATA_PATHS_FNAME = "data_paths.txt"

_CHECKPOINT_COMPATIBILITY_KEYS = ("model", "train_module", "init_seed")
_RUNTIME_ONLY_TRAIN_MODULE_KEYS = {"rank_microbatch_size"}


def checkpoint_compatibility_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select every model- and optimizer-related setting that must agree across a checkpoint restore.

    The complete ``model`` and ``train_module`` trees are compared, including optimizer groups,
    scheduler, precision, clipping, parallelism, activation checkpointing, and state-dict options.
    Rank microbatch size only controls how the unchanged global batch is partitioned at runtime,
    so it may differ when a checkpoint moves to a new GPU topology and is excluded.

    :param config: A JSON-serializable experiment config dictionary.

    :returns: The JSON-canonicalized subset of ``config`` to compare.

    :raises OLMoConfigurationError: If a required field is missing or the selection is not
        JSON serializable.
    """
    selected: Dict[str, Any] = {}
    for key in _CHECKPOINT_COMPATIBILITY_KEYS:
        if key not in config:
            raise OLMoConfigurationError(
                f"Current or checkpoint config is missing required compatibility field '{key}'"
            )
        selected[key] = deepcopy(config[key])

    train_module = selected["train_module"]
    if not isinstance(train_module, dict):
        raise OLMoConfigurationError("Config field 'train_module' must be a dictionary")
    for key in _RUNTIME_ONLY_TRAIN_MODULE_KEYS:
        train_module.pop(key, None)

    # Checkpoint configs are loaded from JSON, while the current config may still contain
    # StrEnum instances. Canonicalize both through JSON so equal enum values compare equal
    # without weakening any value-level assertion.
    try:
        return json.loads(json.dumps(selected))
    except (TypeError, ValueError) as error:
        raise OLMoConfigurationError(
            f"Checkpoint compatibility config is not JSON serializable: {error}"
        ) from error


def config_differences(expected: Any, actual: Any, path: str = "") -> List[str]:
    """
    Return exact, recursively-addressed differences between two JSON configs.

    :param expected: The config built from the current command.
    :param actual: The config recorded in the checkpoint.
    :param path: The dotted path prefix used in the reported differences.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: List[str] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else key
            if key not in expected:
                differences.append(f"{child}: unexpected checkpoint value {actual[key]!r}")
            elif key not in actual:
                differences.append(f"{child}: missing from checkpoint (expected {expected[key]!r})")
            else:
                differences.extend(config_differences(expected[key], actual[key], child))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        differences = []
        if len(expected) != len(actual):
            differences.append(f"{path}: length {len(actual)} != expected {len(expected)}")
        for idx, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            differences.extend(config_differences(expected_item, actual_item, f"{path}[{idx}]"))
        return differences
    if type(expected) is not type(actual) or expected != actual:
        return [f"{path}: checkpoint {actual!r} != command {expected!r}"]
    return []


@dataclass
class ConfigSaverCallback(Callback):
    """
    A callback that writes an arbitrary JSON-serializable config dictionary (:data:`config`) to every checkpoint
    directory written during training. It will also set the config to save for other callbacks, including
    the :class:`WandBCallback`, :class:`CometCallback`, and others, if not already set.

    .. important:: The :data:`config` should be set *after* initializing the trainer and attaching all
        other callbacks.
    """

    fname: str = "config.json"
    save_data_paths: Optional[bool] = None
    data_paths_fname: Optional[str] = None
    validate_checkpoint_config: bool = False
    """
    If ``True``, abort a checkpoint load when the checkpoint's saved config disagrees with the
    current command's ``model``, ``train_module`` (minus runtime-only fields), or ``init_seed``.
    Useful when chaining runs that branch from each other's checkpoints.
    """

    _config: Optional[Dict[str, Any]] = None

    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """
        The JSON config dictionary to record.
        """
        return self._config

    @config.setter
    def config(self, config: Dict[str, Any]):
        self._config = config
        for callback_name, callback in self.trainer.callbacks.items():
            if (
                isinstance(callback, (WandBCallback, CometCallback, BeakerCallback))
                and callback.config is None
            ):
                log.info(
                    f"Setting config for '{callback_name}' callback of type '{callback.__class__.__name__}'"
                )
                callback.config = config

    def post_checkpoint_saved(self, path: PathOrStr):
        if get_rank() != 0:
            return

        if self.config is None:
            log.warning(f"Config not set on {self.__class__.__name__}, doing nothing")
        else:
            self.trainer.write_file(self.fname, json.dumps(self.config), dir=path)

        if self.save_data_paths is not False:
            if isinstance(self.trainer.data_loader, NumpyDataLoaderBase):
                ds = self.trainer.data_loader.dataset
                all_paths = "\n".join(str(p) for p in ds.paths)
                self.trainer.write_file(
                    self.data_paths_fname or DEFAULT_DATA_PATHS_FNAME, all_paths, dir=path
                )
            elif self.save_data_paths:
                log.warning(
                    f"Unable to save paths for data loader of type '{self.trainer.data_loader.__class__.__name__}' (not implemented)"
                )

    def pre_checkpoint_loaded(self, path: PathOrStr):
        if not self.validate_checkpoint_config:
            return

        error: Optional[str] = None
        if get_rank() == 0:
            try:
                if self.config is None:
                    raise OLMoConfigurationError(
                        f"Config not set on {self.__class__.__name__}; cannot validate checkpoint"
                    )
                with resource_path(path, self.fname).open() as f:
                    checkpoint_config = json.load(f)
                expected = checkpoint_compatibility_config(self.config)
                actual = checkpoint_compatibility_config(checkpoint_config)
                differences = config_differences(expected, actual)
                if differences:
                    detail = "\n - ".join(differences)
                    raise OLMoConfigurationError(
                        "Checkpoint model/train-module hyperparameters do not exactly match "
                        f"the command config:\n - {detail}"
                    )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        error = broadcast_object(error)
        if error is not None:
            raise OLMoConfigurationError(error)
        log.info("Checkpoint model/train-module hyperparameters exactly match command config")
