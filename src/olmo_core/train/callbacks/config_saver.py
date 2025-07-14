import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from olmo_core.aliases import PathOrStr
from olmo_core.data import NumpyDataLoaderBase
from olmo_core.distributed.utils import get_rank

from .beaker import BeakerCallback
from .callback import Callback
from .comet import CometCallback
from .wandb import WandBCallback

log = logging.getLogger(__name__)

DEFAULT_DATA_PATHS_FNAME = "data_paths.txt"


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
