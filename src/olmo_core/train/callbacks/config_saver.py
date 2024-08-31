import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from olmo_core.aliases import PathOrStr
from olmo_core.distributed.utils import get_rank

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class ConfigSaverCallback(Callback):
    """
    A callback that writes an arbitrary JSON-serializable config dictionary to every checkpoint
    directory written during training.
    """

    config: Optional[Dict[str, Any]] = None
    fname: str = "config.json"

    def post_checkpoint_saved(self, path: PathOrStr):
        if get_rank() != 0:
            return

        if self.config is None:
            log.warning(f"Config not set on {self.__class__.__name__}, doing nothing")
            return

        self.trainer.write_file(self.fname, json.dumps(self.config), dir=path)
