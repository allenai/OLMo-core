import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional

from olmo_core.distributed.utils import get_rank
from olmo_core.exceptions import OLMoEnvironmentError

from ..common import TrainingProgress
from .callback import Callback
from .comet import CometCallback
from .wandb import WandBCallback

if TYPE_CHECKING:
    from beaker import Beaker

log = logging.getLogger(__name__)


BEAKER_EXPERIMENT_ID_ENV_VAR = "BEAKER_EXPERIMENT_ID"
BEAKER_RESULT_DIR = "/results"


@dataclass
class BeakerCallback(Callback):
    """
    Adds metadata to the Beaker experiment description when running as a Beaker batch job.
    """

    priority: ClassVar[int] = min(CometCallback.priority - 1, WandBCallback.priority - 1)
    experiment_id: Optional[str] = None
    update_interval: Optional[int] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None
    """
    A JSON-serializable config to save to the results dataset as ``config.json``.
    """
    result_dir: str = BEAKER_RESULT_DIR
    """
    The directory of the Beaker results dataset where the config and other data will be saved.
    """

    _client = None
    _url = None
    _last_update: Optional[float] = None

    @property
    def client(self) -> "Beaker":
        return self._client  # type: ignore

    @client.setter
    def client(self, client: "Beaker"):
        self._client = client

    def post_attach(self):
        if self.enabled is None and BEAKER_EXPERIMENT_ID_ENV_VAR in os.environ:
            self.enabled = True

    def pre_train(self):
        if self.enabled and get_rank() == 0:
            if self.experiment_id is None:
                if BEAKER_EXPERIMENT_ID_ENV_VAR not in os.environ:
                    raise OLMoEnvironmentError(f"missing env var '{BEAKER_EXPERIMENT_ID_ENV_VAR}'")
                else:
                    self.experiment_id = os.environ[BEAKER_EXPERIMENT_ID_ENV_VAR]

            from beaker import Beaker

            self.client = Beaker.from_env()
            log.info(
                f"Running in Beaker experiment {self.client.experiment.url(self.experiment_id)}"
            )

            # Ensure result dataset directory exists.
            result_dir = Path(self.result_dir) / "olmo-core"
            result_dir.mkdir(parents=True, exist_ok=True)

            # Save config to result dir.
            if self.config is not None:
                config_path = result_dir / "config.json"
                with config_path.open("w") as config_file:
                    log.info(f"Saving config to '{config_path}'")
                    json.dump(self.config, config_file)

            # Try saving Python requirements.
            requirements_path = result_dir / "requirements.txt"
            try:
                with requirements_path.open("w") as requirements_file:
                    requirements_file.write(f"# python={platform.python_version()}\n")
                with requirements_path.open("a") as requirements_file:
                    subprocess.call(
                        ["pip", "freeze"],
                        stdout=requirements_file,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                    )
            except Exception as e:
                log.exception(f"Error saving Python packages: {e}")

            # Try to get W&B/Comet URL of experiment.
            for callback in self.trainer.callbacks.values():
                if isinstance(callback, WandBCallback) and callback.enabled:
                    if (url := callback.run.get_url()) is not None:
                        self._url = url
                    break
                elif isinstance(callback, CometCallback) and callback.enabled:
                    if (url := callback.exp.url) is not None:
                        self._url = url
                    break

            self._update()

    def post_step(self):
        update_interval = self.update_interval or self.trainer.metrics_collect_interval
        if self.enabled and get_rank() == 0 and self.step % update_interval == 0:
            # Make sure we don't update too frequently.
            if self._last_update is None or (time.monotonic() - self._last_update) > 10:
                self._update()

    def post_train(self):
        if self.enabled and get_rank() == 0:
            self._update()

    def _update(self):
        self.trainer.run_bookkeeping_op(
            self._set_description,
            self.trainer.training_progress,
            op_name="beaker_set_description",
            allow_multiple=False,
            distributed=False,
        )
        self._last_update = time.monotonic()

    def _set_description(self, progress: TrainingProgress):
        from beaker import BeakerError, HTTPError
        from requests.exceptions import RequestException

        assert self.experiment_id is not None

        description = f"[{progress}] "

        if self.description is not None:
            description = f"{description}{self.description}\n"

        if self._url is not None:
            description = f"{description}{self._url} "

        try:
            self.client.experiment.set_description(self.experiment_id, description.strip())
        except (RequestException, BeakerError, HTTPError) as e:
            log.warning(f"Failed to update Beaker experiment description: {e}")
