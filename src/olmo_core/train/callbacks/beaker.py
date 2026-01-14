import dataclasses
import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional

from olmo_core.distributed.utils import get_rank

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

    _url: str | None = dataclasses.field(repr=False, default=None)
    _last_update: float | None = dataclasses.field(repr=False, default=None)
    _client = None

    @property
    def client(self) -> "Beaker":
        if self._client is None:
            from beaker import Beaker

            self._client = Beaker.from_env(check_for_upgrades=False)
        return self._client  # type: ignore

    def post_attach(self):
        if self.enabled is None:
            from olmo_core.launch.beaker import is_running_in_beaker_batch_job

            self.enabled = is_running_in_beaker_batch_job()

    def pre_train(self):
        if self.enabled and get_rank() == 0:
            if self.experiment_id is None:
                from olmo_core.launch.beaker import get_beaker_experiment_id

                self.experiment_id = get_beaker_experiment_id()

            assert self.experiment_id is not None
            workload = self.client.workload.get(self.experiment_id)
            beaker_url = self.client.workload.url(workload)
            log.info(f"Running in Beaker workload {beaker_url}")

            # Add Beaker URL to W&B/Comet config if available.
            for callback in self.trainer.callbacks.values():
                if isinstance(callback, WandBCallback):
                    log.info(f"Found WandBCallback: enabled={callback.enabled}, run={callback.run}")
                    if callback.enabled and callback.run is not None:
                        callback.run.config.update({"beaker_experiment_url": beaker_url})
                        log.info(f"Added beaker_experiment_url to W&B config: {beaker_url}")
                        break
                elif isinstance(callback, CometCallback):
                    log.info(f"Found CometCallback: enabled={callback.enabled}, exp={callback.exp}")
                    if callback.enabled and callback.exp is not None:
                        callback.exp.log_parameter("beaker_experiment_url", beaker_url)
                        log.info(f"Added beaker_experiment_url to Comet: {beaker_url}")
                        break

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
        from beaker.exceptions import BeakerError, HTTPError, RequestException, RpcError
        from gantry.api import update_workload_description

        description = f"[{progress}] "

        if self.description is not None:
            description = f"{description}{self.description}\n"

        if self._url is not None:
            description = f"{description}{self._url} "

        try:
            update_workload_description(description.strip(), client=self.client)
        except (RequestException, BeakerError, HTTPError, RpcError) as e:
            log.warning(f"Failed to update Beaker experiment description: {e}")

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def __del__(self):
        self.close()
