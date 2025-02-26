import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

from olmo_core.distributed.utils import get_rank
from olmo_core.exceptions import OLMoEnvironmentError

from .callback import Callback
from .comet import CometCallback
from .wandb import WandBCallback

if TYPE_CHECKING:
    from beaker import Beaker

log = logging.getLogger(__name__)


BEAKER_EXPERIMENT_ID_ENV_VAR = "BEAKER_EXPERIMENT_ID"


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

    _client = None
    _url = None

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

            for callback in self.trainer.callbacks.values():
                if isinstance(callback, WandBCallback) and callback.enabled:
                    if (url := callback.run.get_url()) is not None:
                        self._url = url
                    break
                elif isinstance(callback, CometCallback) and callback.enabled:
                    if (url := callback.exp.url) is not None:
                        self._url = url
                    break

    def post_step(self):
        update_interval = self.update_interval or self.trainer.metrics_collect_interval
        if self.enabled and get_rank() == 0 and self.step % update_interval == 0:
            self.trainer.thread_pool.submit(
                self._set_description,
                step=self.step,
                max_steps=self.trainer.max_steps,
                msg=self.description,
            )

    def post_train(self):
        if self.enabled and get_rank() == 0:
            self.trainer.thread_pool.submit(
                self._set_description,
                step=self.step,
                max_steps=self.trainer.max_steps,
                msg=self.description,
            )

    def _set_description(self, *, step: int, max_steps: Optional[int], msg: Optional[str]):
        from beaker import BeakerError, HTTPError
        from requests.exceptions import RequestException

        assert self.experiment_id is not None
        progress: str
        if max_steps is not None:
            perc = max(100, int(100 * step / max_steps))
            progress = f"{perc}%, {step}/{max_steps}"
        else:
            progress = f"{step}/??"

        description = f"[{progress}]"
        if msg is not None:
            description = f"{description} {msg}"
        if self._url is not None:
            description = f"{description} {self._url}"

        try:
            self.client.experiment.set_description(self.experiment_id, description)
        except (RequestException, BeakerError, HTTPError) as e:
            log.warning(f"Failed to update Beaker experiment description: {e}")
