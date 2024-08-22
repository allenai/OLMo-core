import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from olmo_core.distributed.utils import get_rank
from olmo_core.exceptions import OLMoEnvironmentError

from .callback import Callback

log = logging.getLogger(__name__)


WANDB_API_KEY_ENV_VAR = "WANDB_API_KEY"


@dataclass
class WandBCallback(Callback):
    """
    Logs metrics to Weights & Biases.

    .. important::
        Requires the ``wandb`` package and the environment variable ``WANDB_API_KEY``.
    """

    name: Optional[str] = None
    """
    The name to give the W&B run.
    """

    project: Optional[str] = None
    """
    The W&B project to use.
    """

    entity: Optional[str] = None
    """
    The W&B entity to use.
    """

    group: Optional[str] = None
    """
    The W&B group to use.
    """

    tags: Optional[List[str]] = None
    """
    Tags to assign the run.
    """

    config: Optional[Dict[str, Any]] = None
    """
    The config to load to W&B.
    """

    cancel_tags: Optional[List[str]] = field(
        default_factory=lambda: ["cancel", "canceled", "cancelled"]
    )
    """
    If you add any of these tags to a run on W&B, the run will cancel itself.
    Defaults to ``["cancel", "canceled", "cancelled"]``.
    """

    _wandb = None
    _api = None

    def __post_init__(self):
        if get_rank() == 0:
            self.wandb
            if WANDB_API_KEY_ENV_VAR not in os.environ:
                raise OLMoEnvironmentError(f"missing env var '{WANDB_API_KEY_ENV_VAR}'")

    @property
    def wandb(self):
        if self._wandb is None:
            import wandb  # type: ignore

            self._wandb = wandb
        return self._wandb

    @property
    def run(self):
        return self.wandb.run

    def pre_train(self):
        if get_rank() == 0:
            wandb_dir = Path(self.trainer.save_folder) / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            self.wandb.init(
                dir=wandb_dir,
                project=self.project,
                entity=self.entity,
                group=self.group,
                name=self.name,
                tags=self.tags,
                config=self.config,
            )

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if get_rank() == 0:
            self.wandb.log(metrics, step=step)

    def post_step(self):
        if get_rank() == 0 and self.step % self.trainer.cancel_check_interval == 0:
            self.check_if_canceled()

    def post_train(self):
        if get_rank() == 0 and self.run is not None:
            self.wandb.finish(exit_code=0, quiet=True)

    def on_error(self, exc: BaseException):
        del exc
        if get_rank() == 0 and self.run is not None:
            self.wandb.finish(exit_code=1, quiet=True)

    def check_if_canceled(self):
        if self.cancel_tags:
            from requests.exceptions import RequestException
            from wandb.errors import CommError  # type: ignore

            try:
                api = self.wandb.Api(api_key=os.environ[WANDB_API_KEY_ENV_VAR])
                run = api.run(self.run.path)
                for tag in run.tags or []:
                    if tag.lower() in self.cancel_tags:
                        self.trainer.cancel_run("canceled from W&B tag")
                        return
            except (RequestException, CommError):
                log.warning("Failed to communicate with W&B API")
