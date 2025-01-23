import logging
import os
import random
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from comet_ml import Experiment

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import barrier, get_rank, is_distributed
from olmo_core.exceptions import OLMoConfigurationError, OLMoEnvironmentError
from olmo_core.utils import set_env_var

from .callback import Callback

log = logging.getLogger(__name__)

COMET_API_KEY_ENV_VAR = "COMET_API_KEY"
BEAKER_EXPERIMENT_ID_ENV_VAR = "BEAKER_EXPERIMENT_ID"


class CometNotificationSetting(StrEnum):
    """
    Defines the notifications settings for the Comet.ml callback.
    """

    all = "all"
    """
    Send all types notifications.
    """

    end_only = "end_only"
    """
    Only send a notification when the experiment ends (successfully or with a failure).
    """

    failure_only = "failure_only"
    """
    Only send a notification when the experiment fails.
    """

    none = "none"
    """
    Don't send any notifcations.
    """


@dataclass
class CometCallback(Callback):
    """
    Logs metrics to Comet.ml from rank 0.

    .. important::
        Requires the ``comet_ml`` package and the environment variable ``COMET_API_KEY``.

    .. note::
        This callback logs metrics from every single step to Comet.ml, regardless of the value
        of :data:`Trainer.metrics_collect_interval <olmo_core.train.Trainer.metrics_collect_interval>`.
    """

    enabled: bool = True
    """
    Set to false to disable this callback.
    """

    name: Optional[str] = None
    """
    The name to give the Comet.ml experiment.
    """

    project: Optional[str] = None
    """
    The Comet.ml project to use.
    """

    workspace: Optional[str] = None
    """
    The name of the Comet.ml workspace to use.
    """

    experiment_key: Optional[str] = None
    """
    An experiment identifier. Used to link different ranks of the same run to the same experiment.
    """

    tags: Optional[List[str]] = None
    """
    Tags to assign the experiment.
    """

    config: Optional[Dict[str, Any]] = None
    """
    The config to save to Comet.ml.
    """

    cancel_tags: Optional[List[str]] = field(
        default_factory=lambda: ["cancel", "canceled", "cancelled"]
    )
    """
    If you add any of these tags to an experiment on Comet.ml, the run will cancel itself.
    Defaults to ``["cancel", "canceled", "cancelled"]``.
    """

    cancel_check_interval: Optional[int] = None
    """
    Check for cancel tags every this many steps. Defaults to
    :data:`olmo_core.train.Trainer.cancel_check_interval`.
    """

    notifications: CometNotificationSetting = CometNotificationSetting.none
    """
    The notification settings.
    """

    failure_tag: str = "failed"
    """
    The tag to assign to failed experiments.
    """

    _exp = None
    _finalized: bool = False

    @property
    def exp(self) -> Experiment:
        return self._exp  # type: ignore

    @exp.setter
    def exp(self, exp: Experiment):
        self._exp = exp

    @property
    def finalized(self) -> bool:
        return self._finalized

    def finalize(self):
        if not self.finalized:
            self.exp.end()
            self._finalized = True

    def pre_train(self):
        if self.enabled:
            set_env_var("COMET_DISABLE_AUTO_LOGGING", "1")

            import comet_ml as comet

            if COMET_API_KEY_ENV_VAR not in os.environ:
                raise OLMoEnvironmentError(f"missing env var '{COMET_API_KEY_ENV_VAR}'")
            if is_distributed() and self.experiment_key is None:
                if BEAKER_EXPERIMENT_ID_ENV_VAR in os.environ:
                    log.info("Using beaker ID to create 32-50 character Comet.ml experiment id")
                    self.experiment_key = os.environ[BEAKER_EXPERIMENT_ID_ENV_VAR].rjust(32, "0")
                    assert len(self.experiment_key) <= 50
                else:
                    raise OLMoConfigurationError("Experiment key must be set in distributed contexts")

            # Start up an experiment that all the ranks can log to
            if get_rank() == 0:
                comet.Experiment(
                    api_key=os.environ[COMET_API_KEY_ENV_VAR],
                    project_name=self.project,
                    workspace=self.workspace,
                    experiment_key=self.experiment_key,
                    auto_output_logging="simple",
                    display_summary_level=0,
                    distributed_node_identifier=str(get_rank()),
                    log_env_gpu=True,
                    log_env_cpu=True,
                    log_env_network=True,
                    log_env_disk=True,
                    log_env_host=False,
                    log_env_details=True,
                )
            barrier()

            self.exp = comet.ExistingExperiment(
                api_key=os.environ[COMET_API_KEY_ENV_VAR],
                project_name=self.project,
                workspace=self.workspace,
                experiment_key=self.experiment_key,
                auto_output_logging="simple",
                display_summary_level=0,
                distributed_node_identifier=str(get_rank()),
                log_env_gpu=True,
                log_env_cpu=True,
                log_env_network=True,
                log_env_disk=True,
                log_env_host=False,
                log_env_details=True,
            )

            if get_rank() == 0:
                if self.name is not None:
                    self.exp.set_name(self.name)

                if self.tags:
                    self.exp.add_tags(self.tags)

                if self.config is not None:
                    self.exp.log_parameters(self.config)

                if self.notifications == CometNotificationSetting.all:
                    self.exp.send_notification(
                        f"Experiment {self.exp.get_name()} ({self.exp.get_key()})",
                        status="started",
                    )

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if self.enabled:
            self.exp.log_metrics(metrics, step=step)

    def post_step(self):
        cancel_check_interval = self.cancel_check_interval or self.trainer.cancel_check_interval
        if self.enabled and get_rank() == 0 and self.step % cancel_check_interval == 0:
            self.trainer.thread_pool.submit(self.check_if_canceled)

    def post_train(self):
        if self.enabled:
            if get_rank() == 0 and self.notifications in (
                CometNotificationSetting.all,
                CometNotificationSetting.end_only,
            ):
                self.exp.send_notification(
                    f"Experiment {self.exp.get_name()} ({self.exp.get_key()})",
                    status="completed successfully",
                )

            log.info("Finalizing successful Comet.ml experiment...")
            self.finalize()

    def on_error(self, exc: BaseException):
        del exc
        if self.enabled:
            if get_rank() == 0:
                if self.notifications in (
                    CometNotificationSetting.all,
                    CometNotificationSetting.end_only,
                    CometNotificationSetting.failure_only,
                ):
                    self.exp.send_notification(
                        f"Experiment {self.exp.get_name()} ({self.exp.get_key()})",
                        status="failed",
                    )

            log.warning("Finalizing failed Comet.ml experiment...")
            self.exp.add_tag(self.failure_tag)
            self.finalize()

    def check_if_canceled(self):
        if self.enabled and not self.finalized and self.cancel_tags:
            from comet_ml.api import API

            try:
                api = API(api_key=os.environ[COMET_API_KEY_ENV_VAR])
                exp = api.get_experiment_by_key(self.exp.get_key())
                assert exp is not None
                tags = exp.get_tags()
            except Exception as exc:
                log.exception(exc)
                return

            for tag in tags or []:
                if tag.lower() in self.cancel_tags:
                    self.trainer.cancel_run("canceled from Comet.ml tag")
                    return
