import os
from dataclasses import dataclass
from typing import Optional

import requests

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import get_rank
from olmo_core.exceptions import OLMoEnvironmentError

from .callback import Callback

SLACK_WEBHOOK_URL_ENV_VAR = "SLACK_WEBHOOK_URL"
BEAKER_JOB_ID_ENV_VAR = "BEAKER_JOB_ID"
EXC_LINE_LIMIT = 30


class SlackNotificationSetting(StrEnum):
    """
    Defines the notifications settings for the Slack notifier callback.
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
    Don't send any notifications.
    """


@dataclass
class SlackNotifierCallback(Callback):
    name: Optional[str] = None
    """
    A name to give the run.
    """

    notifications: SlackNotificationSetting = SlackNotificationSetting.end_only
    """
    The notification settings.
    """

    enabled: bool = True
    """
    Set to false to disable this callback.
    """

    webhook_url: Optional[str] = None
    """
    The webhook URL to post. If not set, will check the environment variable ``SLACK_WEBHOOK_URL``.
    """

    def post_attach(self):
        if not self.enabled or get_rank() != 0:
            return

        if self.webhook_url is None and SLACK_WEBHOOK_URL_ENV_VAR not in os.environ:
            raise OLMoEnvironmentError(f"missing env var '{SLACK_WEBHOOK_URL_ENV_VAR}'")

    def pre_train(self):
        if not self.enabled or get_rank() != 0:
            return

        if self.notifications == SlackNotificationSetting.all:
            self._post_message("started")

    def post_train(self):
        if not self.enabled or get_rank() != 0:
            return

        if self.notifications in (
            SlackNotificationSetting.all,
            SlackNotificationSetting.end_only,
        ):
            if self.trainer.is_canceled:
                self._post_message("canceled")
            else:
                self._post_message("completed successfully")

    def on_error(self, exc: BaseException):
        if not self.enabled or get_rank() != 0:
            return

        if self.notifications in (
            SlackNotificationSetting.all,
            SlackNotificationSetting.end_only,
            SlackNotificationSetting.failure_only,
        ):
            exc_lines = str(exc).rstrip("\n").split("\n")
            if len(exc_lines) > EXC_LINE_LIMIT:
                exc_lines = exc_lines[:EXC_LINE_LIMIT]
                exc_lines.append("...")
            exc_str = "\n".join(exc_lines)
            self._post_message(f"failed with error:\n```\n{exc_str}\n```")

    def _post_message(self, msg: str):
        webhook_url = self.webhook_url or os.environ.get(SLACK_WEBHOOK_URL_ENV_VAR)
        if webhook_url is None:
            raise OLMoEnvironmentError(f"missing env var '{SLACK_WEBHOOK_URL_ENV_VAR}'")

        progress = (
            f"*Progress:*\n"
            f"- step: {self.step:,d}\n"
            f"- epoch: {self.trainer.epoch}\n"
            f"- tokens: {self.trainer.global_train_tokens_seen:,d}"
        )

        if self.name is not None:
            msg = f"Run `{self.name}` {msg}\n{progress}"
        else:
            msg = f"Run {msg}\n{progress}"

        if BEAKER_JOB_ID_ENV_VAR in os.environ:
            msg = f"{msg}\n*Beaker job:* https://beaker.org/job/{os.environ[BEAKER_JOB_ID_ENV_VAR]}"

        requests.post(webhook_url, json={"text": msg})
