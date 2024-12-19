import os
from dataclasses import dataclass
from typing import Optional

import requests

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import get_rank
from olmo_core.exceptions import OLMoEnvironmentError

from .callback import Callback

SLACK_WEBHOOK_URL_ENV_VAR = "SLACK_WEBHOOK_URL"


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
    Don't send any notifcations.
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
            self._post_message("completed successfully")

    def on_error(self, exc: BaseException):
        if not self.enabled or get_rank() != 0:
            return

        if self.notifications in (
            SlackNotificationSetting.all,
            SlackNotificationSetting.end_only,
            SlackNotificationSetting.failure_only,
        ):
            self._post_message(f"failed with error:\n{exc}")

    def _post_message(self, msg: str):
        webhook_url = self.webhook_url or os.environ.get(SLACK_WEBHOOK_URL_ENV_VAR)
        if webhook_url is None:
            raise OLMoEnvironmentError(f"missing env var '{SLACK_WEBHOOK_URL_ENV_VAR}'")

        if self.name is not None:
            msg = f"Run `{self.name}` {msg}"
        else:
            msg = f"Run {msg}"
        requests.post(webhook_url, json={"text": msg})
