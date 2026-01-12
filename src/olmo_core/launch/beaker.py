"""
Launch experiments on `Beaker <https://beaker.org>`_.
"""

import argparse
import dataclasses
import logging
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Any

import requests
import rich
from beaker import Beaker, BeakerDataset, BeakerJob, BeakerWorkload
from beaker.exceptions import BeakerImageNotFound
from gantry.api import GitRepoState
from gantry.api import Recipe as GantryRecipe
from gantry.callbacks import Callback as GantryCallback

from ..config import Config, StrEnum
from ..distributed.utils import OLMO_SHARED_FS_ENV_VAR
from ..exceptions import OLMoConfigurationError, OLMoEnvironmentError
from ..train.callbacks.beaker import BEAKER_RESULT_DIR
from ..utils import (
    LOG_FILTER_TYPE_ENV_VAR,
    LogFilterType,
    generate_uuid,
    prepare_cli_environment,
)
from ..version import VERSION

log = logging.getLogger(__name__)


__all__ = [
    "OLMoCoreBeakerImage",
    "BeakerLaunchConfig",
    "BeakerEnvVar",
    "BeakerEnvSecret",
    "BeakerWekaBucket",
]


_BEAKER_CLIENT: Beaker | None = None
_DEFAULT_TORCH = "2.9.1".replace(".", "")
_DEFAULT_CUDA = "12.8".replace(".", "")


def is_running_in_beaker() -> bool:
    """
    Check if the current process is running inside of a Beaker job (batch or session).
    """
    # There's a number of different environment variables set by the Beaker executor.
    # Checking any one of these would suffice, but we check a couple to reduce the
    # risk of false positives.
    return "BEAKER_JOB_ID" in os.environ and "BEAKER_NODE_ID" in os.environ


def is_running_in_beaker_batch_job() -> bool:
    """
    Check if the current process is running inside a Beaker batch job (as opposed to a session).
    """
    return is_running_in_beaker() and os.environ.get("BEAKER_JOB_KIND") == "batch"


def get_beaker_experiment_id() -> str | None:
    if is_running_in_beaker_batch_job():
        experiment_id = os.environ.get("BEAKER_EXPERIMENT_ID")
        assert experiment_id is not None
        return experiment_id
    else:
        return None


def get_beaker_client(
    workspace: str | None = None, check_for_upgrades: bool | None = None
) -> Beaker:
    global _BEAKER_CLIENT
    if _BEAKER_CLIENT is None:
        defaults = {}
        if workspace is not None:
            defaults["default_workspace"] = workspace
        if check_for_upgrades is None:
            check_for_upgrades = not is_running_in_beaker()
        _BEAKER_CLIENT = Beaker.from_env(check_for_upgrades=check_for_upgrades, **defaults)
    elif workspace is not None:
        _BEAKER_CLIENT.config.default_workspace = workspace
    return _BEAKER_CLIENT


def close_beaker_client():
    global _BEAKER_CLIENT
    if _BEAKER_CLIENT is not None:
        _BEAKER_CLIENT.close()
        _BEAKER_CLIENT = None


class OLMoCoreBeakerImage(StrEnum):
    """
    Official Beaker images that work well for OLMo-core.

    You can find the full list at
    `beaker.org/ws/ai2/OLMo-core/images <https://beaker.org/ws/ai2/OLMo-core/images>`_, which
    includes *versioned* images that are published with each release of the OLMo-core package.
    """

    # NOTE: when updating default images here, should also update images used in tests at .github/workflows/main.yml
    stable = f"tylerr/olmo-core-tch{_DEFAULT_TORCH}cu{_DEFAULT_CUDA}-2025-11-25"
    """
    Built with the latest compatible stable version of PyTorch.
    """
    stable_cu130 = f"tylerr/olmo-core-tch{_DEFAULT_TORCH}cu130-2025-11-25"
    """
    The stable image with CUDA pinned to 13.0.
    """
    stable_cu128 = f"tylerr/olmo-core-tch{_DEFAULT_TORCH}cu128-2025-11-25"
    """
    The stable image with CUDA pinned to 12.8.
    """

    # Sorted roughly from newest versions to oldest versions
    tch280_cu128 = "tylerr/olmo-core-tch280cu128-2025-11-25"
    """
    Built with torch 2.8.0 and CUDA 12.8.
    """
    tch271_cu128 = "tylerr/olmo-core-tch271cu128-2025-11-25"
    """
    Built with torch 2.7.1 and CUDA 12.8.
    """
    tch270_cu128 = "petew/olmo-core-tch270cu128-2025-05-16"
    """
    Built with torch 2.7.0 and CUDA 12.8.
    Battle tested when training Olmo3 7B and 32B. No TransformerEngine or flash-attention-3.
    """
    tch271_cu126 = "petew/olmo-core-tch271cu126-2025-09-15"
    """
    Built with torch 2.7.1 and CUDA 12.6. No TransformerEngine or flash-attention-3.
    """


@dataclass
class BeakerEnvVar(Config):
    name: str
    value: str


@dataclass
class BeakerEnvSecret(Config):
    name: str
    secret: str


@dataclass
class BeakerWekaBucket(Config):
    bucket: str
    mount: str


@dataclass
class BeakerLaunchConfig(Config):
    """
    Config for launching experiments on Beaker.
    """

    name: str
    """
    A name to assign the Beaker experiment.
    """

    cmd: list[str]
    """
    The command to run in the container.
    """

    torchrun: bool | None = None
    """
    Launch the command with ``torchrun``. Defaults to true for multi-GPU jobs.
    """

    budget: str | None = None
    """
    The budget group to assign.
    """

    task_name: str = "train"
    """
    A name to assign the Beaker tasks created.
    """

    workspace: str | None = None
    """
    The Beaker workspace to use.
    """

    description: str | None = None
    """
    A description for the experiment.
    """

    beaker_image: str = OLMoCoreBeakerImage.stable
    """
    The Beaker image to use.

    Suitable images can be found at `beaker.org/ws/ai2/OLMo-core/images <https://beaker.org/ws/ai2/OLMo-core/images>`_.
    """

    num_nodes: int = 1
    """
    The number of nodes to use.
    """

    num_gpus: int = 8
    """
    The number of GPUs to use per node.
    """

    shared_memory: str = "10GiB"
    """
    The amount of shared memory to use.
    """

    clusters: list[str] = dataclasses.field(default_factory=lambda: ["ai2/jupiter"])
    """
    The allowed clusters to run on.
    """

    hostnames: list[str] | None = None
    """
    Manual hostname constraints. Takes priority over :data:`clusters` and other placement filters.
    """

    gpu_types: list[str] | None = None
    """Cluster GPU type constraints."""

    tags: list[str] | None = None
    """Cluster tag constraints."""

    shared_filesystem: bool = False
    """
    Set this to true if the save folder and working directory for each node is part of a global
    shared filesystem (like weka or NFS).
    """

    priority: str = "normal"
    """
    The job priority.
    """

    preemptible: bool = True
    """
    If the job should be preemptible.
    """

    retries: int | None = None
    """
    The number of times to retry the experiment if it fails.
    """

    env_vars: list[BeakerEnvVar] = dataclasses.field(default_factory=list)
    """
    Additional env vars to include.
    """

    env_secrets: list[BeakerEnvSecret] = dataclasses.field(default_factory=list)
    """
    Environment variables to add from secrets.
    """

    google_credentials_secret: str | None = None
    """Name of the Beaker secret containing Google credentials JSON, if needed."""

    aws_config_secret: str | None = None
    """The name of the Beaker secret containing an AWS config file, if needed."""

    aws_credentials_secret: str | None = None
    """The name of the Beaker secret containing an AWS credentials file, if needed."""

    weka_buckets: list[BeakerWekaBucket] = dataclasses.field(default_factory=list)
    """
    Weka buckets to attach and where to attach them.
    """

    allow_dirty: bool = False
    """
    Allow running with uncommitted changed.
    """

    host_networking: bool | None = None
    """Enable host-networking."""

    git: GitRepoState = dataclasses.field(default_factory=GitRepoState.from_env)
    """
    Git configuration, specifies where to clone your source code from and which commit to check out.
    If not set, this will be initialized automatically from your working directory.
    """

    result_dir: str = BEAKER_RESULT_DIR
    """
    The directory of the Beaker results dataset.
    """

    system_python: bool = True
    """Use the system Python installation in the Beaker image."""

    num_execution_units: int | None = None
    """
    Number of "execution units", defaults to 1. An "execution unit" is abstraction
    for any node-using entity of which 1 or more copies are run, where each unit wants its nodes to be
    from colocated hardware (e.g., a model replica for large jobs, or a full distributed model for small jobs).

    For example, when training with HSDP it would make sense to set ``num_execution_units`` to
    the replica degree of the device mesh.
    """

    follow: bool = True
    """Follow the experiment logs locally after launching."""

    slack_notifications: bool | None = None
    """
    Get Slack notifications for experiment status updates when following logs.
    Defaults to true if ``follow`` is true and the env var `SLACK_WEBHOOK_URL` is set.
    """

    launch_timeout: int | None = None
    """
    A timeout in seconds to wait for the job to start after submitting it.
    If the job doesn't start in time a timeout error will be raised.
    """

    step_timeout: int | None = None
    """
    A timeout in seconds to wait for the first training step when ``follow=True``.
    If a step isn't detected in a time a timeout error will be raised.
    """

    step_soft_timeout: int | None = None
    """
    A soft timeout in seconds to wait for the first training step when ``follow=True``.
    If a step isn't detected in a time warning will be issued.
    """

    @property
    def default_env_vars(self) -> list[tuple[str, str]]:
        """
        Default env vars to add to the experiment.
        """
        env_vars: list[tuple[str, str]] = [
            ("NCCL_DEBUG", "INFO"),
            (LOG_FILTER_TYPE_ENV_VAR, LogFilterType.local_rank0_only),
            ("OMP_NUM_THREADS", "8"),
            ("R2_PROFILE", "R2"),
            ("S3_PROFILE", "S3"),
            ("WEKA_PROFILE", "WEKA"),
            ("NUM_NODES", str(self.num_nodes)),
            ("OLMO_CORE_VERSION", VERSION),
            ("FORCE_COLOR", "1"),  # for 'rich' because Beaker supports ANSI colors in logs
            ("TORCH_LOGS", "recompiles,graph_breaks"),
            ("PYTORCH_KERNEL_CACHE_PATH", "/root/.cache/torch/kernels"),
        ]
        if self.shared_filesystem:
            env_vars.append((OLMO_SHARED_FS_ENV_VAR, "1"))
        return env_vars

    @property
    def beaker(self) -> Beaker:
        """
        The Beaker client.
        """
        return get_beaker_client(workspace=self.workspace)

    def _get_env_vars(self) -> list[tuple[str, str]]:
        env_vars: list[tuple[str, str]] = []
        env_var_names: set[str] = set()
        for var in self.env_vars:
            env_vars.append((var.name, var.value))
            env_var_names.add(var.name)
        for name, val in self.default_env_vars:
            if name not in env_var_names:
                env_vars.append((name, val))
        return env_vars

    def _resolve_beaker_image(self) -> str:
        image = self.beaker_image
        try:
            return self.beaker.image.get(image).id
        except BeakerImageNotFound as exc:
            # Image name was already a full name, so it probably doesn't exist.
            if "/" in image:
                raise

            # Try pre-pending 'petew', since that's the account that we usually build the images from.
            try:
                return self.beaker.image.get(f"petew/{image}").id
            except BeakerImageNotFound:
                raise exc

    def launch(
        self,
        follow: bool | None = None,
        slack_notifications: bool | None = None,
        launch_timeout: int | None = None,
        step_timeout: int | None = None,
        step_soft_timeout: int | None = None,
        torchrun: bool | None = None,
    ) -> BeakerWorkload:
        """
        Launch a Beaker experiment using this config.

        .. tip::
            You can preview what the Beaker experiment spec would like using
            :meth:`build_experiment_spec()`.

        :param follow: Stream the logs and follow the experiment until completion.
        :param torchrun: Launch the target command with ``torchrun``. This will default to ``True``
            if ``num_gpus > 1`` and ``False`` otherwise.
        :param entrypoint: Provide an optional entrypoint program if ``torchrun`` is ``False``.
            Defaults to 'python'.
        :param slack_notifications: If ``follow=True``, send Slack notifications when the run launches,
            fails, or succeeds. This requires the env var ``SLACK_WEBHOOK_URL``.
        :param launch_timeout: A timeout in seconds to wait for the job to start after submitting it.
            If the job doesn't start in time a timeout error will be raised.

        :returns: The Beaker experiment.
        """
        follow = follow if follow is not None else self.follow
        slack_notifications = (
            slack_notifications if slack_notifications is not None else self.slack_notifications
        )
        launch_timeout = launch_timeout if launch_timeout is not None else self.launch_timeout
        step_timeout = step_timeout if step_timeout is not None else self.step_timeout
        step_soft_timeout = (
            step_soft_timeout if step_soft_timeout is not None else self.step_soft_timeout
        )
        torchrun = torchrun if torchrun is not None else self.torchrun
        if torchrun is None:
            if self.num_gpus > 1 or (self.num_gpus >= 1 and self.num_nodes > 1):
                torchrun = True
            else:
                torchrun = False

        if self.git.is_dirty and not self.allow_dirty:
            raise RuntimeError(
                "You have uncommitted changes! Set 'allow_dirty=True' in your launch config to force."
            )

        # Check for webhook URL env var if needed.
        slack_webhook_url: str | None = None
        if follow and slack_notifications is not False:
            from olmo_core.train.callbacks.slack_notifier import (
                SLACK_WEBHOOK_URL_ENV_VAR,
            )

            if SLACK_WEBHOOK_URL_ENV_VAR in os.environ:
                slack_webhook_url = os.environ[SLACK_WEBHOOK_URL_ENV_VAR]
            else:
                # Pull from secret if available.
                for env_secret in self.env_secrets:
                    if env_secret.name == SLACK_WEBHOOK_URL_ENV_VAR:
                        secret = self.beaker.secret.get(env_secret.secret)
                        slack_webhook_url = self.beaker.secret.read(secret)
                        break

            if slack_notifications is None:
                slack_notifications = slack_webhook_url is not None
            elif slack_notifications and slack_webhook_url is None:
                raise OLMoEnvironmentError(
                    f"Missing env var / secret '{SLACK_WEBHOOK_URL_ENV_VAR}' for Slack notifications"
                )

        if not follow and slack_notifications:
            raise OLMoConfigurationError("Slack notifications require 'follow=True'")
        if not follow and step_timeout is not None:
            raise OLMoConfigurationError("Step timeout requires 'follow=True'")
        if not follow and step_soft_timeout is not None:
            raise OLMoConfigurationError("Step soft timeout requires 'follow=True'")

        recipe = GantryRecipe(
            self.cmd,
            name=self.name,
            task_name=self.task_name,
            description=self.description,
            workspace=self.workspace,
            budget=self.budget,
            priority=self.priority,
            preemptible=self.preemptible,
            # Inputs.
            beaker_image=self._resolve_beaker_image(),
            env_vars=self._get_env_vars(),
            env_secrets=[(s.name, s.secret) for s in self.env_secrets],
            google_credentials_secret=self.google_credentials_secret,
            aws_config_secret=self.aws_config_secret,
            aws_credentials_secret=self.aws_credentials_secret,
            weka=[(b.bucket, b.mount) for b in self.weka_buckets],
            # Outputs.
            results=self.result_dir,
            # Python settings.
            system_python=self.system_python,
            # Git settings.
            git_repo=self.git,
            allow_dirty=self.allow_dirty,
            # Resources.
            gpus=self.num_gpus,
            shared_memory=self.shared_memory,
            # Placement.
            clusters=self.clusters,
            hostnames=self.hostnames,
            gpu_types=self.gpu_types,
            tags=self.tags,
            # Multi-node settings.
            replicas=self.num_nodes if self.num_nodes > 1 else None,
            leader_selection=self.num_nodes > 1,
            host_networking=self.host_networking
            if self.host_networking is not None
            else self.num_nodes > 1,
            propagate_failure=True if self.num_nodes > 1 else None,
            propagate_preemption=True if self.num_nodes > 1 else None,
            synchronized_start_timeout="90m" if self.num_nodes > 1 else None,
            # Retry settings.
            retries=self.retries,
            # Callbacks.
            callbacks=[
                GantryMonitorCallback(
                    slack_webhook_url=slack_webhook_url if slack_notifications else None,
                    step_timeout=step_timeout,
                    step_soft_timeout=step_soft_timeout,
                )
            ],
        )

        return recipe.launch(
            show_logs=follow,
            start_timeout=launch_timeout,
            inactive_timeout=step_timeout,
            inactive_soft_timeout=step_soft_timeout,
        )


# Regex for detecting training (and eval) steps in logs.
_STEP_REGEX = re.compile(r"\[olmo_core\..+\].+\[.*step\=\d+.*\]")


@GantryCallback.register("olmo_core.monitor")
@dataclass(kw_only=True)
class GantryMonitorCallback(GantryCallback):
    slack_webhook_url: str | None
    step_timeout: int | None = None
    step_soft_timeout: int | None = None

    _start_time: float = dataclasses.field(repr=False, default_factory=time.monotonic)
    _first_step_detected: bool = dataclasses.field(repr=False, default=False)
    _last_step_time: float = dataclasses.field(repr=False, default=0.0)
    _last_inactivity_warning: float = dataclasses.field(repr=False, default=0.0)

    @property
    def workload_name(self) -> str:
        return f"{self.beaker.user_name}/{self.workload.experiment.name}"

    @property
    def workload_url(self) -> str:
        return self.beaker.workload.url(self.workload)

    def on_start(self, job: BeakerJob):
        del job
        self._start_time = time.monotonic()
        if self.slack_webhook_url is not None:
            requests.post(
                self.slack_webhook_url,
                json={
                    "text": f":check: Workload <{self.workload_url}|*{self.workload_name}*> has started! :runner:"
                },
            )

    def on_start_timeout(self, job: BeakerJob):
        del job
        if self.slack_webhook_url is not None:
            requests.post(
                self.slack_webhook_url,
                json={
                    "text": f":warning: Workload <{self.workload_url}|*{self.workload_name}*> failed to start in time!"
                },
            )

    def on_log(self, job: BeakerJob, log_line: str, log_time: float):
        del job, log_time
        if self.step_timeout is not None or self.step_soft_timeout is not None:
            if _STEP_REGEX.search(log_line) is not None:
                self._first_step_detected = True
                self._last_step_time = time.monotonic()

    def on_no_new_logs(self, job: BeakerJob):
        del job
        cur_time = time.monotonic()

        # If (a) we've detected training steps already or (b) the run has been up for over 30 min,
        # then we warn if we haven't detected new steps within the past `step_soft_timeout` seconds.
        # But we only send a warning at most once per hour.
        if (
            self.step_soft_timeout is not None
            and (
                self._first_step_detected
                or (cur_time - self._start_time) > max(self.step_soft_timeout, 1800)
            )
            and (cur_time - self._last_step_time) > self.step_soft_timeout
            and (cur_time - self._last_inactivity_warning) > 3600
        ):
            if self.slack_webhook_url is not None:
                requests.post(
                    self.slack_webhook_url,
                    json={
                        "text": f":zzz: Workload <{self.workload_url}|*{self.workload_name}*> hasn't stepped recently!"
                    },
                )
                self._last_inactivity_warning = time.monotonic()

        # If (a) we've detected training steps already or (b) the run has been up for over 60 min,
        # then we kill the job if we haven't detected new steps within the past `step_timeout` seconds.
        if (
            self.step_timeout is not None
            and (
                self._first_step_detected
                or (cur_time - self._start_time) > max(self.step_timeout, 3600)
            )
            and (cur_time - self._last_step_time) > self.step_timeout
        ):
            if self.slack_webhook_url is not None:
                requests.post(
                    self.slack_webhook_url,
                    json={
                        "text": f":warning: Workload <{self.workload_url}|*{self.workload_name}*> failed to step in time!"
                    },
                )
            self.interrupt_workload()

    def on_timeout(self, job: BeakerJob):
        del job
        if self.slack_webhook_url is not None:
            requests.post(
                self.slack_webhook_url,
                json={
                    "text": f":warning: Workload <{self.workload_url}|*{self.workload_name}*> failed to complete in time!"
                },
            )

    def on_inactive_timeout(self, job: BeakerJob):
        del job
        if self.slack_webhook_url is not None:
            requests.post(
                self.slack_webhook_url,
                json={
                    "text": f":zzz: Workload <{self.workload_url}|*{self.workload_name}*> appears to be inactive!"
                },
            )

    def on_inactive_soft_timeout(self, job: BeakerJob):
        del job
        if self.slack_webhook_url is not None:
            requests.post(
                self.slack_webhook_url,
                json={
                    "text": f":zzz: Workload <{self.workload_url}|*{self.workload_name}*> appears to be inactive!"
                },
            )
            self._last_inactivity_warning = time.monotonic()

    def on_preemption(self, job: BeakerJob):
        del job
        if self.slack_webhook_url is not None:
            requests.post(
                self.slack_webhook_url,
                json={
                    "text": f":warning: Workload <{self.workload_url}|*{self.workload_name}*> was preempted!"
                },
            )

    def on_cancellation(self, job: BeakerJob | None):
        del job
        if self.slack_webhook_url is not None:
            requests.post(
                self.slack_webhook_url,
                json={
                    "text": f":warning: Workload <{self.workload_url}|*{self.workload_name}*> was canceled!"
                },
            )

    def on_failure(
        self,
        job: BeakerJob,
        *,
        metrics: dict[str, Any] | None = None,
        results_ds: BeakerDataset | None = None,
    ):
        del job, metrics, results_ds
        if self.slack_webhook_url is not None:
            requests.post(
                self.slack_webhook_url,
                json={
                    "text": f":check-failed: Workload <{self.workload_url}|*{self.workload_name}*> failed!"
                },
            )

    def on_success(
        self,
        job: BeakerJob,
        *,
        metrics: dict[str, Any] | None = None,
        results_ds: BeakerDataset | None = None,
    ):
        del job, metrics, results_ds
        if self.slack_webhook_url is not None:
            requests.post(
                self.slack_webhook_url,
                json={
                    "text": f":check: Workload <{self.workload_url}|*{self.workload_name}*> succeeded!"
                },
            )


def _parse_args():
    parser = argparse.ArgumentParser(
        "olmo_core.launch.beaker",
        usage="python -m olmo_core.launch.beaker [OPTIONS...] -- [CMD...]",
        description=textwrap.dedent(
            """
            Launch a command on Beaker.
            """
        ),
        epilog=textwrap.dedent(
            """
            examples:
              ❯ python -m olmo_core.launch.beaker -- echo "Hello, World!"
            """
        ),
        formatter_class=type(  # type: ignore[arg-type]
            "CustomFormatter",
            (
                argparse.ArgumentDefaultsHelpFormatter,
                argparse.RawDescriptionHelpFormatter,
            ),
            {},
        ),
    )
    parser.add_argument(
        "--name", type=str, default="olmo-core-test", help="A name to assign to the run."
    )
    parser.add_argument(
        "--task-name", type=str, default="main", help="A name to assign to the task."
    )
    parser.add_argument("--gpus", type=int, default=0, help="The number of GPUs per node/replica.")
    parser.add_argument("--nodes", type=int, default=1, help="The number of nodes/replicas.")
    parser.add_argument("--budget", type=str, help="The Beaker budget account to use.")
    parser.add_argument("--workspace", type=str, help="The Beaker workspace to use.")
    parser.add_argument(
        "--description", type=str, help="A description to assign to the Beaker experiment."
    )
    parser.add_argument(
        "--cluster",
        type=str,
        nargs="*",
        help="""Clusters to launch on (multiple allowed).""",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        nargs="*",
        help="""GPU type constraints (multiple allowed).""",
    )
    parser.add_argument(
        "--tag",
        type=str,
        nargs="*",
        help="""Cluster tag constraints (multiple allowed).""",
    )
    parser.add_argument(
        "--priority",
        choices=["low", "normal", "high", "urgent"],
        default="low",
        help="The priority level.",
    )
    parser.add_argument(
        "--preemptible",
        action="store_true",
        help="""If the job should be preemptible.""",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="""Allow launching with uncommitted changes.""",
        default=False,
    )
    parser.add_argument(
        "--beaker-image",
        type=str,
        default=OLMoCoreBeakerImage.stable,
        help="""The Beaker image to use.""",
    )
    parser.add_argument(
        "--shared-filesystem",
        action="store_true",
        help="""Use this flag if the save folder and working directory for each node is part of a global
        shared filesystem (like weka or NFS).""",
    )
    parser.add_argument("--weka", type=str, nargs="*", help="Weka buckets to mount at '/weka/'.")
    parser.add_argument(
        "--torchrun",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="""If the command should be run via torchrun. This will default to true multi-GPU jobs.""",
    )
    parser.add_argument(
        "--slack-notifications",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="""Send Slack notifications.""",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="""Set debugging env vars, like 'CUDA_LAUNCH_BLOCKING=1'.""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="""Do a dry run where the launch config is printed.""",
    )
    parser.add_argument(
        "--env",
        type=str,
        nargs="*",
        help="""Environment variables to add to the Beaker experiment.
        Should be in the form '{NAME}={VALUE}'. Multiple allowed, space separated.""",
    )
    parser.add_argument(
        "--env-secret",
        type=str,
        nargs="*",
        help="""Environment variables to add to the Beaker experiment from Beaker secrets.
        Should be in the form '{NAME}={SECRET_NAME}'. Multiple allowed, space separated.""",
    )

    if len(sys.argv) < 3 or "--" not in sys.argv:
        parser.print_help()
        sys.exit(1)

    sep_index = sys.argv.index("--")
    args = sys.argv[1:sep_index]
    command = sys.argv[sep_index + 1 :]
    opts = parser.parse_args(args)
    return opts, command


def _build_config(opts: argparse.Namespace, command: list[str]) -> BeakerLaunchConfig:
    env_vars: list[BeakerEnvVar] = []
    if opts.debug:
        env_vars.append(BeakerEnvVar(name="CUDA_LAUNCH_BLOCKING", value="1"))
        env_vars.append(BeakerEnvVar(name="NCCL_DEBUG", value="INFO"))
    for e in opts.env or []:
        if "=" not in e:
            raise ValueError(f"Invalid env var '{e}', must be in the form NAME=VALUE")
        name, value = e.split("=", 1)
        env_vars.append(BeakerEnvVar(name=name, value=value))
    env_secrets: list[BeakerEnvSecret] = []
    for e in opts.env_secret or []:
        if "=" not in e:
            raise ValueError(f"Invalid env secret '{e}', must be in the form NAME=SECRET_NAME")
        name, secret = e.split("=", 1)
        env_secrets.append(BeakerEnvSecret(name=name, secret=secret))
    return BeakerLaunchConfig(
        name=f"{opts.name}-{generate_uuid()[:8]}",
        budget=opts.budget,
        cmd=command,
        env_vars=env_vars,
        env_secrets=env_secrets,
        task_name=opts.task_name,
        description=opts.description,
        clusters=opts.cluster,
        gpu_types=opts.gpu_type,
        tags=opts.tag,
        num_nodes=opts.nodes,
        num_gpus=opts.gpus,
        preemptible=opts.preemptible,
        priority=opts.priority,
        beaker_image=opts.beaker_image,
        slack_notifications=opts.slack_notifications,
        workspace=opts.workspace,
        allow_dirty=opts.allow_dirty,
        shared_filesystem=opts.shared_filesystem,
        weka_buckets=[
            BeakerWekaBucket(bucket=bucket, mount=f"/weka/{bucket}") for bucket in (opts.weka or [])
        ],
        torchrun=opts.torchrun,
    )


def main():
    opts, command = _parse_args()
    prepare_cli_environment()
    try:
        config = _build_config(opts, command)
        if opts.dry_run:
            rich.print(config)
        else:
            config.launch(follow=True)
    finally:
        close_beaker_client()


if __name__ == "__main__":
    main()
