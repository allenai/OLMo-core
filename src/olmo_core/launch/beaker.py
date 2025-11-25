"""
Launch experiments on `Beaker <https://beaker.org>`_.
"""

import argparse
import hashlib
import logging
import os
import re
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import List, Literal, Optional, Set, Tuple

import requests
import rich
from beaker import (
    Beaker,
    Dataset,
    DatasetConflict,
    DatasetNotFound,
    Experiment,
    ExperimentConflict,
    ExperimentSpec,
    ImageNotFound,
    Job,
    Priority,
    RetrySpec,
    TaskResources,
    TaskSpec,
)
from rich.prompt import Confirm

from ..config import Config, StrEnum
from ..distributed.utils import OLMO_SHARED_FS_ENV_VAR
from ..exceptions import (
    BeakerExperimentFailedError,
    OLMoConfigurationError,
    OLMoEnvironmentError,
)
from ..train.callbacks.beaker import BEAKER_RESULT_DIR
from ..utils import (
    LOG_FILTER_TYPE_ENV_VAR,
    LogFilterType,
    generate_uuid,
    prepare_cli_environment,
)
from ..version import VERSION
from .select_beaker_hosts import get_beaker_hostname_constraints
from .utils import GIT_BRANCH_ENV_VAR, GIT_REF_ENV_VAR, GIT_REPO_URL_ENV_VAR, GitConfig

log = logging.getLogger(__name__)


__all__ = [
    "OLMoCoreBeakerImage",
    "BeakerLaunchConfig",
    "BeakerEnvVar",
    "BeakerEnvSecret",
    "BeakerWekaBucket",
    "BeakerPriority",
]


BeakerPriority = Priority

_DEFAULT_TORCH = "2.9.1".replace(".", "")
_DEFAULT_CUDA = "12.8".replace(".", "")


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
    tch270_cu128 = "olmo-core-tch270cu128-2025-05-16"
    """
    Built with torch 2.7.0 and CUDA 12.8.
    Battle tested when training Olmo3 7B and 32B. No TransformerEngine or flash-attention-3.
    """
    tch271_cu126 = "olmo-core-tch271cu126-2025-09-15"
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


DEFAULT_SETUP_STEPS = (
    f'if [[ -z "${GIT_BRANCH_ENV_VAR}" ]]; then',
    f'  git clone "${GIT_REPO_URL_ENV_VAR}" .',
    "else",
    f'  git clone -b "${GIT_BRANCH_ENV_VAR}" --single-branch "${GIT_REPO_URL_ENV_VAR}" .',
    "fi",
    f'git checkout "${GIT_REF_ENV_VAR}"',
    "git submodule update --init --recursive",
    "conda shell.bash activate base",
    "pip install -e '.[all]'",
    "pip freeze",
)


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


@dataclass
class BeakerLaunchConfig(Config):
    """
    Config for launching experiments on Beaker.
    """

    name: str
    """
    A name to assign the Beaker experiment.
    """

    cmd: List[str]
    """
    The command to run in the container.
    """

    budget: Optional[str] = None
    """
    The budget group to assign.
    """

    task_name: str = "train"
    """
    A name to assign the Beaker tasks created.
    """

    workspace: Optional[str] = None
    """
    The Beaker workspace to use.
    """

    description: Optional[str] = None
    """
    A description for the experiment.
    """

    setup_steps: List[str] = field(default_factory=lambda: list(DEFAULT_SETUP_STEPS))
    """
    A list of shell commands to run for cloning your repo, installing dependencies,
    and other arbitrary setup steps.
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

    clusters: List[str] = field(default_factory=lambda: ["ai2/jupiter"])
    """
    The allowed clusters to run on.
    """

    shared_filesystem: bool = False
    """
    Set this to true if the save folder and working directory for each node is part of a global
    shared filesystem (like weka or NFS).
    """

    priority: Priority = Priority.normal
    """
    The job priority.
    """

    preemptible: bool = True
    """
    If the job should be preemptible.
    """

    retries: Optional[int] = None
    """
    The number of times to retry the experiment if it fails.
    """

    env_vars: List[BeakerEnvVar] = field(default_factory=list)
    """
    Additional env vars to include.
    """

    env_secrets: List[BeakerEnvSecret] = field(default_factory=list)
    """
    Environment variables to add from secrets.
    """

    nfs: bool = False
    """
    Attach the NFS drive.
    """

    weka_buckets: List[BeakerWekaBucket] = field(default_factory=list)
    """
    Weka buckets to attach and where to attach them.
    """

    allow_dirty: bool = False
    """
    Allow running with uncommitted changed.
    """

    host_networking: Optional[bool] = None

    git: Optional[GitConfig] = field(default_factory=GitConfig.from_env)
    """
    Git configuration, specifies where to clone your source code from and which commit to check out.
    If not set, this will be initialized automatically from your working directory.
    """

    result_dir: str = BEAKER_RESULT_DIR
    """
    The directory of the Beaker results dataset.
    """

    use_hostname_constraints: bool = False
    """
    Uses hostname constraints to restrict the hostnames on which the experiment runs. This is currently
    only supported for Augusta clusters, and can benefit performance by forcing the use of colocated nodes.

    This is NOT recommended to be used with lower priority preemptible jobs, since hostname constraints are not
    updated on preemption.
    """

    hostnames: Optional[List[str]] = None
    """
    Manual hostname constraints. Takes priority over :data:`clusters` and :data:`use_hostname_constraints`.
    """

    num_execution_units: Optional[int] = None
    """
    Number of "execution units", defaults to ``max(1, num_nodes // 32)``. An "execution unit" is abstraction
    for any node-using entity of which 1 or more copies are run, where each unit wants its nodes to be
    from colocated hardware (e.g., a model replica for large jobs, or a full distributed model for small jobs).

    For internal experiments, this defaults to the number of data-parallel model replicas instead.
    """

    launch_timeout: Optional[int] = None
    """
    A timeout in seconds to wait for the job to start after submitting it.
    If the job doesn't start in time a timeout error will be raised.
    """

    # NOTE: don't assign a type here because omegaconf can't validate arbitrary classes
    #  _beaker: Optional[Beaker] = None
    _beaker = None

    @property
    def default_env_vars(self) -> List[Tuple[str, str]]:
        """
        Default env vars to add to the experiment.
        """
        env_vars: List[Tuple[str, str]] = [
            ("NCCL_DEBUG", "INFO"),
            (LOG_FILTER_TYPE_ENV_VAR, LogFilterType.local_rank0_only),
            ("OMP_NUM_THREADS", "8"),
            ("R2_PROFILE", "R2"),
            ("S3_PROFILE", "S3"),
            ("WEKA_PROFILE", "WEKA"),
            ("NUM_NODES", str(self.num_nodes)),
            ("OLMO_CORE_VERSION", VERSION),
            ("FORCE_COLOR", "1"),  # for 'rich' because Beaker supports ANSI colors in logs
        ]
        if self.shared_filesystem:
            env_vars.append((OLMO_SHARED_FS_ENV_VAR, "1"))
        return env_vars

    @property
    def beaker(self) -> Beaker:
        """
        The Beaker client.
        """
        if self._beaker is None:
            kwargs = {}
            if self.workspace:
                kwargs["default_workspace"] = self.workspace
            self._beaker = Beaker.from_env(**kwargs)
        return self._beaker

    def _get_env_vars(self) -> List[Tuple[str, str]]:
        env_vars: List[Tuple[str, str]] = []
        env_var_names: Set[str] = set()
        for var in self.env_vars:
            env_vars.append((var.name, var.value))
            env_var_names.add(var.name)
        for name, val in self.default_env_vars:
            if name not in env_var_names:
                env_vars.append((name, val))
        return env_vars

    def _get_torchrun_cmd(self) -> List[str]:
        assert self.num_nodes >= 1

        torchrun: List[str]
        if self.num_nodes == 1:
            torchrun = ["torchrun", f"--nproc-per-node={self.num_gpus}"]
        else:
            torchrun = [
                "torchrun",
                f"--nnodes={self.num_nodes}:{self.num_nodes}",
                f"--nproc-per-node={self.num_gpus}",
                "--rdzv_id=12347",
                "--rdzv_backend=static",
                '--rdzv_endpoint="${BEAKER_LEADER_REPLICA_HOSTNAME}:29400"',
                '--node_rank="${BEAKER_REPLICA_RANK}"',
                "--rdzv_conf='read_timeout=420'",
            ]

        return torchrun

    def _create_script_dataset(self, script_name: str, script: List[str]) -> Dataset:
        workspace_id = self.beaker.workspace.get(self.workspace).id

        # Hash contents.
        sha256_hash = hashlib.sha256()
        for line in script:
            sha256_hash.update(line.encode())

        # Create unique name for dataset.
        dataset_name = f"olmo-core-v{VERSION}-{workspace_id}-{sha256_hash.hexdigest()[:6]}"

        dataset: Dataset
        try:
            dataset = self.beaker.dataset.get(dataset_name)
        except DatasetNotFound:
            # Create it.
            log.info(f"Creating script dataset '{dataset_name}'...")
            try:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    tmpdir = Path(tmpdirname)
                    script_path = tmpdir / script_name
                    with open(script_path, "w") as script_file:
                        for line in script:
                            script_file.write(line + "\n")
                    dataset = self.beaker.dataset.create(dataset_name, script_path)
            except DatasetConflict:  # could be in a race with another process.
                time.sleep(1.0)
                dataset = self.beaker.dataset.get(dataset_name)

        return dataset

    def _resolve_beaker_image(self) -> str:
        image = self.beaker_image
        try:
            return self.beaker.image.get(image).id
        except ImageNotFound as exc:
            # Image name was already a full name, so it probably doesn't exist.
            if "/" in image:
                raise

            # Try pre-pending 'petew', since that's the account that we usually build the images from.
            try:
                return self.beaker.image.get(f"petew/{image}").id
            except ImageNotFound:
                raise exc

    def build_experiment_spec(
        self, torchrun: Optional[bool] = None, entrypoint: Optional[str] = None
    ) -> ExperimentSpec:
        """
        Get the Beaker experiment spec corresponding to this config instance.
        """
        if torchrun is None:
            if "torchrun" in self.cmd:
                torchrun = False
            else:
                torchrun = self.num_gpus > 1

        if self.git is None:
            raise OLMoConfigurationError(
                f"{self.__class__.__name__}.git field is required!\n"
                "You either need to instantiate your launch config from a valid git repository folder or set the 'git' field manually."
            )

        if self.git.is_dirty and not self.allow_dirty:
            raise RuntimeError(
                "You have uncommitted changes! Set 'allow_dirty=True' in your launch config to force."
            )

        if not self.git.is_public and self.setup_steps == DEFAULT_SETUP_STEPS:
            raise OLMoConfigurationError(
                "It looks like your repository is private and private repositories will require "
                "custom 'setup_steps' in order to clone the repo."
            )

        entrypoint_script = [
            "#!/usr/bin/env bash",
            "set -exo pipefail",
            "[[ -d /var/lib/tcpxo/lib64 ]] && export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:$LD_LIBRARY_PATH",
            # Setup the kernel cache directory used by pytorch
            "mkdir -p /root/.cache/torch/kernels && export PYTORCH_KERNEL_CACHE_PATH=/root/.cache/torch/kernels",
            "mkdir -p /olmo-core-runtime",
            "cd /olmo-core-runtime",
        ] + self.setup_steps

        if torchrun:
            if self.num_nodes > 1 and any(["augusta" in cluster for cluster in self.clusters]):
                entrypoint_script.append(
                    "BEAKER_REPLICA_RANK=$("
                    "python -m olmo_core.launch.reorder_ranks_in_gcp "
                    "--verbose "
                    "${BEAKER_REPLICA_RANK} "
                    "${BEAKER_REPLICA_COUNT} "
                    "${BEAKER_LEADER_REPLICA_HOSTNAME}"
                    ")"
                )
                entrypoint_script.append("export BEAKER_REPLICA_RANK=$BEAKER_REPLICA_RANK")
            entrypoint_script.append("exec " + " ".join(self._get_torchrun_cmd()) + ' "$@"')
        elif entrypoint:
            entrypoint_script.append(f'{entrypoint} "$@"')
        elif self.cmd and os.path.isfile(self.cmd[0]) and self.cmd[0].endswith(".py"):
            entrypoint_script.append('python "$@"')
        else:
            entrypoint_script.append('exec "$@"')

        entrypoint_dataset = self._create_script_dataset("entrypoint.sh", entrypoint_script)

        if self.hostnames:
            constraints_kwargs = {"hostname": self.hostnames}
        elif (
            self.use_hostname_constraints
            and len(self.clusters) == 1
            and "augusta" in self.clusters[0]
        ):
            if self.retries is not None and self.retries > 0:
                raise OLMoConfigurationError(
                    "Hostname constraints cannot be used for beaker jobs with retries, since constraints do not update on retry."
                )

            host_name_constraints = get_beaker_hostname_constraints(
                self.num_nodes,
                self.num_execution_units or max(1, self.num_nodes // 32),
                1,
                "us-central1-b",
                beaker_cluster=self.clusters[0],
                beaker_priority=self.priority,
            )
            assert (
                len(host_name_constraints) == 1 and len(host_name_constraints[0]) >= self.num_nodes
            )
            constraints_kwargs = {"hostname": host_name_constraints[0]}
        else:
            constraints_kwargs = {"cluster": self.clusters}

        task_spec = (
            TaskSpec.new(
                self.task_name,
                beaker_image=self._resolve_beaker_image(),
                priority=self.priority,
                preemptible=self.preemptible,
                arguments=self.cmd,
                command=["bash", "/olmo-core/entrypoint.sh"],
                replicas=self.num_nodes if self.num_nodes > 1 else None,
                leader_selection=self.num_nodes > 1,
                host_networking=(
                    self.host_networking
                    if self.host_networking is not None
                    else (
                        self.num_nodes > 1
                        or any(["augusta" in cluster for cluster in self.clusters])
                    )
                ),
                propagate_failure=True if self.num_nodes > 1 else None,
                propagate_preemption=True if self.num_nodes > 1 else None,
                synchronized_start_timeout="90m" if self.num_nodes > 1 else None,
                resources=TaskResources(gpu_count=self.num_gpus, shared_memory=self.shared_memory),
                result_path=self.result_dir,
            )
            .with_dataset("/olmo-core", beaker=entrypoint_dataset.id)
            .with_constraint(**constraints_kwargs)
            .with_env_var(GIT_REPO_URL_ENV_VAR, self.git.repo_url)
            .with_env_var(GIT_REF_ENV_VAR, self.git.ref)
        )

        if self.git.branch is not None:
            task_spec = task_spec.with_env_var(GIT_BRANCH_ENV_VAR, self.git.branch)

        for name, val in self._get_env_vars():
            task_spec = task_spec.with_env_var(name=name, value=val)

        for env_secret in self.env_secrets or []:
            task_spec = task_spec.with_env_var(name=env_secret.name, secret=env_secret.secret)

        if self.nfs:
            task_spec = task_spec.with_dataset(
                "/net/nfs.cirrascale", host_path="/net/nfs.cirrascale"
            )
            task_spec = task_spec.with_dataset("/net/nfs", host_path="/net/nfs.cirrascale")

        if self.weka_buckets:
            for bucket in self.weka_buckets:
                task_spec = task_spec.with_dataset(bucket.mount, weka=bucket.bucket)

        return ExperimentSpec(
            description=self.description,
            budget=self.budget,
            tasks=[task_spec],
            retry=None if not self.retries else RetrySpec(allowed_task_retries=self.retries),
        )

    def launch(
        self,
        follow: bool = False,
        torchrun: Optional[bool] = None,
        entrypoint: Optional[str] = None,
        slack_notifications: Optional[bool] = None,
        launch_timeout: Optional[int] = None,
        step_timeout: Optional[int] = None,
        step_soft_timeout: Optional[int] = None,
    ) -> Experiment:
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
        if launch_timeout is None:
            launch_timeout = self.launch_timeout

        # Check for webhook URL env var if needed.
        slack_webhook_url: Optional[str] = None
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
                        slack_webhook_url = self.beaker.secret.read(env_secret.secret)
                        break

            if slack_notifications is None:
                slack_notifications = slack_webhook_url is not None
            elif slack_notifications and slack_webhook_url is None:
                raise OLMoEnvironmentError(
                    f"Missing env var / secret '{SLACK_WEBHOOK_URL_ENV_VAR}' for Slack notifications"
                )

        spec = self.build_experiment_spec(torchrun=torchrun, entrypoint=entrypoint)
        experiment = self.beaker.experiment.create(self.name, spec)
        log.info(f"Experiment submitted, see progress at {self.beaker.experiment.url(experiment)}")

        if not follow:
            return experiment

        try:
            follow_experiment(
                self.beaker,
                experiment,
                slack_webhook_url=slack_webhook_url,
                launch_timeout=launch_timeout,
                step_timeout=step_timeout,
                step_soft_timeout=step_soft_timeout,
            )
        except KeyboardInterrupt:
            log.warning("Caught keyboard interrupt...")
            if Confirm.ask("Would you like to cancel the experiment?"):
                try:
                    self.beaker.experiment.stop(experiment)
                except ExperimentConflict:
                    log.warning(
                        f"Experiment already stopped: {self.beaker.experiment.url(experiment)}"
                    )
                else:
                    log.warning(f"Experiment stopped: {self.beaker.experiment.url(experiment)}")
            else:
                log.info(
                    "You can follow the experiment on the Beaker UI: "
                    f"{self.beaker.experiment.url(experiment)}"
                )

        return experiment


# Regex for detecting training (and eval) steps in logs.
_STEP_REGEX = re.compile(r"\[olmo_core\..+\].+\[.*step\=\d+.*\]")


def follow_experiment(
    beaker: Beaker,
    experiment: Experiment,
    tail: bool = False,
    slack_webhook_url: Optional[str] = None,
    launch_timeout: Optional[int] = None,
    step_timeout: Optional[int] = None,
    step_soft_timeout: Optional[int] = None,
):
    start_time = time.monotonic()

    # Wait for job to be created...
    job: Optional[Job] = beaker.experiment.tasks(experiment.id)[0].latest_job  # type: ignore
    if job is None:
        log.info("Waiting for job to be created...")
        while job is None:
            if launch_timeout is not None and (time.monotonic() - start_time) > launch_timeout:
                try:
                    beaker.experiment.stop(experiment)
                except ExperimentConflict:
                    pass
                raise TimeoutError(
                    f"Job failed to be created within {launch_timeout} seconds. "
                    f"Experiment has been stopped: {beaker.experiment.url(experiment)}"
                )
            else:
                time.sleep(1.0)
                job = beaker.experiment.tasks(experiment.id)[0].latest_job  # type: ignore

    # Pull events until job is running (or fails)...
    events = set()
    while True:
        for event in sorted(
            beaker.job.summarized_events(job), key=lambda event: event.latest_occurrence
        ):
            if event not in events:
                events.add(event)
                log.info(f"❯ {event.latest_message}")

        job = beaker.job.get(job.id)
        if job.is_finalized or job.is_running:
            break
        elif launch_timeout is not None and (time.monotonic() - start_time) > launch_timeout:
            try:
                beaker.experiment.stop(experiment)
            except ExperimentConflict:
                pass
            raise TimeoutError(
                f"Job failed to start within {launch_timeout} seconds. "
                f"Experiment has been stopped: {beaker.experiment.url(experiment)}"
            )
        else:
            time.sleep(1.0)

    if slack_webhook_url is not None:
        _send_slack_notification_for_event(beaker, experiment, "launched", slack_webhook_url)

    queue: Queue = Queue()
    sentinel = object()

    def fill_queue():
        assert job is not None
        try:
            for line_bytes in beaker.job.follow(
                job,
                include_timestamps=False,
                since=None if not tail else timedelta(seconds=10),
            ):
                line = line_bytes.decode(errors="ignore")
                if line.endswith("\n"):
                    line = line[:-1]
                queue.put(line)
        except Exception as e:
            queue.put(e)
        finally:
            queue.put(sentinel)

    thread = Thread(target=fill_queue, daemon=True)
    thread.start()

    # Stream logs...
    log.info("Showing logs:")
    print()
    first_step_detected = False
    start_time = time.monotonic()
    last_step_time = 0.0
    last_inactivity_warning = 0.0
    last_status_check = start_time
    while True:
        try:
            result = queue.get(timeout=1.0)
            if result is sentinel:
                break
            elif isinstance(result, Exception):
                raise result
            else:
                assert isinstance(result, str)
                if (
                    step_timeout is not None or step_soft_timeout is not None
                ) and _STEP_REGEX.search(result) is not None:
                    first_step_detected = True
                    last_step_time = time.monotonic()
                print(result)
        except Empty:
            cur_time = time.monotonic()

            # If (a) we've detected training steps already or (b) the run has been up for over 30 min,
            # then we warn if we haven't detected new steps within the past `step_soft_timeout` seconds.
            # But we only send a warning at most once per hour.
            if (
                slack_webhook_url is not None
                and step_soft_timeout is not None
                and (first_step_detected or (cur_time - start_time) > max(step_soft_timeout, 1800))
                and (cur_time - last_step_time) > step_soft_timeout
                and (cur_time - last_inactivity_warning) > 3600
            ):
                _send_slack_notification_for_event(
                    beaker, experiment, "inactive", slack_webhook_url
                )
                last_inactivity_warning = cur_time

            # If (a) we've detected training steps already or (b) the run has been up for over 60 min,
            # then we kill the job if we haven't detected new steps within the past `step_timeout` seconds.
            if (
                step_timeout is not None
                and (first_step_detected or (cur_time - start_time) > max(step_timeout, 3600))
                and (cur_time - last_step_time) > step_timeout
            ):
                try:
                    beaker.experiment.stop(experiment)
                except ExperimentConflict:
                    pass
                raise TimeoutError(
                    f"No training steps detected within {step_timeout} seconds. "
                    f"Experiment has been stopped: {beaker.experiment.url(experiment)}"
                )

            # Periodically check if the job is finalized in case the log streaming thread gets stuck.
            if (cur_time - last_status_check) > 5 * 60:
                job = beaker.job.get(job.id)
                last_status_check = cur_time
                if job.status.finalized is not None:
                    break

    print()
    log.info("End logs")

    # Refresh the job.
    job = beaker.job.get(job.id)
    exit_code = job.status.exit_code

    if exit_code is None:
        if slack_webhook_url is not None:
            _send_slack_notification_for_event(beaker, experiment, "failed", slack_webhook_url)
        raise BeakerExperimentFailedError(
            f"Experiment failed, see {beaker.experiment.url(experiment)} for details"
        )
    elif exit_code > 0:
        if slack_webhook_url is not None:
            _send_slack_notification_for_event(beaker, experiment, "failed", slack_webhook_url)
        raise BeakerExperimentFailedError(
            f"Experiment exited with non-zero code ({exit_code}), "
            f"see {beaker.experiment.url(experiment)} for details"
        )
    else:
        log.info(f"Experiment completed successfully: {beaker.experiment.url(experiment)}")
        if slack_webhook_url is not None:
            _send_slack_notification_for_event(beaker, experiment, "succeeded", slack_webhook_url)


def _send_slack_notification_for_event(
    beaker: Beaker,
    experiment: Experiment,
    event: Literal["launched", "succeeded", "failed", "inactive"],
    webhook_url: str,
):
    workload_name = experiment.full_name
    workload_url = beaker.experiment.url(experiment)

    if event == "launched":
        text = f":check: Run <{workload_url}|*{workload_name}*> has launched! :runner:"
    elif event == "failed":
        text = f":check-failed: Run <{workload_url}|*{workload_name}*> failed!"
    elif event == "succeeded":
        text = f":check: Run <{workload_url}|*{workload_name}*> succeeded!"
    elif event == "inactive":
        text = f":warning: Run <{workload_url}|*{workload_name}*> appears to be stuck!"
    else:
        raise ValueError(f"Unknown event: {event}")

    try:
        requests.post(webhook_url, json={"text": text})
    except Exception as e:
        log.exception(f"Failed to send Slack notification: {e}")


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
        default=["ai2/jupiter", "ai2/ceres", "ai2/saturn", "ai2/prometheus"],
        help="""Clusters to launch on (multiple allowed).""",
    )
    parser.add_argument(
        "--priority",
        choices=[p.value for p in Priority],
        default=Priority.normal,
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
        help="""If the command should be run via torchrun. This will default to true when '--gpus' is greater than 1.""",
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


def _build_config(opts: argparse.Namespace, command: List[str]) -> BeakerLaunchConfig:
    env_vars: List[BeakerEnvVar] = []
    if opts.debug:
        env_vars.append(BeakerEnvVar(name="CUDA_LAUNCH_BLOCKING", value="1"))
        env_vars.append(BeakerEnvVar(name="NCCL_DEBUG", value="INFO"))
    for e in opts.env or []:
        if "=" not in e:
            raise ValueError(f"Invalid env var '{e}', must be in the form NAME=VALUE")
        name, value = e.split("=", 1)
        env_vars.append(BeakerEnvVar(name=name, value=value))
    env_secrets: List[BeakerEnvSecret] = []
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
        num_nodes=opts.nodes,
        num_gpus=opts.gpus,
        preemptible=opts.preemptible,
        priority=opts.priority,
        beaker_image=opts.beaker_image,
        workspace=opts.workspace,
        allow_dirty=opts.allow_dirty,
        shared_filesystem=opts.shared_filesystem,
        weka_buckets=[
            BeakerWekaBucket(bucket=bucket, mount=f"/weka/{bucket}") for bucket in (opts.weka or [])
        ],
    )


def main():
    opts, command = _parse_args()
    prepare_cli_environment()
    config = _build_config(opts, command)
    if opts.dry_run:
        rich.print(config)
    else:
        config.launch(torchrun=opts.torchrun, follow=True)


if __name__ == "__main__":
    main()
