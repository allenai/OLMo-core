"""
Launch experiments on `Beaker <https://beaker.org>`_.
"""

import hashlib
import logging
import tempfile
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Set, Tuple

from beaker import (
    Beaker,
    Dataset,
    DatasetConflict,
    DatasetNotFound,
    Experiment,
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
from ..exceptions import BeakerExperimentFailedError, OLMoConfigurationError
from ..train.callbacks.beaker import BEAKER_RESULT_DIR
from ..utils import LOG_FILTER_TYPE_ENV_VAR, LogFilterType
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

_DEFAULT_TORCH = "2.7.0".replace(".", "")
_DEFAULT_CUDA = "12.8".replace(".", "")


class OLMoCoreBeakerImage(StrEnum):
    """
    Official Beaker images that work well for OLMo-core.

    You can find the full list at
    `beaker.org/ws/ai2/OLMo-core/images <https://beaker.org/ws/ai2/OLMo-core/images>`_, which
    includes *versioned* images that are published with each release of the OLMo-core package.
    """

    # NOTE: when updating default images here, should also update images used in tests at .github/workflows/main.yml

    stable = f"olmo-core-tch{_DEFAULT_TORCH}cu{_DEFAULT_CUDA}-2025-05-16"
    """
    Built with the latest compatible stable version of PyTorch.
    """

    stable_cu126 = f"olmo-core-tch{_DEFAULT_TORCH}cu126-2025-05-16"
    """
    The stable image with CUDA pinned to 12.6.
    """

    stable_cu128 = f"olmo-core-tch{_DEFAULT_TORCH}cu128-2025-05-16"
    """
    The stable image with CUDA pinned to 12.8.
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
    The command to run in the container via ``torchrun``.
    """

    budget: str
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

    clusters: List[str] = field(default_factory=lambda: ["ai2/jupiter-cirrascale-2"])
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

    This is NOT recommended to be used lower priority preemptible jobs, since hostname constraints are not
    updated on preemption.
    """

    num_execution_units: Optional[int] = None
    """
    Number of \"execution units\", defaults to ``max(1, num_nodes // 32)``. An \"execution unit\" is abstraction
    for any node-using entity of which 1 or more copies are run, where each unit wants its nodes to be
    from colocated hardware (e.g., a model replica for large jobs, or a full distributed model for small jobs).
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
            self._beaker = Beaker.from_env(default_workspace=self.workspace)
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
        self, torchrun: bool = True, entrypoint: Optional[str] = None
    ) -> ExperimentSpec:
        """
        Get the Beaker experiment spec corresponding to this config instance.
        """
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
            "set -exuo pipefail",
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
            entrypoint_script.append(" ".join(self._get_torchrun_cmd()) + ' "$@"')
        else:
            entrypoint = entrypoint or "python"
            entrypoint_script.append(f'{entrypoint} "$@"')

        entrypoint_dataset = self._create_script_dataset("entrypoint.sh", entrypoint_script)

        if (
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
        self, follow: bool = False, torchrun: bool = True, entrypoint: Optional[str] = None
    ) -> Experiment:
        """
        Launch a Beaker experiment using this config.

        .. tip::
            You can preview what the Beaker experiment spec would like using
            :meth:`build_experiment_spec()`.

        :param follow: Stream the logs and follow the experiment until completion.
        :param torchrun: Launch the target command with ``torchrun``.
        :param entrypoint: Provide an optional entrypoint program if ``torchrun`` is ``False``.
            Defaults to 'python'.

        :returns: The Beaker experiment.
        """
        spec = self.build_experiment_spec(torchrun=torchrun, entrypoint=entrypoint)
        experiment = self.beaker.experiment.create(self.name, spec)
        log.info(f"Experiment submitted, see progress at {self.beaker.experiment.url(experiment)}")

        if not follow:
            return experiment

        try:
            follow_experiment(self.beaker, experiment)
        except KeyboardInterrupt:
            log.warning("Caught keyboard interrupt...")
            if Confirm.ask("Would you like to cancel the experiment?"):
                self.beaker.experiment.stop(experiment)
                log.warning(f"Experiment stopped: {self.beaker.experiment.url(experiment)}")
            else:
                log.info(
                    "You can follow the experiment on the Beaker UI: "
                    f"{self.beaker.experiment.url(experiment)}"
                )

        return experiment


def follow_experiment(beaker: Beaker, experiment: Experiment, tail: bool = False):
    # Wait for job to start...
    job: Optional[Job] = beaker.experiment.tasks(experiment.id)[0].latest_job  # type: ignore
    if job is None:
        log.info("Waiting for job to be created...")
        while job is None:
            time.sleep(1.0)
            job = beaker.experiment.tasks(experiment.id)[0].latest_job  # type: ignore

    # Pull events until job is running (or fails)...
    events = set()
    while not (job.is_finalized or job.is_running):
        job = beaker.job.get(job.id)
        for event in sorted(
            beaker.job.summarized_events(job), key=lambda event: event.latest_occurrence
        ):
            if event not in events:
                events.add(event)
                log.info(f"❯ {event.latest_message}")
                if event.status.lower() == "started":
                    break
        else:
            time.sleep(1.0)
            continue
        break

    # Stream logs...
    log.info("Showing logs:")
    print()
    for line_bytes in beaker.job.follow(
        job,
        include_timestamps=False,
        since=None if not tail else timedelta(seconds=10),
    ):
        line = line_bytes.decode(errors="ignore")
        if line.endswith("\n"):
            line = line[:-1]
        print(line)
    print()
    log.info("End logs")

    # Refresh the job.
    job = beaker.job.get(job.id)
    exit_code = job.status.exit_code

    if exit_code is None:
        raise BeakerExperimentFailedError(
            f"Experiment failed, see {beaker.experiment.url(experiment)} for details"
        )
    elif exit_code > 0:
        raise BeakerExperimentFailedError(
            f"Experiment exited with non-zero code ({exit_code}), "
            f"see {beaker.experiment.url(experiment)} for details"
        )
    else:
        log.info(f"Experiment completed successfully: {beaker.experiment.url(experiment)}")
