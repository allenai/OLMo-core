"""
Launch experiments on `Beaker <https://beaker.org>`_.
"""

import binascii
import hashlib
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

from beaker import (
    Beaker,
    BeakerDataset,
    BeakerDatasetFileAlgorithmType,
    BeakerExperimentSpec,
    BeakerImage,
    BeakerJob,
    BeakerRetrySpec,
    BeakerSortOrder,
    BeakerTaskResources,
    BeakerTaskSpec,
    BeakerWorkload,
    BeakerWorkloadStatus,
)
from beaker.exceptions import BeakerDatasetConflict, BeakerImageNotFound
from rich.prompt import Confirm

from ..config import Config, StrEnum
from ..distributed.utils import OLMO_SHARED_FS_ENV_VAR
from ..exceptions import BeakerExperimentFailedError, OLMoConfigurationError
from ..utils import LOG_FILTER_TYPE_ENV_VAR, LogFilterType
from ..version import VERSION
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


class BeakerPriority(StrEnum):
    low = "low"
    normal = "normal"
    high = "high"
    urgent = "urgent"


_DEFAULT_TORCH = "2.7.0".replace(".", "")
_DEFAULT_CUDA = "12.6".replace(".", "")
_DEFAULT_BUILD_ACCOUNT = "petew"


class OLMoCoreBeakerImage(StrEnum):
    """
    Official Beaker images that work well for OLMo-core.

    You can find the full list at
    `beaker.org/ws/ai2/OLMo-core/images <https://beaker.org/ws/ai2/OLMo-core/images>`_, which
    includes *versioned* images that are published with each release of the OLMo-core package.
    """

    stable = f"olmo-core-tch{_DEFAULT_TORCH}cu{_DEFAULT_CUDA}"
    """
    Built with the latest compatible stable version of PyTorch.
    """

    stable_cu126 = f"olmo-core-tch{_DEFAULT_TORCH}cu126"
    """
    The stable image with CUDA pinned to 12.6.
    """

    stable_cu128 = f"olmo-core-tch{_DEFAULT_TORCH}cu128"
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

    priority: BeakerPriority = BeakerPriority.normal
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

    def _create_script_dataset(
        self, beaker: Beaker, script_name: str, script: List[str]
    ) -> BeakerDataset:
        workspace = beaker.workspace.get(self.workspace)

        # Hash contents.
        sha256_hash = hashlib.sha256()
        for line in script:
            sha256_hash.update((line + "\n").encode())

        # Create unique name for dataset.
        dataset_name = f"olmo-core-v{VERSION}-{workspace.id}-{sha256_hash.hexdigest()[:6]}"

        def get_dataset() -> Optional[BeakerDataset]:
            matching_datasets = list(
                beaker.dataset.list(
                    workspace=workspace, name_or_description=dataset_name, results=False
                )
            )
            if matching_datasets:
                return matching_datasets[0]
            else:
                return None

        dataset = get_dataset()
        if dataset is None:
            # Create it.
            log.info(f"Creating script dataset '{dataset_name}'...")
            try:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    tmpdir = Path(tmpdirname)
                    script_path = tmpdir / script_name
                    with open(script_path, "w") as script_file:
                        for line in script:
                            script_file.write(line + "\n")
                    dataset = beaker.dataset.create(dataset_name, script_path)
            except BeakerDatasetConflict:  # could be in a race with another process.
                time.sleep(1.0)
                dataset = get_dataset()

        if dataset is None:
            raise RuntimeError(f"Failed to resolve entrypoint dataset '{dataset_name}'")

        # Verify contents.
        ds_files = list(beaker.dataset.list_files(dataset))
        for retry in range(1, 4):
            ds_files = list(beaker.dataset.list_files(dataset))
            if len(ds_files) >= 1:
                break
            else:
                time.sleep(1.5**retry)

        if len(ds_files) != 1:
            raise RuntimeError(
                f"Entrypoint dataset {beaker.dataset.url(dataset)} is missing the "
                f"required entrypoint file. Please run again."
            )

        if ds_files[0].HasField("digest"):
            digest = ds_files[0].digest
            expected_value = binascii.hexlify(digest.value).decode()
            hasher = BeakerDatasetFileAlgorithmType(digest.algorithm).hasher()
            for line in script:
                hasher.update((line + "\n").encode())
            actual_value = binascii.hexlify(hasher.digest()).decode()
            if actual_value != expected_value:
                raise RuntimeError(
                    f"Checksum failed for entrypoint dataset {beaker.dataset.url(dataset)}\n"
                    f"This could be a bug, or it could mean someone has tampered with the dataset.\n"
                    f"If you're sure no one has tampered with it, you can delete the dataset from "
                    f"the Beaker dashboard and try again.\n"
                    f"Expected digest:\n{digest}"
                )

        return dataset

    def build_experiment_spec(
        self, beaker: Beaker, torchrun: bool = True, entrypoint: Optional[str] = None
    ) -> BeakerExperimentSpec:
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

        entrypoint_dataset = self._create_script_dataset(beaker, "entrypoint.sh", entrypoint_script)

        task_spec = (
            BeakerTaskSpec.new(
                self.task_name,
                beaker_image=resolve_beaker_image(beaker, self.beaker_image).id,
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
                propagate_failure=False if self.num_nodes > 1 else None,
                propagate_preemption=True if self.num_nodes > 1 else None,
                synchronized_start_timeout="90m" if self.num_nodes > 1 else None,
                resources=BeakerTaskResources(
                    gpu_count=self.num_gpus, shared_memory=self.shared_memory
                ),
            )
            .with_dataset("/olmo-core", beaker=entrypoint_dataset.id)
            .with_constraint(cluster=self.clusters)
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

        return BeakerExperimentSpec(
            description=self.description,
            budget=self.budget,
            tasks=[task_spec],
            retry=None if not self.retries else BeakerRetrySpec(allowed_task_retries=self.retries),
        )

    def launch(
        self, follow: bool = False, torchrun: bool = True, entrypoint: Optional[str] = None
    ) -> BeakerWorkload:
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
        with Beaker.from_env(default_workspace=self.workspace) as beaker:
            spec = self.build_experiment_spec(beaker, torchrun=torchrun, entrypoint=entrypoint)
            workload = beaker.experiment.create(name=self.name, spec=spec)
            log.info(f"Experiment submitted, see progress at {beaker.workload.url(workload)}")

            if not follow:
                return workload

            try:
                follow_experiment(beaker, workload)
            except KeyboardInterrupt:
                log.warning("Caught keyboard interrupt...")
                if Confirm.ask("Would you like to cancel the experiment?"):
                    beaker.workload.cancel(workload)
                    log.warning(f"Experiment stopped: {beaker.workload.url(workload)}")
                else:
                    log.info(
                        "You can follow the experiment on the Beaker UI: "
                        f"{beaker.workload.url(workload)}"
                    )

            return workload


def follow_experiment(beaker: Beaker, workload: BeakerWorkload, tail_lines: Optional[int] = None):
    # Wait for job to start...
    while (job := beaker.workload.get_latest_job(workload)) is None:
        log.info("Waiting for job to be created...")
        time.sleep(1.0)

    def refresh_job() -> BeakerJob:
        assert job is not None
        return beaker.job.get(job.id)

    # Pull events until job is running (or fails)...
    events = set()
    while not (job.status.HasField("finalized") or job.status.HasField("started")):
        job = refresh_job()
        for event in beaker.job.list_summarized_events(
            job, sort_order=BeakerSortOrder.descending, sort_field="latest_occurrence"
        ):
            event_hashable = (event.latest_occurrence.ToSeconds(), event.latest_message)
            if event_hashable not in events:
                events.add(event_hashable)
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
    time.sleep(2.0)  # wait a moment to make sure logs are available before experiment finishes
    for job_log in beaker.job.logs(job, follow=True, tail_lines=tail_lines):
        print(job_log.message.decode())
    print()
    log.info("End logs")

    # Waiting for job to finalize...
    job = refresh_job()
    if not job.status.HasField("finalized"):
        log.info("Waiting for job to finalize...")
        while not (job := refresh_job()).status.HasField("finalized"):
            time.sleep(1.0)

    status = job.status.status
    if status == BeakerWorkloadStatus.succeeded:
        log.info(f"Job completed successfully: {beaker.workload.url(workload)}")
    elif status == BeakerWorkloadStatus.canceled:
        raise BeakerExperimentFailedError(
            f"Job was canceled, see {beaker.workload.url(workload)} for details"
        )
    elif status == BeakerWorkloadStatus.failed:
        raise BeakerExperimentFailedError(
            f"Job failed with exit code {job.status.exit_code}, see {beaker.workload.url(workload)} for details"
        )
    else:
        raise ValueError(f"unexpected job status '{status}'")


def resolve_beaker_image(beaker: Beaker, image: str) -> BeakerImage:
    try:
        return beaker.image.get(image)
    except BeakerImageNotFound:
        pass

    # If image name is already a full name then it probably doesn't exist.
    if "/" in image:
        raise BeakerImageNotFound(image)

    # Try pre-pending 'petew', since that's the account that we usually build the images from.
    try:
        return beaker.image.get(f"petew/{image}")
    except BeakerImageNotFound:
        pass

    matches = [im for im in beaker.image.list(name_or_description=image) if im.name == image]
    if not matches:
        raise BeakerImageNotFound(image)
    elif len(matches) == 1:
        return matches[0]

    current_user = beaker.user.get()
    author_ids = [current_user.id]
    if current_user.name != _DEFAULT_BUILD_ACCOUNT:
        author_ids.append(beaker.user.get(_DEFAULT_BUILD_ACCOUNT).id)

    for author_id in author_ids:
        matches_for_author = [im for im in matches if im.author_id == author_id]
        if matches_for_author:
            return matches_for_author[0]

    raise BeakerImageNotFound(image)
