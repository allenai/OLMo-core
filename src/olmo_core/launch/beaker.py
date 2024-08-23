import hashlib
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

from beaker import (
    Beaker,
    Dataset,
    DatasetConflict,
    DatasetNotFound,
    Experiment,
    ExperimentSpec,
    Job,
    Priority,
    TaskResources,
    TaskSpec,
)

from ..config import Config
from ..distributed.utils import OLMO_SHARED_FS_ENV_VAR
from ..exceptions import BeakerExperimentFailedError, OLMoConfigurationError
from ..utils import LOG_FILTER_TYPE_ENV_VAR, LogFilterType
from ..version import VERSION
from .utils import ensure_repo

log = logging.getLogger(__name__)


__all__ = ["BeakerLaunchConfig", "BeakerPriority"]


BeakerPriority = Priority


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

    setup_steps: List[str] = field(
        default_factory=lambda: [
            "conda shell.bash activate base",
            "pip install -e '.[all]'",
            "pip freeze",
        ]
    )
    """
    A list of shell commands to run for installing dependencies, running arbitrary scripts,
    and other setup steps.
    """

    beaker_image: str = "ai2/pytorch2.4.0-cuda12.1-python3.11"
    """
    The Beaker image to use.
    """

    num_nodes: int = 1
    """
    The number of nodes to use.
    """

    num_gpus: int = 8
    """
    The number of GPUs to use per node.
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

    env_vars: List[Tuple[str, str]] = field(default_factory=list)
    """
    Additional env vars to include.
    """

    env_secrets: List[Tuple[str, str]] = field(default_factory=list)
    """
    Environment variables to add from secrets.
    """

    nfs: bool = False
    """
    Attach the NFS drive.
    """

    weka_buckets: List[Tuple[str, str]] = field(default_factory=list)
    """
    Weka buckets to attach and where to attach them,
    e.g. ``("oe-training-default", "/weka/oe-training-default")``.
    """

    allow_dirty: bool = False
    """
    Allow running with uncommitted changed.
    """

    _beaker: Optional[Beaker] = None

    @property
    def default_env_vars(self) -> List[Tuple[str, str]]:
        """
        Default env vars to add to the experiment.
        """
        env_vars = [
            ("NCCL_DEBUG", "INFO"),
            (LOG_FILTER_TYPE_ENV_VAR, LogFilterType.local_rank0_only),
            ("OMP_NUM_THREADS", "8"),
        ]
        if self.shared_filesystem:
            env_vars.append((OLMO_SHARED_FS_ENV_VAR, "1"))
        if self.num_nodes > 1:
            env_vars.extend(
                [
                    ("NCCL_IB_HCA", "^=mlx5_bond_0"),
                    ("NCCL_SOCKET_IFNAME", "ib"),
                ]
            )
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
        for name, val in self.env_vars:
            env_vars.append((name, val))
            env_var_names.add(name)
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

        return torchrun + self.cmd

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

    def build_experiment_spec(self) -> ExperimentSpec:
        """
        Get the Beaker experiment spec corresponding to this config instance.
        """
        # Get repository account, name, and current ref.
        github_account, github_repo, git_ref, is_public = ensure_repo(self.allow_dirty)

        if not is_public:
            raise OLMoConfigurationError(
                "Only public repositories are supported at the moment. "
                "Please use beaker-gantry to launch jobs with private repos."
            )

        entrypoint_script = [
            "#!/usr/bin/env bash",
            "set -exuo pipefail",
            "mkdir -p /olmo-core-runtime",
            "cd /olmo-core-runtime",
            f"git clone https://github.com/{github_account}/{github_repo} .",
            f"git checkout {git_ref}",
            "git submodule update --init --recursive",
            *self.setup_steps,
            'exec "$@" 2>&1',
        ]

        entrypoint_dataset = self._create_script_dataset("entrypoint.sh", entrypoint_script)

        task_spec = (
            TaskSpec.new(
                self.task_name,
                beaker_image=self.beaker_image,
                priority=self.priority,
                preemptible=self.preemptible,
                arguments=self._get_torchrun_cmd(),
                command=["bash", "/olmo-core/entrypoint.sh"],
                replicas=self.num_nodes if self.num_nodes > 1 else None,
                leader_selection=self.num_nodes > 1,
                host_networking=self.num_nodes > 1,
                propagate_failure=True if self.num_nodes > 1 else None,
                propagate_preemption=True if self.num_nodes > 1 else None,
                synchronized_start_timeout="90m" if self.num_nodes > 1 else None,
                resources=TaskResources(gpu_count=self.num_gpus, shared_memory="10GiB"),
            )
            .with_dataset("/olmo-core", beaker=entrypoint_dataset.id)
            .with_constraint(cluster=self.clusters)
        )

        for name, val in self._get_env_vars():
            task_spec = task_spec.with_env_var(name=name, value=val)

        for name, secret in self.env_secrets or []:
            task_spec = task_spec.with_env_var(name=name, secret=secret)

        if self.nfs:
            task_spec = task_spec.with_dataset(
                "/net/nfs.cirrascale", host_path="/net/nfs.cirrascale"
            )
            task_spec = task_spec.with_dataset("/net/nfs", host_path="/net/nfs.cirrascale")

        if self.weka_buckets:
            for source, target in self.weka_buckets:
                task_spec = task_spec.with_dataset(target, weka=source)

        return ExperimentSpec(description=self.description, budget=self.budget, tasks=[task_spec])

    def _follow_experiment(self, experiment: Experiment):
        print("-------------------- Logs ----------------------")

        # Wait for job to start...
        job: Optional[Job] = self.beaker.experiment.tasks(experiment.id)[0].latest_job  # type: ignore
        if job is None:
            print("Waiting for job to launch..", end="")
            while job is None:
                time.sleep(1.0)
                print(".", end="")
                job = beaker.experiment.tasks(experiment.id)[0].latest_job  # type: ignore

        exit_code: Optional[int] = job.status.exit_code
        stream_logs = exit_code is None and not job.is_finalized
        if stream_logs:
            print()
            for line_bytes in self.beaker.job.follow(
                job,
                include_timestamps=False,
            ):
                line = line_bytes.decode(errors="ignore")
                if line.endswith("\n"):
                    line = line[:-1]
                print(line)
            print("-------------------- End logs ----------------------")
            print()

            # Refresh the job.
            job = self.beaker.job.get(job.id)
            exit_code = job.status.exit_code

        if exit_code is None:
            raise BeakerExperimentFailedError(
                f"Experiment failed, see {self.beaker.experiment.url(experiment)} for details"
            )
        elif exit_code > 0:
            raise BeakerExperimentFailedError(
                f"Experiment exited with non-zero code ({exit_code}), "
                f"see {self.beaker.experiment.url(experiment)} for details"
            )
        else:
            print("Experiment completed successfully")

    def launch(self, follow: bool = False) -> Experiment:
        """
        Launch a Beaker experiment using this config.

        .. tip::
            You can preview what the Beaker experiment spec would like using
            :meth:`build_experiment_spec()`.

        :param follow: Stream the logs and follow the experiment until completion.

        :returns: The Beaker experiment.
        """
        spec = self.build_experiment_spec()
        experiment = self.beaker.experiment.create(self.name, spec)
        log.info(f"Experiment submitted, see progress at {self.beaker.experiment.url(experiment)}")

        if not follow:
            return experiment

        try:
            self._follow_experiment(experiment)
        except KeyboardInterrupt:
            print(
                f"You can cancel the experiment on the Beaker UI: {self.beaker.experiment.url(experiment)}"
            )
            raise

        return experiment
