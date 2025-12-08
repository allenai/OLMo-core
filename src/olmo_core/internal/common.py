import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

import torch
from beaker import Beaker, BeakerError, SecretNotFound

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import is_url
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerEnvVar,
    BeakerLaunchConfig,
    BeakerWekaBucket,
    OLMoCoreBeakerImage,
    is_running_in_beaker_batch_job,
)
from olmo_core.train.callbacks.beaker import BEAKER_RESULT_DIR
from olmo_core.utils import generate_uuid

log = logging.getLogger(__name__)

GOOGLE_CLUSTERS = [
    "ai2/augusta",
]


@lru_cache()
def get_beaker_client() -> Optional[Beaker]:
    try:
        return Beaker.from_env(check_for_upgrades=False)
    except BeakerError:
        return None


@lru_cache()
def get_beaker_username() -> Optional[str]:
    beaker = get_beaker_client()
    if beaker is not None:
        return beaker.account.whoami().name
    else:
        return None


def beaker_secret_exists(secret: str, workspace: Optional[str] = None) -> bool:
    beaker = get_beaker_client()
    if beaker is None:
        raise RuntimeError(
            "Environment not configured correctly for Beaker, you may be missing the BEAKER_TOKEN env var."
        )

    try:
        beaker.secret.get(secret, workspace=workspace)
        return True
    except SecretNotFound:
        return False


def _to_beaker_env_secret(
    name: str, secret: str, *, workspace: Optional[str] = None, required: bool = True
) -> Optional[BeakerEnvSecret]:
    # Assume beaker secret exists if we are running in a batch job (e.g., during a training job)
    # so that we don't DOS beaker.
    if is_running_in_beaker_batch_job() or beaker_secret_exists(secret, workspace=workspace):
        return BeakerEnvSecret(name=name, secret=secret)
    elif required:
        raise OLMoConfigurationError(
            f"Secret {secret} not configured in beaker workspace {workspace}"
        )
    else:
        log.info(f"Secret {secret} not configured in beaker workspace {workspace}")
        return None


def get_root_dir(cluster: str) -> str:
    if cluster in [
        "ai2/test-h100",
        "ai2/jupiter",
        "ai2/saturn",
        "ai2/ceres",
        "ai2/neptune",
        "ai2/titan",
        "ai2/rhea",
        "ai2/phobos",
    ]:
        return "/weka/oe-training-default/ai2-llm"
    elif cluster in GOOGLE_CLUSTERS:
        return "gs://ai2-llm"
    elif "local" in cluster:
        return "gs://ai2-llm"
    raise OLMoConfigurationError(f"Unknown cluster: {cluster}")


def get_work_dir(root_dir: str) -> str:
    if is_url(root_dir):
        return "./dataset-cache"
    elif (beaker_username := get_beaker_username()) is not None:
        return f"{root_dir}/checkpoints/{beaker_username.lower()}/dataset-cache"
    else:
        return f"{root_dir}/checkpoints/dataset-cache"


def build_launch_config(
    *,
    name: str,
    cmd: List[str],
    cluster: str,
    root_dir: Optional[str] = None,
    task_name: str = "train",
    workspace: str = "ai2/OLMo-core",
    budget: str = "ai2/oe-base",
    nccl_debug: Union[bool, str] = False,
    flight_recorder: bool = False,
    beaker_image: str = OLMoCoreBeakerImage.stable,
    num_nodes: int = 1,
    use_hostname_constraints: bool = False,
    num_execution_units: Optional[int] = None,
) -> BeakerLaunchConfig:
    weka_buckets: List[BeakerWekaBucket] = []

    default_root_dir = get_root_dir(cluster)
    if root_dir is None:
        root_dir = default_root_dir
    elif root_dir != default_root_dir:
        log.warning(
            f"Overriding default root_dir for {cluster=} to {root_dir} ({default_root_dir=})."
        )

    if root_dir.startswith("/weka/"):
        weka_buckets.append(BeakerWekaBucket("oe-training-default", "/weka/oe-training-default"))

    beaker_user = get_beaker_username()
    if beaker_user is None:
        raise RuntimeError(
            "Environment not configured correctly for Beaker, you may be missing the BEAKER_TOKEN env var."
        )

    google_creds = (
        _to_beaker_env_secret(
            name="GOOGLE_CREDENTIALS",
            secret="GOOGLE_CREDENTIALS",
            required=False,
            workspace=workspace,
        )
        if cluster not in GOOGLE_CLUSTERS
        else None
    )
    env_secrets = [
        _to_beaker_env_secret(
            name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN", workspace=workspace
        ),
        _to_beaker_env_secret(
            name="WANDB_API_KEY",
            secret=f"{beaker_user}_WANDB_API_KEY",
            required=True,
            workspace=workspace,
        ),
        _to_beaker_env_secret(
            name="COMET_API_KEY",
            secret=f"{beaker_user}_COMET_API_KEY",
            required=False,
            workspace=workspace,
        ),
        _to_beaker_env_secret(
            name="AWS_CONFIG", secret=f"{beaker_user}_AWS_CONFIG", workspace=workspace
        ),
        _to_beaker_env_secret(
            name="AWS_CREDENTIALS", secret=f"{beaker_user}_AWS_CREDENTIALS", workspace=workspace
        ),
        _to_beaker_env_secret(
            name="R2_ENDPOINT_URL", secret="R2_ENDPOINT_URL", required=False, workspace=workspace
        ),
        _to_beaker_env_secret(
            name="WEKA_ENDPOINT_URL",
            secret="WEKA_ENDPOINT_URL",
            required=False,
            workspace=workspace,
        ),
        _to_beaker_env_secret(
            name="SLACK_WEBHOOK_URL",
            secret="SLACK_WEBHOOK_URL",
            required=False,
            workspace=workspace,
        ),
        google_creds,
    ]

    env_vars: List[BeakerEnvVar] = []
    if isinstance(nccl_debug, str):
        env_vars.append(BeakerEnvVar(name="NCCL_DEBUG", value=nccl_debug))
    else:
        env_vars.append(BeakerEnvVar(name="NCCL_DEBUG", value="INFO" if nccl_debug else "WARN"))
    if flight_recorder:
        # https://github.com/pytorch/tutorials/blob/main/unstable_source/flight_recorder_tutorial.rst
        fr_dump_location = Path(BEAKER_RESULT_DIR) / "flightrecorder" / "nccl_trace_rank_"
        env_vars += [
            BeakerEnvVar(name="TORCH_NCCL_TRACE_BUFFER_SIZE", value="2000"),
            BeakerEnvVar(name="TORCH_NCCL_DUMP_ON_TIMEOUT", value="1"),
            BeakerEnvVar(name="TORCH_FR_DUMP_TEMP_FILE", value=str(fr_dump_location)),
        ]

    launch_config = BeakerLaunchConfig(
        name=f"{name}-{generate_uuid()[:8]}",
        budget=budget,
        cmd=cmd,
        task_name=task_name,
        workspace=workspace,
        clusters=[cluster],
        weka_buckets=weka_buckets,
        beaker_image=beaker_image,
        num_nodes=num_nodes,
        num_gpus=8,
        use_hostname_constraints=use_hostname_constraints,
        num_execution_units=num_execution_units,
        shared_filesystem=not is_url(root_dir),
        allow_dirty=False,
        env_vars=env_vars,
        env_secrets=[env_secret for env_secret in env_secrets if env_secret is not None],
        setup_steps=[
            # Clone repo.
            'git clone "$REPO_URL" .',
            'git checkout "$GIT_REF"',
            "git submodule update --init --recursive",
            # Setup python environment.
            "conda shell.bash activate base",
            #  "pip install 'ai2-olmo-eval @ git+https://git@github.com/allenai/OLMo-in-loop-evals.git@epwalsh/debug'",
            "pip install -e '.[all]'",
            #  "pip install --upgrade beaker-py",
            # Quickly try a new version of PyTorch like this
            #  "pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128",
            "pip freeze",
            # Move AWS credentials from env to relevant files
            "mkdir -p ~/.aws",
            "printenv AWS_CONFIG > ~/.aws/config",
            "printenv AWS_CREDENTIALS > ~/.aws/credentials",
        ],
    )

    if cluster == "ai2/augusta":
        # Print out host metadata for easy debugging.
        launch_config.setup_steps.insert(
            0,
            """ID=$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/id); """
            """TOPOLOGY=$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/attributes/physical_host_topology); """
            """printf 'Google Instance Metadata: {"id":"%s","physical_host_topology":%s}' "$ID" "$TOPOLOGY" | tr -d '[:space:]'; echo""",
        )

    if google_creds:
        launch_config.setup_steps += [
            "mkdir -p ~/.google",
            f"printenv {google_creds.name} > ~/.google/credentials.json",
            "export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.google/credentials.json",
        ]

    return launch_config


CLUSTER_TO_GPU_TYPE = {
    "ai2/jupiter": "NVIDIA H100 80GB HBM3",
    "ai2/augusta": "NVIDIA H100 80GB HBM3",
    "ai2/ceres": "NVIDIA H100 80GB HBM3",
    "ai2/titan": "NVIDIA B200",
}


def get_gpu_type(cluster: str) -> str:
    if cluster in CLUSTER_TO_GPU_TYPE:
        return CLUSTER_TO_GPU_TYPE[cluster]
    elif cluster == "local":
        return torch.get_default_device().type
    else:
        log.warning(f"Missing cluster '{cluster}' in CLUSTER_TO_GPU_TYPE mapping")
        beaker = get_beaker_client()
        assert beaker is not None
        nodes = beaker.cluster.nodes(cluster)
        assert nodes and nodes[0].limits.gpu_type
        return nodes[0].limits.gpu_type
