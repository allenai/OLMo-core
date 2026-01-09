import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

import torch
from beaker import Beaker, BeakerGpuType
from beaker.exceptions import BeakerError, BeakerSecretNotFound

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import is_url
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerEnvVar,
    BeakerLaunchConfig,
    BeakerWekaBucket,
    OLMoCoreBeakerImage,
    get_beaker_client,
    is_running_in_beaker_batch_job,
)
from olmo_core.train.callbacks.beaker import BEAKER_RESULT_DIR
from olmo_core.utils import generate_uuid

log = logging.getLogger(__name__)

GOOGLE_CLUSTERS = [
    "ai2/augusta",
]


def maybe_get_beaker_client(workspace: str | None = None) -> Optional[Beaker]:
    try:
        return get_beaker_client(workspace=workspace)
    except BeakerError:
        return None


@lru_cache()
def get_beaker_username() -> Optional[str]:
    beaker = maybe_get_beaker_client()
    if beaker is not None:
        return beaker.user_name
    else:
        return None


def beaker_secret_exists(secret: str, workspace: Optional[str] = None) -> bool:
    beaker = maybe_get_beaker_client(workspace=workspace)
    if beaker is None:
        raise RuntimeError(
            "Environment not configured correctly for Beaker, you may be missing the BEAKER_TOKEN env var."
        )

    try:
        beaker.secret.get(secret)
        return True
    except BeakerSecretNotFound:
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
    if cluster.startswith("ai2/"):
        beaker = get_beaker_client()
        cl = beaker.cluster.get(cluster)
        tags = set(cl.tags)
        if "storage:weka" in tags:
            return "/weka/oe-training-default/ai2-llm"
        else:
            return "gs://ai2-llm"
    else:
        return "gs://ai2-llm"


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
        num_execution_units=num_execution_units,
        shared_filesystem=not is_url(root_dir),
        allow_dirty=False,
        env_vars=env_vars,
        env_secrets=[env_secret for env_secret in env_secrets if env_secret is not None],
        google_credentials_secret="GOOGLE_CREDENTIALS" if root_dir.startswith("gs://") else None,
        aws_config_secret=f"{beaker_user}_AWS_CONFIG",
        aws_credentials_secret=f"{beaker_user}_AWS_CREDENTIALS",
    )

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
    elif cluster.startswith("ai2/"):
        beaker = get_beaker_client()
        cl = beaker.cluster.get(cluster)
        for node in beaker.node.list(cluster=cl, limit=1):
            if (gpu_type := node.node_resources.gpu_type) > 0:
                return BeakerGpuType(gpu_type).name.replace("_", " ")
        else:
            raise RuntimeError(f"Could not determine GPU type for cluster '{cluster}'")
    elif cluster == "local":
        return torch.get_default_device().type
    else:
        raise ValueError(f"Unknown cluster '{cluster}'")
