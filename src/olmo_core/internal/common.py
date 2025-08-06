import logging
from functools import lru_cache
from typing import List, Optional

import torch
from beaker import Beaker, BeakerError

from olmo_core.io import is_url
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerEnvVar,
    BeakerLaunchConfig,
    BeakerWekaBucket,
    OLMoCoreBeakerImage,
)
from olmo_core.utils import generate_uuid

log = logging.getLogger(__name__)


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


def get_root_dir(cluster: str) -> str:
    root_dir: str = "weka://oe-training-default/ai2-llm"
    if "cirrascale" in cluster or cluster == "ai2/test-h100":
        root_dir = "/weka/oe-training-default/ai2-llm"
    elif "google" in cluster:
        root_dir = "gs://ai2-llm"
    elif "local" in cluster:
        root_dir = "gs://ai2-llm"
    return root_dir


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
    root_dir: str,
    cmd: List[str],
    cluster: str,
    task_name: str = "train",
    workspace: str = "ai2/OLMo-core",
    budget: str = "ai2/oe-training",
    nccl_debug: bool = False,
    beaker_image: str = OLMoCoreBeakerImage.stable,
    num_nodes: int = 1,
) -> BeakerLaunchConfig:
    weka_buckets: List[BeakerWekaBucket] = []
    if root_dir.startswith("/weka/"):
        weka_buckets.append(BeakerWekaBucket("oe-training-default", "/weka/oe-training-default"))

    beaker_user = get_beaker_username()
    if beaker_user is None:
        raise RuntimeError(
            "Environment not configured correctly for Beaker, you may be missing the BEAKER_TOKEN env var."
        )

    return BeakerLaunchConfig(
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
        shared_filesystem=not is_url(root_dir),
        allow_dirty=False,
        env_vars=[BeakerEnvVar(name="NCCL_DEBUG", value="INFO" if nccl_debug else "WARN")],
        env_secrets=[
            BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
            BeakerEnvSecret(name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"),
            BeakerEnvSecret(name="COMET_API_KEY", secret=f"{beaker_user}_COMET_API_KEY"),
            BeakerEnvSecret(name="AWS_CONFIG", secret=f"{beaker_user}_AWS_CONFIG"),
            BeakerEnvSecret(name="AWS_CREDENTIALS", secret=f"{beaker_user}_AWS_CREDENTIALS"),
            BeakerEnvSecret(name="R2_ENDPOINT_URL", secret="R2_ENDPOINT_URL"),
            BeakerEnvSecret(name="WEKA_ENDPOINT_URL", secret="WEKA_ENDPOINT_URL"),
            BeakerEnvSecret(name="SLACK_WEBHOOK_URL", secret="SLACK_WEBHOOK_URL"),
        ],
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


CLUSTER_TO_GPU_TYPE = {
    "ai2/jupiter-cirrascale-2": "NVIDIA H100 80GB HBM3",
    "ai2/test-h100": "NVIDIA H100 80GB HBM3",
    "ai2/pluto-cirrascale": "NVIDIA H100",
    "ai2/augusta-google-1": "NVIDIA H100",
    "ai2/titan-cirrascale": "NVIDIA B200",
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
