import logging
from typing import List, Optional

from beaker import Beaker

from olmo_core.io import is_url
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerLaunchConfig,
    BeakerWekaBucket,
    OLMoCoreBeakerImage,
)
from olmo_core.utils import generate_uuid

log = logging.getLogger(__name__)
_BEAKER_CLIENT: Optional[Beaker] = None
_BEAKER_USERNAME: Optional[str] = None


def get_beaker_client() -> Beaker:
    global _BEAKER_CLIENT

    if _BEAKER_CLIENT is None:
        _BEAKER_CLIENT = Beaker.from_env()

    return _BEAKER_CLIENT


def get_beaker_username() -> str:
    global _BEAKER_USERNAME

    if _BEAKER_USERNAME is None:
        _BEAKER_USERNAME = get_beaker_client().account.whoami().name

    return _BEAKER_USERNAME


def get_root_dir(cluster: str) -> str:
    root_dir: str = "weka://oe-training-default/ai2-llm"
    if "jupiter" in cluster:
        root_dir = "/weka/oe-training-default/ai2-llm"
    elif "augusta" in cluster:
        root_dir = "gs://ai2-llm"
    return root_dir


def get_work_dir(root_dir: str) -> str:
    return (
        "./dataset-cache"
        if is_url(root_dir)
        else f"{root_dir}/checkpoints/{get_beaker_username().lower()}/dataset-cache"
    )


def build_launch_config(
    *,
    name: str,
    root_dir: str,
    cmd: List[str],
    cluster: str,
    task_name: str = "train",
    workspace: str = "ai2/OLMo-core",
    budget: str = "ai2/oe-training",
) -> BeakerLaunchConfig:
    weka_buckets: List[BeakerWekaBucket] = []
    if root_dir.startswith("/weka/"):
        weka_buckets.append(BeakerWekaBucket("oe-training-default", "/weka/oe-training-default"))

    beaker_user = get_beaker_username()

    return BeakerLaunchConfig(
        name=f"{name}-{generate_uuid()[:8]}",
        budget=budget,
        cmd=cmd,
        task_name=task_name,
        workspace=workspace,
        clusters=[cluster],
        weka_buckets=weka_buckets,
        beaker_image=OLMoCoreBeakerImage.nightly,  # some features require nightly at the moment
        num_nodes=1,
        num_gpus=8,
        shared_filesystem=not is_url(root_dir),
        allow_dirty=False,
        env_secrets=[
            BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
            BeakerEnvSecret(name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"),
            BeakerEnvSecret(name="COMET_API_KEY", secret=f"{beaker_user}_COMET_API_KEY"),
            BeakerEnvSecret(name="AWS_CONFIG", secret=f"{beaker_user}_AWS_CONFIG"),
            BeakerEnvSecret(name="AWS_CREDENTIALS", secret=f"{beaker_user}_AWS_CREDENTIALS"),
            BeakerEnvSecret(name="R2_ENDPOINT_URL", secret="R2_ENDPOINT_URL"),
            BeakerEnvSecret(name="WEKA_ENDPOINT_URL", secret="WEKA_ENDPOINT_URL"),
        ],
        setup_steps=[
            # Clone repo.
            'git clone "$REPO_URL" .',
            'git checkout "$GIT_REF"',
            "git submodule update --init --recursive",
            # Setup python environment.
            "conda shell.bash activate base",
            "pip install -e '.[all]'",
            "pip freeze",
            # Move AWS credentials from env to relevant files
            "mkdir -p ~/.aws",
            "printenv AWS_CONFIG > ~/.aws/config",
            "printenv AWS_CREDENTIALS > ~/.aws/credentials",
        ],
    )


CLUSTER_TO_GPU_TYPE = {
    "ai2/jupiter-cirrascale-2": "NVIDIA H100 80GB HBM3",
    "ai2/pluto-cirrascale": "NVIDIA H100",
    "ai2/augusta-google-1": "NVIDIA H100",
}


def get_gpu_type(cluster: str) -> str:
    if cluster in CLUSTER_TO_GPU_TYPE:
        return CLUSTER_TO_GPU_TYPE[cluster]
    else:
        log.warning(f"Missing cluster '{cluster}' in CLUSTER_TO_GPU_TYPE mapping")
        beaker = get_beaker_client()
        nodes = beaker.cluster.nodes(cluster)
        assert nodes and nodes[0].limits.gpu_type
        return nodes[0].limits.gpu_type
