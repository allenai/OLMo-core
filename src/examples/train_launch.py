"""
An example of how to launch the training script on Beaker.
Run this with:

    python src/examples/train_launch.py
"""

from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.utils import generate_uuid, prepare_cli_environment

LAUNCH_CONFIG = BeakerLaunchConfig(
    name=f"olmo-core-test-{generate_uuid()[:8]}",
    budget="ai2/oe-training",
    cmd=["src/examples/train.py"],
    task_name="train",
    workspace="ai2/OLMo-core",
    description="Testing OLMo-core launch utilities",
    clusters=["ai2/allennlp-cirrascale"],
    num_nodes=1,
    num_gpus=4,
    shared_filesystem=True,
    nfs=True,
    allow_dirty=True,
)

if __name__ == "__main__":
    prepare_cli_environment()
    LAUNCH_CONFIG.launch(follow=True)
