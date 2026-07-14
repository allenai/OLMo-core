"""
Launch a vLLM model server on Beaker.

Usage:
    python src/scripts/beaker/serve_vllm.py \
        --model-path /weka/oe-adapt-default/baileyk/checkpoints/.../step3502-hf \
        --num-gpus 4 \
        --cluster ai2/saturn-cirrascale

The job will start a vLLM HTTP server (OpenAI-compatible API) on port 8000.
Once running, find the node hostname in the Beaker UI and call the server at:
    http://<hostname>:8000/v1
"""

import argparse
import sys

from rich import print

from olmo_core.launch.beaker import (
    BeakerLaunchConfig,
    BeakerWekaBucket,
    OLMoCoreBeakerImage,
)
from olmo_core.utils import generate_uuid, prepare_cli_environment

MODEL_PATH = (
    "/weka/oe-adapt-default/baileyk/checkpoints/olmo-sft/"
    "olmo-for-science-dolci-100k-drtulu-sera-toolu-50k/step3502-hf"
)


def build_config(model_path: str, num_gpus: int, clusters: list[str]) -> BeakerLaunchConfig:
    cmd = [
        "vllm",
        "serve",
        model_path,
        "--tool-call-parser",
        "hermes",
        "--enable-auto-tool-choice",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        str(num_gpus),
    ]

    return BeakerLaunchConfig(
        name=f"vllm-serve-{generate_uuid()[:8]}",
        budget="ai2/oe-adapt",
        cmd=cmd,
        task_name="serve",
        workspace="ai2/OLMo-core",
        beaker_image=OLMoCoreBeakerImage.stable,
        clusters=clusters,
        num_nodes=1,
        num_gpus=num_gpus,
        shared_filesystem=True,
        torchrun=False,
        weka_buckets=[
            BeakerWekaBucket(bucket="oe-adapt-default", mount="/weka/oe-adapt-default"),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch vLLM server on Beaker")
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Path to the HF model checkpoint (on Weka)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs (use enough to fit the model; 4 is a safe default for 7B)",
    )
    parser.add_argument(
        "--cluster",
        dest="clusters",
        action="append",
        default=["ai2/saturn-cirrascale", "ai2/neptune-cirrascale"],
        help="Beaker cluster(s) to target (can specify multiple times)",
    )
    args = parser.parse_args()

    prepare_cli_environment()

    config = build_config(args.model_path, args.num_gpus, args.clusters)
    print(config)
    print(
        f"\n[bold green]Launching vLLM server for:[/bold green] {args.model_path}\n"
        f"[bold]GPUs:[/bold] {args.num_gpus}  |  "
        f"[bold]Clusters:[/bold] {args.clusters}\n"
    )
    config.launch(follow=False, torchrun=False)
    print(
        "\n[bold yellow]Next steps:[/bold yellow]\n"
        "1. Open the Beaker UI and find your experiment\n"
        "2. Click on the job → find the [bold]node hostname[/bold] (e.g. neptune-node-42.reviz.ai2.in)\n"
        "3. From your interactive session, set:\n"
        "   [bold]export VLLM_BASE_URL=http://<hostname>:8000/v1[/bold]\n"
        "4. Wait for the server to print 'Application startup complete' in the logs\n"
        "5. Then run SWE-agent (see README for exact command)\n"
    )
