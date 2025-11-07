import gc
import json
import logging
import os
import shutil
import tempfile
from fnmatch import fnmatch
from tempfile import TemporaryDirectory
from typing import List, Optional, Dict, Tuple, Any

import click
import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from cached_path import cached_path
from torch.distributed.checkpoint import TensorStorageMetadata

from olmo_core.aliases import PathOrStr
from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_model_and_optim_state, \
    get_checkpoint_metadata, load_state_dict, save_state_dict
from olmo_core.io import file_exists, join_path, copy_file
from olmo_core.nn.transformer import TransformerConfig, TransformerBlockConfig
from olmo_core.optim import OptimConfig
from olmo_core.utils import prepare_cli_environment


log = logging.getLogger(__name__)


def merge_checkpoints(
    model_paths: List[str],
    output_path: str,
) -> None:
    # sanity check
    if any(p.rstrip("/").endswith("model_and_optim") for p in model_paths):
        raise ValueError("Checkpoint paths must not end in 'model_and_optim'")
    if output_path.rstrip("/").endswith("model_and_optim"):
        raise ValueError("Output path must not end in 'model_and_optim'")

    # merge checkpoints
    checkpoint_paths = [join_path(p, "model_and_optim") for p in model_paths]
    checkpoint_metadata = [get_checkpoint_metadata(path) for path in checkpoint_paths]
    merged_state_dict = {}
    for i, (path, metadata) in enumerate(zip(checkpoint_paths, checkpoint_metadata)):
        for key, meta in metadata.state_dict_metadata.items():
            if not isinstance(meta, TensorStorageMetadata):
                if key in merged_state_dict:
                    log.info(f"Skipping non-tensor '{key}' from checkpoint {i} because we already have a value...")
                else:
                    log.info(f"Loading non-tensor '{key}' from checkpoint {i}...")
                    mini_state_dict = {key: None}
                    load_state_dict(path, mini_state_dict)
                    merged_state_dict[key] = mini_state_dict[key]
            else:
                if key not in merged_state_dict:
                    merged_state_dict[key] = torch.zeros(meta.size)
                log.info(f"Loading '{key}' from checkpoint {i}...")
                tensor = torch.empty_like(merged_state_dict[key])
                load_state_dict(path, {key: tensor})
                merged_state_dict[key].add_(tensor, alpha=1 / len(checkpoint_paths))
        gc.collect()

    # create the output dir
    log.info(f"Saving merged checkpoint to {output_path}...")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    def ignore_subdir(path: str, files):
        if os.path.basename(path) == "model_and_optim":
            return files
        else:
            return []
    shutil.copytree(model_paths[0], output_path, ignore=ignore_subdir)
    save_state_dict(join_path(output_path, "model_and_optim"), merged_state_dict)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--model",
    "-m",
    "model_paths",
    multiple=True,
    required=True,
    help="Model checkpoint path. Should be specified multiple times for different checkpoints"
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    help="Output directory for the merged checkpoint",
)
def main(model_paths: tuple, output_path: str):
    """
    Merge OLMo-core model checkpoints by averaging their weights.

    Weights are accumulated in float32 for numerical stability, then converted
    back to the original dtype of the checkpoints.

    Optimizer state is taken from the first model in the list.

    Examples:

    \b
    # Merge different local model checkpoints
    python merge_core_checkpoints.py \\
        --model gs://ai2-llm/checkpoints/stego32-highlr-filter3/step656000 \\
        --model gs://ai2-llm/checkpoints/stego32-highlr-filter3/step655000 \\
        --output ./merged_checkpoint
    """
    merge_checkpoints(model_paths=list(model_paths), output_path=output_path)


if __name__ == "__main__":
    prepare_cli_environment()
    main()
