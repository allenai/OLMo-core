#!/usr/bin/env python3
"""
Script to merge/average HuggingFace model checkpoints.

This script loads multiple HuggingFace checkpoints and averages their weights
to produce a merged checkpoint. Weight averaging is performed in float32 for
numerical stability, then converted back to the original dtype.

Example usage:
    python merge_hf_checkpoints.py \\
        --model allenai/OLMo-2-0425-1B \\
        --revisions stage1-step1000000-tokens2098B stage1-step1010000-tokens2119B \\
        --output ./merged_checkpoint
"""

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional

import click
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def merge_checkpoints(
    model_name_or_paths: List[str],
    revisions: Optional[List[Optional[str]]],
    output_dir: str,
    device: str = "cpu",
) -> None:
    """
    Merge multiple HuggingFace checkpoints by averaging their weights.

    Args:
        model_name_or_paths: List of model paths or HF Hub model IDs
        revisions: List of revisions (one per model, or single revision for all), or None
        output_dir: Directory where the merged model will be saved
        device: Device to load models on (default: cpu)
    """
    if not model_name_or_paths:
        raise ValueError("At least one checkpoint must be provided")

    # Handle revision list - default to None (will use "main" for HF Hub, ignored for local)
    if revisions is None or len(revisions) == 0:
        revisions = [None] * len(model_name_or_paths)
    elif len(revisions) == 1:
        revisions = revisions * len(model_name_or_paths)
    elif len(revisions) != len(model_name_or_paths):
        raise ValueError(
            f"Number of revisions ({len(revisions)}) must match number of models "
            f"({len(model_name_or_paths)}) or be 1"
        )

    n_checkpoints = len(model_name_or_paths)
    log.info(f"Merging {n_checkpoints} checkpoints...")

    # Track original dtype per tensor from first checkpoint
    original_dtypes: Dict[str, torch.dtype] = {}
    accumulated_state_dict = {}

    # Disable gradient computation - we're only doing inference/merging, no training
    with torch.no_grad():
        for i, (model_path, revision) in enumerate(zip(model_name_or_paths, revisions), 1):
            log.info(
                f"Loading checkpoint {i}/{n_checkpoints}: {model_path}"
                + (f" (revision: {revision})" if revision else "")
            )

            # Load model
            load_kwargs = {
                "dtype": "auto",
                "low_cpu_mem_usage": False,
            }
            if revision is not None:
                load_kwargs["revision"] = revision

            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs).to(device)
            state_dict = model.state_dict()
            del model

            # Store original dtype per tensor from first checkpoint
            if len(original_dtypes) <= 0:
                original_dtypes = {key: value.dtype for key, value in state_dict.items()}
                log.info(f"Stored original dtypes for {len(original_dtypes)} tensors")

            # Accumulate weights in float32
            log.info(f"Accumulating weights from checkpoint {i}...")
            for key in list(state_dict.keys()):
                value = state_dict.pop(key)
                # Convert to float32 for accumulation (only if floating point)
                if value.is_floating_point():
                    value_float32 = value.float()
                else:
                    # For non-floating point tensors (e.g., integers), keep as-is
                    value_float32 = value
                del value

                if key not in accumulated_state_dict:
                    accumulated_state_dict[key] = value_float32
                else:
                    accumulated_state_dict[key] += value_float32
                del value_float32

            # Free memory
            del state_dict
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Average and convert back to original dtypes
        log.info("Averaging weights and converting back to original dtypes...")
        averaged_state_dict = {}
        for key in list(accumulated_state_dict.keys()):
            value = accumulated_state_dict.pop(key)
            # Average
            value /= n_checkpoints
            # Convert back to original dtype for this specific tensor
            averaged_state_dict[key] = value.to(original_dtypes[key])
            del value

    # Load a fresh model with the first checkpoint's config to save with
    log.info("Creating model with averaged weights...")
    config_kwargs = {}
    if revisions[0] is not None:
        config_kwargs["revision"] = revisions[0]
    config = AutoConfig.from_pretrained(model_name_or_paths[0], **config_kwargs)
    merged_model = AutoModelForCausalLM.from_config(config)

    # Load the averaged state dict
    merged_model.load_state_dict(averaged_state_dict)

    # Save the merged model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving merged checkpoint to {output_path}...")
    merged_model.save_pretrained(output_path)

    # Also save the tokenizer from the first checkpoint
    from transformers import AutoTokenizer

    log.info("Saving tokenizer...")
    tokenizer_kwargs = {}
    if revisions[0] is not None:
        tokenizer_kwargs["revision"] = revisions[0]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_paths[0], **tokenizer_kwargs)
    tokenizer.save_pretrained(output_path)

    log.info(f"Successfully saved merged checkpoint to {output_path}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--model",
    "-m",
    "model_name_or_paths",
    multiple=True,
    required=True,
    help="Model path or HF Hub model ID. Can be specified multiple times for different models, "
    "or once if using --revisions to specify multiple revisions of the same model.",
)
@click.option(
    "--revisions",
    "-r",
    multiple=True,
    default=None,
    help="Revision(s) to use for HF Hub models. If one revision is provided with multiple models, it applies to all. "
    "Otherwise, number of revisions must match number of models. Defaults to 'main' for HF Hub models, ignored for local paths.",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    required=True,
    help="Output directory for the merged checkpoint",
)
@click.option(
    "--device",
    "-d",
    default="cpu",
    help="Device to load models on (default: cpu)",
)
def main(
    model_name_or_paths: tuple,
    revisions: tuple,
    output_dir: str,
    device: str,
):
    """
    Merge HuggingFace model checkpoints by averaging their weights.

    Weights are accumulated in float32 for numerical stability, then converted
    back to the original dtype of the checkpoints.

    Examples:

    \b
    # Merge multiple revisions of the same model
    python merge_hf_checkpoints.py \\
        --model allenai/OLMo-2-0425-1B \\
        --revisions stage1-step1000000-tokens2098B \\
        --revisions stage1-step1010000-tokens2119B \\
        --revisions stage1-step1020000-tokens2140B \\
        --output ./merged_checkpoint

    \b
    # Merge different local model checkpoints
    python merge_hf_checkpoints.py \\
        --model /path/to/checkpoint1 \\
        --model /path/to/checkpoint2 \\
        --output ./merged_checkpoint
    """
    model_list: List[str] = list(model_name_or_paths)
    revisions_list: Optional[List[Optional[str]]] = list(revisions) if revisions else None

    # If only one model is specified but multiple revisions, expand model list
    if revisions_list and len(model_list) == 1 and len(revisions_list) > 1:
        model_list = model_list * len(revisions_list)

    merge_checkpoints(
        model_name_or_paths=model_list,
        revisions=revisions_list,
        output_dir=output_dir,
        device=device,
    )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
