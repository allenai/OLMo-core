"""Load and convert Hugging Face SigLIP vision-only checkpoints."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .config import VisionEncoderConfig

__all__ = [
    "load_siglip_hf_vision_state_dict",
    "siglip_hf_state_dict_to_vision",
]


class SiglipLoaderError(RuntimeError):
    """Raised when a Hugging Face SigLIP checkpoint cannot be mapped exactly."""


def load_siglip_hf_vision_state_dict(
    model_id: str,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> Dict[str, torch.Tensor]:
    """Load only the encoder tensors from a Hugging Face SigLIP checkpoint.

    The text tower, post-layer norm, and pooling head are excluded. Returned keys have the
    ``vision_model.`` prefix removed and can be passed directly to
    :func:`siglip_hf_state_dict_to_vision`.

    :param model_id: Hugging Face repository ID.
    :param revision: Immutable Hugging Face commit to load.
    :param cache_dir: Optional Hugging Face cache directory.
    :param local_files_only: Refuse network access when ``True``.

    :returns: The SigLIP embeddings and encoder state dictionary on CPU.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    checkpoint_path = hf_hub_download(
        repo_id=model_id,
        filename="model.safetensors",
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    prefixes = ("vision_model.embeddings.", "vision_model.encoder.")
    state_dict: Dict[str, torch.Tensor] = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            if key.startswith(prefixes):
                state_dict[key.removeprefix("vision_model.")] = checkpoint.get_tensor(key)

    if not state_dict:
        raise SiglipLoaderError(f"No SigLIP vision encoder weights were found in {model_id!r}")
    return state_dict


def siglip_hf_state_dict_to_vision(
    hf_state_dict: Dict[str, torch.Tensor],
    vision_cfg: VisionEncoderConfig,
) -> Dict[str, torch.Tensor]:
    """Convert a Hugging Face SigLIP encoder into the configured native layout."""
    from .image_vit import siglip_state_dict_to_vision_encoder

    try:
        return siglip_state_dict_to_vision_encoder(
            hf_state_dict, n_blocks=vision_cfg.image_num_layers
        )
    except KeyError as error:
        raise SiglipLoaderError(f"Missing required SigLIP encoder weight: {error}") from error
