"""Load and convert Hugging Face SigLIP vision-only checkpoints."""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

import torch

from .config import VisionEncoderConfig

__all__ = [
    "load_siglip_hf_vision_state_dict",
    "siglip_hf_state_dict_to_vision",
    "vision_state_fingerprint",
]


class SiglipLoaderError(RuntimeError):
    """Raised when a Hugging Face SigLIP checkpoint cannot be mapped exactly."""


def _require(state_dict: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in state_dict:
        raise SiglipLoaderError(f"Missing required Hugging Face SigLIP key: {key!r}")
    return state_dict[key]


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
    """Convert a Hugging Face SigLIP encoder into OLMo-core's native layout.

    :param hf_state_dict: State returned by :func:`load_siglip_hf_vision_state_dict`, or the
        ``vision_model`` state dictionary from a Hugging Face ``SiglipVisionModel``.
    :param vision_cfg: Target OLMo-core vision configuration.

    :returns: A state dictionary accepted strictly by
        :class:`~olmo_core.nn.vision.VisionTransformer`.
    """
    out: Dict[str, torch.Tensor] = {}
    patch_weight = _require(hf_state_dict, "embeddings.patch_embedding.weight")
    out["patch_embedding.weight"] = patch_weight.reshape(patch_weight.shape[0], -1)
    out["patch_embedding.bias"] = _require(hf_state_dict, "embeddings.patch_embedding.bias")
    out["positional_embedding"] = _require(hf_state_dict, "embeddings.position_embedding.weight")

    for layer_idx in range(vision_cfg.image_num_layers):
        source = f"encoder.layers.{layer_idx}"
        target = f"blocks.{layer_idx}"
        for hf_name, native_name in (
            ("layer_norm1", "attn_norm"),
            ("layer_norm2", "ffn_norm"),
        ):
            for suffix in ("weight", "bias"):
                out[f"{target}.{native_name}.{suffix}"] = _require(
                    hf_state_dict, f"{source}.{hf_name}.{suffix}"
                )
        for hf_name, native_name in (
            ("q_proj", "wq"),
            ("k_proj", "wk"),
            ("v_proj", "wv"),
            ("out_proj", "wo"),
        ):
            for suffix in ("weight", "bias"):
                out[f"{target}.attn.{native_name}.{suffix}"] = _require(
                    hf_state_dict, f"{source}.self_attn.{hf_name}.{suffix}"
                )
        for hf_name, native_name in (("fc1", "w1"), ("fc2", "w2")):
            for suffix in ("weight", "bias"):
                out[f"{target}.ffn.{native_name}.{suffix}"] = _require(
                    hf_state_dict, f"{source}.mlp.{hf_name}.{suffix}"
                )
    return out


def vision_state_fingerprint(
    state_dict: Dict[str, torch.Tensor], *, samples_per_tensor: int = 32
) -> str:
    """Return a deterministic sampled SHA-256 fingerprint for a vision state dictionary.

    Tensor names, shapes, dtypes, and evenly spaced values are hashed. Sampling keeps the startup
    invariant cheap enough for a large vision tower while still detecting an incorrect revision,
    namespace, or tensor load.

    :param state_dict: Vision state dictionary to fingerprint.
    :param samples_per_tensor: Maximum number of values sampled from each tensor.

    :returns: A lowercase SHA-256 hex digest.
    """
    if samples_per_tensor <= 0:
        raise ValueError("samples_per_tensor must be positive")

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().reshape(-1)
        digest.update(name.encode())
        digest.update(str(tuple(state_dict[name].shape)).encode())
        digest.update(str(state_dict[name].dtype).encode())
        if tensor.numel() == 0:
            continue
        count = min(samples_per_tensor, tensor.numel())
        indices = torch.linspace(0, tensor.numel() - 1, count, dtype=torch.long)
        values = tensor.index_select(0, indices).to(torch.float64).contiguous().numpy()
        digest.update(values.tobytes())
    return digest.hexdigest()
