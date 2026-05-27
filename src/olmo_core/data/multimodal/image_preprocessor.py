"""
Image preprocessing for the multimodal data pipeline.

Converts a PIL image (or HWC ``numpy`` array) into the
``(n_patches, patch_dim)`` patch tensor + ``(n_patches,)`` validity mask
that the vision tower expects.

Patch flatten convention is **channel-first** (``[c, kh, kw]``) — matching the
HuggingFace ``Conv2d`` patch embedding semantics our parity tests verified
against ``openai/clip-vit-large-patch14-336`` and
``google/siglip-so400m-patch14-384``.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ...config import Config, StrEnum

__all__ = [
    "NormalizeStyle",
    "ImagePreprocessorConfig",
    "ImagePreprocessor",
]


# ---------------------------------------------------------------------------
# Standard mean / std constants (RGB, [0, 1] scale)
# ---------------------------------------------------------------------------

OPENAI_CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


class NormalizeStyle(StrEnum):
    """How to normalize pixel values before patchification."""

    siglip = "siglip"
    """SigLIP / SigLIP2: scale ``[0, 1] → [-1, 1]`` via ``image * 2 - 1``."""

    openai = "openai"
    """OpenAI CLIP: ``(image - OPENAI_CLIP_MEAN) / OPENAI_CLIP_STD``."""


@dataclass
class ImagePreprocessorConfig(Config):
    """
    Configuration for :class:`ImagePreprocessor`.
    """

    patch_size: int = 14
    """Pixel size of each square patch. Must divide every crop dimension."""

    normalize: NormalizeStyle = NormalizeStyle.siglip
    """Normalization style; pick the one matching your vision encoder."""

    pad_value: float = 0.0
    """Pixel value (in the ``[0, 1]`` scale, pre-normalize) used to pad the
    image when its aspect ratio differs from the target crop."""

    def build(self) -> "ImagePreprocessor":
        """Instantiate an :class:`ImagePreprocessor` for this config."""
        return ImagePreprocessor(self)


class ImagePreprocessor:
    """Resize, normalize, and patchify a single image."""

    def __init__(self, cfg: ImagePreprocessorConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Resize + pad
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float_hwc(image) -> np.ndarray:
        """Coerce input to a float32 HWC array in [0, 1]."""
        if isinstance(image, np.ndarray):
            arr = image
        else:
            # PIL.Image or anything with .convert / np.asarray support
            arr = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.max() > 1.5:  # likely uint8 range
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] != 3:
            raise ValueError(f"expected RGB image, got shape {arr.shape}")
        return arr

    def resize_and_pad(
        self,
        image,
        target_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize aspect-preserving to fit ``target_size``, then pad.

        :param image: PIL image, HWC ``np.uint8`` / ``np.float32`` array.
        :param target_size: ``(target_h, target_w)``.
        :returns: ``(image_arr, mask_arr)``:
            - ``image_arr``: ``(target_h, target_w, 3)`` ``float32`` in
              ``[0, 1]`` (pre-normalize). Padded regions equal ``pad_value``.
            - ``mask_arr``: ``(target_h, target_w)`` ``float32`` with ``1.0``
              where the image content lives and ``0.0`` where it was padded.
        """
        cfg = self.cfg
        arr = self._to_float_hwc(image)
        src_h, src_w, _ = arr.shape
        tgt_h, tgt_w = target_size

        # Scale to fit (aspect-preserving). Use the dimension that's the tighter limit.
        scale = min(tgt_h / src_h, tgt_w / src_w)
        new_h = max(1, int(round(src_h * scale)))
        new_w = max(1, int(round(src_w * scale)))

        # Bilinear resize via numpy. Keeping it dependency-free.
        resized = _bilinear_resize(arr, new_h, new_w)

        # Pad to target.
        out = np.full((tgt_h, tgt_w, 3), cfg.pad_value, dtype=np.float32)
        mask = np.zeros((tgt_h, tgt_w), dtype=np.float32)
        pad_top = (tgt_h - new_h) // 2
        pad_left = (tgt_w - new_w) // 2
        out[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
        mask[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = 1.0
        return out, mask

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply the configured normalization (in-place safe)."""
        cfg = self.cfg
        if cfg.normalize == NormalizeStyle.siglip:
            return image * 2.0 - 1.0
        elif cfg.normalize == NormalizeStyle.openai:
            mean = np.asarray(OPENAI_CLIP_MEAN, dtype=np.float32)[None, None, :]
            std = np.asarray(OPENAI_CLIP_STD, dtype=np.float32)[None, None, :]
            return (image - mean) / std
        else:
            raise NotImplementedError(f"unsupported normalize style: {cfg.normalize}")

    # ------------------------------------------------------------------
    # Patchify (channel-first flatten)
    # ------------------------------------------------------------------

    def patchify(self, image: np.ndarray) -> np.ndarray:
        """Reshape ``(H, W, 3)`` to ``(n_patches, 3 * p * p)`` with C-first flatten.

        The output order for each patch matches the natural flatten of a
        HuggingFace ``Conv2d(kernel=p, stride=p)`` weight reshaped via
        ``.reshape(D, -1)``: index ``i = c * p * p + kh * p + kw`` selects
        pixel ``(c, kh, kw)`` within the patch.
        """
        p = self.cfg.patch_size
        h, w, c = image.shape
        if h % p != 0 or w % p != 0:
            raise ValueError(f"image size ({h}, {w}) is not divisible by patch_size {p}")
        # (H, W, C) → (h_patches, p, w_patches, p, C)
        x = image.reshape(h // p, p, w // p, p, c)
        # → (h_patches, w_patches, C, p, p) so flatten is C-first per patch.
        x = x.transpose(0, 2, 4, 1, 3)
        return x.reshape((h // p) * (w // p), c * p * p).astype(np.float32, copy=False)

    def patchify_mask(self, mask: np.ndarray) -> np.ndarray:
        """Per-pixel mask ``(H, W)`` → per-patch coverage ``(n_patches,)``.

        Returns the mean of the per-pixel mask within each patch, so partially
        padded patches receive a fractional weight in ``(0, 1)``.
        """
        p = self.cfg.patch_size
        h, w = mask.shape
        if h % p != 0 or w % p != 0:
            raise ValueError(f"mask size ({h}, {w}) is not divisible by patch_size {p}")
        x = (
            mask.reshape(h // p, p, w // p, p)
            .transpose(0, 2, 1, 3)
            .reshape((h // p) * (w // p), p * p)
        )
        return x.mean(axis=-1).astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Convenience: full pipeline for a single crop
    # ------------------------------------------------------------------

    def preprocess(
        self,
        image,
        target_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize, normalize, and patchify a single image.

        :returns: ``(patches, mask)`` with shapes ``(n_patches, 3 * p * p)``
            and ``(n_patches,)`` respectively.
        """
        image_arr, mask_arr = self.resize_and_pad(image, target_size)
        image_arr = self.normalize(image_arr)
        return self.patchify(image_arr), self.patchify_mask(mask_arr)


# ---------------------------------------------------------------------------
# Dependency-free bilinear resize
# ---------------------------------------------------------------------------


def _bilinear_resize(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Bilinear resize of an HWC float32 array.

    Implemented in pure numpy so the preprocessor has no torchvision or PIL
    dependency at the resize step. Aligns corners the same way ``PIL.Image
    .resize(..., BILINEAR)`` does (half-pixel offsets).
    """
    src_h, src_w, c = image.shape
    if src_h == new_h and src_w == new_w:
        return image.astype(np.float32, copy=False)

    # Compute source coordinates of each output pixel (half-pixel sampling).
    y = (np.arange(new_h, dtype=np.float32) + 0.5) * (src_h / new_h) - 0.5
    x = (np.arange(new_w, dtype=np.float32) + 0.5) * (src_w / new_w) - 0.5
    y0 = np.clip(np.floor(y).astype(np.int64), 0, src_h - 1)
    x0 = np.clip(np.floor(x).astype(np.int64), 0, src_w - 1)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    wy = np.clip(y - y0, 0.0, 1.0)
    wx = np.clip(x - x0, 0.0, 1.0)

    # Gather and blend the four neighbors.
    top_left = image[y0[:, None], x0[None, :], :]
    top_right = image[y0[:, None], x1[None, :], :]
    bot_left = image[y1[:, None], x0[None, :], :]
    bot_right = image[y1[:, None], x1[None, :], :]

    wy_2d = wy[:, None, None]
    wx_2d = wx[None, :, None]
    top = top_left * (1 - wx_2d) + top_right * wx_2d
    bot = bot_left * (1 - wx_2d) + bot_right * wx_2d
    return (top * (1 - wy_2d) + bot * wy_2d).astype(np.float32)
