"""
Multi-crop preprocessing for the multimodal data pipeline.

Implements two crop strategies from the Molmo2 reference:

- :attr:`CropMode.resize` — a single full-resolution crop, the image resized
  + padded to ``base_image_input_size``.
- :attr:`CropMode.overlap_and_resize` — a global low-resolution view plus
  one to ``max_crops`` overlapping regional crops chosen by aspect-ratio fit.
  Each regional crop's borders (the overlap margins) are masked from the
  pooling so the model never attends to a patch from two neighbouring crops.

Output for every image is a dict matching what
:class:`~olmo_core.nn.vision.MultimodalTransformer.forward` expects (minus the
prompt/response text, which is added by the upstream preprocessor).
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from ...config import Config, StrEnum
from .image_preprocessor import ImagePreprocessor, ImagePreprocessorConfig
from .tokens import MultimodalTokenizerConfig

__all__ = [
    "CropMode",
    "MultiCropPreprocessorConfig",
    "MultiCropPreprocessor",
    "MultiCropOutput",
    "select_tiling",
    "arange_for_pooling",
]


class CropMode(StrEnum):
    """How an image is decomposed into crops."""

    resize = "resize"
    """One crop, image resized + padded to ``base_image_input_size``."""

    overlap_and_resize = "overlap_and_resize"
    """A global low-res view + ``≤ max_crops`` overlapping regional crops."""


@dataclass
class MultiCropOutput:
    """Per-image preprocessing result."""

    image_tokens: np.ndarray
    """``(n_image_tokens,)`` token IDs to interleave into the prompt."""

    images: np.ndarray
    """``(n_crops, n_patches_per_crop, 3 * p * p)`` pre-patchified pixels."""

    image_masks: np.ndarray
    """``(n_crops, n_patches_per_crop)`` per-patch validity in ``[0, 1]``."""

    pooled_patches_idx: np.ndarray
    """``(n_pooled, pool_h * pool_w)`` indices into the flattened
    ``(n_crops * n_patches_per_crop)`` patch axis; ``-1`` = padding."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def select_tiling(
    h: int,
    w: int,
    crop_window_size: int,
    max_num_crops: int,
) -> Tuple[int, int]:
    """Choose a ``(rows, cols)`` tiling for an ``h × w`` image.

    Searches every ``(i, j)`` with ``i * j ≤ max_num_crops`` and returns the
    layout that requires the least up-scaling (or the least down-scaling if
    every option requires shrinking).
    """
    tilings: List[Tuple[int, int]] = []
    for i in range(1, max_num_crops + 1):
        for j in range(1, max_num_crops + 1):
            if i * j <= max_num_crops:
                tilings.append((i, j))
    # Sort so ties favour fewer crops.
    tilings.sort(key=lambda x: (x[0] * x[1], x[0]))
    candidate = np.array(tilings, dtype=np.int32)  # (n, 2)
    candidate_res = candidate * crop_window_size  # pixel size of each tiling

    with np.errstate(divide="ignore"):
        required = candidate_res.astype(np.float32) / np.asarray([h, w], dtype=np.float32)
    required = np.min(required, axis=-1, keepdims=True)  # (n, 1)
    if np.all(required < 1):
        ix = int(np.argmax(required))  # least downscaling
    else:
        required = np.where(required < 1.0, 10e9, required)
        ix = int(np.argmin(required))  # least upscaling
    return int(tilings[ix][0]), int(tilings[ix][1])


def arange_for_pooling(idx_arr: np.ndarray, pool_h: int, pool_w: int) -> np.ndarray:
    """Pad a 2-D ``(H, W)`` patch-index map and arrange into pool groups.

    Pads with ``-1`` so each dim becomes a multiple of the corresponding pool
    size, then rearranges to ``(H // pool_h, W // pool_w, pool_h * pool_w)``.
    """
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    padded = np.pad(
        idx_arr,
        [[h_pad // 2, (h_pad + 1) // 2], [w_pad // 2, (w_pad + 1) // 2]],
        mode="constant",
        constant_values=-1,
    )
    H, W = padded.shape
    nH, nW = H // pool_h, W // pool_w
    # (nH, pool_h, nW, pool_w) → (nH, nW, pool_h, pool_w) → (nH, nW, pool_h*pool_w)
    return (
        padded.reshape(nH, pool_h, nW, pool_w)
        .transpose(0, 2, 1, 3)
        .reshape(nH, nW, pool_h * pool_w)
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MultiCropPreprocessorConfig(Config):
    """Configuration for :class:`MultiCropPreprocessor`."""

    base_image_input_size: Tuple[int, int] = (336, 336)
    """Output crop size in pixels. Must be a multiple of ``patch_size``."""

    pool_h: int = 2
    """Pooling group height (patches)."""

    pool_w: int = 2
    """Pooling group width (patches)."""

    crop_mode: CropMode = CropMode.resize
    """Which crop strategy to use."""

    max_crops: int = 6
    """Maximum number of regional crops for :attr:`CropMode.overlap_and_resize`.
    Includes the global view, so set to e.g. 6 for ``1 global + ≤ 5 regional``."""

    overlap_margins: Tuple[int, int] = (2, 2)
    """``(left, right)`` patch counts that are dropped from each crop's pooling
    indices because they overlap with the neighbour. Only used in
    :attr:`CropMode.overlap_and_resize`."""

    use_col_tokens: bool = True
    """Insert ``<im_col>`` after every row of pooled patches."""

    use_low_res_token_for_global: bool = False
    """In overlap mode, use ``<im_low>`` (not ``<im_patch>``) for the global view's
    placeholder tokens. ``False`` matches Molmo2's default — global and regional
    crops share ``<im_patch>``, which keeps the model's splice contract simple
    (the splice writes features at every ``<im_patch>`` position). Set to
    ``True`` only if your model is configured to splice both token IDs."""

    use_low_res_start_token: bool = False
    """In overlap mode, open the global view with ``<low_res_im_start>`` instead
    of ``<im_start>``. ``False`` matches Molmo2's default."""

    image_preprocessor: ImagePreprocessorConfig = field(default_factory=ImagePreprocessorConfig)
    """Resize + normalize + patchify settings for each crop."""

    def build(self, tokenizer: MultimodalTokenizerConfig) -> "MultiCropPreprocessor":
        """Instantiate a :class:`MultiCropPreprocessor` bound to a tokenizer."""
        return MultiCropPreprocessor(self, tokenizer)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class MultiCropPreprocessor:
    """Multi-crop preprocessor that produces a :class:`MultiCropOutput` per image."""

    def __init__(
        self,
        cfg: MultiCropPreprocessorConfig,
        tokenizer: MultimodalTokenizerConfig,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.image_preprocessor: ImagePreprocessor = cfg.image_preprocessor.build()

        h, w = cfg.base_image_input_size
        p = cfg.image_preprocessor.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError(
                f"base_image_input_size {cfg.base_image_input_size} not divisible by patch_size {p}"
            )
        self.crop_patch_h = h // p
        self.crop_patch_w = w // p

    # ------------------------------------------------------------------
    # Token-row construction
    # ------------------------------------------------------------------

    def _row_tokens(self, n_per_row: int, patch_token: int) -> np.ndarray:
        row = np.full(n_per_row, patch_token, dtype=np.int64)
        if self.cfg.use_col_tokens:
            row = np.concatenate([row, [self.tokenizer.image_col_id]], dtype=np.int64)
        return row

    # ------------------------------------------------------------------
    # `resize` mode
    # ------------------------------------------------------------------

    def _resize_mode(self, image) -> MultiCropOutput:
        cfg = self.cfg
        h, w = cfg.base_image_input_size
        patches, mask = self.image_preprocessor.preprocess(image, target_size=(h, w))
        # patches: (n_patches, p*p*3); mask: (n_patches,)
        n_patches = patches.shape[0]

        # Build the 2-D patch grid and the pooling groups.
        grid = np.arange(n_patches, dtype=np.int64).reshape(self.crop_patch_h, self.crop_patch_w)
        groups = arange_for_pooling(grid, cfg.pool_h, cfg.pool_w)  # (nH, nW, pool*pool)
        nH, nW = groups.shape[:2]
        pooled_patches_idx = groups.reshape(nH * nW, cfg.pool_h * cfg.pool_w)

        # Image tokens: <im_start> [<patch>×nW <im_col>]×nH <im_end>
        row = self._row_tokens(nW, self.tokenizer.image_patch_id)
        body = np.tile(row, nH)
        image_tokens = np.concatenate(
            [
                np.asarray([self.tokenizer.image_start_id], dtype=np.int64),
                body,
                np.asarray([self.tokenizer.image_end_id], dtype=np.int64),
            ]
        )

        return MultiCropOutput(
            image_tokens=image_tokens,
            images=patches[None, ...],  # (1, n_patches, patch_dim)
            image_masks=mask[None, ...],  # (1, n_patches)
            pooled_patches_idx=pooled_patches_idx,
        )

    # ------------------------------------------------------------------
    # `overlap-and-resize` mode
    # ------------------------------------------------------------------

    def _overlap_mode(self, image) -> MultiCropOutput:
        cfg = self.cfg
        ip = self.image_preprocessor
        patch_size = cfg.image_preprocessor.patch_size
        crop_size = cfg.base_image_input_size[0]
        assert (
            cfg.base_image_input_size[0] == cfg.base_image_input_size[1]
        ), "overlap_and_resize requires square crops"

        left_margin, right_margin = cfg.overlap_margins
        crop_patches = crop_size // patch_size
        crop_window_patches = crop_patches - (left_margin + right_margin)
        if crop_window_patches <= 0:
            raise ValueError(
                f"overlap_margins {cfg.overlap_margins} consume the entire crop "
                f"(crop has {crop_patches} patches per side)"
            )
        crop_window_size = crop_window_patches * patch_size
        total_margin_pixels = patch_size * (left_margin + right_margin)

        # Source image dims.
        arr = ip._to_float_hwc(image)
        src_h, src_w = arr.shape[:2]
        tiling = select_tiling(
            max(src_h - total_margin_pixels, 1),
            max(src_w - total_margin_pixels, 1),
            crop_window_size,
            cfg.max_crops,
        )

        # Resize the source image to exactly fit the chosen tiling, with margins.
        target_h = tiling[0] * crop_window_size + total_margin_pixels
        target_w = tiling[1] * crop_window_size + total_margin_pixels
        src_image, src_mask = ip.resize_and_pad(image, (target_h, target_w))
        src_image = ip.normalize(src_image)

        # Slide overlapping windows of size `crop_size` with stride `crop_window_size`.
        n_crops = tiling[0] * tiling[1]
        crop_arr = np.zeros((n_crops, crop_size, crop_size, 3), dtype=np.float32)
        mask_arr = np.zeros((n_crops, crop_size, crop_size), dtype=np.float32)
        patch_idx_arr = np.zeros((n_crops, self.crop_patch_h, self.crop_patch_w), dtype=np.int64)
        on_crop = 0
        for i in range(tiling[0]):
            y0 = i * crop_window_size
            for j in range(tiling[1]):
                x0 = j * crop_window_size
                crop_arr[on_crop] = src_image[y0 : y0 + crop_size, x0 : x0 + crop_size]
                mask_arr[on_crop] = src_mask[y0 : y0 + crop_size, x0 : x0 + crop_size]
                patch_idx = np.arange(
                    self.crop_patch_h * self.crop_patch_w, dtype=np.int64
                ).reshape(self.crop_patch_h, self.crop_patch_w)
                patch_idx = patch_idx + on_crop * self.crop_patch_h * self.crop_patch_w
                # Mask the overlap region; neighbour crops own those patches.
                if i != 0:
                    patch_idx[:left_margin, :] = -1
                if j != 0:
                    patch_idx[:, :left_margin] = -1
                if i != tiling[0] - 1:
                    patch_idx[-right_margin:, :] = -1
                if j != tiling[1] - 1:
                    patch_idx[:, -right_margin:] = -1
                patch_idx_arr[on_crop] = patch_idx
                on_crop += 1

        # Pool the regional layout (the tiled patch_idx map spans the full image grid).
        full_grid = np.zeros(
            (tiling[0] * self.crop_patch_h, tiling[1] * self.crop_patch_w), dtype=np.int64
        )
        on_crop = 0
        for i in range(tiling[0]):
            for j in range(tiling[1]):
                ys, xs = i * self.crop_patch_h, j * self.crop_patch_w
                full_grid[ys : ys + self.crop_patch_h, xs : xs + self.crop_patch_w] = patch_idx_arr[
                    on_crop
                ]
                on_crop += 1
        regional_groups = arange_for_pooling(full_grid, cfg.pool_h, cfg.pool_w)
        nH_r, nW_r = regional_groups.shape[:2]
        regional_pooled_idx = regional_groups.reshape(nH_r * nW_r, cfg.pool_h * cfg.pool_w)
        # Each regional crop occupies (crop_patch_h * crop_patch_w) slots in the
        # flat image_features tensor; shift the indices because the global crop
        # is going to be inserted at the front.
        regional_pooled_idx = np.where(
            regional_pooled_idx >= 0,
            regional_pooled_idx + self.crop_patch_h * self.crop_patch_w,
            -1,
        )

        # Build the global low-res view.
        h_g, w_g = cfg.base_image_input_size
        global_patches, global_mask = ip.preprocess(image, target_size=(h_g, w_g))
        global_grid = np.arange(self.crop_patch_h * self.crop_patch_w, dtype=np.int64).reshape(
            self.crop_patch_h, self.crop_patch_w
        )
        global_groups = arange_for_pooling(global_grid, cfg.pool_h, cfg.pool_w)
        nH_g, nW_g = global_groups.shape[:2]
        global_pooled_idx = global_groups.reshape(nH_g * nW_g, cfg.pool_h * cfg.pool_w)

        pooled_patches_idx = np.concatenate([global_pooled_idx, regional_pooled_idx], axis=0)

        # Patchify all crops (global + regional) into one stacked tensor.
        global_patches_4d = global_patches[None, ...]  # (1, P, dim)
        global_mask_2d = global_mask[None, ...]  # (1, P)
        regional_patches = np.stack([ip.patchify(crop_arr[c]) for c in range(n_crops)], axis=0)
        regional_masks = np.stack([ip.patchify_mask(mask_arr[c]) for c in range(n_crops)], axis=0)
        images = np.concatenate([global_patches_4d, regional_patches], axis=0)
        masks = np.concatenate([global_mask_2d, regional_masks], axis=0)

        # Build the token sequence.
        global_patch_token = (
            self.tokenizer.image_low_id
            if cfg.use_low_res_token_for_global
            else self.tokenizer.image_patch_id
        )
        global_start_token = (
            self.tokenizer.low_res_image_start_id
            if cfg.use_low_res_start_token
            else self.tokenizer.image_start_id
        )
        global_row = self._row_tokens(nW_g, global_patch_token)
        global_body = np.tile(global_row, nH_g)
        regional_row = self._row_tokens(nW_r, self.tokenizer.image_patch_id)
        regional_body = np.tile(regional_row, nH_r)

        image_tokens = np.concatenate(
            [
                np.asarray([global_start_token], dtype=np.int64),
                global_body,
                np.asarray([self.tokenizer.image_end_id], dtype=np.int64),
                np.asarray([self.tokenizer.image_start_id], dtype=np.int64),
                regional_body,
                np.asarray([self.tokenizer.image_end_id], dtype=np.int64),
            ]
        )

        return MultiCropOutput(
            image_tokens=image_tokens,
            images=images,
            image_masks=masks,
            pooled_patches_idx=pooled_patches_idx,
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def __call__(self, image) -> MultiCropOutput:
        """Preprocess a single image into a :class:`MultiCropOutput`.

        :param image: A PIL image or HWC ``np.ndarray`` (uint8 or float32).
        """
        if self.cfg.crop_mode == CropMode.resize:
            return self._resize_mode(image)
        elif self.cfg.crop_mode == CropMode.overlap_and_resize:
            return self._overlap_mode(image)
        else:
            raise NotImplementedError(f"unsupported crop_mode: {self.cfg.crop_mode}")
