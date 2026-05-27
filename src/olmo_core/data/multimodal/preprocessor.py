"""
Top-level multimodal preprocessor.

Combines :class:`MultiCropPreprocessor` with a text tokenizer to produce the
full per-example dict expected by
:class:`~olmo_core.nn.vision.MultimodalTransformer.forward`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...config import Config
from .multicrop import MultiCropPreprocessor, MultiCropPreprocessorConfig
from .tokens import MultimodalTokenizerConfig

__all__ = [
    "MultimodalPreprocessorConfig",
    "MultimodalPreprocessor",
]


@dataclass
class MultimodalPreprocessorConfig(Config):
    """Configuration for :class:`MultimodalPreprocessor`."""

    tokenizer: MultimodalTokenizerConfig = field(default_factory=MultimodalTokenizerConfig.dolma2)
    """Tokenizer config defining image special token IDs."""

    multicrop: MultiCropPreprocessorConfig = field(default_factory=MultiCropPreprocessorConfig)
    """Multi-crop / image preprocessing settings."""

    max_sequence_length: int = 2048
    """Hard upper bound on the produced token sequence length. Sequences are
    truncated from the right (response tail dropped)."""

    prompt_template: str = "{image}\n{prompt}"
    """Template for the prompt portion. Must contain a single ``{image}`` placeholder
    where image tokens are inserted, and may contain ``{prompt}``. When *image*
    is ``None`` the ``{image}`` placeholder is dropped (the surrounding newline
    is removed as well to avoid a dangling line break)."""

    response_template: str = " {response}"
    """Template for the response portion. May contain ``{response}``. The
    leading space matches Molmo's convention where the response starts after
    a delimiter."""

    add_eos: bool = True
    """If ``True``, append the base tokenizer's ``eos_token_id`` to the response
    (with ``loss_mask=1``) so the model learns to stop."""


class MultimodalPreprocessor:
    """Build per-example training dicts from raw ``(prompt, response, image)``.

    The HuggingFace tokenizer is supplied at construction time so callers can
    decide when to pay the (potentially network-bound) load cost — typically
    once per dataset.
    """

    def __init__(self, cfg: MultimodalPreprocessorConfig, hf_tokenizer):
        self.cfg = cfg
        self.tokenizer = hf_tokenizer  # PreTrainedTokenizerBase
        self.multicrop: MultiCropPreprocessor = cfg.multicrop.build(cfg.tokenizer)

        # Cache the base EOS id; we'll use it for loss-masking when present.
        self._eos_id: Optional[int] = cfg.tokenizer.base.eos_token_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> np.ndarray:
        if not text:
            return np.empty((0,), dtype=np.int64)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return np.asarray(ids, dtype=np.int64)

    def _split_prompt_template(self) -> Tuple[str, str]:
        """Split ``prompt_template`` at the ``{image}`` placeholder."""
        tmpl = self.cfg.prompt_template
        if "{image}" not in tmpl:
            raise ValueError(
                "prompt_template must contain a single '{image}' placeholder; got: " f"{tmpl!r}"
            )
        pre, post = tmpl.split("{image}", 1)
        return pre, post

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        prompt: str,
        response: str,
        image: Optional[Any] = None,
    ) -> Dict[str, np.ndarray]:
        """Build the per-example dict.

        :param prompt: User-side text.
        :param response: Assistant-side text (the supervised target).
        :param image: A PIL image or HWC ``np.ndarray`` (uint8 or float32).
            Pass ``None`` for text-only examples — the ``{image}`` placeholder
            is stripped from the prompt template.
        :returns: Dict with keys ``input_tokens``, ``loss_masks``, ``images``,
            ``pooled_patches_idx``. For text-only examples the two image fields
            are zero-sized arrays so a collator can still stack them.
        """
        cfg = self.cfg

        # Multi-crop preprocessing (if any image).
        if image is not None:
            mc_out = self.multicrop(image)
            image_tokens = mc_out.image_tokens
            images = mc_out.images
            pooled_patches_idx = mc_out.pooled_patches_idx
        else:
            image_tokens = np.empty((0,), dtype=np.int64)
            # Zero-sized arrays so the collator can still concat across the batch.
            patch_size = cfg.multicrop.image_preprocessor.patch_size
            patch_dim = 3 * patch_size * patch_size
            pool_size = cfg.multicrop.pool_h * cfg.multicrop.pool_w
            images = np.zeros((0, 0, patch_dim), dtype=np.float32)
            pooled_patches_idx = np.zeros((0, pool_size), dtype=np.int64)

        # Compose the text portion. We split the prompt template at {image} so
        # the image-token sequence (which is already token IDs) is spliced in
        # between two tokenized text chunks.
        pre_tmpl, post_tmpl = self._split_prompt_template()
        try:
            pre_text = pre_tmpl.format(prompt=prompt)
            post_text = post_tmpl.format(prompt=prompt)
        except KeyError as e:
            raise ValueError(f"prompt_template references unknown field: {e}") from e
        if image is None:
            # Strip the dangling newline that the {image} placeholder used to anchor.
            post_text = post_text.lstrip("\n")

        resp_text = cfg.response_template.format(response=response)

        pre_ids = self._encode(pre_text)
        post_ids = self._encode(post_text)
        resp_ids = self._encode(resp_text)

        if cfg.add_eos and self._eos_id is not None:
            resp_ids = np.concatenate([resp_ids, np.asarray([self._eos_id], dtype=np.int64)])

        # Concatenate everything.
        input_tokens = np.concatenate([pre_ids, image_tokens, post_ids, resp_ids])
        loss_masks = np.concatenate(
            [
                np.zeros(len(pre_ids) + len(image_tokens) + len(post_ids), dtype=np.float32),
                np.ones(len(resp_ids), dtype=np.float32),
            ]
        )

        # Truncate from the right; the prompt + image tokens are load-bearing.
        if len(input_tokens) > cfg.max_sequence_length:
            input_tokens = input_tokens[: cfg.max_sequence_length]
            loss_masks = loss_masks[: cfg.max_sequence_length]

        return {
            "input_tokens": input_tokens,
            "loss_masks": loss_masks,
            "images": images,
            "pooled_patches_idx": pooled_patches_idx,
        }
