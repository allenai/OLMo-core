"""
Multimodal tokenizer configuration.

Wraps an existing :class:`~olmo_core.data.TokenizerConfig` and reserves six
new token IDs for the image-token layout used by Molmo-style VLMs.
"""

from dataclasses import dataclass
from typing import List, Tuple

from ...config import Config
from ..tokenizer import TokenizerConfig

__all__ = [
    "MultimodalTokenizerConfig",
    "IMAGE_SPECIAL_TOKENS",
]

#: Order-sensitive list of image special tokens added to the base tokenizer.
#: The list order determines the ID assignment: each token gets
#: ``base.vocab_size + i`` for its position ``i``.
IMAGE_SPECIAL_TOKENS: Tuple[str, ...] = (
    "<im_start>",
    "<im_end>",
    "<im_patch>",
    "<im_col>",
    "<low_res_im_start>",
    "<im_low>",
)


@dataclass
class MultimodalTokenizerConfig(Config):
    """
    A tokenizer config that extends a base text tokenizer with six image
    special tokens.

    Token IDs are assigned deterministically starting at ``base.vocab_size``
    in the order of :data:`IMAGE_SPECIAL_TOKENS`. Use
    :meth:`load_hf_tokenizer` to obtain an ``AutoTokenizer`` with the
    matching special tokens registered.
    """

    base: TokenizerConfig
    """The underlying text tokenizer (e.g. dolma2)."""

    @classmethod
    def dolma2(cls) -> "MultimodalTokenizerConfig":
        """Multimodal tokenizer built on ``allenai/dolma2-tokenizer``."""
        return cls(base=TokenizerConfig.dolma2())

    # ------------------------------------------------------------------
    # Token IDs
    # ------------------------------------------------------------------

    @property
    def image_start_id(self) -> int:
        """Token ID for ``<im_start>``."""
        return self.base.vocab_size + IMAGE_SPECIAL_TOKENS.index("<im_start>")

    @property
    def image_end_id(self) -> int:
        """Token ID for ``<im_end>``."""
        return self.base.vocab_size + IMAGE_SPECIAL_TOKENS.index("<im_end>")

    @property
    def image_patch_id(self) -> int:
        """Token ID for ``<im_patch>`` (the splice placeholder)."""
        return self.base.vocab_size + IMAGE_SPECIAL_TOKENS.index("<im_patch>")

    @property
    def image_col_id(self) -> int:
        """Token ID for ``<im_col>`` (row break inside a crop)."""
        return self.base.vocab_size + IMAGE_SPECIAL_TOKENS.index("<im_col>")

    @property
    def low_res_image_start_id(self) -> int:
        """Token ID for ``<low_res_im_start>`` (opens the global low-res view)."""
        return self.base.vocab_size + IMAGE_SPECIAL_TOKENS.index("<low_res_im_start>")

    @property
    def image_low_id(self) -> int:
        """Token ID for ``<im_low>`` (low-res patch placeholder)."""
        return self.base.vocab_size + IMAGE_SPECIAL_TOKENS.index("<im_low>")

    @property
    def special_token_ids(self) -> List[int]:
        """All six image special token IDs in registration order."""
        return [self.base.vocab_size + i for i in range(len(IMAGE_SPECIAL_TOKENS))]

    # ------------------------------------------------------------------
    # Vocab sizing
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total vocab size including image special tokens."""
        return self.base.vocab_size + len(IMAGE_SPECIAL_TOKENS)

    def padded_vocab_size(self, pad_multiple: int = 128) -> int:
        """Vocab size rounded up to a multiple of ``pad_multiple``.

        Use this for the LM's ``vocab_size`` field so the embedding rows for
        image tokens line up with concrete indices.
        """
        return pad_multiple * ((self.vocab_size + pad_multiple - 1) // pad_multiple)

    # ------------------------------------------------------------------
    # HF tokenizer
    # ------------------------------------------------------------------

    def load_hf_tokenizer(self):
        """Load the underlying HuggingFace tokenizer with image tokens added.

        Adds tokens via ``add_special_tokens``; their assigned IDs are
        guaranteed to match :attr:`special_token_ids` because HuggingFace
        appends new tokens consecutively starting at the current vocab size.

        When ``HF_HUB_OFFLINE`` is set, falls back to ``local_files_only=True``
        so the tokenizer is loaded from the HF cache (``HF_HOME``) without
        the revision API check that ``HF_HUB_OFFLINE`` itself refuses.

        :returns: A ``PreTrainedTokenizerBase`` instance.
        """
        import os

        from transformers import AutoTokenizer

        if self.base.identifier is None:
            raise ValueError(
                "MultimodalTokenizerConfig.base.identifier must be set to load an HF tokenizer"
            )
        kwargs = {}
        if os.environ.get("HF_HUB_OFFLINE", "").strip() in {"1", "true", "True"}:
            kwargs["local_files_only"] = True
        tok = AutoTokenizer.from_pretrained(self.base.identifier, **kwargs)
        added = tok.add_special_tokens({"additional_special_tokens": list(IMAGE_SPECIAL_TOKENS)})
        if added != len(IMAGE_SPECIAL_TOKENS):
            raise RuntimeError(
                f"Expected to add {len(IMAGE_SPECIAL_TOKENS)} image tokens but HF added {added}; "
                f"the base tokenizer may already register one of them."
            )
        # Sanity: the assigned IDs must match what our properties report.
        for tok_str, expected_id in zip(IMAGE_SPECIAL_TOKENS, self.special_token_ids):
            assigned = tok.convert_tokens_to_ids(tok_str)
            if assigned != expected_id:
                raise RuntimeError(
                    f"HF assigned ID {assigned} for '{tok_str}' but config expects {expected_id}; "
                    f"base.vocab_size may be inconsistent with the HF tokenizer's actual size."
                )
        return tok
