from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoTokenizer

from ..config import Config, StrEnum

__all__ = [
    "TokenizerConfig",
    "ByteTokenizerConfig",
    "TokenizerName",
]


class TokenizerName(StrEnum):
    """
    An enumeration of tokenizer identifiers commonly used OLMo researchers.
    """

    dolma2 = "allenai/dolma2-tokenizer"
    """
    The dolma2 tokenizer.
    """

    gpt_neox_olmo_dolma_v1_5 = "allenai/gpt-neox-olmo-dolma-v1_5"
    """
    A modified GPT NeoX tokenizer.
    """

    gpt2 = "gpt2"
    """
    The base GPT2 tokenizer.
    """


@dataclass
class TokenizerConfig(Config):
    """
    A configuration class that represents a tokenizer.
    """

    vocab_size: int
    """
    The vocab size.
    """

    eos_token_id: int
    """
    The end-of-sentence token ID.
    """

    pad_token_id: int
    """
    The padding token ID.
    """

    bos_token_id: Optional[int] = None
    """
    The begin-of-sentence token ID.
    """

    identifier: Optional[str] = None
    """
    The identifier of the tokenizer. Could be a path or HuggingFace identifier.
    """

    def padded_vocab_size(self, pad_multiple: int = 128) -> int:
        """
        Returns the vocab size padded to be a multiple of ``pad_multiple``.
        This is useful to set model embeddings to this number to increase throughput.
        """
        return pad_multiple * ((self.vocab_size + pad_multiple - 1) // pad_multiple)

    @classmethod
    def dolma2(cls) -> "TokenizerConfig":
        """
        Get a :data:`~TokenizerName.dolma2` tokenizer config.
        """
        return cls(
            vocab_size=100278,
            eos_token_id=100257,
            pad_token_id=100277,
            identifier=TokenizerName.dolma2,
        )

    @classmethod
    def gpt_neox_olmo_dolma_v1_5(cls) -> "TokenizerConfig":
        """
        Get a :data:`~TokenizerName.gpt_neox_olmo_dolma_v1_5` tokenizer config.
        """
        return cls(
            vocab_size=50280,
            eos_token_id=50279,
            pad_token_id=1,
            identifier=TokenizerName.gpt_neox_olmo_dolma_v1_5,
        )

    @classmethod
    def gpt2(cls) -> "TokenizerConfig":
        """
        Get a :data:`~TokenizerName.gpt2` tokenizer config.
        """
        return cls(
            vocab_size=50257,
            eos_token_id=50256,
            bos_token_id=50256,
            pad_token_id=50256,
            identifier=TokenizerName.gpt2,
        )

    @classmethod
    def from_hf(cls, identifier: str) -> "TokenizerConfig":
        """
        Initialize a tokenizer config from a model on HuggingFace.

        :param identifier: The HF model identifier, e.g. "meta-llama/Llama-3.2-1B".
        """
        import json

        from cached_path import cached_path

        try:
            config_path = cached_path(f"hf://{identifier}/config.json")
        except FileNotFoundError:
            config_path = cached_path(f"hf://{identifier}/tokenizer_config.json")

        with config_path.open() as f:
            config = json.load(f)

        return cls(
            vocab_size=config["vocab_size"],
            eos_token_id=config["eos_token_id"],
            pad_token_id=config.get("pad_token_id", config["eos_token_id"]),
            bos_token_id=config.get("bos_token_id"),
            identifier=identifier,
        )


@dataclass
class ByteTokenizerConfig(TokenizerConfig):
    special_tokens: list[str] = field(default_factory=lambda: [])
    original_identifier: Optional[str] = None

    @classmethod
    def from_tokenizer_config(cls, tokenizer_config):
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.identifier)

        # all_special_tokens does not contain added special tokens (e.g. <|endofprompt|> for OLMo-2)
        # this is an attempt to include all of them, but it may not be exhaustive.
        special_tokens = sorted(set(
            hf_tokenizer.all_special_tokens
            + list(hf_tokenizer.get_added_vocab().keys())  # type: ignore
        ))

        return cls(
            vocab_size=256 + len(special_tokens),
            special_tokens=special_tokens,
            # convention: 256 bytes first, then special tokens, and no bos (as in OLMo)
            pad_token_id=256 + special_tokens.index(hf_tokenizer.pad_token),
            eos_token_id=256 + special_tokens.index(hf_tokenizer.eos_token),
            original_identifier=tokenizer_config.identifier,
        )