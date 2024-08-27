from dataclasses import dataclass
from typing import Optional

from ..config import Config, StrEnum


class TokenizerName(StrEnum):
    """
    An enumeration of supported tokenizer names.
    """

    dolma2 = "allenai/dolma2-tokenizer"
    gpt_neox_olmo_dolma_v1_5 = "allenai/gpt-neox-olmo-dolma-v1_5"
    gpt2 = "gpt2"


@dataclass
class TokenizerConfig(Config):
    """
    A configuration class that represents a tokenizer.
    """

    vocab_size: int
    eos_token_id: int
    pad_token_id: int
    bos_token_id: Optional[int] = None
    identifier: Optional[TokenizerName] = None

    def padded_vocab_size(self, pad_multiple: int = 128) -> int:
        """
        Returns the vocab size padded to be a multiple of ``pad_multiple``.
        This is useful to set model embeddings to this number to increase throughput.
        """
        return pad_multiple * ((self.vocab_size + pad_multiple - 1) // pad_multiple)

    @classmethod
    def dolma2(cls) -> "TokenizerConfig":
        return cls(
            vocab_size=100278,
            eos_token_id=100257,
            pad_token_id=100277,
            identifier=TokenizerName.dolma2,
        )

    @classmethod
    def gpt_neox_olmo_dolma_v1_5(cls) -> "TokenizerConfig":
        return cls(
            vocab_size=50280,
            eos_token_id=50279,
            pad_token_id=1,
            identifier=TokenizerName.gpt_neox_olmo_dolma_v1_5,
        )

    @classmethod
    def gpt2(cls) -> "TokenizerConfig":
        return cls(
            vocab_size=50280,
            eos_token_id=50256,
            pad_token_id=50256,
            identifier=TokenizerName.gpt2,
        )
