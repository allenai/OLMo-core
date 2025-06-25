import logging
from dataclasses import dataclass
from typing import Optional

from ..config import Config, StrEnum

log = logging.getLogger(__name__)

__all__ = [
    "TokenizerConfig",
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

    olmo2instruct = "allenai/OLMo-2-1124-7B-Instruct"
    """
    The OLMo-2-1124-7B-Instruct tokenizer.
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
    def olmo2instruct(cls) -> "TokenizerConfig":
        """
        Get a :data:`~TokenizerName.dolma2` tokenizer config.
        """
        return cls(
            vocab_size=100278,
            eos_token_id=100257,
            pad_token_id=100277,
            bos_token_id=100257,
            identifier=TokenizerName.olmo2instruct,
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
    def from_hf(cls, identifier: str, url_scheme: str = "hf://") -> "TokenizerConfig":
        """
        Initialize a tokenizer config from a model on HuggingFace.

        :param identifier: The HF model identifier, e.g. "meta-llama/Llama-3.2-1B".
        :param url_scheme: The URI scheme to use to download the tokenizer config.
            Defaults to "hf://", which is the default for HuggingFace. Could be an
            alternative like "gs://" for GCS or an empty string for local files.
        """
        import json

        from cached_path import cached_path

        try:
            config_path = cached_path(f"{url_scheme}{identifier}/config.json")
        except FileNotFoundError:
            config_path = cached_path(f"{url_scheme}{identifier}/tokenizer_config.json")

        with config_path.open() as f:
            config = json.load(f)

        log.info(f"Found config: {config}")
        eos_token_id = config.get("eos_token_id")
        pad_token_id = config.get("pad_token_id")
        bos_token_id = config.get("bos_token_id")

        log.info(f"Found EOS token ID {eos_token_id}")
        log.info(f"Found PAD token ID {pad_token_id}")
        log.info(f"Found BOS token ID {bos_token_id}")

        def find_token_id_by_content(content: str) -> int | None:
            for token_id_str, token_info in config["added_tokens_decoder"].items():
                if token_info["content"] == content:
                    return int(token_id_str)
            return None

        if "added_tokens_decoder" in config:
            log.info("Found added_tokens_decoder in config")
            if "eos_token" in config and eos_token_id is None:
                eos_token_id = find_token_id_by_content(config["eos_token"])
                if eos_token_id:
                    log.info(f"Found EOS token ID {eos_token_id} for token '{config['eos_token']}'")
            if "pad_token" in config and pad_token_id is None:
                pad_token_id = find_token_id_by_content(config["pad_token"])
                if pad_token_id:
                    log.info(f"Found PAD token ID {pad_token_id} for token '{config['pad_token']}'")
            if "bos_token" in config and bos_token_id is None:
                log.info(f"Trying to find BOS token ID for token '{config['bos_token']}'")
                bos_token_id = find_token_id_by_content(config["bos_token"])
                if bos_token_id:
                    log.info(f"Found BOS token ID {bos_token_id} for token '{config['bos_token']}'")

        if eos_token_id is None:
            raise ValueError(f"EOS token ID not found for token '{config['eos_token']}'")
        if pad_token_id is None:
            raise ValueError(f"PAD token ID not found for token '{config['pad_token']}'")

        return cls(
            vocab_size=config["vocab_size"],
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            identifier=identifier,
        )
