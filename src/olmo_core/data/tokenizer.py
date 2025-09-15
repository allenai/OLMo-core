from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoTokenizer
import numpy as np
from functools import lru_cache

from ..config import Config, StrEnum
from ..nn.blt import utils as blt_utils

__all__ = [
    "TokenizerConfig",
    "ByteTokenizerConfig",
    "ByteTokenizer",
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

    dolma2_sigdig = "allenai/dolma2-tokenizer-sigdig"
    """
    The R2L dolma2 tokenizer.
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
    def dolma2_sigdig(cls) -> "TokenizerConfig":
        """
        Get a :data:`~TokenizerName.dolma2_sigdig` tokenizer config.
        """
        return cls(
            vocab_size=100278,
            eos_token_id=100257,
            pad_token_id=100277,
            bos_token_id=100257,
            identifier=TokenizerName.dolma2_sigdig,
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
    special_tokens_first: bool = True
    original_identifier: Optional[str] = None
    bpe_token_end_id: Optional[int] = None

    @classmethod
    def from_tokenizer_config(cls, tokenizer_config):
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.identifier)

        # all_special_tokens does not contain added special tokens (e.g. <|endofprompt|> for OLMo-2)
        # this is an attempt to include all of them, but it may not be exhaustive.
        special_tokens = sorted(set(
            hf_tokenizer.all_special_tokens
            + list(hf_tokenizer.get_added_vocab().keys())  # type: ignore
        ))

        # enforce special token
        bos_token = hf_tokenizer.bos_token if hf_tokenizer.bos_token is not None else "<|bos|>"

        if bos_token not in special_tokens:
            special_tokens.insert(0, bos_token)

        return cls(
            vocab_size=len(special_tokens) + 256,
            special_tokens=special_tokens,
            # convention: special tokens first, then 256 bytes (as in BLT)
            bos_token_id=special_tokens.index(bos_token),
            pad_token_id=special_tokens.index(hf_tokenizer.pad_token),
            eos_token_id=special_tokens.index(hf_tokenizer.eos_token),
            original_identifier=tokenizer_config.identifier,
        )

    @classmethod
    def blt(cls) -> "ByteTokenizerConfig":
        special_tokens = [
            "<pad>",
            "<bos>",
            "<eos>",
            "<bpe_token_end>", # reserved in BLT tokenizer, but unused in released checkpoints
        ]

        return cls(
            vocab_size=len(special_tokens) + 256,
            special_tokens=special_tokens,
            bos_token_id=special_tokens.index("<bos>"),
            pad_token_id=special_tokens.index("<pad>"),
            eos_token_id=special_tokens.index("<bos>"), # DIVERGENCE FROM BLT. set bos and eos to the same
            bpe_token_end_id=special_tokens.index("<bpe_token_end>"),
            # slightly hacky, but this must match the dataset tokenizer, so dolma2
            original_identifier=TokenizerConfig.dolma2().identifier,
        )
    
    @classmethod
    def hnet(cls) -> "ByteTokenizerConfig":
        special_tokens = [
            "<bos>",
            "<eos>",
        ]

        return cls(
            vocab_size=256,
            special_tokens=special_tokens,
            special_tokens_first=False,
            bos_token_id=254,
            pad_token_id=255,
            eos_token_id=255,
            # slightly hacky, but this must match the dataset tokenizer, so dolma2
            original_identifier=TokenizerConfig.dolma2().identifier,
        )
    
    def build(self):
        return ByteTokenizer(self)


class ByteTokenizer:
    TOKEN_ID_KEY = -1

    def __init__(self, tokenizer_config: ByteTokenizerConfig):
        self.config = tokenizer_config
        self.hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.original_identifier)
        if self.config.special_tokens_first:
            self.offset = len(tokenizer_config.special_tokens)
            self.special_tokens_offset = 0
        else:
            self.offset = 0
            self.special_tokens_offset = self.config.vocab_size - len(tokenizer_config.special_tokens)

        self.byte_sequences = {}
    
        for key, value in self.hf_tokenizer.get_vocab().items():
            if key in self.config.special_tokens:
                byte_sequence = [self.special_tokens_offset + self.config.special_tokens.index(key)]
            elif value == self.hf_tokenizer.eos_token_id and self.eos_token_id is not None:
                byte_sequence = [self.eos_token_id]
            elif value == self.hf_tokenizer.bos_token_id and self.bos_token_id is not None:
                byte_sequence = [self.bos_token_id]
            elif value == self.hf_tokenizer.pad_token_id and self.pad_token_id is not None:
                byte_sequence = [self.pad_token_id]
            else:
                byte_sequence = [self.offset + i for i in blt_utils.chars_to_bytes(key)]

            assert self.byte_sequences.get(value) is None
            self.byte_sequences[value] = byte_sequence

        self.byte_trie = {}

        for token_id, byte_sequence in self.byte_sequences.items():
            current_dict = self.byte_trie
            for byte in byte_sequence[::-1]: # retrieved from the back so store in reverse order
                if byte not in current_dict:
                    current_dict[byte] = {}
                current_dict = current_dict[byte]
            current_dict[ByteTokenizer.TOKEN_ID_KEY] = token_id

    @property
    def bos_token_id(self):
        return self.config.bos_token_id

    @property
    def eos_token_id(self):
        return self.config.eos_token_id
    
    @property
    def pad_token_id(self):
        return self.config.pad_token_id

    @property
    def bpe_token_end_id(self):
        # TODO(benjaminm): backwards compat, remove once new checkpoints
        return self.config.bpe_token_end_id if self.config.bpe_token_end_id is not None else 3

    def expand_byte_ids(self, byte_ids: list[int], n_last: Optional[int] = None) -> list[int]:
        # search in the byte tree for the longest matching token at every byte position
        expanded_ids = []
        for i in range(len(byte_ids)):
            if n_last is not None and i < len(byte_ids) - n_last:
                continue

            current_dict = self.byte_trie
            current_expansion = None

            for i in range(i, -1, -1):
                byte = byte_ids[i]

                if byte == self.bpe_token_end_id:
                    # skip bpe token end markers, needed for generation
                    continue

                try:
                    current_dict = current_dict[byte]
                    if ByteTokenizer.TOKEN_ID_KEY in current_dict:
                        current_expansion = current_dict[ByteTokenizer.TOKEN_ID_KEY]
                except KeyError:
                    assert current_expansion is not None
                    break

            expanded_ids.append(current_expansion)

        return expanded_ids

    def patch_ids_to_byte_ids(self, input_ids: list[int]):
        return [byte_token_id for token_id in input_ids for byte_token_id in self.byte_sequences[token_id]]

    def encode(self, string: str, add_special_tokens=False):
        input_ids = self.hf_tokenizer.encode(string, add_special_tokens=add_special_tokens)
        return self.patch_ids_to_byte_ids(input_ids)

    def decode(self, tokens: list[int]) -> str:
        return self.decode_to_bytes(tokens).decode("utf-8", errors="replace")

    def decode_to_bytes(self, tokens: list[int]) -> bytes:
        utf8_bytes = [min(tokens - self.offset, 255) for tokens in tokens if tokens >= self.offset]
        return bytes(utf8_bytes)

    def get_tokens_and_patch_lengths(self, original_input_ids: list[int], add_bos=False, strip_pad=False, skip_last=False):
        if add_bos and self.bos_token_id is not None:
            byte_tokens = [self.bos_token_id]
            patch_lengths = [1]
        else:
            byte_tokens = []
            patch_lengths = []

        for idx, token in enumerate(original_input_ids):
            # optionally skip last token to keep the length the same if add_bos=True
            if skip_last and idx == len(original_input_ids) - 1:
                break

            token_byte_tokens = self.patch_ids_to_byte_ids([int(token)])

            if strip_pad and all(t == self.pad_token_id for t in token_byte_tokens):
                # skip padding tokens
                continue

            patch_lengths.append(len(token_byte_tokens))
            byte_tokens.extend(token_byte_tokens)

        return byte_tokens, patch_lengths

    @lru_cache(maxsize=1024)
    def _is_spacelike(self, token_id: int) -> bool:
        """
        Check if a token ID is spacelike.
        """
        byte = token_id - self.offset
        # see https://github.com/kjslag/spacebyte/blob/321111315c92bce0bc2f9f1630cb0bc82b897c57/spacebyte.py#L137-L145.
        is_spacelike = (
            (byte < ord('0')) |
            ((ord('9') < byte) & (byte < ord('A'))) | 
            ((ord('Z') < byte) & (byte < ord('a'))) |
            ((ord('z') < byte) & (byte < 0b1000_0000)) |
            (0b1100_0000 <= byte)
        )
        return is_spacelike

    @lru_cache(maxsize=1024)
    def _is_strict_spacelike(self, token_id: int) -> bool:
        """
        Check if a token ID is strictly spacelike (only space, tab, newline, carriage return).
        """
        byte = token_id - self.offset
        return byte in {ord(' '), ord('\t'), ord('\n'), ord('\r')}

    def get_space_patch_lengths(self, input_ids: list[int], max_patch_length: int = 16, kind: str = "strict_end_before_space") -> list[int]:
        patch_lengths = []
        current_length = 0

        special_tokens = {self.bos_token_id, self.eos_token_id, self.pad_token_id}

        all_spacelike = [self._is_spacelike(token) for token in input_ids]

        if kind == "spacebyte":
            for token_idx, token in enumerate(input_ids):
                current_length += 1

                spacelike = all_spacelike[token_idx]
                previous_spacelike = all_spacelike[token_idx - 1] if token_idx > 0 else False

                if (not previous_spacelike and spacelike) or current_length >= max_patch_length or token in special_tokens:
                    patch_lengths.append(current_length)
                    current_length = 0
        elif kind == "spacebyte_end_before_space":
            for token_idx, token in enumerate(input_ids):
                current_length += 1

                spacelike = all_spacelike[token_idx]
                next_spacelike = all_spacelike[token_idx + 1] if token_idx < len(input_ids) - 1 else True

                if (not spacelike and next_spacelike) or current_length >= max_patch_length or token in special_tokens:
                    patch_lengths.append(current_length)
                    current_length = 0
        elif kind == "strict_end_before_space":
            all_strict_spacelike = [self._is_strict_spacelike(token) for token in input_ids]
            in_strict_prefix = True

            for token_idx, token in enumerate(input_ids):
                current_length += 1

                spacelike = all_spacelike[token_idx]
                strict_spacelike = all_strict_spacelike[token_idx]
                next_spacelike = all_spacelike[token_idx + 1] if token_idx < len(input_ids) - 1 else True
                next_strict_spacelike = all_strict_spacelike[token_idx + 1] if token_idx < len(input_ids) - 1 else True

                if not strict_spacelike:
                    in_strict_prefix = False

                if in_strict_prefix:
                    continue

                if (spacelike != next_spacelike) or (strict_spacelike != next_strict_spacelike) or current_length >= max_patch_length or token in special_tokens:
                    patch_lengths.append(current_length)
                    in_strict_prefix = True
                    current_length = 0

        if current_length > 0:
            patch_lengths.append(current_length)

        return patch_lengths
