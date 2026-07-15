from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence

from .util import get_data_path, is_data_file


class Tokenizer(metaclass=ABCMeta):
    """
    An abstract base class for tokenizer's used by this library.

    .. tip::
        Use the :class:`TokenizerConfig` to build tokenizer objects.
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """
        The tokenizer's vocabulary size.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """
        The tokenizer's padding token ID.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """
        The tokenizer's end-of-sentence token ID.
        """
        raise NotImplementedError

    @property
    def bos_token_id(self) -> Optional[int]:
        """
        The tokenizer's begin-of-sentence token ID.
        """
        return None

    def encode(self, input: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a string into token IDs.

        :param input: The string to encode.
        :param add_special_tokens: Include special tokens (like end-of-sentence tokens)
            in the encoding.
        """
        return self.encode_batch([input], add_special_tokens=add_special_tokens)[0]

    @abstractmethod
    def encode_batch(
        self, inputs: Sequence[str], add_special_tokens: bool = True
    ) -> List[List[int]]:
        """
        Encode a batch of inputs into token IDs.

        :param inputs: The strings to encode.
        :param add_special_tokens: Include special tokens (like end-of-sentence tokens)
            in the encodings.
        """
        raise NotImplementedError

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a sequence of token IDs into a string.

        :param token_ids: The encoding to decode.
        :param skip_special_tokens: Exclude special tokens in the decoded string.
        """
        return self.decode_batch([token_ids], skip_special_tokens=skip_special_tokens)[0]

    @abstractmethod
    def decode_batch(
        self, batch_token_ids: Sequence[Sequence[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode a batch of token IDs into a strings.

        :param batch_token_ids: The batch encoding to decode.
        :param skip_special_tokens: Exclude special tokens in the decoded strings.
        """
        raise NotImplementedError


class HFTokenizer(Tokenizer):
    """
    A :class:`Tokenizer` implementation that uses the `tokenizers <https://github.com/huggingface/tokenizers>`_
    library from HuggingFace.
    """

    def __init__(
        self,
        identifier: str,
        *,
        pad_token_id: int,
        eos_token_id: int,
        bos_token_id: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ):
        from tokenizers import Tokenizer as BaseTokenizer

        base_tokenizer: BaseTokenizer
        if Path(identifier).is_file():
            base_tokenizer = BaseTokenizer.from_file(identifier)
        elif is_data_file(identifier):
            with get_data_path(identifier) as path:
                base_tokenizer = BaseTokenizer.from_file(str(path))
        else:
            base_tokenizer = BaseTokenizer.from_pretrained(identifier)

        self._base_tokenizer = base_tokenizer
        self._vocab_size = vocab_size
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id
        self._bos_token_id = bos_token_id

    @property
    def vocab_size(self) -> int:
        return self._vocab_size or self._base_tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._bos_token_id

    def _add_special_tokens(self, token_ids: List[int]) -> List[int]:
        if len(token_ids) == 0 or token_ids[-1] != self.eos_token_id:
            token_ids.append(self.eos_token_id)
        if self.bos_token_id is not None and token_ids[0] != self.bos_token_id:
            token_ids.insert(0, self.bos_token_id)
        return token_ids

    def encode_batch(
        self, inputs: Sequence[str], add_special_tokens: bool = True
    ) -> List[List[int]]:
        batch_encoding = self._base_tokenizer.encode_batch(
            inputs, add_special_tokens=add_special_tokens
        )
        if add_special_tokens:
            return [self._add_special_tokens(encoding.ids) for encoding in batch_encoding]
        else:
            return [encoding.ids for encoding in batch_encoding]

    def decode_batch(
        self, batch_token_ids: Sequence[Sequence[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        return self._base_tokenizer.decode_batch(
            batch_token_ids, skip_special_tokens=skip_special_tokens
        )
