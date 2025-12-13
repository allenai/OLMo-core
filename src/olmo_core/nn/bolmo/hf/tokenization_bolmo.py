from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

# Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
# Also implemented in https://docs.rs/tokenizers/latest/src/tokenizers/pre_tokenizers/byte_level.rs.html#13-39
_CHARS_TO_BYTES = {
    "Ā": 0, "ā": 1, "Ă": 2, "ă": 3, "Ą": 4, "ą": 5, "Ć": 6, "ć": 7, "Ĉ": 8,
    "ĉ": 9, "Ċ": 10, "ċ": 11, "Č": 12, "č": 13, "Ď": 14, "ď": 15, "Đ": 16,
    "đ": 17, "Ē": 18, "ē": 19, "Ĕ": 20, "ĕ": 21, "Ė": 22, "ė": 23, "Ę": 24,
    "ę": 25, "Ě": 26, "ě": 27, "Ĝ": 28, "ĝ": 29, "Ğ": 30, "ğ": 31, "Ġ": 32,
    "!": 33, '"': 34, "#": 35, "$": 36, "%": 37, "&": 38, "'": 39, "(": 40,
    ")": 41, "*": 42, "+": 43, ",": 44, "-": 45, ".": 46, "/": 47, "0": 48,
    "1": 49, "2": 50, "3": 51, "4": 52, "5": 53, "6": 54, "7": 55, "8": 56,
    "9": 57, ":": 58, ";": 59, "<": 60, "=": 61, ">": 62, "?": 63, "@": 64, 
    "A": 65, "B": 66, "C": 67, "D": 68, "E": 69, "F": 70, "G": 71, "H": 72,
    "I": 73, "J": 74, "K": 75, "L": 76, "M": 77, "N": 78, "O": 79, "P": 80,
    "Q": 81, "R": 82, "S": 83, "T": 84, "U": 85, "V": 86, "W": 87, "X": 88,
    "Y": 89, "Z": 90, "[": 91, "\\": 92, "]": 93, "^": 94, "_": 95, "`": 96, 
    "a": 97, "b": 98, "c": 99, "d": 100, "e": 101, "f": 102, "g": 103,
    "h": 104, "i": 105, "j": 106, "k": 107, "l": 108, "m": 109, "n": 110,
    "o": 111, "p": 112, "q": 113, "r": 114, "s": 115, "t": 116, "u": 117,
    "v": 118, "w": 119, "x": 120, "y": 121, "z": 122, "{": 123, "|": 124,
    "}": 125, "~": 126, "ġ": 127, "Ģ": 128, "ģ": 129, "Ĥ": 130, "ĥ": 131,
    "Ħ": 132, "ħ": 133, "Ĩ": 134, "ĩ": 135, "Ī": 136, "ī": 137, "Ĭ": 138,
    "ĭ": 139, "Į": 140, "į": 141, "İ": 142, "ı": 143, "Ĳ": 144, "ĳ": 145,
    "Ĵ": 146, "ĵ": 147, "Ķ": 148, "ķ": 149, "ĸ": 150, "Ĺ": 151, "ĺ": 152,
    "Ļ": 153, "ļ": 154, "Ľ": 155, "ľ": 156, "Ŀ": 157, "ŀ": 158, "Ł": 159,
    "ł": 160, "¡": 161, "¢": 162, "£": 163, "¤": 164, "¥": 165, "¦": 166,
    "§": 167, "¨": 168, "©": 169, "ª": 170, "«": 171, "¬": 172, "Ń": 173,
    "®": 174, "¯": 175, "°": 176, "±": 177, "²": 178, "³": 179, "´": 180,
    "µ": 181, "¶": 182, "·": 183, "¸": 184, "¹": 185, "º": 186, "»": 187,
    "¼": 188, "½": 189, "¾": 190, "¿": 191, "À": 192, "Á": 193, "Â": 194,
    "Ã": 195, "Ä": 196, "Å": 197, "Æ": 198, "Ç": 199, "È": 200, "É": 201,
    "Ê": 202, "Ë": 203, "Ì": 204, "Í": 205, "Î": 206, "Ï": 207, "Ð": 208,
    "Ñ": 209, "Ò": 210, "Ó": 211, "Ô": 212, "Õ": 213, "Ö": 214, "×": 215,
    "Ø": 216, "Ù": 217, "Ú": 218, "Û": 219, "Ü": 220, "Ý": 221, "Þ": 222,
    "ß": 223, "à": 224, "á": 225, "â": 226, "ã": 227, "ä": 228, "å": 229,
    "æ": 230, "ç": 231, "è": 232, "é": 233, "ê": 234, "ë": 235, "ì": 236,
    "í": 237, "î": 238, "ï": 239, "ð": 240, "ñ": 241, "ò": 242, "ó": 243,
    "ô": 244, "õ": 245, "ö": 246, "÷": 247, "ø": 248, "ù": 249, "ú": 250,
    "û": 251, "ü": 252, "ý": 253, "þ": 254, "ÿ": 255,
}
_BYTES_TO_CHARS = {v: k for k, v in _CHARS_TO_BYTES.items()}

def _bytes_to_chars(byte_sequence: bytes) -> str:
    return "".join(_BYTES_TO_CHARS[byte] for byte in byte_sequence)

def _chars_to_bytes(char_sequence: str) -> list:
    return list(bytes(_CHARS_TO_BYTES[char] for char in char_sequence))

@dataclass
class BolmoTokenizerConfig:
    vocab_size: int
    bos_token_id: int
    pad_token_id: int
    eos_token_id: int
    bpe_token_end_id: int
    special_tokens: list[str] = field(default_factory=lambda: [])
    special_tokens_first: bool = True
    original_identifier: Optional[str] = None


    @classmethod
    def bolmo(cls) -> "BolmoTokenizerConfig":
        special_tokens = [
            "<pad>",
            "<bos>",
            "<eos>",
            "<bpe_token_end>",
        ]

        return cls(
            # *2 to accomodate fused boundary tokens
            vocab_size=(len(special_tokens) + 256) * 2,
            special_tokens=special_tokens,
            bos_token_id=special_tokens.index("<bos>"),
            pad_token_id=special_tokens.index("<pad>"),
            eos_token_id=special_tokens.index("<bos>"),
            bpe_token_end_id=special_tokens.index("<bpe_token_end>"),
            original_identifier="allenai/dolma2-tokenizer",
        )
    
    def build(self):
        return BolmoTokenizer(tokenizer_config=self)


class BolmoTokenizer(PreTrainedTokenizer):
    TOKEN_ID_KEY = -1

    def __init__(self, **kwargs):
        tokenizer_config = kwargs.pop("tokenizer_config", BolmoTokenizerConfig.bolmo())

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
                byte_sequence = [self.offset + i for i in _chars_to_bytes(key)]

            assert self.byte_sequences.get(value) is None
            self.byte_sequences[value] = byte_sequence

        self.byte_trie = {}

        for token_id, byte_sequence in self.byte_sequences.items():
            current_dict = self.byte_trie
            for byte in byte_sequence[::-1]: # retrieved from the back so store in reverse order
                if byte not in current_dict:
                    current_dict[byte] = {}
                current_dict = current_dict[byte]
            current_dict[BolmoTokenizer.TOKEN_ID_KEY] = token_id

        self.add_bos_token = True
        self.add_eos_token = False

        super().__init__(
            bos_token=self.config.special_tokens[self.config.bos_token_id],
            eos_token=self.config.special_tokens[self.config.eos_token_id],
            pad_token=self.config.special_tokens[self.config.pad_token_id],
            extra_ids=0,
        )

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
        return self.config.bpe_token_end_id 

    @property
    def vocab_size(self):
        return self.config.vocab_size

    def _convert_id_to_token(self, index):
        if index < self.offset:
            return self.config.special_tokens[index - self.special_tokens_offset]

        if index >= self.offset + 256 and index < self.offset * 2 + 256:
            # special token with fused boundary
            return self.config.special_tokens[index - self.offset - 256] + "b"

        return _BYTES_TO_CHARS[index - self.offset - 256 - self.offset] + "b" if index >= self.offset + 256 else _BYTES_TO_CHARS[index - self.offset]

    def _convert_token_to_id(self, token):
        if token in self.config.special_tokens:
            return self.config.special_tokens.index(token)

        if token in [x + "b" for x in self.config.special_tokens]:
            # special token with fused boundary
            return 256 + self.config.special_tokens.index(token[:-1])

        if len(token) > 1 and token[-1] == "b":
            return self.offset + 256 + _CHARS_TO_BYTES[token[0]]
        else:
            return self.offset + _CHARS_TO_BYTES[token]

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

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

                if byte >= self.offset + 256:
                    # ignore fused boundary
                    byte -= self.offset + 256

                try:
                    current_dict = current_dict[byte]
                    if BolmoTokenizer.TOKEN_ID_KEY in current_dict:
                        current_expansion = current_dict[BolmoTokenizer.TOKEN_ID_KEY]
                except KeyError:
                    assert current_expansion is not None
                    break

            expanded_ids.append(current_expansion)

        return expanded_ids

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None, already_has_special_tokens: bool = False
    ) -> list[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
    ) -> list[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:
        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```
        if token_ids_1 is None, only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        tokens = self.convert_ids_to_tokens(self._bolmo_encode(text))
        return tokens

    def _patch_ids_to_byte_ids(self, input_ids: list[int]):
        return [byte_token_id for token_id in input_ids for byte_token_id in self.byte_sequences[token_id]]

    def _bolmo_encode(self, string: str, add_special_tokens=False):
        input_ids = self.hf_tokenizer.encode(string, add_special_tokens=add_special_tokens)
        return self._patch_ids_to_byte_ids(input_ids)

    def _bolmo_decode(self, tokens: list[int]) -> str:
        return self._decode_to_bytes(tokens).decode("utf-8", errors="replace")

    def _decode_to_bytes(self, tokens: list[int]) -> bytes:
        tokens_without_boundary = []
        for token in tokens:
            if token >= (self.offset + 256):
                token -= self.offset + 256

            tokens_without_boundary.append(token)

        utf8_bytes = [min(token - self.offset, 255) for token in tokens_without_boundary if token >= self.offset]
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

            token_byte_tokens = self._patch_ids_to_byte_ids([int(token)])

            if strip_pad and all(t == self.pad_token_id for t in token_byte_tokens):
                # skip padding tokens
                continue

            patch_lengths.append(len(token_byte_tokens))
            byte_tokens.extend(token_byte_tokens)

        return byte_tokens, patch_lengths

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()