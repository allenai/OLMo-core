from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional
from transformers import AutoTokenizer

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
class ByteTokenizerConfig:
    vocab_size: int
    bos_token_id: int
    pad_token_id: int
    eos_token_id: int
    bpe_token_end_id: int
    special_tokens: list[str] = field(default_factory=lambda: [])
    special_tokens_first: bool = True
    original_identifier: Optional[str] = None


    @classmethod
    def bolmo(cls) -> "ByteTokenizerConfig":
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
        return self.config.bpe_token_end_id 

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
