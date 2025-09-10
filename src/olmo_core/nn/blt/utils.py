import math
import random
from functools import partial
from typing import Optional, Any

from tokenizers import pre_tokenizers
from transformers import AutoTokenizer
from transformers.models.gpt2 import tokenization_gpt2
import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask


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

def bytes_to_chars(byte_sequence: bytes) -> str:
    return "".join(_BYTES_TO_CHARS[byte] for byte in byte_sequence)

def chars_to_bytes(char_sequence: str) -> list:
    return list(bytes(_CHARS_TO_BYTES[char] for char in char_sequence))

# adapted from BLT's patch_ids_from_lengths
def lengths_to_ids(lengths, total_len):
    bs = lengths.shape[0]
    # Create a tensor of cumulative sums of the lengths
    cum_d = torch.cat(
        [
            torch.zeros(bs, 1, dtype=lengths.dtype, device=lengths.device),
            lengths.cumsum(dim=-1),
        ],
        dim=-1,
    )
    # FIXME: this seems slow since we broadcast to num_lengths x total_len? 
    patch_ids = (cum_d.unsqueeze(-1) <= torch.arange(total_len, device=cum_d.device)[None]).sum(
        dim=-2
    ) - 1
    # commented so we don't need to synchronize
    # assert not (
    #     torch.max(patch_ids) > lengths.shape[-1] or torch.min(patch_ids) < 0
    # ), f"{torch.max(patch_ids)} > {lengths.shape[-1]} or {torch.min(patch_ids)} < 0"
    return patch_ids


# from BLT's create_patch_mask_from_ids, `window` removed.
def _create_patch_mask_from_ids(
    patch_ids, num_patches, patches_as_queries=False
):
    """
    Creates a tensor of shape [bs, byte_seq_len, patch_seq_len] where each element at position (i, j, k)
    is True if the patch id at position (i, j) is less than or equal to k.
    Args:
        patch_ids (torch.Tensor): Tensor of shape [bs, byte_seq_len] containing patch ids.
        num_patches (int): Total number of patches.
        patches_as_queries (bool): If True, the patches are used as queries
    Returns:
        torch.Tensor: Tensor of shape [bs, q_len, kv_len] with the desired mask.
    """
    bs, seq_len = patch_ids.shape
    if not patches_as_queries:
        q_ids = patch_ids.unsqueeze(-1).expand(bs, seq_len, num_patches)
        kv_ids = (
            torch.arange(num_patches, device=patch_ids.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(bs, seq_len, num_patches)
        )
    else:
        kv_ids = patch_ids.unsqueeze(1).expand(bs, num_patches, seq_len)
        q_ids = (
            torch.arange(num_patches, device=patch_ids.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(bs, num_patches, seq_len)
        )
    mask = q_ids == kv_ids
    return mask


# from BLT's cross_attn_mask, `window` removed, `N` removed.
def cross_attn_mask(
    patch_ids,
    patch_lengths,
    patches_as_queries=False,
    cross_attn_k=1,
    block_mask=True,
):
    bs = patch_ids.shape[0]
    with torch.no_grad():
        # Create the patch mask
        # FIXME: this seems problematic, is there a way to avoid materializing the full [bs, byte_seq_len, patch_seq_len] mask?
        cross_mask = _create_patch_mask_from_ids(
            patch_ids,
            patch_lengths.shape[1],
            patches_as_queries=patches_as_queries,
        ).repeat_interleave(cross_attn_k, dim=1 if patches_as_queries else -1)
        q_len = patch_lengths.shape[1] * cross_attn_k if patches_as_queries else patch_ids.shape[1]
        kv_len = patch_ids.shape[1] if patches_as_queries else patch_lengths.shape[1] * cross_attn_k
        assert cross_mask.shape == (
            bs,
            q_len,
            kv_len,
        ), f"{cross_mask.shape} != {(bs, q_len, kv_len)}"

        def patch_mask(b, h, q_idx, kv_idx):
            return cross_mask[b, q_idx, kv_idx]

        block_mask = create_block_mask(
            patch_mask,
            B=bs,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            _compile=False,  # TODO(benjaminm): _compile=True causes failures in eval, why? and how large is the speed diff?
            device=patch_ids.device,
        )
        return block_mask


def log1mexp(x):
    """Computes log(1 - exp(x)) in a numerically stable way for x < 0."""
    # For x < log(0.5), use log1p(-exp(x)) directly
    # For x >= log(0.5), use log(-expm1(x)) to avoid precision issues
    log_half = -math.log(2)
    return torch.where(x < log_half, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x)))


def jsd(logprobs, logtargets, epsilon=1e-3):
    log_p = logprobs.clip(max=-epsilon)
    log_1mp = log1mexp(log_p)

    log_q = logtargets.clip(max=-epsilon)
    log_1mq = log1mexp(log_q)

    logP = torch.stack([log_p, log_1mp], dim=-1)
    logQ = torch.stack([log_q, log_1mq], dim=-1)

    stacked = torch.stack([logP, logQ], dim=0)  # [2, B, T, 2]
    # log mixture: logM = log(0.5 * (exp(logP) + exp(logQ)))
    logM = torch.logsumexp(stacked, dim=0) - math.log(2.0)

    kl_PM = (logP.exp() * (logP - logM)).sum(dim=-1)
    kl_QM = (logQ.exp() * (logQ - logM)).sum(dim=-1)

    return 0.5 * (kl_PM + kl_QM)


def binary_cross_entropy_with_logprobs(logprobs, targets, epsilon=1e-3):
    logprobs = logprobs.float().clip(max=-epsilon)
    targets = targets.float()
    return - (targets * logprobs + (1 - targets) * log1mexp(logprobs))


# patch adding dropout
def _bpe(token, self, p_bpe_dropout):
    word = tuple(token)
    pairs = tokenization_gpt2.get_pairs(word)

    if not pairs:
        return token

    while True:
        bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
        if bigram not in self.bpe_ranks:
            break

        # MODIFIED
        # with probability = dropout, skip this merge (treat as if it's not in ranks)
        if p_bpe_dropout > 0 and random.random() < p_bpe_dropout:
            # remove this pair from consideration, but continue loop
            pairs = {p for p in pairs if p != bigram}
            if not pairs:
                break
            continue

        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
            except ValueError:
                new_word.extend(word[i:])
                break
            else:
                new_word.extend(word[i:j])
                i = j

            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = tokenization_gpt2.get_pairs(word)
    word = " ".join(word)
    return word


class Noiser:
    def __init__(self, subword_tokenizer, p_ctrl_char=0.01, p_bpe_dropout=0.01):
        self.ctrl_char = chr(0xFAFAF) # in private use area
        self.ctrl_char_bytes = self.ctrl_char.encode("utf-8")
        self.subword_tokenizer = subword_tokenizer
        self.p_ctrl_char = p_ctrl_char
        self.p_bpe_dropout = p_bpe_dropout

        # for noise_ctrl_char
        self.subword_tokenizer_with_ctrl_split = AutoTokenizer.from_pretrained(subword_tokenizer.name_or_path)
        self.subword_tokenizer_with_ctrl_split.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Split(self.ctrl_char, behavior="removed"), subword_tokenizer.backend_tokenizer.pre_tokenizer]
        )

        # for noise_bpe_dropout
        self.subword_tokenizer_with_bpe_dropout = AutoTokenizer.from_pretrained(subword_tokenizer.name_or_path, use_fast=False)
        self.subword_tokenizer_with_bpe_dropout.bpe = partial(
            _bpe,
            self=self.subword_tokenizer_with_bpe_dropout,
            p_bpe_dropout=p_bpe_dropout,
        )

    def noise_ctrl_char_preset_boundaries(self, subword_input_ids, byte_boundaries: set[int], byte_tokenizer: Optional[Any] = None):
        if byte_tokenizer is None:
            # problematic if the last token is not full valid utf-8
            text_bytes = self.subword_tokenizer.decode(subword_input_ids).encode("utf-8")
        else:
            text_bytes = byte_tokenizer.decode_to_bytes(byte_tokenizer.patch_ids_to_byte_ids(subword_input_ids.tolist()))

        text_bytes_with_split_char = b""
        for i in range(len(text_bytes)):
            text_bytes_with_split_char += text_bytes[i:i+1]

            if i in byte_boundaries:
                text_bytes_with_split_char += self.ctrl_char_bytes

        text_with_split_char = text_bytes_with_split_char.decode("utf-8", errors="ignore")

        return self.subword_tokenizer_with_ctrl_split.encode(text_with_split_char)

    def noise_ctrl_char(self, subword_input_ids):
        text = self.subword_tokenizer.decode(subword_input_ids)

        text_with_split_char = ""
        for i in range(len(text)):
            if i > 0 and text_with_split_char[-1] != self.ctrl_char and random.random() < self.p_ctrl_char:
                text_with_split_char += self.ctrl_char

            text_with_split_char += text[i]

        return self.subword_tokenizer_with_ctrl_split.encode(text_with_split_char)

    def noise_bpe_dropout(self, subword_input_ids):
        text = self.subword_tokenizer.decode(subword_input_ids)

        return self.subword_tokenizer_with_bpe_dropout.encode(text)

def get_dolma2_space_mask():
    DOLMA2_TOKENIZER = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

    space_mask = torch.zeros(len(DOLMA2_TOKENIZER), dtype=torch.bool)

    for token, token_id in DOLMA2_TOKENIZER.get_vocab().items():
        if token.startswith("Ġ") or token.startswith("Ċ") or token.startswith("ĉ"):
            space_mask[token_id] = True

    return space_mask

def get_blt_space_mask():
    offset = 4

    space_mask = torch.zeros(256 + offset, dtype=torch.bool)
    space_ids = [i + offset for i in list(" \t\n".encode("utf-8"))]

    for space_id in space_ids:
        space_mask[space_id] = True

    return space_mask

def _pad(tensors: list[torch.Tensor], multiple_of: int, direction: str, value):
    max_len = max(t.size(0) for t in tensors)
    if multiple_of > 1:
        # Round up max_len to the nearest multiple_of
        max_len = ((max_len + multiple_of - 1) // multiple_of) * multiple_of
    padded = []
    for t in tensors:
        if direction == "left":
            pad_shape = (max_len - t.size(0), 0)
        elif direction == "right":
            pad_shape = (0, max_len - t.size(0))
        else:
            raise ValueError(f"Unknown direction: {direction}. Must be 'left' or 'right'.")
        padded.append(F.pad(t, pad_shape, value=value))
    return torch.stack(padded, dim=0)

def pad_right(
    tensors: list[torch.Tensor],
    multiple_of: int = 128,
    value=0,
):
    return _pad(tensors, multiple_of, direction="right", value=value)

def pad_left(
    tensors: list[torch.Tensor],
    multiple_of: int = 128,
    value=0,
):
    return _pad(tensors, multiple_of, direction="left", value=value)