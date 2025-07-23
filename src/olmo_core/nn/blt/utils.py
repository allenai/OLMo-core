from typing import Dict, Any, List

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

def get_original_labels(batch: Dict[str, Any], label_ignore_index: int = -100) -> torch.Tensor:
    max_byte_len = batch["input_ids"].shape[-1]

    # Labels are just input IDs shifted to the left (first item is ignored).
    labels, attention_mask, patch_lens = (
        batch["original_input_ids"].clone(),
        batch.get("original_attention_mask"),
        batch.get("patch_lens"),
    )
    if attention_mask is not None:
        labels.masked_fill_(attention_mask == 0.0, label_ignore_index)
    if patch_lens is not None:
        labels.masked_fill_(~patch_lens.cumsum(-1) > max_byte_len, value=label_ignore_index)
    # Shift and pad.
    return F.pad(labels[..., 1:], (0, 1, 0, 0), value=label_ignore_index)