from typing import Callable, ClassVar

import torch
from torch._utils import _get_available_device_type, _get_device_module
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    create_block_mask,
    flex_attention,
)

# FlexAttention mask type. For each mask type, we initialize it at most once per
# batch. To record what it is initialized, FLEX_ATTN_MASK_T is used as the key to
# track the initialized mask.
FLEX_ATTN_MASK_T = tuple[str, int | None]


def has_cuda_capability(major: int, minor: int) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
        major,
        minor,
    )


def get_device_info() -> tuple[str, torch.device]:
    device_type = _get_available_device_type() or "cuda"
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


device_type, device_module = get_device_info()


class FlexAttention(torch.nn.Module):
    """FlexAttention module that uses torch.nn.attention.flex_attention.

    This module is a wrapper around torch.nn.attention.flex_attention. This module
    implements certain common attention types, such as causal and block_causal.

    Args:
        attn_mask_type (str): The type of attention mask. Currently, we support
            "causal" and "block_causal". "causal" means the lower triangle of the
            attention matrix is masked. "block_causal" means the attention matrix
            is divided into blocks, where block boundary is defined by EOS token,
            and the lower triangle of each block is masked.
        fixed_block_size (int | None): The block size to be used to perform attention.
            If specified, each sequence will be further divided to blocks, where each
            block has the maximum size of ``fixed_block_size``. A query will only attend
            to the keys within the same block.
    """

    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, mode="max-autotune", options={"triton.cudagraphs": False}
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(
        create_block_mask, mode="max-autotune", options={"triton.cudagraphs": False}
    )
    used_attn_mask_types: ClassVar[set[FLEX_ATTN_MASK_T]] = set()
    block_masks: ClassVar[dict[FLEX_ATTN_MASK_T, BlockMask]] = {}
    sink_mask_cache: ClassVar[dict[tuple, BlockMask]] = {}

    attn_mask_type: str

    def __init__(self, attn_mask_type: str, fixed_block_size: int | None = None) -> None:
        super().__init__()
        if attn_mask_type not in ["causal", "block_causal"]:
            raise ValueError(f"Unrecognized attn_mask_type {attn_mask_type}.")
        self.attn_mask_type = attn_mask_type
        self.fixed_block_size = fixed_block_size

        FlexAttention.used_attn_mask_types.add(self.mask_key)

    @property
    def mask_key(self) -> FLEX_ATTN_MASK_T:
        return (self.attn_mask_type, self.fixed_block_size)

    @classmethod
    def clear_sink_mask_cache(cls):
        cls.sink_mask_cache.clear()

    def forward(self, q, k, v, sink_weights=None, sliding_window=0, enable_gqa=False):
        """
        q : (B, H_q, S_q, D)
        k : (B, H_kv, S_kv, D)   -- without sink
        v : (B, H_kv, S_kv, D)
        sink_weights : (H_q,) or (H, M)   -- broadcast to all queries
        sliding_window : int
        enable_gqa : bool
        """
        if sink_weights is None:
            block_mask = FlexAttention.block_masks[self.mask_key]
            return FlexAttention.flex_attn(q, k, v, block_mask=block_mask)

        B, H_q, S_q, D = q.shape
        _, H_kv, S_kv, _ = k.shape
        sink_idx = S_kv  # sink occupies final key slot

        sink_k = k.new_empty(B, H_kv, 1, D)  # Faster - no initialization needed
        sink_v = v.new_zeros(B, H_kv, 1, D)  # Keep zeros for sink_v (needed for nullifying)

        k_ext = torch.cat([k, sink_k], dim=2)
        v_ext = torch.cat([v, sink_v], dim=2)

        mask_cache_key = (B, H_q, S_q, S_kv + 1, sliding_window if sliding_window else -1)

        if mask_cache_key not in FlexAttention.sink_mask_cache:
            if len(FlexAttention.sink_mask_cache) > 32:
                FlexAttention.sink_mask_cache.clear()

            # masks ensure sinks are included in softmax
            if sliding_window is not None and sliding_window > 0:
                mask_mod = FlexAttention._get_sliding_window_with_sink_mask_mod(
                    sliding_window, sink_idx
                )
            else:
                mask_mod = FlexAttention._get_causal_with_sink_mask_mod(sink_idx)

            FlexAttention.sink_mask_cache[
                mask_cache_key
            ] = FlexAttention.compiled_create_block_mask(mask_mod, B, H_q, S_q, S_kv + 1)

        block_mask = FlexAttention.sink_mask_cache[mask_cache_key]

        # overwrite the dummy sink scores with actual sink weights
        def score_mod(score, b, h_q, q_idx, kv_idx):
            return torch.where(
                kv_idx == sink_idx,
                sink_weights[h_q].to(score.dtype) + 0.0,  # cast + keep grad
                score,
            )

        return FlexAttention.flex_attn(
            q, k_ext, v_ext, block_mask=block_mask, score_mod=score_mod, enable_gqa=enable_gqa
        )

    @staticmethod
    def _get_causal_with_sink_mask_mod(sink_idx):
        """
        Returns a mask_mod function that
        - only allows kv_idx ≤ q_idx (causal)
        - or if kv_idx == sink_idx (always allow the sink)
        """
        orig = FlexAttention._get_causal_mask_mod()

        def causal_with_sink(b, h, q_idx, kv_idx):
            return orig(b, h, q_idx, kv_idx) | (kv_idx == sink_idx)

        return causal_with_sink

    @staticmethod
    def _get_sliding_window_with_sink_mask_mod(window: int, sink_idx: int):
        """
        Returns a mask_mod function that
        - only allows kv_idx ≤ q_idx (causal)
        - and only if (q_idx - kv_idx) ≤ window
        - or if kv_idx == sink_idx (always allow the sink)
        """

        def sliding_mod(b, h, q_idx, kv_idx):
            # causal within window
            keep = (kv_idx <= q_idx) & (q_idx - kv_idx <= window)
            # always allow the sink slot
            return keep | (kv_idx == sink_idx)

        return sliding_mod

    @staticmethod
    def _get_causal_mask_mod() -> _mask_mod_signature:
        def causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return q_idx >= kv_idx

        return causal_mask

    @staticmethod
    def _get_block_causal_mask_mod(batch: torch.Tensor, eos_id: int) -> _mask_mod_signature:
        # batch is [b, s, h, d] shape
        mask = batch == eos_id
        mask[:, -1] = True
        acc_mask = torch.cumsum(torch.where(mask, 1, 0), dim=1)
        # Optimize: use empty_like for faster initialization
        seq_idx = torch.empty_like(acc_mask, dtype=torch.int32)
        seq_idx[:, 0] = 0  # Only first position needs to be 0
        seq_idx[:, 1:] = acc_mask[:, :-1]

        def block_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_idx[b, q_idx] == seq_idx[b, kv_idx]) & (q_idx >= kv_idx)

        return block_causal_mask

    @staticmethod
    def _fixed_block_mask_mod(
        mask_mod: _mask_mod_signature, fixed_block_size: int
    ) -> _mask_mod_signature:
        """
        Given an arbirary mask_mod, divide the input sequence to blocks
        and only allow attention within the same block.

        Args:
            mask_mod: The mask mod to apply to the documents
            fixed_block_size: The number of tokens in each block.
        """

        # Credit to @drisspg.
        def blocked_mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # Get the block index of the query and key
            q_block = q_idx // fixed_block_size
            kv_block = kv_idx // fixed_block_size
            # Only allow attention within the same block
            same_block = q_block == kv_block
            # Apply the original mask mod
            inner_mask = mask_mod(b, h, q_idx % fixed_block_size, kv_idx % fixed_block_size)

            return same_block & inner_mask

        blocked_mask_mod.__name__ = (
            f"blocked_mask_mod_{mask_mod.__name__}_fixed_block_size_{fixed_block_size}"
        )

        return blocked_mask_mod

    @staticmethod
    @torch.no_grad()
    def init_attention_mask(batch: torch.Tensor, eos_id: int | None) -> None:
        # batch is [b, s, h, d] shape
        for mask_key in FlexAttention.used_attn_mask_types:
            attn_mask_type, fixed_block_size = mask_key
            match attn_mask_type:
                case "causal":
                    if FlexAttention.block_masks.get(mask_key, None) is not None:
                        continue
                    # We don't care about batch dimension --
                    # all samples have the same lower triangle mask.
                    batch_dimension = 1
                    mask_mod = FlexAttention._get_causal_mask_mod()
                case "block_causal":
                    if eos_id is None:
                        raise RuntimeError("eos_id must be provided for block_causal mask.")
                    batch_dimension = batch.shape[0]
                    mask_mod = FlexAttention._get_block_causal_mask_mod(batch, eos_id)
                case _:
                    raise RuntimeError(f"Shouldn't reach here. {attn_mask_type}")

            if fixed_block_size is not None and fixed_block_size > 0:
                mask_mod = FlexAttention._fixed_block_mask_mod(mask_mod, fixed_block_size)

            seq_len = batch.shape[1]
            block_mask = FlexAttention.compiled_create_block_mask(
                mask_mod, batch_dimension, None, seq_len, seq_len
            )
            FlexAttention.block_masks[mask_key] = block_mask
