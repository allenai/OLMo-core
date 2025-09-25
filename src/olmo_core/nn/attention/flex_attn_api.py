# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from typing import Callable, ClassVar, Optional, Tuple

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    and_masks,
    create_block_mask,
    flex_attention,
)

# FlexAttention mask type. For each mask type, we initialize it at most once per
# batch. To record what it is initialized, FLEX_ATTN_MASK_T is used as the key to
# track the initialized mask.
FLEX_ATTN_MASK_T = tuple[str, int | None]


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

    # We registered flex_attention related attributes as class variables as we
    # need to amortize the cost of compilation.
    # Disable CUDA graphs to avoid tensor overwriting issues
    # Using options only (can't use both mode and options together)
    flex_attn: ClassVar[Callable] = torch.compile(flex_attention)
    # Same for create_block_mask - disable CUDA graphs
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    used_attn_mask_types: ClassVar[set[FLEX_ATTN_MASK_T]] = set()
    # Attention mask type to the created BlockMask.
    # This allows us to keep track the created block masks for each
    # new batch. We will use this to update the block mask when a
    # new batch is created. This also allows user to create different
    # block masks for different layers.
    block_masks: ClassVar[dict[FLEX_ATTN_MASK_T, BlockMask]] = {}
    # Cache for sink block masks to avoid recompilation
    sink_block_masks: ClassVar[dict[tuple[int, int, int, int, int], BlockMask]] = {}

    # Instance variables.
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

    def forward(self, q, k, v, sink_weights=None, sliding_window=0, enable_gqa=False, block_mask=None, scale=None):
        """
        q : (B, H_q, S_q, D)
        k : (B, H_kv, S_kv, D)   -- without sink
        v : (B, H_kv, S_kv, D)
        sink_weights : (H_q,) or (H, M)   -- broadcast to all queries
        sliding_window : int
        enable_gqa : bool
        block_mask : Optional BlockMask for custom masking (used when sink_weights is None)
        scale : Optional scale factor for attention scores
        """
        if sink_weights is None:
            # Use provided block_mask or fall back to class's default mask
            if block_mask is None:
                block_mask = FlexAttention.block_masks.get(self.mask_key)
                if block_mask is None:
                    # Create a simple causal mask if no block mask exists
                    block_mask = self.get_causal_block_mask(
                        q.shape[2], q.device,
                        window_size=(sliding_window, 0) if sliding_window else None
                    )
            return FlexAttention.flex_attn(q, k, v, block_mask=block_mask, scale=scale, enable_gqa=enable_gqa)

        B, H_q, S_q, D = q.shape
        _, H_kv, S_kv, _ = k.shape
        sink_idx = S_kv  # sink occupies final key slot

        sink_k = k.new_zeros(B, H_kv, 1, D)  # this needn't be 0's since it's overwritten
        sink_v = v.new_zeros(B, H_kv, 1, D)  # 0 value nullifies sink weight in output

        k_ext = torch.cat([k, sink_k], dim=2)
        v_ext = torch.cat([v, sink_v], dim=2)

        cache_key = (B, H_q, S_q, S_kv + 1, sliding_window if sliding_window else -1)
        
        if cache_key not in FlexAttention.sink_block_masks:
            if sliding_window is not None and sliding_window > 0:
                mask_mod = FlexAttention._get_sliding_window_with_sink_mask_mod(
                    sliding_window, sink_idx
                )
            else:
                mask_mod = FlexAttention._get_causal_with_sink_mask_mod(sink_idx)

            block_mask = FlexAttention.compiled_create_block_mask(mask_mod, B, H_q, S_q, S_kv + 1)
            FlexAttention.sink_block_masks[cache_key] = block_mask
        else:
            block_mask = FlexAttention.sink_block_masks[cache_key]

        # overwrite the dummy sink scores with actual sink weights
        def score_mod(score, b, h_q, q_idx, kv_idx):
            return torch.where(
                kv_idx == sink_idx,
                sink_weights[h_q].to(score.dtype) + 0.0,  # cast + keep grad
                score,
            )

        return FlexAttention.flex_attn(
            q, k_ext, v_ext, block_mask=block_mask, score_mod=score_mod, scale=scale, enable_gqa=enable_gqa
        )

    @classmethod
    def clear_sink_mask_cache(cls):
        """Clear the cached sink block masks to free memory."""
        cls.sink_block_masks.clear()
    
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
        seq_idx = torch.zeros_like(acc_mask, dtype=torch.int32)
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
    def get_mask_mod(
        window_size: Optional[Tuple[int, int]] = None,
        doc_lens: Optional[Tuple[int, ...]] = None,
        device: Optional[torch.device] = None,
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Create a mask modification function combining causal, sliding window, and document masks.

        Args:
            window_size: Optional tuple (left_window, right_window) for sliding window attention.
                        (-1, -1) means no window restriction.
            doc_lens: Optional tuple of document lengths for intra-document masking.
            device: Required device when using document masking.

        Returns:
            A mask modification function for flex attention.
        """
        mask_mods = []

        def _causal_mask_mod(
            B: torch.Tensor, H: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            return q_idx >= kv_idx

        mask_mods.append(_causal_mask_mod)

        if window_size is not None and window_size != (-1, -1):

            def _sliding_window_mask_mod(
                B: torch.Tensor, H: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
            ) -> torch.Tensor:
                assert window_size is not None
                return torch.logical_and(
                    q_idx - kv_idx <= window_size[0], kv_idx - q_idx <= window_size[1]
                )

            mask_mods.append(_sliding_window_mask_mod)

        if doc_lens is not None:
            if device is None:
                raise ValueError("Device is required for intra-document masking mod")

            document_ids = torch.cat(
                [torch.full((int(doc_len),), i, device=device) for i, doc_len in enumerate(doc_lens)]
            )

            def _document_masking_mask_mod(
                B: torch.Tensor, H: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
            ) -> torch.Tensor:
                return document_ids[q_idx] == document_ids[kv_idx]

            mask_mods.append(_document_masking_mask_mod)

        return and_masks(*mask_mods)

    @staticmethod
    def create_causal_block_mask(
        seq_len: int,
        device: torch.device,
        window_size: Optional[Tuple[int, int]] = None,
        doc_lens: Optional[Tuple[int, ...]] = None,
        block_size: int = 128,
    ) -> BlockMask:
        """Create a causal block mask with optional sliding window and document boundaries.

        Args:
            seq_len: Sequence length
            device: Device to create the mask on
            window_size: Optional sliding window size (left, right)
            doc_lens: Optional document lengths as a tuple
            block_size: Block size for block mask creation

        Returns:
            BlockMask for flex attention
        """
        if doc_lens is not None:
            token_count = int(sum(doc_lens))
            if token_count % seq_len != 0:
                raise ValueError("Sum of document lengths is not a multiple of sequence length")

            # For intra-document masking, we merge the batch size dimension into the sequence dimension.
            return create_block_mask(
                FlexAttention.get_mask_mod(window_size, doc_lens=doc_lens, device=device),
                B=1,
                H=None,
                Q_LEN=token_count,
                KV_LEN=token_count,
                device=device.type,
                BLOCK_SIZE=block_size,
            )

        else:
            return create_block_mask(
                FlexAttention.get_mask_mod(window_size, device=device),
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device.type,
                BLOCK_SIZE=block_size,
            )

    @staticmethod
    def get_causal_block_mask(
        seq_len: int,
        device: torch.device,
        window_size: Optional[Tuple[int, int]] = None,
        doc_lens: Optional[torch.Tensor] = None,
        block_size: int = 128,
    ) -> BlockMask:
        """Create a causal block mask with optional sliding window and document boundaries.

        This is the main public API for creating causal block masks without sinks.

        Args:
            seq_len: Sequence length
            device: Device to create the mask on
            window_size: Optional sliding window size (left, right)
            doc_lens: Optional document lengths as a tensor
            block_size: Block size for block mask creation

        Returns:
            BlockMask for flex attention
        """
        if doc_lens is not None:
            doc_lens_list = tuple(doc_lens.flatten().tolist())
            return FlexAttention.create_causal_block_mask(
                seq_len, device, window_size, doc_lens_list, block_size
            )

        return FlexAttention.create_causal_block_mask(
            seq_len, device, window_size, doc_lens=None, block_size=block_size
        )

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


def init_attention_mask(batch: torch.Tensor, eos_id: int | None) -> None:
    FlexAttention.init_attention_mask(batch, eos_id)