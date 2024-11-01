import logging
from functools import cached_property
from typing import List, Optional, Sequence, cast

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh

from olmo_core.config import StrEnum
from olmo_core.data.utils import get_cumulative_document_lengths
from olmo_core.doc_utils import beta_feature
from olmo_core.utils import get_default_device

from ..buffer_cache import BufferCache
from ..layer_norm import LayerNorm, LayerNormConfig
from ..utils import selective_checkpointing_context_fn
from .block import TransformerBlock, TransformerBlockConfig
from .init import InitMethod

__all__ = [
    "Transformer",
    "TransformerDataParallelWrappingStrategy",
    "TransformerActivationCheckpointingMode",
]


log = logging.getLogger(__name__)


class TransformerDataParallelWrappingStrategy(StrEnum):
    """
    An enumeration of the different wrapping strategy for the data parallel implementations.
    """

    full = "full"
    """
    Wrap each block (only applies to FSDP).
    """
    fine_grained = "fine_grained"
    """
    Wrap certain modules within each block in addition to wrapping each block (only applies to FSDP).
    """


@beta_feature
class TransformerActivationCheckpointingMode(StrEnum):
    """
    An enumeration of the different activation checkpointing modes.
    """

    full = "full"
    """Checkpoint every block."""
    selected_blocks = "selected_blocks"
    """Checkpoint only selected blocks."""
    selected_modules = "selected_modules"
    """Checkpoint only selected modules."""
    selected_ops = "selected_ops"
    """Checkpoint only a specific set of operations."""


class Transformer(nn.Module):
    """
    A typical "Llama-style" transformer implementation.

    :param d_model: The model dimensionality.
    :param vocab_size: The vocab size.
    :param n_layers: The number of transformer layers/blocks.
    :param block: The block configuration.
    :param layer_norm: The layer norm config for the final layer norm.
    :param bias: Whether to use a bias in the final linear layer.
    :param dtype: The datatype to use for the linear output layer.
    :param init_device: The device used when initializing parameters.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        block: TransformerBlockConfig,
        layer_norm: LayerNormConfig,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        init_method: InitMethod = InitMethod.normal,
        init_device: str = "cpu",
        init_seed: int = 0,
    ):
        super().__init__()
        cache = BufferCache()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model, dtype=dtype, device=init_device)
        self.blocks = nn.ModuleList(
            [
                block.build(
                    d_model=d_model,
                    block_idx=block_idx,
                    init_device=init_device,
                    cache=cache,
                )
                for block_idx in range(n_layers)
            ]
        )
        self.norm = layer_norm.build(d_model, init_device=init_device)
        self.w_out = nn.Linear(d_model, vocab_size, bias=bias, dtype=dtype, device=init_device)
        self.init_method = InitMethod(init_method)
        self.init_seed = init_seed
        self._cache = cache

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            if p.numel() > 0:
                return p.device
        return get_default_device()

    @torch.no_grad()
    def init_weights(
        self,
        *,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the model weights.

        :param max_seq_len: The maximum sequence length expected during training. This is used
            to warm up the RoPE cache.
        :param device: The device the local copy of the model will be trained on.
        """
        device = device or self.device
        generator = torch.Generator(device).manual_seed(self.init_seed)

        if self.embeddings is not None:
            self.init_method.init_embeddings(self.embeddings, generator=generator)

        for block in self.blocks:
            # This might fail if it's wrapped.
            #  assert isinstance(block, TransformerBlock)
            block = cast(TransformerBlock, block)
            att = block.attention

            # Norms.
            block_norms: List[LayerNorm] = [block.attention_norm, block.feed_forward_norm]
            if hasattr(att, "q_norm") and att.q_norm is not None:
                block_norms.append(att.q_norm)
            if hasattr(att, "k_norm") and att.k_norm is not None:
                block_norms.append(att.k_norm)
            for norm in block_norms:
                norm.reset_parameters()

            # Attention weights.
            self.init_method.init_attention(
                att, block_idx=block.block_idx, num_blocks=len(self.blocks), generator=generator
            )

            # Feed-forward weights.
            if hasattr(block, "feed_forward"):
                self.init_method.init_feed_forward(
                    block.feed_forward,
                    block_idx=block.block_idx,
                    num_blocks=len(self.blocks),
                    generator=generator,
                )
            else:
                self.init_method.init_feed_forward_moe(
                    block.feed_forward_moe,
                    block_idx=block.block_idx,
                    num_blocks=len(self.blocks),
                    generator=generator,
                )

            # Warm up RoPE cache.
            if max_seq_len is not None and att.rope is not None:
                att.rope.warmup_cache(max_seq_len, device)

        if self.norm is not None:
            self.norm.reset_parameters()

        if self.w_out is not None:
            self.init_method.init_final_w_out(self.w_out, d_model=self.d_model, generator=generator)

    def forward(
        self,
        input_ids: torch.Tensor,
        doc_lens: Optional[torch.Tensor] = None,
        max_doc_lens: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """
        Run the transformer on the token input IDs.

        :param input_ids: The token input IDs, shape ``(batch_size, seq_len)``.
        :param doc_lens: Document lengths to use in attention for intra-document masking.
            Shape ``(batch_size, max_docs)``.
            Required together with ``max_doc_lens`` when using intra-document masking.
        :param max_doc_lens: Maximum document length for each instance in the batch.
            Required together with ``doc_lens`` when using intra-document masking.

        :returns: The output logits.
        """
        max_doc_len: Optional[int] = None
        cu_doc_lens: Optional[torch.Tensor] = None
        if doc_lens is not None and max_doc_lens is not None:
            max_doc_len = max(max_doc_lens)
            cu_doc_lens = get_cumulative_document_lengths(doc_lens)

        # passthrough for non-existent layers, allows easy pipeline parallel configuration
        h = self.embeddings(input_ids) if self.embeddings is not None else input_ids

        for block in self.blocks:
            h = block(h, max_doc_len=max_doc_len, cu_doc_lens=cu_doc_lens)

        h = self.norm(h) if self.norm is not None else h
        out = self.w_out(h) if self.w_out is not None else h
        return out

    def apply_activation_checkpointing(
        self,
        mode: TransformerActivationCheckpointingMode,
        block_interval: Optional[int] = None,
        modules: Optional[List[str]] = None,
    ):
        """
        Apply activation checkpointing to the model.

        .. warning::
            Usually this does not need to be called directly, as :meth:`TransformerConfig.build()`
            will call it for you.

        :param mode: Determines how to apply activation checkpointing.
        :param block_interval: Required when :data:`mode` is "selected_blocks". Determines
            which blocks are wrapped.
        :param modules: Required when :data:`mode` is "selected_modules". A list of modules names
            to wrap for activation checkpointing. Globs are supported.
        """
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper as ptd_checkpoint_wrapper,
        )

        if (
            mode == TransformerActivationCheckpointingMode.selected_blocks
            and block_interval is None
        ):
            raise ValueError("'block_interval' is required for 'selected_blocks' mode")

        if mode == TransformerActivationCheckpointingMode.selected_modules and modules is None:
            raise ValueError("'modules' is required for 'selected_modules' mode")

        # TODO: only preserve RNG state if dropout is active
        preserve_rng_state = True

        if mode == TransformerActivationCheckpointingMode.selected_modules:
            from fnmatch import fnmatch

            assert modules is not None
            for name, module in self.named_modules():
                for pattern in modules:
                    if fnmatch(name, pattern):
                        break
                else:
                    continue

                parent_name = ".".join(name.split(".")[:-1])
                parent = self if not parent_name else self.get_submodule(parent_name)
                module = ptd_checkpoint_wrapper(module, preserve_rng_state=preserve_rng_state)
                parent.register_module(name.split(".")[-1], module)
                log.info(f"Wrapped '{name}' for activation checkpointing")
        else:
            for block_idx, block in enumerate(self.blocks):
                if mode == TransformerActivationCheckpointingMode.selected_blocks:
                    assert block_interval is not None
                    if block_idx % block_interval == 0:
                        block = ptd_checkpoint_wrapper(block, preserve_rng_state=preserve_rng_state)
                elif mode == TransformerActivationCheckpointingMode.full:
                    block = ptd_checkpoint_wrapper(block, preserve_rng_state=preserve_rng_state)
                elif mode == TransformerActivationCheckpointingMode.selected_ops:
                    block = ptd_checkpoint_wrapper(
                        block,
                        context_fn=selective_checkpointing_context_fn,
                        preserve_rng_state=preserve_rng_state,
                    )

                self.blocks.register_module(str(block_idx), block)

        log.info(f"Applied {mode} activation checkpointing to the model")

    def apply_compile(self):
        """
        Apply ``torch.compile()`` to each transformer block, which makes compilation efficient
        due to repeated structure.

        .. warning::
            Usually this does not need to be called directly, as :meth:`TransformerConfig.build()`
            will call it for you.

            If you do use this directly note that it must be called after
            :meth:`apply_activation_checkpointing()` but before :meth:`apply_fsdp()` or :meth:`apply_ddp()`.
        """
        for block_id, block in self.blocks.named_children():
            block = torch.compile(block, fullgraph=False)
            self.blocks.register_module(block_id, block)  # type: ignore

        log.info("Compiling each transformer block with torch.compile")

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
    ):
        """
        Apply FSDP(2) to the model.

        .. warning::
            Usually this does not need to be called directly, as :meth:`TransformerConfig.build()`
            will call it for you.

        :param dp_mesh: The data parallel device mesh.
        :param param_dtype: The data type to materialize params in. Defaults to the current param dtype.
        :param reduce_dtype: The data type for gradient reduction.
        :pp_enabled: If pipeline parallelism is also enabled.
        :wrapping_strategy: The wrapping strategy.
        """
        # Adapted from
        # https://github.com/pytorch/torchtitan/blob/90c889e972b56b9faadebbb78fc985dedc537ed9/torchtitan/parallelisms/parallelize_llama.py#L289

        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype or self.w_out.weight.dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)

        for block_id, block in enumerate(self.blocks):
            reshard_after_forward = True
            if pp_enabled:
                # For PP, do not reshard after forward to avoid per-microbatch
                # all-gathers, which can be expensive and non-overlapped
                reshard_after_forward = False
            elif wrapping_strategy == TransformerDataParallelWrappingStrategy.full:
                # As an optimization, do not reshard after forward for the last
                # transformer block since FSDP would prefetch it immediately
                reshard_after_forward = int(block_id) < len(self.blocks) - 1

            if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
                if hasattr(block, "feed_forward"):
                    fully_shard(
                        block.feed_forward,
                        reshard_after_forward=reshard_after_forward,
                        **fsdp_config,
                    )
                else:
                    fully_shard(
                        block.feed_forward_moe,
                        reshard_after_forward=reshard_after_forward,
                        **fsdp_config,
                    )

            fully_shard(block, reshard_after_forward=reshard_after_forward, **fsdp_config)

        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fully_shard(self.embeddings, reshard_after_forward=not pp_enabled, **fsdp_config)
            fully_shard(self.w_out, reshard_after_forward=False, **fsdp_config)

        fully_shard(self, reshard_after_forward=not pp_enabled, **fsdp_config)

        if dp_mesh is None:
            log.info("Applied FSDP2 to the model")
        else:
            log.info("Applied FSDP2 with hybrid sharding to the model")

    def apply_ddp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        compile_enabled: bool = False,
        autograd_compile_enabled: bool = False,
    ):
        """
        Apply DDP to the model.

        .. warning::
            Usually this does not need to be called directly, as :meth:`TransformerConfig.build()`
            will call it for you.
        """
        from torch.distributed._composable.replicate import replicate

        # Adapted from
        # https://github.com/pytorch/torchtitan/blob/90c889e972b56b9faadebbb78fc985dedc537ed9/torchtitan/parallelisms/parallelize_llama.py#L328
        if compile_enabled:
            if autograd_compile_enabled:
                torch._dynamo.config.optimize_ddp = "python_reducer_without_compiled_forward"  # type: ignore
            else:
                torch._dynamo.config.optimize_ddp = "ddp_optimizer"  # type: ignore

        replicate(self, device_mesh=dp_mesh, bucket_cap_mb=100)

        log.info("Applied DDP to the model")

    @cached_property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @cached_property
    def num_non_embedding_params(self) -> int:
        return self.num_params - self.embeddings.weight.numel()

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Get the approximate number of flops per token.
        """
        n, h, q, t = (
            len(self.blocks),
            self.blocks[0].attention.n_heads,
            self.d_model // self.blocks[0].attention.n_heads,
            seq_len,
        )

        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        flop_per_token = 6 * self.num_non_embedding_params + 12 * n * h * q * t

        return flop_per_token
