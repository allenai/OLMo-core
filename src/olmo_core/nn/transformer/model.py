import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import RowwiseParallel, parallelize_module

from olmo_core.config import StrEnum
from olmo_core.data.utils import get_cumulative_document_lengths
from olmo_core.distributed.utils import hide_from_torch, unhide_from_torch
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import get_default_device, mark_dynamic, move_to_device

from ..attention import (
    Attention,
    AttentionBase,
    FusedAttention,
    RingAttentionLoadBalancer,
    RingAttentionLoadBalancerType,
)
from ..buffer_cache import BufferCache
from ..functional import l2_normalize
from ..lm_head import LMHeadConfig, LMOutputWithLoss
from ..rope import RoPEBuffers, RotaryEmbeddingBase
from ..utils import selective_checkpointing_context_fn
from .block import (
    MoETransformerBlock,
    NormalizedTransformerBlock,
    TransformerBlock,
    TransformerBlockBase,
    TransformerBlockConfig,
)
from .init import InitMethod

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

__all__ = [
    "Transformer",
    "NormalizedTransformer",
    "MoETransformer",
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
    Wrap each block and the LM head (only applies to FSDP).
    """

    blocks = "blocks"
    """
    Like full but the LM head is not wrapped separately (only applies to FSDP).
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
        lm_head: LMHeadConfig,
        dtype: torch.dtype = torch.float32,
        init_method: InitMethod = InitMethod.normal,
        init_device: str = "cpu",
        init_seed: int = 0,
    ):
        super().__init__()

        cache = BufferCache()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_attn_heads = block.attention.n_heads
        self.dtype = dtype

        self.embeddings = nn.Embedding(vocab_size, d_model, dtype=dtype, device=init_device)
        self.blocks = nn.ModuleDict()
        for block_idx in range(n_layers):
            block_ = block.build(
                d_model=d_model,
                block_idx=block_idx,
                init_device=init_device,
                cache=cache,
            )
            self._validate_block(block_)
            self.blocks[str(block_idx)] = block_
        self.lm_head = lm_head.build(
            d_model=d_model, vocab_size=vocab_size, init_device=init_device
        )

        self.init_device = init_device
        self.init_method = InitMethod(init_method)
        self.init_seed = init_seed

        self._cache = cache
        self._compile_enabled = False
        self._device: Optional[torch.device] = None
        self._cp_load_balancer: Optional[RingAttentionLoadBalancer] = None

        # Cache the value of these properties up-front in case the parameters are removed
        # later, like for pipeline parallelism.
        self.num_params
        self.num_non_embedding_params

    def _validate_block(self, block: TransformerBlockBase):
        del block

    def compute_auxiliary_losses(
        self, total_bz: Union[int, torch.Tensor], reset: bool = True
    ) -> Dict[str, torch.Tensor]:
        del total_bz, reset
        return {}

    def reset_auxiliary_losses(self):
        pass

    def compute_auxiliary_metrics(
        self, total_bz: Union[int, torch.Tensor], reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        del total_bz, reset
        return {}

    def reset_auxiliary_metrics(self):
        pass

    @property
    def is_moe(self) -> bool:
        return False

    @property
    def device(self) -> torch.device:
        if self._device is None:
            for p in self.parameters():
                if p.numel() > 0:
                    self._device = p.device
                    break
            else:
                self._device = get_default_device()
        return self._device

    @property
    def compile_enabled(self) -> bool:
        return self._compile_enabled

    def get_rope_buffers(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> Optional[RoPEBuffers]:
        if device is None:
            device = self.device
        for block in self.blocks.values():
            rope = cast(Optional[RotaryEmbeddingBase], block.attention.rope)  # type: ignore
            if rope is not None:
                return rope.get_buffers(seq_len, device)
        return None

    @torch.no_grad()
    def init_weights(
        self,
        *,
        max_seq_len: Optional[int] = None,
        max_local_microbatch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Generator:
        """
        Initialize the model weights.

        :param max_seq_len: The maximum sequence length expected. This is used
            to warm up the RoPE cache.
        :param max_local_microbatch_size: The maximum local (rank) micro-batch size (in tokens)
            expected. This is used to warm-up some MoE cache.
        :param device: The device the local copy of the model will be trained on.
        """
        device = device or self.device
        self.to_empty(device=device)

        generator = torch.Generator(device).manual_seed(self.init_seed)

        if self.embeddings is not None:
            self.init_method.init_embeddings(
                self.embeddings, d_model=self.d_model, generator=generator
            )

        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()  # type: ignore

        for block in self.blocks.values():
            # This might fail if it's wrapped.
            #  assert isinstance(block, TransformerBlock)
            block = cast(TransformerBlock, block)
            att = cast(Union[Attention, FusedAttention], block.attention)

            # Attention weights.
            self.init_method.init_attention(
                att,
                d_model=self.d_model,
                block_idx=block.block_idx,
                num_blocks=self.n_layers,
                generator=generator,
            )

            # Feed-forward weights.
            if hasattr(block, "feed_forward"):
                self.init_method.init_feed_forward(
                    block.feed_forward,
                    d_model=self.d_model,
                    block_idx=block.block_idx,
                    num_blocks=self.n_layers,
                    generator=generator,
                )
            else:
                block = cast(MoETransformerBlock, block)
                if max_local_microbatch_size is not None:
                    block.feed_forward_moe.warmup_cache(max_local_microbatch_size)
                self.init_method.init_feed_forward_moe(
                    block.feed_forward_moe,
                    d_model=self.d_model,
                    block_idx=block.block_idx,
                    num_blocks=self.n_layers,
                    generator=generator,
                )

            # Warm up RoPE cache.
            if max_seq_len is not None and att.rope is not None:
                att.rope.warmup_cache(max_seq_len, device)

        if self.lm_head is not None:
            self.init_method.init_final_w_out(
                self.lm_head.w_out, d_model=self.d_model, generator=generator
            )

        return generator

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        ignore_index: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        B, S = input_ids.shape
        out_kwargs: Dict[str, Any] = {}

        # Prepare document length inputs.
        max_doc_len: Optional[int] = None
        cu_doc_lens: Optional[torch.Tensor] = None
        if (doc_lens := kwargs.pop("doc_lens", None)) is not None and (
            max_doc_lens := kwargs.pop("max_doc_lens", None)
        ) is not None:
            max_doc_len = max(max_doc_lens)
            cu_doc_lens = get_cumulative_document_lengths(doc_lens)

        # Shard inputs and RoPE buffers on sequence dimension if using context parallelism.
        if (cp_load_balancer := self._cp_load_balancer) is not None:
            inputs = [input_ids]
            seq_dims = [1]
            pad_values: List[Union[int, float]] = [0]
            keys = ["input_ids"]

            # NOTE: initialize buffer(s) on CPU to avoid possible host-device sync when sharding.
            rope_buffers = self.get_rope_buffers(S, torch.device("cpu"))
            if rope_buffers is not None:
                if rope_buffers.pos_sin is not None:
                    inputs.append(rope_buffers.pos_sin)
                    seq_dims.append(0)
                    pad_values.append(0.0)
                    keys.append("pos_sin")
                if rope_buffers.pos_cos is not None:
                    inputs.append(rope_buffers.pos_cos)
                    seq_dims.append(0)
                    pad_values.append(0.0)
                    keys.append("pos_cos")
                if rope_buffers.freqs_cis is not None:
                    inputs.append(rope_buffers.freqs_cis)
                    seq_dims.append(0)
                    pad_values.append(0.0)
                    keys.append("freqs_cis")

            if labels is not None:
                inputs.append(labels)
                seq_dims.append(1)
                pad_values.append(ignore_index)
                keys.append("labels")

            if cu_doc_lens is not None:
                # NOTE: Can only shard properly here if 'input_ids' is flat, i.e. a single instance.
                # TODO: (epwalsh) We could just flatten all of the inputs here, but then we risk going
                # beyond the model's maximum sequence length, which might be okay at least
                # with relative positional encodings, but then again if you're resorting to context
                # parallelism you can probably only fit a single instance at a time anyway.
                if B != 1:
                    raise RuntimeError(
                        f"Rank micro-batches must consist of a single instance when using "
                        f"context parallelism with intra-document masking (got {B} instances)"
                    )
                inputs, additional_inputs = cp_load_balancer.batch_shard_by_document(
                    inputs=inputs,
                    seq_dims=seq_dims,
                    cu_doc_lens=cu_doc_lens,
                    pad_values=pad_values,
                    length_multiple=16,
                )
                for key, value in additional_inputs.items():
                    out_kwargs[key] = move_to_device(value, self.device)
            else:
                inputs = cp_load_balancer.batch_shard(
                    inputs=inputs,
                    seq_dims=seq_dims,
                    pad_values=pad_values,
                )

            for key, value in zip(keys, inputs):
                out_kwargs[key] = move_to_device(value, self.device)

            input_ids = out_kwargs.pop("input_ids")
            labels = out_kwargs.pop("labels", None)
        else:
            input_ids = move_to_device(input_ids, self.device)
            labels = move_to_device(labels, self.device)
            out_kwargs["max_doc_len"] = max_doc_len
            out_kwargs["cu_doc_lens"] = move_to_device(cu_doc_lens, self.device)

        return (
            input_ids,
            labels,
            out_kwargs,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run the transformer on the token input IDs.

        :param input_ids: The token input IDs, shape ``(batch_size, seq_len)``.

        :returns: The logits if ``labels`` is ``None`` or the losses if ``labels`` is not ``None``.
        """
        input_ids, labels, kwargs = self._prepare_inputs(
            input_ids, labels, ignore_index=ignore_index, **kwargs
        )
        loss_div_factor = move_to_device(loss_div_factor, self.device)

        # Get embeddings but pass-through for non-existent layers to allow easy
        # pipeline parallel configuration.
        h = self.embeddings(input_ids) if self.embeddings is not None else input_ids

        # Run each block.
        for block in self.blocks.values():
            # Mark sizes as dynamic for torch.compile().
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
            h = block(h, **kwargs)

        # Get final logits but again pass-through in case of pipeline parallelism.
        if self.lm_head is not None:
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
                if labels is not None:
                    mark_dynamic(labels, (0, 1), strict=False)
            return self.lm_head(
                h,
                labels,
                ignore_index=ignore_index,
                loss_reduction=loss_reduction,
                z_loss_multiplier=z_loss_multiplier,
                loss_div_factor=loss_div_factor,
                return_logits=return_logits,
            )
        else:
            return h

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        float8_enabled: bool = False,
    ):
        """
        Apply tensor parallelism to the model.

        :param loss_parallel: Set to ``True`` if parallelizing the loss function as well.
        :param float8_enabled: Set this to ``True`` if training with float8 linear layers.
        """
        if self.embeddings is not None:
            parallelize_module(
                self.embeddings,
                device_mesh=tp_mesh,
                parallelize_plan=RowwiseParallel(
                    input_layouts=Replicate(),
                    use_local_output=False,
                ),
            )

        # Apply tensor/sequence parallelism to every transformer block.
        # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
        #       by folding (and unfolding) the batch dimension and the sequence dimension.
        #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_tp(tp_mesh, float8_enabled=float8_enabled)

        if self.lm_head is not None:
            self.lm_head.apply_tp(tp_mesh, input_layouts=(Shard(1), Replicate()))

    def apply_cp(self, cp_mesh: DeviceMesh, load_balancer: RingAttentionLoadBalancerType):
        """
        Prepare the model for context-parallelism (CP).

        :param cp_mesh: The CP device mesh.
        :param load_balancer: The load balancing method.
        """
        self._cp_load_balancer = load_balancer.build(cp_mesh)
        for block in self.blocks.values():
            cast(AttentionBase, block.attention).apply_cp(cp_mesh, load_balancer=load_balancer)

    def apply_activation_checkpointing(
        self,
        mode: TransformerActivationCheckpointingMode,
        block_interval: Optional[int] = None,
        modules: Optional[List[str]] = None,
    ):
        """
        Apply activation checkpointing to the model.

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
            for block_idx, block in enumerate(self.blocks.values()):
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

    def apply_compile(self):
        """
        Apply ``torch.compile()`` to each transformer block, which makes compilation efficient
        due to repeated structure.

        .. warning::
            This must be called after :meth:`apply_activation_checkpointing()` but
            before :meth:`apply_fsdp()` or :meth:`apply_ddp()`.
        """
        for block_id, block in self.blocks.named_children():
            block = torch.compile(block, fullgraph=False)
            self.blocks.register_module(block_id, block)  # type: ignore

        if self.lm_head is not None:
            self.register_module("lm_head", torch.compile(self.lm_head, fullgraph=False))  # type: ignore

        self._compile_enabled = True

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
            This should generally be called last if using any other parallelism strategies or optimizations
            like :meth:`apply_compile()`.

        :param dp_mesh: The model data parallel device mesh.
        :param param_dtype: The data type to materialize params in. Defaults to the current param dtype.
        :param reduce_dtype: The data type for gradient reduction.
        :pp_enabled: If pipeline parallelism is also enabled.
        :wrapping_strategy: The wrapping strategy.
        """
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype or self.dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)

        for block in self.blocks.values():
            # For PP, do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = False if pp_enabled else True

            if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
                if hasattr(block, "feed_forward"):
                    fully_shard(
                        block.feed_forward,  # type: ignore
                        reshard_after_forward=reshard_after_forward,
                        **fsdp_config,
                    )
                else:
                    fully_shard(
                        block.feed_forward_moe,  # type: ignore
                        reshard_after_forward=reshard_after_forward,
                        **fsdp_config,
                    )

            fully_shard(block, reshard_after_forward=reshard_after_forward, **fsdp_config)

        if (
            wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained
            and self.embeddings is not None
        ):
            fully_shard(self.embeddings, reshard_after_forward=not pp_enabled, **fsdp_config)

        if (
            wrapping_strategy != TransformerDataParallelWrappingStrategy.blocks
            and self.lm_head is not None
        ):
            fully_shard(self.lm_head, reshard_after_forward=False, **fsdp_config)

        fully_shard(self, reshard_after_forward=not pp_enabled, **fsdp_config)
        # Some inputs need to be on CPU initially, but FSDP will move everything to model's
        # device if we don't hide it.
        self.register_forward_pre_hook(_hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True)
        self.register_forward_pre_hook(
            _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        )

    def apply_ddp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        compile_enabled: bool = False,
        autograd_compile_enabled: bool = False,
    ):
        """
        Apply DDP to the model.
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
        # Some inputs need to be on CPU initially, but DDP will move everything to model's
        # device if we don't hide it.
        self.register_forward_pre_hook(_hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True)
        self.register_forward_pre_hook(
            _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        )

    @cached_property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @cached_property
    def num_non_embedding_params(self) -> int:
        return self.num_params - self.embeddings.weight.numel()

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Get the approximate number of flops per token.
        """
        n, h, q, t = (
            self.n_layers,
            self.n_attn_heads,
            self.d_model // self.n_attn_heads,
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


@beta_feature
class NormalizedTransformer(Transformer):
    """
    A nGPT transformer implementation, to be used with the :class:`NormalizedTransformerBlock` block
    type.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        block: TransformerBlockConfig,
        lm_head: LMHeadConfig,
        dtype: torch.dtype = torch.float32,
        init_method: InitMethod = InitMethod.normalized,
        init_device: str = "cpu",
        init_seed: int = 0,
    ):
        super().__init__(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=lm_head,
            dtype=dtype,
            init_method=init_method,
            init_device=init_device,
            init_seed=init_seed,
        )

    def _validate_block(self, block: TransformerBlockBase):
        if not isinstance(block, NormalizedTransformerBlock):
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}' requires a '{NormalizedTransformerBlock.__name__}' block"
            )

    @torch.no_grad()
    def init_weights(
        self,
        *,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Generator:
        generator = super().init_weights(max_seq_len=max_seq_len, device=device)
        self.normalize_matrices()
        return generator

    @torch.no_grad()
    def normalize_matrices(self):
        """
        Normalize the weights in all matrices. This should be called after each optimizer step, which
        the :class:`~olmo_core.train.train_module.TransformerTrainModule` will handle for you.
        """
        if self.embeddings is not None:
            self._normalize_matrix(self.embeddings.weight)

        for block in self.blocks.values():
            if hasattr(block, "normalize_matrices"):
                block.normalize_matrices()  # type: ignore

        if self.lm_head is not None:
            self.lm_head.normalize_matrices()  # type: ignore

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        float8_enabled: bool = False,
    ):
        del tp_mesh, float8_enabled

        raise NotImplementedError(
            "TP is not implemented yet for the normalized transformer variant"
        )

    def apply_compile(self):
        super().apply_compile()
        self.normalize_matrices = torch.compile(self.normalize_matrices)


@beta_feature
class MoETransformer(Transformer):
    """
    An MoE transformer implementation, to be used with one of the
    :class:`MoETransformerBlock` block types.
    """

    @property
    def is_moe(self) -> bool:
        return True

    def _validate_block(self, block: TransformerBlockBase):
        if not isinstance(block, MoETransformerBlock):
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}' requires a '{MoETransformerBlock.__name__}' block"
            )

    def compute_auxiliary_losses(
        self, total_bz: Union[int, torch.Tensor], reset: bool = True
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for block in self.blocks.values():
            for loss_name, loss_val in (
                cast(MoETransformerBlock, block).compute_losses(total_bz, reset=reset).items()
            ):
                loss_val.div_(self.n_layers)
                if loss_name in out:
                    out[loss_name] += loss_val
                else:
                    out[loss_name] = loss_val
        return out

    def reset_auxiliary_losses(self):
        for block in self.blocks.values():
            cast(MoETransformerBlock, block).reset_losses()

    def compute_auxiliary_metrics(
        self, total_bz: Union[int, torch.Tensor], reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}
        for block_idx, block in self.blocks.items():
            for metric_name, metric_val in (
                cast(MoETransformerBlock, block).compute_metrics(total_bz, reset=reset).items()
            ):
                out[f"block {int(block_idx):02d}/{metric_name}"] = metric_val
        return out

    def reset_auxiliary_metrics(self):
        for block in self.blocks.values():
            cast(MoETransformerBlock, block).reset_metrics()

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        for block in self.blocks.values():
            cast(MoETransformerBlock, block).apply_ep(ep_mesh, **kwargs)

    def prepare_experts_for_fsdp(
        self,
        world_mesh: DeviceMesh,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
    ):
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        for block in self.blocks.values():
            cast(MoETransformerBlock, block).feed_forward_moe.prepare_experts_for_fsdp(
                world_mesh=world_mesh,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=param_dtype or self.dtype, reduce_dtype=reduce_dtype
                ),
                reshard_after_forward=not pp_enabled,
            )

    def prepare_experts_for_ddp(self, world_mesh: DeviceMesh):
        for block in self.blocks.values():
            cast(MoETransformerBlock, block).feed_forward_moe.prepare_experts_for_ddp(
                world_mesh=world_mesh,
            )


def _hide_cpu_inputs_from_torch(m, args, kwargs) -> Optional[Tuple[Any, Dict[str, Any]]]:
    del m
    if (doc_lens := kwargs.get("doc_lens")) is not None:
        kwargs["doc_lens"] = hide_from_torch(doc_lens)
    return (args, kwargs)


def _unhide_cpu_inputs_from_torch(m, args, kwargs) -> Optional[Tuple[Any, Dict[str, Any]]]:
    del m
    if (doc_lens := kwargs.get("doc_lens")) is not None:
        kwargs["doc_lens"] = unhide_from_torch(doc_lens)
    return (args, kwargs)
