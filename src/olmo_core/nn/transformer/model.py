import logging
from collections import defaultdict
from functools import cached_property, lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import RowwiseParallel, parallelize_module

from olmo_core.data.tokenizer import ByteTokenizerConfig
from olmo_core.data.utils import get_cumulative_document_lengths, get_labels
from olmo_core.distributed.parallel import get_pp_mesh
from olmo_core.distributed.utils import hide_from_torch, unhide_from_torch
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.utils import get_default_device, mark_dynamic, move_to_device
from olmo_core.nn.blt.config import BLTConfig
from olmo_core.nn.functional.cross_entropy_loss import cross_entropy_loss
from olmo_core.nn.attention import FlashAttention2Backend

from ..attention import (
    Attention,
    FusedAttention,
    RingAttentionLoadBalancer,
    RingAttentionLoadBalancerType,
)
from ..buffer_cache import BufferCache
from ..functional import l2_normalize
from ..lm_head import LMHeadConfig, LMOutputWithLoss
from ..moe import MoEBase
from ..blt.config import LocalEncoderConfig, LocalDecoderConfig
from ..blt import utils as blt_utils
from ..rope import RoPEBuffers, RotaryEmbeddingBase
from ..utils import selective_checkpointing_context_fn
from .block import (
    MoETransformerBlock,
    NormalizedTransformerBlock,
    TransformerBlock,
    TransformerBlockBase,
)
from .config import (
    TransformerActivationCheckpointingMode,
    TransformerBlockConfig,
    TransformerDataParallelWrappingStrategy,
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
        init_std: float = 0.02,
        block_overrides: Optional[Dict[int, TransformerBlockConfig]] = None,
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
            block_config = block
            if block_overrides is not None and block_idx in block_overrides:
                block_config = block_overrides[block_idx]
            self.blocks[str(block_idx)] = self._validate_block(
                block_config.build(
                    d_model=d_model,
                    block_idx=block_idx,
                    n_layers=n_layers,
                    init_device=init_device,
                    cache=cache,
                )
            )
        self.lm_head = lm_head.build(
            d_model=d_model, vocab_size=vocab_size, init_device=init_device
        )

        self.init_device = init_device
        self.init_method = InitMethod(init_method)
        self.init_seed = init_seed
        self.init_std = init_std

        self._cache = cache
        self._pp_enabled = False
        self._pp_group_size = 1
        self._fp8_enabled = False
        self._precompute_float8_dynamic_scale_for_fsdp = False
        self._compile_enabled = False
        self._device: Optional[torch.device] = None
        self._cp_load_balancer: Optional[RingAttentionLoadBalancer] = None
        self._tp_enabled = False
        self._tp_mesh: Optional[DeviceMesh] = None
        self._fsdp_enabled = False

        # Cache the value of these properties up-front in case the parameters are removed
        # later, like for pipeline parallelism.
        self.num_params
        self.num_non_embedding_params

    def _validate_block(self, block: TransformerBlockBase) -> TransformerBlockBase:
        return block

    def compute_auxiliary_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        del reset
        return {}

    def reset_auxiliary_metrics(self):
        pass

    @property
    def pp_enabled(self) -> bool:
        return self._pp_enabled

    @property
    def fp8_enabled(self) -> bool:
        return self._fp8_enabled

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    @property
    def fsdp_enabled(self) -> bool:
        return self._fsdp_enabled

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
    ) -> Dict[int, Optional[RoPEBuffers]]:
        """
        Get the RoPE buffers to pass to each layer.
        """
        if device is None:
            device = self.device
        rope_buffers = {}
        for key, block in self.blocks.items():
            rope = cast(Optional[RotaryEmbeddingBase], block.attention.rope)  # type: ignore
            rope_buffers[int(key)] = None if rope is None else rope.get_buffers(seq_len, device)
        return rope_buffers

    @torch.no_grad()
    def init_weights(
        self,
        *,
        max_seq_len: Optional[int] = None,
        max_local_microbatch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        world_mesh: Optional[DeviceMesh] = None,
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

        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()  # type: ignore

        seed = self.init_seed
        if world_mesh is not None and self.pp_enabled:
            seed += get_pp_mesh(world_mesh).get_local_rank()

        generator = torch.Generator(device).manual_seed(seed)

        if self.embeddings is not None:
            self.init_method.init_embeddings(
                self.embeddings,
                d_model=self.d_model,
                std=self.init_std,
                generator=generator,
            )

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
                std=self.init_std,
                generator=generator,
            )

            # Feed-forward weights.
            if hasattr(block, "feed_forward"):
                self.init_method.init_feed_forward(
                    block.feed_forward,
                    d_model=self.d_model,
                    block_idx=block.block_idx,
                    num_blocks=self.n_layers,
                    std=self.init_std,
                    generator=generator,
                )

            # MoE weights.
            if hasattr(block, "feed_forward_moe"):
                block = cast(MoETransformerBlock, block)
                if max_local_microbatch_size is not None:
                    block.feed_forward_moe.warmup_cache(max_local_microbatch_size)
                self.init_method.init_feed_forward_moe(
                    block.feed_forward_moe,
                    d_model=self.d_model,
                    block_idx=block.block_idx,
                    num_blocks=self.n_layers,
                    std=self.init_std,
                    generator=generator,
                )

            # Warm up attention backend cache.
            if max_seq_len is not None and att.backend is not None:
                att.backend.warmup_cache(max_seq_len, device)

            # Warm up RoPE cache.
            if max_seq_len is not None and att.rope is not None:
                att.rope.warmup_cache(max_seq_len, device)

        if self.lm_head is not None:
            self.init_method.init_final_w_out(
                self.lm_head.w_out,
                d_model=self.d_model,
                std=self.init_std,
                generator=generator,
            )

        return generator

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Dict[str, Any],
        Dict[int, Dict[str, Any]],
        Dict[str, Any],
    ]:
        # NOTE: with pipeline parallelism input_ids might actually be an intermediate output,
        # so we have to be careful here.
        B, S = input_ids.shape[:2]

        all_block_kwargs: Dict[str, Any] = {}
        per_block_kwargs: Dict[int, Dict[str, Any]] = defaultdict(dict)
        lm_head_kwargs: Dict[str, Any] = dict(
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            return_logits=return_logits,
            logits_to_keep=logits_to_keep,
        )

        if loss_div_factor is not None:
            loss_div_factor = move_to_device(loss_div_factor, self.device)
            lm_head_kwargs["loss_div_factor"] = loss_div_factor
            all_block_kwargs["loss_div_factor"] = loss_div_factor

        # Prepare document length inputs.
        max_doc_len: Optional[int] = None
        cu_doc_lens: Optional[torch.Tensor] = None
        doc_lens: Optional[torch.Tensor] = None
        cache_leftpad: Optional[torch.Tensor] = kwargs.pop("cache_leftpad", None)

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
            for block_idx, rope_buffers in self.get_rope_buffers(S, torch.device("cpu")).items():
                if rope_buffers is not None:
                    if rope_buffers.pos_sin is not None:
                        inputs.append(rope_buffers.pos_sin)
                        seq_dims.append(0)
                        pad_values.append(0.0)
                        keys.append(f"block_{block_idx}.pos_sin")
                    if rope_buffers.pos_cos is not None:
                        inputs.append(rope_buffers.pos_cos)
                        seq_dims.append(0)
                        pad_values.append(0.0)
                        keys.append(f"block_{block_idx}.pos_cos")
                    if rope_buffers.freqs_cis is not None:
                        inputs.append(rope_buffers.freqs_cis)
                        seq_dims.append(0)
                        pad_values.append(0.0)
                        keys.append(f"block_{block_idx}.freqs_cis")

            if labels is not None:
                inputs.append(labels)
                seq_dims.append(1)
                pad_values.append(ignore_index)
                keys.append("labels")

            if cache_leftpad is not None:
                raise NotImplementedError("cache_leftpad is not supported with context parallelism")

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
                    all_block_kwargs[key] = move_to_device(value, self.device)

            else:
                inputs = cp_load_balancer.batch_shard(
                    inputs=inputs,
                    seq_dims=seq_dims,
                    pad_values=pad_values,
                )

            for key, value in zip(keys, inputs):
                if key.startswith("block_"):
                    block_key, key = key.split(".", 1)
                    block_idx = int(block_key.replace("block_", ""))
                    per_block_kwargs[block_idx][key] = move_to_device(value, self.device)
                else:
                    all_block_kwargs[key] = move_to_device(value, self.device)

            input_ids = all_block_kwargs.pop("input_ids")
            labels = all_block_kwargs.pop("labels", None)
        else:
            input_ids = move_to_device(input_ids, self.device)
            labels = move_to_device(labels, self.device)

            if (max_doc_len is not None or cu_doc_lens is not None) and cache_leftpad is not None:
                raise ValueError("max_doc_len/cu_doc_lens and cache_leftpad are mutually exclusive")
            if max_doc_len is not None or cu_doc_lens is not None:
                all_block_kwargs["max_doc_len"] = max_doc_len
                all_block_kwargs["cu_doc_lens"] = move_to_device(cu_doc_lens, self.device)
            if cache_leftpad is not None:
                all_block_kwargs["cache_leftpad"] = move_to_device(cache_leftpad, self.device)

        return (
            input_ids,
            labels,
            all_block_kwargs,
            per_block_kwargs,
            lm_head_kwargs,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run the transformer on the token input IDs.

        :param input_ids: The token input IDs, shape ``(batch_size, seq_len)``.
        :param labels: The token labels, shape ``(batch_size, seq_len)``.
        :param ignore_index: The index to ignore in the loss computation. Default is -100.
        :param loss_reduction: The reduction method for the loss. Can be "mean", "sum", or "none".
        :param z_loss_multiplier: Optional multiplier for the z-loss regularization term.
        :param loss_div_factor: Optional divisor for the loss, can be a scalar or tensor.
        :param return_logits: Whether to return logits along with the loss when labels are provided.
        :param logits_to_keep: Number of positions to keep from the end of the sequence (if int),
            or tensor specifying which positions to keep. Default is 0 (keep all).

        :returns: The logits if ``labels`` is ``None`` or the losses if ``labels`` is not ``None``.
        """
        (
            input_ids,
            labels,
            all_block_kwargs,
            per_block_kwargs,
            lm_head_kwargs,
        ) = self._prepare_inputs(
            input_ids,
            labels,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            loss_div_factor=loss_div_factor,
            return_logits=return_logits,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # Get embeddings but pass-through for non-existent layers to allow easy
        # pipeline parallel configuration.
        h = self.embeddings(input_ids) if self.embeddings is not None else input_ids

        dtype = h.dtype

        # Run each block.
        for block_key, block in self.blocks.items():
            block_idx = int(block_key)
            block_kwargs = per_block_kwargs.get(block_idx, {})
            # Mark sizes as dynamic for torch.compile().
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
            h = block(h, **all_block_kwargs, **block_kwargs)

        # Get final logits but again pass-through in case of pipeline parallelism.
        if self.lm_head is not None:
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
                if labels is not None:
                    mark_dynamic(labels, (0, 1), strict=False)
            # NOTE: When TP is active we can't pass 'labels=None' or the hook from 'PrepareModuleInput'
            # will throw an exception.
            if labels is not None:
                lm_head_kwargs["labels"] = labels
            return self.lm_head(h, **lm_head_kwargs)
        else:
            return h

    def apply_fp8(self, float8_config: Float8Config):
        """
        Use an FP8 recipe on most linear layers.
        """
        if not float8_config.enabled:
            return

        modules_to_ignore = set()
        if self.lm_head is not None:
            modules_to_ignore.add("lm_head.w_out")

        float8_config.apply_float8_linear(self, modules_to_ignore=modules_to_ignore)

        self._fp8_enabled = True
        self._precompute_float8_dynamic_scale_for_fsdp = (
            float8_config.should_precompute_float8_dynamic_scale_for_fsdp
        )

    def apply_pp(self, pp_mesh: DeviceMesh):
        """
        Prepare the model for pipeline parallelism after it's been split into stages.
        """
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_pp(pp_mesh)
        self._pp_enabled = True
        self._pp_group_size = pp_mesh.size()

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: Optional[bool] = None):
        """
        Apply tensor parallelism to the model.

        :param loss_parallel: Set to ``True`` if parallelizing the loss function as well.
        :param float8_enabled: Set this to ``True`` if training with float8 linear layers.
        """
        if float8_enabled is None:
            float8_enabled = self.fp8_enabled
        elif not float8_enabled and self.fp8_enabled:
            raise OLMoConfigurationError(
                "Got 'float8_enabled=False', but FP8 has already been enabled"
            )

        if self.embeddings is not None:
            parallelize_module(
                self.embeddings,
                device_mesh=tp_mesh,
                parallelize_plan=RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                    use_local_output=False,
                ),
            )

        # Apply tensor/sequence parallelism to every transformer block.
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_tp(tp_mesh, input_layout=Shard(1), float8_enabled=float8_enabled)

        if self.lm_head is not None:
            self.lm_head.apply_tp(tp_mesh, input_layouts=(Shard(1), Replicate()))

        self._tp_enabled = True
        self._tp_mesh = tp_mesh

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        load_balancer: RingAttentionLoadBalancerType,
        head_stride: int = 1,
    ):
        """
        Prepare the model for context-parallelism (CP).

        :param cp_mesh: The CP device mesh.
        :param load_balancer: The load balancing method.
        """
        self._cp_load_balancer = load_balancer.build(cp_mesh)
        for block in self.blocks.values():
            cast(TransformerBlockBase, block).apply_cp(
                cp_mesh, load_balancer, head_stride=head_stride
            )
        if self.lm_head is not None:
            self.lm_head.apply_cp(cp_mesh, load_balancer)

    def apply_activation_checkpointing(
        self,
        mode: TransformerActivationCheckpointingMode,
        block_interval: Optional[int] = None,
        modules: Optional[List[str]] = None,
        activation_memory_budget: Optional[float] = None,
    ):
        """
        Apply activation checkpointing to the model.

        :param mode: Determines how to apply activation checkpointing.
        :param block_interval: Required when :data:`mode` is "selected_blocks". Determines
            which blocks are wrapped.
        :param modules: Required when :data:`mode` is "selected_modules". A list of modules names
            to wrap for activation checkpointing. Globs are supported.
        :param activation_memory_budget: The memory budget for activation checkpointing in the range
            [0, 1]. 0 corresponds to the memory usage when recomputing all activations, and 1
            corresponds to the memory usage when recomputing no activations (which is the default).
            Requires compilation to be enabled.
        """

        if mode == TransformerActivationCheckpointingMode.budget:
            if activation_memory_budget is None:
                raise ValueError("'activation_memory_budget' is required for 'budget' mode")
            if activation_memory_budget < 0 or activation_memory_budget > 1:
                raise ValueError("'activation_memory_budget' must be in the range [0, 1]")
            torch._functorch.config.activation_memory_budget = activation_memory_budget
            return

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
        preserve_rng_state = False

        if mode == TransformerActivationCheckpointingMode.selected_modules:
            from fnmatch import fnmatch

            assert modules is not None
            wrapped_modules: Set[str] = set()
            for name, module in self.named_modules():
                for pattern in modules:
                    if fnmatch(name, pattern):
                        break
                else:
                    continue

                if isinstance(module, MoEBase):
                    raise OLMoConfigurationError(
                        "Wrapping an entire MoE module for activation checkpointing is not supported. "
                        "Please try a finer-grained wrapping strategy."
                    )

                # NOTE: have to be careful not to try to wrap submodules of modules that have been wrapped.
                parent_name = ".".join(name.split(".")[:-1])
                if parent_name in wrapped_modules:
                    continue

                parent = self if not parent_name else self.get_submodule(parent_name)
                module = ptd_checkpoint_wrapper(module, preserve_rng_state=preserve_rng_state)
                parent.register_module(name.split(".")[-1], module)
                log.info(f"Wrapped '{name}' for activation checkpointing")
                wrapped_modules.add(name)
        else:
            for block_idx, block in enumerate(self.blocks.values()):
                if mode == TransformerActivationCheckpointingMode.selected_blocks:
                    assert block_interval is not None
                    if block_idx % block_interval == 0:
                        if isinstance(block, MoETransformerBlock):
                            raise OLMoConfigurationError(
                                "Wrapping MoE blocks for activation checkpointing is not supported."
                            )
                        block = ptd_checkpoint_wrapper(block, preserve_rng_state=preserve_rng_state)
                elif mode == TransformerActivationCheckpointingMode.full:
                    if isinstance(block, MoETransformerBlock):
                        raise OLMoConfigurationError(
                            "Wrapping MoE blocks for activation checkpointing is not supported."
                        )
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
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_compile()

        if self.lm_head is not None:
            self.lm_head.compile(fullgraph=False)

        self._compile_enabled = True

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
        prefetch_factor: int = 0,
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
        :prefetch_factor: For tuning the prefetch settings. 0 is the default, and higher values result
            in more aggressive prefetching.
        :wrapping_strategy: The wrapping strategy.
        """
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype or self.dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)
        # For PP, do not reshard after forward to avoid per-microbatch all-gathers,
        # which can be expensive and non-overlapped
        reshard_after_forward = False if pp_enabled else True

        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_fsdp(
                dp_mesh=dp_mesh,
                prefetch_factor=prefetch_factor,
                wrapping_strategy=wrapping_strategy,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
            )

        if self.embeddings is not None:
            fully_shard(
                self.embeddings,
                reshard_after_forward=reshard_after_forward,
                **fsdp_config,
            )
            # Embedding params are not needed for backwards computation.
            cast(FSDPModule, self.embeddings).set_unshard_in_backward(False)

        if (
            wrapping_strategy != TransformerDataParallelWrappingStrategy.blocks
            and self.lm_head is not None
        ):
            fully_shard(self.lm_head, reshard_after_forward=False, **fsdp_config)

        fully_shard(self, reshard_after_forward=reshard_after_forward, **fsdp_config)
        # Some inputs need to be on CPU initially, but FSDP will move everything to model's
        # device if we don't hide it.
        self.register_forward_pre_hook(_hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True)
        self.register_forward_pre_hook(
            _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        )

        if prefetch_factor > 0:
            blocks = cast(List[FSDPModule], list(self.blocks.values()))
            for i in range(len(blocks)):
                block = blocks[i]
                if i + 1 < len(blocks):
                    block.set_modules_to_forward_prefetch(blocks[i + 1 : i + 1 + prefetch_factor])
                elif isinstance(self.lm_head, FSDPModule):
                    block.set_modules_to_forward_prefetch([self.lm_head])

        self._fsdp_enabled = True

    def apply_ddp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        compile_enabled: bool = False,
        autograd_compile_enabled: bool = False,
    ):
        """
        Apply DDP to the model.
        """
        from torch.distributed._composable.replicate import replicate

        # Cast model explicitly to the specified dtype before applying DDP
        target_dtype = param_dtype or self.dtype
        if target_dtype != self.dtype:
            self.to(dtype=target_dtype)

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

    def post_batch(self, dry_run: bool = False):
        """
        Should be called right after the final backward of a complete batch but before the optimizer step.
        """
        del dry_run

    def post_optim_step(self):
        """
        Should be called right after an optimizer step.
        """
        if self.fp8_enabled and self._precompute_float8_dynamic_scale_for_fsdp:
            from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

            precompute_float8_dynamic_scale_for_fsdp(self)


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
        init_std: float = 0.02,
        block_overrides: Optional[Dict[int, TransformerBlockConfig]] = None,
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
            init_std=init_std,
            block_overrides=block_overrides,
        )

    def _validate_block(self, block: TransformerBlockBase) -> TransformerBlockBase:
        if not isinstance(block, NormalizedTransformerBlock):
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}' requires a '{NormalizedTransformerBlock.__name__}' block"
            )
        return block

    @torch.no_grad()
    def init_weights(self, *args, **kwargs) -> torch.Generator:
        generator = super().init_weights(*args, **kwargs)
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
        float8_enabled: Optional[bool] = None,
    ):
        del tp_mesh, float8_enabled

        raise NotImplementedError(
            "TP is not implemented yet for the normalized transformer variant"
        )

    def apply_compile(self):
        super().apply_compile()
        self.normalize_matrices = torch.compile(self.normalize_matrices)

    def post_optim_step(self):
        super().post_optim_step()
        self.normalize_matrices()


@beta_feature
class MoETransformer(Transformer):
    """
    An MoE transformer implementation, to be used with one of the
    :class:`MoETransformerBlock` block types.
    """

    @property
    def is_moe(self) -> bool:
        return True

    def compute_auxiliary_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        mean_offset = 1.0
        if self.pp_enabled:
            # Change the divisor to 'world_size // pp_group_size'
            mean_offset = self._pp_group_size

        out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}
        for block_idx, block in self.blocks.items():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            block_metrics = block.compute_metrics(reset=reset)
            for metric_name, (metric_val, reduce_type) in block_metrics.items():
                out[f"block {int(block_idx):02d}/{metric_name}"] = (
                    metric_val,
                    reduce_type,
                )

                if self.pp_enabled and reduce_type == ReduceType.mean:
                    metric_val = metric_val.float() * mean_offset

                if metric_name not in out:
                    out[metric_name] = (metric_val, reduce_type)
                elif reduce_type in (ReduceType.mean, ReduceType.sum):
                    out[metric_name] = (
                        out[metric_name][0] + metric_val,
                        reduce_type,
                    )
                elif reduce_type == ReduceType.max:
                    out[metric_name] = (
                        torch.max(out[metric_name][0], metric_val),
                        reduce_type,
                    )
                else:
                    raise NotImplementedError(reduce_type)
        return out

    def reset_auxiliary_metrics(self):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            cast(MoETransformerBlock, block).reset_metrics()

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            block.apply_ep(ep_mesh, **kwargs)

    def prepare_experts_for_fsdp(
        self,
        world_mesh: DeviceMesh,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
    ):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            reshard_after_forward = True
            if pp_enabled or block.ep_enabled or block.tp_enabled:
                reshard_after_forward = False
            block.feed_forward_moe.prepare_experts_for_fsdp(
                world_mesh=world_mesh,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=param_dtype or self.dtype, reduce_dtype=reduce_dtype
                ),
                reshard_after_forward=reshard_after_forward,
            )

    def prepare_experts_for_ddp(self, world_mesh: DeviceMesh):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            cast(MoETransformerBlock, block).feed_forward_moe.prepare_experts_for_ddp(
                world_mesh=world_mesh,
            )

    def post_batch(self, dry_run: bool = False):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            block.feed_forward_moe.post_batch(dry_run=dry_run)


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


def log1mexp(x):
    """Computes log(1 - exp(x)) in a numerically stable way for x < 0."""
    # For x < log(0.5), use log1p(-exp(x)) directly
    # For x >= log(0.5), use log(-expm1(x)) to avoid precision issues
    log_half = -math.log(2)
    return torch.where(x < log_half, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x)))


class BLTTransformer(Transformer):
    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        block: TransformerBlockConfig,
        local_encoder: LocalEncoderConfig,
        local_decoder: LocalDecoderConfig,
        lm_head: LMHeadConfig,
        dtype: torch.dtype = torch.float32,
        init_method: InitMethod = InitMethod.normal,
        init_device: str = "cpu",
        init_seed: int = 0,
        init_std: float = 0.02,
        block_overrides: Optional[Dict[int, TransformerBlockConfig]] = None,
    ):
        # accessed in super() so we need a placeholder
        self.num_non_embedding_params = -1
        self.num_params = -1

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
            init_std=init_std,
            block_overrides=block_overrides,
        )

        del self.num_non_embedding_params
        del self.num_params

        # replaced by local encoder, maybe move local encoder embeddings outside?
        self.embeddings = None
        # need local dimension, not global dimension as in super()
        self.lm_head = lm_head.build(
            d_model=local_decoder.d_model, vocab_size=vocab_size, init_device=init_device
        )

        self.local_encoder = local_encoder.build(vocab_size, d_global_model=d_model)
        self.local_decoder = local_decoder.build(vocab_size, d_global_model=d_model)

         # TODO(benjaminm): generalize
        self.space_mask_dolma2 = blt_utils.get_dolma2_space_mask()
        self.eos_token_dolma2 = 100257
        self.space_mask_blt = blt_utils.get_blt_space_mask()
        self.end_of_subword_token_blt = 3
        self.eos_token_blt = 1
        self.vocab_size_blt = (4 + 256)

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
        prefetch_factor: int = 0,
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
        :prefetch_factor: For tuning the prefetch settings. 0 is the default, and higher values result
            in more aggressive prefetching.
        :wrapping_strategy: The wrapping strategy.
        """
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype or self.dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)
        # For PP, do not reshard after forward to avoid per-microbatch all-gathers,
        # which can be expensive and non-overlapped
        reshard_after_forward = False if pp_enabled else True

        self.local_encoder.apply_fsdp(  # type: ignore
            dp_mesh=dp_mesh,
            prefetch_factor=prefetch_factor,
            wrapping_strategy=wrapping_strategy,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
        )
        # TODO(benjaminm): this seems to cause perf deg? only tested on single GPU.
        self.local_decoder.apply_fsdp(  # type: ignore
            dp_mesh=dp_mesh,
            prefetch_factor=prefetch_factor,
            wrapping_strategy=wrapping_strategy,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
        )

        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_fsdp(
                dp_mesh=dp_mesh,
                prefetch_factor=prefetch_factor,
                wrapping_strategy=wrapping_strategy,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
            )

        if (
            wrapping_strategy != TransformerDataParallelWrappingStrategy.blocks
            and self.lm_head is not None
        ):
            fully_shard(self.lm_head, reshard_after_forward=False, **fsdp_config)

        fully_shard(self, reshard_after_forward=reshard_after_forward, **fsdp_config)
        # Some inputs need to be on CPU initially, but FSDP will move everything to model's
        # device if we don't hide it.
        self.register_forward_pre_hook(_hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True)
        self.register_forward_pre_hook(
            _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        )

        if prefetch_factor > 0:
            blocks = cast(List[FSDPModule], list(self.blocks.values()))
            for i in range(len(blocks)):
                block = blocks[i]
                if i + 1 < len(blocks):
                    block.set_modules_to_forward_prefetch(blocks[i + 1 : i + 1 + prefetch_factor])
                elif isinstance(self.lm_head, FSDPModule):
                    block.set_modules_to_forward_prefetch([self.lm_head])

        self._fsdp_enabled = True

    @cached_property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @cached_property
    def num_non_embedding_params(self) -> int:
        num_embeddings = 0

        hash_embeddings = self.local_encoder.hash_embeddings if self.local_encoder.hash_embeddings is not None else []

        for embedding_module in [self.local_encoder.embedding] + list(hash_embeddings):  # type: ignore[attr-defined]
            num_embeddings += embedding_module.weight.numel()   # type: ignore[attr-defined]

        return self.num_params - num_embeddings

    def _prepare_inputs(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        blt_config: Optional[BLTConfig] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any], Dict[int, Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        input_ids, labels, all_block_kwargs, per_block_kwargs, lm_head_kwargs = super()._prepare_inputs(
            input_ids,
            labels,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            loss_div_factor=loss_div_factor,
            return_logits=return_logits,
            **kwargs,
        )

        local_encoder_kwargs = {}
        local_decoder_kwargs = {}
        extra_kwargs = {}

        patch_lens: Optional[torch.Tensor] = None
        patch_ids: Optional[torch.Tensor] = None
        original_input_ids: Optional[torch.Tensor] = None

        if (patch_lens := kwargs.pop("patch_lens", None)) is not None:
            if blt_config is not None and blt_config.patching == "space":
                patch_lens = kwargs["space_patch_lens"]
                assert patch_lens is not None, "space_patch_lens must be present if patch_lens is present"

            patch_lens = move_to_device(patch_lens, self.device)
            patch_ids = blt_utils.lengths_to_ids(patch_lens, input_ids.shape[-1])
            original_input_ids = kwargs.pop("original_input_ids", None) # must be present if patch_lens is present

        local_encoder_kwargs["patch_lens"] = patch_lens
        local_encoder_kwargs["patch_ids"] = patch_ids
        extra_kwargs["original_input_ids"] = move_to_device(original_input_ids, self.device)

        if (expanded_input_ids := kwargs.pop("expanded_input_ids", None)) is not None:
            local_encoder_kwargs["expanded_input_ids"] = move_to_device(expanded_input_ids, self.device)
        else:
            local_encoder_kwargs["expanded_input_ids"] = None

        if (teacher_inputs_embeds := kwargs.pop("teacher_inputs_embeds", None)) is not None:
            extra_kwargs["teacher_inputs_embeds"] = move_to_device(teacher_inputs_embeds, self.device)

        if blt_config is not None and blt_config.patching != "dolma2":
            # can't use attributes relying on dolma2 patching
            extra_kwargs["original_input_ids"] = None

        return (
            input_ids,
            labels,
            all_block_kwargs,
            per_block_kwargs,
            lm_head_kwargs,
            local_encoder_kwargs,
            local_decoder_kwargs,
            extra_kwargs,
        )

    def apply_compile(self):
        """
        Apply ``torch.compile()`` to each transformer block, which makes compilation efficient
        due to repeated structure.

        .. warning::
            This must be called after :meth:`apply_activation_checkpointing()` but
            before :meth:`apply_fsdp()` or :meth:`apply_ddp()`.
        """
        super().apply_compile()

        self.local_encoder.apply_compile()  # type: ignore
        self.local_decoder.apply_compile()  # type: ignore

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
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
        input_ids, labels, all_block_kwargs, per_block_kwargs, lm_head_kwargs, local_encoder_kwargs, local_decoder_kwargs, _ = self._prepare_inputs(
            input_ids,
            labels,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            loss_div_factor=loss_div_factor,
            return_logits=return_logits,
            **kwargs,
        )

        h_byte, h_patch, (_, boundary_logprobs), boundary_mask = self.local_encoder(
            input_ids,
            teacher_force_boundaries=False,
            **local_encoder_kwargs,
        )

        # TEMP DEBUG
        h_patch_global = h_patch.to(torch.bfloat16)

        # Run each block.
        for block_key, block in self.blocks.items():
            block_idx = int(block_key)
            block_kwargs = per_block_kwargs.get(block_idx, {})
            # Mark sizes as dynamic for torch.compile().
            if self.compile_enabled:
                mark_dynamic(h_patch_global, (0, 1), strict=False)
            h_patch_global = block(h_patch_global, **all_block_kwargs, **block_kwargs)

        h_patch_after_global = h_patch_global.to(h_patch.dtype)

        h_out = self.local_decoder(
            embeds=h_byte,
            patch_embeds=h_patch_after_global,
            patch_residuals=h_patch,
            boundary_logprobs=boundary_logprobs,
            boundary_mask=boundary_mask,
            **local_decoder_kwargs,
        )

        if labels is not None:
            lm_head_kwargs["labels"] = labels

        return self.lm_head(h_out, **lm_head_kwargs)


class BLTDistillTransformer(BLTTransformer):
    def __init__(self, *args, teacher: Optional[Transformer], share_blocks: bool, use_teacher_embs_with_vocab_size: Optional[int], dtype, init_device, **kwargs):
        super().__init__(*args, dtype=dtype, **kwargs)

        self.teacher = teacher
        self.share_blocks = share_blocks
        # if not None, keep & use the teacher embeddings (for distilling from stage1 local decoder transfer)
        self.use_teacher_embs_with_vocab_size = use_teacher_embs_with_vocab_size

        if self.use_teacher_embs_with_vocab_size is not None:
            self.teacher_embeddings = nn.Embedding(
                self.use_teacher_embs_with_vocab_size,
                self.d_model,
                dtype=dtype,
                device=init_device,
            )
        else:
            self.teacher_embeddings = None

        if self.teacher is not None and self.share_blocks:
            self.teacher.blocks.clear()

    # adapted from LMHead._finalize_loss
    def _finalize_loss(self, loss: torch.Tensor | float, loss_div_factor: Optional[Union[torch.Tensor, float]] = None):
        if self.lm_head.tp_enabled or self.lm_head.cp_enabled:
            # would need extra adjustments as per LMHead._finalize_loss
            raise NotImplementedError("Loss division is not implemented for TP or CP.")

        if loss_div_factor is not None:
            if isinstance(loss_div_factor, torch.Tensor):
                loss = loss / loss_div_factor
            else:
                loss = loss / float(loss_div_factor)

        return loss

    def apply_fp8(self, float8_config: Float8Config):
        """
        Use an FP8 recipe on the (potentially frozen) backbone layers.
        Do not use FP8 for local encoder / decoder.
        This is mainly to speed up Stage 1 training (where the backbone is frozen).
        """
        if not float8_config.enabled:
            return

        float8_config.apply_float8_linear(self.blocks)
        if self.teacher is not None:
            float8_config.apply_float8_linear(self.teacher.blocks)

        self._fp8_enabled = True
        self._precompute_float8_dynamic_scale_for_fsdp = (
            float8_config.should_precompute_float8_dynamic_scale_for_fsdp
        )

    def apply_compile(self):
        """
        Apply ``torch.compile()`` to each transformer block, which makes compilation efficient
        due to repeated structure.

        .. warning::
            This must be called after :meth:`apply_activation_checkpointing()` but
            before :meth:`apply_fsdp()` or :meth:`apply_ddp()`.
        """
        super().apply_compile()

        if self.teacher is not None:
            self.teacher.apply_compile()

    def apply_fsdp(self, *args, **kwargs):
        if self.teacher is not None:
            self.teacher.apply_fsdp(*args, **kwargs)

        super().apply_fsdp(*args, **kwargs)

    # teacher forward patched to (1) use the student blocks if blocks are shared, (2) return the required extra hidden state information
    def teacher_forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        skip_blocks: bool = False,
        zero_bos: bool = True,
        hidden_states_to_return: Optional[list[int]] = None,
        **kwargs,
    ) -> Tuple[Union[LMOutputWithLoss, None], Tuple[list[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """
        Run the transformer on the token input IDs.

        :param input_ids: The token input IDs, shape ``(batch_size, seq_len)``.

        :returns: The logits if ``labels`` is ``None`` or the losses if ``labels`` is not ``None``.
        """
        out_hidden_states = []

        if isinstance(self.teacher, BLTTransformer):
            raise NotImplementedError()
        elif isinstance(self.teacher, Transformer):
            input_ids, labels, all_block_kwargs, per_block_kwargs, lm_head_kwargs = self.teacher._prepare_inputs(
                input_ids,
                labels,
                ignore_index=ignore_index,
                loss_reduction=loss_reduction,
                z_loss_multiplier=z_loss_multiplier,
                loss_div_factor=loss_div_factor,
                return_logits=return_logits,
                **kwargs,
            )

            if inputs_embeds is not None:
                h_emb = inputs_embeds
            else:
                h_emb = self.teacher.embeddings(input_ids)

            if skip_blocks:
                return None, ([], None, h_emb)

            if self.share_blocks:
                blocks = self.blocks
            else:
                blocks = self.teacher.blocks

            # Run each block.
            dtype = h_emb.dtype
            global_dtype = torch.bfloat16 if isinstance(self.blocks["0"].attention.backend, FlashAttention2Backend) else dtype  # type: ignore

            h = h_emb.to(global_dtype)

            # Run each block.
            for block_key, block in blocks.items():
                block_idx = int(block_key)
                block_kwargs = per_block_kwargs.get(block_idx, {})
                # Mark sizes as dynamic for torch.compile().
                if self.compile_enabled:
                    mark_dynamic(h, (0, 1), strict=False)
                h = block(h, **all_block_kwargs, **block_kwargs)

                if block_idx in (hidden_states_to_return or []):
                    out_hidden_states.append(h.detach().clone().to(dtype))

            h = h.to(dtype)

            # Get final logits but again pass-through in case of pipeline parallelism.
            if self.teacher.lm_head is not None:
                if self.teacher.compile_enabled:
                    mark_dynamic(h, (0, 1), strict=False)
                    if labels is not None:
                        mark_dynamic(labels, (0, 1), strict=False)
                # NOTE: When TP is active we can't pass 'labels=None' or the hook from 'PrepareModuleInput'
                # will throw an exception.
                if labels is not None:
                    lm_head_kwargs["labels"] = labels

                out = LMOutputWithLoss(
                    logits=self.teacher.lm_head(h, **lm_head_kwargs),
                    loss=torch.tensor(torch.nan),
                    ce_loss=torch.tensor(torch.nan),
                    z_loss=None,
                )

                return out, (out_hidden_states, h, h_emb)
            else:
                return None, (out_hidden_states, h, h_emb)
        else:
            return None, ([], None, None)

    def fix_init(self, blt_config, embedding_init_path: Optional[str] = None):
        # fsdp requires fwd through root first
        dummy_size = 128
        self(
            input_ids=torch.zeros((1, dummy_size), dtype=torch.long, device=self.device),
            expanded_input_ids=torch.zeros((1, dummy_size), dtype=torch.long, device=self.device),
            labels=torch.zeros((1, dummy_size), dtype=torch.long, device=self.device),
            patch_lens=torch.ones((1, dummy_size), dtype=torch.long, device=self.device),
            original_input_ids=torch.zeros((1, dummy_size), dtype=torch.long, device=self.device),
            blt_config=blt_config
        )
        teacher_embs = self.teacher.embeddings.weight if self.teacher is not None else self.teacher_embeddings.weight  # type: ignore
        self.local_encoder.fix_init(embedding_init_path, teacher_embs)  # type: ignore

        for block in list(self.local_encoder.blocks.values()) + list(self.local_decoder.blocks.values()):  # type: ignore
            if hasattr(block, "xlstm"):
                # disable input gate
                block.xlstm.igate_preact.bias.data.fill_(blt_config.xlstm_igate_bias_init)  # type: ignore

    def _rep_compare_fn(self, blt_config: BLTConfig):
        if blt_config.rep_compare_fn == "l2":
            def l2_compare_fn(x, y):
                return torch.linalg.norm(x - y, dim=-1) / math.sqrt(x.shape[-1])

            rep_compare_fn = l2_compare_fn
        elif blt_config.rep_compare_fn == "cos_dist":
            def cos_dist_compare_fn(x, y):
                return 1 - F.cosine_similarity(x, y, dim=-1)

            rep_compare_fn = cos_dist_compare_fn
        elif blt_config.rep_compare_fn == "l2_rmsnorm":
            def l2_rmsnorm_compare_fn(x, y):
                uncentered_y_std = torch.sqrt(
                    torch.mean(torch.square(y), dim=-1, keepdim=True).clip(blt_config.epsilon)
                )

                return torch.linalg.norm((x - y) / uncentered_y_std, dim=-1) / math.sqrt(x.shape[-1])

            rep_compare_fn = l2_rmsnorm_compare_fn
        else:
            raise ValueError(f"Unknown distillation rep_compare_fn '{blt_config.rep_compare_fn}'")

        return rep_compare_fn

    def _compute_hnet_embed_loss(
        self,
        h_byte,
        teacher_embeds,
        byte_mask,
        true_patch_ids,
        blt_config,
        metrics={},
    ):
        rep_compare_fn = self._rep_compare_fn(blt_config)

        teacher_embs_repeated = torch.gather(
            teacher_embeds,
            dim=1,
            index=true_patch_ids.clip(max=teacher_embeds.shape[1] - 1).unsqueeze(-1).expand(-1, -1, teacher_embeds.shape[-1]),
        )

        elementwise_hnet_embed_loss = rep_compare_fn(
            h_byte[:, 1:], # skip first embedding to produce offset as in H-Net paper (match first patch byte to prev emb)
            teacher_embs_repeated[:, :-1]
        )
        hnet_embed_loss_mask = byte_mask[:, 1:]

        hnet_embed_loss = (elementwise_hnet_embed_loss * hnet_embed_loss_mask.float()).mean()
        metrics[f"blt/hnet_embed_loss"] = hnet_embed_loss / hnet_embed_loss_mask.float().mean()
        return hnet_embed_loss

    def _compute_ratio_loss(
        self,
        boundary_logprobs,
        boundary_mask,
        byte_mask,
        blt_config,
        metrics={},
    ):
        true_ratio = (boundary_mask * byte_mask).float().mean() / byte_mask.float().mean()
        average_prob = (torch.exp(boundary_logprobs) * byte_mask).float().mean() / byte_mask.float().mean()

        ratio_loss = (
            (1 - true_ratio) * (1 - average_prob) +
            (true_ratio) * (average_prob) * (blt_config.target_ratio - 1)
        ) * blt_config.target_ratio / (blt_config.target_ratio - 1)
        metrics["blt/ratio_loss"] = ratio_loss
        return ratio_loss

    def _compute_alm_style_loss(
        self,
        logprobs: torch.Tensor,
        main_path_logprobs: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        teacher_main_path_logprobs: torch.Tensor,
        byte_mask: torch.Tensor,
        patch_mask: torch.Tensor,
        true_patch_ids: torch.Tensor,
        true_patch_lens: torch.Tensor,
        debiasing_logprobs: Optional[torch.Tensor],
        blt_config: BLTConfig,
        metrics={},
    ):
        if blt_config.div_fn == "kl":
            def kl_div_fn(log_y_true, log_y_pred):
                log_y_true = (log_y_true.float() / blt_config.binarization_temp) - blt_config.epsilon
                log_y_pred = (log_y_pred.float() / blt_config.binarization_temp) - blt_config.epsilon

                e = (
                    torch.exp(log_y_true) * log_y_true
                    + (-torch.expm1(log_y_true) * log1mexp(log_y_true))
                )
                ce = (
                    torch.exp(log_y_true) * log_y_pred
                    + (-torch.expm1(log_y_true) * log1mexp(log_y_pred))
                )

                return (e - ce)

            div_fn = kl_div_fn
        elif blt_config.div_fn == "reverse_kl":
            def reverse_kl_div_fn(log_y_true, log_y_pred):
                log_y_true = (log_y_true.float() / blt_config.binarization_temp) - blt_config.epsilon
                log_y_pred = (log_y_pred.float() / blt_config.binarization_temp) - blt_config.epsilon

                e = (
                    torch.exp(log_y_pred) * log_y_pred
                    + (-torch.expm1(log_y_pred) * log1mexp(log_y_pred))
                )
                ce = (
                    torch.exp(log_y_pred) * log_y_true
                    + (-torch.expm1(log_y_pred) * log1mexp(log_y_true))
                )

                return (e - ce)

            div_fn = reverse_kl_div_fn
        elif blt_config.div_fn == "tvd":
            def tvd_div_fn(log_y_true, log_y_pred):
                log_y_true = (log_y_true.float() / blt_config.binarization_temp) - blt_config.epsilon
                log_y_pred = (log_y_pred.float() / blt_config.binarization_temp) - blt_config.epsilon

                # TODO(benjaminm): how does this scale with temp?
                return torch.abs(torch.exp(log_y_true) - torch.exp(log_y_pred))

            div_fn = tvd_div_fn
        elif blt_config.div_fn == "tvd_temp_limit":
            def tvd_temp_limit_div_fn(log_y_true, log_y_pred):
                return torch.abs(log_y_true - log_y_pred)

            div_fn = tvd_temp_limit_div_fn
        else:
            raise ValueError(f"Unknown distillation div_fn '{blt_config.div_fn}'")

        main_path_patch_logprobs = torch.zeros((patch_mask.shape[0], patch_mask.shape[1]), device=main_path_logprobs.device, dtype=main_path_logprobs.dtype)
        #assert (patch_ids[:, 2:] - 1).max().item() < main_path_patch_logprobs.shape[1]
        #assert (patch_ids[:, 2:] - 1).min().item() >= 0
        patch_ids_to_select = true_patch_ids[:, 1:] - 1
        main_path_patch_logprobs = main_path_patch_logprobs.scatter_reduce(
            src=main_path_logprobs,
            dim=1,
            index=patch_ids_to_select,
            reduce="sum",
            include_self=False,
        )

        y_hat = main_path_patch_logprobs[:, :-1]
        y_true = teacher_main_path_logprobs

        if blt_config.do_alm_debiasing:
            if debiasing_logprobs is None:
                space_mask_padded_blt = F.pad(
                    self.space_mask_blt.to(y_hat.device),
                    (0, logprobs.shape[-1] - len(self.space_mask_blt)),
                    value=0
                )[None, None, :]
                space_mask_padded_dolma2 = F.pad(
                    self.space_mask_dolma2.to(y_true.device),
                    (0, teacher_logprobs.shape[-1] - len(self.space_mask_dolma2)),
                    value=0
                )[None, None, :]
                patch_end_indices = torch.cumsum(true_patch_lens, dim=1) - 1
                minus_inf = torch.tensor(float('-inf'), device=logprobs.device)
                y_space_hat_all = torch.where(space_mask_padded_blt.bool(), logprobs, minus_inf).logsumexp(dim=-1)  
                y_space_hat = torch.gather(y_space_hat_all, dim=1, index=patch_end_indices[:, 1:])
                y_space_true = torch.where(space_mask_padded_dolma2.bool(), teacher_logprobs[:, 1:], minus_inf).logsumexp(dim=-1)

                y_hat = y_hat + y_space_hat
                y_true = y_true + y_space_true
            else:
                y_hat = y_hat + debiasing_logprobs

        local_decoder_loss_simple = (div_fn(y_true, y_hat) * patch_mask[:, :-1]).mean()
        metrics["blt/local_decoder_teacher_mean_p_simple"] = (torch.exp(y_true) * patch_mask[:, :-1]).mean() / (patch_mask[:, :-1].float().mean() + blt_config.epsilon)
        metrics["blt/local_decoder_loss_simple"] = local_decoder_loss_simple / (patch_mask[:, :-1].float().mean() + blt_config.epsilon)
        metrics["blt/local_decoder_mae_simple"] = (torch.abs(y_true - y_hat) * patch_mask[:, :-1]).mean() / (patch_mask[:, :-1].float().mean() + blt_config.epsilon)

        return local_decoder_loss_simple, metrics

    def _compute_local_encoder_loss(
        self,
        h_patch,
        teacher_embeds,
        patch_mask,
        seq_sorted_indices,
        true_patch_ids,
        blt_config,
        student_hidden_states=None,
        teacher_hidden_states=None,
        metrics={}
    ):
        teacher_indices_to_select = torch.gather(
            true_patch_ids,
            dim=1,
            index=seq_sorted_indices,
        )
        mask = patch_mask & (teacher_indices_to_select < teacher_embeds.shape[1])
        teacher_indices_to_select = torch.where(
            mask,
            teacher_indices_to_select,
            torch.zeros_like(teacher_indices_to_select),
        ).unsqueeze(-1).expand(-1, -1, teacher_embeds.shape[-1])
        aligned_teacher_embeds = torch.gather(
            teacher_embeds,
            dim=1,
            index=teacher_indices_to_select,
        )

        rep_compare_fn = self._rep_compare_fn(blt_config)

        elementwise_local_encoder_loss = rep_compare_fn(h_patch, aligned_teacher_embeds)
        local_encoder_loss = (elementwise_local_encoder_loss * mask.float()).mean()
        metrics["blt/local_encoder_loss"] = local_encoder_loss / (mask.float().mean() + blt_config.epsilon)
        metrics["blt/local_encoder_cos_sim"] = (F.cosine_similarity(
            h_patch.float(),
            aligned_teacher_embeds.float(),
            dim=-1,
        ) * mask.float()).mean() / (mask.float().mean() + blt_config.epsilon)

        local_encoder_loss *= blt_config.encoder_loss_no_lookahead_weight

        for lookahead_idx in range(blt_config.encoder_loss_lookahead):
            assert student_hidden_states is not None
            assert teacher_hidden_states is not None
            
            current_aligned_teacher_embeds = torch.gather(
                teacher_hidden_states[lookahead_idx],
                dim=1,
                index=teacher_indices_to_select,
            )

            elementwise_local_encoder_loss = rep_compare_fn(
                student_hidden_states[lookahead_idx],
                current_aligned_teacher_embeds,
            )
            local_encoder_loss_lookahead = (elementwise_local_encoder_loss * mask.float()).mean()
            metrics[f"blt/local_encoder_loss_lookahead_{lookahead_idx}"] = local_encoder_loss_lookahead / (mask.float().mean() + blt_config.epsilon)
            metrics[f"blt/local_encoder_cos_sim_lookahead_{lookahead_idx}"] = (F.cosine_similarity(
                student_hidden_states[lookahead_idx].float(),
                current_aligned_teacher_embeds.float(),
                dim=-1,
            ) * mask.float()).mean() / (mask.float().mean() + blt_config.epsilon)

            local_encoder_loss += local_encoder_loss_lookahead * blt_config.encoder_loss_lookahead_weights[lookahead_idx]

        return local_encoder_loss

    def _block_forward(
        self,
        h_patch: torch.Tensor,
        hidden_states_to_return: Optional[list[int]] = None,
        limit: Optional[int] = None,
        all_block_kwargs: Optional[Dict[str, Any]] = None,
        per_block_kwargs: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h_patch_global = h_patch

        out_hidden_states = []

        for block_key, block in self.blocks.items():
            block_idx = int(block_key)

            if limit is not None and block_idx + 1 > limit:
                break

            all_block_kwargs = all_block_kwargs or {}
            block_kwargs = per_block_kwargs.get(block_idx, {}) if per_block_kwargs is not None else {}

            # Mark sizes as dynamic for torch.compile().
            if self.compile_enabled:
                mark_dynamic(h_patch_global, (0, 1), strict=False)

            # TODO(benjaminm): possibly no_grad for non-trainable layers
            h_patch_global = block(h_patch_global, **all_block_kwargs, **block_kwargs)

            if int(block_idx) in (hidden_states_to_return or []):
                out_hidden_states.append(h_patch_global)

        return h_patch_global, out_hidden_states

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        patch_loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        blt_config: Optional[BLTConfig] = None,
        **kwargs,
    ):
        if blt_config is None:
            raise ValueError("`blt_config` must be provided for BLTDistillTransformer.forward")

        skip_blocks = blt_config.skip_blocks
        skip_teacher_blocks = blt_config.skip_teacher_blocks
        skip_teacher = blt_config.skip_teacher

        input_ids, labels, all_block_kwargs, per_block_kwargs, lm_head_kwargs, local_encoder_kwargs, local_decoder_kwargs, extra_kwargs = self._prepare_inputs(
            input_ids,
            labels,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            loss_div_factor=loss_div_factor,
            blt_config=blt_config,
            return_logits=True,  # needed for distillation
            **kwargs,
        )
        assert labels is not None

        # losses (incl. distillation losses)
        metrics = {}

        # loss masks
        byte_mask = labels != ignore_index
        shifted_patch_lens = F.pad(
            local_encoder_kwargs["patch_lens"][:, 1:],
            (0, 1),
        )
        patch_mask = shifted_patch_lens != 0

        patch_end_indices = torch.cumsum(local_encoder_kwargs["patch_lens"], dim=1) - 1
        patch_end_indices = torch.where(
            patch_end_indices < byte_mask.shape[1],
            patch_end_indices,
            torch.zeros_like(patch_end_indices), # effectively mask out, index 0 is always start
        )

        boundary_labels = torch.zeros_like(byte_mask, dtype=torch.float32)
        boundary_labels.scatter_(1, patch_end_indices, 1.0)
        if blt_config.skip_boundary_before_eos:
            boundary_labels[:, :-1] = torch.where( # no boundary before eos
                input_ids[:, 1:] == self.eos_token_blt,
                0.0,
                boundary_labels[:, :-1]
            )
            boundary_labels[:, 0] = 1 # bos must still be boundary

        h_byte, h_patch, boundary_logprobs, boundary_mask = self.local_encoder(
            input_ids,
            boundary_predictor_backprop_through_encoder=blt_config.boundary_predictor_backprop_through_encoder,
            teacher_force_boundaries=blt_config.teacher_force_boundaries,
            boundary_threshold=blt_config.boundary_threshold,
            true_boundary_mask=boundary_labels > 0.5,
            **local_encoder_kwargs,
        )

        L = byte_mask.shape[1]
        token_idx = (
            torch.arange(L, device=byte_mask.device)[None, :] + (~boundary_mask).long() * L  # type: ignore
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)[:, :patch_mask.shape[1]]
        last_increasing_index = ((seq_sorted_indices[:, 1:] - seq_sorted_indices[:, :-1]) < 0).max(-1)
        patch_mask = (
            (torch.arange(patch_mask.shape[1], device=patch_mask.device)[None, :] <= last_increasing_index.indices[:, None]) |
            (torch.zeros_like(patch_mask) == last_increasing_index.values[:, None]) # case where never not increasing (no padding)
        )
        patch_ids = torch.cumsum(boundary_mask.flip(1), -1).flip(1)
        patch_ids = (patch_ids.max(1, keepdim=True).values - patch_ids).clip(max=patch_mask.shape[1] - 1)
        patch_ids = torch.where(byte_mask, patch_ids, torch.full_like(patch_ids, fill_value=patch_mask.shape[1] - 1)) # TODO(benjaminm): need to adjust byte mask?
        seq_sorted_indices = torch.where(
            patch_mask,
            seq_sorted_indices,
            torch.ones_like(seq_sorted_indices),
        )
        if self.local_decoder.fuse_boundaries:
            shift_boundary_mask = (boundary_mask[:, 1:] & byte_mask[:, 1:])
            label_offsets = shift_boundary_mask * self.vocab_size_blt
            labels[:, :-1] += label_offsets

        # compute the boundary loss
        elementwise_boundary_loss = blt_utils.binary_cross_entropy_with_logprobs(
            boundary_logprobs,
            boundary_labels,
        )
        boundary_byte_mask = byte_mask.clone()
        boundary_byte_mask[:, byte_mask.shape[1]-self.local_encoder.boundary_predictor_lookahead:] = False  # type: ignore
        boundary_loss = (elementwise_boundary_loss * boundary_byte_mask).mean()
        elementwise_boundary_acc = (torch.exp(boundary_logprobs) > 0.5) == (boundary_labels > 0)
        boundary_acc = (elementwise_boundary_acc * boundary_byte_mask).float().mean()
        metrics["blt/boundary_loss"] = boundary_loss / boundary_byte_mask.float().mean()
        metrics["blt/boundary_acc"] = boundary_acc / boundary_byte_mask.float().mean()
        metrics["blt/boundary_mean"] = (boundary_mask * boundary_byte_mask).float().mean() / boundary_byte_mask.float().mean()
        metrics["blt/boundary_label_mean"] = (boundary_labels * boundary_byte_mask).float().mean() / boundary_byte_mask.float().mean()

        # First, run the teacher.
        if not skip_teacher:
            with (torch.no_grad() if blt_config.teacher_blocks_no_grad else nullcontext()):
                assert not isinstance(self.teacher, BLTTransformer)
                input_ids_for_teacher = torch.concatenate(
                    [
                        torch.full(
                            (extra_kwargs["original_input_ids"].shape[0], 1),
                            fill_value=self.eos_token_dolma2,
                            dtype=extra_kwargs["original_input_ids"].dtype,
                            device=extra_kwargs["original_input_ids"].device
                        ),
                        extra_kwargs["original_input_ids"][:, :-1]
                    ],
                    1,
                )
                teacher_loss_mask = shifted_patch_lens != 0

                if blt_config.use_student_patch_reps_for_teacher:
                    inputs_embeds_for_teacher = h_patch
                else:
                    inputs_embeds_for_teacher = None

                teacher_out, (teacher_hidden_states, teacher_last_hidden_state, teacher_embeds) = self.teacher_forward(
                    input_ids_for_teacher,
                    inputs_embeds=inputs_embeds_for_teacher,
                    labels=None, # we will compute loss ourselves
                    return_logits=True,
                    skip_blocks=skip_blocks or skip_teacher_blocks,
                    zero_bos=True,
                    hidden_states_to_return=list(range(blt_config.encoder_loss_lookahead)),
                    **kwargs,
                )
                teacher_logits = teacher_out.logits if teacher_out is not None else None
                if teacher_logits is not None:
                    teacher_logprobs = F.log_softmax(teacher_logits.float() / blt_config.temperature, dim=-1) # type: ignore
                    teacher_main_path_logprobs = torch.gather(teacher_logprobs[:, :-1], -1, input_ids_for_teacher[:, 1:].unsqueeze(-1)).squeeze(-1)

                    # behind flag since it's compute intensive
                    if blt_config.compute_teacher_ce:
                        teacher_labels = F.pad(
                            input_ids_for_teacher[:, 1:],
                            (0, 1),
                            value=ignore_index,
                        )

                        teacher_ce_loss, _ = cross_entropy_loss(
                            teacher_logits.view(-1, teacher_logits.shape[-1]),  # type: ignore
                            teacher_labels.view(-1),
                            ignore_index=ignore_index,
                        )
                        metrics["blt/teacher_ce_loss"] = teacher_ce_loss
                    else:
                        teacher_ce_loss = torch.nan
                else:
                    teacher_logprobs = None
                    teacher_main_path_logprobs = None
                    teacher_ce_loss = torch.nan
        else:
            teacher_embeds = None
            teacher_last_hidden_state = None
            teacher_logprobs = None
            teacher_main_path_logprobs = None
            teacher_hidden_states = None
            teacher_loss_mask = None
            teacher_ce_loss = torch.nan

        # Run each block.
        if not skip_blocks:
            if blt_config.use_oracle_patch_reps:
                assert teacher_last_hidden_state is not None
                assert blt_config.teacher_force_boundaries

                if blt_config.skip_boundary_before_eos:
                    # need to align
                    teacher_indices_to_select = torch.gather(
                        local_encoder_kwargs["patch_ids"],
                        dim=1,
                        index=seq_sorted_indices,
                    )
                    mask = patch_mask & (teacher_indices_to_select < teacher_last_hidden_state.shape[1])
                    teacher_indices_to_select = torch.where(
                        mask,
                        teacher_indices_to_select,
                        torch.zeros_like(teacher_indices_to_select),
                    ).unsqueeze(-1).expand(-1, -1, teacher_last_hidden_state.shape[-1])
                    h_patch_after_global = torch.gather(
                        teacher_last_hidden_state,
                        dim=1,
                        index=teacher_indices_to_select,
                    )
                else:
                    h_patch_after_global = teacher_last_hidden_state

                with (torch.no_grad() if blt_config.student_blocks_no_grad else nullcontext()):
                    _, student_hidden_states = self._block_forward(
                        h_patch,
                        hidden_states_to_return=list(range(blt_config.encoder_loss_lookahead)),
                        limit=blt_config.encoder_loss_lookahead,
                        all_block_kwargs=all_block_kwargs,
                        per_block_kwargs=per_block_kwargs,
                    )
            else:
                dtype = h_patch.dtype
                global_dtype = torch.bfloat16 if isinstance(self.blocks["0"].attention.backend, FlashAttention2Backend) else dtype  # type: ignore

                with (torch.no_grad() if blt_config.student_blocks_no_grad else nullcontext()):
                    h_patch_after_global, student_hidden_states = self._block_forward(
                        h_patch.to(global_dtype),
                        hidden_states_to_return=list(range(blt_config.encoder_loss_lookahead)),
                        all_block_kwargs=all_block_kwargs,
                        per_block_kwargs=per_block_kwargs,
                    )
                h_patch_after_global = h_patch_after_global.to(dtype)
                student_hidden_states = [x.to(dtype) for x in student_hidden_states]

            if blt_config.decoder_backprop_through_encoder:
                h_byte_for_decoder = h_byte
                h_patch_after_global_for_decoder = h_patch_after_global
                h_patch_for_decoder = h_patch
            else:
                h_byte_for_decoder = h_byte.detach()
                h_patch_after_global_for_decoder = h_patch_after_global.detach()
                h_patch_for_decoder = h_patch.detach()

            if blt_config.decoder_backprop_through_boundary_predictor:
                boundary_logprobs_for_decoder = boundary_logprobs if boundary_logprobs is not None else None
            else:
                boundary_logprobs_for_decoder = boundary_logprobs.detach() if boundary_logprobs is not None else None

            (h_out_for_true_boundaries, h_out_for_all_boundaries), h_out_for_logits, _ = self.local_decoder(
                embeds=h_byte_for_decoder,
                patch_embeds=h_patch_after_global_for_decoder,
                patch_residuals=h_patch_for_decoder,
                boundary_logprobs=boundary_logprobs_for_decoder,
                boundary_mask=boundary_mask,
                **local_decoder_kwargs,
            )

            if self.local_decoder.fuse_boundaries:
                # no separate boundary logits
                true_boundary_logits = None
                all_boundary_logits = None
            else:
                true_boundary_logits = self.lm_head(h_out_for_true_boundaries, **lm_head_kwargs)
                all_boundary_logits = self.lm_head(h_out_for_all_boundaries, **lm_head_kwargs)

            logits = self.lm_head(h_out_for_logits, **lm_head_kwargs)
            logprobs = F.log_softmax(logits.float() / blt_config.temperature, dim=-1)
            main_path_logprobs = torch.gather(logprobs, -1, labels[:, :-1].clip(min=0).unsqueeze(-1)).squeeze(-1)
        else:
            student_hidden_states = None
            true_boundary_logits = None
            all_boundary_logits = None
            logits = None
            logprobs = None
            main_path_logprobs = None

        # compute CE
        if not skip_blocks:
            assert logits is not None
            assert labels is not None

            ce_loss, _ = cross_entropy_loss(logits.view(-1, logits.shape[-1]), labels[:, :logits.shape[1]].reshape(-1))
            metrics["blt/ce_loss"] = ce_loss

            if not self.local_decoder.fuse_boundaries:
                assert true_boundary_logits is not None
                assert all_boundary_logits is not None

                if blt_config.merge_boundary_loss:
                    boundary_ce_loss, _ = cross_entropy_loss(
                        true_boundary_logits.view(-1, true_boundary_logits.shape[-1]),
                        torch.full(
                            (true_boundary_logits.shape[0] * true_boundary_logits.shape[1],),
                            fill_value=self.end_of_subword_token_blt,
                            device=true_boundary_logits.device,
                            dtype=torch.long
                        ),
                    )
                    metrics["blt/boundary_ce_loss"] = boundary_ce_loss

                    ce_loss = ce_loss + boundary_ce_loss * patch_mask.shape[1] / byte_mask.shape[1]
                    output_boundary_loss = torch.nan
                else:
                    all_output_boundary_logprobs = F.log_softmax(all_boundary_logits, dim=-1)[..., self.end_of_subword_token_blt]

                    if blt_config.use_output_boundary_jsd:
                        elementwise_boundary_jsd_loss = blt_utils.jsd(
                            all_output_boundary_logprobs,
                            boundary_logprobs[:, 1:],
                        )
                        output_boundary_loss = (elementwise_boundary_jsd_loss * boundary_byte_mask[:, 1:]).mean()
                        metrics["blt/output_boundary_loss"] = output_boundary_loss / (boundary_byte_mask[:, 1:].float().mean() + blt_config.epsilon)
                    else:
                        # shift one
                        elementwise_boundary_ce_loss = blt_utils.binary_cross_entropy_with_logprobs(
                            all_output_boundary_logprobs,
                            boundary_mask[:, 1:],
                        )
                        output_boundary_loss = (elementwise_boundary_ce_loss * boundary_byte_mask[:, 1:]).mean()
                        metrics["blt/output_boundary_loss"] = output_boundary_loss / (boundary_byte_mask[:, 1:].float().mean() + blt_config.epsilon)

                    metrics["blt/output_boundary_logmae"] = (
                        torch.abs(all_output_boundary_logprobs - boundary_logprobs[:, 1:]) * boundary_byte_mask[:, 1:]
                    ).mean() / (boundary_byte_mask[:, 1:].float().mean() + blt_config.epsilon)

                true_boundary_positives = boundary_mask[:, 1:] & boundary_byte_mask[:, 1:]
                true_boundary_negatives = (~boundary_mask[:, 1:]) & boundary_byte_mask[:, 1:]

                metrics["blt/boundary_true_positives"] = (
                    ((all_boundary_logits.argmax(-1) == self.end_of_subword_token_blt) & true_boundary_positives).float().mean()
                    / (true_boundary_positives.float().mean() + blt_config.epsilon)
                )
                metrics["blt/boundary_true_negatives"] = (
                    ((all_boundary_logits.argmax(-1) != self.end_of_subword_token_blt) & true_boundary_negatives).float().mean()
                    / (true_boundary_negatives.float().mean() + blt_config.epsilon)
                )
            else:
                output_boundary_loss = torch.nan
        else:
            output_boundary_loss = torch.nan
            ce_loss = torch.nan

        # could also have some version of the encoder loss for BLT teacher but not implemented for now
        if not skip_teacher and not isinstance(self.teacher, BLTTransformer) and teacher_embeds is not None:
            local_encoder_loss = self._compute_local_encoder_loss(
                h_patch=h_patch,
                teacher_embeds=teacher_embeds,
                patch_mask=patch_mask,
                seq_sorted_indices=seq_sorted_indices,
                true_patch_ids=local_encoder_kwargs["patch_ids"],
                blt_config=blt_config,
                student_hidden_states=student_hidden_states,
                teacher_hidden_states=teacher_hidden_states,
                metrics=metrics,
            )
        else:
            local_encoder_loss = torch.nan

        if not skip_blocks and not skip_teacher and not skip_teacher_blocks:
            assert logprobs is not None
            assert main_path_logprobs is not None
            assert teacher_logprobs is not None
            assert teacher_main_path_logprobs is not None
            assert teacher_embeds is not None
            assert teacher_loss_mask is not None

            if not self.local_decoder.fuse_boundaries and blt_config.teacher_force_boundaries:
                assert true_boundary_logits is not None

                debiasing_logprobs = F.log_softmax(
                    true_boundary_logits.float() / blt_config.temperature,
                    dim=-1
                )[..., self.end_of_subword_token_blt]  # type: ignore
            else:
                debiasing_logprobs = None

            local_decoder_loss, metrics = self._compute_alm_style_loss(
                logprobs,
                main_path_logprobs,
                teacher_logprobs,
                teacher_main_path_logprobs,
                byte_mask,
                teacher_loss_mask,
                local_encoder_kwargs["patch_ids"],
                local_encoder_kwargs["patch_lens"],
                debiasing_logprobs,
                blt_config,
                metrics,
            )
        else:
            local_decoder_loss = torch.nan

        # H-Net style embedding loss
        if teacher_embeds is not None and not isinstance(self.teacher, BLTTransformer):
            hnet_embed_loss = self._compute_hnet_embed_loss(
                h_byte=h_byte,
                teacher_embeds=teacher_embeds,
                byte_mask=byte_mask,
                true_patch_ids=local_encoder_kwargs["patch_ids"],
                blt_config=blt_config,
                metrics=metrics,
            )
        else:
            hnet_embed_loss = torch.nan

        # H-Net ratio loss
        if boundary_logprobs is not None:
            ratio_loss = self._compute_ratio_loss(
                boundary_logprobs,
                boundary_mask,
                boundary_byte_mask,
                blt_config,
                metrics=metrics,
            )
        else:
            ratio_loss = torch.nan

        # finalize losses
        # NOTE: loss_div_factor is at *byte sequence level*.
        if loss_div_factor is not None:
            loss_div_factor = loss_div_factor / (h_byte.shape[0] * h_byte.shape[1])

        if patch_loss_div_factor is not None:
            patch_loss_div_factor = patch_loss_div_factor / (h_patch.shape[0] * h_patch.shape[1])

        if loss_div_factor is not None and patch_loss_div_factor is not None:
            combined_loss_div_factor = loss_div_factor + patch_loss_div_factor
        else:
            combined_loss_div_factor = None

        # TODO(benjaminm): fix if possible by accumulating div factor across batches and dividing at the end
        if blt_config.merge_boundary_loss:
            ce_loss = self._finalize_loss(ce_loss, loss_div_factor=combined_loss_div_factor)
        else:
            ce_loss = self._finalize_loss(ce_loss, loss_div_factor=loss_div_factor)
        boundary_loss = self._finalize_loss(boundary_loss, loss_div_factor=loss_div_factor)
        output_boundary_loss = self._finalize_loss(output_boundary_loss, loss_div_factor=loss_div_factor)
        local_encoder_loss = self._finalize_loss(local_encoder_loss, loss_div_factor=patch_loss_div_factor)
        local_decoder_loss = self._finalize_loss(local_decoder_loss, loss_div_factor=loss_div_factor)
        hnet_embed_loss = self._finalize_loss(hnet_embed_loss, loss_div_factor=loss_div_factor)
        teacher_ce_loss = self._finalize_loss(teacher_ce_loss, loss_div_factor=patch_loss_div_factor)

        loss = 0.0
        for loss_idx, (loss_name, loss_weight) in enumerate(zip(blt_config.losses, blt_config.loss_weights)):
            if blt_config.loss_schedules is not None:
                schedule = blt_config.loss_schedules[loss_idx]
                if schedule.startswith("linear_decrease"):
                    _, n_steps = schedule.split(":")
                    n_steps = int(n_steps)

                    schedule_multiplier = max(1.0 - kwargs["step"] / n_steps, 0.0)
                elif schedule == "constant":
                    schedule_multiplier = 1.0
                else:
                    raise ValueError(f"Unknown loss schedule '{schedule}'")

                metrics[f"blt/{loss_name}_loss_schedule_multiplier"] = schedule_multiplier
            else:
                schedule_multiplier = 1.0

            if loss_weight == 0.0:
                continue
            if loss_name == "ce":
                loss = loss + ce_loss * loss_weight * schedule_multiplier
            elif loss_name == "local_encoder":
                loss = loss + local_encoder_loss * loss_weight * schedule_multiplier
            elif loss_name == "local_decoder":
                loss = loss + local_decoder_loss * loss_weight * schedule_multiplier
            elif loss_name == "boundary" and boundary_loss is not None:
                loss = loss + boundary_loss * loss_weight * schedule_multiplier
            elif loss_name == "output_boundary":
                loss = loss + output_boundary_loss * loss_weight * schedule_multiplier
            elif loss_name == "hnet_embed":
                loss = loss + hnet_embed_loss * loss_weight * schedule_multiplier
            elif loss_name == "ratio":
                loss = loss + ratio_loss * loss_weight * schedule_multiplier
            elif loss_name == "teacher_ce":
                loss = loss + teacher_ce_loss * loss_weight * schedule_multiplier
            else:
                raise ValueError(f"Unknown distillation loss '{loss_name}'")

        output = LMOutputWithLoss(
            logits=logits,
            loss=loss,  # type: ignore
            ce_loss=ce_loss,  # type: ignore
            z_loss=None,
        )

        return output, metrics

    def student_forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        blt_config: Optional[BLTConfig] = None,
        **kwargs,
    ):
        if blt_config is None:
            raise ValueError("`blt_config` must be provided for student_forward")

        input_ids, labels, all_block_kwargs, per_block_kwargs, lm_head_kwargs, local_encoder_kwargs, local_decoder_kwargs, extra_kwargs = self._prepare_inputs(
            input_ids,
            labels,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            loss_div_factor=loss_div_factor,
            blt_config=blt_config,
            return_logits=True,  # needed for distillation
            **kwargs,
        )
        if labels is None:
            labels = F.pad(
                input_ids[:, 1:],
                (0, 1),
                value=ignore_index,
            )

        byte_mask = labels != ignore_index
        shifted_patch_lens = F.pad(
            local_encoder_kwargs["patch_lens"][:, 1:],
            (0, 1),
        )
        patch_mask = shifted_patch_lens != 0

        patch_end_indices = torch.cumsum(local_encoder_kwargs["patch_lens"], dim=1) - 1
        patch_end_indices = torch.where(
            patch_end_indices < byte_mask.shape[1],
            patch_end_indices,
            torch.zeros_like(patch_end_indices), # effectively mask out, index 0 is always start
        )
        boundary_labels = torch.zeros_like(byte_mask, dtype=torch.float32)
        boundary_labels.scatter_(1, patch_end_indices, 1.0)
        if blt_config.skip_boundary_before_eos:
            boundary_labels[:, :-1] = torch.where( # no boundary before eos
                input_ids[:, 1:] == self.eos_token_blt,
                0.0,
                boundary_labels[:, :-1]
            )
            boundary_labels[:, 0] = 1 # bos must still be boundary

        h_byte, h_patch, boundary_logprobs, boundary_mask = self.local_encoder(
            input_ids,
            teacher_force_boundaries=blt_config.teacher_force_boundaries,
            boundary_threshold=blt_config.boundary_threshold,
            true_boundary_mask=boundary_labels > 0.5,
            **local_encoder_kwargs
        )

        L = byte_mask.shape[1]
        token_idx = (
            torch.arange(L, device=byte_mask.device)[None, :] + (~boundary_mask).long() * L  # type: ignore
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)[:, :patch_mask.shape[1]]
        last_increasing_index = ((seq_sorted_indices[:, 1:] - seq_sorted_indices[:, :-1]) < 0).max(-1)
        patch_mask = (
            (torch.arange(patch_mask.shape[1], device=patch_mask.device)[None, :] <= last_increasing_index.indices[:, None]) |
            (torch.zeros_like(patch_mask) == last_increasing_index.values[:, None]) # case where never not increasing (no padding)
        )
        if self.local_decoder.fuse_boundaries:
            shift_boundary_mask = (boundary_mask[:, 1:] & byte_mask[:, 1:])
            label_offsets = shift_boundary_mask * self.vocab_size_blt
            labels[:, :-1] += label_offsets

        dtype = h_patch.dtype
        global_dtype = torch.bfloat16 if isinstance(self.blocks["0"].attention.backend, FlashAttention2Backend) else dtype  # type: ignore

        h_patch_after_global, _ = self._block_forward(
            h_patch.to(global_dtype),
            all_block_kwargs=all_block_kwargs,
            per_block_kwargs=per_block_kwargs,
        )
        h_patch_after_global = h_patch_after_global.to(dtype)

        (h_out_for_boundaries, _), h_out_for_logits, _ = self.local_decoder(
            embeds=h_byte,
            patch_embeds=h_patch_after_global,
            patch_residuals=h_patch,
            boundary_logprobs=boundary_logprobs,
            boundary_mask=boundary_mask,
            **local_decoder_kwargs,
        )
        logits = self.lm_head(h_out_for_logits, **lm_head_kwargs)
        if self.local_decoder.fuse_boundaries:
            ce_loss, _ = cross_entropy_loss(logits.view(-1, logits.shape[-1]), labels.view(-1))  # type: ignore

            # replace plain (single byte) logits with byte + boundary logits where boundaries occur
            probs = F.softmax(logits.float(), dim=-1)
            probs[..., :self.vocab_size_blt] += probs[..., self.vocab_size_blt:self.vocab_size_blt*2]
            logits = torch.log(probs)
            logits[..., self.vocab_size_blt:self.vocab_size_blt*2] = -100_000
        else:
            logits = torch.concatenate([
                logits,
                torch.zeros((logits.shape[0], 1, logits.shape[2]), device=logits.device, dtype=logits.dtype)
            ], dim=1)

            ce_loss, _ = cross_entropy_loss(logits.view(-1, logits.shape[-1]), labels.view(-1))  # type: ignore

            if blt_config.eval_add_boundary_logp:
                boundary_logits = self.lm_head(h_out_for_boundaries, **lm_head_kwargs)

                main_path_logprobs = torch.gather(F.log_softmax(logits[:, :-1].float(), dim=-1), -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                main_path_boundary_logprobs = F.log_softmax(boundary_logits.float(), dim=-1)[..., self.end_of_subword_token_blt]  # type: ignore

                y_hat = main_path_logprobs.scatter_add(
                    src=torch.where(
                        patch_mask[:, 1:],
                        main_path_boundary_logprobs,
                        torch.zeros_like(main_path_boundary_logprobs),
                    ),
                    dim=-1,
                    index=torch.where(
                        patch_mask[:, 1:],
                        seq_sorted_indices[:, 1:] - 1,
                        torch.zeros_like(seq_sorted_indices[:, 1:])
                    ),
                )
                remaining_logpmass = log1mexp(y_hat)
                remaining_logp_uniform = remaining_logpmass - math.log(logits.shape[2] - 1)  # -1 to skip the main path token
                logits.zero_()
                logits[:, :-1, :] = remaining_logp_uniform.unsqueeze(-1)
                logits.scatter_(
                    -1,
                    input_ids[:, 1:].unsqueeze(-1),
                    y_hat.to(logits.dtype).unsqueeze(-1),
                )

        return LMOutputWithLoss(
            logits=logits,
            loss=ce_loss,
            ce_loss=ce_loss,
            z_loss=None,
        ), {}

    def subword_forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        blt_config: Optional[BLTConfig] = None,
        **kwargs,
    ):
        if blt_config is None:
            raise ValueError("`blt_config` must be provided for student_forward")

        if not blt_config.teacher_force_boundaries:
            raise ValueError("subword_forward only works with teacher_force_boundaries=True")

        input_ids, labels, all_block_kwargs, per_block_kwargs, lm_head_kwargs, local_encoder_kwargs, local_decoder_kwargs, extra_kwargs = self._prepare_inputs(
            input_ids,
            labels,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            loss_div_factor=loss_div_factor,
            blt_config=blt_config,
            return_logits=True,  # needed for distillation
            **kwargs,
        )

        byte_mask = labels != ignore_index
        shifted_patch_lens = F.pad(
            local_encoder_kwargs["patch_lens"][:, 1:],
            (0, 1),
        )
        patch_mask = shifted_patch_lens != 0

        patch_end_indices = torch.cumsum(local_encoder_kwargs["patch_lens"], dim=1) - 1
        patch_end_indices = torch.where(
            patch_end_indices < byte_mask.shape[1],
            patch_end_indices,
            torch.zeros_like(patch_end_indices), # effectively mask out, index 0 is always start
        )
        boundary_labels = torch.zeros_like(byte_mask, dtype=torch.float32)
        boundary_labels.scatter_(1, patch_end_indices, 1.0)

        h_byte, h_patch, boundary_logprobs, boundary_mask = self.local_encoder(
            input_ids,
            teacher_force_boundaries=blt_config.teacher_force_boundaries,
            boundary_threshold=blt_config.boundary_threshold,
            true_boundary_mask=boundary_labels > 0.5,
            **local_encoder_kwargs
        )

        L = byte_mask.shape[1]
        token_idx = (
            torch.arange(L, device=byte_mask.device)[None, :] + (~boundary_mask).long() * L  # type: ignore
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)[:, :patch_mask.shape[1]]
        last_increasing_index = ((seq_sorted_indices[:, 1:] - seq_sorted_indices[:, :-1]) < 0).max(-1)
        patch_mask = (
            (torch.arange(patch_mask.shape[1], device=patch_mask.device)[None, :] <= last_increasing_index.indices[:, None]) |
            (torch.zeros_like(patch_mask) == last_increasing_index.values[:, None]) # case where never not increasing (no padding)
        )

        dtype = h_patch.dtype
        global_dtype = torch.bfloat16 if isinstance(self.blocks["0"].attention.backend, FlashAttention2Backend) else dtype  # type: ignore

        h_patch_after_global, _ = self._block_forward(
            h_patch.to(global_dtype),
            all_block_kwargs=all_block_kwargs,
            per_block_kwargs=per_block_kwargs,
        )
        h_patch_after_global = h_patch_after_global.to(dtype)

        (h_out_for_boundaries, _), h_out_for_logits, _ = self.local_decoder(
            embeds=h_byte,
            patch_embeds=h_patch_after_global,
            patch_residuals=h_patch,
            boundary_logprobs=boundary_logprobs,
            boundary_mask=boundary_mask,
            **local_decoder_kwargs,
        )
        logits = self.lm_head(h_out_for_logits, **lm_head_kwargs)
        boundary_logits = self.lm_head(h_out_for_boundaries, **lm_head_kwargs)

        logits = torch.concatenate([
            logits,
            torch.zeros((logits.shape[0], 1, logits.shape[2]), device=logits.device, dtype=logits.dtype)
        ], dim=1)

        ce_loss, _ = cross_entropy_loss(logits.view(-1, logits.shape[-1]), labels.view(-1))  # type: ignore

        main_path_patch_logprobs = torch.zeros(patch_mask.shape, device=logits.device, dtype=torch.float32)
        main_path_logprobs = torch.gather(F.log_softmax(logits[:, :-1].float(), dim=-1), -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        y_hat = main_path_patch_logprobs.scatter_reduce(
            src=main_path_logprobs,
            dim=1,
            index=local_encoder_kwargs["patch_ids"][:, 1:] - 1,
            reduce="sum",
            include_self=False,
        )[:, :-1]

        if blt_config.eval_add_boundary_logp:
            main_path_boundary_logprobs = F.log_softmax(boundary_logits.float(), dim=-1)[..., self.end_of_subword_token_blt]  # type: ignore
            # ignore last
            main_path_boundary_logprobs[:, :-1] = (main_path_boundary_logprobs[:, :-1] * patch_mask[:, 2:])
            y_hat = y_hat + main_path_boundary_logprobs

        logits = torch.zeros(
            (patch_mask.shape[0], patch_mask.shape[1], self.teacher.embeddings.weight.shape[0]),  #  type: ignore
            dtype=torch.float32,
            device=logits.device
        )
        remaining_logpmass = log1mexp(y_hat)
        remaining_logp_uniform = remaining_logpmass - math.log(logits.shape[2] - 1)  # -1 to skip the main path token
        logits[:, :-2, :] = remaining_logp_uniform[:, 1:].unsqueeze(-1) # offset since bos skipped
        logits.scatter_(
            -1,
            extra_kwargs["original_input_ids"][:, 1:-1].unsqueeze(-1),
            y_hat[:, 1:].to(logits.dtype).unsqueeze(-1), # offset since bos skipped
        )

        return LMOutputWithLoss(
            logits=logits,
            loss=ce_loss,
            ce_loss=ce_loss,
            z_loss=None,
        ), {}

    def prefill_boundary_prediction_forward(
        self,
        input_ids: torch.Tensor,
        blt_config: BLTConfig,
        last_token_is_boundary: bool = False,
        sequence_start_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        input_ids, labels, all_block_kwargs, per_block_kwargs, lm_head_kwargs, local_encoder_kwargs, local_decoder_kwargs, extra_kwargs = self._prepare_inputs(
            input_ids,
            blt_config=blt_config,
            **kwargs,
        )

        byte_mask = torch.ones_like(input_ids, dtype=torch.bool) # temp
        if not last_token_is_boundary:
            local_encoder_kwargs["patch_lens"] = local_encoder_kwargs["patch_lens"][:, :-1]

        patch_end_indices = torch.cumsum(local_encoder_kwargs["patch_lens"], dim=1) - 1
        patch_end_indices = torch.where(
            patch_end_indices < byte_mask.shape[1],
            patch_end_indices,
            torch.zeros_like(patch_end_indices), # effectively mask out, index 0 is always start
        )
        boundary_labels = torch.zeros_like(byte_mask, dtype=torch.float32)
        boundary_labels.scatter_(1, patch_end_indices, 1.0)

        h_byte, h_patch, boundary_logprobs, boundary_mask = self.local_encoder.inference_forward(  # type: ignore
            input_ids,
            teacher_force_boundaries=blt_config.teacher_force_boundaries,
            boundary_threshold=blt_config.boundary_threshold,
            true_boundary_mask=boundary_labels > 0.5,
            boundary_state=blt_utils.MaskState(torch.full((input_ids.shape[0],), fill_value=last_token_is_boundary, device=input_ids.device, dtype=torch.bool)),
            pad_state=blt_utils.MaskState(torch.full((input_ids.shape[0],), fill_value=last_token_is_boundary and not self.local_decoder.fuse_boundaries, device=input_ids.device, dtype=torch.bool)),
            sequence_start_indices=sequence_start_indices,
            **local_encoder_kwargs
        )

        return boundary_mask, (h_byte, h_patch, boundary_logprobs, boundary_mask)

    def inference_forward(
        self,
        input_ids: torch.Tensor,
        blt_config: BLTConfig,
        boundary_state: torch.Tensor,
        pad_state: torch.Tensor,
        sequence_start_indices: Optional[torch.Tensor] = None,
        cached_encoder_outputs: Optional[Any] = None,
        **kwargs,
    ):
        input_ids, labels, all_block_kwargs, per_block_kwargs, lm_head_kwargs, local_encoder_kwargs, local_decoder_kwargs, extra_kwargs = self._prepare_inputs(
            input_ids,
            blt_config=blt_config,
            **kwargs,
        )

        if cached_encoder_outputs is not None:
            h_byte, h_patch, boundary_logprobs, boundary_mask = cached_encoder_outputs
        else:
            h_byte, h_patch, _, _ = self.local_encoder.inference_forward(  # type: ignore
                input_ids,
                boundary_state=boundary_state,
                pad_state=pad_state,
                sequence_start_indices=sequence_start_indices,
                **local_encoder_kwargs
            )
            boundary_logprobs = boundary_mask = None

        if h_patch.numel() > 0:
            # we need to convert from right-pad to left-pad and back for prefill
            # since flash attention expects left-pad and local/enc dec expect right-pad global tokens
            # should add better left-pad support but this only affects prefill so OK for now
            # although super inefficient!
            if boundary_mask is not None: # prefill
                n_boundaries = boundary_mask.sum(-1)

                for i, current_n_boundaries in enumerate(n_boundaries):
                    h_patch[i, -current_n_boundaries:] = h_patch[i, :current_n_boundaries].clone()

            h_patch_after_global, _ = self._block_forward(
                h_patch.to(torch.bfloat16),
                all_block_kwargs=all_block_kwargs,
                per_block_kwargs=per_block_kwargs,
            )
            h_patch_after_global = h_patch_after_global.to(h_patch.dtype)

            if boundary_mask is not None: # prefill
                n_boundaries = boundary_mask.sum(-1)

                for i, current_n_boundaries in enumerate(n_boundaries):
                    h_patch_after_global[i, :current_n_boundaries] = h_patch_after_global[i, -current_n_boundaries:].clone()
        else:
            h_patch_after_global = h_patch

        _, _, h_out = self.local_decoder.inference_forward(  # type: ignore
            embeds=h_byte,
            patch_embeds=h_patch_after_global,
            patch_residuals=h_patch,
            boundary_logprobs=boundary_logprobs,
            boundary_mask=boundary_mask,
            boundary_state=boundary_state,
            sequence_start_indices=sequence_start_indices,
            **local_decoder_kwargs,
        )
        logits = self.lm_head(h_out, **lm_head_kwargs)

        return logits