import logging
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
                self.embeddings, d_model=self.d_model, std=self.init_std, generator=generator
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

            # Warm up RoPE cache.
            if max_seq_len is not None and att.rope is not None:
                att.rope.warmup_cache(max_seq_len, device)

        if self.lm_head is not None:
            self.init_method.init_final_w_out(
                self.lm_head.w_out, d_model=self.d_model, std=self.init_std, generator=generator
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
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any], Dict[str, Any]]:
        # NOTE: with pipeline parallelism input_ids might actually be an intermediate output,
        # so we have to be careful here.
        B, S = input_ids.shape[:2]

        block_kwargs: Dict[str, Any] = {}
        encoder_decoder_kwargs: Dict[str, Any] = {}

        lm_head_kwargs: Dict[str, Any] = dict(
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            return_logits=return_logits,
        )

        if loss_div_factor is not None:
            loss_div_factor = move_to_device(loss_div_factor, self.device)
            lm_head_kwargs["loss_div_factor"] = loss_div_factor
            block_kwargs["loss_div_factor"] = loss_div_factor

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
                    block_kwargs[key] = move_to_device(value, self.device)
            else:
                inputs = cp_load_balancer.batch_shard(
                    inputs=inputs,
                    seq_dims=seq_dims,
                    pad_values=pad_values,
                )

            for key, value in zip(keys, inputs):
                block_kwargs[key] = move_to_device(value, self.device)

            input_ids = block_kwargs.pop("input_ids")
            labels = block_kwargs.pop("labels", None)
        else:
            input_ids = move_to_device(input_ids, self.device)
            labels = move_to_device(labels, self.device)
            block_kwargs["max_doc_len"] = max_doc_len
            block_kwargs["cu_doc_lens"] = move_to_device(cu_doc_lens, self.device)

        return (
            input_ids,
            labels,
            block_kwargs,
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
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run the transformer on the token input IDs.

        :param input_ids: The token input IDs, shape ``(batch_size, seq_len)``.

        :returns: The logits if ``labels`` is ``None`` or the losses if ``labels`` is not ``None``.
        """
        input_ids, labels, block_kwargs, lm_head_kwargs = self._prepare_inputs(
            input_ids,
            labels,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            loss_div_factor=loss_div_factor,
            return_logits=return_logits,
            **kwargs,
        )

        # Get embeddings but pass-through for non-existent layers to allow easy
        # pipeline parallel configuration.
        h = self.embeddings(input_ids) if self.embeddings is not None else input_ids

        # Run each block.
        for block in self.blocks.values():
            # Mark sizes as dynamic for torch.compile().
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
            h = block(h, **block_kwargs)

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

    def apply_cp(self, cp_mesh: DeviceMesh, load_balancer: RingAttentionLoadBalancerType):
        """
        Prepare the model for context-parallelism (CP).

        :param cp_mesh: The CP device mesh.
        :param load_balancer: The load balancing method.
        """
        self._cp_load_balancer = load_balancer.build(cp_mesh)
        for block in self.blocks.values():
            cast(TransformerBlockBase, block).apply_cp(cp_mesh, load_balancer)
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
            fully_shard(self.embeddings, reshard_after_forward=reshard_after_forward, **fsdp_config)
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

    def _validate_block(self, block: TransformerBlockBase) -> TransformerBlockBase:
        if not isinstance(block, MoETransformerBlock):
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}' requires a '{MoETransformerBlock.__name__}' block"
            )
        return block

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
            block = cast(MoETransformerBlock, block)
            block_metrics = block.compute_metrics(reset=reset)
            for metric_name, (metric_val, reduce_type) in block_metrics.items():
                out[f"block {int(block_idx):02d}/{metric_name}"] = (metric_val, reduce_type)

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
                    out[metric_name] = (torch.max(out[metric_name][0], metric_val), reduce_type)
                else:
                    raise NotImplementedError(reduce_type)
        return out

    def reset_auxiliary_metrics(self):
        for block in self.blocks.values():
            cast(MoETransformerBlock, block).reset_metrics()

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        for block in self.blocks.values():
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

    def post_batch(self, dry_run: bool = False):
        for block in self.blocks.values():
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
        prepend_embedding_to_global: bool = False,
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

        if prepend_embedding_to_global:
            self.prepend_embedding = nn.Embedding(1, self.d_model, dtype=self.dtype)
        else:
            self.prepend_embedding = None

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        input_ids, labels, block_kwargs, lm_head_kwargs = super()._prepare_inputs(
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
        encoder_cross_attn_mask: Optional[BlockMask] = None
        decoder_cross_attn_mask: Optional[BlockMask] = None

        if (patch_lens := kwargs.pop("patch_lens", None)) is not None:
            if blt_config is not None and blt_config.patching == "space":
                patch_lens = kwargs["space_patch_lens"]
                assert patch_lens is not None, "space_patch_lens must be present if patch_lens is present"

            patch_lens = move_to_device(patch_lens, self.device)
            patch_ids = blt_utils.lengths_to_ids(patch_lens, input_ids.shape[-1])
            original_input_ids = kwargs.pop("original_input_ids", None) # must be present if patch_lens is present

            encoder_cross_attn_mask = blt_utils.cross_attn_mask(
                patch_ids,
                patch_lens,
                patches_as_queries=True,
                cross_attn_k=self.local_encoder.blt_k or 1,  # type: ignore
                block_mask=True,
            )

            _decoder_patch_ids = blt_utils.lengths_to_ids(patch_lens[:, 1:], input_ids.shape[-1])
            decoder_cross_attn_mask = blt_utils.cross_attn_mask(
                _decoder_patch_ids,
                patch_lens,
                patches_as_queries=False,
                cross_attn_k=self.local_decoder.blt_k or 1,  # type: ignore
                block_mask=True,
            )

        local_encoder_kwargs["patch_lens"] = patch_lens
        local_encoder_kwargs["patch_ids"] = patch_ids
        local_encoder_kwargs["cross_attn_mask"] = encoder_cross_attn_mask
        local_decoder_kwargs["patch_lens"] = patch_lens
        local_decoder_kwargs["patch_ids"] = patch_ids
        local_decoder_kwargs["cross_attn_mask"] = decoder_cross_attn_mask
        extra_kwargs["original_input_ids"] = move_to_device(original_input_ids, self.device)

        if (constituent_input_ids := kwargs.pop("constituent_input_ids", None)) is not None:
            extra_kwargs["constituent_input_ids"] = move_to_device(constituent_input_ids, self.device)

        if (teacher_inputs_embeds := kwargs.pop("teacher_inputs_embeds", None)) is not None:
            extra_kwargs["teacher_inputs_embeds"] = move_to_device(teacher_inputs_embeds, self.device)

        if blt_config is not None and blt_config.patching != "dolma2":
            # can't use attributes relying on dolma2 patching
            extra_kwargs["original_input_ids"] = None
            extra_kwargs["constituent_input_ids"] = None

        return (
            input_ids,
            labels,
            block_kwargs,
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

        for block in self.local_encoder.blocks.values():  # type: ignore
            block = cast(TransformerBlockBase, block)
            block.apply_compile()

        for block in self.local_decoder.blocks.values():  # type: ignore
            block = cast(TransformerBlockBase, block)
            block.apply_compile()

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
        input_ids, labels, block_kwargs, lm_head_kwargs, local_encoder_kwargs, local_decoder_kwargs, _ = self._prepare_inputs(
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

        if self.prepend_embedding is not None:
            h_patch_global = torch.cat([
                self.prepend_embedding.weight.unsqueeze(0).expand(h_patch_global.shape[0], -1, -1),
                h_patch_global,
            ], dim=1)

        # Run each block.
        for block in self.blocks.values():
            # Mark sizes as dynamic for torch.compile().
            if self.compile_enabled:
                mark_dynamic(h_patch, (0, 1), strict=False)
            h_patch_global = block(h_patch_global, **block_kwargs)

        if self.prepend_embedding is not None:
            h_patch_global = h_patch_global[:, 1:]

        h_patch = h_patch_global.to(h_patch.dtype)

        h_out = self.local_decoder(
            embeds=h_byte,
            patch_embeds=h_patch,
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

        if self.teacher is not None and self.use_teacher_embs_with_vocab_size is not None:
            self.teacher_embeddings = nn.Embedding(
                self.use_teacher_embs_with_vocab_size,
                self.teacher.d_model,
                dtype=dtype,
                device=init_device,
            )
        else:
            self.teacher_embeddings = None

        if self.teacher is not None:
            if self.share_blocks:
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
    def _teacher_forward(
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
    ) -> Tuple[Union[torch.Tensor, LMOutputWithLoss, None], Tuple[list[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """
        Run the transformer on the token input IDs.

        :param input_ids: The token input IDs, shape ``(batch_size, seq_len)``.

        :returns: The logits if ``labels`` is ``None`` or the losses if ``labels`` is not ``None``.
        """
        out_hidden_states = []

        if isinstance(self.teacher, BLTTransformer):
            input_ids, labels, block_kwargs, lm_head_kwargs, local_encoder_kwargs, local_decoder_kwargs, extra_kwargs = self.teacher._prepare_inputs(
                input_ids,
                labels,
                ignore_index=ignore_index,
                loss_reduction=loss_reduction,
                z_loss_multiplier=z_loss_multiplier,
                loss_div_factor=loss_div_factor,
                return_logits=return_logits,
                **kwargs,
            )

            h_byte, h_patch, _, _ = self.teacher.local_encoder(input_ids, **local_encoder_kwargs)
            h_emb = h_patch

            if skip_blocks:
                return None, ([], None, h_emb)

            if self.share_blocks:
                blocks = self.blocks
            else:
                blocks = self.teacher.blocks

            if self.teacher_embeddings is not None:
                h_patch_global = self.teacher_embeddings(extra_kwargs["original_input_ids"][:, :-1])
            elif zero_bos:
                h_patch_global = h_patch[:, 1:]
            else:
                h_patch_global = h_patch

            # Run each block.
            for block_idx, block in blocks.items():
                # Mark sizes as dynamic for torch.compile().
                if self.compile_enabled:
                    mark_dynamic(h_patch_global, (0, 1), strict=False)
                h_patch_global = block(h_patch_global, **block_kwargs)

                if int(block_idx) in (hidden_states_to_return or []):
                    out_hidden_states.append(h_patch_global.detach().clone())

            if zero_bos:
                h_patch_after_global = torch.zeros_like(h_patch)
                h_patch_after_global[:, 1:] = h_patch_global
            else:
                h_patch_after_global = h_patch_global

            h_out = self.teacher.local_decoder(
                embeds=h_byte,
                patch_embeds=h_patch_after_global,
                **local_decoder_kwargs,
            )

            if self.teacher.lm_head is not None:
                if labels is not None:
                    lm_head_kwargs["labels"] = labels
                return self.teacher.lm_head(h_out, **lm_head_kwargs), (out_hidden_states, h_out, h_emb)
            else:
                return None, (out_hidden_states, h_out, h_emb)
        elif isinstance(self.teacher, Transformer):
            input_ids, labels, block_kwargs, lm_head_kwargs = self.teacher._prepare_inputs(
                input_ids,
                labels,
                ignore_index=ignore_index,
                loss_reduction=loss_reduction,
                z_loss_multiplier=z_loss_multiplier,
                loss_div_factor=loss_div_factor,
                return_logits=return_logits,
                **kwargs,
            )

            h_emb = self.teacher.embeddings(input_ids)

            if inputs_embeds is not None:
                # not ideal, to support <bos> difference
                h_emb[:, :inputs_embeds.shape[1]] = inputs_embeds

            if skip_blocks:
                return None, ([], None, h_emb)

            if self.share_blocks:
                blocks = self.blocks
            else:
                blocks = self.teacher.blocks

            # Run each block.
            h = h_emb

            for block in blocks.values():
                # Mark sizes as dynamic for torch.compile().
                if self.teacher.compile_enabled:
                    mark_dynamic(h, (0, 1), strict=False)
                h = block(h, **block_kwargs)

                if int(block.block_idx) in (hidden_states_to_return or []):
                    out_hidden_states.append(h.detach().clone())

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
                return self.teacher.lm_head(h, **lm_head_kwargs), (out_hidden_states, h, h_emb)
            else:
                return None, (out_hidden_states, h, h_emb)
        else:
            return None, ([], None, None)

    def fix_init(self, embedding_init_path: Optional[str] = None):
        self.local_encoder.fix_init(embedding_init_path, self.teacher.embeddings.weight)  # type: ignore

    def _compute_alm_style_loss(
        self,
        main_path_logprobs: torch.Tensor,
        boundary_logprobs: Optional[torch.Tensor],
        boundary_labels: Optional[torch.Tensor],
        teacher_logprobs: torch.Tensor,
        teacher_main_path_logprobs: torch.Tensor,
        byte_mask: torch.Tensor,
        patch_mask: torch.Tensor,
        patch_lens: torch.Tensor,
        patch_ids: torch.Tensor,
        constituent_input_ids: torch.Tensor,
        ignore_index: int,
        blt_config: BLTConfig,
        metrics,
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

        ## EXHAUSTIVE LOSS
        # -2: -1 to start from 0 (because of <bos> token), -1 to shift labels right
        teacher_logprobs_indices = (patch_ids[:, 1:] - 2) * teacher_logprobs.shape[2] + constituent_input_ids[:, :-1]
        teacher_logprobs_mask = (constituent_input_ids[:, :-1] != ignore_index) & (teacher_logprobs_indices >= 0)
        teacher_logprobs_to_compare = torch.gather(
            input=teacher_logprobs.view(teacher_logprobs.shape[0], -1),
            dim=-1,
            index=torch.where(
                teacher_logprobs_mask,
                teacher_logprobs_indices,
                torch.zeros_like(teacher_logprobs_indices)
            ),
        )
        # this equals teacher_main_path_logprobs[0]
        # teacher_logprobs_to_compare[0][torch.cumsum(local_encoder_kwargs["patch_lens"][0, 1:], dim=0) - 1][1:]

        local_decoder_loss_exhaustive = 0.0
        local_decoder_denom = 0

        for offset in range(blt_config.n_distill_offsets):
            student_logprobs_to_compare = torch.zeros_like(teacher_logprobs_to_compare)

            # write to last position of each patch
            student_logprobs_mask = (patch_ids[:, 1:] < patch_lens.shape[1])
            masked_offset_patch_ids = torch.where(
                student_logprobs_mask,
                patch_ids[:, 1:] - 1,
                torch.zeros_like(patch_ids[:, 1:])
            )
            # we can't include patches with length < offset
            student_logprobs_mask &= torch.take_along_dim(
                patch_lens[:, 1:] - 1 - offset >= 0,
                masked_offset_patch_ids,
                dim=1,
            )
            # exclude the last n (n=offset) bytes from each patch
            if offset > 0:
                student_logprobs_mask[:, :-offset] = (
                    masked_offset_patch_ids[:, :-offset] == masked_offset_patch_ids[:, offset:]
                )

            student_logprobs_indices = torch.take_along_dim(
                torch.cumsum(patch_lens[:, 1:], dim=1) - 1 - offset,
                masked_offset_patch_ids,
                dim=1,
            )

            student_logprobs_indices = torch.where(
                student_logprobs_mask,
                torch.clamp(student_logprobs_indices, min=0), # should not be necessary, for dummy batch
                torch.zeros_like(student_logprobs_indices),
            )

            student_logprobs_to_compare = student_logprobs_to_compare.scatter_add(
                dim=1,
                index=student_logprobs_indices,
                src=main_path_logprobs,
            )
            nonzero_mask = (student_logprobs_to_compare != 0) & (teacher_logprobs_mask)

            if blt_config.add_boundary_logp:
                assert boundary_logprobs is not None

                student_logprobs_to_compare += boundary_logprobs[:, 1:]

            current_local_decoder_loss = (
                div_fn(teacher_logprobs_to_compare, student_logprobs_to_compare) * nonzero_mask.float()
            ).sum()
            current_local_decoder_denom = nonzero_mask.float().sum()

            local_decoder_loss_exhaustive += current_local_decoder_loss * blt_config.distill_offset_weights[offset]
            local_decoder_denom += current_local_decoder_denom

            metrics[f"blt/local_decoder_loss_{offset}"] = current_local_decoder_loss / (current_local_decoder_denom + blt_config.epsilon)
            metrics[f"blt/local_decoder_teacher_mean_p_{offset}"] = (teacher_logprobs_to_compare * nonzero_mask.float()).sum() / (current_local_decoder_denom + blt_config.epsilon)
            metrics[f"blt/local_decoder_mean_numel_{offset}"] = current_local_decoder_denom / nonzero_mask.numel()

            if blt_config.add_boundary_logp:
                assert boundary_labels is not None and boundary_logprobs is not None

                metrics[f"blt/boundary_acc_{offset}"] = (
                    (((torch.exp(boundary_logprobs) > 0.5) == (boundary_labels > 0))[:, 1:] * nonzero_mask.float()).sum()
                    / (current_local_decoder_denom + blt_config.epsilon)
                )

        # compute the boundary loss over the remaining boundaries (those can never occur)
        if blt_config.add_boundary_logp:
            assert boundary_logprobs is not None and boundary_labels is not None and boundary_logprobs is not None

            remainder_target_logprobs = torch.full_like(boundary_logprobs, math.log(blt_config.epsilon))
            remainder_mask = byte_mask[:, 1:] & (~teacher_logprobs_mask)
            remainder_boundary_loss = (
                div_fn(
                    remainder_target_logprobs[:, 1:],
                    boundary_logprobs[:, 1:],
                ) * remainder_mask.float()
            ).sum()

            local_decoder_loss_exhaustive += remainder_boundary_loss
            local_decoder_denom += remainder_mask.float().sum()

            metrics[f"blt/local_decoder_remainder_boundary_loss"] = remainder_boundary_loss / (remainder_mask.float().sum() + blt_config.epsilon)
            metrics[f"blt/local_decoder_remainder_boundary_mean_numel"] = remainder_mask.float().sum() / remainder_mask.numel()
            metrics[f"blt/local_decoder_remainder_boundary_acc"] = (
                (((torch.exp(boundary_logprobs) > 0.5) == (boundary_labels > 0))[:, 1:] * remainder_mask.float()).sum()
                / (remainder_mask.float().sum() + blt_config.epsilon)
            )

        metrics["blt/local_decoder_loss_exhaustive"] = local_decoder_loss_exhaustive / (local_decoder_denom + blt_config.epsilon)
        local_decoder_loss_exhaustive = local_decoder_loss_exhaustive / byte_mask.numel()

        ## SIMPLE LOSS
        main_path_patch_logprobs = torch.zeros((patch_mask.shape[0], patch_mask.shape[1]), device=main_path_logprobs.device, dtype=main_path_logprobs.dtype)
        main_path_patch_logprobs = main_path_patch_logprobs.scatter_reduce(
            src=main_path_logprobs,
            dim=1,
            index=patch_ids[:, 1:] - 1,
            reduce="sum",
            include_self=False,
        )
        # `main_path_patch_logprobs` are at this point probabilities for the first regular token onwards (since byte-level input ids include <bos>)
        # the last position is written to by padding tokens (those with patch_ids == seq_length), so we need to disregard those.
        # also `teacher_main_path_logprobs` contains the prediction starting from the *second* token (since the teacher does not use <bos>).
        # so we need to shift appropriately.
        y_hat = main_path_patch_logprobs[:, 1:-1]
        # the teacher has had one more id passed to it (since we reduced seq length of the student by one to make room for <bos>).
        # so we need to truncate by an extra position here.
        y_true = teacher_main_path_logprobs[:, :-1]

        if boundary_logprobs is not None and blt_config.add_boundary_logp:
            patch_end_indices = torch.cumsum(patch_lens, dim=1) - 1
            y_hat = y_hat + torch.gather(boundary_logprobs, -1, patch_end_indices)[:, 2:]

        local_decoder_loss_simple = (div_fn(y_true, y_hat) * patch_mask[:, 1:-1]).mean()
        metrics["blt/local_decoder_teacher_mean_p_simple"] = (torch.exp(y_true) * patch_mask[:, 1:-1]).mean() / patch_mask[:, 1:-1].float().mean()
        metrics["blt/local_decoder_loss_simple"] = local_decoder_loss_simple / patch_mask[:, 1:-1].float().mean()
        metrics["blt/local_decoder_mae_simple"] = (torch.abs(y_true - y_hat) * patch_mask[:, 1:-1]).mean() / patch_mask[:, 1:-1].float().mean()

        return local_decoder_loss_exhaustive, local_decoder_loss_simple, metrics

    @lru_cache(maxsize=100_000)
    def _backend_tokenize(self, byte_str):
        if not hasattr(self, "_tokenizer"):
            # hardcode for now
            self._tokenizer = ByteTokenizerConfig.blt().build()

        return tuple(x.id for x in self._tokenizer.hf_tokenizer.backend_tokenizer.model.tokenize(byte_str))

    def _noise_strict(
        self,
        input_ids,
        labels,
        p,
        epsilon=1e-6,
        **kwargs: Dict[str, Any],
    ) -> tuple[torch.Tensor, Dict[str, Any], torch.Tensor]:
        if not hasattr(self, "_tokenizer"):
            # hardcode for now
            self._tokenizer = ByteTokenizerConfig.blt().build()

        indices = torch.randperm(input_ids.shape[0])[:int(input_ids.shape[0] * p)]

        with torch.no_grad():
            _, _, (_, boundary_logprobs), boundary_mask = self.local_encoder(
                input_ids[indices],
                patch_lens=torch.tensor([[input_ids.shape[1]]], device=input_ids.device, dtype=torch.long).expand(indices.shape[0], -1),
                patch_ids=torch.zeros_like(input_ids[indices]),
                cross_attn_mask=None,
            )

        B, L = boundary_mask.shape
        token_idx = (
            torch.arange(L, device=boundary_mask.device)[None, :]
            + (~boundary_mask).long() * L
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)[:, :kwargs["patch_lens"].shape[1]]  # type: ignore
        patch_lens = kwargs["patch_lens"].clone()  # type: ignore

        # get / unshard teacher embeddings
        teacher_embeddings = self.teacher.embeddings(torch.arange(len(self.teacher.embeddings.weight), device=input_ids.device))  # type: ignore
        teacher_inputs_embeds = teacher_embeddings[kwargs["original_input_ids"]]

        for i, idx in enumerate(indices):
            teacher_inputs_embeds[idx] = 0
            patch_lens[idx, 1:] = 0

            to_write = {}
            # len(subword_ids) should never be zero during regular training, but make sure (and could be zero in dry run batch)
            counts = torch.full(
                (len(teacher_inputs_embeds[idx]),),
                fill_value=epsilon,
                dtype=torch.float32,
                device=teacher_inputs_embeds.device
            )

            prev_i = 1

            for j in range(seq_sorted_indices.shape[1] - 1):
                curr_i = seq_sorted_indices[i, j + 1] # offset bos

                if curr_i + 1 <= prev_i:
                    break

                current_bytes = self._tokenizer.decode_to_bytes(input_ids[idx, prev_i:curr_i + 1])
                current_byte_str = blt_utils.bytes_to_chars(current_bytes)
                subword_ids = self._backend_tokenize(current_byte_str)

                for idx_in_patch, subword_id in enumerate(subword_ids):
                    if idx_in_patch not in to_write:
                        to_write[idx_in_patch] = ([], [])

                    to_write[idx_in_patch][0].append(j)
                    to_write[idx_in_patch][1].append(subword_id)

                patch_lens[idx, j + 1] = len(current_bytes)
                prev_i = curr_i + 1

            for idx_in_patch, (js, subword_ids) in to_write.items():
                teacher_inputs_embeds[idx, js] += teacher_embeddings[subword_ids]
                counts[js] += 1

            teacher_inputs_embeds[idx] /= counts.unsqueeze(-1)
            labels[idx, prev_i:] = -100

            # cant use
            kwargs["original_input_ids"][idx] = 0 # type: ignore
            kwargs["constituent_input_ids"][idx] = 0 # type: ignore

        kwargs["teacher_inputs_embeds"] = teacher_inputs_embeds  # type: ignore
        kwargs["patch_lens"] = patch_lens  # type: ignore

        return labels, kwargs, indices

    def _noise(
        self,
        input_ids,
        labels,
        p,
        **kwargs: Dict[str, Any],
    ) -> tuple[torch.Tensor, Dict[str, Any], torch.Tensor]:
        if not hasattr(self, "_tokenizer"):
            # hardcode for now
            self._tokenizer = ByteTokenizerConfig.blt().build()
        if not hasattr(self, "_noiser"):
            self._noiser = blt_utils.Noiser(self._tokenizer.hf_tokenizer)

        indices = torch.randperm(input_ids.shape[0])[:int(input_ids.shape[0] * p)]

        with torch.no_grad():
            _, _, (_, boundary_logprobs), boundary_mask = self.local_encoder(
                input_ids[indices],
                patch_lens=torch.tensor([[input_ids.shape[1]]], device=input_ids.device, dtype=torch.long).expand(indices.shape[0], -1),
                patch_ids=torch.zeros_like(input_ids[indices]),
                cross_attn_mask=None,
            )

        # -1 / 1: to skip bos
        L = input_ids.shape[1] - 1
        boundary_indices = (
            torch.arange(L, device=input_ids.device)[None, :] + (~boundary_mask)[:, 1:].long() * L  # type: ignore
        )
        boundary_indices = torch.argsort(boundary_indices, dim=1)[:, :kwargs["patch_lens"].shape[1]]  # type: ignore

        for i, idx in enumerate(indices):
            original_input_ids = kwargs["original_input_ids"][idx] # type: ignore
            noised_original_input_ids = self._noiser.noise_ctrl_char_preset_boundaries(
                original_input_ids,
                set(boundary_indices[i][:boundary_mask[i].sum()].tolist()),
                byte_tokenizer=self._tokenizer,
            )[:len(original_input_ids)]

            while len(noised_original_input_ids) < len(original_input_ids):
                noised_original_input_ids.append(self._tokenizer.hf_tokenizer.pad_token_id)

            noised_input_ids, noised_patch_lengths = self._tokenizer.get_tokens_and_patch_lengths(
                noised_original_input_ids,
                add_bos=True,
                skip_last=True
            )
            noised_patch_lengths = torch.tensor(
                noised_patch_lengths,
                device=original_input_ids.device,
                dtype=original_input_ids.dtype,
            )
            # make sure lengths do not surpass byte_sequence_length
            noised_patch_lengths = torch.where(
                torch.cumsum(noised_patch_lengths, dim=0) > input_ids.shape[1],
                torch.zeros_like(noised_patch_lengths),
                noised_patch_lengths,
            )

            kwargs["original_input_ids"][idx] = torch.tensor(  # type: ignore
                noised_original_input_ids,
                device=original_input_ids.device,
                dtype=original_input_ids.dtype,
            )
            kwargs["patch_lens"][idx] = noised_patch_lengths  # type: ignore
            labels[idx, len(noised_input_ids):] = -100  # type: ignore 

        return labels, kwargs, indices

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
        use_oracle_patch_reps = blt_config.use_oracle_patch_reps

        if blt_config.p_boundary_noise > 0:
            labels, kwargs, noised_indices = self._noise_strict(input_ids, labels, p=blt_config.p_boundary_noise, **kwargs)
        else:
            noised_indices = None

        input_ids, labels, block_kwargs, lm_head_kwargs, local_encoder_kwargs, local_decoder_kwargs, extra_kwargs = self._prepare_inputs(
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

        # losses (incl. distillation losses)
        metrics = {}

        # First, run the teacher.
        if not skip_teacher:
            with torch.no_grad():
                input_ids_for_teacher = input_ids if isinstance(self.teacher, BLTTransformer) else extra_kwargs["original_input_ids"]

                teacher_logits, (teacher_hidden_states, teacher_last_hidden_state, teacher_embeds) = self._teacher_forward(
                    input_ids_for_teacher,
                    inputs_embeds=extra_kwargs.get("teacher_inputs_embeds"),
                    labels=None, # we will compute loss ourselves
                    return_logits=True,
                    skip_blocks=skip_blocks or skip_teacher_blocks,
                    zero_bos=True,
                    hidden_states_to_return=list(range(blt_config.encoder_loss_lookahead)),
                    **kwargs,
                )
                if teacher_logits is not None:
                    teacher_logprobs = F.log_softmax(teacher_logits.float() / blt_config.temperature, dim=-1) # type: ignore
                    teacher_main_path_logprobs = torch.gather(teacher_logprobs[:, :-1], -1, input_ids_for_teacher[:, 1:].unsqueeze(-1)).squeeze(-1)
                else:
                    teacher_logprobs = None
                    teacher_main_path_logprobs = None
        else:
            teacher_embeds = None
            teacher_last_hidden_state = None
            teacher_logprobs = None
            teacher_main_path_logprobs = None
            teacher_hidden_states = None

        # loss masks
        byte_mask = labels != ignore_index
        shifted_patch_lens = F.pad(
            local_encoder_kwargs["patch_lens"][:, 1:],
            (0, 1),

        )
        patch_mask = shifted_patch_lens != 0

        patch_end_indices = torch.cumsum(local_encoder_kwargs["patch_lens"], dim=1) - 1
        boundary_labels = torch.zeros_like(byte_mask, dtype=torch.float32)
        boundary_labels.scatter_(1, patch_end_indices, 1.0)

        if blt_config.teacher_force_interpolation_steps != 0:
            teacher_force_interpolation_ratio = min(kwargs["step"] / blt_config.teacher_force_interpolation_steps, 1.0)
            metrics[f"blt/teacher_force_interpolation_ratio"] = teacher_force_interpolation_ratio
        else:
            teacher_force_interpolation_ratio = None

        h_byte, h_patch, (boundary_logprobs_for_loss, boundary_logprobs), boundary_mask = self.local_encoder(
            input_ids,
            boundary_predictor_backprop_through_encoder=blt_config.boundary_predictor_backprop_through_encoder,
            teacher_force_boundaries=blt_config.teacher_force_boundaries,
            boundary_threshold=blt_config.boundary_threshold,
            teacher_force_interpolation_ratio=teacher_force_interpolation_ratio,
            **local_encoder_kwargs,
        )
        if boundary_logprobs is not None:
            if blt_config.decoder_backprop_through_add_boundary_logp:
                boundary_logprobs_for_decoder_loss = boundary_logprobs_for_loss
            else:
                boundary_logprobs_for_decoder_loss = boundary_logprobs_for_loss.detach()
        else:
            boundary_logprobs = None
            boundary_logprobs_for_decoder_loss = None
            boundary_labels = None

        # Run each block.
        if not skip_blocks:
            if use_oracle_patch_reps:
                if teacher_last_hidden_state is None:
                    raise ValueError("`last_hidden_state` must be provided when `use_oracle_patch_reps` is True")
                h_patch_after_global = torch.zeros_like(h_patch)
                h_patch_after_global[:, 1:] = teacher_last_hidden_state[:, :-1]
            else:
                # need to start with the first token since <bos> token is not known to the global transformer
                # and for consistency with the use_oracle_patch_reps=True case.
                if self.prepend_embedding is not None:
                    h_patch_global = h_patch.clone()
                    h_patch_global[:, 0] = self.prepend_embedding.weight.expand(h_patch.shape[0], -1)
                else:
                    h_patch_global = h_patch[:, 1:]

                for block in self.blocks.values():
                    # Mark sizes as dynamic for torch.compile().
                    if self.compile_enabled:
                        mark_dynamic(h_patch_global, (0, 1), strict=False)
                    h_patch_global = block(h_patch_global, **block_kwargs)

                if self.prepend_embedding is not None:
                    h_patch_global = h_patch_global[:, 1:]

                h_patch_after_global = torch.zeros_like(h_patch)
                h_patch_after_global[:, 1:] = h_patch_global

            if blt_config.decoder_backprop_through_encoder:
                h_byte_for_decoder = h_byte
                h_patch_for_decoder = h_patch_after_global
            else:
                h_byte_for_decoder = h_byte.detach()
                h_patch_for_decoder = h_patch_after_global.detach()

            if blt_config.decoder_backprop_through_boundary_predictor:
                boundary_logprobs_for_decoder = boundary_logprobs if boundary_logprobs is not None else None
            else:
                boundary_logprobs_for_decoder = boundary_logprobs.detach() if boundary_logprobs is not None else None

            h_out = self.local_decoder(
                embeds=h_byte_for_decoder,
                patch_embeds=h_patch_for_decoder,
                boundary_logprobs=None if blt_config.teacher_force_boundaries else boundary_logprobs_for_decoder,
                boundary_mask=None if blt_config.teacher_force_boundaries else boundary_mask,
                **local_decoder_kwargs,
            )

            logits = self.lm_head(h_out, **lm_head_kwargs)
            logprobs = F.log_softmax(logits.float() / blt_config.temperature, dim=-1)
            main_path_logprobs = torch.gather(logprobs[:, :-1], -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        else:
            h_out = None
            logits = None
            logprobs = None
            main_path_logprobs = None

        # compute CE
        if not skip_blocks:
            ce_loss, _ = cross_entropy_loss(logits.view(-1, logits.shape[-1]), labels.view(-1))  # type: ignore
            metrics["blt/ce_loss"] = ce_loss
        else:
            ce_loss = torch.nan

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

        # could also have some version of the encoder loss for BLT teacher but not implemented for now
        if not skip_teacher and not isinstance(self.teacher, BLTTransformer):
            assert teacher_embeds is not None

            # the offset is OLMo specific: we use a <bos> token, but OLMo doesn't so shift by one.
            elementwise_local_encoder_loss = rep_compare_fn(
                h_patch[:, 1:],
                teacher_embeds[:, :-1],
            )
            local_encoder_loss = (elementwise_local_encoder_loss * patch_mask[:, :-1].float()).mean()
            metrics["blt/local_encoder_loss"] = local_encoder_loss / patch_mask[:, :-1].float().mean()
            metrics["blt/local_encoder_cos_sim"] = (F.cosine_similarity(
                h_patch[:, 1:].float(),
                teacher_embeds[:, :-1].float(),
                dim=-1,
            ) * patch_mask[:, :-1].float()).mean() / patch_mask[:, :-1].float().mean()

            # add lookahead (FuLA-style) losses
            h_lookahead = h_patch[:, 1:]  # skip <bos> token

            for lookahead_idx in range(blt_config.encoder_loss_lookahead):
                assert teacher_hidden_states is not None

                h_lookahead = self.blocks[str(lookahead_idx)](h_lookahead, **block_kwargs)

                elementwise_local_encoder_loss = rep_compare_fn(
                    h_lookahead,
                    teacher_hidden_states[lookahead_idx][:, :-1],
                )
                local_encoder_loss_lookahead = (elementwise_local_encoder_loss * patch_mask[:, :-1].float()).mean()
                metrics[f"blt/local_encoder_loss_lookahead_{lookahead_idx}"] = local_encoder_loss_lookahead / patch_mask[:, :-1].float().mean()
                metrics[f"blt/local_encoder_cos_sim_lookahead_{lookahead_idx}"] = (F.cosine_similarity(
                    h_lookahead.float(),
                    teacher_hidden_states[lookahead_idx][:, :-1].float(),
                    dim=-1,
                ) * patch_mask[:, :-1].float()).mean() / patch_mask[:, :-1].float().mean()

                local_encoder_loss += local_encoder_loss_lookahead * blt_config.encoder_loss_lookahead_weights[lookahead_idx]
        else:
            local_encoder_loss = torch.nan

        if not skip_blocks and not skip_teacher and not skip_teacher_blocks:
            assert logprobs is not None
            assert main_path_logprobs is not None
            assert teacher_logprobs is not None
            assert teacher_main_path_logprobs is not None
            assert teacher_embeds is not None

            if isinstance(self.teacher, BLTTransformer):
                elementwise_mse_local_decoder_loss = rep_compare_fn(
                    h_out,
                    teacher_last_hidden_state,
                )
                mse_local_decoder_loss = (elementwise_mse_local_decoder_loss * byte_mask).mean()

                # we can use standard KL!
                elementwise_kl_local_decoder_loss = F.kl_div(
                    logprobs.float(),
                    teacher_logprobs.float(),
                    reduction="none",
                    log_target=True,
                ).sum(-1)
                kl_local_decoder_loss = (elementwise_kl_local_decoder_loss * byte_mask).mean()

                if blt_config.decoder_use_mse_loss:
                    local_decoder_loss = mse_local_decoder_loss
                else:
                    local_decoder_loss = kl_local_decoder_loss

                metrics["blt/mse_local_decoder_loss"] = mse_local_decoder_loss / byte_mask.float().mean()
                metrics["blt/kl_local_decoder_loss"] = kl_local_decoder_loss / byte_mask.float().mean()
                metrics["blt/kl_local_decoder_teacher_mean_p"] = (
                    (torch.exp(teacher_main_path_logprobs) * byte_mask[:, 1:].float()).mean() / byte_mask[:, 1:].float().mean()
                )
                metrics["blt/kl_local_decoder_acc"] = (
                    ((logprobs.argmax(-1) == teacher_logprobs.argmax(-1)) * byte_mask).float().mean()
                ) / byte_mask.float().mean()
            else:
                if blt_config.decoder_use_mse_loss:
                    raise ValueError("MSE loss not implemented for ALM-style distillation")

                noised_mask = torch.zeros(input_ids.shape[0], device=self.device, dtype=torch.bool)
                noised_mask[noised_indices] = True
                non_noised_indices = torch.where(~noised_mask)[0]

                # use an ALM-style loss over non-noised examples
                local_decoder_loss_exhaustive, local_decoder_loss_simple, metrics = self._compute_alm_style_loss(
                    main_path_logprobs[non_noised_indices],
                    boundary_logprobs_for_decoder_loss[non_noised_indices] if boundary_logprobs_for_decoder_loss is not None else None,
                    boundary_labels[non_noised_indices] if boundary_labels is not None else None,
                    teacher_logprobs[non_noised_indices],
                    teacher_main_path_logprobs[non_noised_indices],
                    byte_mask[non_noised_indices],
                    patch_mask[non_noised_indices],
                    local_encoder_kwargs["patch_lens"][non_noised_indices],
                    local_encoder_kwargs["patch_ids"][non_noised_indices],
                    extra_kwargs["constituent_input_ids"][non_noised_indices],
                    ignore_index,
                    blt_config,
                    metrics,
                )

                if blt_config.use_exhaustive_decoder_loss:
                    local_decoder_loss = local_decoder_loss_exhaustive
                else:
                    local_decoder_loss = local_decoder_loss_simple
        else:
            local_decoder_loss = torch.nan

        # H-Net style embedding loss
        if teacher_embeds is not None and not isinstance(self.teacher, BLTTransformer):
            teacher_embs_repeated = torch.gather(
                teacher_embeds,
                dim=1,
                index=(local_encoder_kwargs["patch_ids"][:, 1:] - 1).unsqueeze(-1).expand(-1, -1, teacher_embeds.shape[-1]),
            )

            if blt_config.hnet_embed_loss_use_offset:
                elementwise_hnet_embed_loss = rep_compare_fn(
                    h_byte[:, 2:], # skip bos and first embedding to produce offset as in H-Net paper (match first patch byte to prev emb)
                    teacher_embs_repeated[:, :-1]
                )
                hnet_embed_loss_mask = byte_mask[:, 2:]
            else:
                elementwise_hnet_embed_loss = rep_compare_fn(h_byte[:, 1:], teacher_embs_repeated)
                hnet_embed_loss_mask = byte_mask[:, 1:]

            hnet_embed_loss = (elementwise_hnet_embed_loss * hnet_embed_loss_mask.float()).mean()
            metrics[f"blt/hnet_embed_loss"] = hnet_embed_loss / hnet_embed_loss_mask.float().mean()
        else:
            hnet_embed_loss = torch.nan

        # compute the boundary loss
        if boundary_logprobs is not None:
            assert boundary_labels is not None

            boundary_byte_mask = byte_mask.clone()
            # shouldn't compute boundary loss over wrong boundaries
            # slight inaccuracies here because the div factor is not adjusted accordingly
            # but should not have a big impact (hopefully :) )
            if noised_indices is not None:
                boundary_byte_mask[noised_indices] = False

            elementwise_boundary_loss = blt_utils.binary_cross_entropy_with_logprobs(
                boundary_logprobs_for_loss,
                boundary_labels,
            )
            boundary_loss = (elementwise_boundary_loss * boundary_byte_mask).mean()
            boundary_acc = ((boundary_mask == (boundary_labels > 0)) * boundary_byte_mask).float().mean()
            metrics["blt/boundary_loss"] = boundary_loss / boundary_byte_mask.float().mean()
            metrics["blt/boundary_acc"] = boundary_acc / boundary_byte_mask.float().mean()
            metrics["blt/boundary_mean"] = (boundary_mask * boundary_byte_mask).float().mean() / boundary_byte_mask.float().mean()
            metrics["blt/boundary_threshold"] = torch.where(
                boundary_mask,
                torch.exp(boundary_logprobs),
                torch.ones_like(boundary_logprobs),
            ).min(-1).values.mean()
        else:
            boundary_loss = torch.nan

        # H-Net ratio loss
        if boundary_logprobs is not None:
            true_ratio = (boundary_mask * byte_mask).float().mean() / byte_mask.float().mean()
            average_prob = (torch.exp(boundary_logprobs) * byte_mask).float().mean() / byte_mask.float().mean()

            ratio_loss = (
                (1 - true_ratio) * (1 - average_prob) +
                (true_ratio) * (average_prob) * (blt_config.target_ratio - 1)
            ) * blt_config.target_ratio / (blt_config.target_ratio - 1)
            metrics["blt/ratio_loss"] = ratio_loss
        else:
            ratio_loss = torch.nan

        # finalize losses
        # NOTE: loss_div_factor is at *byte sequence level*.
        if loss_div_factor is not None:
            loss_div_factor = loss_div_factor / (h_byte.shape[0] * h_byte.shape[1])
        
        if patch_loss_div_factor is not None:
            patch_loss_div_factor = patch_loss_div_factor / (h_patch.shape[0] * h_patch.shape[1])

        ce_loss = self._finalize_loss(ce_loss, loss_div_factor=loss_div_factor)
        boundary_loss = self._finalize_loss(boundary_loss, loss_div_factor=loss_div_factor)
        local_encoder_loss = self._finalize_loss(local_encoder_loss, loss_div_factor=patch_loss_div_factor)
        # TODO: check div factor after update / take constituent_input_ids -100's into account
        local_decoder_loss = self._finalize_loss(local_decoder_loss, loss_div_factor=loss_div_factor)

        loss = 0.0
        for loss_name, loss_weight in zip(blt_config.losses, blt_config.loss_weights):
            if loss_weight == 0.0:
                continue
            if loss_name == "ce":
                loss = loss + ce_loss * loss_weight
            elif loss_name == "local_encoder":
                loss = loss + local_encoder_loss * loss_weight
            elif loss_name == "local_decoder":
                loss = loss + local_decoder_loss * loss_weight
            elif loss_name == "boundary" and boundary_loss is not None:
                loss = loss + boundary_loss * loss_weight
            elif loss_name == "hnet_embed":
                loss = loss + hnet_embed_loss * loss_weight
            elif loss_name == "ratio":
                loss = loss + ratio_loss * loss_weight
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
            raise ValueError("`blt_config` must be provided for original_trunk_forward")

        input_ids, labels, block_kwargs, lm_head_kwargs, local_encoder_kwargs, local_decoder_kwargs, extra_kwargs = self._prepare_inputs(
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

        h_byte, h_patch, (_, boundary_logprobs), boundary_mask = self.local_encoder(
            input_ids,
            teacher_force_boundaries=blt_config.teacher_force_boundaries,
            boundary_threshold=blt_config.boundary_threshold,
            **local_encoder_kwargs
        )

        if self.prepend_embedding is not None:
            h_patch_global = h_patch.clone()
            h_patch_global[:, 0] = self.prepend_embedding.weight.expand(h_patch.shape[0], -1)
        else:
            h_patch_global = h_patch[:, 1:]  # skip the first token, which is <bos>

        for block in self.blocks.values():
            # Mark sizes as dynamic for torch.compile().
            if self.compile_enabled:
                mark_dynamic(h_patch_global, (0, 1), strict=False)
            h_patch_global = block(h_patch_global, **block_kwargs)

        h_patch_after_global = h_patch_global

        if self.prepend_embedding is not None:
            h_patch_after_global = h_patch_after_global[:, 1:]

        h_patch = torch.zeros_like(h_patch)
        h_patch[:, 1:] = h_patch_after_global

        h_out = self.local_decoder(
            embeds=h_byte,
            patch_embeds=h_patch,
            boundary_logprobs=None if blt_config.teacher_force_boundaries else boundary_logprobs,
            boundary_mask=None if blt_config.teacher_force_boundaries else boundary_mask,
            **local_decoder_kwargs,
        )
        logits = self.lm_head(h_out, **lm_head_kwargs)

        if blt_config.eval_add_boundary_logp:
            raise NotImplementedError("`eval_add_boundary_logp` is not implemented for student_forward")

        ce_loss, _ = cross_entropy_loss(logits.view(-1, logits.shape[-1]), labels.view(-1))  # type: ignore

        return LMOutputWithLoss(
            logits=logits,
            loss=ce_loss,
            ce_loss=ce_loss,
            z_loss=None,
        ), {}

    def original_head_forward(
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
        if self.prepend_embedding is not None:
            raise NotImplementedError("`prepend_embedding` is not implemented for original_head_forward")

        if self.teacher is None or isinstance(self.teacher, BLTTransformer):
            raise ValueError("`teacher` must be provided and be a subword-level transformer for original_trunk_forward.")

        input_ids, labels, block_kwargs, lm_head_kwargs, local_encoder_kwargs, local_decoder_kwargs, extra_kwargs = self._prepare_inputs(
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

        h_byte, h_patch, _, _ = self.local_encoder(input_ids, **local_encoder_kwargs)

        teacher_logits: torch.Tensor
        teacher_logits, (_, _, teacher_embeds) = self._teacher_forward(  # type: ignore
            extra_kwargs["original_input_ids"],
            # this could leak information since we take the true last embedding
            # but not a problem for log likelihood eval since last token logits are unused
            inputs_embeds=h_patch[:, 1:],
            labels=None, # we will compute loss ourselves
            return_logits=True,
            skip_blocks=False,
            **kwargs,
        )
        teacher_labels = get_labels({"input_ids": extra_kwargs["original_input_ids"]}, label_ignore_index=ignore_index)
        ce_loss, _ = cross_entropy_loss(teacher_logits.view(-1, teacher_logits.shape[-1]), teacher_labels.view(-1))

        return LMOutputWithLoss(
            logits=teacher_logits,
            loss=ce_loss,
            ce_loss=ce_loss,
            z_loss=None,
        ), {}

    def original_trunk_forward(
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
        if self.prepend_embedding is not None:
            raise NotImplementedError("`prepend_embedding` is not implemented for original_head_forward")

        if blt_config is None:
            raise ValueError("`blt_config` must be provided for original_trunk_forward")

        if self.teacher is None or isinstance(self.teacher, BLTTransformer):
            raise ValueError("`teacher` must be provided and be a subword-level transformer for original_trunk_forward.")

        input_ids, labels, block_kwargs, lm_head_kwargs, local_encoder_kwargs, local_decoder_kwargs, extra_kwargs = self._prepare_inputs(
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

        teacher_logits: torch.Tensor
        teacher_embeds: torch.Tensor
        teacher_logits, (_, teacher_last_hidden_state, teacher_embeds) = self._teacher_forward(  # type: ignore
            extra_kwargs["original_input_ids"],
            # this could leak information since we take the true last embedding
            # but not a problem for log likelihood eval since last token logits are unused
            labels=None, # we will compute loss ourselves
            skip_blocks=False,
            **kwargs,
        )
        assert teacher_last_hidden_state is not None, "Teacher forward must return last_hidden_state if skip_blocks=False"

        h_byte, h_patch, (_, boundary_logprobs), boundary_mask = self.local_encoder(input_ids, **local_encoder_kwargs)

        h_patch[:, 1:] = teacher_last_hidden_state[:, :-1]

        h_out = self.local_decoder(
            embeds=h_byte,
            patch_embeds=h_patch,
            **local_decoder_kwargs,
        )

        logits = self.lm_head(h_out, **lm_head_kwargs)
        logprobs = F.log_softmax(logits.float(), dim=-1)
        main_path_logprobs = torch.gather(logprobs[:, :-1], -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        main_path_patch_logprobs = torch.zeros((teacher_embeds.shape[0], teacher_embeds.shape[1]), device=logprobs.device, dtype=logprobs.dtype)
        main_path_patch_logprobs = main_path_patch_logprobs.scatter_reduce(
            src=main_path_logprobs,
            dim=1,
            index=local_encoder_kwargs["patch_ids"][:, 1:] - 1,
            reduce="sum",
            include_self=False,
        )
        y_hat = main_path_patch_logprobs[:, 1:-1]

        if boundary_logprobs is not None:
            patch_end_indices = torch.cumsum(local_encoder_kwargs["patch_lens"], dim=1) - 1
            if blt_config.eval_add_boundary_logp:
                y_hat = y_hat + torch.gather(boundary_logprobs, -1, patch_end_indices)[:, 2:]

        remaining_logpmass = log1mexp(y_hat)
        remaining_logp_uniform = remaining_logpmass - math.log(teacher_logits.shape[2] - 1)  # -1 to skip the main path token

        teacher_logits.zero_()
        teacher_logits[:, :-2, :] = remaining_logp_uniform.unsqueeze(-1)
        teacher_logits.scatter_(
            -1,
            extra_kwargs["original_input_ids"][:, 1:-1].unsqueeze(-1),
            y_hat.to(teacher_logits.dtype).unsqueeze(-1),
        )
        teacher_labels = get_labels({"input_ids": extra_kwargs["original_input_ids"]}, label_ignore_index=ignore_index)
        ce_loss, _ = cross_entropy_loss(teacher_logits.view(-1, teacher_logits.shape[-1]), teacher_labels.view(-1))

        return LMOutputWithLoss(
            logits=teacher_logits,
            loss=ce_loss,
            ce_loss=ce_loss,
            z_loss=None,
        ), {}