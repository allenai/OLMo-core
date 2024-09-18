import logging
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Sequence, cast

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from olmo_core.config import Config, DType, StrEnum
from olmo_core.data.utils import get_cumulative_document_lengths
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import get_default_device, has_flash_attn

from ..attention import AttentionConfig, AttentionType
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForwardConfig
from ..layer_norm import LayerNorm, LayerNormConfig, LayerNormType
from ..rope import RoPEConfig, RoPEType
from .block import TransformerBlock, TransformerBlockConfig, TransformerBlockType
from .init import InitMethod

__all__ = [
    "TransformerConfig",
    "Transformer",
    "TransformerActivationCheckpointingConfig",
    "TransformerActivationCheckpointingMode",
]


log = logging.getLogger(__name__)


class TransformerActivationCheckpointingMode(StrEnum):
    full = "full"
    """Checkpoint every block."""
    selected_blocks = "selected_blocks"
    """Checkpoint only selected blocks."""
    selected_modules = "selected_modules"
    """Checkpoint only selected modules."""


@dataclass
class TransformerActivationCheckpointingConfig(Config):
    """
    Defines the activation checkpointing strategy for a transformer model.
    """

    mode: TransformerActivationCheckpointingMode = TransformerActivationCheckpointingMode.full

    block_interval: Optional[int] = None
    """
    Required when :data:`mode` is "selected_blocks". Determines which blocks are wrapped.
    """

    modules: Optional[List[str]] = None
    """
    Required when :data:`mode` is "selected_modules". A list of modules names to wrap for
    activation checkpointing. Globs are supported.
    """

    def __post_init__(self):
        if (
            self.mode == TransformerActivationCheckpointingMode.selected_blocks
            and self.block_interval is None
        ):
            raise OLMoConfigurationError(
                "'block_interval' is required for 'selected_blocks' activation checkpointing"
            )
        elif (
            self.mode == TransformerActivationCheckpointingMode.selected_modules
            and self.modules is None
        ):
            raise OLMoConfigurationError(
                "'modules' is required for 'selected_modules' activation checkpointing"
            )


@dataclass
class TransformerConfig(Config):
    """
    A config for easily building transformer models.

    See :class:`Transformer` for a description of the parameters.
    """

    d_model: int
    vocab_size: int
    n_layers: int
    block: TransformerBlockConfig
    layer_norm: LayerNormConfig
    bias: bool = True
    dtype: DType = DType.float32
    init_method: InitMethod = InitMethod.normal
    init_seed: int = 0
    compile: bool = False
    dp_config: Optional[DataParallelConfig] = None
    ac_config: Optional[TransformerActivationCheckpointingConfig] = None

    def build(
        self,
        *,
        init_device: str = "cpu",
        device: Optional[torch.device] = None,
        dp_mesh: Optional[DeviceMesh] = None,
        max_seq_len: Optional[int] = None,
    ) -> "Transformer":
        """
        Build the model corresponding to this config, potentially applying activation checkpointing,
        compilation, FSDP or DDP, etc, and eventually calling :meth:`Transformer.init_weights()`.

        :param init_device: The device to put the parameters on during initialization. In a
            distributed setting it usually makes sense to set this to "meta".
        :param device: The device to put the model on after initialization.
        :param dp_mesh: Data parallel device mesh. This can be used to configure hybrid sharding
            with FSDP. See :func:`~olmo_core.distributed.utils.init_hybrid_shard_mesh()` for
            easily creating such a mesh.
        :param max_seq_len: The maximum sequence length expected.
        """
        device = device or get_default_device()

        log.info(
            f"Building transformer with {self.num_params:,d} total params, "
            f"{self.num_non_embedding_params:,d} non-embedding params"
        )
        model = Transformer(
            d_model=self.d_model,
            vocab_size=self.vocab_size,
            n_layers=self.n_layers,
            block=self.block,
            layer_norm=self.layer_norm,
            bias=self.bias,
            dtype=self.dtype.as_pt(),
            init_method=self.init_method,
            init_device=init_device,
            init_seed=self.init_seed,
        )
        log.info("%s", model)

        # Maybe apply activation checkpointing.
        if self.ac_config is not None:
            model.apply_activation_checkpointing(
                self.ac_config.mode,
                block_interval=self.ac_config.block_interval,
                modules=self.ac_config.modules,
            )

        # Maybe compile.
        if self.compile:
            model.apply_compile()

        # Maybe wrap for data parallel.
        if self.dp_config is not None:
            if self.dp_config.name == DataParallelType.fsdp:
                model.apply_fsdp(
                    dp_mesh=dp_mesh,
                    param_dtype=self.dp_config.param_dtype.as_pt()
                    if self.dp_config.param_dtype is not None
                    else None,
                    reduce_dtype=self.dp_config.reduce_dtype.as_pt(),
                )
            elif self.dp_config.name == DataParallelType.ddp:
                model.apply_ddp(dp_mesh=dp_mesh, compile_enabled=self.compile)
            else:
                raise NotImplementedError(self.dp_config.name)

        # Materialize and init parameters.
        if device != torch.device(init_device):
            model.to_empty(device=device)
        model.init_weights(max_seq_len=max_seq_len, device=device)

        return model

    @property
    def num_params(self) -> int:
        """
        The total number of parameters that a model from this config would have.
        """

        def layer_norm_params(layer_norm: LayerNormConfig) -> int:
            ln_params = 0
            if layer_norm.elementwise_affine:
                ln_params += self.d_model
                if layer_norm.bias:
                    ln_params += self.d_model
            return ln_params

        num_params = 0

        # Embedding params.
        num_params += self.d_model * self.vocab_size

        block_params = 0

        n_heads = self.block.attention.n_heads
        n_kv_heads = self.block.attention.n_kv_heads or n_heads
        head_dim = self.d_model // n_heads

        # Block attention Q projection.
        block_params += self.d_model * self.d_model
        if self.block.attention.bias:
            block_params += self.d_model

        # Block attention KV projections.
        block_params += 2 * self.d_model * n_kv_heads * head_dim
        if self.block.attention.bias:
            block_params += 2 * n_kv_heads * head_dim

        # Block attention QK norm.
        if self.block.attention.qk_norm is not None:
            block_params += 2 * layer_norm_params(self.block.attention.qk_norm)

        # Block attention out.
        block_params += self.d_model * self.d_model
        if self.block.attention.bias:
            block_params += self.d_model

        # Block attention norm.
        block_params += layer_norm_params(self.block.layer_norm)

        # Block feed forward.
        block_params += 3 * self.d_model * self.block.feed_forward.hidden_size
        if self.block.feed_forward.bias:
            block_params += 2 * self.block.feed_forward.hidden_size + self.d_model

        # Block feed forward norm.
        block_params += layer_norm_params(self.block.layer_norm)

        # All block params.
        num_params += self.n_layers * block_params

        # Final layer norm.
        num_params += layer_norm_params(self.layer_norm)

        # Final FF out.
        num_params += self.d_model * self.vocab_size
        if self.bias:
            num_params += self.vocab_size

        return num_params

    @property
    def num_non_embedding_params(self) -> int:
        """
        The number of parameters excluding embedding parameters.
        """
        return self.num_params - self.d_model * self.vocab_size

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Get the approximate number of flops per token.
        """
        n, h, q, t = (
            self.n_layers,
            self.block.attention.n_heads,
            self.d_model // self.block.attention.n_heads,
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

    @classmethod
    def olmo_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B OLMo model config.
        """
        return cls.llama2_1B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            **kwargs,
        )

    @classmethod
    def olmo_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 7B OLMo model config.
        """
        return cls.llama2_7B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            **kwargs,
        )

    @classmethod
    def olmo_13B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 13B OLMo model config.
        """
        return cls.llama2_13B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            **kwargs,
        )

    @classmethod
    def llama2_271M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 271M Llama2-like model config.
        """
        return cls.llama_like(
            d_model=1024,
            vocab_size=vocab_size,
            n_layers=16,
            n_heads=8,
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=18,
            n_heads=16,
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 7B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=4096,
            vocab_size=vocab_size,
            n_layers=32,
            n_heads=32,
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_13B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 7B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=5120,
            vocab_size=vocab_size,
            n_layers=40,
            n_heads=40,
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_26B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 26B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=5120,
            vocab_size=vocab_size,
            n_layers=80,
            n_heads=40,
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_70B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 70B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=8192,
            vocab_size=vocab_size,
            n_layers=80,
            n_heads=64,
            n_kv_heads=8,
            rope_theta=kwargs.pop("rope_theta", 10_000),
            hidden_size_multiplier=1.3,
            hidden_size_multiple_of=4096,
            **kwargs,
        )

    @classmethod
    def llama3_8B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        An 8B Llama3-like model config.
        """
        return cls.llama_like(
            d_model=4096,
            vocab_size=vocab_size,
            n_layers=32,
            n_heads=32,
            n_kv_heads=8,
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiplier=1.3,
            hidden_size_multiple_of=1024,
            **kwargs,
        )

    @classmethod
    def llama3_70B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 70B Llama3-like model config.
        """
        return cls.llama_like(
            d_model=8196,
            vocab_size=vocab_size,
            n_layers=80,
            n_heads=64,
            n_kv_heads=8,
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiplier=1.3,
            hidden_size_multiple_of=4096,
            **kwargs,
        )

    @classmethod
    def llama3_405B(
        cls,
        vocab_size: int,
        **kwargs,
    ) -> "TransformerConfig":
        """
        A 405B Llama3-like model config.
        """
        return cls.llama_like(
            d_model=16384,
            vocab_size=vocab_size,
            n_layers=126,
            n_heads=128,
            n_kv_heads=8,
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiplier=1.2,
            hidden_size_multiple_of=4096,
            **kwargs,
        )

    @classmethod
    def llama_like(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        qk_norm: bool = False,
        rope_theta: int = 500_000,
        rope_type: Optional[RoPEType] = None,
        hidden_size_multiple_of: int = 256,
        hidden_size_multiplier: Optional[float] = None,
        fused_ops: Optional[bool] = None,
        use_flash: Optional[bool] = None,
        block_name: TransformerBlockType = TransformerBlockType.default,
        dtype: DType = DType.float32,
        compile: bool = False,
        **kwargs,
    ) -> "TransformerConfig":
        """
        Create a Llama-like model configuration.

        :param hidden_size_multiple_of: Ensure the FFN hidden size is a multiple of this value.
        :param hidden_size_multiplier: Custom multiplier for the FFN hidden size.
        :param fused_ops: Use fused operations where possible. Defaults to ``True`` if flash-attn is
            installed and ``compile=False``, otherwise ``False``.
        :param use_flash: Use flash-attn. Defaults to ``True`` if flash-attn is
            installed and ``compile=False``, otherwise ``False``.
        :param dtype: The default data type to use for all parameters.
        """
        if fused_ops is None:
            fused_ops = False if compile else has_flash_attn()
        if use_flash is None:
            use_flash = False if compile else has_flash_attn()

        # Resolve hidden size of FFN in blocks.
        hidden_size = int(8 * d_model / 3)
        if hidden_size_multiplier is not None:
            hidden_size = int(hidden_size_multiplier * hidden_size)
        hidden_size = hidden_size_multiple_of * (
            (hidden_size + hidden_size_multiple_of - 1) // hidden_size_multiple_of
        )

        # Configure global layer norm.
        layer_norm = LayerNormConfig(
            name=LayerNormType.fused_rms if fused_ops else LayerNormType.rms,
            eps=1e-5,
            bias=False,
            #  dtype=dtype,  # TODO: allow low precision LN?
        )

        # Decide on attention/rope implementations.
        att_type = AttentionType.default
        if rope_type is None:
            rope_type = RoPEType.default
            if fused_ops and n_kv_heads is None:  # fused attention not compatible with MQA/GQA.
                att_type = AttentionType.fused
                rope_type = RoPEType.fused

        # Configure blocks.
        block = TransformerBlockConfig(
            name=block_name,
            attention=AttentionConfig(
                name=att_type,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                bias=False,
                rope=RoPEConfig(name=rope_type, theta=rope_theta),
                qk_norm=layer_norm if qk_norm else None,
                use_flash=use_flash,
                dtype=dtype,
            ),
            feed_forward=FeedForwardConfig(hidden_size=hidden_size, bias=False, dtype=dtype),
            layer_norm=layer_norm,
        )

        return cls(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            layer_norm=layer_norm,
            bias=False,
            dtype=dtype,
            compile=compile,
            **kwargs,
        )


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
            self.init_method.init_feed_forward(
                block.feed_forward,
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
        out = self.w_out(h).float() if self.w_out is not None else h
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
        """
        # Adapted from
        # https://github.com/pytorch/torchtitan/blob/90c889e972b56b9faadebbb78fc985dedc537ed9/torchtitan/parallelisms/parallelize_llama.py#L289

        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype or self.w_out.weight.dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)

        for block_id, block in enumerate(self.blocks):
            if pp_enabled:
                # For PP, do not reshard after forward to avoid per-microbatch
                # all-gathers, which can be expensive and non-overlapped
                reshard_after_forward = False
            else:
                # As an optimization, do not reshard after forward for the last
                # transformer block since FSDP would prefetch it immediately
                reshard_after_forward = int(block_id) < len(self.blocks) - 1
            fully_shard(block, reshard_after_forward=reshard_after_forward, **fsdp_config)

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
