import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.distributed import DeviceMesh

from olmo_core.config import Config, DType
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.utils import get_default_device, has_flash_attn

from ..attention import AttentionConfig, AttentionType
from ..feed_forward import FeedForwardConfig
from ..layer_norm import LayerNormConfig, LayerNormType
from ..rope import RoPEConfig, RoPEType
from .block import TransformerBlockConfig, TransformerBlockType
from .init import InitMethod
from .model import (
    Transformer,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelWrappingStrategy,
)

__all__ = [
    "TransformerDataParallelConfig",
    "TransformerActivationCheckpointingConfig",
]


log = logging.getLogger(__name__)


@dataclass
class TransformerDataParallelConfig(DataParallelConfig):
    wrapping_strategy: TransformerDataParallelWrappingStrategy = (
        TransformerDataParallelWrappingStrategy.full
    )
    """
    Wrapping strategy.
    """


@beta_feature
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

    :param compile: Whether to compile the model with ``torch.compile``.
    :param dp_config: Data parallel configuration.
    :param ac_config: Activation checkpointing configuration.
    :param float8_config: Float8 training configuration.

    See :class:`Transformer` for a description of the other parameters.
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
    dp_config: Optional[TransformerDataParallelConfig] = None
    ac_config: Optional[TransformerActivationCheckpointingConfig] = None
    float8_config: Optional[Float8Config] = None

    def build(
        self,
        *,
        init_device: str = "cpu",
        device: Optional[torch.device] = None,
        dp_mesh: Optional[DeviceMesh] = None,
        max_seq_len: Optional[int] = None,
    ) -> Transformer:
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

        # Maybe convert linear layers to Float8 linear layers.
        if self.float8_config is not None and self.float8_config.enabled:
            if self.float8_config.compile is None and self.compile:
                self.float8_config.compile = True
            self.float8_config.convert_to_float8_training(model, modules_to_ignore={"w_out"})

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
                    wrapping_strategy=self.dp_config.wrapping_strategy,
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
        if "moe" not in self.block.name:
            assert self.block.feed_forward is not None
            block_params += 3 * self.d_model * self.block.feed_forward.hidden_size
            if self.block.feed_forward.bias:
                block_params += 2 * self.block.feed_forward.hidden_size + self.d_model
        else:
            assert self.block.feed_forward_moe is not None
            block_params += self.block.feed_forward_moe.num_params(self.d_model)

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
            layer_norm_eps=1e-6,
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
            layer_norm_eps=1e-6,
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
            layer_norm_eps=1e-6,
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
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 8),
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
            n_layers=kwargs.pop("n_layers", 18),
            n_heads=kwargs.pop("n_heads", 16),
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
            n_layers=kwargs.pop("n_layers", 32),
            n_heads=kwargs.pop("n_heads", 32),
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
            n_layers=kwargs.pop("n_layers", 40),
            n_heads=kwargs.pop("n_heads", 40),
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
            n_layers=kwargs.pop("n_layers", 80),
            n_heads=kwargs.pop("n_heads", 40),
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
            n_layers=kwargs.pop("n_layers", 80),
            n_heads=kwargs.pop("n_heads", 64),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
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
            n_layers=kwargs.pop("n_layers", 32),
            n_heads=kwargs.pop("n_heads", 32),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
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
            n_layers=kwargs.pop("n_layers", 80),
            n_heads=kwargs.pop("n_heads", 64),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
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
            n_layers=kwargs.pop("n_layers", 126),
            n_heads=kwargs.pop("n_heads", 128),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
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
        layer_norm_eps: float = 1e-5,
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
            eps=layer_norm_eps,
            bias=False,
            dtype=dtype,
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
