import logging
from dataclasses import dataclass
from typing import Optional

from olmo_core.config import Config, DType, StrEnum

from ..attention import AttentionConfig, AttentionType
from ..feed_forward import FeedForwardConfig, FeedForwardType
from ..layer_norm import LayerNormConfig, LayerNormType
from ..lm_head import LMHeadConfig, LMHeadType
from ..moe import MoEConfig, MoERouterConfig, MoEType, SharedMLPConfig
from ..rope import RoPEConfig, RoPEScalingConfig, RoPEType
from .block import TransformerBlockConfig, TransformerBlockType
from .init import InitMethod
from .model import MoETransformer, NormalizedTransformer, Transformer

log = logging.getLogger(__name__)


class TransformerType(StrEnum):
    """
    An enumeration of transformer implementations.
    """

    default = "default"
    """
    ➡️ :class:`Transformer`
    """

    normalized = "normalized"
    """
    ➡️ :class:`NormalizedTransformer` (nGPT)
    """

    moe = "moe"
    """
    ➡️ :class:`MoETransformer`
    """


@dataclass
class TransformerConfig(Config):
    """
    A config for easily building transformer models.

    :param name: The name of the implementation.

    See :class:`Transformer` for a description of the other parameters.
    """

    d_model: int
    vocab_size: int
    n_layers: int
    block: TransformerBlockConfig
    lm_head: LMHeadConfig
    name: TransformerType = TransformerType.default
    dtype: DType = DType.float32
    init_method: InitMethod = InitMethod.normal
    init_seed: int = 0

    def build(
        self,
        *,
        init_device: str = "cpu",
    ) -> Transformer:
        """
        Build the model corresponding to this config.

        :param init_device: The device to put the parameters on during initialization. In a
            distributed setting it usually makes sense to set this to "meta".
        """
        log.info(
            f"Building transformer with {self.num_params:,d} total params, "
            f"{self.num_non_embedding_params:,d} non-embedding params"
        )
        model: Transformer
        if self.name == TransformerType.default:
            model = Transformer(
                d_model=self.d_model,
                vocab_size=self.vocab_size,
                n_layers=self.n_layers,
                block=self.block,
                lm_head=self.lm_head,
                dtype=self.dtype.as_pt(),
                init_method=self.init_method,
                init_device=init_device,
                init_seed=self.init_seed,
            )
        elif self.name == TransformerType.normalized:
            model = NormalizedTransformer(
                d_model=self.d_model,
                vocab_size=self.vocab_size,
                n_layers=self.n_layers,
                block=self.block,
                lm_head=self.lm_head,
                dtype=self.dtype.as_pt(),
                init_method=self.init_method,
                init_device=init_device,
                init_seed=self.init_seed,
            )
        elif self.name == TransformerType.moe:
            model = MoETransformer(
                d_model=self.d_model,
                vocab_size=self.vocab_size,
                n_layers=self.n_layers,
                block=self.block,
                lm_head=self.lm_head,
                dtype=self.dtype.as_pt(),
                init_method=self.init_method,
                init_device=init_device,
                init_seed=self.init_seed,
            )
        else:
            raise NotImplementedError(self.name)

        log.info("%s", model)

        return model

    @property
    def num_params(self) -> int:
        """
        The total number of parameters that a model from this config would have.
        """

        num_params = 0

        # Embedding params.
        num_params += self.d_model * self.vocab_size

        block_params = 0

        # Block attn and MLP scaling factors.
        if self.block.name == TransformerBlockType.normalized:
            block_params += 2 * self.d_model

        # Block attention params.
        block_params += self.block.attention.num_params(self.d_model)

        # Block attention norm.
        if self.block.layer_norm is not None:
            block_params += self.block.layer_norm.num_params(self.d_model)

        # Block feed forward.
        if self.block.feed_forward is not None:
            block_params += self.block.feed_forward.num_params(self.d_model)
        elif self.block.feed_forward_moe is not None:
            block_params += self.block.feed_forward_moe.num_params(self.d_model)

        # Block feed forward norm.
        if self.block.layer_norm is not None:
            block_params += self.block.layer_norm.num_params(self.d_model)

        # All block params.
        num_params += self.n_layers * block_params

        # LM head.
        num_params += self.lm_head.num_params(self.d_model, self.vocab_size)

        return num_params

    @property
    def num_active_params(self) -> int:
        """
        The total number of active parameters that a model from this config would have.
        """
        num_params = self.num_params
        if self.block.feed_forward_moe is None:
            return num_params
        diff_per_block = self.block.feed_forward_moe.num_params(
            self.d_model
        ) - self.block.feed_forward_moe.num_active_params(self.d_model)
        total_diff = self.n_layers * diff_per_block
        return num_params - total_diff

    @property
    def num_non_embedding_params(self) -> int:
        """
        The number of parameters excluding embedding parameters.
        """
        return self.num_params - self.d_model * self.vocab_size

    @property
    def num_active_non_embedding_params(self) -> int:
        """
        The number of active parameters excluding embedding parameters.
        """
        return self.num_active_params - self.d_model * self.vocab_size

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
    def olmo2_190M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=768,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 12),
            n_heads=kwargs.pop("n_heads", 12),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_370M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=1024,
            hidden_size_multiplier=1.4,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_600M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=1344,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_760M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=1536,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
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
    def olmo2_3B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=3328,
            hidden_size_multiplier=1.4,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
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
    def olmo2_13B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
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
    def olmo2_32B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 32B OLMo model config.
        """
        d_model = 5120
        return cls.llama_like(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=kwargs.pop("n_layers", 64),
            n_heads=kwargs.pop("n_heads", 40),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiple_of=kwargs.pop("hidden_size_multiple_of", 512),
            hidden_size_multiplier=kwargs.pop("hidden_size_multiplier", 27648 / (8 * d_model / 3)),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def smallmoe(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        d_model = kwargs.pop("d_model", 768)
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 12),
            n_heads=kwargs.pop("n_heads", 12),
            name=kwargs.pop("name", TransformerType.moe),
            block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            feed_forward_moe=MoEConfig(
                name=MoEType.default,
                num_experts=32,
                hidden_size=int(0.5 * d_model),
                router=MoERouterConfig(top_k=4, bias=False),
                shared_mlp=SharedMLPConfig(hidden_size=d_model * 2, bias=False),
                lb_loss_weight=0.01,
                z_loss_weight=0.001,
            ),
        )

    @classmethod
    def olmoe2_large(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        d_model = kwargs.pop("d_model", 4096)
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 32),
            n_heads=kwargs.pop("n_heads", 32),
            name=kwargs.pop("name", TransformerType.moe),
            block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            feed_forward_moe=MoEConfig(
                name=MoEType.default,
                num_experts=64,
                hidden_size=int(0.5 * d_model),
                router=MoERouterConfig(top_k=8, bias=False),
                shared_mlp=SharedMLPConfig(hidden_size=d_model * 2, bias=False),
                lb_loss_weight=0.01,
                z_loss_weight=0.001,
            ),
        )

    @classmethod
    def olmoe_1B_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        d_model = kwargs.pop("d_model", 2048)
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            name=kwargs.pop("name", TransformerType.moe),
            block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            feed_forward_moe=MoEConfig(
                name=MoEType.dropless,
                num_experts=64,
                hidden_size=int(0.5 * d_model),
                router=MoERouterConfig(top_k=8, bias=False),
                lb_loss_weight=0.01,
                z_loss_weight=0.001,
            ),
        )

    @classmethod
    def ngpt_271M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 271M nGPT model config.
        """
        return cls.ngpt_like(
            d_model=1024,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            **kwargs,
        )

    @classmethod
    def ngpt_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B nGPT model config.
        """
        return cls.ngpt_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 18),
            n_heads=kwargs.pop("n_heads", 16),
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
    def llama3_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B Llama3-like model config.
        """
        return cls.llama_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 32),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiplier=1.5,
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
        fused_ops: bool = False,
        use_flash: bool = False,
        block_name: TransformerBlockType = TransformerBlockType.default,
        dtype: DType = DType.float32,
        rope_scaling: Optional[RoPEScalingConfig] = None,
        feed_forward: Optional[FeedForwardConfig] = None,
        feed_forward_moe: Optional[MoEConfig] = None,
        **kwargs,
    ) -> "TransformerConfig":
        """
        Create a Llama-like model configuration.

        :param hidden_size_multiple_of: Ensure the FFN hidden size is a multiple of this value.
        :param hidden_size_multiplier: Custom multiplier for the FFN hidden size.
        :param fused_ops: Use fused operations where possible.
        :param use_flash: Use flash-attn.
        :param dtype: The default data type to use for all parameters.
        """
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

        # Feed-forward.
        if feed_forward is None and feed_forward_moe is None:
            feed_forward = FeedForwardConfig(hidden_size=hidden_size, bias=False, dtype=dtype)

        # Configure blocks.
        block = TransformerBlockConfig(
            name=block_name,
            attention=AttentionConfig(
                name=att_type,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                bias=False,
                rope=RoPEConfig(name=rope_type, theta=rope_theta, scaling=rope_scaling),
                qk_norm=layer_norm if qk_norm else None,
                use_flash=use_flash,
                dtype=dtype,
            ),
            feed_forward=feed_forward,
            feed_forward_moe=feed_forward_moe,
            layer_norm=layer_norm,
        )

        return cls(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
            dtype=dtype,
            **kwargs,
        )

    @classmethod
    def ngpt_like(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        qk_norm: bool = True,
        rope_theta: int = 500_000,
        hidden_size_multiple_of: int = 256,
        hidden_size_multiplier: Optional[float] = None,
        use_flash: bool = False,
        dtype: DType = DType.float32,
        **kwargs,
    ) -> "TransformerConfig":
        """
        Create an nGPT-like model configuration.
        """
        # Resolve hidden size of FFN in blocks.
        hidden_size = int(8 * d_model / 3)
        if hidden_size_multiplier is not None:
            hidden_size = int(hidden_size_multiplier * hidden_size)
        hidden_size = hidden_size_multiple_of * (
            (hidden_size + hidden_size_multiple_of - 1) // hidden_size_multiple_of
        )

        # Configure blocks.
        block = TransformerBlockConfig(
            name=TransformerBlockType.normalized,
            attention=AttentionConfig(
                name=AttentionType.normalized,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                qk_norm=None if not qk_norm else LayerNormConfig(name=LayerNormType.l2_norm),
                rope=RoPEConfig(name=RoPEType.default, theta=rope_theta),
                use_flash=use_flash,
                dtype=dtype,
            ),
            feed_forward=FeedForwardConfig(
                name=FeedForwardType.normalized, hidden_size=hidden_size, dtype=dtype
            ),
        )

        return cls(
            name=TransformerType.normalized,
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=LMHeadConfig(name=LMHeadType.normalized, dtype=dtype),
            dtype=dtype,
            init_method=InitMethod.normalized,
            **kwargs,
        )
