import logging
from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Dict, List, Optional

from olmo_core.config import Config, DType, StrEnum
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import ensure_multiple_of

from ..attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    SlidingWindowAttentionConfig,
)
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForwardConfig, FeedForwardType
from ..layer_norm import LayerNormConfig, LayerNormType
from ..lm_head import LMHeadConfig, LMHeadType
from ..moe import MoEConfig, MoERouterConfig, MoEType
from ..rope import RoPEConfig, RoPEScalingConfig, RoPEType
from .init import InitMethod

if TYPE_CHECKING:
    from .block import TransformerBlockBase
    from .model import Transformer

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
    budget = "budget"
    """Checkpoint based on a budget."""


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


class TransformerBlockType(StrEnum):
    """
    An enumeration of the different transformer block implementations.
    """

    default = "default"
    """
    ➡️ :class:`TransformerBlock`
    """

    reordered_norm = "reordered_norm"
    """
    ➡️ :class:`ReorderedNormTransformerBlock`
    """

    normalized = "normalized"
    """
    ➡️ :class:`NormalizedTransformerBlock`
    """

    moe = "moe"
    """
    ➡️ :class:`MoETransformerBlock`
    """

    moe_reordered_norm = "moe_reordered_norm"
    """
    ➡️ :class:`MoEReorderedNormTransformerBlock`
    """

    moe_hybrid = "moe_hybrid"
    """
    ➡️ :class:`MoEHybridTransformerBlock`
    """

    moe_hybrid_reordered_norm = "moe_hybrid_reordered_norm"
    """
    ➡️ :class:`MoEHybridReorderedNormTransformerBlock`
    """

    default_scaled = "default_scaled"
    """
    ➡️ :class:`LayerNormScaledTransformerBlock` (applies LayerNorm Scaling)
    """


@dataclass
class TransformerBlockConfig(Config):
    """
    A configuration class for easily building transformer blocks.
    """

    attention: AttentionConfig
    """
    The attention config.
    """
    layer_norm: Optional[LayerNormConfig] = None
    """
    The layer norm config.
    """
    feed_forward: Optional[FeedForwardConfig] = None
    """
    The feed-forward config, required for non-MoE blocks.
    """
    feed_forward_moe: Optional[MoEConfig] = None
    """
    The config for the MoE feed-forward layer. Required for MoE blocks.
    """
    name: TransformerBlockType = TransformerBlockType.default
    """
    The block type.
    """
    dropout: Optional[float] = None
    """
    Dropout probability.
    """

    def build(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "TransformerBlockBase":
        from .block import (
            LayerNormScaledTransformerBlock,
            MoEHybridReorderedNormTransformerBlock,
            MoEHybridTransformerBlock,
            MoEReorderedNormTransformerBlock,
            MoETransformerBlock,
            NormalizedTransformerBlock,
            ReorderedNormTransformerBlock,
            TransformerBlock,
        )

        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            init_device=init_device,
            cache=cache,
        )

        try:
            if self.name == TransformerBlockType.default:
                return TransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.default_scaled:
                return LayerNormScaledTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.reordered_norm:
                return ReorderedNormTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.normalized:
                return NormalizedTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe:
                return MoETransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe_reordered_norm:
                return MoEReorderedNormTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe_hybrid:
                return MoEHybridTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe_hybrid_reordered_norm:
                return MoEHybridReorderedNormTransformerBlock(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e

    def num_params(self, d_model: int) -> int:
        block_params = 0

        # Block attn and MLP scaling factors.
        if self.name == TransformerBlockType.normalized:
            block_params += 2 * d_model

        # Block attention params.
        block_params += self.attention.num_params(d_model)
        if self.layer_norm is not None:
            block_params += self.layer_norm.num_params(d_model)

        # Block feed forward (dense and/or sparse).
        if self.feed_forward is not None:
            block_params += self.feed_forward.num_params(d_model)
            if self.layer_norm is not None:
                block_params += self.layer_norm.num_params(d_model)
        if self.feed_forward_moe is not None:
            block_params += self.feed_forward_moe.num_params(d_model)
            if self.layer_norm is not None:
                block_params += self.layer_norm.num_params(d_model)

        return block_params

    def num_active_params(self, d_model: int) -> int:
        num_params = self.num_params(d_model)
        if self.feed_forward_moe is None:
            return num_params

        num_inactive_params = self.feed_forward_moe.num_params(
            d_model
        ) - self.feed_forward_moe.num_active_params(d_model)
        return num_params - num_inactive_params


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
    init_std: float = 0.02
    freeze_params: Optional[List[str]] = None
    block_overrides: Optional[Dict[int, TransformerBlockConfig]] = None

    def build(
        self,
        *,
        init_device: str = "cpu",
    ) -> "Transformer":
        """
        Build the model corresponding to this config.

        :param init_device: The device to put the parameters on during initialization. In a
            distributed setting it usually makes sense to set this to "meta".
        """
        from .model import MoETransformer, NormalizedTransformer, Transformer

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
                init_std=self.init_std,
                block_overrides=self.block_overrides,
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
                init_std=self.init_std,
                block_overrides=self.block_overrides,
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
                init_std=self.init_std,
                block_overrides=self.block_overrides,
            )
        else:
            raise NotImplementedError(self.name)

        if self.freeze_params:
            for name, param in model.named_parameters():
                for pattern in self.freeze_params:
                    if fnmatch(name, pattern):
                        param.requires_grad = False
                        log.info(f"Param '{name}' will be frozen")
                        break
                else:
                    log.info(f"Param '{name}' will be trainable")

        log.info("%s", model)
        log.info(
            f"Built model with:\n"
            f"- {model.num_params:,d} total params\n"
            f"- {model.num_non_embedding_params:,d} non-embedding params\n"
            f"- {model.num_trainable_params:,d} trainable params"
        )

        return model

    @property
    def num_params(self) -> int:
        """
        The total number of parameters that a model from this config would have.
        """
        num_params = 0

        # Embedding params.
        num_params += self.d_model * self.vocab_size

        # All block params.
        num_block_params = self.block.num_params(self.d_model)
        if self.block_overrides is None:
            num_params += self.n_layers * num_block_params
        else:
            for idx in range(self.n_layers):
                if idx in self.block_overrides:
                    num_params += self.block_overrides[idx].num_params(self.d_model)
                else:
                    num_params += num_block_params

        # LM head.
        num_params += self.lm_head.num_params(self.d_model, self.vocab_size)

        return num_params

    @property
    def num_active_params(self) -> int:
        """
        The total number of active parameters that a model from this config would have.
        """
        num_active_params = 0

        # Embedding params.
        num_active_params += self.d_model * self.vocab_size

        # All block active params.
        num_active_block_params = self.block.num_active_params(self.d_model)
        if self.block_overrides is None:
            num_active_params += self.n_layers * num_active_block_params
        else:
            for idx in range(self.n_layers):
                if idx in self.block_overrides:
                    num_active_params += self.block_overrides[idx].num_active_params(self.d_model)
                else:
                    num_active_params += num_active_block_params

        # LM head.
        num_active_params += self.lm_head.num_params(self.d_model, self.vocab_size)

        return num_active_params

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
    def olmo2_30M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=256,
            n_layers=kwargs.pop("n_layers", 4),
            n_heads=kwargs.pop("n_heads", 8),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

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
        A 1B OLMo2 model config.

        This is different from the OLMo 1B from the old OLMo trainer.
        """
        return cls.llama2_1B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            hidden_size_multiplier=1.5,
            **kwargs,
        )

    @classmethod
    def olmo2_1B_v2(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B OLMo2 model config.

        This matches the OLMo 1B from the old OLMo trainer.
        """
        return cls.llama2_1B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            n_layers=kwargs.pop("n_layers", 16),
            hidden_size_multiplier=kwargs.pop("hidden_size_multiplier", 1.5),
            **kwargs,
        )

    @classmethod
    def olmo2_3B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 3B OLMo2 model config.
        """
        return cls.llama_like(
            d_model=3328,
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
    def olmo2_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 7B OLMo2 model config.
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
        A 13B OLMo2 model config.
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
        A 32B OLMo2 model config.
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
    def olmo3_190M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 190M OLMo3 model config.
        """
        config = cls.olmo2_190M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 7B OLMo3 model config.
        """
        config = cls.olmo2_7B(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_32B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 32B OLMo3 model config.
        """
        config = cls.olmo2_32B(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

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
                router=MoERouterConfig(top_k=4),
                shared_mlp=FeedForwardConfig(hidden_size=d_model * 2),
                lb_loss_weight=0.01,
                z_loss_weight=0.001,
            ),
        )

    @classmethod
    def small_hybrid_moe(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        d_model = kwargs.pop("d_model", 768)
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 12),
            n_heads=kwargs.pop("n_heads", 12),
            name=kwargs.pop("name", TransformerType.moe),
            block_name=kwargs.pop("block_name", TransformerBlockType.moe_hybrid_reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            feed_forward=FeedForwardConfig(hidden_size=d_model * 2, bias=False),
            feed_forward_moe=MoEConfig(
                name=MoEType.default,
                num_experts=32,
                hidden_size=int(0.5 * d_model),
                router=MoERouterConfig(top_k=4),
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
                router=MoERouterConfig(top_k=8),
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

        Note: Llama2 doesn't have a 1B. We made this up.
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
        use_flash: Optional[bool] = None,
        attn_backend: Optional[AttentionBackendName] = None,
        sliding_window: Optional[SlidingWindowAttentionConfig] = None,
        block_name: TransformerBlockType = TransformerBlockType.default,
        block_mods: Optional[
            Dict[int, Callable[[TransformerBlockConfig], TransformerBlockConfig]]
        ] = None,
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
        :param block_mods: A dictionary of block indices to functions that take the base block config and return a modified block config.
        :param dtype: The default data type to use for all parameters.
        """
        # Resolve hidden size of FFN in blocks.
        hidden_size = int(8 * d_model / 3)
        if hidden_size_multiplier is not None:
            hidden_size = int(hidden_size_multiplier * hidden_size)
        hidden_size = ensure_multiple_of(hidden_size, hidden_size_multiple_of)

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
                backend=attn_backend,
                sliding_window=sliding_window,
                dtype=dtype,
            ),
            feed_forward=feed_forward,
            feed_forward_moe=feed_forward_moe,
            layer_norm=layer_norm,
        )

        if block_mods and kwargs.get("block_overrides"):
            raise OLMoConfigurationError(
                "`block_mods` and `block_overrides` cannot be used together."
            )
        block_overrides = None
        if block_mods:
            block_overrides = {i: block_mods[i](block.copy()) for i in block_mods}
        elif kwargs.get("block_overrides"):
            block_overrides = kwargs.get("block_overrides")

        return cls(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
            dtype=dtype,
            block_overrides=block_overrides,
            **kwargs,
        )

    @classmethod
    def llama_like_moe(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        num_experts: int,
        top_k: int,
        expert_hidden_size: int,
        shared_expert_hidden_size: Optional[int] = None,
        dropless: bool = False,
        capacity_factor: Optional[float] = None,
        lb_loss_weight: float = 0.01,
        z_loss_weight: Optional[float] = 0.001,
        reordered_norm: bool = False,
        hybrid: bool = False,
        **kwargs,
    ) -> "TransformerConfig":
        block_name: TransformerBlockType
        if reordered_norm:
            block_name = (
                TransformerBlockType.moe_hybrid_reordered_norm
                if hybrid
                else TransformerBlockType.moe_reordered_norm
            )
        else:
            block_name = TransformerBlockType.moe_hybrid if hybrid else TransformerBlockType.moe
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            n_heads=n_heads,
            name=TransformerType.moe,
            block_name=block_name,
            qk_norm=kwargs.pop("qk_norm", reordered_norm),
            feed_forward_moe=MoEConfig(
                name=MoEType.default if not dropless else MoEType.dropless,
                num_experts=num_experts,
                hidden_size=expert_hidden_size,
                capacity_factor=capacity_factor,
                router=MoERouterConfig(top_k=top_k),
                shared_mlp=None
                if shared_expert_hidden_size is None
                else FeedForwardConfig(hidden_size=shared_expert_hidden_size, bias=False),
                lb_loss_weight=lb_loss_weight,
                z_loss_weight=z_loss_weight,
            ),
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
        hidden_size = ensure_multiple_of(hidden_size, hidden_size_multiple_of)

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

    def with_rope_scaling(
        self, rope_scaling: RoPEScalingConfig, full_attn_layers_only: bool = True
    ) -> "TransformerConfig":
        """
        Return a copy of this config with the given RoPE scaling scheme applied.
        """
        new_config = self.copy()
        if new_config.block.attention.rope is None:
            raise ValueError("Cannot apply RoPE scaling to a model without RoPE.")
        if new_config.block_overrides:
            raise ValueError("Cannot apply RoPE scaling when block_overrides are already set.")

        def apply_scaling(block_config: TransformerBlockConfig) -> None:
            rope_config = block_config.attention.rope
            if rope_config is None:
                raise ValueError("Cannot apply RoPE scaling to a layer without RoPE.")
            rope_config = rope_config.copy()
            rope_config.scaling = rope_scaling
            block_config.attention.rope = rope_config

        if not full_attn_layers_only:
            apply_scaling(new_config.block)
            return new_config

        # Add rope scaling only to layers that do not use sliding window attention
        # We supply "block_overrides" for the layers we want to scale.
        overrides: Dict[int, TransformerBlockConfig] = {}
        for i in range(new_config.n_layers):
            sliding_window_cfg = new_config.block.attention.sliding_window
            if sliding_window_cfg and sliding_window_cfg.should_use_swa(i, new_config.n_layers):
                continue
            block_copy = new_config.block.copy()
            apply_scaling(block_copy)
            overrides[i] = block_copy

        new_config.block_overrides = overrides or None
        return new_config
