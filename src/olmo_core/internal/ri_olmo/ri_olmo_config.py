from dataclasses import dataclass

from olmo_core.nn.attention import AttentionBackendName, GateConfig, GateGranularity
from olmo_core.nn.feed_forward import ActivationFunction
from olmo_core.nn.transformer import TransformerConfig


@dataclass
class RicursiveTransformerConfig(TransformerConfig):
    @classmethod
    def ri_olmo_v1(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        The RI-Olmo v1 model series baseline.
        """
        return cls.gemma3_like(
            vocab_size=vocab_size,
            n_kv_heads=8,
            head_dim=128,
            global_layer_interval=5,
            activation=ActivationFunction.silu,
            gate=GateConfig(
                granularity=GateGranularity.elementwise,
                full_precision=True,
            ),
            attn_backend=kwargs.pop(
                "attn_backend",
                AttentionBackendName.flash_3,
            ),
            **kwargs,
        )

    #############################################################################
    #                       Ricursive-Olmo V1 Model Series                      #
    #############################################################################

    @classmethod
    def ri_olmo_v1_260M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 260M RI-Olmo model config.

        259,551,360 total params
        195,326,080 non-embedding params
        """
        return cls.ri_olmo_v1(
            d_model=640,
            hidden_size=640 * 8,
            n_layers=10,
            n_heads=8,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def ri_olmo_v1_709M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 709M RI-Olmo model config.

        708,903,680 total params
        606,143,232 non-embedding params
        """
        return cls.ri_olmo_v1(
            d_model=1024,
            hidden_size=1024 * 8,
            n_layers=15,
            n_heads=16,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def ri_olmo_v1_1p3B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1.3B RI-Olmo model config.

        1,253,157,120 total params
        1,124,706,560 non-embedding params
        """
        return cls.ri_olmo_v1(
            d_model=1280,
            hidden_size=1280 * 8,
            n_layers=20,
            n_heads=16,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def ri_olmo_v1_2B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 2.2B RI-Olmo model config.

        2,156,558,080 total params
        2,002,417,408 non-embedding params
        """
        return cls.ri_olmo_v1(
            d_model=1536,
            hidden_size=1536 * 8,
            n_layers=25,
            n_heads=24,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def ri_olmo_v1_4B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 4.3B RI-Olmo model config.

        4,312,000,000 total params
        4,106,479,104 non-embedding params
        """
        return cls.ri_olmo_v1(
            d_model=2048,
            hidden_size=2048 * 8,
            n_layers=30,
            n_heads=32,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def ri_olmo_v1_8B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        An 8B RI-Olmo model config.

        8,588,259,840 total params
        8,331,358,720 non-embedding params
        """
        return cls.ri_olmo_v1(
            d_model=2560,
            hidden_size=2560 * 8,
            n_layers=40,
            n_heads=40,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def ri_olmo_v1_15B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 15B RI-Olmo model config.

        15,087,541,760 total params
        14,779,260,416 non-embedding params
        """
        return cls.ri_olmo_v1(
            d_model=3072,
            hidden_size=3072 * 8,
            n_layers=50,
            n_heads=48,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def ri_olmo_v1_34B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 34B RI-Olmo model config.

        34,084,000,000 total params
        33,672,958,208 non-embedding params
        """
        return cls.ri_olmo_v1(
            d_model=4096,
            hidden_size=4096 * 8,
            n_layers=65,
            n_heads=64,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def ri_olmo_v1_65B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 65B RI-Olmo model config.

        64,782,689,280 total params
        64,268,887,040 non-embedding params
        """
        return cls.ri_olmo_v1(
            d_model=5120,
            hidden_size=5120 * 8,
            n_layers=80,
            n_heads=80,
            vocab_size=vocab_size,
            **kwargs,
        )
