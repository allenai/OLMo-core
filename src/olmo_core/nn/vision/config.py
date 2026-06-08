from dataclasses import dataclass, field
from typing import Tuple

import torch.nn as nn
from typing_extensions import Self

from olmo_core.config import DType, StrEnum
from olmo_core.nn.config import ModuleConfig

__all__ = [
    "VisionEncoderType",
    "VisionBlockType",
    "VisionBlockConfig",
    "VisionEncoderConfig",
]


class VisionEncoderType(StrEnum):
    """
    An enumeration of supported vision encoder architectures.
    """

    openai = "openai"
    """
    OpenAI CLIP ViT-style encoder (with CLS token).
    Default config matches ViT-L/14-336.
    ➡️ :class:`~olmo_core.nn.vision.VisionTransformer`
    """

    siglip = "siglip"
    """
    SigLIP-style encoder (no CLS token).
    Default config matches SigLIP-SO400M-14-378.
    ➡️ :class:`~olmo_core.nn.vision.VisionTransformer`
    """

    siglip2 = "siglip2"
    """
    SigLIP2-style encoder (no CLS token; same architecture as SigLIP, improved training).
    ➡️ :class:`~olmo_core.nn.vision.VisionTransformer`
    """


class VisionBlockType(StrEnum):
    """
    An enumeration of the different vision transformer block implementations.
    """

    standard = "standard"
    """
    Standard pre-LN ViT block (multi-head attention + MLP).
    ➡️ :class:`~olmo_core.nn.vision.ViTBlock`
    """


@dataclass
class VisionBlockConfig(ModuleConfig):
    """
    Configuration for a single vision transformer block.

    Mirrors the language-model :class:`~olmo_core.nn.transformer.TransformerBlockConfig`
    pattern: the block implementation is selected via :attr:`name` and :meth:`build`
    dispatches to the matching public block class. Today only
    :attr:`VisionBlockType.standard` is available; additional block variants can be
    registered here as they are added.

    The block reads its layer dimensions from the parent :class:`VisionEncoderConfig`
    passed to :meth:`build`.
    """

    name: VisionBlockType = VisionBlockType.standard
    """The vision block implementation to use."""

    def build(self, cfg: "VisionEncoderConfig", init_device: str = "cpu") -> nn.Module:
        """
        Instantiate the block.

        :param cfg: The parent vision encoder configuration supplying layer dimensions.
        :param init_device: Device on which to initialise parameters.
        :returns: A :class:`~olmo_core.nn.vision.ViTBlock` instance.
        """
        from .image_vit import ViTBlock

        if self.name == VisionBlockType.standard:
            return ViTBlock(cfg, init_device=init_device)
        else:
            raise NotImplementedError(self.name)


@dataclass
class VisionEncoderConfig(ModuleConfig):
    """
    Configuration for a vision transformer encoder.

    Default field values correspond to OpenAI CLIP ViT-L/14-336.
    Use the class-method factories for other standard checkpoints:

    ==========================================  =======================
    Factory                                     Checkpoint
    ==========================================  =======================
    :meth:`VisionEncoderConfig`                CLIP ViT-L/14-336
    :meth:`siglip_b16_224`                      SigLIP B/16-224
    :meth:`siglip_l16_384`                      SigLIP L/16-384
    :meth:`siglip_so400m_patch14_224`           SigLIP SO400M/14-224
    :meth:`siglip_so400m`                       SigLIP SO400M/14-378
    :meth:`siglip2_b16_256`                     SigLIP2 B/16-256
    :meth:`siglip2_l16_256`                     SigLIP2 L/16-256
    :meth:`siglip2_so400m_patch14_378`          SigLIP2 SO400M/14-378
    :meth:`siglip2_so400m_patch16_256`          SigLIP2 SO400M/16-256
    ==========================================  =======================

    Example usage::

        cfg = VisionEncoderConfig()           # CLIP ViT-L/14-336
        cfg = VisionEncoderConfig.siglip2_so400m_patch14_378()
        vit = cfg.build(init_device="cpu")
        # images: (B, n_patches, patch_size**2 * 3)  -- pre-patchified
        # outputs: list of per-layer hidden states
    """

    name: VisionEncoderType = VisionEncoderType.openai
    """The vision encoder architecture."""

    image_default_input_size: Tuple[int, int] = (336, 336)
    """Default (height, width) of input images in pixels."""

    image_patch_size: int = 14
    """Patch size in pixels (square patches assumed)."""

    image_emb_dim: int = 1024
    """Hidden dimension of the vision transformer."""

    image_num_heads: int = 16
    """Number of attention heads."""

    image_num_key_value_heads: int = 16
    """Number of key/value heads. Equal to ``image_num_heads`` for MHA; less for GQA."""

    image_num_layers: int = 23
    """
    Number of transformer blocks to instantiate. The full HF CLIP ViT-L/14 tower has
    24 layers; the default of 23 keeps only the layers needed when features are read
    from the second-to-last layer (index ``-2``), dropping the unused final block.
    """

    image_head_dim: int = 64
    """Per-head dimension. Must satisfy ``image_emb_dim == image_num_heads * image_head_dim``."""

    image_mlp_dim: int = 4096
    """Hidden size of the per-block feed-forward network."""

    image_mlp_activations: str = "quick_gelu"
    """
    Activation name for the ViT FFN. One of ``quick_gelu`` (CLIP),
    ``gelu_pytorch_tanh`` (SigLIP), or ``gelu``.
    """

    image_num_pos: int = 577
    """Number of positional embedding slots. For CLIP ViT-L/14-336: 24*24 patches + 1 CLS = 577."""

    image_norm_eps: float = 1e-5
    """Epsilon for layer normalisation inside the vision transformer."""

    attention_dropout: float = 0.0
    """Dropout probability for attention weights."""

    residual_dropout: float = 0.0
    """Dropout probability applied after the attention output projection."""

    use_cls_token: bool = True
    """
    Whether to prepend a learnable CLS token (CLIP-style). SigLIP-style encoders set
    this to ``False``.
    """

    patch_embedding_bias: bool = False
    """
    Whether the patch-embedding projection has a bias term. CLIP omits it; SigLIP
    includes it.
    """

    use_pre_ln: bool = True
    """
    Whether a LayerNorm is applied to the embeddings before the transformer blocks
    (CLIP-style). SigLIP-style encoders set this to ``False``.
    """

    block: VisionBlockConfig = field(default_factory=VisionBlockConfig)
    """
    Configuration selecting the transformer block implementation. Defaults to the
    standard pre-LN ViT block; see :class:`VisionBlockConfig`.
    """

    initializer_range: float = 0.02
    """Std dev for normal-distribution weight initialisation."""

    dtype: DType = DType.float32
    """Default parameter dtype."""

    @property
    def image_num_patch(self) -> Tuple[int, int]:
        """Number of patches along (height, width) for the default input size."""
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size

    # ------------------------------------------------------------------
    # SigLIP factory methods
    # ------------------------------------------------------------------

    @classmethod
    def siglip_b16_224(cls) -> Self:
        """
        Returns a :class:`VisionEncoderConfig` matching SigLIP ViT-B/16-224
        (``google/siglip-base-patch16-224``).
        """
        return cls(
            name=VisionEncoderType.siglip,
            use_cls_token=False,
            patch_embedding_bias=True,
            use_pre_ln=False,
            image_default_input_size=(224, 224),
            image_patch_size=16,
            image_emb_dim=768,
            image_num_heads=12,
            image_num_key_value_heads=12,
            image_num_layers=12,
            image_head_dim=64,
            image_mlp_dim=3072,
            image_mlp_activations="gelu_pytorch_tanh",
            image_num_pos=196,  # 14*14 = 196, no CLS
            image_norm_eps=1e-6,
        )

    @classmethod
    def siglip_l16_384(cls) -> Self:
        """
        Returns a :class:`VisionEncoderConfig` matching SigLIP ViT-L/16-384
        (``google/siglip-large-patch16-384``).
        """
        return cls(
            name=VisionEncoderType.siglip,
            use_cls_token=False,
            patch_embedding_bias=True,
            use_pre_ln=False,
            image_default_input_size=(384, 384),
            image_patch_size=16,
            image_emb_dim=1024,
            image_num_heads=16,
            image_num_key_value_heads=16,
            image_num_layers=24,
            image_head_dim=64,
            image_mlp_dim=4096,
            image_mlp_activations="gelu_pytorch_tanh",
            image_num_pos=576,  # 24*24 = 576, no CLS
            image_norm_eps=1e-6,
        )

    @classmethod
    def siglip_so400m_patch14_224(cls) -> Self:
        """
        Returns a :class:`VisionEncoderConfig` matching SigLIP SO400M/14-224
        (``google/siglip-so400m-patch14-224``).
        """
        return cls(
            name=VisionEncoderType.siglip,
            use_cls_token=False,
            patch_embedding_bias=True,
            use_pre_ln=False,
            image_default_input_size=(224, 224),
            image_patch_size=14,
            image_emb_dim=1152,
            image_num_heads=16,
            image_num_key_value_heads=16,
            image_num_layers=27,
            image_head_dim=72,
            image_mlp_dim=4304,
            image_mlp_activations="gelu_pytorch_tanh",
            image_num_pos=256,  # 16*16 = 256, no CLS
            image_norm_eps=1e-6,
        )

    @classmethod
    def siglip_so400m(cls) -> Self:
        """
        Returns a :class:`VisionEncoderConfig` matching SigLIP SO400M/14-378
        (``google/siglip-so400m-patch14-378``).
        """
        return cls(
            name=VisionEncoderType.siglip,
            use_cls_token=False,
            patch_embedding_bias=True,
            use_pre_ln=False,
            image_default_input_size=(378, 378),
            image_patch_size=14,
            image_emb_dim=1152,
            image_num_heads=16,
            image_num_key_value_heads=16,
            image_num_layers=27,
            image_head_dim=72,
            image_mlp_dim=4304,
            image_mlp_activations="gelu_pytorch_tanh",
            image_num_pos=729,  # 27*27 = 729, no CLS
            image_norm_eps=1e-6,
        )

    # ------------------------------------------------------------------
    # SigLIP2 factory methods
    # ------------------------------------------------------------------

    @classmethod
    def siglip2_b16_256(cls) -> Self:
        """
        Returns a :class:`VisionEncoderConfig` matching SigLIP2 ViT-B/16-256
        (``google/siglip2-base-patch16-256``).
        """
        return cls(
            name=VisionEncoderType.siglip2,
            use_cls_token=False,
            patch_embedding_bias=True,
            use_pre_ln=False,
            image_default_input_size=(256, 256),
            image_patch_size=16,
            image_emb_dim=768,
            image_num_heads=12,
            image_num_key_value_heads=12,
            image_num_layers=12,
            image_head_dim=64,
            image_mlp_dim=3072,
            image_mlp_activations="gelu_pytorch_tanh",
            image_num_pos=256,  # 16*16 = 256, no CLS
            image_norm_eps=1e-6,
        )

    @classmethod
    def siglip2_l16_256(cls) -> Self:
        """
        Returns a :class:`VisionEncoderConfig` matching SigLIP2 ViT-L/16-256
        (``google/siglip2-large-patch16-256``).
        """
        return cls(
            name=VisionEncoderType.siglip2,
            use_cls_token=False,
            patch_embedding_bias=True,
            use_pre_ln=False,
            image_default_input_size=(256, 256),
            image_patch_size=16,
            image_emb_dim=1024,
            image_num_heads=16,
            image_num_key_value_heads=16,
            image_num_layers=24,
            image_head_dim=64,
            image_mlp_dim=4096,
            image_mlp_activations="gelu_pytorch_tanh",
            image_num_pos=256,  # 16*16 = 256, no CLS
            image_norm_eps=1e-6,
        )

    @classmethod
    def siglip2_so400m_patch14_378(cls) -> Self:
        """
        Returns a :class:`VisionEncoderConfig` matching SigLIP2 SO400M/14-378
        (``google/siglip2-so400m-patch14-378``).
        """
        return cls(
            name=VisionEncoderType.siglip2,
            use_cls_token=False,
            patch_embedding_bias=True,
            use_pre_ln=False,
            image_default_input_size=(378, 378),
            image_patch_size=14,
            image_emb_dim=1152,
            image_num_heads=16,
            image_num_key_value_heads=16,
            image_num_layers=27,
            image_head_dim=72,
            image_mlp_dim=4304,
            image_mlp_activations="gelu_pytorch_tanh",
            image_num_pos=729,  # 27*27 = 729, no CLS
            image_norm_eps=1e-6,
        )

    @classmethod
    def siglip2_so400m_patch16_256(cls) -> Self:
        """
        Returns a :class:`VisionEncoderConfig` matching SigLIP2 SO400M/16-256
        (``google/siglip2-so400m-patch16-256``).
        """
        return cls(
            name=VisionEncoderType.siglip2,
            use_cls_token=False,
            patch_embedding_bias=True,
            use_pre_ln=False,
            image_default_input_size=(256, 256),
            image_patch_size=16,
            image_emb_dim=1152,
            image_num_heads=16,
            image_num_key_value_heads=16,
            image_num_layers=27,
            image_head_dim=72,
            image_mlp_dim=4304,
            image_mlp_activations="gelu_pytorch_tanh",
            image_num_pos=256,  # 16*16 = 256, no CLS
            image_norm_eps=1e-6,
        )

    def build(self, init_device: str = "cpu") -> nn.Module:
        """
        Instantiate the vision encoder on ``init_device``.

        :param init_device: Device string (e.g. ``"cpu"``, ``"meta"``).
        :returns: A :class:`~olmo_core.nn.vision.VisionTransformer` instance configured
            for the selected encoder variant.
        """
        from .image_vit import VisionTransformer

        return VisionTransformer(self, init_device=init_device)
