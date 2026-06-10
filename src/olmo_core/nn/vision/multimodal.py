from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from olmo_core.config import Config
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.vision.config import VisionEncoderConfig
from olmo_core.nn.vision.connector import VisionConnectorConfig

__all__ = [
    "MultimodalLMConfig",
    "MultimodalLM",
]


@dataclass
class MultimodalLMConfig(Config):
    """
    Configuration for a multimodal (vision-language) transformer.

    Composes a language model (:class:`~olmo_core.nn.transformer.TransformerConfig`),
    a vision encoder (:class:`~olmo_core.nn.vision.VisionEncoderConfig`), and
    a vision-to-language connector
    (:class:`~olmo_core.nn.vision.VisionConnectorConfig`) into a single module
    that splices projected image features into the LM embedding stream.

    Example::

        lm_cfg = TransformerConfig.olmo2_1M(vocab_size=50000)
        vis_cfg = VisionEncoderConfig()           # CLIP ViT-L/14-336
        conn_cfg = VisionConnectorConfig.from_vision_encoder(vis_cfg, output_dim=lm_cfg.d_model)
        cfg = MultimodalLMConfig(
            lm=lm_cfg, vision=vis_cfg, connector=conn_cfg, image_patch_token_id=49152,
        )
        model = cfg.build()
    """

    lm: TransformerConfig
    """Language model configuration."""

    vision: VisionEncoderConfig
    """Vision encoder configuration."""

    connector: VisionConnectorConfig
    """Vision-to-language connector configuration."""

    image_patch_token_id: int = 0
    """
    LM vocabulary ID of the ``<im_patch>`` placeholder token. Positions in
    ``input_ids`` matching this ID receive projected image features via ``+=``
    during the forward pass. The data preprocessor is responsible for ensuring
    the number of occurrences of this token in the sequence matches the number
    of pooled image features the connector produces.
    """

    vit_layers: Tuple[int, ...] = (-1,)
    """
    Indices of the ViT hidden-state layers to extract and concatenate before
    the connector. Negative indices count from the last layer. For example,
    ``(-1,)`` uses only the final layer; a two-layer selection such as ``(-2, -9)``
    requires :attr:`connector.num_input_layers` to be ``2``.
    """

    def build(self, init_device: str = "cpu") -> "MultimodalLM":
        """
        Instantiate the multimodal model on ``init_device``.

        :param init_device: Device string (e.g. ``"cpu"``, ``"meta"``).
        :returns: A :class:`MultimodalLM`.
        """
        return MultimodalLM(self, init_device=init_device)


class MultimodalLM(nn.Module):
    """
    Vision-language model: vision encoder + connector + language model.

    Forward pass flow:

    1. Look up LM token embeddings for ``input_ids``.
    2. If images are provided, run them through the vision tower, extract the
       configured ViT layers, optionally strip the CLS / register prefix, and
       gather/pool/project via the connector to produce one feature per
       ``<im_patch>`` placeholder token.
    3. Add the projected image features back into the LM embedding sequence
       at every position where ``input_ids == image_patch_token_id``.
    4. Run the LM with the modified embeddings.

    :param cfg: Multimodal model configuration.
    :param init_device: Device on which to initialise parameters.
    """

    def __init__(self, cfg: MultimodalLMConfig, init_device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.lm = cfg.lm.build(init_device=init_device)
        self.vision = cfg.vision.build(init_device=init_device)
        self.connector = cfg.connector.build(init_device=init_device)

    def _encode_images(
        self,
        images: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode pre-patchified images into LM-space pooled features.

        :param images: Shape ``(B, n_crops, n_patches, patch_dim)``.
        :param pooled_patches_idx: Shape ``(B, n_pooled, pool_size)`` —
            indices into the flattened ``(n_crops * n_patches)`` patch axis
            for each pool group, with ``-1`` marking padded slots.
        :returns: Shape ``(B, n_pooled, lm_d_model)``.
        """
        B, T, N, _ = images.shape

        # Flatten crop dim into batch dim for the ViT.
        hidden_states: List[torch.Tensor] = self.vision(images.reshape(B * T, N, -1))

        # Select configured layers and concat along feature dim.
        selected = [hidden_states[i] for i in self.cfg.vit_layers]
        features = torch.cat(selected, dim=-1) if len(selected) > 1 else selected[0]

        # Strip prefix tokens (CLS for CLIP-style).
        num_prefix = getattr(self.vision, "num_prefix_tokens", 0)
        if num_prefix > 0:
            features = features[:, num_prefix:]

        # Reshape back to (B, T*N, dim) — the connector indexes into the
        # flat crop-patch axis.
        features = features.reshape(B, T * features.shape[1], features.shape[-1])

        return self.connector(features, pooled_patches_idx)

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        pooled_patches_idx: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run the vision-language forward pass.

        :param input_ids: Token IDs, shape ``(B, seq_len)``. Positions equal to
            :attr:`cfg.image_patch_token_id` will be overwritten (via ``+=``)
            with projected image features.
        :param images: Pre-patchified image patches, shape
            ``(B, n_crops, n_patches, patch_dim)``. Pass ``None`` for
            text-only batches.
        :param pooled_patches_idx: Per-group patch indices,
            shape ``(B, n_pooled, pool_size)``. Required when *images* is not
            ``None``. ``n_pooled`` must equal the number of
            ``<im_patch>`` tokens per sequence.
        :param labels: Target token IDs, shape ``(B, seq_len)``.
        :returns: Logits or loss (same as
            :meth:`~olmo_core.nn.transformer.Transformer.forward`).
        """
        assert (
            self.lm.embeddings is not None
        ), "MultimodalLM requires the LM to have an embedding table"

        device = self.lm.device
        input_ids = input_ids.to(device)
        if labels is not None:
            labels = labels.to(device)

        # Compute LM token embeddings with any configured scale / norm.
        h = self.lm.embeddings(input_ids)
        if self.lm.embed_scale is not None:
            h = h * self.lm.embed_scale
        if self.lm.embedding_norm is not None:
            h = self.lm.embedding_norm(h)

        if images is not None:
            if pooled_patches_idx is None:
                raise ValueError("`pooled_patches_idx` is required when `images` is provided")

            images = images.to(device)
            pooled_patches_idx = pooled_patches_idx.to(device)

            image_features = self._encode_images(images, pooled_patches_idx)  # (B, n_pooled, d)

            # Splice into LM embeddings at every <im_patch> position.
            is_image_patch = input_ids.view(-1) == self.cfg.image_patch_token_id
            n_patches_in_seq = int(is_image_patch.sum())
            n_features = image_features.shape[0] * image_features.shape[1]
            if n_patches_in_seq != n_features:
                raise ValueError(
                    f"Number of <im_patch> tokens in input_ids ({n_patches_in_seq}) does not "
                    f"match the number of projected image features ({n_features}). The data "
                    f"preprocessor must insert exactly one <im_patch> per pooled feature."
                )
            d = h.shape[-1]
            # ``.contiguous()`` guards against a non-contiguous ``h`` (e.g. from a fused
            # embedding_norm), for which ``.view()`` would raise. ``flat`` is a view of
            # the contiguous ``h``, so the in-place add below propagates back into ``h``.
            h = h.contiguous()
            flat = h.view(-1, d)
            flat[is_image_patch] = flat[is_image_patch] + image_features.reshape(-1, d)

        return self.lm(input_ids, input_embeddings=h, labels=labels, **kwargs)
