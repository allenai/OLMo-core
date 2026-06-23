from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

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

    # -- model introspection (mirrors the Transformer API used by the trainer / callbacks) --

    @property
    def num_params(self) -> int:
        """Total number of parameters (LM + vision + connector)."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_non_embedding_params(self) -> int:
        """All parameters excluding the LM token-embedding table (the vision encoder
        and connector have no token embeddings, so they count in full)."""
        vision_connector = sum(
            p.numel() for m in (self.vision, self.connector) for p in m.parameters()
        )
        return self.lm.num_non_embedding_params + vision_connector

    @property
    def is_moe(self) -> bool:
        return self.lm.is_moe

    def num_flops_per_token(self, seq_len: int) -> int:
        """Idealized FLOPs/token for the language model only. The vision encoder /
        connector run per-image (not per-token), so their FLOPs are reported separately
        via :meth:`image_encoder_flops` and added to the throughput/MFU accounting."""
        return self.lm.num_flops_per_token(seq_len)

    @property
    def _n_vision_params(self) -> int:
        if not hasattr(self, "_n_vision_params_cache"):
            self._n_vision_params_cache = sum(p.numel() for p in self.vision.parameters())
        return self._n_vision_params_cache

    @property
    def _n_connector_params(self) -> int:
        if not hasattr(self, "_n_connector_params_cache"):
            self._n_connector_params_cache = sum(p.numel() for p in self.connector.parameters())
        return self._n_connector_params_cache

    def image_encoder_flops(
        self, n_crops: int, n_patches_per_crop: int, n_pooled_tokens: int
    ) -> int:
        """Idealized FLOPs for the vision half of one batch, for MFU accounting.

        The ViT processes every (padded) crop in the batch, so ``n_crops`` should be the
        full ``B * n_crops`` of the images tensor. The encoder is **frozen** → forward-only
        (2 FLOPs/param/patch for the linear layers, plus the attention score+context
        quadratic ``4·L·P·d`` per patch). The connector is **trained** → 6 FLOPs/param
        (fwd+bwd) per pooled output token.

        :param n_crops: total number of image crops processed by the ViT this batch.
        :param n_patches_per_crop: patches per crop fed to the ViT (``P``).
        :param n_pooled_tokens: number of pooled ``<im_patch>`` tokens produced.
        """
        d = self.cfg.vision.image_emb_dim
        n_layers = self.cfg.vision.image_num_layers
        n_raw = n_crops * n_patches_per_crop
        vit = n_raw * (2 * self._n_vision_params + 4 * n_layers * n_patches_per_crop * d)
        connector = n_pooled_tokens * 6 * self._n_connector_params
        return int(vit + connector)

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
        token_type_ids: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        example_ids: Optional[torch.Tensor] = None,
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
        :param token_type_ids: Optional ``(B, seq_len)`` tensor marking image tokens
            (non-zero) vs. text tokens (zero). When provided, image tokens attend to
            each other **bidirectionally** (causal order is ignored among image
            tokens) while text stays causal, matching HF Molmo2's attention mask.
            Requires the dense (``"torch"``) attention backend.
        :param subsegment_ids: Optional ``(B, seq_len)`` int tensor marking subsegments
            for packed multi-annotation data: a shared prefix (``ATTEND_ALL`` id) that
            branches into several mutually-isolated response branches (one id each). A
            query may only attend to keys with a ``>=`` subsegment id (matching mm_olmo's
            ``attention_mask & (subseg_q <= subseg_k)``). Requires ``position_ids`` and
            the dense (``"torch"``) attention backend.
        :param position_ids: Optional ``(B, seq_len)`` int tensor of explicit RoPE
            positions. Required with ``subsegment_ids`` (parallel branches share an
            overlapping position range and cannot use sequential positions).
        :returns: Logits or loss (same as
            :meth:`~olmo_core.nn.transformer.Transformer.forward`).
        """
        if subsegment_ids is not None and position_ids is None:
            raise ValueError("`position_ids` is required when `subsegment_ids` is provided")
        assert (
            self.lm.embeddings is not None
        ), "MultimodalLM requires the LM to have an embedding table"

        device = self.lm.device
        input_ids = input_ids.to(device)
        if labels is not None:
            labels = labels.to(device)

        # Compute LM token embeddings with any configured scale / norm. We embed here
        # (rather than inside ``self.lm``) so image features can be spliced in below.
        # Under FSDP the embedding weight is a sharded ``DTensor`` that only the LM's own
        # forward would unshard, so gather it to a full tensor for the lookup (a no-op for
        # DDP / single-GPU where the weight is already a plain tensor).
        emb = self.lm.embeddings
        emb_weight = emb.weight
        if isinstance(emb_weight, DTensor):
            emb_weight = emb_weight.full_tensor()
        h = F.embedding(input_ids, emb_weight, padding_idx=emb.padding_idx)
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

            # Tie the connector output into the autograd graph on *every* forward that ran
            # the vision path, even when no rows are spliced below (e.g. an all-text
            # microbatch handed a dummy zero crop by the collator). This adds exactly 0 to
            # the activations but keeps the connector's FSDP reduce-scatter — and the vision
            # all-gather — firing on every rank each step, so collectives stay in lockstep
            # across ranks regardless of how text-only vs image examples are distributed.
            h = h + 0.0 * image_features.sum().to(h.dtype)

            # Keep only valid pooled rows (a row is padding iff *all* its patch
            # indices are -1, e.g. added by a batch collator to equalize ``n_pooled``
            # across examples). Selecting in row-major order keeps each example's
            # features aligned with its ``<im_patch>`` positions, so batches with a
            # variable number of image tokens per example work. For unpadded / B=1
            # inputs every row is valid and this is a no-op.
            valid_rows = (pooled_patches_idx >= 0).any(dim=-1)  # (B, n_pooled)
            image_features = image_features[valid_rows]  # (total_valid, d)

            # Splice into LM embeddings at every <im_patch> position.
            is_image_patch = input_ids.view(-1) == self.cfg.image_patch_token_id
            n_patches_in_seq = int(is_image_patch.sum())
            n_features = image_features.shape[0]
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

        # Build a bidirectional allow-mask among image tokens (causal elsewhere),
        # matching HF Molmo2's `token_type_ids`-based mask. ``or_mask[b, 1, q, k]``
        # is True where both q and k are image tokens, so they attend regardless of
        # causal order; it is OR'd onto the causal base inside each attention block.
        or_mask: Optional[torch.Tensor] = None
        if token_type_ids is not None:
            is_image = token_type_ids.to(device) != 0  # (B, S)
            or_mask = (is_image[:, :, None] & is_image[:, None, :]).unsqueeze(1)  # (B, 1, S, S)

        # Build a restrictive subsegment (branch-isolation) allow-mask: a query at
        # position q may attend a key at position k only if ``subseg[q] <= subseg[k]``,
        # matching mm_olmo's ``attention_mask & subsegment_mask``. The shared prefix uses
        # the largest id so it only attends itself, while each branch attends the prefix
        # and itself but not sibling branches.
        # ``example_ids`` (sequence packing) adds a block-diagonal keep-mask so a token never
        # attends across a packed-example boundary: AND ``example_ids[q] == example_ids[k]``
        # onto the subsegment rule. For a single (unpacked) example every id is equal, so it
        # is a no-op. The OR'd image mask is scoped too, since the whole expression is
        # ``(causal | or_mask) & and_mask``.
        and_mask: Optional[torch.Tensor] = None
        seg_rule: Optional[torch.Tensor] = None
        if subsegment_ids is not None:
            seg = subsegment_ids.to(device)
            seg_rule = seg[:, :, None] <= seg[:, None, :]  # (B, S, S)
        if example_ids is not None:
            eid = example_ids.to(device)
            same_example = eid[:, :, None] == eid[:, None, :]  # (B, S, S)
            combined = same_example & seg_rule if seg_rule is not None else same_example
            and_mask = combined.unsqueeze(1)  # (B, 1, S, S)
        elif seg_rule is not None:
            and_mask = seg_rule.unsqueeze(1)

        if position_ids is not None:
            position_ids = position_ids.to(device)

        return self.lm(
            input_ids,
            input_embeddings=h,
            labels=labels,
            or_mask=or_mask,
            and_mask=and_mask,
            position_ids=position_ids,
            **kwargs,
        )
