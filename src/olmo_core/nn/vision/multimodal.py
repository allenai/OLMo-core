from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

from ...config import Config
from ..lm_head import LMOutputWithLoss
from ..transformer.config import (
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
from .config import VisionBackboneConfig
from .connector import VisionConnectorConfig

__all__ = [
    "MultimodalTransformerConfig",
    "MultimodalTransformer",
]


@dataclass
class MultimodalTransformerConfig(Config):
    """
    Configuration for a multimodal (vision-language) transformer.

    Composes a language model (:class:`~olmo_core.nn.transformer.TransformerConfig`),
    a vision encoder (:class:`~olmo_core.nn.vision.VisionBackboneConfig`), and
    a vision-to-language connector
    (:class:`~olmo_core.nn.vision.VisionConnectorConfig`) into a single module
    matching Molmo's image-feature splice mechanics.

    Example::

        lm_cfg = TransformerConfig.olmo2_1M(vocab_size=50000)
        vis_cfg = VisionBackboneConfig()           # CLIP ViT-L/14-336
        conn_cfg = VisionConnectorConfig.from_vision_backbone(vis_cfg, output_dim=lm_cfg.d_model)
        cfg = MultimodalTransformerConfig(
            lm=lm_cfg, vision=vis_cfg, connector=conn_cfg, image_patch_token_id=49152,
        )
        model = cfg.build()
    """

    lm: TransformerConfig
    """Language model configuration."""

    vision: VisionBackboneConfig
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
    ``(-1,)`` uses only the final layer; ``(-2, -9)`` matches Molmo's two-layer
    extraction and requires :attr:`connector.num_input_layers` to be ``2``.
    """

    def build(self, init_device: str = "cpu") -> "MultimodalTransformer":
        """
        Instantiate the multimodal model on ``init_device``.

        :param init_device: Device string (e.g. ``"cpu"``, ``"meta"``).
        :returns: A :class:`MultimodalTransformer`.
        """
        return MultimodalTransformer(self, init_device=init_device)


class MultimodalTransformer(nn.Module):
    """
    Vision-language model: vision encoder + connector + language model.

    Forward pass flow (matching Molmo's reference implementation):

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

    def __init__(self, cfg: MultimodalTransformerConfig, init_device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.lm = cfg.lm.build(init_device=init_device)
        self.vision = cfg.vision.build(init_device=init_device)
        self.connector = cfg.connector.build(init_device=init_device)

    # ------------------------------------------------------------------
    # TrainModule interface (delegated to the wrapped LM)
    # ------------------------------------------------------------------

    def post_batch(self, dry_run: bool = False) -> None:
        """Hook called by the train module after each batch's backward pass."""
        self.lm.post_batch(dry_run=dry_run)

    def post_optim_step(self) -> None:
        """Hook called by the train module after each optimizer step."""
        self.lm.post_optim_step()

    def reset_auxiliary_metrics(self) -> None:
        """Reset the LM's auxiliary metrics (MoE load-balancing etc.)."""
        self.lm.reset_auxiliary_metrics()

    def compute_auxiliary_metrics(self, reset: bool = True):
        """Return the LM's auxiliary metrics; vision/connector contribute none."""
        return self.lm.compute_auxiliary_metrics(reset=reset)

    def num_flops_per_token(self, seq_len: int) -> int:
        """Approximate FLOPs/token. Counts the LM only — vision FLOPs depend on
        ``n_crops × n_patches`` which the LM seq_len doesn't capture."""
        return self.lm.num_flops_per_token(seq_len)

    @property
    def num_params(self) -> int:
        """Total parameter count across LM, vision, and connector."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_non_embedding_params(self) -> int:
        """Total params excluding the LM token embedding table.

        Used by speed-monitor callbacks. Vision patch-embedding and pooling
        attention are kept (they're not "vocabulary embeddings")."""
        return self.num_params - self.lm.embeddings.weight.numel()

    # ------------------------------------------------------------------
    # Distributed: materialize weights + apply FSDP/DDP
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_weights(
        self,
        *,
        max_seq_len: Optional[int] = None,
        max_local_microbatch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        world_mesh: Optional[DeviceMesh] = None,
        model_part_idx: int = 0,
    ) -> torch.Generator:
        """Materialise parameters on ``device`` and initialise them.

        Matches :meth:`~olmo_core.nn.transformer.Transformer.init_weights`'s
        signature so the same trainer-level call site works for both. Each
        sub-module (LM, vision, connector) is materialised separately to
        avoid double-``to_empty`` issues under FSDP — each ``to_empty`` on a
        FSDP-wrapped param is a collective, and overlapping/redundant calls
        across ranks deadlock.
        """
        target_device = device or next(iter(self.parameters())).device

        # LM handles its own materialisation + InitMethod + RoPE cache.
        gen = self.lm.init_weights(
            max_seq_len=max_seq_len,
            max_local_microbatch_size=max_local_microbatch_size,
            device=target_device,
            world_mesh=world_mesh,
            model_part_idx=model_part_idx,
        )

        # Materialise vision and connector separately, then init their params.
        self.vision.to_empty(device=target_device)
        self.vision.reset_parameters()
        self.connector.to_empty(device=target_device)
        self.connector.reset_parameters()

        return gen

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
    ) -> None:
        """Apply FSDP2 (``fully_shard``) to vision, connector, LM, and the
        composite model.

        The LM is wrapped via its own :meth:`Transformer.apply_fsdp`, which
        handles block-level sharding, the embedding table, and the LM head.
        Vision blocks are individually sharded; the connector is sharded as
        a single unit; the whole :class:`MultimodalTransformer` gets a final
        outer ``fully_shard`` so cross-submodule unsharding remains cheap.

        :param dp_mesh: The data-parallel device mesh.
        :param param_dtype: Mixed-precision parameter dtype.
        :param reduce_dtype: Gradient reduction dtype.
        :param pp_enabled: Whether pipeline parallelism is also enabled.
            Currently unsupported for multimodal models; passed through to
            the LM only.
        :param prefetch_factor: Forwarded to LM block prefetching.
        :param wrapping_strategy: Forwarded to LM FSDP wrapping.
        """
        # 1. Delegate LM wrapping to Transformer.apply_fsdp.
        self.lm.apply_fsdp(
            dp_mesh=dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            pp_enabled=pp_enabled,
            prefetch_factor=prefetch_factor,
            wrapping_strategy=wrapping_strategy,
        )

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype or self.lm.dtype, reduce_dtype=reduce_dtype
        )
        fsdp_kwargs = dict(mesh=dp_mesh, mp_policy=mp_policy)
        reshard_after_forward = not pp_enabled

        # 2. Each vision block gets its own FSDP unit.
        for block in self.vision.blocks:
            fully_shard(block, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)

        # 3. Connector is small enough to wrap as a single unit.
        fully_shard(self.connector, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)

        # 4. Top-level wrap so the composite all-gather happens in one shot.
        fully_shard(self, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)

        # Match Transformer's behaviour: don't unshard the (large) text
        # embedding table during backward, since it isn't needed there.
        if isinstance(self.lm.embeddings, FSDPModule):
            self.lm.embeddings.set_unshard_in_backward(False)

    def apply_ddp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Apply DDP to the composite model.

        Cheap to implement because DDP doesn't need per-submodule wrapping —
        we just replicate the whole :class:`MultimodalTransformer`.
        """
        from torch.distributed._composable.replicate import replicate

        if param_dtype is not None and param_dtype != self.lm.dtype:
            self.to(dtype=param_dtype)
        replicate(self, device_mesh=dp_mesh, bucket_cap_mb=100)

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
        ), "MultimodalTransformer requires the LM to have an embedding table"

        # The base Transformer's _prepare_inputs would move input_ids to device,
        # but we lookup embeddings before delegating to it. Move proactively.
        emb_device = self.lm.embeddings.weight.device
        if input_ids.device != emb_device:
            input_ids = input_ids.to(emb_device)
        if images is not None and images.device != emb_device:
            images = images.to(emb_device)
        if pooled_patches_idx is not None and pooled_patches_idx.device != emb_device:
            pooled_patches_idx = pooled_patches_idx.to(emb_device)

        # Compute LM token embeddings with any configured scale / norm.
        h = self.lm.embeddings(input_ids)
        if self.lm.embed_scale is not None:
            h = h * self.lm.embed_scale
        if self.lm.embedding_norm is not None:
            h = self.lm.embedding_norm(h)

        if images is not None:
            if pooled_patches_idx is None:
                raise ValueError("`pooled_patches_idx` is required when `images` is provided")

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
            h.view(-1, d)[is_image_patch] = h.view(-1, d)[is_image_patch] + image_features.reshape(
                -1, d
            )

        return self.lm(input_ids, input_embeddings=h, labels=labels, **kwargs)
