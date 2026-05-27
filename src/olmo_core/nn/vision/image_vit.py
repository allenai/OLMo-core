import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation

from .config import VisionBackboneConfig

__all__ = [
    "VisionTransformer",
    "SiglipVisionTransformer",
]


class _ViTAttention(nn.Module):
    """Multi-head dot-product attention for a vision transformer block."""

    def __init__(self, cfg: VisionBackboneConfig, init_device: str = "cpu"):
        super().__init__()
        self.embed_dim = cfg.image_emb_dim
        self.num_heads = cfg.image_num_heads
        self.head_dim = cfg.image_head_dim
        self.num_kv_heads = cfg.image_num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.initializer_range = cfg.initializer_range
        dtype = cfg.dtype.as_pt()

        self.wq = nn.Linear(
            self.embed_dim,
            self.num_heads * self.head_dim,
            bias=True,
            device=init_device,
            dtype=dtype,
        )
        self.wk = nn.Linear(
            self.embed_dim,
            self.num_kv_heads * self.head_dim,
            bias=True,
            device=init_device,
            dtype=dtype,
        )
        self.wv = nn.Linear(
            self.embed_dim,
            self.num_kv_heads * self.head_dim,
            bias=True,
            device=init_device,
            dtype=dtype,
        )
        self.wo = nn.Linear(
            self.num_heads * self.head_dim,
            self.embed_dim,
            bias=True,
            device=init_device,
            dtype=dtype,
        )

        self.attn_drop = nn.Dropout(cfg.attention_dropout) if cfg.attention_dropout > 0 else None
        self.resid_drop = nn.Dropout(cfg.residual_dropout)

    def reset_parameters(self):
        for w in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(w.weight, std=self.initializer_range)
            nn.init.zeros_(w.bias)

    def forward(self, x: torch.Tensor, kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape
        src = x if kv is None else kv

        q = self.wq(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.wk(src).reshape(B, src.shape[1], self.num_kv_heads, self.head_dim)
        v = self.wv(src).reshape(B, src.shape[1], self.num_kv_heads, self.head_dim)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            is_causal=False,
            dropout_p=self.attn_drop.p if self.attn_drop is not None and self.training else 0.0,
        ).transpose(1, 2)

        out = out.reshape(B, N, self.embed_dim)
        out = self.wo(out)
        out = self.resid_drop(out)
        return out


class _ViTMLP(nn.Module):
    """Two-layer feed-forward network used inside each ViT block."""

    def __init__(self, cfg: VisionBackboneConfig, init_device: str = "cpu"):
        super().__init__()
        dtype = cfg.dtype.as_pt()
        self.w1 = nn.Linear(
            cfg.image_emb_dim, cfg.image_mlp_dim, bias=True, device=init_device, dtype=dtype
        )
        self.w2 = nn.Linear(
            cfg.image_mlp_dim, cfg.image_emb_dim, bias=True, device=init_device, dtype=dtype
        )
        self.act = get_activation(cfg.image_mlp_activations)
        self._emb_dim = cfg.image_emb_dim
        self._mlp_dim = cfg.image_mlp_dim

    def reset_parameters(self):
        nn.init.trunc_normal_(self.w1.weight, std=math.sqrt(1 / self._emb_dim), a=-2.0, b=2.0)
        nn.init.trunc_normal_(self.w2.weight, std=math.sqrt(1 / self._mlp_dim), a=-2.0, b=2.0)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class _ViTBlock(nn.Module):
    """Standard pre-LN ViT residual block (attention + MLP)."""

    def __init__(self, cfg: VisionBackboneConfig, init_device: str = "cpu"):
        super().__init__()
        dtype = cfg.dtype.as_pt()
        self.attn_norm = nn.LayerNorm(
            cfg.image_emb_dim, eps=cfg.image_norm_eps, device=init_device, dtype=dtype
        )
        self.ffn_norm = nn.LayerNorm(
            cfg.image_emb_dim, eps=cfg.image_norm_eps, device=init_device, dtype=dtype
        )
        self.attn = _ViTAttention(cfg, init_device=init_device)
        self.ffn = _ViTMLP(cfg, init_device=init_device)

    def reset_parameters(self):
        self.attn_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.attn.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


def _interpolate_pos_emb(
    pos_emb: torch.Tensor,  # (num_pos, emb_dim)
    target_patch_num: Tuple[int, int],
    num_prefix: int = 1,
) -> torch.Tensor:
    """Bicubic-interpolate positional embeddings to a different spatial resolution."""
    cls_emb = pos_emb[:num_prefix]  # (num_prefix, emb_dim)
    patch_emb = pos_emb[num_prefix:]  # (H0*W0, emb_dim)

    h0 = w0 = int(math.sqrt(patch_emb.shape[0]))
    h1, w1 = target_patch_num

    if h0 == h1 and w0 == w1:
        return pos_emb

    patch_emb = patch_emb.reshape(1, h0, w0, -1).permute(0, 3, 1, 2)  # (1, D, H0, W0)
    patch_emb = F.interpolate(
        patch_emb, size=(h1, w1), mode="bicubic", align_corners=False, antialias=True
    )
    patch_emb = patch_emb.permute(0, 2, 3, 1).reshape(h1 * w1, -1)  # (H1*W1, D)

    return torch.cat([cls_emb, patch_emb], dim=0)


class VisionTransformer(nn.Module):
    """
    OpenAI CLIP-style vision transformer with a prepended CLS token.

    Matches ``VisionTransformer`` in the Molmo reference implementation and is
    suitable for weights from OpenAI CLIP ViT-L/14-336.

    The forward pass accepts pre-patchified images and returns the hidden states
    from every block, so callers can select specific layers (e.g. ``[-2, -9]``).

    :param cfg: Vision backbone configuration.
    :param init_device: Device on which to initialise parameters.
    """

    def __init__(self, cfg: VisionBackboneConfig, init_device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        dtype = cfg.dtype.as_pt()

        # Patch embedding: flattened patch pixels → emb_dim.
        patch_pixels = cfg.image_patch_size * cfg.image_patch_size * 3
        self.patch_embedding = nn.Linear(
            patch_pixels, cfg.image_emb_dim, bias=False, device=init_device, dtype=dtype
        )

        # CLS token and positional embeddings.
        self.num_prefix_tokens: int = 1
        self.class_embedding = nn.Parameter(
            torch.zeros(cfg.image_emb_dim, device=init_device, dtype=dtype)
        )
        self.positional_embedding = nn.Parameter(
            torch.zeros(cfg.image_num_pos, cfg.image_emb_dim, device=init_device, dtype=dtype)
        )

        self.pre_ln = nn.LayerNorm(
            cfg.image_emb_dim, eps=cfg.image_norm_eps, device=init_device, dtype=dtype
        )
        self.blocks = nn.ModuleList(
            [_ViTBlock(cfg, init_device=init_device) for _ in range(cfg.image_num_layers)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Re-initialise all parameters."""
        scale = self.cfg.image_emb_dim**-0.5
        nn.init.normal_(self.class_embedding, std=scale)
        nn.init.normal_(self.positional_embedding, std=scale)
        nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.pre_ln.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()

    def _add_pos_emb(self, x: torch.Tensor, patch_num: Tuple[int, int]) -> torch.Tensor:
        pos_emb = _interpolate_pos_emb(
            self.positional_embedding, patch_num, num_prefix=self.num_prefix_tokens
        )
        return x + pos_emb[None, :, :].to(x.dtype)

    def forward(
        self, x: torch.Tensor, patch_num: Optional[Tuple[int, int]] = None
    ) -> List[torch.Tensor]:
        """
        Encode pre-patchified images.

        :param x: Float tensor of shape ``(batch, n_patches, patch_size**2 * 3)``.
        :param patch_num: Spatial grid ``(H, W)`` of patches. Defaults to
            ``cfg.image_num_patch``.
        :returns: List of length ``image_num_layers``, each element a tensor of shape
            ``(batch, 1 + n_patches, image_emb_dim)``.
        """
        if patch_num is None:
            patch_num = self.cfg.image_num_patch

        B, N, _ = x.shape
        x = self.patch_embedding(x)

        # Prepend CLS token.
        cls = self.class_embedding[None, None, :].expand(B, 1, -1).to(x.dtype)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, D)
        x = self._add_pos_emb(x, patch_num)
        x = self.pre_ln(x)

        hidden_states: List[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            hidden_states.append(x)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    """
    SigLIP-style vision transformer without a CLS token.

    Matches ``SiglipVisionTransformer`` in the Molmo reference implementation and is
    suitable for weights from SigLIP-SO400M-14-378.

    Like :class:`VisionTransformer`, the forward pass returns hidden states from every
    block. There is no CLS token, so each output tensor has shape
    ``(batch, n_patches, image_emb_dim)``.

    :param cfg: Vision backbone configuration. Use :meth:`VisionBackboneConfig.siglip_so400m`
        for the SigLIP-SO400M defaults.
    :param init_device: Device on which to initialise parameters.
    """

    def __init__(self, cfg: VisionBackboneConfig, init_device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        dtype = cfg.dtype.as_pt()

        patch_pixels = cfg.image_patch_size * cfg.image_patch_size * 3
        self.patch_embedding = nn.Linear(
            patch_pixels, cfg.image_emb_dim, bias=True, device=init_device, dtype=dtype
        )

        self.num_prefix_tokens: int = 0  # no CLS token
        self.positional_embedding = nn.Parameter(
            torch.zeros(cfg.image_num_pos, cfg.image_emb_dim, device=init_device, dtype=dtype)
        )

        self.blocks = nn.ModuleList(
            [_ViTBlock(cfg, init_device=init_device) for _ in range(cfg.image_num_layers)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Re-initialise all parameters."""
        scale = self.cfg.image_emb_dim**-0.5
        nn.init.normal_(self.positional_embedding, std=scale)
        nn.init.normal_(self.patch_embedding.weight, std=0.02)
        nn.init.zeros_(self.patch_embedding.bias)
        for block in self.blocks:
            block.reset_parameters()

    def _add_pos_emb(self, x: torch.Tensor, patch_num: Tuple[int, int]) -> torch.Tensor:
        pos_emb = _interpolate_pos_emb(self.positional_embedding, patch_num, num_prefix=0)
        return x + pos_emb[None, :, :].to(x.dtype)

    def forward(
        self, x: torch.Tensor, patch_num: Optional[Tuple[int, int]] = None
    ) -> List[torch.Tensor]:
        """
        Encode pre-patchified images.

        :param x: Float tensor of shape ``(batch, n_patches, patch_size**2 * 3)``.
        :param patch_num: Spatial grid ``(H, W)`` of patches. Defaults to
            ``cfg.image_num_patch``.
        :returns: List of length ``image_num_layers``, each element a tensor of shape
            ``(batch, n_patches, image_emb_dim)``.
        """
        if patch_num is None:
            patch_num = self.cfg.image_num_patch

        B, N, _ = x.shape
        x = self.patch_embedding(x)
        x = self._add_pos_emb(x, patch_num)

        hidden_states: List[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            hidden_states.append(x)
        return hidden_states
