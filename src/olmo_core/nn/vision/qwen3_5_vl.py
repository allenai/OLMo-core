"""
Native multimodal **Qwen3.5-VL** support for OLMo-core.

Loads a HuggingFace ``Qwen3_5ForConditionalGeneration`` checkpoint (e.g.
``Qwen/Qwen3.5-0.8B``) into an OLMo-core model and runs image→text inference.

The architecture has three parts:

1. A **native vision tower** (:class:`Qwen3_5VisionModel`) — a dynamic-resolution
   ViT with a ``Conv3d`` patch embedding, bilinearly-interpolated learned position
   embeddings, 2D rotary attention, and a spatial-merge **patch merger** that
   produces one embedding per ``<image_pad>`` placeholder token. Ported faithfully
   from ``transformers`` so HF vision weights load 1:1.

2. The existing OLMo-core **Qwen3.5 hybrid LM** (GatedDeltaNet + full attention),
   reused unchanged. Its weights are converted by
   :func:`~olmo_core.nn.hf.convert.convert_qwen3_5_state_from_hf` (which already
   normalizes the ``model.language_model.*`` nesting and skips vision keys).

3. **M-RoPE** — 3D rotary position embeddings. Image tokens get distinct
   temporal/height/width positions; text tokens collapse to standard 1D RoPE. The
   cos/sin tables are precomputed here and injected into the LM's full-attention
   blocks via the ``pos_sin`` / ``pos_cos`` arguments of
   :meth:`~olmo_core.nn.transformer.Transformer.forward`.

Image preprocessing reuses the HF ``AutoProcessor`` (``pixel_values`` /
``image_grid_thw`` / ``input_ids``); only the *model* is native OLMo-core.

.. note::
    The 0.8B checkpoint has ``deepstack_visual_indexes == []`` (deepstack disabled),
    so vision features are injected only at the embedding layer via a masked scatter.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import Self

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.rope import build_mrope_sin_cos, compute_inv_freqs
from olmo_core.nn.transformer import Transformer, TransformerConfig

__all__ = [
    "Qwen3_5VisionConfig",
    "Qwen3_5VisionModel",
    "Qwen3_5VL",
    "qwen3_5_get_rope_index",
    "load_qwen3_5_vl_from_hf",
]


# ---------------------------------------------------------------------------
# Vision config
# ---------------------------------------------------------------------------


@dataclass
class Qwen3_5VisionConfig:
    """Configuration for the Qwen3.5 native vision tower."""

    depth: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_heads: int = 12
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 1024
    num_position_embeddings: int = 2304
    hidden_act: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_hf(cls, vc: Any) -> Self:
        get = (
            (lambda k, d=None: vc.get(k, d))
            if isinstance(vc, dict)
            else (lambda k, d=None: getattr(vc, k, d))
        )
        return cls(
            depth=get("depth"),
            hidden_size=get("hidden_size"),
            intermediate_size=get("intermediate_size"),
            num_heads=get("num_heads"),
            in_channels=get("in_channels", 3),
            patch_size=get("patch_size"),
            temporal_patch_size=get("temporal_patch_size"),
            spatial_merge_size=get("spatial_merge_size"),
            out_hidden_size=get("out_hidden_size"),
            num_position_embeddings=get("num_position_embeddings"),
            hidden_act=get("hidden_act", "gelu_pytorch_tanh"),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


# ---------------------------------------------------------------------------
# Vision tower (ported from transformers Qwen3_5VisionModel)
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class _VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: int = 10000) -> None:
        super().__init__()
        inv_freq = compute_inv_freqs(theta, dim, torch.device("cpu"))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class _VisionPatchEmbed(nn.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.in_channels = cfg.in_channels
        self.temporal_patch_size = cfg.temporal_patch_size
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.hidden_size
        ks = [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size]
        self.proj = nn.Conv3d(
            cfg.in_channels, cfg.hidden_size, kernel_size=ks, stride=ks, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, self.embed_dim)


class _VisionMLP(nn.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class _VisionAttention(nn.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.hidden_size, cfg.hidden_size * 3, bias=True)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_len = x.shape[0]
        q, k, v = self.qkv(x).reshape(seq_len, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)  # (seq, heads, head_dim)

        # (1, heads, seq, head_dim)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        # Block-diagonal attention by image via cu_seqlens (no causal mask).
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outs = []
        for q_i, k_i, v_i in zip(
            torch.split(q, lengths, dim=2),
            torch.split(k, lengths, dim=2),
            torch.split(v, lengths, dim=2),
        ):
            outs.append(
                F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=False, scale=self.scaling)
            )
        attn = torch.cat(outs, dim=2)  # (1, heads, seq, head_dim)
        attn = attn.squeeze(0).transpose(0, 1).reshape(seq_len, -1)
        return self.proj(attn)


class _VisionBlock(nn.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.attn = _VisionAttention(cfg)
        self.mlp = _VisionMLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings
        )
        x = x + self.mlp(self.norm2(x))
        return x


class _VisionPatchMerger(nn.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.hidden_size = cfg.hidden_size * (cfg.spatial_merge_size**2)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, cfg.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3_5VisionModel(nn.Module):
    """Native Qwen3.5 vision tower (patch embed → blocks → spatial-merge merger)."""

    def __init__(self, cfg: Qwen3_5VisionConfig, init_device: str = "cpu") -> None:
        super().__init__()
        self.cfg = cfg
        self.spatial_merge_size = cfg.spatial_merge_size
        self.num_grid_per_side = int(cfg.num_position_embeddings**0.5)

        self.patch_embed = _VisionPatchEmbed(cfg)
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size)
        self.rotary_pos_emb = _VisionRotaryEmbedding(cfg.head_dim // 2)
        self.blocks = nn.ModuleList([_VisionBlock(cfg) for _ in range(cfg.depth)])
        self.merger = _VisionPatchMerger(cfg)
        self.to(init_device)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge = self.spatial_merge_size
        grid = grid_thw.tolist()
        max_hw = max(max(h, w) for _, h, w in grid)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total = sum(t * h * w for t, h, w in grid)
        pos_ids = torch.empty((total, 2), dtype=torch.long, device=device)
        offset = 0
        for t, h, w in grid:
            mh, mw = h // merge, w // merge
            br = torch.arange(mh, device=device)
            bc = torch.arange(mw, device=device)
            ir = torch.arange(merge, device=device)
            ic = torch.arange(merge, device=device)
            row = (
                (br[:, None, None, None] * merge + ir[None, None, :, None])
                .expand(mh, mw, merge, merge)
                .reshape(-1)
            )
            col = (
                (bc[None, :, None, None] * merge + ic[None, None, None, :])
                .expand(mh, mw, merge, merge)
                .reshape(-1)
            )
            coords = torch.stack((row, col), dim=-1)
            if t > 1:
                coords = coords.repeat(t, 1)
            n = coords.shape[0]
            pos_ids[offset : offset + n] = coords
            offset += n
        emb = freq_table[pos_ids].flatten(1)
        return emb

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid = grid_thw.tolist()
        device = self.pos_embed.weight.device
        n_side = self.num_grid_per_side
        merge = self.spatial_merge_size

        idx_list: List[List[int]] = [[] for _ in range(4)]
        weight_list: List[List[float]] = [[] for _ in range(4)]
        for _, h, w in grid:
            h_idxs = torch.linspace(0, n_side - 1, h)
            w_idxs = torch.linspace(0, n_side - 1, w)
            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_idxs.int() + 1).clip(max=n_side - 1)
            w_ceil = (w_idxs.int() + 1).clip(max=n_side - 1)
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor
            base_h = h_floor * n_side
            base_h_ceil = h_ceil * n_side
            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_t = torch.tensor(idx_list, dtype=torch.long, device=device)
        w_t = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos = self.pos_embed(idx_t).to(device) * w_t[:, :, None]
        patch_pos = pos[0] + pos[1] + pos[2] + pos[3]
        patch_pos = patch_pos.split([h * w for _, h, w in grid])

        out = []
        for pe, (t, h, w) in zip(patch_pos, grid):
            pe = pe.repeat(t, 1)
            pe = (
                pe.view(t, h // merge, merge, w // merge, merge, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            out.append(pe)
        return torch.cat(out)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Encode patchified pixels into merged per-token image features.

        :param pixel_values: ``(total_patches, in_ch * t_patch * patch * patch)``.
        :param grid_thw: ``(num_images, 3)`` of ``(t, h, w)`` patch-grid dims.
        :returns: ``(num_merged_tokens, out_hidden_size)``.
        """
        x = self.patch_embed(pixel_values)
        x = x + self.fast_pos_embed_interpolate(grid_thw)

        rotary = self.rot_pos_emb(grid_thw).reshape(x.shape[0], -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            0, dtype=torch.int32
        )
        cu = F.pad(cu, (1, 0), value=0)

        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu, position_embeddings=position_embeddings)
        return self.merger(x)


# ---------------------------------------------------------------------------
# M-RoPE
# ---------------------------------------------------------------------------


def qwen3_5_get_rope_index(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    image_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Compute 3D ``(3, 1, seq)`` position ids for a single ``B==1`` sequence.

    Ported from ``Qwen3_5Model.get_rope_index`` for the image-only path: text spans
    use a shared running 1D position; each image block lays out temporal/height/width
    positions over its merged grid.
    """
    assert input_ids.shape[0] == 1, "only B==1 supported"
    device = input_ids.device
    ids = input_ids[0]
    # token type: 1 where image placeholder, else 0.
    is_image = ids == image_token_id

    pos_list = []
    current = 0
    img_iter = iter(image_grid_thw.tolist()) if image_grid_thw is not None else iter([])
    i = 0
    n = ids.shape[0]
    while i < n:
        if is_image[i]:
            t, h, w = next(img_iter)
            lt, lh, lw = t, h // spatial_merge_size, w // spatial_merge_size
            p_t = torch.arange(lt, device=device).repeat_interleave(lh * lw) + current
            p_h = torch.arange(lh, device=device).repeat_interleave(lw).repeat(lt) + current
            p_w = torch.arange(lw, device=device).repeat(lh * lt) + current
            pos_list.append(torch.stack([p_t, p_h, p_w], dim=0))
            current += max(lh, lw)
            i += lt * lh * lw
        else:
            # contiguous text run
            j = i
            while j < n and not is_image[j]:
                j += 1
            text_len = j - i
            rng = torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + current
            pos_list.append(rng)
            current += text_len
            i = j
    positions = torch.cat(pos_list, dim=1)  # (3, seq)
    return positions[:, None, :]  # (3, 1, seq)


# ---------------------------------------------------------------------------
# Top-level VL model
# ---------------------------------------------------------------------------


class Qwen3_5VL(nn.Module):
    """Native multimodal Qwen3.5: vision tower + OLMo-core Qwen3.5 hybrid LM."""

    def __init__(
        self,
        vision: Qwen3_5VisionModel,
        lm: Transformer,
        *,
        image_token_id: int,
        vision_start_token_id: int,
        spatial_merge_size: int,
        head_dim: int,
        partial_rotary_factor: float,
        rope_theta: int,
        mrope_section: List[int],
    ) -> None:
        super().__init__()
        self.vision = vision
        self.lm = lm
        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id
        self.spatial_merge_size = spatial_merge_size
        self.head_dim = head_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.mrope_section = mrope_section

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        logits_to_keep: int = 0,
    ) -> torch.Tensor:
        device = self.device
        input_ids = input_ids.to(device)
        h = self.lm.embeddings(input_ids)  # (B, S, d_model)

        if pixel_values is not None and image_grid_thw is not None:
            img = self.vision(pixel_values.to(device), image_grid_thw.to(device)).to(h.dtype)
            mask = input_ids == self.image_token_id  # (B, S)
            h = h.clone()
            h[mask] = img

        position_ids = qwen3_5_get_rope_index(
            input_ids,
            image_grid_thw,
            image_token_id=self.image_token_id,
            vision_start_token_id=self.vision_start_token_id,
            spatial_merge_size=self.spatial_merge_size,
        )
        pos_sin, pos_cos = build_mrope_sin_cos(
            position_ids,
            head_dim=self.head_dim,
            partial_rotary_factor=self.partial_rotary_factor,
            theta=self.rope_theta,
            mrope_section=self.mrope_section,
            device=device,
            dtype=h.dtype,
        )
        return self.lm(
            input_ids,
            input_embeddings=h,
            pos_sin=pos_sin,
            pos_cos=pos_cos,
            logits_to_keep=logits_to_keep,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        *,
        max_new_tokens: int = 40,
        eos_token_ids: Tuple[int, ...] = (),
    ) -> List[int]:
        """Greedy decode (no KV cache; image is re-encoded each step)."""
        gen = input_ids.to(self.device)
        start = gen.shape[1]
        for _ in range(max_new_tokens):
            logits = self.forward(gen, pixel_values, image_grid_thw, logits_to_keep=1)
            nxt = int(logits[0, -1].argmax().item())
            if nxt in eos_token_ids:
                break
            gen = torch.cat([gen, torch.tensor([[nxt]], device=gen.device)], dim=1)
        return gen[0, start:].tolist()


# ---------------------------------------------------------------------------
# HF loading
# ---------------------------------------------------------------------------


def _load_vision_from_hf(vision: Qwen3_5VisionModel, hf_state: Dict[str, torch.Tensor]) -> None:
    """Load ``model.visual.*`` HF weights into the native vision tower (1:1 names)."""
    own = dict(vision.state_dict())
    mapped: Dict[str, torch.Tensor] = {}
    for k, v in hf_state.items():
        if not k.startswith("model.visual."):
            continue
        nk = k[len("model.visual.") :]
        if nk in own:
            mapped[nk] = v
    missing = set(own) - set(mapped)
    # inv_freq is a non-persistent buffer recomputed at init.
    missing = {m for m in missing if not m.endswith("inv_freq")}
    if missing:
        raise RuntimeError(f"vision tower missing weights for: {sorted(missing)[:8]}")
    vision.load_state_dict(mapped, strict=False)


def load_qwen3_5_vl_from_hf(
    model_id: str = "Qwen/Qwen3.5-0.8B",
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    hf_model: Optional[Any] = None,
) -> Tuple["Qwen3_5VL", Any]:
    """Load a HF ``Qwen3_5ForConditionalGeneration`` checkpoint into :class:`Qwen3_5VL`.

    :param hf_model: An already-loaded HF ``Qwen3_5ForConditionalGeneration`` to
        convert from. When given, its config/weights are reused instead of
        instantiating a fresh copy — useful to avoid holding two HF copies in
        memory (e.g. in parity tests that also keep the HF model for comparison).
    :returns: ``(model, processor)`` — ``processor`` is the HF ``AutoProcessor`` used
        for image/text preprocessing.
    """
    import transformers
    from transformers import AutoProcessor

    from olmo_core.nn.hf.convert import convert_qwen3_5_state_from_hf

    if hf_model is None:
        hf_model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype
        )
    hf_config = hf_model.config
    text_cfg = hf_config.text_config
    vis_cfg = Qwen3_5VisionConfig.from_hf(hf_config.vision_config)

    hf_state = hf_model.state_dict()

    # --- LM ---
    lm_cfg = TransformerConfig.qwen3_5_from_hf_config(
        text_cfg, attn_backend=AttentionBackendName.torch, dtype=DType.float32
    )
    lm = lm_cfg.build(init_device="cpu")
    lm_state = convert_qwen3_5_state_from_hf(text_cfg, hf_state)
    lm.load_state_dict(lm_state, strict=True)

    # --- Vision ---
    vision = Qwen3_5VisionModel(vis_cfg, init_device="cpu")
    _load_vision_from_hf(vision, hf_state)

    rp = getattr(text_cfg, "rope_parameters", None) or {}
    model = Qwen3_5VL(
        vision,
        lm,
        image_token_id=hf_config.image_token_id,
        vision_start_token_id=hf_config.vision_start_token_id,
        spatial_merge_size=vis_cfg.spatial_merge_size,
        head_dim=text_cfg.head_dim,
        partial_rotary_factor=text_cfg.partial_rotary_factor,
        rope_theta=rp.get("rope_theta", getattr(text_cfg, "rope_theta", 10_000_000)),
        mrope_section=rp.get("mrope_section", [11, 11, 10]),
    )
    del hf_model
    model = model.to(device=device, dtype=dtype).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor
