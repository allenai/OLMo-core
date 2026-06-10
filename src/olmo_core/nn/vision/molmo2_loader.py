"""
HuggingFace Molmo2 → :class:`~olmo_core.nn.vision.MultimodalLM`
state-dict converter.

Loads weights from a public Molmo2 checkpoint on HuggingFace (e.g.
``allenai/Molmo2-O-7B``) into our composite multimodal model.

The two architectures differ in three places that this converter handles:

1. **LM token embedding split.** HF Molmo2 keeps the base vocab and the extra
   image-special-token vocab in two separate parameters
   (``transformer.wte.embedding`` and ``transformer.wte.new_embedding``);
   our :class:`~olmo_core.nn.transformer.Transformer` uses a single
   ``embeddings.weight`` table. We concatenate.

2. **Fused QKV and gated MLP in the LM.** HF Molmo2 has
   ``self_attn.att_proj`` (fused Q+K+V) and ``mlp.ff_proj`` (fused gate+up
   for SwiGLU). Our LM uses ``w_q`` / ``w_k`` / ``w_v`` and ``w1`` / ``w3``.
   We split along the output dimension.

3. **Patch-embedding flatten order in the vision encoder.** Molmo2's HF
   ``image_processing_molmo2.batch_pixels_to_patches`` lays patches out
   spatial-first (``kh, kw, c``), while our
   :class:`~olmo_core.data.multimodal.ImagePreprocessor` uses channel-first
   (``c, kh, kw``) to match the HuggingFace ``Conv2d`` patch-embedding
   convention our CLIP/SigLIP parity tests verified against. We permute
   the patch-embedding weight to bridge the two.

Vision blocks, the connector pooling cross-attention, the connector SwiGLU
projector, and the LM head all map one-to-one (no per-tensor surgery).

Usage::

    cfg = MultimodalLMConfig(lm=olmo3_7B(...), vision=..., connector=...)
    model = MultimodalLM(cfg)
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained("allenai/Molmo2-O-7B", trust_remote_code=True)
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf.state_dict(), cfg)
    model.load_state_dict(converted)
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from ...config import DType
from ..transformer import TransformerConfig
from .config import VisionEncoderConfig, VisionEncoderType
from .connector import ImagePoolingType, ImageProjectorType, VisionConnectorConfig
from .multimodal import MultimodalLMConfig

log = logging.getLogger(__name__)

__all__ = [
    "molmo2_hf_state_dict_to_multimodal_lm",
    "molmo2_config_from_hf_config",
    "ensure_default_rope_registered",
    "reinit_rope_buffers",
]


class Molmo2LoaderError(RuntimeError):
    """Raised when the HF Molmo2 state dict can't be mapped to our model."""


def ensure_default_rope_registered() -> None:
    """Re-register ``ROPE_INIT_FUNCTIONS["default"]`` if upstream removed it.

    transformers ≥ 5 dropped the ``"default"`` key from
    ``modeling_rope_utils.ROPE_INIT_FUNCTIONS``, but the Molmo2 modeling
    code bundled inside HF checkpoints still references it.  Call this
    **before** ``from_pretrained`` so the model can be instantiated.
    Safe to call repeatedly; a no-op when ``"default"`` is already present.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _default_rope(config, device=None, seq_len=None, **kwargs):
        base = getattr(config, "rope_theta", 10000.0)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _default_rope


def reinit_rope_buffers(model: "torch.nn.Module") -> None:
    """Re-initialise non-persistent RoPE ``inv_freq`` buffers after loading.

    transformers ≥ 5 uses meta-device fast-init in ``from_pretrained``,
    which skips ``__init__`` for non-persistent buffers.  This leaves
    ``inv_freq`` as uninitialised memory and breaks positional encoding for
    all sequence positions beyond 0.  Call this **after** ``from_pretrained``
    on any Molmo2 model loaded with ``trust_remote_code=True``.

    :param model: An HF Molmo2 model returned by ``from_pretrained``.
    """
    for _, mod in model.named_modules():
        if hasattr(mod, "rope_init_fn") and hasattr(mod, "inv_freq") and hasattr(mod, "config"):
            inv_freq, attn_scaling = mod.rope_init_fn(mod.config, None)
            mod.register_buffer("inv_freq", inv_freq, persistent=False)
            mod.original_inv_freq = mod.inv_freq
            mod.attention_scaling = attn_scaling


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require(hf_sd: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in hf_sd:
        raise Molmo2LoaderError(f"missing required HF Molmo2 key: {key!r}")
    return hf_sd[key]


def _maybe(hf_sd: Dict[str, torch.Tensor], key: str) -> Optional[torch.Tensor]:
    return hf_sd.get(key)


def _attention_dims(lm_cfg: TransformerConfig) -> tuple[int, int, int]:
    """Read ``(n_heads, n_kv_heads, head_dim)`` from the LM's attention config.

    OLMo-3 configs put attention dims under
    ``lm.block.attention`` (or ``lm.block.sequence_mixer`` for newer configs).
    Try both.
    """
    block = lm_cfg.block
    seq_mixer = getattr(block, "attention", None) or getattr(block, "sequence_mixer", None)
    if seq_mixer is None:
        raise Molmo2LoaderError(
            "Unable to find attention config on TransformerBlockConfig; "
            "tried `attention` and `sequence_mixer`"
        )
    n_heads = getattr(seq_mixer, "n_heads", None) or getattr(seq_mixer, "num_heads", None)
    n_kv_heads = (
        getattr(seq_mixer, "n_kv_heads", None)
        or getattr(seq_mixer, "num_kv_heads", None)
        or n_heads
    )
    head_dim = getattr(seq_mixer, "head_dim", None)
    if head_dim is None and n_heads is not None:
        head_dim = lm_cfg.d_model // n_heads
    if n_heads is None or head_dim is None:
        raise Molmo2LoaderError(
            f"Could not derive (n_heads, n_kv_heads, head_dim) from {seq_mixer!r}"
        )
    if n_kv_heads is None:
        n_kv_heads = n_heads
    return int(n_heads), int(n_kv_heads), int(head_dim)


def _has_qk_norm(lm_cfg: TransformerConfig) -> bool:
    block = lm_cfg.block
    seq_mixer = getattr(block, "attention", None) or getattr(block, "sequence_mixer", None)
    return getattr(seq_mixer, "qk_norm", None) is not None


def _convert_patch_embedding(hf_patch_weight: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Permute Molmo2's spatial-first patch weight to our C-first convention.

    HF layout (per row): ``[kh=0,kw=0,c=0], [kh=0,kw=0,c=1], [kh=0,kw=0,c=2],
    [kh=0,kw=1,c=0], …`` — i.e. ``(p, p, 3)`` flattened.

    Our layout (per row): ``[c=0,kh=0,kw=0], [c=0,kh=0,kw=1], …,
    [c=1,kh=0,kw=0], …`` — i.e. ``(3, p, p)`` flattened (matches a
    HuggingFace ``Conv2d.weight.reshape(D, -1)`` from our CLIP/SigLIP parity
    tests).
    """
    D, total = hf_patch_weight.shape
    expected = patch_size * patch_size * 3
    if total != expected:
        raise Molmo2LoaderError(
            f"HF patch embedding weight has shape {(D, total)} but expected "
            f"{(D, expected)} given patch_size={patch_size}"
        )
    # HF stores rows as (p, p, c). Permute → (c, p, p).
    return (
        hf_patch_weight.reshape(D, patch_size, patch_size, 3)
        .permute(0, 3, 1, 2)
        .reshape(D, expected)
        .contiguous()
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def molmo2_hf_state_dict_to_multimodal_lm(
    hf_state_dict: Dict[str, torch.Tensor],
    cfg: MultimodalLMConfig,
) -> Dict[str, torch.Tensor]:
    """Convert a Molmo2 HF state dict to our :class:`MultimodalLM` layout.

    :param hf_state_dict: Output of ``hf_model.state_dict()`` where
        ``hf_model`` is a :class:`transformers.AutoModelForCausalLM` /
        ``Molmo2ForConditionalGeneration`` loaded from one of
        ``allenai/Molmo2-*``.
    :param cfg: The target multimodal config. Used to read per-module
        dimensions (n_heads, head_dim, n_layers, patch_size) so the converter
        can split fused weights correctly.
    :returns: A new ``Dict[str, Tensor]`` keyed for
        :meth:`~olmo_core.nn.vision.MultimodalLM.load_state_dict`.
    :raises Molmo2LoaderError: when the HF state dict is missing a required
        key or has an unexpected shape.
    """
    out: Dict[str, torch.Tensor] = {}

    lm_cfg = cfg.lm
    n_layers = lm_cfg.n_layers
    n_heads, n_kv_heads, head_dim = _attention_dims(lm_cfg)
    fused_attn_dims = (
        n_heads * head_dim,
        n_kv_heads * head_dim,
        n_kv_heads * head_dim,
    )
    has_qk_norm = _has_qk_norm(lm_cfg)

    # --- LM: token embeddings -----------------------------------------------
    base_emb = _require(hf_state_dict, "model.transformer.wte.embedding")
    new_emb = _require(hf_state_dict, "model.transformer.wte.new_embedding")
    out["lm.embeddings.weight"] = torch.cat([base_emb, new_emb], dim=0)

    # --- LM: final norm + lm head ------------------------------------------
    out["lm.lm_head.norm.weight"] = _require(hf_state_dict, "model.transformer.ln_f.weight")

    # HF Molmo2's lm_head is sized to the *base* vocab only — image special
    # tokens are inputs-only and never predicted. Our OLMo-core LM uses a
    # single vocab_size for both input and output, so pad the lm_head with
    # zero rows for the image-token slots.
    #
    # NOTE: zero rows give those positions a *finite* logit (0), not -inf, so
    # they enter the softmax denominator and shift the cross-entropy loss
    # relative to HF (which has no such columns). Argmax/greedy generation is
    # unaffected as long as a real token outscores 0, but exact *loss* parity
    # requires masking the extra output IDs to -inf at the loss/forward layer
    # (handled where the loss is computed, not in this weight converter). The
    # logit-parity tests therefore compare only the base-vocab columns.
    hf_lm_head = _require(hf_state_dict, "lm_head.weight")
    extra_rows = lm_cfg.vocab_size - hf_lm_head.shape[0]
    if extra_rows < 0:
        raise Molmo2LoaderError(
            f"HF lm_head has {hf_lm_head.shape[0]} rows but our LM "
            f"vocab_size is only {lm_cfg.vocab_size}"
        )
    if extra_rows > 0:
        # new_zeros preserves the source tensor's device and dtype, so this
        # works when the HF state dict is already on CUDA.
        pad = hf_lm_head.new_zeros((extra_rows, hf_lm_head.shape[1]))
        out["lm.lm_head.w_out.weight"] = torch.cat([hf_lm_head, pad], dim=0)
    else:
        out["lm.lm_head.w_out.weight"] = hf_lm_head

    # --- LM: per-block --------------------------------------------------------
    for i in range(n_layers):
        src = f"model.transformer.blocks.{i}"
        dst = f"lm.blocks.{i}"

        # Pre-attn / pre-mlp norms.
        out[f"{dst}.attention_norm.weight"] = _require(hf_state_dict, f"{src}.attn_norm.weight")
        out[f"{dst}.feed_forward_norm.weight"] = _require(hf_state_dict, f"{src}.ff_norm.weight")

        # Fused QKV → split.
        att_w = _require(hf_state_dict, f"{src}.self_attn.att_proj.weight")
        q_w, k_w, v_w = att_w.split(fused_attn_dims, dim=0)
        out[f"{dst}.attention.w_q.weight"] = q_w.contiguous()
        out[f"{dst}.attention.w_k.weight"] = k_w.contiguous()
        out[f"{dst}.attention.w_v.weight"] = v_w.contiguous()
        if (att_b := _maybe(hf_state_dict, f"{src}.self_attn.att_proj.bias")) is not None:
            q_b, k_b, v_b = att_b.split(fused_attn_dims, dim=0)
            out[f"{dst}.attention.w_q.bias"] = q_b.contiguous()
            out[f"{dst}.attention.w_k.bias"] = k_b.contiguous()
            out[f"{dst}.attention.w_v.bias"] = v_b.contiguous()

        # QK-norm if present.
        if has_qk_norm:
            out[f"{dst}.attention.q_norm.weight"] = _require(
                hf_state_dict, f"{src}.self_attn.q_norm.weight"
            )
            out[f"{dst}.attention.k_norm.weight"] = _require(
                hf_state_dict, f"{src}.self_attn.k_norm.weight"
            )

        # Attn output projection.
        out[f"{dst}.attention.w_out.weight"] = _require(
            hf_state_dict, f"{src}.self_attn.attn_out.weight"
        )

        # Fused SwiGLU → split.
        # HF forward: x, gate = ff_proj.chunk(2); x = act(gate) * x → ff_out.
        # So ff_proj.weight[:H] feeds the *multiplier* branch (our w3),
        #   ff_proj.weight[H:2H] feeds the *gate* branch (our w1).
        ff_proj_w = _require(hf_state_dict, f"{src}.mlp.ff_proj.weight")
        mul_w, gate_w = ff_proj_w.chunk(2, dim=0)
        out[f"{dst}.feed_forward.w3.weight"] = mul_w.contiguous()
        out[f"{dst}.feed_forward.w1.weight"] = gate_w.contiguous()
        out[f"{dst}.feed_forward.w2.weight"] = _require(hf_state_dict, f"{src}.mlp.ff_out.weight")

    # --- Vision: patch embedding + positional ---------------------------------
    vit_layers = cfg.vision.image_num_layers
    p = cfg.vision.image_patch_size
    hf_patch_w = _require(hf_state_dict, "model.vision_backbone.image_vit.patch_embedding.weight")
    out["vision.patch_embedding.weight"] = _convert_patch_embedding(hf_patch_w, p)
    out["vision.patch_embedding.bias"] = _require(
        hf_state_dict, "model.vision_backbone.image_vit.patch_embedding.bias"
    )
    out["vision.positional_embedding"] = _require(
        hf_state_dict, "model.vision_backbone.image_vit.positional_embedding"
    )

    # --- Vision: per-block ----------------------------------------------------
    for i in range(vit_layers):
        src = f"model.vision_backbone.image_vit.transformer.resblocks.{i}"
        dst = f"vision.blocks.{i}"
        # Norms.
        for hf_name, ours_name in (("attention_norm", "attn_norm"), ("ffn_norm", "ffn_norm")):
            for suffix in ("weight", "bias"):
                out[f"{dst}.{ours_name}.{suffix}"] = _require(
                    hf_state_dict, f"{src}.{hf_name}.{suffix}"
                )
        # QKV + output projection (no fusing on the vision side).
        for proj in ("wq", "wk", "wv", "wo"):
            for suffix in ("weight", "bias"):
                out[f"{dst}.attn.{proj}.{suffix}"] = _require(
                    hf_state_dict, f"{src}.attention.{proj}.{suffix}"
                )
        # Plain MLP (no SwiGLU in the ViT).
        for proj in ("w1", "w2"):
            for suffix in ("weight", "bias"):
                out[f"{dst}.ffn.{proj}.{suffix}"] = _require(
                    hf_state_dict, f"{src}.feed_forward.{proj}.{suffix}"
                )

    # --- Connector: pooling cross-attention + projector SwiGLU --------------
    for proj in ("wq", "wk", "wv", "wo"):
        for suffix in ("weight", "bias"):
            out[f"connector.pooling.{proj}.{suffix}"] = _require(
                hf_state_dict, f"model.vision_backbone.image_pooling_2d.{proj}.{suffix}"
            )
    for proj in ("w1", "w2", "w3"):
        out[f"connector.projector.{proj}.weight"] = _require(
            hf_state_dict, f"model.vision_backbone.image_projector.{proj}.weight"
        )

    return out


# ---------------------------------------------------------------------------
# HF Molmo2 config → MultimodalLMConfig
# ---------------------------------------------------------------------------


def _build_lm_config(text_cfg, total_vocab_size: int) -> TransformerConfig:
    """Build a :class:`TransformerConfig` matching an HF Molmo2 text_config.

    Dispatches to ``qwen3_4B`` / ``qwen3_8B`` (qk_norm_type="qwen3"),
    ``olmo3_7B`` (default qk_norm), or raises if no factory matches.

    TODO: this dispatch hard-codes a growing ``(qk_norm_type, hidden_size,
    n_layers)`` lookup table mapping to named factory methods, and ignores most
    of the HF ``text_config`` fields (intermediate_size, num_attention_heads,
    head_dim, rms_norm_eps, tie_word_embeddings, etc.). It will be hard to
    maintain as more checkpoints are added. Replace it with a generic builder
    that constructs the ``TransformerConfig`` directly from the HF fields rather
    than matching dimensions to a preset.
    """
    from ..attention import AttentionBackendName

    hidden_size = text_cfg.hidden_size
    n_layers = text_cfg.num_hidden_layers
    qk_norm_type = getattr(text_cfg, "qk_norm_type", None)
    rope_theta = text_cfg.rope_theta

    # Branch by qk_norm_type then dimensions.
    if qk_norm_type == "qwen3" and hidden_size == 2560 and n_layers == 36:
        return TransformerConfig.qwen3_4B(
            vocab_size=total_vocab_size,
            rope_theta=rope_theta,
            attn_backend=AttentionBackendName.torch,
            dtype=DType.float32,
        )
    if qk_norm_type == "qwen3" and hidden_size == 4096 and n_layers == 36:
        return TransformerConfig.qwen3_8B(
            vocab_size=total_vocab_size,
            rope_theta=rope_theta,
            attn_backend=AttentionBackendName.torch,
            dtype=DType.float32,
        )
    if hidden_size == 4096 and n_layers == 32:
        # Olmo3-7B layout (per-channel qk_norm).
        rope_scaling_layers = getattr(text_cfg, "rope_scaling_layers", None)
        if rope_scaling_layers is not None:
            # Molmo2-O-7B uses per-layer YaRN attention scaling (attention_factor ≈ 1.208)
            # on a subset of layers.  MultimodalLM currently applies a single global
            # RoPE config, so the per-layer attention_factor is silently ignored.  Logit
            # magnitudes will differ from HF's reference by ~3-4 nats for short sequences;
            # predicted argmax still matches for text-only paths.
            log.warning(
                "Molmo2 text config has 'rope_scaling_layers=%s', indicating per-layer "
                "YaRN attention scaling (attention_factor=%.4f).  "
                "MultimodalLM does not support per-layer RoPE variants; "
                "logit magnitudes may differ from the HF reference.",
                rope_scaling_layers[:4],
                text_cfg.rope_scaling.get("attention_factor", float("nan")),
            )
        return TransformerConfig.olmo3_7B(
            vocab_size=total_vocab_size,
            rope_theta=rope_theta,
            attn_backend=AttentionBackendName.torch,
            dtype=DType.float32,
        )
    raise Molmo2LoaderError(
        f"No matching TransformerConfig factory for HF Molmo2 text_config: "
        f"hidden_size={hidden_size}, n_layers={n_layers}, qk_norm_type={qk_norm_type!r}"
    )


def _effective_vit_layers(vit_cfg, vit_layers: list[int]) -> int:
    """Mirror HF Molmo2's optimization: drop ViT blocks above
    ``max(vit_layers) + 1`` since they're never read."""
    total = vit_cfg.num_hidden_layers
    # Resolve negative indices to positive.
    resolved = [(layer if layer >= 0 else layer + total) for layer in vit_layers]
    last_needed = max(resolved) + 1
    return min(last_needed, total)


def _build_vision_config(vit_cfg, vit_layers: list[int]) -> VisionEncoderConfig:
    """Build a :class:`VisionEncoderConfig` matching an HF Molmo2 vit_config.

    Molmo2 ViTs match SigLIP-style (no CLS token, ``gelu_pytorch_tanh``
    activation), so we use :attr:`VisionEncoderType.siglip`. We also mirror
    HF Molmo2's truncation of unused upper ViT blocks: if
    ``adapter_config.vit_layers`` is ``[-3, -9]``, only layers 0..24 are
    instantiated (saves storage and compute).
    """
    raw_size = vit_cfg.image_default_input_size
    image_size: Tuple[int, int] = (int(raw_size[0]), int(raw_size[1]))
    return VisionEncoderConfig(
        name=VisionEncoderType.siglip,
        # SigLIP-style variant switches: no CLS token, a biased patch projection,
        # and no pre-LayerNorm. These must be set explicitly — `name` alone no
        # longer selects the architecture.
        use_cls_token=False,
        patch_embedding_bias=True,
        use_pre_ln=False,
        image_default_input_size=image_size,
        image_patch_size=vit_cfg.image_patch_size,
        image_emb_dim=vit_cfg.hidden_size,
        image_num_heads=vit_cfg.num_attention_heads,
        image_num_key_value_heads=vit_cfg.num_key_value_heads,
        image_num_layers=_effective_vit_layers(vit_cfg, vit_layers),
        image_head_dim=vit_cfg.head_dim,
        image_mlp_dim=vit_cfg.intermediate_size,
        image_mlp_activations=vit_cfg.hidden_act,
        image_num_pos=vit_cfg.image_num_pos,
        image_norm_eps=vit_cfg.layer_norm_eps,
        dtype=DType.float32,
    )


def _build_connector_config(adapter_cfg, vit_hidden_size: int) -> VisionConnectorConfig:
    """Build :class:`VisionConnectorConfig` from HF Molmo2 adapter_config.

    ``vit_layers`` in the adapter config tells us how many ViT layers are
    concatenated into the pooling cross-attention's input dim
    (``num_input_layers``).
    """
    num_input_layers = len(adapter_cfg.vit_layers)
    return VisionConnectorConfig(
        image_emb_dim=vit_hidden_size,
        image_num_heads=adapter_cfg.num_attention_heads,
        image_num_key_value_heads=adapter_cfg.num_key_value_heads,
        image_head_dim=adapter_cfg.head_dim,
        output_dim=adapter_cfg.text_hidden_size,
        num_input_layers=num_input_layers,
        pooling_type=ImagePoolingType.attention_meanq,
        pooling_attention_mask=adapter_cfg.pooling_attention_mask,
        projector_type=ImageProjectorType.mlp,
        mlp_hidden_size=adapter_cfg.intermediate_size,
        dtype=DType.float32,
    )


def molmo2_config_from_hf_config(hf_config: Any) -> MultimodalLMConfig:
    """Build a :class:`MultimodalLMConfig` from a Molmo2 HF config.

    :param hf_config: A loaded HF Molmo2 config — typically
        ``transformers.AutoConfig.from_pretrained("allenai/Molmo2-*",
        trust_remote_code=True)``.
    :returns: A :class:`MultimodalLMConfig` whose model architecture
        matches the HF Molmo2 layout, ready to receive converted weights.
    """
    text_cfg = hf_config.text_config
    vit_cfg = hf_config.vit_config
    adapter_cfg = hf_config.adapter_config

    # HF Molmo2 keeps the extra image-token embeddings in a separate
    # parameter (``new_embedding``); after concatenation our model needs the
    # combined vocab.
    total_vocab = text_cfg.vocab_size + text_cfg.additional_vocab_size
    lm_cfg = _build_lm_config(text_cfg, total_vocab_size=total_vocab)
    vis_cfg = _build_vision_config(vit_cfg, adapter_cfg.vit_layers)
    conn_cfg = _build_connector_config(adapter_cfg, vit_hidden_size=vit_cfg.hidden_size)

    # Resolve vit_layers to **absolute** layer indices in the (possibly
    # truncated) ViT we just built. HF Molmo2 stores ``vit_layers`` as
    # negative indices into the *original* (untruncated) ViT, so e.g.
    # ``[-3, -9]`` with 27 layers means absolute ``[24, 18]``. Once we
    # truncate to 25 layers, our model's hidden_states list is 25 long and
    # we want to read indices ``[24, 18]`` directly.
    full_vit_layers = vit_cfg.num_hidden_layers
    resolved_vit_layers = tuple(
        layer if layer >= 0 else layer + full_vit_layers for layer in adapter_cfg.vit_layers
    )

    return MultimodalLMConfig(
        lm=lm_cfg,
        vision=vis_cfg,
        connector=conn_cfg,
        image_patch_token_id=hf_config.image_patch_id,
        vit_layers=resolved_vit_layers,
    )
