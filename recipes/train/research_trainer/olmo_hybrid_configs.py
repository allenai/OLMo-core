"""``allenai/Olmo-Hybrid-7B`` model config for the CTC-suite hybrid-vs-not comparison.

Kept beside :mod:`olmo3_configs` and out of ``train_ctc_suite.py`` for the same reason: the shared
family tables there stay one line per family.

── WHY THIS MODEL ────────────────────────────────────────────────────────────────────────────
Every chunked-attention number in this project's headline suite is measured on Qwen3.5, whose
backbone is a 3:1 Gated-DeltaNet:full-attention hybrid. That leaves an unanswered question the
suite cannot settle on its own: is the dense-vs-chunked gap a property of the *task*, or partly a
property of the *backbone*, since the chunk mask only ever touches one layer in four?

``allenai/Olmo-3-1025-7B`` and ``allenai/Olmo-Hybrid-7B`` are the clean control, because they are
the same size and the same training data and differ in exactly the axis in question:

    Olmo-3-1025-7B   3:1 sliding-window(4096) : full attention   -- softmax throughout
    Olmo-Hybrid-7B   3:1 linear (Gated DeltaNet) : full attention -- the Qwen3.5 shape

Both share the dolma2 tokenizer and the same padded 100352-row embedding, so the *existing* olmo3
tokenized shards and marker tokenizer are reused verbatim -- no data rebuild, and the comparison is
not confounded by a retokenization.

── ARCHITECTURE, READ OFF THE CHECKPOINT (never guessed) ─────────────────────────────────────
Shapes below come from the safetensors header of ``allenai/Olmo-Hybrid-7B`` and its ``config.json``,
not from the family resemblance to Qwen3.5. Three of them differ from Qwen3.5 in ways that would
silently produce a wrong model if carried over:

* **``allow_neg_eigval=True``.** ``TransformerConfig.qwen3_5_like`` hardcodes ``False``. Olmo-Hybrid
  sets ``linear_allow_neg_eigval: true``, which doubles the beta range in the delta rule.
* **No RoPE at all.** ``rope_parameters.rope_theta`` is ``null`` -- the full-attention layers are
  NoPE. Qwen3.5 uses partial (25%) RoPE. Passing a RoPEConfig here rotates queries and keys the
  pretrained weights never saw.
* **No attention output gate, and whole-projection QK norm.** ``q_norm``/``k_norm`` are
  ``(3840,)`` -- one gain per *projection*, the OLMo-2 convention -- not ``(head_dim,)`` per-head as
  in Qwen3.5, and there is no ``self_attn.gate`` tensor at all.

The two block types also normalize differently, which the HF tensor names make explicit and which
:data:`olmo_core.nn.hf.convert.HYBRID_GDN_LAYER_KEY_MAP` /
:data:`~olmo_core.nn.hf.convert.HYBRID_ATTN_LAYER_KEY_MAP` already encode:

* GDN layers carry ``input_layernorm`` + ``post_attention_layernorm`` -> ordinary **pre-norm**.
* Attention layers carry ``post_attention_layernorm`` + ``post_feedforward_layernorm`` and *no*
  ``input_layernorm`` -> OLMo-2 **reordered (post) norm**.

olmo-core already ships the export half of this (``convert_hybrid_state_to_hf``, ``model_type:
olmo_hybrid``) and a test fixture that builds exactly this shape at 190M
(``src/test/examples/huggingface/convert_checkpoint_to_hf_hybrid_test.py``); this module is that
fixture's recipe at the real 7B dimensions.
"""

import os
from typing import List, Optional

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerBlockType

__all__ = [
    "OLMO_HYBRID_HF_ID",
    "OLMO_HYBRID_MARKER_TOKENIZER",
    "OLMO_HYBRID_VOCAB_SIZE",
    "OLMO_HYBRID_LAYER_TYPES",
    "olmo_hybrid_7B_ctc",
]

OLMO_HYBRID_HF_ID = "allenai/Olmo-Hybrid-7B"

#: Same padded dolma2 embedding size as Olmo-3 (100278 real ids -> 100352 rows). Olmo-Hybrid's
#: ``config.json`` already states ``vocab_size: 100352``, so unlike Olmo-3 there is nothing to pad.
OLMO_HYBRID_VOCAB_SIZE = 100352

#: Olmo-Hybrid shares Olmo-3's dolma2 tokenizer, so it also shares the PATCHED marker copy in which
#: ``<|extra_id_1|>`` / ``<|extra_id_2|>`` (ids 100266 / 100267, unchanged) are renamed to
#: ``<|box_start|>`` / ``<|box_end|>``. Reusing it is what makes the olmo3 shards valid here.
_TOKENIZER_CANDIDATES = (
    "/scratch/users/prasann/hf_models/Olmo-3-1025-7B-docchunk",
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_olmo3/tokenizer",
)


def _resolve_marker_tokenizer() -> str:
    """Pick the first marker-tokenizer copy that exists on this host.

    :returns: A path to the patched tokenizer dir (falls back to the Berkeley path so an error
        message names a real location if neither is present).
    """
    override = os.environ.get("OLMO3_MARKER_TOKENIZER")
    if override:
        return override
    for path in _TOKENIZER_CANDIDATES:
        if os.path.isdir(path):
            return path
    return _TOKENIZER_CANDIDATES[0]


OLMO_HYBRID_MARKER_TOKENIZER = _resolve_marker_tokenizer()

#: The released ``layer_types``: linear x3 then full, repeated 8 times over 32 layers.
OLMO_HYBRID_LAYER_TYPES: List[str] = [
    "full_attention" if (i % 4) == 3 else "linear_attention" for i in range(32)
]

# --- released dimensions (config.json + safetensors header) -----------------------------------
_D_MODEL = 3840
_N_LAYERS = 32
_N_HEADS = 30  # MHA: num_key_value_heads == num_attention_heads == 30
_HEAD_DIM = 128  # 3840 / 30; the config omits head_dim, so it is the derived value
_INTERMEDIATE = 11008
_LAYER_NORM_EPS = 1e-6
_LINEAR_NUM_KEY_HEADS = 30
_LINEAR_NUM_VALUE_HEADS = 30
_LINEAR_KEY_HEAD_DIM = 96  # q_proj/k_proj are (2880, 3840) = 30 * 96
_LINEAR_VALUE_HEAD_DIM = 192  # v_proj/g_proj are (5760, 3840) = 30 * 192; o_norm is (192,)
_LINEAR_CONV_KERNEL = 4


def olmo_hybrid_7B_ctc(
    vocab_size: int = OLMO_HYBRID_VOCAB_SIZE,
    *,
    document_chunked: bool = False,
    cross_doc_mode: Optional[str] = None,
    attn_backend: Optional[AttentionBackendName] = None,
    dtype: DType = DType.float32,
    **kwargs,
) -> TransformerConfig:
    """Build the CTC-suite ``Olmo-Hybrid-7B`` config.

    When ``document_chunked=True`` the mask is applied to the **full-attention blocks only**, and
    the Gated-DeltaNet blocks are left exactly as pretrained. That is not a simplification -- it is
    the only faithful option (linear attention has no mask to restrict) and it is precisely what
    the Qwen3.5 arms do, which is what makes this an apples-to-apples control for them. It also
    matches :func:`olmo3_configs.olmo3_7B_ctc_swa`, where the mask likewise touches only the 8
    full-attention layers, so all three families chunk the same 1-in-4 layers.

    :param vocab_size: Embedding rows; defaults to the released padded 100352.
    :param document_chunked: Apply :class:`~olmo_core.nn.attention.DocumentChunkedAttention` to the
        full-attention blocks.
    :param cross_doc_mode: Cross-document mask mode; only valid with ``document_chunked=True``.
    :param attn_backend: Attention backend override for the full-attention blocks.
    :param dtype: Parameter dtype for the config's submodules.

    :returns: The transformer config, ready to ``.build()``.

    :raises ValueError: If ``cross_doc_mode`` is passed without ``document_chunked``.
    """
    if cross_doc_mode is not None and not document_chunked:
        raise ValueError("'cross_doc_mode' is only valid when 'document_chunked=True'.")

    layer_norm = LayerNormConfig(
        name=LayerNormType.rms, eps=_LAYER_NORM_EPS, bias=False, dtype=dtype
    )
    feed_forward = FeedForwardConfig(hidden_size=_INTERMEDIATE, bias=False, dtype=dtype)

    # GDN blocks: ordinary pre-norm (HF input_layernorm + post_attention_layernorm).
    gdn_block = TransformerBlockConfig(
        name=TransformerBlockType.default,
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=_LINEAR_NUM_KEY_HEADS,
            n_v_heads=_LINEAR_NUM_VALUE_HEADS,
            head_dim=_LINEAR_KEY_HEAD_DIM,
            expand_v=_LINEAR_VALUE_HEAD_DIM / _LINEAR_KEY_HEAD_DIM,
            # ⚠ Differs from qwen3_5_like, which hardcodes False. Olmo-Hybrid sets
            # linear_allow_neg_eigval: true.
            allow_neg_eigval=True,
            conv_size=_LINEAR_CONV_KERNEL,
            # ⚠ Do NOT pass norm_eps=_LAYER_NORM_EPS (1e-6) here. GatedDeltaNetConfig's default of
            # 1e-5 is what HF OlmoHybridGatedDeltaNet and vLLM hardcode for the GDN o_norm, and the
            # HF export cannot express any other value — so a 1e-6-trained checkpoint is silently
            # served at 1e-5 by vLLM. Measured 2026-08-21: that mismatch flips 96% of per-example
            # generations on a near-chance task (aggregates survive within 1 SE, but only verified
            # after the fact). Training at the default keeps train == eval on every backend.
            # Probe evidence: debug/ctc_olmo_hybrid/hybrid_vllm/. Checkpoints trained before this
            # change (all ctc-olmohyb-* through 2026-08-21) are 1e-6: evaluate those NATIVE, or
            # accept the documented per-example shuffle under vLLM.
            dtype=dtype,
        ),
        feed_forward=feed_forward,
        layer_norm=layer_norm,
    )

    # Full-attention blocks: OLMo-2 reordered (post) norm, whole-projection QK norm, no output
    # gate, and NO RoPE -- rope=None is the NoPE the released config asks for via rope_theta: null.
    attn_block = TransformerBlockConfig(
        name=TransformerBlockType.reordered_norm,
        sequence_mixer=AttentionConfig(
            name=AttentionType.document_chunked if document_chunked else AttentionType.default,
            n_heads=_N_HEADS,
            n_kv_heads=_N_HEADS,
            head_dim=_HEAD_DIM,
            bias=False,
            rope=None,
            gate=None,
            qk_norm=layer_norm,
            use_head_qk_norm=False,
            backend=attn_backend,
            cross_doc_mode=cross_doc_mode if document_chunked else None,
            dtype=dtype,
        ),
        feed_forward=feed_forward,
        layer_norm=layer_norm,
    )

    return TransformerConfig(
        d_model=_D_MODEL,
        vocab_size=vocab_size,
        n_layers=_N_LAYERS,
        block={"gdn": gdn_block, "attn": attn_block},
        block_pattern=["gdn", "gdn", "gdn", "attn"],
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        dtype=dtype,
        **kwargs,
    )
