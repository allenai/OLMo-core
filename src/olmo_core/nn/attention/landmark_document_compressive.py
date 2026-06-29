"""
``DocumentCompressiveLandmarkAttention`` -- the **compressive** analogue of
:class:`~olmo_core.nn.attention.landmark_document.DocumentLandmarkAttention`, completing the trio of
document-chunked attention families (dense / landmark / compressive).

It combines:

* the **compressive** landmark grouped softmax
  (:func:`~olmo_core.nn.attention.landmark.compressive_landmark_grouped_softmax`): a query attends its
  own section fully and each earlier block gated by that block's landmark, and the gate weight is
  distributed by a within-block softmax over *all* the block's tokens -- content **and** landmark --
  so the landmark token contributes its value as a learned compressed summary of its block; and
* the **chunked-document** masking ported from corpus-reasoning (per-token ``chunk_id`` roles,
  ``build_chunked_allowed_mask`` / the ``cross_doc_mode`` pattern family): context chunks are mutually
  isolated while FREE tokens (the trailing query/answer, plus any instruction) bridge across chunks,
  and because the bridge runs through the compressive grouped softmax the query/answer retrieves each
  earlier chunk's content *and* its compressed landmark summary, gated by that chunk's landmarks.

Structurally this subclasses :class:`DocumentLandmarkAttention` and inherits the entire chunked-mask
machinery (``forward`` stashing ``chunk_ids``, the chunk-folded ``_landmark_masks``, the anchor
helper, the cached-generation prefill/decode path, and the Ulysses-CP-with-chunk_ids rejection); only
the eager grouped softmax is swapped for the compressive one, and the cached decode is swapped for the
compressive decode (which also honours ``nonselected_landmark_mass`` at top-k eval).

**Eager only.** Like the landmark doc-chunked runs, this defaults to (and only supports) the eager
grouped-softmax path. The fused compressive kernel has no chunk-mask variant, and the document SFT
data is right-padded with ``PadToLength`` whose pad tail breaks the kernel's positional ``is_mem``
assumption anyway. ``use_kernel=True`` is rejected.
"""

from typing import Optional

import torch

from olmo_core.exceptions import OLMoConfigurationError

from .landmark import compressive_landmark_grouped_softmax
from .landmark_compressive import FastCompressiveLandmarkAttention
from .landmark_document import DocumentLandmarkAttention


class DocumentCompressiveLandmarkAttention(DocumentLandmarkAttention):
    """
    Compressive landmark attention with chunked-document masking
    (``AttentionType.document_compressive_landmark``). See the module docstring.

    :param mem_freq: The number of regular tokens between landmark tokens. The landmark block size is
        ``mem_freq + 1``.
    :param nonselected_landmark_mass: The fraction ``alpha in [0, 1)`` of attention mass reserved, at
        top-k decode time, for the landmark tokens of the *non-selected* blocks (see
        :class:`~olmo_core.nn.attention.landmark_compressive.FastCompressiveLandmarkAttention`). Has no
        effect during training / exact eval.
    :param cross_doc_mode: The chunked-attention pattern name (the pluggable cross-document policy);
        defaults to ``"chunked"``.

    See :class:`DocumentLandmarkAttention` / :class:`LandmarkAttention` / :class:`Attention` for the
    remaining parameters.
    """

    # Cached fast-eval decode: reuse the COMPRESSIVE decode (so the landmark summary keeps contributing
    # at decode, and ``nonselected_landmark_mass`` is honoured under top-k) instead of the plain
    # landmark decode bound by :class:`DocumentLandmarkAttention`. ``init_kv_cache_manager`` and
    # ``clear_landmark_eval_decode`` are inherited unchanged.
    set_landmark_eval_decode = FastCompressiveLandmarkAttention.set_landmark_eval_decode
    _decode_one = FastCompressiveLandmarkAttention._decode_one
    _decode_one_eval = FastCompressiveLandmarkAttention._decode_one_eval
    _compressive_decode_probs = FastCompressiveLandmarkAttention._compressive_decode_probs

    def __init__(
        self,
        *,
        mem_freq: int,
        nonselected_landmark_mass: float = 0.1,
        cross_doc_mode: str = "chunked",
        use_kernel: bool = False,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ):
        if use_kernel:
            raise OLMoConfigurationError(
                "DocumentCompressiveLandmarkAttention is eager-only (use_kernel=True is not "
                "supported): the fused compressive kernel has no chunk-mask path, and PadToLength "
                "pad tails break its positional is_mem assumption."
            )
        super().__init__(
            mem_freq=mem_freq,
            cross_doc_mode=cross_doc_mode,
            use_kernel=False,
            softmax_scale=softmax_scale,
            **kwargs,
        )
        if not (0.0 <= nonselected_landmark_mass < 1.0):
            raise OLMoConfigurationError(
                f"nonselected_landmark_mass must be in [0, 1) (got {nonselected_landmark_mass})"
            )
        self.nonselected_landmark_mass = nonselected_landmark_mass

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Eager compressive grouped softmax with the chunked-document additive mask. Mirrors
        :meth:`LandmarkAttention._eager_forward` but calls
        :func:`~olmo_core.nn.attention.landmark.compressive_landmark_grouped_softmax`. The chunk mask
        is folded into ``attn_mask`` / ``is_mem`` / ``last_section_mask`` by the inherited
        :meth:`DocumentLandmarkAttention._landmark_masks` (it reads the stashed ``self._chunk_ids``).
        """
        B, n_heads, T, _ = q.shape
        attn_mask, is_mem, last_section_mask = self._landmark_masks(
            T, q.device, q.dtype, cu_doc_lens=cu_doc_lens, batch_size=B
        )

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale
        attn = attn + attn_mask
        attn = torch.maximum(
            attn, torch.tensor(torch.finfo(attn.dtype).min, device=attn.device, dtype=attn.dtype)
        )

        # Inference top-k landmark retrieval (exact eval keeps every allowed block). The compressive
        # alpha (``nonselected_landmark_mass``) is applied only on the cached decode path; the eager
        # top-k here zeroes non-selected blocks, matching DocumentLandmark's eager prefill.
        if self._eval_top_k is not None:
            attn = self._apply_topk_landmark_retrieval(attn, T)

        probs = compressive_landmark_grouped_softmax(
            attn,
            dim=-1,
            is_mem=is_mem.expand(B, n_heads, T, T),
            last_section_mask=last_section_mask.expand(B, 1, T, T),
        ).to(q.dtype)

        # shape: (B, n_heads, T, head_dim)
        return torch.matmul(probs, v)
