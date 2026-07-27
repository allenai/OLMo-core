"""OLMo 3 model/tokenizer configuration for the CTC-suite dense-vs-chunked experiment.

Kept out of ``train_ctc_suite.py`` so the shared family tables there stay one-line-per-family
(several agents edit that file concurrently).

The base is ``allenai/Olmo-3-1025-7B`` -- the newest OLMo generation, at the size nearest the
Qwen3.5-4B reference (OLMo 3 ships only 7B and 32B, so 7B it is).

Olmo 3 runs a 3:1 sliding(4096):full attention pattern, which makes it a close structural analogue
of the Qwen3.5 hybrid (3:1 Gated-DeltaNet:full). Qwen3.5's arms apply the document-chunk mask to the
full-attention blocks only and leave the GDN blocks untouched, so the faithful Olmo counterpart --
:func:`olmo3_7B_ctc_swa`, the factory the suite uses -- applies the mask to Olmo's 8 full-attention
layers only and leaves the 24 sliding layers as pretrained. Both arms then share an identical
pretrained backbone and differ *only* in the mask on those 8 layers.

:func:`olmo3_7B_ctc` was the first attempt and is **superseded**: it disabled sliding windows
everywhere (because ``DocumentChunkedAttention`` refuses a sliding window) on the reasoning that
changing both arms identically keeps the comparison internally valid. That reasoning was right about
validity and wrong about cost -- ``olmo3_swa_ablation.py`` measures the *base* model at CE 0.033
native vs 1.362 with the windows removed (11.5k-token passage), so both arms started ~41x degraded,
and contradiction never learned the task from there. It is retained only so those runs stay
reproducible.

Neither factory touches any parameter shape, so the converted distcp base loads unchanged.
"""

import os

from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig

__all__ = [
    "OLMO3_MARKER_TOKENIZER",
    "OLMO3_VOCAB_SIZE",
    "olmo3_7B_ctc",
    "olmo3_7B_ctc_swa",
]

#: Patched dolma2 tokenizer copy in which the unused ``<|extra_id_1|>`` / ``<|extra_id_2|>`` slots
#: (ids 100266 / 100267, unchanged) are RENAMED to the project's canonical ``<|box_start|>`` /
#: ``<|box_end|>`` spellings, which the document-chunk converter and the native eval harness wrap
#: documents with and verify against ``tok.convert_tokens_to_ids``. Stock ``allenai/dolma2-tokenizer``
#: has no such tokens, so pointing at it instead makes the converter's marker-id check fail.
#: Resolution order: ``$OLMO3_MARKER_TOKENIZER`` -> the Berkeley ``/scratch`` copy -> the Beaker
#: weka copy. The same runs execute on both clusters and neither path exists on the other, so a
#: single hardcoded absolute path would break whichever side it was not written for.
_OLMO3_TOKENIZER_CANDIDATES = (
    "/scratch/users/prasann/hf_models/Olmo-3-1025-7B-docchunk",
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_olmo3/tokenizer",
)


def _resolve_marker_tokenizer() -> str:
    """Pick the first marker-tokenizer copy that exists on this host.

    :returns: A path to the patched tokenizer dir (falls back to the Berkeley path so the error
        message names a real location if neither is present).
    """
    override = os.environ.get("OLMO3_MARKER_TOKENIZER")
    if override:
        return override
    for path in _OLMO3_TOKENIZER_CANDIDATES:
        if os.path.isdir(path):
            return path
    return _OLMO3_TOKENIZER_CANDIDATES[0]


OLMO3_MARKER_TOKENIZER = _resolve_marker_tokenizer()

#: olmo-core's padded dolma2 embedding size (100278 real ids -> 100352 rows). This is what
#: ``convert_checkpoint_from_hf.py --model-arch olmo3_7b --tokenizer dolma2`` writes, so the model
#: config must use it or the distcp load shape-mismatches.
OLMO3_VOCAB_SIZE = 100352

#: The released checkpoint's YaRN parameters, read off ``config.json`` (never guessed).
_YARN_FACTOR = 8.0
_YARN_OLD_CONTEXT = 8192


def olmo3_7B_ctc_swa(**kwargs) -> TransformerConfig:
    """Build the CTC-suite Olmo-3 7B config that **keeps the native sliding-window backbone**.

    This is the faithful analogue of the Qwen3.5 arms, and it supersedes :func:`olmo3_7B_ctc`.
    Qwen3.5 applies the document-chunk mask to its full-attention blocks only and leaves the 3-in-4
    Gated-DeltaNet blocks alone; the exact counterpart for Olmo 3 is to apply the mask to its
    full-attention layers only and leave the 3-in-4 sliding-window layers alone. Both arms then
    share the identical pretrained backbone and differ *only* in the mask on those 8 layers.

    The first attempt (:func:`olmo3_7B_ctc`) instead disabled sliding windows everywhere, because
    :class:`~olmo_core.nn.attention.DocumentChunkedAttention` rejects a sliding window. That is
    measurably destructive: ``olmo3_swa_ablation.py`` puts the base model at CE 0.033 native vs
    1.362 with the windows removed on an 11.5k-token passage, and the contradiction runs trained
    from that damaged starting point never learned the task (CE plateaued at ~1.44 from step 200,
    and both arms emitted one constant answer for all 500 eval examples).

    The construction exploits two existing behaviours rather than adding machinery:

    * ``with_rope_scaling(full_attn_layers_only=True)`` already emits ``block_overrides`` for
      exactly the non-sliding layers -- the same 8 layers we want to chunk.
    * ``DocumentChunkedAttention``'s sliding-window rejection keys off the **per-layer** resolved
      ``window_size``, not the presence of a config, so a layer that is not an SWA layer may be
      document-chunked while its neighbours stay windowed.

    :param kwargs: Passed to :meth:`TransformerConfig.olmo3_7B`; ``document_chunked`` /
        ``cross_doc_mode`` are intercepted and applied to the full-attention layers only.

    :returns: The transformer config, ready to ``.build()``.
    """
    from olmo_core.nn.attention import AttentionType

    document_chunked = kwargs.pop("document_chunked", False)
    cross_doc_mode = kwargs.pop("cross_doc_mode", None)
    # Native architecture: SWA pattern [4096, 4096, 4096, -1], YaRN on the full-attention layers.
    config = TransformerConfig.olmo3_7B(**kwargs)
    config = config.with_rope_scaling(
        YaRNRoPEScalingConfig(
            factor=_YARN_FACTOR,
            beta_fast=32,
            beta_slow=1,
            old_context_len=_YARN_OLD_CONTEXT,
        )
    )
    if not document_chunked:
        return config

    overrides = config.block_overrides or {}
    if not overrides:
        raise ValueError(
            "expected with_rope_scaling to emit block_overrides for the full-attention layers; "
            "got none, so there is no set of layers to apply the document-chunk mask to."
        )
    for block in overrides.values():
        mixer = block.sequence_mixer
        mixer.name = AttentionType.document_chunked
        mixer.cross_doc_mode = cross_doc_mode or "chunked"
        # These layers are the non-SWA ones by construction; make that explicit so the per-layer
        # window_size is never resolved for them.
        mixer.sliding_window = None
    return config


def olmo3_7B_ctc(**kwargs) -> TransformerConfig:
    """Build the CTC-suite Olmo-3 7B config (no sliding window, YaRN on every layer).

    .. warning::
       SUPERSEDED by :func:`olmo3_7B_ctc_swa`. Disabling the sliding windows costs the base model
       roughly 41x in cross-entropy before any training happens (see ``olmo3_swa_ablation.py``),
       and the contradiction runs trained on it never learned the task. Kept only so the runs that
       used it remain reproducible.

    :param kwargs: Passed through to :meth:`TransformerConfig.olmo3_7B` -- notably ``vocab_size``,
        ``attn_backend``, and the ``document_chunked`` / ``cross_doc_mode`` pair for the chunked arm.

    :returns: The transformer config, ready to ``.build()``.
    """
    kwargs.setdefault("sliding_window", None)
    config = TransformerConfig.olmo3_7B(**kwargs)
    return config.with_rope_scaling(
        YaRNRoPEScalingConfig(
            factor=_YARN_FACTOR,
            beta_fast=32,
            beta_slow=1,
            old_context_len=_YARN_OLD_CONTEXT,
        ),
        full_attn_layers_only=False,
    )
