"""OLMo 3 model/tokenizer configuration for the CTC-suite dense-vs-chunked experiment.

Kept out of ``train_ctc_suite.py`` so the shared family tables there stay one-line-per-family
(several agents edit that file concurrently).

The base is ``allenai/Olmo-3-1025-7B`` -- the newest OLMo generation, at the size nearest the
Qwen3.5-4B reference (OLMo 3 ships only 7B and 32B, so 7B it is). Two deliberate deviations from the
stock ``TransformerConfig.olmo3_7B`` factory, both applied to **both** arms so the full-vs-chunked
comparison stays internally valid:

1. **Sliding-window attention is disabled.** Native Olmo 3 runs a 3:1 sliding(4096):full layer
   pattern. :class:`~olmo_core.nn.attention.DocumentChunkedAttention` explicitly refuses sliding
   windows ("the chunked mask already restricts visibility"), so the chunked arm *cannot* keep them.
   Keeping SWA in the full arm only would make the two arms differ in two ways at once, confounding
   the very gap being measured. Disabling it in both arms makes the document-chunk mask the single
   manipulated variable, and it is the conservative direction: the full arm is asked to use a
   global receptive field its 24 sliding layers were not pretrained for, so any residual
   dense-over-chunked advantage is, if anything, understated.

2. **YaRN RoPE scaling is applied to every layer** (``full_attn_layers_only=False``). The released
   checkpoint carries ``rope_scaling={"rope_type":"yarn","factor":8.0,
   "original_max_position_embeddings":8192}``, which HF applies to the full-attention layers. With
   the sliding windows removed every layer *is* a full-attention layer, so every layer gets the same
   long-context RoPE treatment; that also keeps the rotary angles at the 8k-rung scale the former
   sliding layers were trained on, rather than extrapolating them to 16k.

Neither change touches any parameter shape, so the converted distcp base loads unchanged.
"""

from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig

__all__ = ["OLMO3_MARKER_TOKENIZER", "OLMO3_VOCAB_SIZE", "olmo3_7B_ctc"]

#: Patched dolma2 tokenizer copy in which the unused ``<|extra_id_1|>`` / ``<|extra_id_2|>`` slots
#: (ids 100266 / 100267, unchanged) are RENAMED to the project's canonical ``<|box_start|>`` /
#: ``<|box_end|>`` spellings, which the document-chunk converter and the native eval harness wrap
#: documents with and verify against ``tok.convert_tokens_to_ids``. Stock ``allenai/dolma2-tokenizer``
#: has no such tokens, so pointing at it instead makes the converter's marker-id check fail.
OLMO3_MARKER_TOKENIZER = "/scratch/users/prasann/hf_models/Olmo-3-1025-7B-docchunk"

#: olmo-core's padded dolma2 embedding size (100278 real ids -> 100352 rows). This is what
#: ``convert_checkpoint_from_hf.py --model-arch olmo3_7b --tokenizer dolma2`` writes, so the model
#: config must use it or the distcp load shape-mismatches.
OLMO3_VOCAB_SIZE = 100352

#: The released checkpoint's YaRN parameters, read off ``config.json`` (never guessed).
_YARN_FACTOR = 8.0
_YARN_OLD_CONTEXT = 8192


def olmo3_7B_ctc(**kwargs) -> TransformerConfig:
    """Build the CTC-suite Olmo-3 7B config (no sliding window, YaRN on every layer).

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
