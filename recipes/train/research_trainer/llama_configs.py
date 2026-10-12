"""OLMo-core :class:`TransformerConfig` factories for the Llama-3 checkpoints used by the CTC
suite, derived from the HF ``config.json`` rather than copied from a neighbouring size.

``olmo_core.nn.transformer.config`` ships named factories only for ``llama3_1B`` / ``llama3_8B`` /
``llama3_70B`` / ``llama3_405B`` -- there is no 3B. Llama-3.2-3B is the newest-generation Llama at
the scale closest to the suite's Qwen3.5-4B reference (Llama 4 is a 109B MoE, Llama 3.3 is 70B
only), so it gets a factory here, built with the generic ``llama_like`` and dimensions taken
verbatim from ``meta-llama/Llama-3.2-3B``'s config:

===========================  ==========  ====================================================
HF config field              value       how it reaches ``llama_like``
===========================  ==========  ====================================================
``hidden_size``              3072        ``d_model``
``num_hidden_layers``        28          ``n_layers``
``num_attention_heads``      24          ``n_heads``
``num_key_value_heads``      8           ``n_kv_heads`` (GQA)
``head_dim``                 128         ``head_dim`` (== 3072/24, passed explicitly anyway)
``intermediate_size``        8192        ``int(8 * d_model / 3)`` = 8192 exactly, and
                                         ``hidden_size_multiple_of=256`` leaves it untouched
``rms_norm_eps``             1e-5        ``layer_norm_eps``
``rope_theta``               500000      ``rope_theta``
``rope_scaling`` (llama3)    f=32        :class:`StepwiseRoPEScalingConfig` -- HF's ``llama3``
                                         rope type IS stepwise scaling, and its defaults
                                         (factor 32, low_freq_factor 1.0 -> proportion 0.0,
                                         high_freq_factor 4.0 -> proportion 0.25,
                                         old_context_len 8192) match this config exactly
``vocab_size``               128256      ``vocab_size`` (caller passes it)
===========================  ==========  ====================================================

:func:`assert_matches_hf` re-reads the HF config at conversion/train time and re-checks all of the
above plus the parameter count, so a wrong-size factory cannot silently load weights into a
mismatched architecture (which loads "successfully" and produces plausible garbage).

.. note::
   Llama-3.2-3B **ties** its input embeddings to the LM head (``tie_word_embeddings: true``), so
   its safetensors contain no ``lm_head.weight``. OLMo-core's transformer keeps them as separate
   parameters, so the converter must copy the embedding matrix into the LM head -- see
   ``convert_llama_base.py``. Forgetting this leaves the LM head at random init.
"""

import json
import os
from typing import Any, Dict

from olmo_core.nn.rope import StepwiseRoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig

#: The patched tokenizer directory the Llama CTC runs use. Llama 3 has no ``<|box_start|>`` /
#: ``<|box_end|>`` tokens, so reserved slots 128002/128003 are renamed to those strings (ids
#: unchanged) by ``src/scripts/data/make_llama_marker_tokenizer.py``. Override with the env var
#: ``LLAMA_MARKER_TOKENIZER`` on a machine that stages it elsewhere (e.g. weka on Beaker).
LLAMA_MARKER_TOKENIZER = os.environ.get(
    "LLAMA_MARKER_TOKENIZER", "/scratch/users/prasann/hf_models/Llama-3.2-3B-marker-tok"
)

#: HF ``config.json`` fields that MUST match the factory below, keyed by HF field name.
LLAMA3_2_3B_HF_SHAPE: Dict[str, Any] = {
    "model_type": "llama",
    "hidden_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 8192,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "vocab_size": 128256,
    "tie_word_embeddings": True,
}

#: Parameter count of the HF (tied) checkpoint: 28 blocks + one embedding matrix + final norm.
#: OLMo-core unties the LM head, so its own ``num_params`` is this plus one more embedding matrix.
LLAMA3_2_3B_HF_NUM_PARAMS = 3_212_749_824


def llama3_2_3B(vocab_size: int = 128256, **kwargs) -> TransformerConfig:
    """Build the OLMo-core config for ``meta-llama/Llama-3.2-3B``.

    :param vocab_size: Embedding-matrix rows (128256 for the stock checkpoint).
    :param kwargs: Forwarded to :meth:`TransformerConfig.llama_like` (e.g. ``attn_backend``,
        ``document_chunked``, ``cross_doc_mode``).

    :returns: The transformer config.
    """
    kwargs.setdefault("rope_scaling", StepwiseRoPEScalingConfig())  # HF rope_type "llama3"
    return TransformerConfig.llama_like(
        d_model=3072,
        vocab_size=vocab_size,
        n_layers=kwargs.pop("n_layers", 28),
        n_heads=kwargs.pop("n_heads", 24),
        n_kv_heads=kwargs.pop("n_kv_heads", 8),
        head_dim=kwargs.pop("head_dim", 128),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=kwargs.pop("layer_norm_eps", 1e-5),
        hidden_size_multiple_of=kwargs.pop("hidden_size_multiple_of", 256),
        **kwargs,
    )


#: HF ``config.json`` fields that MUST match :func:`llama3_1_8B`.
#:
#: ⚠ THIS IS NOT THE 3B TABLE WITH BIGGER NUMBERS. Two fields differ *structurally*, and both fail
#: silently rather than loudly if carried over from Llama-3.2-3B:
#:
#: * ``tie_word_embeddings`` is **False** here and True at 3B. The 3B converter copies the
#:   embedding matrix into the LM head because HF ships no ``lm_head.weight``; 8B ships a real,
#:   separately-trained one. Running the tie-copy on 8B would OVERWRITE a trained LM head with the
#:   embedding matrix -- the model still loads and still generates, just worse.
#: * ``rope_scaling.factor`` is **8.0** here and 32.0 at 3B, and
#:   :class:`StepwiseRoPEScalingConfig` defaults to 32.0 -- i.e. the *default* is wrong for this
#:   model. A mis-scaled RoPE degrades long-context behaviour specifically, which is the axis this
#:   whole suite measures, so it would read as a Llama long-context finding.
LLAMA3_1_8B_HF_SHAPE: Dict[str, Any] = {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 14336,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "vocab_size": 128256,
    "tie_word_embeddings": False,
}

#: Parameter count of the HF checkpoint. UNLIKE the 3B constant this one is already UNTIED (8B
#: ships its own ``lm_head.weight``), so olmo-core's ``num_params`` should equal it exactly with
#: no extra embedding matrix added. Asserted at conversion time by :func:`assert_matches_hf`, so a
#: wrong value here fails a 10-minute conversion job rather than reaching a training run.
LLAMA3_1_8B_HF_NUM_PARAMS = 8_030_261_248


def llama3_1_8B(vocab_size: int = 128256, **kwargs) -> TransformerConfig:
    """Build the OLMo-core config for ``meta-llama/Llama-3.1-8B``.

    Unlike :func:`llama3_2_3B` this maps onto olmo-core's own :meth:`TransformerConfig.llama3_8B`
    factory, whose ``hidden_size_multiplier=1.3`` / ``hidden_size_multiple_of=1024`` reproduce the
    checkpoint's ``intermediate_size`` of 14336. :func:`assert_matches_hf` re-checks that against
    the real ``config.json`` rather than trusting it.

    :param vocab_size: Embedding-matrix rows (128256 for the stock checkpoint).
    :param kwargs: Forwarded to :meth:`TransformerConfig.llama3_8B`.

    :returns: The transformer config.
    """
    # factor=8.0 is Llama-3.1-8B's own value; the class default of 32.0 is Llama-3.2's. See the
    # warning on LLAMA3_1_8B_HF_SHAPE -- this line is why that table exists.
    kwargs.setdefault("rope_scaling", StepwiseRoPEScalingConfig(factor=8.0))
    return TransformerConfig.llama3_8B(
        vocab_size=vocab_size,
        layer_norm_eps=kwargs.pop("layer_norm_eps", 1e-5),
        **kwargs,
    )


def assert_matches_hf(
    hf_dir: str,
    config: TransformerConfig,
    shape: Dict[str, Any] | None = None,
    hf_num_params: int | None = None,
    factory_name: str = "llama3_2_3B",
) -> Dict[str, Any]:
    """Hard-check a built config against the checkpoint's own ``config.json``.

    Checks every shape field and the parameter count. A mismatched architecture does NOT crash on
    load (olmo-core's converter reports missing/unexpected keys but shapes that happen to line up
    load fine), so this is the guard that keeps a wrong factory from producing plausible-looking
    numbers.

    Whether olmo-core carries an *extra* embedding matrix relative to HF is derived from the
    checkpoint's own ``tie_word_embeddings`` rather than assumed: olmo-core always keeps the LM
    head separate, so a tied HF checkpoint (3.2-3B) gains one matrix and an untied one (3.1-8B)
    gains none. Hardcoding either behaviour breaks the other model.

    :param hf_dir: The HF checkpoint directory (must contain ``config.json``).
    :param config: The built :class:`TransformerConfig`.
    :param shape: Expected HF fields; defaults to :data:`LLAMA3_2_3B_HF_SHAPE`.
    :param hf_num_params: The HF checkpoint's parameter count; defaults to the 3B constant.
    :param factory_name: Name used in error messages.

    :returns: The parsed HF config dict.

    :raises SystemExit: On any mismatch.
    """
    shape = LLAMA3_2_3B_HF_SHAPE if shape is None else shape
    hf_num_params = LLAMA3_2_3B_HF_NUM_PARAMS if hf_num_params is None else hf_num_params
    with open(os.path.join(hf_dir, "config.json")) as f:
        raw = json.load(f)
    bad = []
    for key, want in shape.items():
        got = raw.get(key)
        # `head_dim` is OPTIONAL in an HF llama config: Llama-3.2-3B states it explicitly, but
        # Llama-3.1-8B omits it, and transformers then derives it as hidden_size //
        # num_attention_heads. Comparing the absent key against 128 would fail a checkpoint whose
        # geometry is in fact correct, so apply the same default here rather than dropping the
        # field from the table (it still catches a genuinely wrong head_dim, stated or derived).
        if key == "head_dim" and got is None:
            got = raw["hidden_size"] // raw["num_attention_heads"]
        if isinstance(want, bool):
            ok = bool(got) == want
        elif isinstance(want, float):
            ok = got is not None and abs(float(got) - want) < 1e-9
        else:
            ok = got == want
        if not ok:
            bad.append(f"{key}: HF={got!r} expected={want!r}")
    if bad:
        raise SystemExit(
            f"HF config does not match the {factory_name} factory:\n  " + "\n  ".join(bad)
        )
    # OLMo-core always unties the LM head, so it carries one extra embedding matrix relative to a
    # TIED checkpoint and none relative to an untied one.
    tied = bool(raw.get("tie_word_embeddings"))
    extra = raw["vocab_size"] * raw["hidden_size"] if tied else 0
    want_untied = hf_num_params + extra
    if config.num_params != want_untied:
        detail = (
            f"(= HF tied {hf_num_params:,} + untied lm_head {extra:,})"
            if tied
            else f"(= HF {hf_num_params:,}, already untied)"
        )
        raise SystemExit(
            f"param-count mismatch: built config has {config.num_params:,} params, expected "
            f"{want_untied:,} {detail}. The architecture does not match the checkpoint."
        )
    return raw
