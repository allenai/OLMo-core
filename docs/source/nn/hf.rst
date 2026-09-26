``nn.hf``
==============

Tokenizer export safety
-----------------------

Conversion uses the explicit ``tokenizer_id`` (CLI ``--tokenizer``) first, then a
checkpoint-side tokenizer directory, then the saved configuration identifier.
Use ``tokenizer_revision`` / ``--tokenizer-revision`` to pin a Hub revision.
Saved identifiers, including ``allenai/dolma2-tokenizer``, remain supported.
For SFT, select the tokenizer and chat template actually used in training.
An unspecified BOS in the training config preserves the source tokenizer's BOS;
explicit special-token IDs in the config take precedence.
Model and generation configurations use the resolved BOS, EOS, and padding IDs,
including ID 0. Generation without input IDs starts with the exported BOS.
Recognized chat stop tokens are also retained for generation; absent token names
are never mapped to an unknown-token ID and used as stop tokens.

The exporter requires a fast ``tokenizer.json`` and preserves its encoding
backend with a generic fast tokenizer. An independent AutoTokenizer reload
checks backend semantics, special IDs, chat metadata, and encoding/decoding
probes. Only runtime padding/truncation and the exact identity post-processor
are normalized for comparison. A mismatch fails conversion even when numerical
model validation is disabled. ``tokenizer-export-audit.json`` records provenance.
This does not repair training data that was tokenized incorrectly.

OLMoDDP MoE export constraints
------------------------------

Hybrid KDA exports require full attention in every non-KDA layer; sliding-window
layers are rejected. In both hybrid and attention-only EMO models, every routed
layer's evaluation pool must span all experts. Restricted evaluation pools are
rejected because the HF router does not implement document-pool selection.
Forced uniform/random expert assignment and KDA convolution biases are also
unsupported and rejected before conversion.

Both export paths require bias-free, full-precision ``RMSNorm`` or ``FusedRMSNorm``
with one shared epsilon for Q/K, block pre/post, embedding, LM-head, and latent
input norms. Other normalization
operations and variants with different rounding behavior are rejected. All KDA
output norms must share one epsilon, which is preserved as ``linear_norm_eps``;
it may differ from the model RMSNorm epsilon.

Headwise and elementwise attention gates are preserved in both export paths,
including their precision setting. HF supports one shared expert per block
without a shared-expert router; other shared-expert configurations are rejected.

Scalable-softmax exports set ``use_cache=False`` for generation. The HF model
rejects both ``use_cache=True`` and supplied ``past_key_values`` for these models.
Ordinary attention-only exports retain KV caching. Hybrid KDA exports also
require uncached input because the HF reference does not implement recurrent-state
caching.

.. automodule:: olmo_core.nn.hf
   :members:
   :member-order: bysource

.. autodata:: olmo_core.nn.hf.convert.HF_TO_OLMO_CORE_WEIGHT_MAPPINGS
   :no-value:
.. autodata:: olmo_core.nn.hf.convert.HF_TO_OLMO_CORE_MODULE_MAPPINGS
   :no-value:
.. autodata:: olmo_core.nn.hf.convert.MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_WEIGHT_MAPPINGS
   :no-value:
.. autodata:: olmo_core.nn.hf.convert.MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_MODULE_MAPPINGS
   :no-value:
.. autodata:: olmo_core.nn.hf.convert.HF_TO_OLMO_CORE_TEMPLATE_MAPPINGS
   :no-value:
.. autodata:: olmo_core.nn.hf.convert.OLMO_CORE_TO_HF_WEIGHT_MAPPINGS
   :no-value:
.. autodata:: olmo_core.nn.hf.convert.OLMO_CORE_TO_HF_MODULE_MAPPINGS
   :no-value:
.. autodata:: olmo_core.nn.hf.convert.OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS
   :no-value:
.. autodata:: olmo_core.nn.hf.convert.MODEL_TYPE_SPECIFIC_OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS
   :no-value:
