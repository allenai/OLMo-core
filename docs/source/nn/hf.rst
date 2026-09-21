``nn.hf``
==============

Tokenizer export safety
-----------------------

Conversion uses the explicit ``tokenizer_id`` (CLI ``--tokenizer``) first, then a
checkpoint-side tokenizer directory, then the saved configuration identifier.
Use ``tokenizer_revision`` / ``--tokenizer-revision`` to pin a Hub revision.
Implicit legacy ``allenai/dolma2-tokenizer`` identifiers require an explicit
choice: production OLMo-3 data and legacy tokenizer-class defaults can disagree
despite identical vocabularies.

The exporter requires a fast ``tokenizer.json`` and preserves its encoding
backend with a generic fast tokenizer. An independent AutoTokenizer reload
checks backend semantics, special IDs, chat metadata, and encoding/decoding
probes. Only runtime padding/truncation and the exact identity post-processor
are normalized for comparison. A mismatch fails conversion even when numerical
model validation is disabled. ``tokenizer-export-audit.json`` records provenance.
This does not repair training data that was tokenized incorrectly.

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
