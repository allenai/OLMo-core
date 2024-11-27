HuggingFace models
==================

The OLMo-core :class:`~olmo_core.train.Trainer` can be used to fine-tune language models from HuggingFace's ``transformers`` library.

One way to do this would be to manually apply a data parallel wrapper (like DDP or FSDP) to your ``AutoModelForCausalLM`` and then pass that model directly to the trainer. The downside with this approach is that you won't be able to take advantage of all of the optimizations in this library.

Instead we recommend converting your HuggingFace checkpoint into a format that can be loaded into an equivalent OLMo-core :class:`~olmo_core.nn.transformer.Transformer` model, when possible, using the functions provided by :mod:`olmo_core.distributed.checkpoint`.

Below is an example that shows how to convert an OLMo2 or Llama-3 checkpoint on HuggingFace into the right format for OLMo-core.
It would be straight forward to adapt this script to convert in the other direction as well.

.. seealso::
   See the `train a Llama model <llama.html>`_ example to learn how to use OLMo-core's training API to pretrain or fine-tune any Llama-like language model.

.. tab:: ``src/examples/huggingface/convert_checkpoint.py``

   .. literalinclude:: ../../../src/examples/huggingface/convert_checkpoint.py
      :language: py
