Train a Llama model
===================

The following snippets can be found in `src/examples/llama/ <https://github.com/allenai/OLMo-core/tree/main/src/examples/llama>`_.
The ``train.py`` script is meant to be launched via ``torchrun``.
You can also use the :mod:`olmo_core.launch` API to quickly launch this script on Beaker.
See the ``train_launch.py`` snippet for an example of that.

.. tab:: ``train.py``

   .. literalinclude:: ../../../src/examples/llama/train.py
      :language: py

.. tab:: ``train_launch.py``

   .. literalinclude:: ../../../src/examples/llama/train_launch.py
      :language: py
