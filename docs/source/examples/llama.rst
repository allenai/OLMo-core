``Train a Llama model``
=======================

The following snippet is the code from ``src/examples/llama/train.py``.
It's a script meant to be launched via ``torchrun``.
You can also use the :mod:`olmo_core.launch` API to quickly launch this script on Beaker.
See below for an example of that.

``src/examples/llama/train.py``
-------------------------------

.. literalinclude:: ../../../src/examples/llama/train.py
   :language: py

``src/examples/llama/train_launch.py``
--------------------------------------

.. literalinclude:: ../../../src/examples/llama/train_launch.py
   :language: py
