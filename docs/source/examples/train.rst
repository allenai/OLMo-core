``Train a language model``
==========================

The following snippet is the code from ``src/examples/train.py``. It's a script meant to be launched via ``torchrun``.
You can also use the :mod:`olmo_core.launch` API to quickly launch this script on Beaker.
See below for an example of that.

``src/examples/train.py``
-------------------------

.. literalinclude:: ../../../src/examples/train.py
   :language: py

``src/examples/train_launch.py``
--------------------------------

.. literalinclude:: ../../../src/examples/train_launch.py
   :language: py
