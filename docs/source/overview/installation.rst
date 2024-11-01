Installation
============

Prior to installing OLMo-core you should install `PyTorch <https://pytorch.org>`_ according to the official instructions
specific to your operating system and hardware.

Then you can install OLMo-core from `PyPI <https://pypi.org/project/ai2-olmo-core/>`_ with::

    pip install ai2-olmo-core

There are a number of optional dependencies that must be installed to use certain functionality as well, including:

- `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ for flash attention and certain other fused operations.
- `torchao <https://github.com/pytorch/ao>`_ for float8 training (see :mod:`olmo_core.float8`).
- `megablocks <https://github.com/databricks/megablocks>`_ for mixture-of-experts (MoE) models (see :mod:`olmo_core.nn.moe`).
