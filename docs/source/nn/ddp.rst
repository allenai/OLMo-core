``nn.ddp``
==========

.. automodule:: olmo_core.nn.ddp
   :members:
   :member-order: bysource

Training
--------

Use :class:`~olmo_core.train.train_module.transformer.OLMoDDPTrainModuleConfig` with
:class:`~olmo_core.optim.OLMoDDPOptimizerConfig` to train an
:class:`~olmo_core.nn.ddp.OLMoDDPModel`. Parameter groups are synchronized through
:class:`~olmo_core.nn.parallel.MultiGroupDistributedDataParallel`.

.. autoclass:: olmo_core.train.train_module.transformer.OLMoDDPTrainModuleConfig
   :noindex:
   :members:

.. autoclass:: olmo_core.train.train_module.transformer.OLMoDDPTrainModule
   :noindex:
   :members:

.. autoclass:: olmo_core.optim.OLMoDDPOptimizerConfig
   :noindex:
   :members:

.. autoclass:: olmo_core.optim.OLMoDDPOptimizer
   :noindex:
   :members:

Runtime requirements
--------------------

The synchronized expert-parallel path does not require NVSHMEM. The rowwise path
requires the NVSHMEM CUDA extension; build it before starting distributed workers:

.. code-block:: bash

   python -m olmo_core.kernels.build_symm_mem_vdev2d_ext --inplace

CUDA, CMake, and the NVSHMEM headers and libraries must be available to the build.
Keep compiler caches node-local. ``OLMO_TRITON_CACHE_BASE`` sets the parent of the
default per-job, host, and rank Triton cache directories; an explicit
``TRITON_CACHE_DIR`` overrides that layout.
