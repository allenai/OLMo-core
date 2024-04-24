"""
This is a light-weight, experimental rewrite of PyTorch's :class:`~torch.distributed.fsdp.FullyShardedDataParallel`
with a few of improvements, including:

- Well-defined "hands off" handling of buffers. FSDP never shards buffers, they are left as-is.
- Well-defined handling of frozen params. You can mix and match within an FSDP instance as long
  as you're consistent across the process group with which parameters are frozen.
- Full support for CPU-only training and inference via the GLOO backend.
- Low-overhead checkpointing with :mod:`olmo_core.distributed.checkpoint`.

Usage Tips
----------

- Always initialize your optimizer *after* wrapping your model with FSDP.
- When you use initialize model (prior to wrapping with FSDP), use ``device=torch.device("meta")``
  when initializing *parameters* to save memory. :class:`FSDP` will automatically materialize and
  move parameters to the right device when wrapping.
  Then you can use :meth:`FSDP.apply()` to initialize parameters how you want.
- Analogous to with PyTorch's :class:`~torch.distributed.fsdp.FullyShardedDataParallel`, you should
  use :func:`FSDP.clip_grad_norm_()` for clipping gradient norms instead of :func:`torch.nn.utils.clip_grad_norm_()`.
- Use activation checkpointing via :func:`torch.utils.checkpoint.checkpoint()` to save more memory
  during the forward and backward pass at the expense of more computation.
- To save and load checkpoints for your FSDP model and its optimizer, use
  :func:`~olmo_core.distributed.checkpoint.save_model_and_optim_state()` and
  :func:`~olmo_core.distributed.checkpoint.load_model_and_optim_state()`, respectively.

Implementation Details
----------------------

When you wrap a :class:`~torch.nn.Module` with :class:`FSDP`, the wrapping FSDP instance will replace
each original parameter in the module with a :class:`~olmo_core.distributed.tensors.ShardedFlatParameter` instance,
and each rank will only keep a shard of the original data. Buffers are left as-is.

.. note::
    Further, the sharded data for all of the :class:`~olmo_core.distributed.tensors.ShardedFlatParameter`
    instances will be collected into a single :class:`FlatParamHandle`, and each flat parameter will
    just hold a view into a slice of the data managed by the handle. This makes gathering the full
    params more efficient as it only requires a single all-gather per FSDP node.

Forward Pass
~~~~~~~~~~~~

When the :meth:`~torch.nn.Module.forward()` method is called on the wrapping FSDP instance, it will gather
the full unsharded data for each parameter in the desired :class:`~torch.dtype`
(as defined by the :class:`FSDPPrecision` settings) while caching the sharded data behind the scenes.
Then it runs the forward method of the wrapped module, which is completely unsharded at that point.

After the forward method of the wrapped module returns, the wrapping FSDP instance will reshard
the parameters and, if gradients are enabled, register backward hooks to manage the state of parameters
and gradients during the backward pass.

During the first forward pass the root FSDP instance will also record the order of execution of all
FSDP children, and use that order to prefetch the full parameters for its FSDP children during
subsequent forward passes. The number of children that are prefetched at once is controlled by the
``max_prefetch_count`` setting.

.. note::
    When CUDA is available :class:`FSDP` instances utilize multiple CUDA streams in order to overlap
    communication (e.g. unsharding params or reducing gradients) with computation
    (e.g. the forward pass or computing gradients during the backward pass).

Backward Pass
~~~~~~~~~~~~~

At the end of the forward method, the wrapping FSDP instance registers ephemeral "pre-backward" and "post-backward" hooks
to unshard the parameters and reduce-scatter the gradients, respectively, during the backward pass.

At the end of the backward pass the :attr:`~torch.Tensor.grad` attribute of each (non-frozen) parameter will
be the shard of the full gradient corresponding to the shard of the full parameter, i.e. it will
have the same shape/size as the sharded parameter.

Just how the root FSDP instance records the execution order of its FSDP children during the first
forward pass, the root will also record the order during the first backward pass and use that
to prefetch the full parameters of its children during subsequent backward passes.

API Reference
-------------
"""

from .fsdp import FSDP, FSDPDebugConfig, FSDPPrecision, FSDPShardingStrategy

__all__ = ["FSDP", "FSDPDebugConfig", "FSDPPrecision", "FSDPShardingStrategy"]
