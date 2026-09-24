``nn.moe``
==========

.. automodule:: olmo_core.nn.moe
   :members:
   :member-order: bysource

Optional OLMoDDP expert optimizations
-------------------------------------

Set these environment variables before constructing ``RoutedExperts``:

* ``OLMO_PROFILE_SWIGLU_PAIRWISE=1`` combines the two SwiGLU input-gradient
  stores in one Triton kernel. It applies to compiled, contiguous CUDA BF16
  expert activations; eager execution keeps the native PyTorch activation.
  The backward matches the compiled baseline on Torch 2.11 and 2.13, including
  their different sigmoid lowering. Other Torch versions are rejected when
  the optimized backward is used. Large expert buffers use 64-bit indexing.
* ``OLMO_PROFILE_ROUNDED_WGRAD=1`` accumulates expert weight gradients directly
  into their owned FP32 OLMoDDP buckets. Each microbatch's GEMM result is rounded
  to BF16 before FP32 accumulation, preserving the native BF16 gradient path.
  This requires Blackwell, Torch 2.11 or 2.13, ``quack-kernels==0.5.0``, and
  FP32 gradient accumulation and reduction. The tested CuTe DSL version is
  ``nvidia-cutlass-dsl==4.5.3``.
* Also set ``OLMO_PROFILE_ROUNDED_WGRAD_EP=1`` when using expert parallelism or
  the rowwise output/input-gradient buffers. This enables the separately
  tested EP path.

Rounded accumulation requires each expert parameter to be used exactly once
per forward. Duplicate uses, native gradients on the same parameter, writes
after reduction has been scheduled, and detached destination buckets raise
errors. Saved-tensor hooks and activation recomputation retain the original
parameter's ownership. Higher-order derivatives are unsupported.
The rounded GEMM uses an explicit compiler boundary, so compilation must
allow graph breaks around that operation.

The regression tests exercise compiled BF16 SwiGLU values and gradients,
all BF16 gate encodings, large-buffer indexing, sharded Adam updates,
activation recomputation, checkpoint resume, and EP degrees 2, 4, and 8.
Run ``src/test/nn/parallel/swiglu_pairwise_test.py`` on two GPUs and
``src/test/nn/moe/v2/rounded_wgrad_ep_test.py`` on eight Blackwell GPUs.
Set ``OLMO_TEST_LARGE_SWIGLU=1`` to include the allocation exceeding
2**31 elements; that test requires at least 24 GiB of free GPU memory.

Timing depends on shape and hardware. On Torch 2.13, an isolated compiled
SwiGLU forward/backward at 524,288 rows and hidden size 1,024 took
2.08 ms natively versus 1.29 ms paired on B300, and 2.99 ms versus 2.79 ms
on H100. On B300, two isolated weight-gradient shapes with 512 experts
and 131,072 rows took 2.08 ms versus 0.86 ms and 1.05 ms versus 0.43 ms.
These measurements describe those regions, not whole-training throughput.
