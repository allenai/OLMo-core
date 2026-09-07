"""Balanced MB3/MB2 views of prewarmed EP8 buffers at actual medium KDA dimensions."""

from test.nn.moe.v2.batched_lb_ep_test import _run_batched_ep_parity

import pytest
import torch

from olmo_core.testing import run_distributed_test


@pytest.mark.gpu
def test_compiled_balanced_microbatches_and_sharded_adam():
    """Compare explicit partitions with the balanced splitter, changing 16/32/16 sequences.

    Both arms use the same 2/3-sequence microbatch partitions and loss denominator;
    this is not a claim of identical auxiliary-loss granularity to uniform MB2.
    Two production-width KDA/latent-MoE blocks, private EP8 buffers, no recomputation,
    no runtime symmetric allocation, and unchanged BF16/FP32/CuTe policy.
    """
    if torch.cuda.device_count() < 8:
        pytest.skip("requires8 CUDA GPUs")
    run_distributed_test(
        _run_batched_ep_parity,
        world_size=8,
        backend="nccl",
        start_method="spawn",
        func_args=(8, 1536, 1536, 512, True, False, True),
    )
