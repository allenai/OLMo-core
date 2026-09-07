"""Conditional memory-candidate gate at the medium KDA head/latent dimensions."""

from test.nn.moe.v2.batched_lb_ep_test import _run_batched_ep_parity

import pytest
import torch

from olmo_core.testing import run_distributed_test


@pytest.mark.gpu
def test_compiled_kda_attention_only_checkpointing():
    """Recompute only KDA, never EP/EMO: compare three Adam updates and all states.

    Two production-width KDA+latent-MoE blocks, MB4/T8192 and two accumulated
    microbatches, with the existing fused kernels. This is a dropless EP8 fixture,
    not a qualification of the full 24-layer stack or its capacity1.25 workload.
    """
    if torch.cuda.device_count() < 8:
        pytest.skip("requires8 CUDA GPUs")
    run_distributed_test(
        _run_batched_ep_parity,
        world_size=8,
        backend="nccl",
        start_method="spawn",
        func_args=(8, 1536, 1536, 512, True, True),
    )
