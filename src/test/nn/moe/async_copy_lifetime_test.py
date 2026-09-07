"""Allocator lifetime coverage for short-lived asynchronous expert-count copies."""

import pytest
import torch

from olmo_core.nn.moe.utils import async_copy_to_cpu
from olmo_core.utils import get_or_init_stream


@pytest.mark.gpu
@pytest.mark.parametrize("compiled", [False, True])
def test_temporary_source_survives_delayed_copy(compiled):
    """Release the source and churn same-sized allocations before the copy completes."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    def copy_temporary(x):
        return async_copy_to_cpu(x + 7)

    if compiled:
        copy_temporary = torch.compile(copy_temporary)
    source = torch.arange(65536, dtype=torch.int64, device="cuda")
    # Compile first so the delayed copy test is not accidentally serialized by JIT.
    _, _, event = copy_temporary(source)
    event.synchronize()
    for _ in range(8):
        stream = get_or_init_stream(id="dtoh", priority=-5)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(50_000_000)
        copied, _, event = copy_temporary(source)
        churn = [torch.full_like(source, -17) for _ in range(32)]
        del churn
        event.synchronize()
        torch.testing.assert_close(copied, torch.arange(65536, dtype=torch.int64) + 7)
