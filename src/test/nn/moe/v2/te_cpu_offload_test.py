from contextlib import nullcontext

from olmo_core.nn.moe.v2.te import get_cpu_offload_context
from olmo_core.nn.moe.v2.te.cpu_offload import CpuOffloadHandler, CpuOffloadHook
from olmo_core.testing import requires_gpu


def test_te_cpu_offload_module_exposes_public_api():
    # Import-level smoke test: the offload machinery itself allocates CUDA streams, but the module
    # must at least import cleanly and expose its public surface.
    assert callable(get_cpu_offload_context)
    assert CpuOffloadHook is not None
    assert CpuOffloadHandler is not None


@requires_gpu
def test_get_cpu_offload_context_disabled_returns_nullcontext():
    ctx, commit = get_cpu_offload_context(enabled=False, num_layers=2, model_layers=4)
    assert commit is None
    assert isinstance(ctx, nullcontext)
