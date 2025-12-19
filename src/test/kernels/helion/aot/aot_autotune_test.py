import helion
import helion.language as hl
import torch

from olmo_core.kernels.helion.aot.aot_autotune import KernelKey, helion_aot_autotune
from olmo_core.testing import requires_gpu

# ============================================================================
# Test helper kernel - assume this is correct
# ============================================================================


@helion.kernel(autotune_effort="quick")
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    """Sums a 2D tensor along the last dimension."""
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


def sum_kernel_key(x) -> KernelKey:
    """Key function for sum_kernel."""
    return KernelKey(
        numeric_key=x.shape[0], hash_key=(x.shape[0], x.shape[1], x.dtype), exact_key=(x.dtype,)
    )


def primary_inputs():
    """Generate primary test inputs."""
    for size in [128, 256]:
        yield (torch.randn(size, size, device="cuda", dtype=torch.float32),)


# ============================================================================
# Tests for aot_autotune decorator
# ============================================================================


@requires_gpu
def test_aot_autotune_decorator_basic():
    """Test that aot_autotune decorator wraps a kernel correctly."""
    wrapped = helion_aot_autotune(
        config_name="test_basic",
        kernel_key=sum_kernel_key,
        primary_inputs=primary_inputs,
    )(sum_kernel)

    # Should return a callable
    assert callable(wrapped)

    # Should execute correctly
    x = torch.randn(128, 128, device="cuda", dtype=torch.float32)
    result = wrapped(x)
    expected = x.sum(-1)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_aot_autotune_caches_configs():
    """Test that configs are cached between calls."""
    wrapped = helion_aot_autotune(
        config_name="test_cache",
        kernel_key=sum_kernel_key,
        primary_inputs=primary_inputs,
    )(sum_kernel)

    x = torch.randn(128, 128, device="cuda", dtype=torch.float32)

    # Call twice with same shape
    result1 = wrapped(x)
    result2 = wrapped(x)

    # Results should be identical (deterministic)
    torch.testing.assert_close(result1, result2)


@requires_gpu
def test_aot_autotune_handles_different_shapes():
    """Test that different input shapes are handled correctly."""
    wrapped = helion_aot_autotune(
        config_name="test_shapes",
        kernel_key=sum_kernel_key,
        primary_inputs=primary_inputs,
    )(sum_kernel)

    # Test multiple shapes
    for size in [128, 256, 512]:
        x = torch.randn(size, size, device="cuda", dtype=torch.float32)
        result = wrapped(x)
        expected = x.sum(-1)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_aot_autotune_with_secondary_inputs():
    """Test that secondary inputs work for benchmarking."""

    def secondary_inputs():
        """Generate secondary test inputs."""
        for size in range(128, 256, 32):
            yield (torch.randn(size, size, device="cuda", dtype=torch.float32),)

    wrapped = helion_aot_autotune(
        config_name="test_secondary",
        kernel_key=sum_kernel_key,
        primary_inputs=primary_inputs,
        secondary_inputs=secondary_inputs,
    )(sum_kernel)

    # Secondary inputs should not be autotuned but should work
    x = torch.randn(384, 384, device="cuda", dtype=torch.float32)
    result = wrapped(x)
    expected = x.sum(-1)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
