import pytest
import torch

from olmo_core.nn.attention.kv_cache import write_kvcache_
from olmo_core.testing.utils import GPU_MARKS, requires_gpu


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=GPU_MARKS)])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("T", [1, 16])
def test_write_kvcache_basic(device: str, batch_size: int, T: int):
    """Test basic functionality of write_kvcache_."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    max_cache_len = 128
    num_kv_heads = 4
    head_dim = 64

    k_cache = torch.zeros(batch_size, max_cache_len, num_kv_heads, head_dim, device=device)
    v_cache = torch.zeros(batch_size, max_cache_len, num_kv_heads, head_dim, device=device)
    cache_seqlens = torch.zeros(batch_size, dtype=torch.long, device=device)

    k = torch.randn(batch_size, T, num_kv_heads, head_dim, device=device)
    v = torch.randn(batch_size, T, num_kv_heads, head_dim, device=device)

    write_kvcache_(k_cache, v_cache, cache_seqlens, k, v, T)

    # Verify the write
    for b in range(batch_size):
        torch.testing.assert_close(k_cache[b, :T], k[b])
        torch.testing.assert_close(v_cache[b, :T], v[b])
        # Check that the rest of the cache is still zeros
        assert torch.all(k_cache[b, T:] == 0)
        assert torch.all(v_cache[b, T:] == 0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=GPU_MARKS)])
def test_write_kvcache_sequential_writes(device: str):
    """Test sequential writes to the cache."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    batch_size = 2
    max_cache_len = 64
    num_kv_heads = 4
    head_dim = 32

    # Initialize cache tensors
    k_cache = torch.zeros(batch_size, max_cache_len, num_kv_heads, head_dim, device=device)
    v_cache = torch.zeros(batch_size, max_cache_len, num_kv_heads, head_dim, device=device)
    cache_seqlens = torch.zeros(batch_size, dtype=torch.long, device=device)

    # First write
    T1 = 10
    k1 = torch.randn(batch_size, T1, num_kv_heads, head_dim, device=device)
    v1 = torch.randn(batch_size, T1, num_kv_heads, head_dim, device=device)
    write_kvcache_(k_cache, v_cache, cache_seqlens, k1, v1, T1)
    cache_seqlens += T1

    # Verify first write
    for b in range(batch_size):
        torch.testing.assert_close(k_cache[b, :T1], k1[b])
        torch.testing.assert_close(v_cache[b, :T1], v1[b])

    # Second write
    T2 = 5
    k2 = torch.randn(batch_size, T2, num_kv_heads, head_dim, device=device)
    v2 = torch.randn(batch_size, T2, num_kv_heads, head_dim, device=device)
    write_kvcache_(k_cache, v_cache, cache_seqlens, k2, v2, T2)
    cache_seqlens += T2

    # Verify both writes
    for b in range(batch_size):
        # First write should be unchanged
        torch.testing.assert_close(k_cache[b, :T1], k1[b])
        torch.testing.assert_close(v_cache[b, :T1], v1[b])
        # Second write should be at the correct position
        torch.testing.assert_close(k_cache[b, T1 : T1 + T2], k2[b])
        torch.testing.assert_close(v_cache[b, T1 : T1 + T2], v2[b])
        # Rest should be zeros
        assert torch.all(k_cache[b, T1 + T2 :] == 0)
        assert torch.all(v_cache[b, T1 + T2 :] == 0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=GPU_MARKS)])
def test_write_kvcache_different_batch_positions(device: str):
    """Test writing to different positions for different batch items."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    batch_size = 3
    max_cache_len = 64
    num_kv_heads = 2
    head_dim = 32
    T = 4

    # Initialize cache tensors
    k_cache = torch.zeros(batch_size, max_cache_len, num_kv_heads, head_dim, device=device)
    v_cache = torch.zeros(batch_size, max_cache_len, num_kv_heads, head_dim, device=device)
    # Different starting positions for each batch item
    cache_seqlens = torch.tensor([0, 5, 10], dtype=torch.long, device=device)

    # Create key and value tensors to write
    k = torch.randn(batch_size, T, num_kv_heads, head_dim, device=device)
    v = torch.randn(batch_size, T, num_kv_heads, head_dim, device=device)

    # Write to cache
    write_kvcache_(k_cache, v_cache, cache_seqlens, k, v, T)

    # Verify the write at different positions
    torch.testing.assert_close(k_cache[0, 0:T], k[0])
    torch.testing.assert_close(v_cache[0, 0:T], v[0])

    torch.testing.assert_close(k_cache[1, 5 : 5 + T], k[1])
    torch.testing.assert_close(v_cache[1, 5 : 5 + T], v[1])

    torch.testing.assert_close(k_cache[2, 10 : 10 + T], k[2])
    torch.testing.assert_close(v_cache[2, 10 : 10 + T], v[2])


def test_write_kvcache_overflow_error():
    """Test that overflow raises an error."""
    batch_size = 1
    max_cache_len = 10
    num_kv_heads = 2
    head_dim = 16

    # Initialize cache tensors
    k_cache = torch.zeros(batch_size, max_cache_len, num_kv_heads, head_dim)
    v_cache = torch.zeros(batch_size, max_cache_len, num_kv_heads, head_dim)
    cache_seqlens = torch.tensor([8], dtype=torch.long)  # Starting at position 8

    # Try to write 5 tokens (would go to positions 8-12, exceeding max_cache_len=10)
    T = 5
    k = torch.randn(batch_size, T, num_kv_heads, head_dim)
    v = torch.randn(batch_size, T, num_kv_heads, head_dim)

    with pytest.raises(ValueError, match="KV cache overflow"):
        write_kvcache_(k_cache, v_cache, cache_seqlens, k, v, T)


def test_write_kvcache_batch_size_mismatch():
    """Test that batch size mismatches raise errors."""
    batch_size = 2
    max_cache_len = 32
    num_kv_heads = 2
    head_dim = 16
    T = 4

    # Initialize cache tensors
    k_cache = torch.zeros(batch_size, max_cache_len, num_kv_heads, head_dim)
    v_cache = torch.zeros(batch_size, max_cache_len, num_kv_heads, head_dim)
    cache_seqlens = torch.zeros(batch_size, dtype=torch.long)

    # Wrong batch size for k
    k_wrong = torch.randn(batch_size + 1, T, num_kv_heads, head_dim)
    v = torch.randn(batch_size, T, num_kv_heads, head_dim)

    with pytest.raises(AssertionError, match="Key tensor batch size"):
        write_kvcache_(k_cache, v_cache, cache_seqlens, k_wrong, v, T)

    # Wrong batch size for v
    k = torch.randn(batch_size, T, num_kv_heads, head_dim)
    v_wrong = torch.randn(batch_size + 1, T, num_kv_heads, head_dim)

    with pytest.raises(AssertionError, match="Value tensor batch size"):
        write_kvcache_(k_cache, v_cache, cache_seqlens, k, v_wrong, T)

    # Wrong batch size for cache_seqlens
    cache_seqlens_wrong = torch.zeros(batch_size + 1, dtype=torch.long)

    with pytest.raises(AssertionError, match="cache_seqlens batch size"):
        write_kvcache_(k_cache, v_cache, cache_seqlens_wrong, k, v, T)


def test_write_kvcache_sequence_length_mismatch():
    """Test that sequence length mismatches raise errors."""
    batch_size = 1
    max_cache_len = 32
    num_kv_heads = 2
    head_dim = 16
    T = 4

    # Initialize cache tensors
    k_cache = torch.zeros(batch_size, max_cache_len, num_kv_heads, head_dim)
    v_cache = torch.zeros(batch_size, max_cache_len, num_kv_heads, head_dim)
    cache_seqlens = torch.zeros(batch_size, dtype=torch.long)

    # Wrong sequence length for k
    k_wrong = torch.randn(batch_size, T + 1, num_kv_heads, head_dim)
    v = torch.randn(batch_size, T, num_kv_heads, head_dim)

    with pytest.raises(AssertionError, match="Key tensor sequence length"):
        write_kvcache_(k_cache, v_cache, cache_seqlens, k_wrong, v, T)

    # Wrong sequence length for v
    k = torch.randn(batch_size, T, num_kv_heads, head_dim)
    v_wrong = torch.randn(batch_size, T + 1, num_kv_heads, head_dim)

    with pytest.raises(AssertionError, match="Value tensor sequence length"):
        write_kvcache_(k_cache, v_cache, cache_seqlens, k, v_wrong, T)


@requires_gpu
def test_write_kvcache_memory_contiguity():
    """Test that cache remains contiguous after writes."""
    batch_size = 2
    max_cache_len = 64
    num_kv_heads = 4
    head_dim = 32

    # Initialize cache tensors on GPU
    k_cache = torch.zeros(batch_size, max_cache_len, num_kv_heads, head_dim, device="cuda")
    v_cache = torch.zeros(batch_size, max_cache_len, num_kv_heads, head_dim, device="cuda")
    cache_seqlens = torch.zeros(batch_size, dtype=torch.long, device="cuda")

    # Verify initial contiguity
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()

    # Perform multiple writes
    for i in range(3):
        T = 8
        k = torch.randn(batch_size, T, num_kv_heads, head_dim, device="cuda")
        v = torch.randn(batch_size, T, num_kv_heads, head_dim, device="cuda")

        write_kvcache_(k_cache, v_cache, cache_seqlens, k, v, T)
        cache_seqlens += T

        # Check that cache remains contiguous after each write
        assert k_cache.is_contiguous(), f"k_cache not contiguous after write {i + 1}"
        assert v_cache.is_contiguous(), f"v_cache not contiguous after write {i + 1}"
