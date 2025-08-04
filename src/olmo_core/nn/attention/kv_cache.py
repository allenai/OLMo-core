import torch


def write_kvcache_(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    T: int,
) -> None:
    """Write key and value tensors to the KV cache at the appropriate positions.

    Args:
        k_cache: Key cache tensor of shape [batch_size, max_cache_len, num_kv_heads, head_dim]
        v_cache: Value cache tensor of shape [batch_size, max_cache_len, num_kv_heads, head_dim]
        cache_seqlens: Current sequence lengths for each batch item, shape [batch_size]
        k: Key tensor to write, shape [batch_size, T, num_kv_heads, head_dim]
        v: Value tensor to write, shape [batch_size, T, num_kv_heads, head_dim]
        T: Sequence length of the key/value tensors to write

    Raises:
        ValueError: If writing would exceed the cache capacity
    """
    # Assert that k and v have the correct batch sizes
    batch_size = k_cache.shape[0]
    assert k.shape[0] == batch_size, (
        f"Key tensor batch size {k.shape[0]} != cache batch size {batch_size}"
    )
    assert v.shape[0] == batch_size, (
        f"Value tensor batch size {v.shape[0]} != cache batch size {batch_size}"
    )
    assert cache_seqlens.shape[0] == batch_size, (
        f"cache_seqlens batch size {cache_seqlens.shape[0]} != cache batch size {batch_size}"
    )
    assert k.shape[1] == T, f"Key tensor sequence length {k.shape[1]} != write length {T}"
    assert v.shape[1] == T, f"Value tensor sequence length {v.shape[1]} != write length {T}"

    # Compute the absolute positions in the sequence dimension **per batch item**
    # Shape: (B, T)
    seq_positions = cache_seqlens[:, None] + torch.arange(
        T, device=k.device, dtype=cache_seqlens.dtype
    )

    # Sanity-check that we are not writing beyond the allocated cache.
    if (seq_positions >= k_cache.shape[1]).any():
        raise ValueError(
            f"KV cache overflow: max position {seq_positions.max().item()} >= "
            f"max_cache_len ({k_cache.shape[1]})"
        )

    # NOTE: Flash-Attention expects the sequence dimension in K/V cache to be
    # physically contiguous. Using `scatter_` or advanced indexing creates
    # strided layouts which break numerical equivalence.  Write the new slice
    # for each batch item with a plain slice assignment – this keeps the data
    # contiguous in memory and is cheap because batch sizes during inference
    # are small.

    B = k.shape[0]
    seq_positions_long = seq_positions.to(torch.long)

    # Use index_copy_ along the sequence dimension so every (batch, pos)
    # pair is updated exactly once and the cache remains contiguous.
    for b in range(B):
        k_cache[b].index_copy_(0, seq_positions_long[b], k[b])
        v_cache[b].index_copy_(0, seq_positions_long[b], v[b])
