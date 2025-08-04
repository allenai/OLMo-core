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
    B = k.shape[0]
    assert v.shape[0] == B, f"Value tensor batch size {v.shape[0]} != key tensor batch size {B}"
    assert k.shape[1] == T, f"Key tensor sequence length {k.shape[1]} != write length {T}"
    assert v.shape[1] == T, f"Value tensor sequence length {v.shape[1]} != write length {T}"

    # Compute the absolute positions in the sequence dimension **per batch item**
    seq_positions = cache_seqlens[:B, None] + torch.arange(T, device=k.device)

    # Sanity-check that we are not writing beyond the allocated cache.
    if (seq_positions >= k_cache.shape[1]).any():
        raise ValueError(
            f"KV cache overflow: max position {seq_positions.max().item()} >=  max_cache_len ({k_cache.shape[1]})"
        )

    batch_indices = torch.arange(B, device=k.device)[:, None]  # (B, 1)
    k_cache[batch_indices, seq_positions] = k
    v_cache[batch_indices, seq_positions] = v
