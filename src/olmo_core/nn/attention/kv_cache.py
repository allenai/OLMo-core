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
        k_cache: Key cache tensor of shape [batch_size, max_cache_len, ...]
        v_cache: Value cache tensor of shape [batch_size, max_cache_len, ...]
        cache_seqlens: Current sequence lengths for each batch item, shape [batch_size]
        k: Key tensor to write, shape [batch_size, T, ...]
        v: Value tensor to write, shape [batch_size, T, ...]
        T: Sequence length of the key/value tensors to write

    Raises:
        ValueError: If writing would exceed the cache capacity
    """
    seq_positions = cache_seqlens[:, None] + torch.arange(
        T, device=k.device, dtype=cache_seqlens.dtype
    )
    if (seq_positions >= k_cache.shape[1]).any():
        raise ValueError(
            f"KV cache overflow: max position {seq_positions.max().item()} >= "
            f"max_cache_len ({k_cache.shape[1]})"
        )
    k_cache[:, seq_positions] = k
    v_cache[:, seq_positions] = v
