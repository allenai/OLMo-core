import torch


def greedy_selection(logits: torch.Tensor) -> torch.Tensor:
    """
    Deterministically select the next token as the one with the highest logit.

    :param logits: Logits tensor of shape ``(..., vocab_size)``.

    :returns: Selected token indices of shape ``(...)``.
    """
    return logits.argmax(dim=-1)


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Filter logits to keep only the top k tokens.

    :param logits: Logits tensor of shape ``(..., vocab_size)``.
    :param top_k: Number of top tokens to keep.

    :returns: Filtered logits with -inf for tokens outside top k.
    """
    if top_k <= 0:
        return logits

    # Get the kth largest value
    kth_values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
    kth_value = kth_values[..., -1].unsqueeze(-1)

    # Set all logits below the kth value to -inf
    return torch.where(logits < kth_value, torch.full_like(logits, float("-inf")), logits)


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Filter logits using nucleus (top-p) sampling.

    :param logits: Logits tensor of shape ``(..., vocab_size)``.
    :param top_p: Cumulative probability threshold for nucleus sampling.

    :returns: Filtered logits with -inf for tokens outside the nucleus.
    """
    if top_p <= 0.0 or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 0] = False

    # Shift the mask to include the token that crosses the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()

    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )

    # Set filtered tokens to -inf
    return logits.masked_fill(indices_to_remove, float("-inf"))


# @torch.compile(dynamic=True)
def select_next_token(
    logits: torch.Tensor,
    do_sample: bool = True,
    temperature: float = 0.0,
    top_k: int = -1,
    top_p: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample from the logits using temperature scaling with optional top-k and top-p filtering.

    :param logits: Logits tensor of shape ``(..., vocab_size)``.
    :param do_sample: Whether to sample from the distribution. If False, uses greedy selection.
                      If True, applies temperature scaling and optional top-k and top-p filtering.
    :param temperature: Temperature for scaling. Higher values increase randomness.
                        Values < 1.0 make the distribution sharper (more deterministic).
                        Values > 1.0 make the distribution flatter (more random).
                        Value = 0.0 is equivalent to greedy selection.
    :param top_k: Only consider the top k tokens with highest probabilities. -1 means no filtering.
    :param top_p: Only consider the smallest set of tokens whose cumulative
                  probability exceeds this threshold (nucleus sampling). 1.0 means no filtering.
    :param dtype: The dtype of the output tensor. If specified, the input tensor is cast to dtype
        before the operation is performed. This is useful for preventing data type overflows.

    :returns: Sampled token indices of shape ``(...)``.
    """
    if not do_sample or temperature == 0:
        return greedy_selection(logits)

    nan_mask = torch.isnan(logits)
    num_nans = nan_mask.sum().item()
    total_elements = logits.numel()
    nan_percentage = (num_nans / total_elements) * 100 if total_elements > 0 else 0
    batch_nan_info = [i for i in range(logits.shape[0]) if torch.isnan(logits[i]).any()]
    if nan_mask.any():
        breakpoint()

    assert not nan_mask.any(), (
        f"NaN values detected in logits: {num_nans}/{total_elements} ({nan_percentage:.2f}%) "
        f"NaN values in tensor of shape {logits.shape}"
        + (f" in batch elements: {', '.join(map(str, batch_nan_info))}" if batch_nan_info else "")
    )

    scaled_logits = logits / temperature
    if top_k != -1:
        scaled_logits = top_k_filtering(scaled_logits, top_k)
    if top_p != 1.0:
        scaled_logits = top_p_filtering(scaled_logits, top_p)

    probs = torch.softmax(scaled_logits, dim=-1, dtype=dtype)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
