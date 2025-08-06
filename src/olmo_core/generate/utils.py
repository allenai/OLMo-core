import torch


@torch.compile(dynamic=True)
def selective_log_softmax(
    logits: torch.Tensor, index: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Compute log softmax probabilities for selected tokens.

    .. note::
        Compiling this function reduces the required memory by 50% (by avoiding the materialization
        of the full logprobs tensor), which can be significant when working with large vocabularies
        and/or long sequences.

    :param logits: Logits tensor of shape ``(..., num_classes)``.
    :param index: Index tensor of shape ``(...)``, specifying the positions to gather from the log-softmax output.
    :param dtype: The dtype of the output tensor. If specified, the input tensor is cast to dtype
        before the operation is performed. This is useful for preventing data type overflows.

    :returns: Gathered log probabilities with the same shape as ``index``.
    """
    logprobs = logits.log_softmax(dim=-1, dtype=dtype)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
