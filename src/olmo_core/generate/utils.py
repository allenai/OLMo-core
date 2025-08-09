import torch


@torch.compile(dynamic=True)
def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Compute log softmax probabilities for selected tokens.

    .. note::
        torch.compile() performs an optimization that avoids materializing the full log softmax
        tensor, which saves ~50% of the total memory usage of the operation.

    :param logits: The logits tensor of shape ``(batch_size, seq_len, vocab_size)``.
    :param index: The index tensor of shape ``(batch_size, seq_len)``.

    :returns: The log probabilities of shape ``(batch_size, seq_len)``.
    """
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
