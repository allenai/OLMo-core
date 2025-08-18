import torch


@torch.compile(dynamic=True)
def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Compute log softmax probabilities for selected tokens.

    .. note::
        torch.compile() performs an optimization that avoids materializing the full log softmax
        tensor when combined with gather operations, which can save significant memory compared
        to computing the full log softmax and then indexing.

    :param logits: The logits tensor of shape ``(..., vocab_size)``.
    :param index: The index tensor of shape ``(...)``.

    :returns: The log probabilities of shape ``(...)``.
    """
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
