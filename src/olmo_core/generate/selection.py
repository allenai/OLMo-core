import torch


def greedy_selection(logits: torch.Tensor) -> torch.Tensor:
    """
    Deterministically select the next token as the one with the highest logit.
    """
    return logits.argmax(dim=-1)


def temperature_sampling(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample from the logits using temperature scaling.

    Args:
        logits: Logits tensor of shape (..., vocab_size)
        temperature: Temperature for scaling. Higher values increase randomness.
                    Values < 1.0 make the distribution sharper (more deterministic).
                    Values > 1.0 make the distribution flatter (more random).
                    Value = 0.0 is equivalent to greedy selection.

    Returns:
        Sampled token indices of shape (...)
    """
    if temperature < 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    if temperature == 0:
        return greedy_selection(logits)

    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
