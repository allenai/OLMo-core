from olmo_core.utils import format_int


def format_count(count: int) -> str:
    """Format a large count into a human-readable string."""
    return format_int(count)


def format_tokens(tokens: int) -> str:
    """Format number of tokens into a human-readable string."""
    return f"{format_count(tokens)} tokens"
