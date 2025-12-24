def format_count(count: int) -> str:
    """Format a large count into a human-readable string."""
    if count < 1_000:
        return f"{count}"
    elif count < 1_000_000:
        return f"{count / 1_000:.1f}K".replace(".0", "")
    elif count < 1_000_000_000:
        return f"{count / 1_000_000:.1f}M".replace(".0", "")
    elif count < 1_000_000_000_000:
        return f"{count / 1_000_000_000:.1f}B".replace(".0", "")
    else:
        return f"{count / 1_000_000_000_000:.1f}T".replace(".0", "")


def format_tokens(tokens: int) -> str:
    """Format number of tokens into a human-readable string."""
    return f"{format_count(tokens)} tokens"


def get_mix_base_dir(cluster: str) -> str:
    if cluster == "lambda":
        return "/data/caia-mltrain/data/"
    else:
        return "gs://ai2-llm/"
