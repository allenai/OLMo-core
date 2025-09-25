try:
    import transformer_engine.pytorch as te  # type: ignore
except ImportError:
    te = None


def has_te_attn() -> bool:
    """Check if Transformer Engine attention is available."""
    return te is not None


TEDotProductAttention = te.DotProductAttention if te is not None else None
