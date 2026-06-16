try:
    import transformer_engine.pytorch as te  # type: ignore
except (ImportError, OSError) as e:
    te = None
    _te_import_error = e
else:
    _te_import_error = None


def has_te_attn() -> bool:
    """Check if Transformer Engine attention is available."""
    return te is not None


def te_attn_import_error() -> BaseException | None:
    return _te_import_error


TEDotProductAttention = te.DotProductAttention if te is not None else None
